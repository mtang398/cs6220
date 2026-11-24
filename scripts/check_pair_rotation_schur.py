#!/usr/bin/env python3
"""Dense paired-rotation Schur headroom oracle; no native resource claim."""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_large_core_exchange_oracle as previous

ROOT = Path(__file__).resolve().parents[1]
RTOL, CORE, CHECKPOINT = 1e-8, 416, 10


class PairRotation:
    def __init__(self, permutation, cosine, sine):
        self.permutation = np.asarray(permutation)
        self.c, self.s = np.asarray(cosine), np.asarray(sine)
        self.m = len(self.c)
        assert len(self.permutation) > 2*self.m
        assert len(np.unique(self.permutation)) == len(self.permutation)
        assert len(self.s) == self.m

    def to_rotated(self, vector):
        m = self.m
        ordered = vector[self.permutation]
        result = ordered.copy()
        result[:m] = self.c*ordered[:m]+self.s*ordered[m:2*m]
        result[m:2*m] = -self.s*ordered[:m]+self.c*ordered[m:2*m]
        return result

    def to_original(self, vector):
        m = self.m
        result = np.empty_like(vector)
        result[self.permutation[:m]] = self.c*vector[:m]-self.s*vector[m:2*m]
        result[self.permutation[m:2*m]] = self.s*vector[:m]+self.c*vector[m:2*m]
        result[self.permutation[2*m:]] = vector[2*m:]
        return result

    def matrix(self, A):
        m = self.m
        result = A[np.ix_(self.permutation, self.permutation)]
        left, right = result[:, :m].copy(), result[:, m:2*m].copy()
        result[:, :m] = left*self.c+right*self.s
        result[:, m:2*m] = -left*self.s+right*self.c
        del left, right
        top, bottom = result[:m, :].copy(), result[m:2*m, :].copy()
        result[:m, :] = self.c[:, None]*top+self.s[:, None]*bottom
        result[m:2*m, :] = -self.s[:, None]*top+self.c[:, None]*bottom
        return result

    def metadata(self):
        return dict(cosine=self.c.tolist(), sine=self.s.tolist(),
            orthogonality_max_defect=float(np.max(np.abs(self.c*self.c+self.s*self.s-1))),
            identity_pairs=int(np.count_nonzero((self.c == 1.) & (self.s == 0.))),
            permutation_sha256=sha256(self.permutation.tobytes()).hexdigest())


def aligned(permutation, direction, m):
    ordered = direction[permutation]
    length = np.hypot(ordered[:m], ordered[m:2*m])
    c, s = np.ones(m), np.zeros(m)
    nonzero = length > 0
    np.divide(ordered[:m], length, out=c, where=nonzero)
    np.divide(ordered[m:2*m], length, out=s, where=nonzero)
    return PairRotation(permutation, c, s)


def pair_geometry(K, J):
    available = np.ones(len(K), dtype=bool)
    available[J] = False
    partners = []
    for j in J:
        candidates = np.flatnonzero(available)
        partner = int(candidates[np.argmax(K[j, candidates])])
        partners.append(partner)
        available[partner] = False
    partners = np.asarray(partners, dtype=np.int64)
    remaining = np.flatnonzero(available)
    return np.r_[J, partners, remaining], partners, remaining


def fps(K, m):
    selected = [0]
    distance = 2-2*K[:, 0]
    for _ in range(m-1):
        j = int(np.argmax(distance))
        selected.append(j)
        distance = np.minimum(distance, 2-2*K[:, j])
    selected = np.asarray(selected, dtype=np.int64)
    assert len(np.unique(selected)) == m
    return selected


def solve(A, b, rotation, rotated_A, rotated_b, op, *, x0=None,
          capture_prefix=False, deadline=float('inf')):
    """Ordinary reduced CG, literal original-system stopping every iterate."""
    R, F, B, V, D, rhs = op
    m, n = rotation.m, len(b)
    original_norm = float(np.linalg.norm(b))
    initial_rotated = np.zeros(n) if x0 is None else rotation.to_rotated(x0)
    y = initial_rotated[R].copy()
    recursive = rhs-D@y
    p = recursive.copy()
    rho = float(recursive@recursive)
    saved = {}
    iterations, verifies, reconstructions = 0, 0, 0
    status = 'maxiter'
    transition = None
    current = None
    original_residual = None
    started, cpu_started = perf_counter(), process_time()
    for step in range(2001):
        full_rotated = np.empty(n)
        full_rotated[R] = y
        full_rotated[:m] = cho_solve(F, rotated_b[:m]-B@y)
        reconstructions += 1
        current = rotation.to_original(full_rotated)
        original_residual = b-A@current
        verifies += 1
        relative = float(np.linalg.norm(original_residual)/original_norm)
        if step == 0:
            before = np.zeros(n) if x0 is None else x0
            delta = current-before
            before_residual = b-A@before
            after_rotated_residual = rotation.to_rotated(original_residual)
            transition = dict(initial_original_relative_residual=float(np.linalg.norm(before_residual)/original_norm),
                after_coarse_correction_relative_residual=relative,
                correction_norm=float(np.linalg.norm(delta)),
                correction_energy=float(.5*delta@(A@delta)),
                correction_objective_decrease=float(delta@before_residual-.5*delta@(A@delta)),
                new_core_relative_residual=float(np.linalg.norm(after_rotated_residual[:m])/original_norm),
                retained_transformed_tail_max_change=float(np.max(np.abs(full_rotated[R]-initial_rotated[R]))))
        if capture_prefix and step in [CHECKPOINT-1, CHECKPOINT]:
            saved[step] = current.copy()
        if relative <= RTOL:
            status = 'converged'
            break
        if step == 2000:
            break
        if process_time() > deadline:
            status = 'cpu_cap'
            break
        ap = D@p
        denominator = float(p@ap)
        if min(rho, denominator) <= 0 or not np.isfinite(rho+denominator):
            status = 'nonpositive_or_nonfinite_curvature'
            break
        alpha = rho/denominator
        y += alpha*p
        recursive -= alpha*ap
        new_rho = float(recursive@recursive)
        if not np.isfinite(new_rho):
            status = 'nonfinite_residual'
            break
        p *= new_rho/rho
        p += recursive
        rho = new_rho
        iterations += 1
    reduced_true = rhs-D@y
    transformed_residual = rotation.to_rotated(original_residual)
    relative = float(np.linalg.norm(original_residual)/original_norm)
    row = dict(iterations=iterations, status=status, converged=relative <= RTOL,
        original_relative_residual=relative,
        exact_reduced_relative_residual=float(np.linalg.norm(reduced_true)/original_norm),
        recursive_reduced_relative_residual=float(np.linalg.norm(recursive)/original_norm),
        core_original_relative_residual=float(np.linalg.norm(transformed_residual[:m])/original_norm),
        reduced_vs_transformed_original_residual_difference=float(np.linalg.norm(reduced_true-transformed_residual[R])/original_norm),
        original_verification_products=verifies, diagnostic_core_reconstructions=reconstructions,
        diagnostic_Schur_products=iterations+2, transition=transition,
        solution_sha256=sha256(current.tobytes()).hexdigest(),
        diagnostic_wall_seconds=perf_counter()-started,
        diagnostic_cpu_seconds=process_time()-cpu_started)
    return row, saved, current


def tiny_check():
    rng = np.random.default_rng(622121)
    F = rng.normal(size=(13, 13))
    A = F@F.T+.3*np.eye(13)
    b = rng.normal(size=13)
    permutation = rng.permutation(13)
    angle = np.array([.3, -.4, 1.1, .7])
    rotation = PairRotation(permutation, np.cos(angle), np.sin(angle))
    Q = np.column_stack([rotation.to_original(np.eye(13)[:, i]) for i in range(13)])
    Ar, br = rotation.matrix(A), rotation.to_rotated(b)
    orthogonal_error = float(np.linalg.norm(Q.T@Q-np.eye(13)))
    transformed_error = float(np.linalg.norm(Ar-Q.T@A@Q)/np.linalg.norm(A))
    assert orthogonal_error < 1e-14 and transformed_error < 1e-14
    J = np.arange(4)
    op = previous.build(Ar, br, J)
    x0 = rng.normal(size=13)
    row, _, answer = solve(A, b, rotation, Ar, br, op, x0=x0)
    reference = cho_solve(cho_factor(A, lower=True), b)
    error = float(np.linalg.norm(answer-reference)/np.linalg.norm(reference))
    e = reference-x0
    ideal = aligned(permutation, e, 4)
    tail_error = float(np.linalg.norm(ideal.to_rotated(e)[4:8])/np.linalg.norm(e))
    assert row['converged'] and error < 1e-7 and tail_error < 1e-14
    assert row['transition']['retained_transformed_tail_max_change'] == 0
    return dict(orthogonality_error=orthogonal_error,
        transformed_operator_relative_error=transformed_error,
        independent_full_solution_relative_error=error,
        exact_error_pair_tail_fraction=tail_error,
        original_relative_residual=row['original_relative_residual'],
        coarse_correction=row['transition'])


def main():
    output = ROOT/'results/pair-rotation-schur.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    sources = [Path(__file__).resolve(), Path(previous.__file__).resolve()]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(datasets=['smooth', 'mixed'], n=2048, core=CORE, seed=0,
            regularizations=[.001, 1e-5], checkpoint=CHECKPOINT, rtol=RTOL,
            max_iterations=2000, CPU_cap=80, wall_cap=120,
            pairing='Each FPS core coordinate, in deterministic FPS order, takes its nearest still-unused noncore coordinate by Gaussian kernel distance; ties use smallest original row index.',
            branches=['unchanged_restart', 'fixed45_after10', 'remaining_error_oracle_after10',
                      'recent_CG_displacement_after10', 'fixed45_upfront'],
            base='Uninterrupted exact principal-core Schur CG.',
            rotation='Permutation[J,pairedT,remainder]; core columns c*eJ+s*eT, paired tail columns -s*eJ+c*eT. Zero alignment vector uses identity.',
            checkpoint_policy='Full reconstructed x10 and x9; all delayed branches receive identical x10 transformed by Q transpose. Core reconstruction is an implicit Galerkin correction, recorded separately; its work is free in this oracle.',
            stopping='Literal original b-Ax is formed and checked every iteration. No reduced-only endpoint; no residual replacement or fallback.',
            offline='Dense K,A,exact reference solution, pairing, rotations and Schur rebuilds are free oracle resources. Diagnostic runtime is not native performance. No assumed transformed n²-m² kernel-entry cost or memory advantage.',
            interpretation='Exact-error alignment uses unavailable ground truth; only headroom, not an adaptive algorithm.'),
        tiny_check=None, cases=[], failures=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
            elapsed_cpu_seconds=process_time()-cpu_started,
            offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)

    save()
    with threadpool_limits(limits=1):
        try:
            record['tiny_check'] = tiny_check()
            save()
            for dataset in ['smooth', 'mixed']:
                path = ROOT/'data/prepared'/f'synthetic-n2560-{dataset}-s0-lambda0.001.npz'
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                assert len(b) == 2048
                K = np.exp(-cdist(X, X, 'sqeuclidean')/(2*bw*bw))
                J = fps(K, CORE)
                permutation, partners, remaining = pair_geometry(K, J)
                identity = PairRotation(permutation, np.ones(CORE), np.zeros(CORE))
                fixed = PairRotation(permutation, np.full(CORE, np.sqrt(.5)), np.full(CORE, np.sqrt(.5)))
                for lam in [.001, 1e-5]:
                    if process_time()-cpu_started > 70:
                        record['status'] = 'CPU_budget_censored'
                        save()
                        return
                    A = K+lam*np.eye(len(b))
                    reference = cho_solve(cho_factor(A, lower=True), b)
                    reference_residual = float(np.linalg.norm(b-A@reference)/np.linalg.norm(b))
                    Ar, br = identity.matrix(A), identity.to_rotated(b)
                    op = previous.build(Ar, br, np.arange(CORE))
                    baseline, saved, _ = solve(A, b, identity, Ar, br, op, capture_prefix=True,
                                                deadline=cpu_started+76.)
                    case = dict(dataset=dataset, regularization=lam, n=len(b), core=CORE,
                        data_sha256=sha256(path.read_bytes()).hexdigest(),
                        original_A_sha256=sha256(A.tobytes()).hexdigest(),
                        landmarks=J.tolist(), paired_coordinates=partners.tolist(),
                        unpaired_coordinates=remaining.tolist(),
                        reference_original_relative_residual=reference_residual,
                        baseline=baseline, branches=[])
                    record['cases'].append(case)
                    save()
                    if CHECKPOINT not in saved or CHECKPOINT-1 not in saved:
                        case['status'] = 'prefix_unavailable'
                        save()
                        continue
                    x0 = saved[CHECKPOINT]
                    displacement = x0-saved[CHECKPOINT-1]
                    error = reference-x0
                    total_energy = float(error@error)
                    ordered_error = error[permutation]
                    pair_energy = float(ordered_error[:2*CORE]@ordered_error[:2*CORE])
                    outside_energy = float(ordered_error[2*CORE:]@ordered_error[2*CORE:])
                    case['checkpoint'] = dict(original_relative_residual=float(np.linalg.norm(b-A@x0)/np.linalg.norm(b)),
                        x10_sha256=sha256(x0.tobytes()).hexdigest(), x9_sha256=sha256(saved[CHECKPOINT-1].tobytes()).hexdigest(),
                        displacement_sha256=sha256(displacement.tobytes()).hexdigest(),
                        exact_error_norm=float(np.sqrt(total_energy)),
                        paired_error_norm_fraction=float(np.sqrt(pair_energy/total_energy)),
                        outside_pair_error_norm_fraction=float(np.sqrt(outside_energy/total_energy)),
                        outside_pair_error_squared_norm_fraction=float(outside_energy/total_energy))
                    branches = [('unchanged_restart', identity, x0), ('fixed45_after10', fixed, x0),
                        ('remaining_error_oracle_after10', aligned(permutation, error, CORE), x0),
                        ('recent_CG_displacement_after10', aligned(permutation, displacement, CORE), x0),
                        ('fixed45_upfront', fixed, None)]
                    del op, Ar, br
                    for label, rotation, start in branches:
                        if process_time()-cpu_started > 76:
                            case['branches'].append(dict(method=label, status='CPU_budget_not_started', converged=False))
                            save()
                            continue
                        t = perf_counter()
                        Ar, br = rotation.matrix(A), rotation.to_rotated(b)
                        asymmetry = float(np.linalg.norm(Ar-Ar.T)/np.linalg.norm(Ar))
                        op = previous.build(Ar, br, np.arange(CORE))
                        build_seconds = perf_counter()-t
                        result, _, _ = solve(A, b, rotation, Ar, br, op, x0=start,
                                             deadline=cpu_started+78.)
                        transformed_error = rotation.to_rotated(error)
                        result.update(method=label, total_prefix_plus_iterations=result['iterations']+(0 if start is None else CHECKPOINT),
                            rotation=rotation.metadata(),
                            rotated_operator_sha256=sha256(Ar.tobytes()).hexdigest(),
                            rotated_rhs_sha256=sha256(br.tobytes()).hexdigest(),
                            rotated_operator_relative_asymmetry=asymmetry,
                            checkpoint_error_core_norm_fraction=float(np.linalg.norm(transformed_error[:CORE])/np.sqrt(total_energy)),
                            checkpoint_error_paired_tail_norm_fraction=float(np.linalg.norm(transformed_error[CORE:2*CORE])/np.sqrt(total_energy)),
                            diagnostic_rotation_and_build_seconds=build_seconds)
                        case['branches'].append(result)
                        save()
                        print(json.dumps(dict(dataset=dataset, regularization=lam, method=label,
                            total_iterations=result['total_prefix_plus_iterations'], status=result['status'],
                            original_relative_residual=result['original_relative_residual'])), flush=True)
                        del op, Ar, br
                    case['status'] = 'complete'
                    save()
                    del A, reference, saved, branches, x0, displacement, error
                del K
        except Exception as error:
            record['status'] = 'failed'
            record['failures'].append(repr(error))
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
