#!/usr/bin/env python3
"""Dense protected-mean triplet rotation gate; no native resource claim."""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_pair_rotation_schur as pair

ROOT = Path(__file__).resolve().parents[1]
GROUPS, CORE, CHECKPOINT = 208, 416, 10
U = np.ones(3)/np.sqrt(3.)
PLANE = np.column_stack((np.array([1., -1., 0.])/np.sqrt(2.),
                         np.array([1., 1., -2.])/np.sqrt(6.)))


class TripletRotation:
    def __init__(self, groups, remaining, contrasts):
        self.groups = np.asarray(groups)
        self.remaining = np.asarray(remaining)
        self.v = np.asarray(contrasts)
        self.g = len(groups)
        self.m = 2*self.g
        self.n = 3*self.g+len(remaining)
        self.permutation = np.r_[self.groups.T.reshape(-1), self.remaining]
        assert len(np.unique(self.permutation)) == self.n and self.n > 3*self.g
        self.w = np.cross(np.broadcast_to(U, self.v.shape), self.v)
        self.basis = np.stack((np.broadcast_to(U, self.v.shape), self.v, self.w), axis=2)

    def to_rotated(self, vector):
        local = vector[self.groups]
        result = np.empty_like(vector)
        result[:self.g] = local@U
        result[self.g:2*self.g] = np.sum(local*self.v, axis=1)
        result[2*self.g:3*self.g] = np.sum(local*self.w, axis=1)
        result[3*self.g:] = vector[self.remaining]
        return result

    def to_original(self, vector):
        result = np.empty_like(vector)
        result[self.groups] = (vector[:self.g, None]*U+
            vector[self.g:2*self.g, None]*self.v+
            vector[2*self.g:3*self.g, None]*self.w)
        result[self.remaining] = vector[3*self.g:]
        return result

    def matrix(self, A):
        result = A[np.ix_(self.permutation, self.permutation)]
        columns = result[:, :3*self.g].reshape(self.n, 3, self.g)
        transformed = np.einsum('npg,gpc->ncg', columns, self.basis, optimize=True)
        result[:, :3*self.g] = transformed.reshape(self.n, 3*self.g)
        del columns, transformed
        rows = result[:3*self.g, :].reshape(3, self.g, self.n)
        transformed = np.einsum('gpc,pgn->cgn', self.basis, rows, optimize=True)
        result[:3*self.g, :] = transformed.reshape(3*self.g, self.n)
        return result

    def metadata(self):
        gram = np.einsum('gpi,gpj->gij', self.basis, self.basis)
        return dict(contrasts=self.v.tolist(),
            local_orthogonality_max_error=float(np.max(np.abs(gram-np.eye(3)))),
            local_mean_contrast_max_inner_product=float(np.max(np.abs(self.v@U))),
            group_indices_sha256=sha256(self.groups.tobytes()).hexdigest(),
            mean_basis_sha256=sha256(self.groups.tobytes()+U.tobytes()).hexdigest(),
            remaining_indices_sha256=sha256(self.remaining.tobytes()).hexdigest())


def geometry(K, count):
    anchors = pair.fps(K, count)
    available = np.ones(len(K), dtype=bool)
    available[anchors] = False
    groups = np.empty((count, 3), dtype=np.int64)
    groups[:, 0] = anchors
    for i, anchor in enumerate(anchors):
        for j in [1, 2]:
            candidates = np.flatnonzero(available)
            selected = int(candidates[np.argmax(K[anchor, candidates])])
            groups[i, j] = selected
            available[selected] = False
    return groups, np.flatnonzero(available)


def initial_rotation(K, groups, remaining):
    contrasts, values = [], []
    for group in groups:
        small = PLANE.T@K[np.ix_(group, group)]@PLANE
        eig, vectors = eigh((small+small.T)*.5)
        v = PLANE@vectors[:, -1]
        v /= np.linalg.norm(v)
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1.
        contrasts.append(v)
        values.append(eig.tolist())
    return TripletRotation(groups, remaining, np.asarray(contrasts)), values


def adapted_rotation(old, direction):
    local = direction[old.groups]
    centered = local-(local@U)[:, None]*U
    norms = np.linalg.norm(centered, axis=1)
    thresholds = 64*np.finfo(float).eps*np.maximum(np.linalg.norm(local, axis=1), np.finfo(float).tiny)
    keep_old = norms <= thresholds
    contrasts = old.v.copy()
    contrasts[~keep_old] = centered[~keep_old]/norms[~keep_old, None]
    rotation = TripletRotation(old.groups, old.remaining, contrasts)
    return rotation, dict(zero_centered_policy='Retain original matrix-derived contrast when ||centered d_group|| <= 64 eps max(||d_group||, tiny).',
        fallback_group_indices=np.flatnonzero(keep_old).tolist(), fallback_count=int(keep_old.sum()),
        centered_candidate_norms=norms.tolist(), numerical_zero_thresholds=thresholds.tolist(),
        mean_basis_unchanged=rotation.metadata()['mean_basis_sha256'] == old.metadata()['mean_basis_sha256'])


def tiny_check():
    rng = np.random.default_rng(622122)
    factor = rng.normal(size=(17, 17))
    A = factor@factor.T+.3*np.eye(17)
    b = rng.normal(size=17)
    groups = np.array([[7, 2, 14], [3, 5, 11], [0, 8, 16], [1, 4, 12]])
    remaining = np.setdiff1d(np.arange(17), groups.reshape(-1))
    old, _ = initial_rotation(A, groups, remaining)
    changed, info = adapted_rotation(old, rng.normal(size=17))
    fallback, fallback_info = adapted_rotation(old, np.ones(17))
    assert fallback_info['fallback_count'] == len(groups) and np.array_equal(fallback.v, old.v)
    Q = np.column_stack([changed.to_original(np.eye(17)[:, i]) for i in range(17)])
    orth_error = float(np.linalg.norm(Q.T@Q-np.eye(17)))
    Ar, br = changed.matrix(A), changed.to_rotated(b)
    transform_error = float(np.linalg.norm(Ar-Q.T@A@Q)/np.linalg.norm(A))
    oldQ = np.column_stack([old.to_original(np.eye(17)[:, i]) for i in range(17)])
    unchanged_means = bool(np.array_equal(Q[:, :len(groups)], oldQ[:, :len(groups)]))
    op = pair.previous.build(Ar, br, np.arange(changed.m))
    x0 = rng.normal(size=17)
    row, _, answer = pair.solve(A, b, changed, Ar, br, op, x0=x0)
    exact = cho_solve(cho_factor(A, lower=True), b)
    solution_error = float(np.linalg.norm(answer-exact)/np.linalg.norm(exact))
    oracle, _ = adapted_rotation(old, exact-x0)
    oracle_fine = oracle.to_rotated(exact-x0)[2*len(groups):3*len(groups)]
    oracle_error = float(np.linalg.norm(oracle_fine)/np.linalg.norm(exact-x0))
    assert max(orth_error, transform_error, oracle_error) < 1e-13
    assert unchanged_means and row['converged'] and solution_error < 1e-7
    return dict(orthogonality_error=orth_error, transformed_matrix_relative_error=transform_error,
        mean_columns_bitwise_unchanged=unchanged_means, zero_candidate_keeps_old=True,
        independent_original_solution_relative_error=solution_error,
        original_relative_residual=row['original_relative_residual'],
        error_oracle_group_fine_relative_norm=oracle_error)


def main():
    output = ROOT/'results/triplet-rotation-schur.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    prior_path = ROOT/'results/pair-rotation-schur.json'
    prior = json.loads(prior_path.read_text())
    assert prior['status'] == 'complete'
    assert all(sha256((ROOT/path).read_bytes()).hexdigest() == value for path, value in prior['sources'].items())
    sources = [Path(__file__).resolve(), prior_path, *[ROOT/p for p in prior['sources']]]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(datasets=['smooth', 'mixed'], n=2048, groups=GROUPS, core=CORE,
            regularizations=[.001, 1e-5], seed=0, checkpoint=CHECKPOINT, rtol=pair.RTOL,
            grouping='208 FPS anchors, each paired in FPS order with its two nearest unused nonanchor coordinates. Ties use original row order.',
            initial_basis='u=ones/sqrt3; dominant eigenvector of local3x3K restricted to mean-orthogonal plane gives contrast v; w=cross(u,v). Largest-magnitude entry of initial v has positive sign.',
            ordering='First208 mean columns, next208 contrasts, next208 fine columns, then untouched coordinates.',
            branches=['unchanged_restart', 'current_displacement_after10', 'remaining_error_oracle_after10'],
            baseline='Fixed matrix-derived triplet basis from start; captures reconstructed x9/x10.',
            protected_basis='Every update preserves the exact original mean columns; changes only contrasts within mean-orthogonal planes.',
            zero_fallback='Retain original contrast if centered local candidate norm <=64eps*max(local candidate norm,tiny).',
            external_controls='Previously verified coordinate416 and fixed45pair upfront iteration counts, matched operator/data hashes; not native timing comparisons.',
            offline='Dense kernels, exact reference, rotations, pairing and rebuilds are free oracle work. No native runtime, memory, or transformed-entry cost claim.',
            stopping='Reuse repaired pair solver: full original reconstruction and b-Ax check every iteration, strict1e-8; no hidden repair.',
            CPU_cap=80, wall_cap=120), tiny_check=None, cases=[], failures=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started,
            offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp = output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    save()
    with threadpool_limits(limits=1):
        try:
            record['tiny_check'] = tiny_check()
            save()
            for dataset in ['smooth', 'mixed']:
                path = ROOT/'data/prepared'/f'synthetic-n2560-{dataset}-s0-lambda0.001.npz'
                data_hash = sha256(path.read_bytes()).hexdigest()
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                assert len(b) == 2048
                K = np.exp(-cdist(X, X, 'sqeuclidean')/(2*bw*bw))
                groups, remaining = geometry(K, GROUPS)
                old, local_values = initial_rotation(K, groups, remaining)
                for lam in [.001, 1e-5]:
                    if process_time()-cpu_started > 70:
                        record['status'] = 'CPU_budget_censored'
                        save()
                        return
                    A = K+lam*np.eye(len(b))
                    A_hash = sha256(A.tobytes()).hexdigest()
                    external = next(c for c in prior['cases'] if c['dataset'] == dataset and c['regularization'] == lam)
                    assert external['data_sha256'] == data_hash and external['original_A_sha256'] == A_hash
                    pair_fixed = next(c for c in external['branches'] if c['method'] == 'fixed45_upfront')
                    reference = cho_solve(cho_factor(A, lower=True), b)
                    reference_residual = float(np.linalg.norm(b-A@reference)/np.linalg.norm(b))
                    Ar, br = old.matrix(A), old.to_rotated(b)
                    op = pair.previous.build(Ar, br, np.arange(CORE))
                    baseline, saved, _ = pair.solve(A, b, old, Ar, br, op, capture_prefix=True,
                                                    deadline=cpu_started+76.)
                    case = dict(dataset=dataset, regularization=lam, n=len(b), core=CORE,
                        data_sha256=data_hash, original_A_sha256=A_hash,
                        groups=groups.tolist(), remaining=remaining.tolist(),
                        initial_rotation=old.metadata(), local_mean_orthogonal_eigenvalues=local_values,
                        reference_original_relative_residual=reference_residual,
                        baseline=baseline, external_controls=dict(source='results/pair-rotation-schur.json',
                            coordinate416=external['baseline'], fixed45_pair_upfront=pair_fixed), branches=[])
                    record['cases'].append(case)
                    save()
                    if CHECKPOINT not in saved or CHECKPOINT-1 not in saved:
                        case['status'] = 'prefix_unavailable'
                        save()
                        continue
                    x0 = saved[CHECKPOINT]
                    displacement = x0-saved[CHECKPOINT-1]
                    error = reference-x0
                    error_norm = float(np.linalg.norm(error))
                    outside_fraction = float(np.linalg.norm(error[remaining])/error_norm)
                    case['checkpoint'] = dict(original_relative_residual=float(np.linalg.norm(b-A@x0)/np.linalg.norm(b)),
                        x10_sha256=sha256(x0.tobytes()).hexdigest(), x9_sha256=sha256(saved[CHECKPOINT-1].tobytes()).hexdigest(),
                        displacement_sha256=sha256(displacement.tobytes()).hexdigest(),
                        exact_error_norm=error_norm, outside_group_error_norm_fraction=outside_fraction,
                        outside_group_error_squared_norm_fraction=outside_fraction**2)
                    current, current_info = adapted_rotation(old, displacement)
                    oracle, oracle_info = adapted_rotation(old, error)
                    branches = [('unchanged_restart', old, dict(mean_basis_unchanged=True, fallback_count=0)),
                        ('current_displacement_after10', current, current_info),
                        ('remaining_error_oracle_after10', oracle, oracle_info)]
                    del op, Ar, br
                    for label, rotation, info in branches:
                        if process_time()-cpu_started > 76:
                            case['branches'].append(dict(method=label, status='CPU_budget_not_started', converged=False))
                            save()
                            continue
                        t = perf_counter()
                        Ar, br = rotation.matrix(A), rotation.to_rotated(b)
                        asymmetry = float(np.linalg.norm(Ar-Ar.T)/np.linalg.norm(Ar))
                        op = pair.previous.build(Ar, br, np.arange(CORE))
                        setup_seconds = perf_counter()-t
                        result, _, _ = pair.solve(A, b, rotation, Ar, br, op, x0=x0,
                                                 deadline=cpu_started+78.)
                        transformed_error = rotation.to_rotated(error)
                        result.update(method=label, total_prefix_plus_iterations=CHECKPOINT+result['iterations'],
                            rotation=rotation.metadata(), candidate=info,
                            mean_basis_unchanged=rotation.metadata()['mean_basis_sha256'] == old.metadata()['mean_basis_sha256'],
                            rotated_operator_relative_asymmetry=asymmetry,
                            rotated_operator_sha256=sha256(Ar.tobytes()).hexdigest(),
                            rotated_rhs_sha256=sha256(br.tobytes()).hexdigest(),
                            checkpoint_error_mean_norm_fraction=float(np.linalg.norm(transformed_error[:GROUPS])/error_norm),
                            checkpoint_error_contrast_norm_fraction=float(np.linalg.norm(transformed_error[GROUPS:CORE])/error_norm),
                            checkpoint_error_group_fine_norm_fraction=float(np.linalg.norm(transformed_error[CORE:3*GROUPS])/error_norm),
                            diagnostic_rotation_build_seconds=setup_seconds)
                        case['branches'].append(result)
                        save()
                        print(json.dumps(dict(dataset=dataset, regularization=lam, method=label,
                            total_iterations=result['total_prefix_plus_iterations'], status=result['status'],
                            original_relative_residual=result['original_relative_residual'])), flush=True)
                        del op, Ar, br
                    case['status'] = 'complete'
                    save()
                    del A, reference, saved, branches, x0, displacement, error, current, oracle
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
