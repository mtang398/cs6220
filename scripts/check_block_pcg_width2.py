"""Width-two fixed-preconditioner block CG; dense offline accuracy diagnostic.

The two RHS are b and one independent, predeclared Gaussian vector. This is
ordinary block CG with a QR-normalized search block, not multi-preconditioned
CG. Only the preceding search block is retained. Dense A is an offline oracle;
its multiplication counters describe possible kernel passes, not CPU timings.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nystrom_project.numerics import GaussianKernel
from nystrom_project.preconditioners import LowRankPreconditioner
from check_projected_balancing import pcg
from check_power_depth_headroom import refine


class CountedOperator:
    def __init__(self, matrix):
        self.matrix = matrix
        self.calls = self.rhs = 0
        self.widths = {}

    def __matmul__(self, value):
        width = 1 if value.ndim == 1 else value.shape[1]
        self.calls += 1
        self.rhs += width
        self.widths[str(width)] = self.widths.get(str(width), 0) + 1
        return self.matrix @ value


class CountedInverse:
    def __init__(self, state):
        self.state = state
        self.calls = self.rhs = 0

    def apply(self, value):
        self.calls += 1
        self.rhs += 1 if value.ndim == 1 else value.shape[1]
        if value.ndim == 1:
            return self.state.apply(value)
        if not isinstance(self.state, LowRankPreconditioner):
            return self.state.apply(value)
        U, w, lam = self.state.basis, self.state.eigenvalues, self.state.regularization
        coefficients = U.T @ value
        perpendicular = value - U @ coefficients
        correction = U.T @ perpendicular
        coefficients += correction
        perpendicular -= U @ correction
        return perpendicular / lam + U @ (coefficients / (w[:, None] + lam))


def allocated_bytes(*arrays):
    """Unique NumPy backing allocations; views do not get counted twice."""
    roots = {}
    for value in arrays:
        if value is None:
            continue
        base = value
        while isinstance(base.base, np.ndarray):
            base = base.base
        roots[id(base)] = base.nbytes
    return int(sum(roots.values()))


def orthonormal_search(Z):
    """QR plus a width-at-most-two SVD exposes numerical dependence."""
    Q, R, permutation = linalg.qr(Z, mode="economic", pivoting=True, check_finite=False)
    left, singular, _ = linalg.svd(R, full_matrices=False, check_finite=False)
    threshold = 64 * np.finfo(float).eps * max(Z.shape) * singular[0]
    keep = singular > threshold
    return Q @ left[:, keep], singular, float(threshold)


def block_pcg(A, B, inverse, *, rtol=1e-8, maxiter=2500, capture=False):
    """Short-recurrence block CG, Euclidean-QR search blocks and local reorth.

    P_new spans H R_new - P (P^T A P)^-1 (AP)^T H R_new.
    In exact arithmetic fixed H makes this conjugate to every earlier block.
    A second projection against the same previous block reduces rounding error;
    no older vectors or a full Krylov basis are retained. Each block updates all
    currently active RHS; true convergence of the auxiliary RHS deflates it.
    """
    B = np.asarray(B)
    assert B.ndim == 2 and 1 <= B.shape[1] <= 2
    n, initial_width = B.shape
    active = list(range(initial_width))
    norms = np.linalg.norm(B, axis=0)
    assert np.all(norms > 0)
    X = np.zeros_like(B)
    R = B.copy()
    Z = inverse.apply(R)
    P, singular, threshold = orthonormal_search(Z)
    del Z
    retained_peak = allocated_bytes(B, X, R, P)
    # At QR, both Z and Q plus output P may be live. Two width-sized work
    # arrays additionally cover dense BLAS expression temporaries. This is
    # an explicit upper bound for n-vector work, not measured process RSS.
    workspace_peak_bound = retained_peak + 4 * n * initial_width * 8 + 16 * initial_width**2 * 8
    deflations, rank_losses, replacements = [], [], []
    capture_x = []
    max_local_conjugacy = 0.
    max_small_condition = 1.
    true_checks = 0
    converged, breakdown = False, None
    primary_x = None
    iteration = 0
    for iteration in range(1, maxiter + 1):
        if P.shape[1] == 0:
            breakdown = "zero_search_rank"
            break
        AP = A @ P
        S = (P.T @ AP + AP.T @ P) * .5
        retained_peak = max(retained_peak, allocated_bytes(B, X, R, P, AP, S))
        workspace_peak_bound = max(workspace_peak_bound,
            allocated_bytes(B, X, R, P, AP, S) + 4 * n * initial_width * 8 + 16 * initial_width**2 * 8)
        try:
            chol = linalg.cho_factor(S, lower=True, check_finite=False)
            max_small_condition = max(max_small_condition, float(np.linalg.cond(S)))
            alpha = linalg.cho_solve(chol, P.T @ R, check_finite=False)
        except np.linalg.LinAlgError:
            breakdown = "nonpositive_search_gram"
            break
        X += P @ alpha
        R -= AP @ alpha
        if capture:
            capture_x.append(X[:, active.index(0)].copy())
        recursive = np.linalg.norm(R, axis=0) / norms[active]
        suspected = recursive <= rtol
        restart = False
        if np.any(suspected) or iteration % 20 == 0 or iteration == maxiter:
            true_R = B[:, active] - A @ X
            true_checks += 1
            relative = np.linalg.norm(true_R, axis=0) / norms[active]
            if relative[active.index(0)] <= rtol:
                converged = True
                break
            # True auxiliary convergence removes its RHS and solution state.
            if 1 in active and relative[active.index(1)] <= rtol:
                j = active.index(0)
                deflations.append(dict(iteration=iteration, rhs=1,
                    true_relative_residual=float(relative[active.index(1)])))
                X, R = X[:, j:j+1].copy(), R[:, j:j+1].copy()
                true_R = true_R[:, j:j+1].copy()
                suspected = suspected[j:j+1]
                relative = relative[j:j+1]
                active = [0]
            if np.any(suspected & (relative > rtol)):
                R = true_R.copy()
                replacements.append(iteration)
                restart = True
            del true_R
        Z = inverse.apply(R)
        if not restart:
            for _ in range(2):
                coefficients = linalg.cho_solve(chol, AP.T @ Z, check_finite=False)
                Z -= P @ coefficients
        P_next, singular, threshold = orthonormal_search(Z)
        if P_next.shape[1] < len(active):
            rank_losses.append(dict(iteration=iteration,
                search_rank=P_next.shape[1], active_rhs=len(active),
                singular_values=singular.tolist(), threshold=threshold))
        if not restart and P_next.shape[1]:
            # Normalized cross-Gram error, measured from already-known AP.
            numerator = np.linalg.norm(AP.T @ P_next)
            denominator = np.linalg.norm(AP) * np.linalg.norm(P_next)
            max_local_conjugacy = max(max_local_conjugacy, float(numerator / max(denominator, 1e-300)))
        workspace_peak_bound = max(workspace_peak_bound,
            allocated_bytes(B, X, R, P, AP, Z, P_next, S, alpha)
            + 2 * n * initial_width * 8 + 16 * initial_width**2 * 8)
        P = P_next
        del Z, P_next
    primary_x = X[:, active.index(0)].copy()
    # Always charge independent final verification, matching the scalar helper.
    final_residual = B[:, 0] - A @ primary_x
    final_relative = float(np.linalg.norm(final_residual) / norms[0])
    converged = bool(final_relative <= rtol)
    return dict(iterations=iteration, converged=converged,
        true_relative_residual=final_relative, breakdown=breakdown,
        auxiliary_deflations=deflations, numerical_rank_losses=rank_losses,
        residual_replacement_restarts=replacements, true_block_checks=true_checks,
        max_previous_block_conjugacy_relative=max_local_conjugacy,
        max_small_gram_condition=max_small_condition,
        solver_retained_array_bytes_peak=retained_peak,
        solver_array_workspace_bytes_upper_bound=workspace_peak_bound,
        initial_rhs_width=initial_width,
        final_active_rhs=len(active)), primary_x, capture_x


class DenseInverse:
    def __init__(self, M):
        self.factor = linalg.cho_factor(M, lower=True)
    def apply(self, value):
        return linalg.cho_solve(self.factor, value)


def scalar_iterates(A, b, H, count):
    x = np.zeros_like(b)
    r = b.copy()
    z = H.apply(r)
    p, rho = z.copy(), float(r @ z)
    values = []
    for _ in range(count):
        ap = A @ p
        alpha = rho / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        values.append(x.copy())
        z = H.apply(r)
        new_rho = float(r @ z)
        p = z + (new_rho / rho) * p
        rho = new_rho
    return values


def validations():
    rng = np.random.default_rng(8135)
    n = 30
    V, _ = linalg.qr(rng.normal(size=(n, n)))
    A = (V * np.geomspace(1., 80., n)) @ V.T
    T = rng.normal(size=(n, n))
    M = np.eye(n) + .03 * T @ T.T
    H = DenseInverse(M)
    B = rng.normal(size=(n, 2))
    run, x, captured = block_pcg(CountedOperator(A), B, CountedInverse(H), rtol=1e-11, maxiter=100, capture=True)
    exact = linalg.solve(A, B[:, 0], assume_a="pos")
    solution_error = float(np.linalg.norm(x-exact) / np.linalg.norm(exact))
    assert run['converged'] and solution_error < 1e-10, (run, solution_error)
    # Independent block Krylov Galerkin minimizers for the first four steps.
    raw, galerkin_errors = H.apply(B), []
    previous = []
    for step in range(4):
        previous.append(raw)
        Q, _ = linalg.qr(np.column_stack(previous), mode="economic")
        direct = Q @ linalg.solve(Q.T @ A @ Q, Q.T @ B, assume_a="pos")
        galerkin_errors.append(float(np.linalg.norm(captured[step]-direct[:, 0]) / np.linalg.norm(direct[:, 0])))
        raw = H.apply(A @ raw)
    assert max(galerkin_errors) < 1e-10, galerkin_errors
    _, _, width1 = block_pcg(CountedOperator(A), B[:, :1], CountedInverse(H), rtol=1e-14, maxiter=7, capture=True)
    scalar = scalar_iterates(A, B[:, 0], H, 7)
    width1_errors = [float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x, y in zip(width1, scalar)]
    assert max(width1_errors) < 1e-10, width1_errors
    diagonal = np.diag(np.linspace(1., 4., 18))
    rhs = np.column_stack((rng.normal(size=18), np.eye(18)[:, 0]))
    deflated, _, _ = block_pcg(CountedOperator(diagonal), rhs, CountedInverse(DenseInverse(np.eye(18))), rtol=1e-11, maxiter=60)
    assert deflated['converged'] and deflated['auxiliary_deflations'][0]['iteration'] == 1, deflated
    return dict(independent_dense_solution_relative_error=solution_error,
        independent_block_krylov_galerkin_errors_first4=galerkin_errors,
        scalar_width1_iterate_errors_first7=width1_errors,
        auxiliary_eigenvector_rhs_deflation=deflated,
        primary_source='https://etna.ricam.oeaw.ac.at/vol.65.2026/pp63-92.dir/pp63-92.pdf',
        source_scope='Chen et al. Section 3.2 distinguishes full reorthogonalization from memory-limited short recurrences; the latter is independently validated here.')


def main():
    output = ROOT / 'results/block-pcg-width2-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (98, 100))
    start, cpu_start = perf_counter(), process_time()
    baseline_path = ROOT / 'results/power-depth-headroom-diagnostic.json'
    baseline = json.loads(baseline_path.read_text())
    archived = {(c['dataset'], c['regularization'], r['power_passes']): r
        for c in baseline['cases'] for r in c['runs'] if r['method'] == 'power_upfront'}
    record = dict(status='running',
        scope='Established width-two block CG control; dense offline action counts, no production timing or novelty claim',
        protocol=dict(datasets=['synthetic-n640-smooth', 'synthetic-n640-random', 'synthetic-n640-mixed', 'concrete-observed', 'airfoil-observed'],
            seed=0, rank=32, regularizations=[.001, 1e-5], power_passes=[2, 4, 8],
            initial_block='[b, independent Gaussian g]', auxiliary_seed_sequence=[0, 6220, 71],
            power_seed_sequence=[0, 6220, 3], width=2, rtol=1e-8, maxiter=2500,
            blas_threads=1, cpu_cap_seconds=100,
            true_checks='every20 iterations, either RHS suspected stop, and final independent primary check',
            search_normalization='pivoted Euclidean QR and SVD of width<=2 triangular factor',
            recurrence='project preconditioned residual against immediately preceding search block twice; no older history',
            rank_threshold='64 eps max(n,width) sigma_max',
            deflation='remove auxiliary RHS only after true relative residual<=1e-8; retain previous search block for next conjugation',
            memory='exact named retained ndarray backing allocations plus conservative explicitly bounded ndarray workspace; input RHS counted; dense oracle K/A excluded and separately disclosed',
            matrix_load_model='each A@block is one possible kernel pass with that many scalar RHS; setup depth passes each width32'),
        reference_result_sha256=sha256(baseline_path.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),
            ROOT/'scripts/check_projected_balancing.py', ROOT/'scripts/check_power_depth_headroom.py',
            ROOT/'src/nystrom_project/numerics.py', ROOT/'src/nystrom_project/preconditioners.py']}, cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start, elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['validations'] = validations()
        save()
        for name in record['protocol']['datasets']:
            data_path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data = np.load(data_path)
            b = data['y_train']
            n = len(b)
            K = GaussianKernel(data['X_train'], float(data['bandwidth'])).dense()
            omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(n, 32))
            first = refine(K, omega, 2, .001)
            fourth = refine(K, first.basis, 2, .001)
            eighth = refine(K, fourth.basis, 4, .001)
            g = np.random.default_rng(np.random.SeedSequence([0, 6220, 71])).normal(size=n)
            B = np.column_stack((b, g))
            for lam in [.001, 1e-5]:
                A = K + lam*np.eye(n)
                case = dict(dataset=name, regularization=lam, n=n,
                    data_file=str(data_path.relative_to(ROOT)), data_sha256=sha256(data_path.read_bytes()).hexdigest(),
                    auxiliary_sha256=sha256(g.tobytes()).hexdigest(), dense_oracle_K_A_bytes=K.nbytes+A.nbytes, runs=[])
                record['cases'].append(case)
                for depth, base in [(2, first), (4, fourth), (8, eighth)]:
                    if process_time()-cpu_start > 88:
                        record['status']='cpu_budget'; save(); return
                    model = LowRankPreconditioner(base.basis, base.eigenvalues, lam)
                    factor_hash = sha256(model.basis.tobytes()+model.eigenvalues.tobytes()).hexdigest()
                    reference = archived[name, lam, depth]
                    assert factor_hash == reference['factor_sha256']
                    operator, inverse = CountedOperator(A), CountedInverse(model)
                    scalar, _ = pcg(operator, b, inverse, maxiter=2500)
                    scalar.update(method='ordinary_pcg', factor_sha256=factor_hash,
                        reference_iterations=reference['iterations'],
                        iterations_match_archived=scalar['iterations']==reference['iterations'])
                    # Scalar helper's fixed live vector budget: b,x,r,z,p,ap,
                    # true_r at checks. One n-vector expression temporary bound.
                    scalar.update(solver_retained_array_bytes_upper_bound=7*n*8,
                        solver_array_workspace_bytes_upper_bound=8*n*8,
                        memory_count_kind='source-derived seven persistent/check vectors plus one expression vector; not RSS')
                    runs = [(scalar, operator, inverse)]
                    operator, inverse = CountedOperator(A), CountedInverse(model)
                    blocked, _, _ = block_pcg(operator, B, inverse)
                    blocked.update(method='block_pcg_width2', factor_sha256=factor_hash,
                        memory_count_kind='unique named ndarray retained allocations; bounded QR/BLAS work arrays, not RSS')
                    runs.append((blocked, operator, inverse))
                    for run, op, inv in runs:
                        run.update(power_passes=depth, setup_kernel_passes=depth,
                            setup_scalar_rhs=32*depth, solve_kernel_passes=op.calls,
                            solve_scalar_rhs=op.rhs, solve_operator_width_histogram=op.widths,
                            total_kernel_passes=depth+op.calls, total_scalar_rhs=32*depth+op.rhs,
                            inverse_calls=inv.calls, inverse_rhs=inv.rhs,
                            factor_storage_bytes=model.storage_bytes,
                            factor_plus_solver_workspace_upper_bound_bytes=model.storage_bytes+run['solver_array_workspace_bytes_upper_bound'])
                        case['runs'].append(run)
                    save()
                    print(json.dumps(dict(dataset=name, lam=lam, depth=depth,
                        scalar=[scalar['iterations'], scalar['total_kernel_passes'], scalar['total_scalar_rhs']],
                        block=[blocked['iterations'], blocked['total_kernel_passes'], blocked['total_scalar_rhs']],
                        converged=blocked['converged'], breakdown=blocked['breakdown'])), flush=True)
    record['status']='complete'
    save()


if __name__ == '__main__':
    main()
