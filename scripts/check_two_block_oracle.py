#!/usr/bin/env python3
"""Dense two-block commuting-preconditioner oracle, not a deployable method.

All branches share exact dense leaf factors and the entire coupling SVD bank.
PCG runs in an orthogonal eigenbasis of the whitened full operator. Stopping
uses the original residual norm and is confirmed by a dense original A product.
"""
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
sys.path.insert(0, str(ROOT / 'src'))
from nystrom_project.numerics import GaussianKernel


class TwoBlockOracle:
    def __init__(self, A, permutation):
        self.A, self.permutation = A, permutation
        self.n = len(A)
        assert self.n % 2 == 0, 'This diagnostic uses equal-sized blocks.'
        self.half = m = self.n // 2
        p, q = permutation[:m], permutation[m:]
        self.L1 = linalg.cholesky(A[np.ix_(p, p)], lower=True, check_finite=False)
        self.L2 = linalg.cholesky(A[np.ix_(q, q)], lower=True, check_finite=False)
        cross = linalg.solve_triangular(self.L1, A[np.ix_(p, q)], lower=True, check_finite=False)
        cross = linalg.solve_triangular(self.L2, cross.T, lower=True, check_finite=False).T
        self.U, self.sigma, vt = linalg.svd(cross, full_matrices=False, check_finite=False)
        self.V = vt.T.copy()
        self.diagonal = np.r_[1 + self.sigma, 1 - self.sigma]
        if np.min(self.diagonal) <= 0:
            raise ArithmeticError('Exact whitened operator lost positive definiteness; no clipping allowed.')
        # Extra dense oracle cache, disclosed separately; no cheap production norm claim.
        self.LU, self.LV = self.L1 @ self.U, self.L2 @ self.V
        self.checks = dict(minimum_whitened_eigenvalue=float(self.diagonal.min()),
                           maximum_coupling_singular_value=float(self.sigma[0]),
                           coupling_svd_relative_error=float(np.linalg.norm(cross - (self.U*self.sigma)@self.V.T)/np.linalg.norm(cross)),
                           U_orthogonality=float(np.linalg.norm(self.U.T@self.U-np.eye(m), ord=2)),
                           V_orthogonality=float(np.linalg.norm(self.V.T@self.V-np.eye(m), ord=2)))

    def rhs_to_modes(self, r):
        m = self.half
        a = self.U.T @ linalg.solve_triangular(self.L1, r[self.permutation[:m]], lower=True, check_finite=False)
        b = self.V.T @ linalg.solve_triangular(self.L2, r[self.permutation[m:]], lower=True, check_finite=False)
        return np.r_[(a+b)/np.sqrt(2), (a-b)/np.sqrt(2)]

    def mode_residual_to_original(self, r):
        m = self.half
        value = np.empty(self.n)
        value[self.permutation[:m]] = self.LU @ ((r[:m]+r[m:])/np.sqrt(2))
        value[self.permutation[m:]] = self.LV @ ((r[:m]-r[m:])/np.sqrt(2))
        return value

    def solution_from_modes(self, x):
        m = self.half
        value = np.empty(self.n)
        value[self.permutation[:m]] = linalg.solve_triangular(
            self.L1.T, self.U@((x[:m]+x[m:])/np.sqrt(2)), lower=False, check_finite=False)
        value[self.permutation[m:]] = linalg.solve_triangular(
            self.L2.T, self.V@((x[:m]-x[m:])/np.sqrt(2)), lower=False, check_finite=False)
        return value

    def inverse_diagonal(self, subset):
        inv = np.ones(self.n)
        inv[subset] = 1/self.diagonal[subset]
        inv[self.half+subset] = 1/self.diagonal[self.half+subset]
        return inv

    def importance(self, r):
        m = self.half
        return self.sigma**2*(r[:m]**2/self.diagonal[:m]+r[m:]**2/self.diagonal[m:])

    def storage(self, k):
        return dict(
            exact_dense_leaf_factor_bytes=self.L1.nbytes+self.L2.nbytes,
            full_candidate_singular_basis_bytes=self.U.nbytes+self.V.nbytes+self.sigma.nbytes,
            extra_original_residual_mapping_cache_bytes=self.LU.nbytes+self.LV.nbytes,
            operator_eigenvalues_bytes=self.diagonal.nbytes,
            permutation_bytes=self.permutation.nbytes,
            active_subset_index_bytes=8*k,
            active_inverse_diagonal_bytes=8*self.n,
            dense_original_operator_for_verification_bytes=self.A.nbytes,
            setup_extra_whitened_cross_bytes=8*self.half*self.half,
            note='All listed dense factors/banks are shared equally across methods. Setup SVD/LAPACK workspaces are additional; this is not a peak-memory matched production method.')


def solve(oracle, b, k, method, cpu_start):
    schedules = dict(static=[], static_restart8=[8], eta_once8=[8],
                     static_restart8_16_24=[8, 16, 24], eta_periodic8_3=[8, 16, 24])
    checkpoints = schedules[method]
    h = oracle.rhs_to_modes(b)
    x = np.zeros_like(h)
    r = h.copy()
    subset = np.arange(k)
    inverse = oracle.inverse_diagonal(subset)
    z = inverse*r
    p, rho = z.copy(), float(r@z)
    norm_b = np.linalg.norm(b)
    events, progress = [], []
    verification_matvecs = spectral_matvecs = norm_mappings = replacements = 0
    true_rel = 1.
    last_verified = -1
    stopped = 'max_iterations'
    started = perf_counter()
    def verify(iteration):
        nonlocal verification_matvecs, last_verified, true_rel
        full_x = oracle.solution_from_modes(x)
        full_r = b-oracle.A@full_x
        verification_matvecs += 1
        last_verified = iteration
        true_rel = float(np.linalg.norm(full_r)/norm_b)
        gap = float(np.linalg.norm(full_r-oracle.mode_residual_to_original(r))/norm_b)
        progress.append(dict(iteration=iteration, true_relative_residual=true_rel, original_recursive_residual_gap=gap))
        return full_r
    for iteration in range(1, 2001):
        if process_time()-cpu_start > 70:
            stopped = 'global_cpu_guard'
            iteration -= 1
            break
        ap = oracle.diagonal*p
        spectral_matvecs += 1
        curvature = float(p@ap)
        if min(rho, curvature) <= 0 or not np.isfinite(rho+curvature):
            stopped = 'nonpositive_inner_product'
            iteration -= 1
            break
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        original_proxy = np.linalg.norm(oracle.mode_residual_to_original(r))/norm_b
        norm_mappings += 1
        is_checkpoint = iteration in checkpoints
        suspected = original_proxy <= 1e-8
        full_r = None
        if iteration % 20 == 0 or suspected or is_checkpoint or iteration == 2000:
            full_r = verify(iteration)
            if true_rel <= 1e-8:
                stopped = 'converged'
                break
        if is_checkpoint:
            r = oracle.rhs_to_modes(full_r)
            scores = oracle.importance(r)
            old = subset.copy()
            omitted = np.ones(oracle.half, dtype=bool)
            omitted[old] = False
            before = float(scores[omitted].sum())
            if method.startswith('eta_'):
                subset = np.sort(np.argsort(-scores, kind='stable')[:k])
            omitted[:] = True
            omitted[subset] = False
            after = float(scores[omitted].sum())
            inverse = oracle.inverse_diagonal(subset)
            direct_error = float(np.sum(oracle.diagonal*((inverse-1/oracle.diagonal)*r)**2))
            assert abs(direct_error-after) <= 1e-10*max(float(scores.sum()), 1e-300)
            events.append(dict(iteration=iteration, old_subset=old.tolist(), new_subset=subset.tolist(),
                exchanged_modes=int(len(np.setdiff1d(subset, old))), inverse_action_energy_before=before,
                inverse_action_energy_after=after, direct_spectral_energy_after=direct_error,
                retained_sigma_min=float(oracle.sigma[subset].min()), retained_sigma_max=float(oracle.sigma[subset].max()),
                true_relative_residual=true_rel, restart=True))
            z = inverse*r
            p, rho = z.copy(), float(r@z)
            continue
        if suspected:
            r = oracle.rhs_to_modes(full_r)
            z = inverse*r
            p, rho = z.copy(), float(r@z)
            replacements += 1
            continue
        z = inverse*r
        new_rho = float(r@z)
        p = z+(new_rho/rho)*p
        rho = new_rho
    if last_verified != iteration:
        verify(iteration)
    return dict(method=method, active_modes=k, iterations=iteration, converged=bool(true_rel<=1e-8),
        true_relative_residual=true_rel, stop_reason=stopped,
        spectral_full_operator_applications=spectral_matvecs,
        original_full_verification_matvecs=verification_matvecs,
        conceptual_operator_applications_including_verification=spectral_matvecs+verification_matvecs,
        dense_original_norm_mapping_applications=norm_mappings,
        schedule_restarts=len(events), numerical_residual_restarts=replacements,
        accepted_subset_changes=sum(e['exchanged_modes']>0 for e in events),
        total_exchanged_modes=sum(e['exchanged_modes'] for e in events),
        initial_subset=list(range(k)), final_subset=subset.tolist(), events=events, progress=progress,
        oracle_solve_wall_seconds=perf_counter()-started)


def independent_checks():
    rng = np.random.default_rng(622034)
    F = rng.normal(size=(12, 12))
    A = F@F.T+.3*np.eye(12)
    o = TwoBlockOracle(A, np.arange(12))
    m = o.half
    L = linalg.block_diag(o.L1, o.L2)
    B = linalg.solve_triangular(L, A, lower=True, check_finite=False)
    B = linalg.solve_triangular(L, B.T, lower=True, check_finite=False).T
    subset = np.array([0, 2, 4])
    cross = (o.U[:, subset]*o.sigma[subset])@o.V[:, subset].T
    M = np.block([[np.eye(m), cross], [cross.T, np.eye(m)]])
    H = linalg.solve(M, np.eye(12), assume_a='pos')
    q = rng.normal(size=12)
    a, b = o.U.T@q[:m], o.V.T@q[m:]
    modal_r = np.r_[(a+b)/np.sqrt(2), (a-b)/np.sqrt(2)]
    eta = o.importance(modal_r)
    omitted = np.ones(m, dtype=bool)
    omitted[subset] = False
    error = H@q-linalg.solve(B, q, assume_a='pos')
    direct = float(error@B@error)
    formula = float(eta[omitted].sum())
    commutator = float(np.linalg.norm(B@H-H@B)/(np.linalg.norm(B)*np.linalg.norm(H)))
    physical_r = L@q
    mapping_error = float(np.linalg.norm(physical_r-o.mode_residual_to_original(modal_r))/np.linalg.norm(physical_r))
    mode_x = rng.normal(size=12)
    actual_x = o.solution_from_modes(mode_x)
    operator_error = float(np.linalg.norm(A@actual_x-o.mode_residual_to_original(o.diagonal*mode_x))/np.linalg.norm(A@actual_x))
    assert abs(formula-direct)/direct < 1e-11
    assert commutator < 1e-13
    assert mapping_error < 1e-13 and operator_error < 1e-12
    linalg.cholesky(M, lower=True)
    return dict(dense_inverse_energy=direct, eta_sum=formula, eta_relative_error=abs(formula-direct)/direct,
        normalized_commutator=commutator, preconditioner_minimum_eigenvalue=float(linalg.eigvalsh(M)[0]),
        preconditioner_cholesky_spd=True, original_residual_mapping_relative_error=mapping_error,
        spectral_vs_original_operator_relative_error=operator_error)


def main():
    output = ROOT/'results/two-block-oracle-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (73, 75))
    started, cpu_start = perf_counter(), process_time()
    result = dict(status='running', scope='Two-block dense commuting oracle only; no hierarchical, production, runtime, or practical memory advantage claimed',
        protocol=dict(active_modes=[16, 32], regularizations=[.001], optional_regularization=1e-5,
            extension_rule='Run small-lambda batch only if primary batch uses at most 28 CPU seconds and no static case converges before checkpoint 8',
            partition='stable sort along feature with largest training range, split at median; no target dependence',
            methods=['static', 'static_restart8', 'eta_once8', 'static_restart8_16_24', 'eta_periodic8_3'],
            first_checkpoint=8, periodic_checkpoints=[8, 16, 24], maximum_subset_updates=3,
            static_subset='largest coupling singular values', adaptive_subset='largest exact current inverse-action-energy contributions',
            rtol=1e-8, stopping='original full-system relative residual, independently confirmed using dense A',
            recurrence='ordinary PCG in fixed orthogonal spectral coordinates of exact whitened operator; true-residual replacement at matched restarts',
            max_iterations=2000, blas_threads=1, cpu_limit_seconds=75,
            work_note='A diagonal spectral application counts as one conceptual full-operator application; actual oracle timing is not a production cost. Dense norm mappings, setup, and original verification products are separately recorded.',
            storage_note='All methods retain the same complete singular-vector bank and exact dense leaf factors, irrespective of active k. Both are unrealistic quadratic storage.'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py']}, cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_checks'] = independent_checks()
        for lam in (.001, 1e-5):
            if lam == 1e-5:
                if any(v['method']=='static' and v['converged'] and v['iterations']<8
                       for case in result['cases'] for v in case['runs']):
                    result['optional_batch'] = 'not_run_static_case_converged_before_checkpoint'
                    break
                if process_time()-cpu_start > 28:
                    result['optional_batch'] = 'not_run_insufficient_predeclared_cpu_reserve'
                    break
                result['protocol']['regularizations'].append(lam)
                result['optional_batch'] = 'run_under_predeclared_cpu_reserve'
            for name in ('synthetic-n640-smooth', 'synthetic-n640-random', 'concrete-observed', 'airfoil-observed'):
                if process_time()-cpu_start > 65:
                    result['status'] = 'budget_exhausted'
                    save()
                    return
                setup_start = perf_counter()
                path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
                data = np.load(path)
                X, b = data['X_train'], data['y_train']
                axis = int(np.argmax(np.ptp(X, axis=0)))
                permutation = np.argsort(X[:, axis], kind='stable')
                K = GaussianKernel(X, float(data['bandwidth'])).dense()
                K[np.diag_indices(len(K))] += lam
                oracle = TwoBlockOracle(K, permutation)
                setup_seconds = perf_counter()-setup_start
                row = dict(dataset=name, n=len(b), regularization=lam, actual_operator_regularization=lam,
                    prepared_input=str(path.relative_to(ROOT)), prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                    preparation_regularization=.001, stored_reference_solution_ignored=True,
                    partition_axis=axis, partition_axis_range=float(np.ptp(X[:, axis])),
                    partition_indices=permutation.tolist(), partition_sha256=sha256(permutation.tobytes()).hexdigest(),
                    oracle_setup_wall_seconds=setup_seconds, oracle_checks=oracle.checks,
                    storage_by_active_modes={str(k):oracle.storage(k) for k in (16, 32)}, runs=[])
                result['cases'].append(row)
                for k in (16, 32):
                    for method in result['protocol']['methods']:
                        value = solve(oracle, b, k, method, cpu_start)
                        row['runs'].append(value)
                        save()
                        print(json.dumps(dict(dataset=name, regularization=lam, k=k, method=method,
                            iterations=value['iterations'], true_relative_residual=value['true_relative_residual'],
                            converged=value['converged'], subset_changes=value['accepted_subset_changes'],
                            operator_work=value['conceptual_operator_applications_including_verification'])), flush=True)
                        if value['stop_reason'] == 'global_cpu_guard':
                            result['status'] = 'budget_exhausted'
                            save()
                            return
        result['status'] = 'complete'
        save()
        print(json.dumps(dict(status=result['status'], cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__ == '__main__':
    main()
