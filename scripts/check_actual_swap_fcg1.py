#!/usr/bin/env python3
"""Dense finite-pool one-column exchange oracle with one retained A direction.

No implemented selection criterion or production performance claim follows.
Every removal is scored by its actual five/ten-step energy error. Selected
branches are independently rebuilt and continued with true residual checks.
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
sys.path.insert(0, str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel, NystromPreconditioner, initialize_rpcholesky


def stable_apply(state, R):
    U, values, lam = state.basis, state.eigenvalues, state.regularization
    c = U.T@R
    perp = R-U@c
    correction = U.T@perp
    c += correction
    perp -= U@correction
    return perp/lam+U@(c/(values[:, None]+lam))


def scalar_step(A, state, x, r, pair):
    p = state.apply(r)
    ap = A@p
    if pair is not None:
        old, aold = pair
        curv = float(old@aold)
        for _ in range(2):
            coefficient = float(old@ap)/curv
            p -= coefficient*old
            ap -= coefficient*aold
    curvature = float(p@ap)
    numerator = float(r@p)
    if curvature <= 0 or numerator <= 0:
        raise ArithmeticError('Nonpositive FCG1 curvature/progress')
    alpha = numerator/curvature
    x = x+alpha*p
    r = r-alpha*ap
    scale = np.sqrt(curvature)
    return x, r, (p/scale, ap/scale)


def scalar_run(A, b, state, snapshot, xstar, *, clear_history=False,
               limit=2000, cpu_start=None):
    x, r, pair = snapshot
    x, r = x.copy(), r.copy()
    pair = None if clear_history or pair is None else (pair[0].copy(), pair[1].copy())
    norm_b = np.linalg.norm(b)
    progress = {}
    checks = replacements = 0
    true_rel = float(np.linalg.norm(b-A@x)/norm_b)
    checks += 1
    reason = 'max_iterations'
    last_verified = 0
    for step in range(1, limit+1):
        if cpu_start is not None and process_time()-cpu_start > 104:
            step -= 1
            reason = 'cpu_guard'
            break
        x, r, pair = scalar_step(A, state, x, r, pair)
        if step in (5, 10):
            e = x-xstar
            progress[str(step)] = dict(energy=float(e@(A@e)))
        suspected = np.linalg.norm(r)/norm_b <= 1e-8
        if step % 20 == 0 or suspected or step == limit:
            true_r = b-A@x
            true_rel = float(np.linalg.norm(true_r)/norm_b)
            checks += 1
            last_verified = step
            if true_rel <= 1e-8:
                reason = 'converged'
                break
            if suspected:
                r = true_r
                pair = None
                replacements += 1
    if last_verified != step:
        true_rel = float(np.linalg.norm(b-A@x)/norm_b)
        checks += 1
    return dict(remaining_iterations=step, total_iterations=20+step,
        converged=bool(true_rel<=1e-8), true_relative_residual=true_rel,
        stop_reason=reason, full_verification_matvecs=checks,
        continuation_operator_applications=step+checks,
        residual_restarts=replacements, history_cleared=bool(clear_history),
        lookahead=progress)


def augmented_removals(K, state, candidate):
    indices = np.r_[state.indices, candidate]
    aug = NystromPreconditioner(indices, K[:, indices], state.regularization)
    W = K[np.ix_(indices, indices)]
    chol = linalg.cholesky(W, lower=True, check_finite=False)
    Winv = linalg.cho_solve((chol, True), np.eye(len(indices)), check_finite=False)
    # M_remove_i = M_aug - u_i u_i.T. The final removal is the original state.
    U = (K[:, indices]@Winv)/np.sqrt(np.diag(Winv))[None, :]
    HU = stable_apply(aug, U)
    denominator = 1-np.sum(U*HU, axis=0)
    if np.min(denominator) <= 1e-12:
        raise ArithmeticError('Unstable augmented downdate denominator')
    oldHU = stable_apply(state, U)
    g11 = float(U[:, -1]@oldHU[:, -1])
    g22 = np.sum(U[:, :-1]*oldHU[:, :-1], axis=0)
    g12 = U[:, :-1].T@oldHU[:, -1]
    discriminant = (g11+g22)**2-4*g12*g12
    if np.min(discriminant) < -1e-10*max(float(np.max((g11+g22)**2)), 1):
        raise ArithmeticError('Indefinite rank-two Gram discriminant')
    root = np.sqrt(np.maximum(discriminant, 0))
    lo = np.minimum(1, 1+.5*(g11-g22-root))
    hi = np.maximum(1, 1+.5*(g11-g22+root))
    if np.min(lo) <= 0:
        raise ArithmeticError('Nonpositive relative eigenvalue')
    return aug, U, HU, denominator, lo, hi


def batch_lookahead(A, b, aug, HU, denominator, snapshot, xstar):
    count = HU.shape[1]-1
    x, r, pair = snapshot
    X = np.repeat(x[:, None], count, axis=1)
    R = np.repeat(r[:, None], count, axis=1)
    P = np.repeat(pair[0][:, None], count, axis=1)
    AP = np.repeat(pair[1][:, None], count, axis=1)
    W = HU[:, :count]
    energies, residuals = {}, {}
    for step in range(1, 11):
        Z = stable_apply(aug, R)+W*(np.sum(W*R, axis=0)/denominator[:count])[None, :]
        AZ = A@Z
        old_curvature = np.sum(P*AP, axis=0)
        for _ in range(2):
            coefficients = np.sum(P*AZ, axis=0)/old_curvature
            Z -= P*coefficients[None, :]
            AZ -= AP*coefficients[None, :]
        curvature = np.sum(Z*AZ, axis=0)
        numerator = np.sum(R*Z, axis=0)
        if np.min(curvature) <= 0 or np.min(numerator) <= 0:
            raise ArithmeticError('Nonpositive batched FCG1 progress')
        alpha = numerator/curvature
        X += Z*alpha[None, :]
        R -= AZ*alpha[None, :]
        scale = np.sqrt(curvature)
        P, AP = Z/scale[None, :], AZ/scale[None, :]
        if step in (5, 10):
            E = X-xstar[:, None]
            AE = A@E
            energies[step] = np.sum(E*AE, axis=0)
            residuals[step] = np.linalg.norm(AE, axis=0)/np.linalg.norm(b)
    return energies, residuals


def independent_checks():
    rng = np.random.default_rng(622017)
    X = rng.normal(size=(25, 3))
    K = GaussianKernel(X, 2.).dense()
    lam = .07
    indices = np.array([0, 2, 5, 8, 11])
    old = NystromPreconditioner(indices, K[:, indices], lam)
    aug, U, HU, denom, lo, hi = augmented_removals(K, old, 19)
    old_dense = lam*np.eye(len(K))+old.basis@(old.eigenvalues[:, None]*old.basis.T)
    R = rng.normal(size=(len(K), len(indices)))
    fast = stable_apply(aug, R)+HU[:, :-1]*(np.sum(HU[:, :-1]*R, axis=0)/denom[:-1])[None, :]
    inverse_errors, eigen_errors, rank_two_errors = [], [], []
    A = K+lam*np.eye(len(K))
    b = rng.normal(size=len(K))
    xstar = linalg.solve(A, b, assume_a='pos')
    x, r, pair = np.zeros_like(b), b.copy(), None
    for _ in range(4):
        x, r, pair = scalar_step(A, old, x, r, pair)
    energies, _ = batch_lookahead(A, b, aug, HU, denom, (x, r, pair), xstar)
    batch_errors = []
    for i in range(len(indices)):
        kept = np.delete(np.r_[indices, 19], i)
        new = NystromPreconditioner(kept, K[:, kept], lam)
        new_dense = lam*np.eye(len(K))+new.basis@(new.eigenvalues[:, None]*new.basis.T)
        exact = linalg.solve(new_dense, R[:, i], assume_a='pos')
        inverse_errors.append(float(np.linalg.norm(exact-fast[:, i])/np.linalg.norm(exact)))
        ev = linalg.eigvalsh(new_dense, old_dense)
        eigen_errors.append(float(max(abs(ev[0]-lo[i]), abs(ev[-1]-hi[i]))/max(ev[-1], 1)))
        delta = np.outer(U[:, -1], U[:, -1])-np.outer(U[:, i], U[:, i])
        rank_two_errors.append(float(np.linalg.norm(new_dense-old_dense-delta)/np.linalg.norm(new_dense)))
        xx, rr, pp = x.copy(), r.copy(), (pair[0].copy(), pair[1].copy())
        for step in range(1, 11):
            xx, rr, pp = scalar_step(A, new, xx, rr, pp)
            if step in (5, 10):
                e = xx-xstar
                energy = float(e@(A@e))
                batch_errors.append(abs(energy-energies[step][i])/max(energy, 1e-30))
    assert max(inverse_errors) < 1e-11
    assert max(eigen_errors) < 1e-10
    assert max(rank_two_errors) < 1e-11
    assert max(batch_errors) < 1e-8
    return dict(max_inverse_vs_dense_relative_error=max(inverse_errors),
        max_relative_eigenvalue_error=max(eigen_errors), max_rank_two_identity_relative_error=max(rank_two_errors),
        max_batched_vs_independent_rebuild_energy_relative_error=max(batch_errors))


def category(lo, hi):
    distortion = max(hi, 1/lo)
    return 'at_most_2' if distortion <= 2 else ('between_2_and_10' if distortion <= 10 else 'above_10')


def main():
    output = ROOT/'results/actual-swap-fcg1-oracle-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (108, 110))
    start, cpu_start = perf_counter(), process_time()
    paths = [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py']
    result = dict(status='running', scope='Dense finite-pool actual-column swap oracle, not a selection method or production timing result',
        protocol=dict(regularization=.001, seed=0, ranks=[16, 32], checkpoint=20,
            initial_seed_sequence=[0, 6220, 1], candidate_uniform_seed=901,
            candidates='union of 24 uniform unused indices and 8 largest absolute true residual unused indices',
            removals='every existing landmark; exact k unique actual kernel columns retained',
            lookahead_steps=[5, 10], ranking='independently evaluated A-energy error against dense direct solution',
            final_branches='union of top five swaps at each horizon; independent rebuilt-state continuation with preserved history and with cleared history',
            recurrence='FCG1 with two-pass A orthogonalization and one normalized p/Ap pair',
            checkpoint_residual='all branches inherit identical recursive residual; pool uses separately checked true residual',
            controls=['old_uninterrupted_FCG1', 'old_history_cleared_FCG1'],
            relative_eigenvalues='exact rank-two Gram for Mnew relative to Mold; identity eigenvalues included',
            perturbation_categories='descriptive max(upper,1/lower) <=2, (2,10], >10; no selection threshold',
            rtol=1e-8, maximum_total_iterations=2000, blas_threads=1, cpu_limit_seconds=110,
            memory_note='dense A, dense reference solve, batched trajectories, and all candidate records are oracle-only allocations'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths}, cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-start, elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_checks'] = independent_checks()
        for name in ('synthetic-n640-smooth', 'concrete-observed'):
            path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data = np.load(path)
            b = data['y_train']
            kernel = GaussianKernel(data['X_train'], float(data['bandwidth']))
            K = kernel.dense()
            A = K+.001*np.eye(len(K))
            xstar = linalg.solve(A, b, assume_a='pos', check_finite=False)
            for k in (16, 32):
                if process_time()-cpu_start > 95:
                    result['status'] = 'budget_exhausted'
                    save()
                    return
                old = initialize_rpcholesky(kernel, k, .001, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                x, r, pair = np.zeros_like(b), b.copy(), None
                for _ in range(20):
                    x, r, pair = scalar_step(A, old, x, r, pair)
                snapshot = (x, r, pair)
                true_r = b-A@x
                available = np.setdiff1d(np.arange(len(b)), old.indices)
                uniform = np.random.default_rng(901).choice(available, 24, replace=False)
                residual = available[np.argsort(-np.abs(true_r[available]), kind='stable')[:8]]
                pool = np.unique(np.r_[uniform, residual])
                row = dict(dataset=name, rank=k, n=len(b), status='running',
                    prepared_input=str(path.relative_to(ROOT)), prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                    initial_indices=old.indices.tolist(), pool=pool.tolist(), uniform_candidates=uniform.tolist(),
                    residual_candidates=residual.tolist(), checkpoint_true_relative_residual=float(np.linalg.norm(true_r)/np.linalg.norm(b)),
                    checkpoint_recursive_gap=float(np.linalg.norm(true_r-r)/np.linalg.norm(b)),
                    initial_state_bytes=old.storage_bytes, retained_history_bytes=2*len(b)*8+8,
                    controls={}, trials=[], final_branches=[])
                result['cases'].append(row)
                for clear, key in ((False, 'old_uninterrupted_FCG1'), (True, 'old_history_cleared_FCG1')):
                    row['controls'][key] = scalar_run(A, b, old, snapshot, xstar, clear_history=clear, limit=1980, cpu_start=cpu_start)
                for candidate in pool:
                    if process_time()-cpu_start > 96:
                        row['status'] = 'partial_candidate_pool_cpu_guard'
                        result['status'] = 'budget_exhausted'
                        save()
                        return
                    try:
                        aug, U, HU, denominator, lo, hi = augmented_removals(K, old, int(candidate))
                        energies, residuals = batch_lookahead(A, b, aug, HU, denominator, snapshot, xstar)
                    except (ArithmeticError, linalg.LinAlgError) as error:
                        row.setdefault('numerical_skips', []).append(dict(candidate=int(candidate), reason=str(error)))
                        continue
                    for i in range(k):
                        row['trials'].append(dict(add=int(candidate), remove=int(old.indices[i]), remove_position=i,
                            energy5=float(energies[5][i]), energy10=float(energies[10][i]),
                            relative_residual5=float(residuals[5][i]), relative_residual10=float(residuals[10][i]),
                            relative_eigenvalue_min=float(lo[i]), relative_eigenvalue_max=float(hi[i]),
                            multiplicative_distortion=float(max(hi[i], 1/lo[i])),
                            relative_condition_bound=float(hi[i]/lo[i]), perturbation_category=category(lo[i], hi[i]),
                            augmented_downdate_denominator=float(denominator[i])))
                selected = {}
                for horizon in (5, 10):
                    ordered = sorted(row['trials'], key=lambda v: v[f'energy{horizon}'])
                    old_energy = row['controls']['old_uninterrupted_FCG1']['lookahead'][str(horizon)]['energy']
                    row[f'better_than_old_at_{horizon}'] = sum(v[f'energy{horizon}'] < old_energy*(1-1e-8) for v in ordered)
                    row[f'best_at_{horizon}'] = ordered[0]
                    for place, trial in enumerate(ordered[:5], start=1):
                        key = (trial['add'], trial['remove'])
                        selected.setdefault(key, dict(trial=trial, selected_by=[]))['selected_by'].append(dict(horizon=horizon, place=place))
                save()
                for entry in selected.values():
                    if process_time()-cpu_start > 102:
                        row['status'] = 'partial_final_branches_cpu_guard'
                        result['status'] = 'budget_exhausted'
                        save()
                        return
                    trial = entry['trial']
                    indices = np.r_[np.delete(old.indices, trial['remove_position']), trial['add']]
                    new = NystromPreconditioner(indices, K[:, indices], .001)
                    final = dict(**entry, independently_rebuilt_indices=indices.tolist(), state_bytes=new.storage_bytes)
                    final['preserve_history'] = scalar_run(A, b, new, snapshot, xstar, clear_history=False, limit=1980, cpu_start=cpu_start)
                    final['clear_history'] = scalar_run(A, b, new, snapshot, xstar, clear_history=True, limit=1980, cpu_start=cpu_start)
                    errors = []
                    for horizon in (5, 10):
                        actual = final['preserve_history']['lookahead'].get(str(horizon), {}).get('energy')
                        if actual is not None:
                            errors.append(abs(actual-trial[f'energy{horizon}'])/max(actual, 1e-300))
                    final['batched_vs_rebuilt_max_energy_relative_error'] = max(errors, default=0.)
                    row['final_branches'].append(final)
                    save()
                row['status'] = 'complete'
                save()
                print(json.dumps(dict(dataset=name, rank=k, trials=len(row['trials']),
                    old_iterations=row['controls']['old_uninterrupted_FCG1']['total_iterations'],
                    old_restart_iterations=row['controls']['old_history_cleared_FCG1']['total_iterations'],
                    better5=row['better_than_old_at_5'], better10=row['better_than_old_at_10'],
                    selected_count=len(row['final_branches']),
                    best_selected_iterations=min(v['preserve_history']['total_iterations'] for v in row['final_branches']),
                    cpu_seconds=process_time()-cpu_start)), flush=True)
        result['status'] = 'complete'
        save()
        print(json.dumps(dict(status='complete', cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__ == '__main__':
    main()
