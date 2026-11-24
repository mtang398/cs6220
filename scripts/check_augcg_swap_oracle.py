#!/usr/bin/env python3
"""Full-projector AugCG finite-pool oracle; no production or novelty claim."""
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
from check_actual_swap_fcg1 import augmented_removals, stable_apply


def prefix(A, b, apply, steps=20):
    x, r = np.zeros_like(b), b.copy()
    z = apply(r)
    p, rho = z.copy(), float(r@z)
    directions, images = [], []
    for _ in range(steps):
        ap = A@p
        curvature = float(p@ap)
        scale = np.sqrt(curvature)
        directions.append(p/scale)
        images.append(ap/scale)
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        z = apply(r)
        next_rho = float(r@z)
        p = z+(next_rho/rho)*p
        rho = next_rho
    return x, r, p, rho, np.column_stack(directions), np.column_stack(images)


class Projector:
    def __init__(self, P, AP):
        self.P, self.AP = P, AP
        self.G = (P.T@AP+AP.T@P)*.5
        self.L = linalg.cholesky(self.G, lower=True, check_finite=False)
    def small(self, R):
        return linalg.cho_solve((self.L, True), R, check_finite=False)
    def left(self, Z):
        return Z-self.P@self.small(self.AP.T@Z)
    def right(self, R):
        return R-self.AP@self.small(self.P.T@R)
    def correct(self, A, b, x):
        x = x+self.P@self.small(self.P.T@(b-A@x))
        return x, b-A@x
    def constraint(self, r):
        return float(np.max(np.abs(self.P.T@r)/(np.linalg.norm(self.P, axis=0)*max(np.linalg.norm(r), 1e-300))))


def stable_prefix_projector(A, cholA, P):
    transformed = cholA.T@P
    basis, singular, _ = linalg.svd(transformed, full_matrices=False, check_finite=False)
    cutoff = 64*np.finfo(float).eps*max(transformed.shape)*singular[0]
    keep = singular > cutoff
    basis = basis[:, keep]
    Q = linalg.solve_triangular(cholA.T, basis, lower=False, check_finite=False)
    AQ = cholA@basis
    projector = Projector(Q, AQ)
    diagnostics = dict(method='SVD of L_A.T P followed by triangular backsolve; same numerical prefix span',
        raw_columns=P.shape[1], retained_rank=int(np.count_nonzero(keep)),
        A_metric_singular_values=singular.tolist(), singular_value_cutoff=float(cutoff),
        normalized_A_gram_error=float(np.linalg.norm(projector.G-np.eye(Q.shape[1]), ord=2)),
        operator_image_relative_error=float(np.linalg.norm(A@Q-AQ)/np.linalg.norm(AQ)),
        extra_dense_A_cholesky_bytes=cholA.nbytes)
    return projector, diagnostics


def gold_apply(state, projector, R):
    matrix = R[:, None] if R.ndim == 1 else R
    value = projector.left(stable_apply(state, projector.right(matrix)))
    return value[:, 0] if R.ndim == 1 else value


def run(A, b, apply, x0, r0, xstar, *, projector=None, p0=None, rho0=None, cpu_start=None):
    x, r = x0.copy(), r0.copy()
    if p0 is None:
        z = apply(r)
        p, rho = z.copy(), float(r@z)
    else:
        p, rho = p0.copy(), float(rho0)
    norm_b = np.linalg.norm(b)
    lookahead, progress = {}, []
    verifications = replacements = 0
    true_rel = float(np.linalg.norm(b-A@x)/norm_b)
    verifications += 1
    reason, last_verified = 'max_iterations', 0
    max_constraint = projector.constraint(r) if projector is not None else None
    for step in range(1, 1981):
        if cpu_start is not None and process_time()-cpu_start > 104:
            step -= 1
            reason = 'cpu_guard'
            break
        ap = A@p
        curvature = float(p@ap)
        if min(curvature, rho) <= 0 or not np.isfinite(curvature+rho):
            step -= 1
            reason = 'nonpositive_inner_product'
            break
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        if step in (5, 10):
            e = x-xstar
            lookahead[str(step)] = dict(energy=float(e@(A@e)))
        suspected = np.linalg.norm(r)/norm_b <= 1e-8
        if step % 20 == 0 or suspected or step == 1980:
            true_r = b-A@x
            true_rel = float(np.linalg.norm(true_r)/norm_b)
            verifications += 1
            last_verified = step
            entry = dict(step=step, true_relative_residual=true_rel)
            if projector is not None:
                entry['residual_constraint'] = projector.constraint(true_r)
                max_constraint = max(max_constraint, entry['residual_constraint'])
            progress.append(entry)
            if true_rel <= 1e-8:
                reason = 'converged'
                break
            if suspected:
                if projector is not None:
                    x, r = projector.correct(A, b, x)
                    verifications += 2
                else:
                    r = true_r
                z = apply(r)
                p, rho = z.copy(), float(r@z)
                replacements += 1
                continue
        z = apply(r)
        next_rho = float(r@z)
        p = z+(next_rho/rho)*p
        rho = next_rho
    if last_verified != step:
        true_rel = float(np.linalg.norm(b-A@x)/norm_b)
        verifications += 1
    return dict(remaining_iterations=step, total_iterations=20+step, converged=bool(true_rel<=1e-8),
        true_relative_residual=true_rel, stop_reason=reason, lookahead=lookahead,
        verification_matvecs=verifications, continuation_operator_applications=step+verifications,
        numerical_restarts=replacements, maximum_checked_residual_constraint=max_constraint, progress=progress)


def batch(A, b, aug, HU, denominator, projector, x, r, xstar):
    k = HU.shape[1]-1
    X = np.repeat(x[:, None], k, axis=1)
    R = np.repeat(r[:, None], k, axis=1)
    W = HU[:, :k]
    def apply(V):
        V = projector.right(V)
        Z = stable_apply(aug, V)+W*(np.sum(W*V, axis=0)/denominator[:k])[None, :]
        return projector.left(Z)
    Z = apply(R)
    P, rho = Z.copy(), np.sum(R*Z, axis=0)
    energies = {}
    for step in range(1, 11):
        AP = A@P
        curvature = np.sum(P*AP, axis=0)
        if min(float(curvature.min()), float(rho.min())) <= 0:
            raise ArithmeticError('Nonpositive batched constrained PCG curvature')
        alpha = rho/curvature
        X += P*alpha[None, :]
        R -= AP*alpha[None, :]
        if step in (5, 10):
            E = X-xstar[:, None]
            energies[step] = np.sum(E*(A@E), axis=0)
        Z = apply(R)
        next_rho = np.sum(R*Z, axis=0)
        P = Z+P*(next_rho/rho)[None, :]
        rho = next_rho
    return energies


def boundary_action(old_apply, last, alast, U, V, r):
    h = old_apply(r)
    h -= last*float(alast@h)/float(last@alast)
    return h+V[:, 0]*float(V[:, 0]@r)-V[:, 1]*float(V[:, 1]@r)


def independent_checks():
    rng = np.random.default_rng(622019)
    F, Q = rng.normal(size=(30, 30)), rng.normal(size=(30, 8))
    A = F@F.T+3*np.eye(30)
    M = np.eye(30)+.1*Q@Q.T
    H = linalg.solve(M, np.eye(30), assume_a='pos')
    b = rng.normal(size=30)
    x, r, _, _, P, AP = prefix(A, b, lambda v: H@v, steps=5)
    projector = Projector(P, AP)
    candidate = .03*rng.normal(size=(30, 2))
    Hnew = H+candidate@np.diag([1., -1.])@candidate.T
    linalg.cholesky(Hnew)
    Pi = np.eye(30)-P@projector.small(AP.T)
    rr = rng.normal(size=30)
    rr -= P@linalg.solve(P.T@P, P.T@rr, assume_a='pos')
    V = projector.left(candidate)
    full = Pi@Hnew@Pi.T@rr
    short = boundary_action(lambda v: H@v, P[:, -1], AP[:, -1], candidate, V, rr)
    error = float(np.linalg.norm(full-short)/np.linalg.norm(full))
    assert error < 1e-10
    stream = candidate.copy()
    for j in range(P.shape[1]):
        stream -= np.outer(P[:, j], AP[:, j]@stream)/float(P[:, j]@AP[:, j])
    stream_error = float(np.linalg.norm(stream-V)/np.linalg.norm(V))
    assert stream_error < 1e-10
    return dict(full_projector_vs_boundary_relative_error=error,
        streamed_vs_full_projected_update_relative_error=stream_error,
        prefix_A_gram_error=float(np.linalg.norm(P.T@AP-np.eye(5), ord=2)),
        test_residual_orthogonality=projector.constraint(rr), Hnew_cholesky_spd=True)


def prefix_checks(A, old, b, x, r, P, AP, projector, corrected_r):
    rng = np.random.default_rng(622020)
    random = rng.normal(size=len(b))
    random = projector.right(random)
    errors = []
    for rr in (corrected_r, random):
        h = old.apply(rr)
        full = projector.left(h)
        short = h-P[:, -1]*float(AP[:, -1]@h)/float(P[:, -1]@AP[:, -1])
        errors.append(float(np.linalg.norm(full-short)/np.linalg.norm(full)))
    gram = P.T@AP
    return dict(A_conjugacy_spectral_error=float(np.linalg.norm(gram-np.eye(len(gram)), ord=2)),
        normalized_projector_A_gram_condition=float(np.linalg.cond(projector.G)),
        raw_prefix_A_gram_condition=float(np.linalg.cond((gram+gram.T)*.5)),
        recursive_residual_orthogonality=projector.constraint(r),
        true_residual_orthogonality=projector.constraint(b-A@x),
        corrected_residual_orthogonality=projector.constraint(corrected_r),
        boundary_vs_full_on_corrected_residual=errors[0], boundary_vs_full_on_random_orthogonal_residual=errors[1],
        checkpoint_recursive_residual_gap=float(np.linalg.norm(b-A@x-r)/np.linalg.norm(b)))


def main():
    output = ROOT/'results/augcg-swap-oracle-diagnostic.json'
    resume = '--resume' in sys.argv
    if output.exists() and not resume:
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (108, 110))
    started, cpu_start = perf_counter(), process_time()
    pool_path = ROOT/'results/actual-swap-fcg1-oracle-diagnostic.json'
    pools = json.loads(pool_path.read_text())
    sources = [Path(__file__).resolve(), ROOT/'scripts/check_actual_swap_fcg1.py', ROOT/'src/nystrom_project/numerics.py']
    result = dict(status='running', scope='Dense full-projector AugCG oracle with actual-column swaps; no production or novelty claim',
        protocol=dict(regularization=.001, ranks=[16, 32], checkpoint=20, seed=0,
            initial_seed_sequence=[0, 6220, 1], prefix='ordinary fixed-preconditioner PCG without global reorthogonalization',
            projector='Same saved prefix span, A-metric SVD normalized using dense A Cholesky; Pi formed from its stable measured Gram',
            initial_correction='same Galerkin correction for every old/new projected branch',
            projected_recurrence='ordinary PCG with Pi H Pi.T on constrained residuals',
            lookahead=[5, 10], final_branches='union of top five at each horizon, independently rebuilt states',
            short_checks='if positive, union of best one at each horizon; streamed rank-two update plus old-PCG boundary action',
            candidate_pool='exact archived FCG1 pool for smooth/Concrete; mixed uses same uniform24+residual8 rule',
            optional_mixed_rule='after first four cases complete, begin mixed cases only before75 CPU seconds',
            rtol=1e-8, max_total_iterations=2000, blas_threads=1, cpu_limit_seconds=110,
            memory_note='Full P/AP, H0, Hnew, original/projected low-rank vectors, dense A/reference solve and batch trajectories are oracle allocations; no free future-candidate projection'),
        candidate_source=dict(path=str(pool_path.relative_to(ROOT)), sha256=sha256(pool_path.read_bytes()).hexdigest()),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}, cases=[])
    prior_cpu = prior_wall = 0.
    if resume:
        current_sources = result['source_sha256']
        result = json.loads(output.read_text())
        prior_cpu, prior_wall = result['elapsed_cpu_seconds'], result['elapsed_wall_seconds']
        for case in result['cases']:
            case.setdefault('execution_source_sha256', result['source_sha256']['scripts/check_augcg_swap_oracle.py'])
        result.setdefault('source_versions', []).append(dict(source_sha256=result['source_sha256'],
            archive='results/augcg-swap-oracle-stable-projector-source-v1.zip'))
        result['source_sha256'] = current_sources
        result['status'] = 'running'
    def save():
        result.update(elapsed_wall_seconds=prior_wall+perf_counter()-started, elapsed_cpu_seconds=prior_cpu+process_time()-cpu_start)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_checks'] = independent_checks()
        for name in ('synthetic-n640-smooth', 'concrete-observed', 'synthetic-n640-mixed'):
            if all(any(c['dataset']==name and c['rank']==k and c['status']=='complete' for c in result['cases']) for k in (16, 32)):
                continue
            path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            if name.endswith('mixed') and (process_time()-cpu_start > 75 or not path.exists()):
                result['optional_mixed'] = 'not_run_budget_or_missing_prepared_input'
                break
            data = np.load(path)
            b = data['y_train']
            kernel = GaussianKernel(data['X_train'], float(data['bandwidth']))
            K = kernel.dense()
            A = K+.001*np.eye(len(b))
            cholA = linalg.cholesky(A, lower=True, check_finite=False)
            xstar = linalg.cho_solve((cholA, True), b, check_finite=False)
            for k in (16, 32):
                if any(c['dataset']==name and c['rank']==k and c['status']=='complete' for c in result['cases']):
                    continue
                if process_time()-cpu_start > 95:
                    result['status'] = 'budget_exhausted'
                    save()
                    return
                old = initialize_rpcholesky(kernel, k, .001, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                x, r, next_p, next_rho, P, AP = prefix(A, b, old.apply)
                projector, projector_diagnostics = stable_prefix_projector(A, cholA, P)
                xg, rg = projector.correct(A, b, x)
                archived = [c for c in pools['cases'] if c['dataset']==name and c['rank']==k]
                if archived:
                    assert archived[0]['initial_indices']==old.indices.tolist()
                    pool = np.asarray(archived[0]['pool'])
                    pool_provenance = 'exact archived FCG1 pool; available to this offline experiment before prefix'
                else:
                    available = np.setdiff1d(np.arange(len(b)), old.indices)
                    uniform = np.random.default_rng(901).choice(available, 24, replace=False)
                    residual = available[np.argsort(-np.abs((b-A@x)[available]), kind='stable')[:8]]
                    pool = np.unique(np.r_[uniform, residual])
                    pool_provenance = 'mixed oracle pool uses checkpoint residual; a production stream would need an earlier declared bank or charge replay/projection'
                row = dict(dataset=name, rank=k, n=len(b), status='running', initial_indices=old.indices.tolist(),
                    execution_source_sha256=result['source_sha256']['scripts/check_augcg_swap_oracle.py'],
                    prepared_input=str(path.relative_to(ROOT)), prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                    pool=pool.tolist(), pool_provenance=pool_provenance,
                    stable_projector=projector_diagnostics,
                    prefix_checks=prefix_checks(A, old, b, x, r, P, AP, projector, rg),
                    checkpoint_energy=float((x-xstar)@(A@(x-xstar))),
                    corrected_checkpoint_energy=float((xg-xstar)@(A@(xg-xstar))),
                    storage=dict(H0_state_bytes=old.storage_bytes, full_prefix_P_AP_bytes=P.nbytes+AP.nbytes,
                        normalized_projector_P_AP_bytes=projector.P.nbytes+projector.AP.nbytes,
                        full_prefix_Gram_factor_bytes=projector.G.nbytes+projector.L.nbytes,
                        dense_A_cholesky_bytes=cholA.nbytes,
                        single_swap_original_U_bytes=2*len(b)*8, single_swap_projected_V_bytes=2*len(b)*8,
                        boundary_p_Ap_bytes=2*len(b)*8,
                        predeclared_original_bank_bytes=len(pool)*(k+1)*len(b)*8,
                        predeclared_projected_bank_bytes=len(pool)*(k+1)*len(b)*8,
                        note='Bank exploits sharing of one negative-update vector per added candidate. Original plus projected banks count separately; gold also retains full history. Dense matrices and batch workspaces are additional.'),
                    controls={}, trials=[], final_branches=[], streamed_checks=[])
                result['cases'].append(row)
                row['controls']['old_uninterrupted_PCG'] = run(A, b, old.apply, x, r, xstar, p0=next_p, rho0=next_rho, cpu_start=cpu_start)
                row['controls']['old_restart_PCG'] = run(A, b, old.apply, x, b-A@x, xstar, cpu_start=cpu_start)
                row['controls']['old_projected_PCG'] = run(A, b, lambda rr: gold_apply(old, projector, rr), xg, rg, xstar, projector=projector, cpu_start=cpu_start)
                for candidate in pool:
                    if process_time()-cpu_start > 95:
                        row['status'] = 'partial_pool'
                        result['status'] = 'budget_exhausted'
                        save()
                        return
                    aug, U, HU, denominator, lo, hi = augmented_removals(K, old, int(candidate))
                    energies = batch(A, b, aug, HU, denominator, projector, xg, rg, xstar)
                    for i in range(k):
                        row['trials'].append(dict(add=int(candidate), remove=int(old.indices[i]), remove_position=i,
                            energy5=float(energies[5][i]), energy10=float(energies[10][i]),
                            relative_eigenvalue_min=float(lo[i]), relative_eigenvalue_max=float(hi[i])))
                selected, short_selected = {}, {}
                for horizon in (5, 10):
                    ordered = sorted(row['trials'], key=lambda v: v[f'energy{horizon}'])
                    old_energy = row['controls']['old_projected_PCG']['lookahead'][str(horizon)]['energy']
                    row[f'better_than_old_projected_at_{horizon}'] = sum(v[f'energy{horizon}'] < old_energy*(1-1e-8) for v in ordered)
                    row[f'best_at_{horizon}'] = ordered[0]
                    for trial in ordered[:5]:
                        selected[(trial['add'], trial['remove'])] = trial
                    short_selected[(ordered[0]['add'], ordered[0]['remove'])] = ordered[0]
                save()
                for trial in selected.values():
                    indices = np.r_[np.delete(old.indices, trial['remove_position']), trial['add']]
                    new = NystromPreconditioner(indices, K[:, indices], .001)
                    final = run(A, b, lambda rr: gold_apply(new, projector, rr), xg, rg, xstar, projector=projector, cpu_start=cpu_start)
                    errors = [abs(final['lookahead'][str(h)]['energy']-trial[f'energy{h}'])/final['lookahead'][str(h)]['energy'] for h in (5, 10)]
                    row['final_branches'].append(dict(trial=trial, gold=final,
                        independent_rebuild_max_lookahead_relative_error=max(errors), new_state_bytes=new.storage_bytes))
                    save()
                positive = any(row[f'better_than_old_projected_at_{h}'] for h in (5, 10))
                if positive:
                    for trial in short_selected.values():
                        aug, _, HU, denominator, _, _ = augmented_removals(K, old, trial['add'])
                        positions = [trial['remove_position'], k]
                        Udiff = HU[:, positions]/np.sqrt(denominator[positions])[None, :]
                        exact_V = projector.left(Udiff)
                        stream_V = Udiff.copy()
                        for j in range(20):
                            stream_V -= np.outer(P[:, j], AP[:, j]@stream_V)/float(P[:, j]@AP[:, j])
                        def short_apply(rr):
                            return boundary_action(old.apply, P[:, -1], AP[:, -1], Udiff, stream_V, rr)
                        initial_action = short_apply(rg)
                        indices = np.r_[np.delete(old.indices, trial['remove_position']), trial['add']]
                        new = NystromPreconditioner(indices, K[:, indices], .001)
                        exact_action = gold_apply(new, projector, rg)
                        check = dict(trial=trial,
                            streamed_V_relative_error=float(np.linalg.norm(stream_V-exact_V)/np.linalg.norm(exact_V)),
                            initial_boundary_action_relative_error=float(np.linalg.norm(initial_action-exact_action)/np.linalg.norm(exact_action)),
                            note='No full projector is used by short apply. Full P is retained only for diagnostics and possible numerical residual-restart correction, which is explicitly counted.')
                        check['short_continuation'] = run(A, b, short_apply, xg, rg, xstar, projector=projector, cpu_start=cpu_start)
                        row['streamed_checks'].append(check)
                        save()
                row['status'] = 'complete'
                save()
                print(json.dumps(dict(dataset=name, rank=k, trials=len(row['trials']),
                    controls={key:v['total_iterations'] for key,v in row['controls'].items()},
                    better5=row['better_than_old_projected_at_5'], better10=row['better_than_old_projected_at_10'],
                    best_selected_total=min(v['gold']['total_iterations'] for v in row['final_branches']),
                    prefix_conjugacy_error=row['prefix_checks']['A_conjugacy_spectral_error'],
                    boundary_error=row['prefix_checks']['boundary_vs_full_on_corrected_residual'],
                    cpu_seconds=process_time()-cpu_start)), flush=True)
        result['status'] = 'complete'
        save()
        print(json.dumps(dict(status='complete', cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__ == '__main__':
    main()
