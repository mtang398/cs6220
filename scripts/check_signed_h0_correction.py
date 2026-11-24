#!/usr/bin/env python3
"""Signed rank-four correction of unrefined FIRST8 inverse.

Dense offline arithmetic with analytical query counts. No native resource claim.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from threadpoolctl import threadpool_limits
import check_core_precision_trajectories as prior
from check_schur_inplace_frontier_v2 import Budget, SharedKernel, ROOT


def run(K, b, selected, lam, method, helper, *, cpu_cap=3., signed_correction=None):
    started, cpu_started = perf_counter(), process_time()
    n, m = len(b), len(selected)
    R = np.setdiff1d(np.arange(n), selected)
    C = K[np.ix_(selected, selected)]+lam*np.eye(m)
    B = K[np.ix_(selected, R)]
    D = K[np.ix_(R, R)]+lam*np.eye(len(R))
    counts = dict(action_core_products=0, boundary_core_products=0, factor_builds=0,
                  Schur_actions=0, boundary_cross_products=0, original_verifications=0)
    builds = []
    score = np.zeros(m)
    transition = None
    current_policy = method == 'current_four_H2_to_H1'
    static_prefix = method == 'FIRST8_H2prefix4_then_H1'

    def build(mask):
        factor = prior.precision.MixedFactor(C, mask, helper)
        counts['factor_builds'] += 1
        builds.append(factor.metadata())
        return factor

    factor = build(np.arange(8) if method.startswith('FIRST8') else [])

    def refine(rhs, corrections, kind, *, collect=False):
        nonlocal score
        answer = factor.solve(rhs)
        for step in range(corrections):
            residual = rhs-C@answer
            counts[kind+'_core_products'] += 1
            delta = factor.solve(residual)
            if collect and step == 0:
                score += prior.normalized_score(factor, answer, delta, np.linalg.norm(rhs))
            answer += delta
        return answer

    full_x = np.zeros(n)
    full_residual = b.copy()
    norm_b = float(np.linalg.norm(b))
    inner_target = .5e-8*norm_b
    cycles = []
    total_iterations = 0
    action_orders = []
    status = 'correction_cycle_cap'
    original_relative = 1.
    for cycle in range(4):
        counts_before = dict(counts)
        initial_relative = float(np.linalg.norm(full_residual)/norm_b)
        # All correction RHS values come from the verified full original
        # residual. Retain and increment the accumulated original-space x.
        top_rhs = refine(full_residual[selected], 4, 'boundary')
        reduced_rhs = full_residual[R]-B.T@top_rhs
        counts['boundary_cross_products'] += 1
        v = np.zeros(len(R))
        r = reduced_rhs.copy()
        p = r.copy()
        rr = float(r@r)
        inner_status = 'already_below_absolute_target' if np.linalg.norm(r) <= inner_target else 'maxiter'
        iterations = 0
        if inner_status == 'maxiter':
            for iterations in range(1, 2001):
                if process_time()-cpu_started > cpu_cap:
                    inner_status = 'cpu_screen_cap'
                    iterations -= 1
                    break
                t = B@p
                prefix = cycle == 0 and iterations <= 4 and (current_policy or static_prefix)
                if method.startswith('uniform600_H'):
                    order = int(method[-1])
                else:
                    order = 0 if method == 'FIRST8_H0' else (2 if prefix else 1)
                top = refine(t, order, 'action', collect=current_policy and prefix)
                if signed_correction is not None:
                    W,signs=signed_correction
                    top += W@(signs*(W.T@t))
                ap = D@p-B.T@top
                counts['Schur_actions'] += 1
                curvature = float(p@ap)
                if curvature <= 0 or not np.isfinite(curvature):
                    inner_status = 'nonpositive_curvature'
                    break
                alpha = rr/curvature
                v += alpha*p
                r -= alpha*ap
                action_orders.append(order)
                if np.linalg.norm(r) <= inner_target:
                    inner_status = 'absolute_recursive_target'
                    break
                next_rr = float(r@r)
                p = r+(next_rr/rr)*p
                rr = next_rr
                if current_policy and cycle == 0 and iterations == 4:
                    mask = prior.precision.top8(score)
                    factor = build(mask)
                    transition = dict(cycle=0, after_inner_iteration=4, promoted=mask.tolist(),
                        score_top8_fraction=float(score[mask].sum()/score.sum()),
                        recurrence_reset=False, reduced_rhs_recomputed=False)
        total_iterations += iterations
        correction_top_rhs = full_residual[selected]-B@v
        counts['boundary_cross_products'] += 1
        u = refine(correction_top_rhs, 4, 'boundary')
        full_x[selected] += u
        full_x[R] += v
        full_residual = b-K@full_x-lam*full_x
        counts['original_verifications'] += 1
        original_relative = float(np.linalg.norm(full_residual)/norm_b)
        cycles.append(dict(cycle=cycle, initial_original_relative_residual=initial_relative,
            inner_iterations=iterations, inner_status=inner_status,
            inner_recursive_relative_to_original_b=float(np.linalg.norm(r)/norm_b),
            final_original_relative_residual=original_relative,
            final_landmark_relative_residual=float(np.linalg.norm(full_residual[selected])/norm_b),
            final_tail_relative_residual=float(np.linalg.norm(full_residual[R])/norm_b),
            operation_counts={key:counts[key]-counts_before[key] for key in counts}))
        if original_relative <= 1e-8:
            status = 'converged'
            break
        if inner_status in ['cpu_screen_cap', 'nonpositive_curvature', 'maxiter']:
            status = inner_status
            break
    paid_core_products = counts['action_core_products']+counts['boundary_core_products']
    entry_components = dict(FPS_Gaussian_entries=n*m,
        streamed_factor_entries=counts['factor_builds']*m*(m+1)//2,
        Schur_cross_and_tail_entries=counts['Schur_actions']*(n*n-m*m),
        core_product_entries=paid_core_products*m*m,
        boundary_cross_entries=counts['boundary_cross_products']*m*(n-m),
        original_verification_entries=counts['original_verifications']*n*n)
    return dict(method=method, m=m, status=status, converged=original_relative <= 1e-8,
        total_inner_iterations=total_iterations, correction_cycles=len(cycles)-1,
        pre_repair_original_relative_residual=cycles[0]['final_original_relative_residual'],
        pre_repair_passed=cycles[0]['final_original_relative_residual'] <= 1e-8,
        final_original_relative_residual=original_relative, cycles=cycles,
        counts=counts, paid_core_products=paid_core_products,
        entry_components=entry_components, analytical_solver_Gaussian_entries=sum(entry_components.values()),
        FPS_squared_distance_pair_evaluations=n*m, factor_builds=builds,
        transition=transition, action_orders=action_orders,
        full_solution_sha256=sha256(full_x.tobytes()).hexdigest(),
        original_residual_sha256=sha256(full_residual.tobytes()).hexdigest(),
        offline_diagnostic_seconds=perf_counter()-started,
        offline_diagnostic_cpu_seconds=process_time()-cpu_started)


def main():
    from scipy.linalg import eigh,solve_triangular,cho_solve
    output=ROOT/'results/signed-h0-correction.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
    sources=[Path(__file__).resolve(),Path(prior.__file__).resolve(),Path(prior.precision.__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},protocol=dict(
        question='Does a signed rank4 H0 correction remove core-product cost with useful accuracy?',
        construction='E=L^-1 C L^-T-I; eig(theta,V). Select4 largestabs(-theta/(1+theta)); W=L^-T V_selected sqrt(abs(-theta/(1+theta))); signs=sign(-theta/(1+theta)). Hcorr=H0+W diag(signs) W.T.',
        core=584,controls=['same-core H0','strong592 H1 and H2prefix from prior pilot'],
        common='H4 boundary, inner .5e-8 norm(original b), max3 repairs, original relative1e-8.',
        scope='Dense free eigensystem construction, static representation gate; no learned acquisition, native timing or full-memory claim. Optimal spectral truncation only in whitened inverse error, not coupled Schur error or CG iterations.',limits='120wall/80CPU/5CPU per solve;BLAS1'),runs=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    save();helper=prior.precision.Helper(ROOT/'results/mixed_column_factor_first8.so')
    cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
    with threadpool_limits(limits=1):
        for name,filename in cases:
            path=ROOT/'data/prepared'/filename
            with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
            kernel=SharedKernel(X,bw,Budget([X,b]));J=kernel.farthest(584);K=kernel.block(X,X)
            C=K[np.ix_(J,J)]+1e-5*np.eye(584);factor=prior.precision.MixedFactor(C,np.arange(8),helper);L=factor.dense_L()
            E=solve_triangular(L,C,lower=True);E=solve_triangular(L,E.T,lower=True).T-np.eye(584);E=(E+E.T)*.5
            theta,V=eigh(E);assert theta[0]>-1
            coeff=-theta/(1+theta);selected=np.argsort(np.abs(coeff))[-4:];signs=np.sign(coeff[selected])
            W=solve_triangular(L.T,V[:,selected],lower=False)*np.sqrt(np.abs(coeff[selected]))
            fullW=solve_triangular(L.T,V,lower=False)
            reconstructed=cho_solve((L,True),np.eye(584))+(fullW*coeff)@fullW.T
            exact=cho_solve((np.linalg.cholesky(C),True),np.eye(584))
            identity=float(np.linalg.norm(reconstructed-exact)/np.linalg.norm(exact));assert identity<1e-7
            remain=np.delete(coeff,selected)
            info=dict(selected_theta=theta[selected].tolist(),correction_signs=signs.tolist(),
                remaining_whitened_inverse_error_spectral_fraction=float(max(abs(remain))/max(abs(coeff))),
                remaining_whitened_inverse_error_Frobenius_fraction=float(np.linalg.norm(remain)/np.linalg.norm(coeff)),
                full_rank_identity_relative_error=identity,minimum_corrected_whitened_inverse_eigenvalue=float(min(1.,min(1/(1+theta[selected])))),
                retained_factor_metadata_and_correction_bytes=factor.low.nbytes+factor.wide.nbytes+factor.high.nbytes+factor.offsets.nbytes+W.nbytes+signs.nbytes)
            for mode in ['unchanged','signed_rank4']:
                row=run(K,b,J,1e-5,'FIRST8_H0',helper,cpu_cap=5.,signed_correction=(W,signs) if mode=='signed_rank4' else None)
                row.update(dataset=name,correction_mode=mode,oracle=info,data_sha256=sha256(path.read_bytes()).hexdigest())
                rec['runs'].append(row);save();print(json.dumps({k:row[k] for k in ['dataset','correction_mode','status','total_inner_iterations','analytical_solver_Gaussian_entries']}),flush=True)
    rec['status']='complete';save()

if __name__=='__main__':main()
