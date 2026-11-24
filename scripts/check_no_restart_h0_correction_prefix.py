#!/usr/bin/env python3
"""Matched no-restart coupled H0 correction after a paid prefix.

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


def run(K, b, selected, lam, method, helper, *, cpu_cap=3., signed_correction=None, prefix_mode="none"):
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
                if prefix_mode!='none':
                    order=1 if cycle==0 and iterations<=4 else 0
                top = refine(t, order, 'action', collect=current_policy and prefix)
                ap = D@p-B.T@top
                if signed_correction is not None and (cycle>0 or iterations>4):
                    W,signs=signed_correction
                    ap -= W@(signs*(W.T@p))
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
                if cycle==0 and iterations==4 and prefix_mode in ['restart','oracle_after4']:
                    replacement=D@v-B.T@refine(B@v,0,'action')
                    if signed_correction is not None:
                        W,signs=signed_correction
                        replacement-=W@(signs*(W.T@v))
                    counts['Schur_actions']+=1
                    r=reduced_rhs-replacement
                    p=r.copy();rr=float(r@r)
                    transition=dict(after_iteration=4,new_operator_residual_recomputed=True,
                        original_b_scaled_inner_residual=float(np.linalg.norm(r)/norm_b))
                    if np.linalg.norm(r)<=inner_target:
                        inner_status='absolute_true_target_at_transition';break
                    continue
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
    return dict(method=method, prefix_mode=prefix_mode, m=m, status=status, converged=original_relative <= 1e-8,
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
    from scipy.linalg import eigh,solve_triangular,cho_solve,cholesky
    output=ROOT/'results/no-restart-h0-correction-prefix.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
    files=[Path(__file__).resolve(),Path(prior.__file__).resolve(),Path(prior.precision.__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(
        question='Does the same paid prefix with no restart preserve coupled rank4 headroom?'  ,
        mechanism='G=B.T(C^-1-H0)B. Retain4 eigenpairs of G with largestabs eigenvalue. Apply S0-W diag(signs)W.T directly in tail coordinates.',
        rank=4,core_selection='Largest integer m<=592 with 2m^2+43m-112 +32(n-m)+32 <=726272 bytes, matching592 factor+metadata only; not full-memory proof.',
        scope='Optimal G directions supplied free; charge four H1 actions but no restart. Changing-action ordinary CG is a heuristic, certified only by literal original residual and common repairs. Same-core prefix control repeated for bitwise comparison. Strong592 controls retained from preceding pilot. No native/learned claim.',
        common='H4 boundary, inner .5e-8 norm(original b), max3 additional repairs, literal original1e-8.',limits='120wall/80CPU/5CPU per solve,BLAS1'),runs=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    save();helper=prior.precision.Helper(ROOT/'results/mixed_column_factor_first8.so')
    cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
    with threadpool_limits(limits=1):
        for name,filename in cases:
            path=ROOT/'data/prepared'/filename
            with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
            n=len(b);m=max(j for j in range(8,min(592,n-1)+1) if 2*j*j+43*j-112+32*(n-j)+32<=726272)
            kernel=SharedKernel(X,bw,Budget([X,b]));J=kernel.farthest(m);R=np.setdiff1d(np.arange(n),J);K=kernel.block(X,X)
            C=K[np.ix_(J,J)]+1e-5*np.eye(m);B=K[np.ix_(J,R)];D=K[np.ix_(R,R)]+1e-5*np.eye(len(R))
            factor=prior.precision.MixedFactor(C,np.arange(8),helper);L=factor.dense_L()
            E=solve_triangular(L,C,lower=True);E=solve_triangular(L,E.T,lower=True).T-np.eye(m);E=(E+E.T)*.5
            theta,V=eigh(E);assert theta[0]>-1
            coefficients=-theta/(1+theta)
            Z=B.T@solve_triangular(L.T,V,lower=False)
            G=(Z*coefficients)@Z.T;G=(G+G.T)*.5
            values,U=eigh(G);selected=np.argsort(abs(values))[-4:];signs=np.sign(values[selected]);W=U[:,selected]*np.sqrt(abs(values[selected]))
            S0=D-B.T@cho_solve((L,True),B);Sc=S0-(W*signs)@W.T;Sc=(Sc+Sc.T)*.5
            try:cholesky(Sc,lower=True);spd=True
            except np.linalg.LinAlgError:spd=False
            rest=np.delete(values,selected)
            info=dict(minimum_Schur_eigenvalue=float(eigh(Sc,eigvals_only=True,subset_by_index=[0,0])[0]),Schur_cholesky_passed=spd,
                remaining_coupled_spectral_fraction=float(max(abs(rest))/max(abs(values))),remaining_coupled_Frobenius_fraction=float(np.linalg.norm(rest)/np.linalg.norm(values)),
                retained_factor_metadata_correction_bytes=2*m*m+43*m-112+W.nbytes+signs.nbytes,selected_values=values[selected].tolist())
            for mode in ['prefix','oracle_no_restart']:
                row=run(K,b,kernel.farthest(592) if mode=='strong592_prefix' else J,1e-5,'FIRST8_H0',helper,cpu_cap=5.,signed_correction=(W,signs) if mode=='oracle_no_restart' else None,prefix_mode='prefix' if mode=='strong592_prefix' else mode)
                row.update(dataset=name,correction_mode=mode,oracle=info,data_sha256=sha256(path.read_bytes()).hexdigest())
                rec['runs'].append(row);save();print(json.dumps({k:row[k] for k in ['dataset','m','correction_mode','status','total_inner_iterations','analytical_solver_Gaussian_entries']}),flush=True)
    rec['status']='complete';save()

if __name__=='__main__':main()
