#!/usr/bin/env python3
"""Current-direction energy proxy for choosing core-refinement accuracy.

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


def run(K, b, selected, lam, method, helper, *, cpu_cap=3.):
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
    selector_trace = []
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
                    order = 2 if prefix else 1
                dp = D@p
                if method == 'FIRST8_energy_selector':
                    z = factor.solve(t)
                    core_residual = t-C@z
                    counts['action_core_products'] += 1
                    delta = factor.solve(core_residual)
                    top = z+delta
                    approximate_curvature = float(p@dp-t@top)
                    energy_proxy = max(float(core_residual@delta), 0.)
                    threshold = .1*inner_target/max(np.sqrt(rr), inner_target)
                    proxy_ratio = np.sqrt(energy_proxy/max(approximate_curvature, np.finfo(float).tiny))
                    order = 3 if approximate_curvature <= 0 or proxy_ratio > threshold else 1
                    selector_trace.append(dict(cycle=cycle, iteration=iterations,
                        proxy_ratio=float(proxy_ratio), threshold=float(threshold), order=order))
                    if order == 3:
                        for _ in range(2):
                            extra_residual = t-C@top
                            counts['action_core_products'] += 1
                            top += factor.solve(extra_residual)
                else:
                    if method == 'FIRST8_H3': order = 3
                    if method == 'FIRST8_H3prefix4_then_H1': order = 3 if cycle == 0 and iterations <= 4 else 1
                    top = refine(t, order, 'action', collect=current_policy and prefix)
                ap = dp-B.T@top
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
        transition=transition, action_orders=action_orders, selector_trace=selector_trace,
        full_solution_sha256=sha256(full_x.tobytes()).hexdigest(),
        original_residual_sha256=sha256(full_residual.tobytes()).hexdigest(),
        offline_diagnostic_seconds=perf_counter()-started,
        offline_diagnostic_cpu_seconds=process_time()-cpu_started)


def main():
    output=ROOT/'results/core-energy-selector.json'
    if output.exists(): raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85))
    start,cpu=perf_counter(),process_time()
    files=[Path(__file__).resolve(),Path(prior.__file__).resolve(),Path(prior.precision.__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
    modes=['FIRST8_H1','FIRST8_H2prefix4_then_H1','FIRST8_H3','FIRST8_H3prefix4_then_H1','FIRST8_energy_selector']
    cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(
        question='Can existing current core residuals select cheaper refinement without random probes?',
        selector='z=H0t; e=t-Cz; delta=H0e; h=z+delta. Choose H3 if sqrt(max(e.T delta,0)/(p.T Dp-t.T h)) > 0.1*inner_target/max(norm(recursive_r),inner_target), else H1. Nonpositive approximate curvature chooses H3.',
        interpretation='Heuristic energy proxy, not a certified action-norm error bound. Threshold fixed before runs; no sweep.',
        controls=modes[:-1],core=592,first_FP64_columns=8,regularization=1e-5,seed=0,
        scope='Dense development trajectories and analytical kernel-entry counts; no native timing/full-memory/novelty claim.',
        common='H4 boundary; inner absolute .5e-8 norm(original b); at most3 outer repairs; literal original relative residual1e-8.',
        limits='120 wall,80 CPU,5 CPU per solve,BLAS1'),runs=[],failures=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        temp=output.with_suffix('.json.tmp');temp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');temp.replace(output)
    save();helper=prior.precision.Helper(ROOT/'results/mixed_column_factor_first8.so')
    with threadpool_limits(limits=1):
        try:
            for name,filename in cases:
                path=ROOT/'data/prepared'/filename
                with np.load(path) as d: X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
                kernel=SharedKernel(X,bw,Budget([X,b]));selected=kernel.farthest(592);K=kernel.block(X,X)
                for method in modes:
                    if process_time()-cpu>72: rec['status']='cpu_budget';save();return
                    row=run(K,b,selected,1e-5,method,helper,cpu_cap=5.)
                    row.update(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest())
                    rec['runs'].append(row);save()
                    print(json.dumps({k:row[k] for k in ['dataset','method','status','total_inner_iterations','paid_core_products','analytical_solver_Gaussian_entries']}),flush=True)
        except Exception as e:rec['status']='failed';rec['failures'].append(repr(e));save();raise
    rec['status']='complete';save()

if __name__=='__main__':main()
