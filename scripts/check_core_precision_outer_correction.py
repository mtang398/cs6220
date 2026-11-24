#!/usr/bin/env python3
"""Common full-original-residual correction and static-prefix confound gate.

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
                top = refine(t, order, 'action', collect=current_policy and prefix)
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
    output = ROOT/'results/core-precision-outer-correction.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (43, 45))
    started, cpu_started = perf_counter(), process_time()
    old_path = ROOT/'results/airfoil-core-error-diagnosis.json'
    old = json.loads(old_path.read_text())
    assert all(sha256((ROOT/path).read_bytes()).hexdigest() == h for path, h in old['sources'].items())
    sources = [Path(__file__), old_path, *[ROOT/path for path in old['sources']]]
    helper = prior.precision.Helper(ROOT/'results/mixed_column_factor.so')
    methods = ['FIRST8_H1', 'FIRST8_H2prefix4_then_H1', 'current_four_H2_to_H1',
               'uniform600_H1', 'uniform600_H2', 'uniform600_H3']
    cases = [('smooth', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('mixed', 'synthetic-n2560-mixed-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(cases=cases, methods=methods, seed=0, regularization=1e-5,
            mixed_core=592, uniform_core=600, promotion_count=8,
            maximum_additional_correction_cycles=3, original_relative_tolerance=1e-8,
            absolute_inner_target='.5e-8*||original b|| for every correction; not scaled by the new residual.',
            boundary_corrections=4,
            correction='From verified full r, q=rR-B.T Hb rS; solve current Sa v≈q; u=Hb(rS-Bv); add[u,v] to accumulated full x. Recompute literal b-Ax. Never reconstruct totalx from total-tail constraints.',
            prefix='H2 only in first4 inner actions of first cycle; H1 afterward and in all correction cycles. Current score/rebuild occurs only once in that prefix. FIRST8-prefix has FIRST8 factor from the start and never rebuilds.',
            accounting='Counts include each factor/setup/rebuild, boundary4 solve, Schur action, original verification and correction cross products. FPS distance pairs separately reported; existing FPS also evaluates Gaussian exp. No native runtime/fullmemory claim.',
            per_attempt_cpu_cap=3., total_cpu_cap=43, command_wall_cap=60,
            scope='One dense common-repair/confound gate, not a new iterative-refinement principle or feedback precision policy.'),
        runs=[], failures=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
                      elapsed_cpu_seconds=process_time()-cpu_started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)

    save()
    with threadpool_limits(limits=1):
        try:
            for name, filename in cases:
                path = ROOT/'data/prepared'/filename
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                kernel = SharedKernel(X, bw, Budget([X, b]))
                indices = kernel.farthest(600)
                K = kernel.block(X, X)
                for method in methods:
                    if process_time()-cpu_started >= 37.:
                        row = dict(method=method, status='global_cpu_budget_not_started', converged=False)
                    else:
                        m = 600 if method.startswith('uniform600') else 592
                        row = run(K, b, indices[:m], 1e-5, method, helper)
                    row.update(dataset=name, n=len(b), regularization=1e-5,
                               data_sha256=sha256(path.read_bytes()).hexdigest())
                    record['runs'].append(row)
                    save()
                    print(json.dumps({key:row.get(key) for key in ['dataset', 'method', 'status', 'total_inner_iterations', 'correction_cycles', 'pre_repair_original_relative_residual', 'final_original_relative_residual', 'paid_core_products', 'analytical_solver_Gaussian_entries']}), flush=True)
        except Exception as error:
            record['status'] = 'failed'
            record['failures'].append(dict(error=repr(error)))
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
