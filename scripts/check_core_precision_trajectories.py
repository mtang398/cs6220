#!/usr/bin/env python3
"""Dense changing-accuracy trajectories with rebuilt mixed factors.

Analytical kernel-entry accounting only; no native timing or memory claim.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
import check_core_precision_allocation as precision
from check_schur_inplace_frontier_v2 import Budget, SharedKernel, ROOT


def normalized_score(factor, z, q, tnorm):
    zn, qn = z/tnorm, q/tnorm
    values = np.empty(factor.m)
    for j in range(factor.m):
        vector = zn[j]*qn[j:]+qn[j]*zn[j:]
        vector[0] -= zn[j]*qn[j]
        values[j] = factor.epsilon[j]*np.linalg.norm(vector)/(4*(factor.m-j))
    return values


def run(K, b, selected, lam, method, helper, *, cpu_cap=3.):
    started, cpu_started = perf_counter(), process_time()
    n, m = len(b), len(selected)
    remaining = np.setdiff1d(np.arange(n), selected)
    C = K[np.ix_(selected, selected)]+lam*np.eye(m)
    B = K[np.ix_(selected, remaining)]
    D = K[np.ix_(remaining, remaining)]+lam*np.eye(n-m)
    counts = dict(action_core_products=0, boundary_core_products=0, score_core_products=0,
                  factor_builds=0, Schur_actions=0, boundary_cross_products=0,
                  original_verifications=0, diagnostic_core_products=0,
                  diagnostic_cross_products=0)
    builds = []
    score = np.zeros(m)
    transition = None
    pending_old = None

    def build(mask):
        factor = precision.MixedFactor(C, mask, helper)
        counts['factor_builds'] += 1
        builds.append(factor.metadata())
        return factor

    if method == 'FIRST8_H1':
        factor = build(np.arange(8))
    else:
        factor = build([])
        if method == 'epsilon_H1':
            mask = precision.top8(factor.epsilon/(4*np.arange(m, 0, -1)))
            factor = build(mask)
        elif method == 'initial_b_H1':
            t = b[selected]
            z = factor.solve(t)
            q = factor.solve(t-C@z)
            counts['score_core_products'] += 1
            mask = precision.top8(normalized_score(factor, z, q, np.linalg.norm(t)))
            factor = build(mask)

    def refine(rhs, order, kind, *, collect=False, use_factor=None):
        nonlocal score
        use = factor if use_factor is None else use_factor
        answer = use.solve(rhs)
        for correction in range(order):
            residual = rhs-C@answer
            counts[kind+'_core_products'] += 1
            delta = use.solve(residual)
            if collect and correction == 0:
                score += normalized_score(use, answer, delta, np.linalg.norm(rhs))
            answer += delta
        return answer

    top = refine(b[selected], 4, 'boundary')
    rhs = b[remaining]-B.T@top
    counts['boundary_cross_products'] += 1
    x = np.zeros(n-m)
    r = rhs.copy()
    p = r.copy()
    rr = float(r@r)
    nb = float(np.linalg.norm(b))
    setup = perf_counter()-started
    relative = None
    landmark_relative = None
    tail_relative = None
    status = 'maxiter'
    verification_history = []
    actions = []

    def verify(iteration):
        nonlocal relative, landmark_relative, tail_relative
        top_rhs = b[selected]-B@x
        counts['boundary_cross_products'] += 1
        top = refine(top_rhs, 4, 'boundary')
        full = np.empty(n)
        full[selected] = top
        full[remaining] = x
        original_residual = b-K@full-lam*full
        counts['original_verifications'] += 1
        relative = float(np.linalg.norm(original_residual)/nb)
        landmark_relative = float(np.linalg.norm(original_residual[selected])/nb)
        tail_relative = float(np.linalg.norm(original_residual[remaining])/nb)
        verification_history.append(dict(iteration=iteration, relative=relative,
            landmark_relative=landmark_relative, tail_relative=tail_relative))

    for iteration in range(1, 2001):
        if process_time()-cpu_started > cpu_cap:
            status = 'cpu_screen_cap'
            iteration -= 1
            break
        t = B@p
        collect = method == 'current_four_H2_to_H1' and iteration <= 4
        order = 2 if method in ['uniform592_H2', 'uniform600_H2'] or collect else 1
        top = refine(t, order, 'action', collect=collect)
        ap = D@p-B.T@top
        counts['Schur_actions'] += 1
        if pending_old is not None:
            old_top = refine(t, 2, 'diagnostic', use_factor=pending_old)
            gap = B.T@(old_top-top)
            counts['diagnostic_cross_products'] += 1
            old_action = ap-gap
            transition.update(next_action_iteration=iteration,
                action_gap_norm=float(np.linalg.norm(gap)),
                relative_action_gap=float(np.linalg.norm(gap)/np.linalg.norm(old_action)),
                next_action_direction_norm=float(np.linalg.norm(p)))
            pending_old = None
        curvature = float(p@ap)
        if curvature <= 0 or not np.isfinite(curvature):
            status = 'nonpositive_curvature'
            break
        alpha = rr/curvature
        x += alpha*p
        r -= alpha*ap
        actions.append(order)
        if np.linalg.norm(r) <= .5e-8*nb:
            verify(iteration)
            status = 'converged' if relative <= 1e-8 else 'recursive_stop_true_residual_failed'
            break
        next_rr = float(r@r)
        p = r+(next_rr/rr)*p
        rr = next_rr
        if method == 'current_four_H2_to_H1' and iteration == 4:
            mask = precision.top8(score)
            old_factor = factor
            factor = build(mask)
            pending_old = old_factor
            transition = dict(after_iteration=4, promoted=mask.tolist(),
                score_top8_fraction=float(score[mask].sum()/score.sum()),
                reduced_rhs_recomputed=False, recurrence_reset=False)
    if not verification_history or verification_history[-1]['iteration'] != iteration:
        verify(iteration)
    core_products = sum(counts[key] for key in ['action_core_products', 'boundary_core_products', 'score_core_products'])
    entry_components = dict(
        FPS_Gaussian_entries=n*m,
        streamed_factor_entries=counts['factor_builds']*m*(m+1)//2,
        Schur_cross_and_tail_entries=counts['Schur_actions']*(n*n-m*m),
        core_product_entries=core_products*m*m,
        boundary_cross_entries=counts['boundary_cross_products']*m*(n-m),
        original_verification_entries=counts['original_verifications']*n*n)
    diagnostic_entries = counts['diagnostic_core_products']*m*m+counts['diagnostic_cross_products']*m*(n-m)
    return dict(method=method, m=m, status=status, converged=relative <= 1e-8,
        iterations=iteration, true_relative_residual=relative,
        landmark_relative_residual=landmark_relative, tail_relative_residual=tail_relative,
        setup_diagnostic_seconds=setup, total_diagnostic_seconds=perf_counter()-started,
        total_diagnostic_cpu_seconds=process_time()-cpu_started,
        counts=counts, paid_core_products=core_products, entry_components=entry_components,
        analytical_solver_Gaussian_entries=sum(entry_components.values()),
        FPS_squared_distance_pair_evaluations=n*m,
        diagnostic_extra_Gaussian_entries=diagnostic_entries,
        accounting_note='FPS uses SharedKernel.farthest, which evaluates both squared distances and exp. Its Gaussian entries are charged once; distance counts are separate and not added as time. Dense K/C caches and CPU times are offline only. Transition gap comparison is an extra diagnostic, not used by solver.',
        factor_builds=builds, action_orders=actions, transition=transition,
        verification_history=verification_history)


def first8_heldout(K, b, selected, lam, helper):
    trace, C, B, cf, directions = precision.trajectory(K, b, selected, lam)
    low = precision.MixedFactor(C, [], helper)
    first = precision.MixedFactor(C, np.arange(8), helper)
    rows = []
    for index, p in enumerate(directions[4:], 5):
        t = B@p
        reference = linalg.cho_solve(cf, t)
        errors = []
        for factor, order in [(first, 1), (low, 2)]:
            answer = factor.solve(t)
            for _ in range(order):
                answer += factor.solve(t-C@answer)
            errors.append(float(np.linalg.norm(B.T@(reference-answer))))
        rows.append(dict(iteration=index, FIRST8_H1_error=errors[0], uniform_H2_error=errors[1]))
    ratios = np.array([row['FIRST8_H1_error']/max(row['uniform_H2_error'], np.finfo(float).tiny) for row in rows])
    return dict(scope='Extra frozen-exact-trajectory offline accuracy check, not algorithm selection work.',
                exact_iterations=trace['iterations'], heldout_count=len(rows),
                at_most_uniform_H2_count=sum(row['FIRST8_H1_error'] <= row['uniform_H2_error'] for row in rows),
                median_ratio=float(np.median(ratios)) if len(rows) else None,
                maximum_ratio=float(np.max(ratios)) if len(rows) else None, history=rows)


def main():
    output = ROOT/'results/core-precision-trajectories.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (43, 45))
    started, cpu_started = perf_counter(), process_time()
    old_path = ROOT/'results/core-precision-allocation.json'
    old = json.loads(old_path.read_text())
    assert all(sha256((ROOT/path).read_bytes()).hexdigest() == h for path, h in old['sources'].items())
    helper = precision.Helper(ROOT/'results/mixed_column_factor.so')
    sources = [Path(__file__), old_path, *[ROOT/path for path in old['sources']]]
    methods = ['FIRST8_H1', 'epsilon_H1', 'initial_b_H1', 'current_four_H2_to_H1', 'uniform592_H2', 'uniform600_H2']
    cases = [('smooth', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('mixed', 'synthetic-n2560-mixed-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(cases=cases, methods=methods, regularization=1e-5, seed=0,
            candidate_core=592, promotions=8, boundary_corrections=4,
            current_policy='First4 actual H2 action RHS, reuse actual z0/q from first correction, normalize score by||t||². Rebuild original C once, continue H1 with no reset or RHS replacement.',
            stopping='Recursive residual<=.5e-8||b|| then literal original residual<=1e-8. Failure is retained, with no hidden repair.',
            static_FIRST8='Direct single mixed build, no preliminary all32 factor or mask selection.',
            FPS='Existing SharedKernel.farthest evaluates distance and Gaussian exp; per-solver hypothetical counts charged separately from shared offline setup.',
            accounting='All analytical solver Gaussian entries include FPS, streamed factor/rebuild, score core products, actions, boundary4 products, reconstruction and original verification. Extra gap/reference diagnostics are separately disclosed.',
            scope='Dense actual inexact trajectories; no native runtime or full-memory claim.',
            per_attempt_cpu_cap=3., total_cpu_cap=43, wall_cap=60), runs=[], heldout_FIRST8=[], failures=[])

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
                        m = 600 if method == 'uniform600_H2' else 592
                        row = run(K, b, indices[:m], 1e-5, method, helper)
                    row.update(dataset=name, n=len(b), regularization=1e-5,
                               data_sha256=sha256(path.read_bytes()).hexdigest())
                    record['runs'].append(row)
                    save()
                    print(json.dumps({key:row.get(key) for key in ['dataset', 'method', 'status', 'iterations', 'true_relative_residual', 'paid_core_products', 'analytical_solver_Gaussian_entries']}), flush=True)
                if process_time()-cpu_started < 37.:
                    check = first8_heldout(K, b, indices[:592], 1e-5, helper)
                    check['dataset'] = name
                    record['heldout_FIRST8'].append(check)
                    save()
        except Exception as error:
            record['status'] = 'failed'
            record['failures'].append(dict(error=repr(error)))
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
