#!/usr/bin/env python3
"""Airfoil residual decomposition before any repair; dense diagnostic only."""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
import check_core_precision_trajectories as prior
from check_schur_inplace_frontier_v2 import Budget, SharedKernel, ROOT


def diagnose(K, b, J, C, B, D, rhs, y, recursive, factor, initial_factor, method):
    remaining = np.setdiff1d(np.arange(len(b)), J)
    norm_b = float(np.linalg.norm(b))
    cf = linalg.cho_factor(C, lower=True)
    exact = lambda t:linalg.cho_solve(cf, t)

    def refined(use, t, order):
        if use is None:
            return exact(t)
        answer = use.solve(t)
        for _ in range(order):
            answer += use.solve(t-C@answer)
        return answer

    relative = lambda vector:float(np.linalg.norm(vector)/norm_b)
    exact_rhs = b[remaining]-B.T@exact(b[J])
    final_g4 = b[remaining]-B.T@refined(factor, b[J], 4)
    t = B@y
    exact_action = D@y-B.T@exact(t)
    if method == 'preformed_exact_S':
        S = D-B.T@exact(B)
        current_action = S@y
        order = None
    else:
        order = 2 if method == 'uniform592_H2' else 1
        current_action = D@y-B.T@refined(factor, t, order)
    action4 = D@y-B.T@refined(factor, t, 4)
    current_true = rhs-current_action
    recurrence_gap = current_true-recursive
    change_rhs = final_g4-rhs
    order_difference = current_action-action4
    top_rhs = b[J]-B@y
    top = refined(factor, top_rhs, 4)
    full = np.empty(len(b))
    full[J] = top
    full[remaining] = y
    original = b-K@full-1e-5*full
    long_original = b.astype(np.longdouble)-K.astype(np.longdouble)@full.astype(np.longdouble)-np.longdouble(1e-5)*full.astype(np.longdouble)
    original_tail = original[remaining]
    sum_components = change_rhs+order_difference+recurrence_gap
    closure = original_tail-recursive-sum_components
    initial_top4 = refined(initial_factor, b[J], 4)
    initial_core_defect = b[J]-C@initial_top4
    mapped_initial_defect = B.T@exact(initial_core_defect)
    final_core_defect = top_rhs-C@top
    exact_top = exact(top_rhs)
    exact_full = full.copy()
    exact_full[J] = exact_top
    exact_boundary_original = b-K@exact_full-1e-5*exact_full
    C_long = C.astype(np.longdouble)
    initial_defect_long = b[J].astype(np.longdouble)-C_long@initial_top4.astype(np.longdouble)
    core_reference = exact(b[J])
    core_reference_defect = b[J]-C@core_reference
    diagnostic = dict(
        recursive_relative_norm=relative(recursive),
        final_operator_true_relative_norm=relative(current_true),
        final_operator_true_minus_recursive=relative(recurrence_gap),
        original_relative_norm=relative(original),
        original_landmark_relative_norm=relative(original[J]),
        original_tail_relative_norm=relative(original_tail),
        longdouble_original_relative_norm=relative(long_original),
        longdouble_vs_float_original_difference=relative(long_original-original),
        original_with_exact_final_boundary=relative(exact_boundary_original),
        initial_g4_minus_exact_g=relative(rhs-exact_rhs),
        final_g4_minus_exact_g=relative(final_g4-exact_rhs),
        final_g4_minus_initial_g4=relative(change_rhs),
        action_order_difference_Sj_minus_S4=relative(order_difference),
        current_action_minus_exact_action=relative(current_action-exact_action),
        H4_action_minus_exact_action=relative(action4-exact_action),
        decomposition_closure_error=relative(closure),
        initial_core_H4_relative_residual=float(np.linalg.norm(initial_core_defect)/np.linalg.norm(b[J])),
        initial_core_H4_longdouble_relative_residual=float(np.linalg.norm(initial_defect_long)/np.linalg.norm(b[J])),
        initial_core_H4_amplified_defect=relative(mapped_initial_defect),
        initial_rhs_error_minus_amplified_defect=relative(rhs-exact_rhs-mapped_initial_defect),
        final_core_H4_relative_residual=float(np.linalg.norm(final_core_defect)/np.linalg.norm(top_rhs)),
        exact_core_reference_relative_residual=float(np.linalg.norm(core_reference_defect)/np.linalg.norm(b[J])),
        exact_core_reference_longdouble_relative_residual=float(np.linalg.norm(b[J].astype(np.longdouble)-C_long@core_reference.astype(np.longdouble))/np.linalg.norm(b[J])),
        final_action_order=order,
        explanation='tail-r_recursive=(new_g4-initial_g4)+(Sj-S4)y+(current_operator_true-r_recursive)+closure. For changing factors the third term includes intentional previous-operator mismatch as well as arithmetic drift.',
        endpoint_vectors=dict(y=y.tolist(), recursive_residual=recursive.tolist(),
            current_operator_true_residual=current_true.tolist(), original_tail_residual=original_tail.tolist(),
            boundary_rhs_change=change_rhs.tolist(), action_order_difference=order_difference.tolist()))
    if initial_factor is not None and relative(rhs-exact_rhs) > 1e-9:
        initial_top6 = refined(initial_factor, b[J], 6)
        rhs6 = b[remaining]-B.T@initial_top6
        diagnostic['H6_boundary_diagnostic'] = dict(
            initial_g6_minus_exact_g=relative(rhs6-exact_rhs),
            initial_g6_minus_initial_g4=relative(rhs6-rhs),
            initial_core_H6_relative_residual=float(np.linalg.norm(b[J]-C@initial_top6)/np.linalg.norm(b[J])),
            scope='Post-solve diagnostic only. No trajectory or boundary policy changed.')
    return diagnostic


# Reuse the frozen three compressed trajectories exactly, adding endpoint
# capture only. The scientific recurrence and stopping conditions are unchanged.
CAPTURE_SOURCE = inspect.getsource(prior.run)
assert CAPTURE_SOURCE.count("top = refine(b[selected], 4, 'boundary')") == 1
CAPTURE_SOURCE = CAPTURE_SOURCE.replace("top = refine(b[selected], 4, 'boundary')",
    "initial_factor = factor\n    top = refine(b[selected], 4, 'boundary')")
assert CAPTURE_SOURCE.count('return dict(method=method, m=m, status=status,') == 1
CAPTURE_SOURCE = CAPTURE_SOURCE.replace('return dict(method=method, m=m, status=status,',
    'return dict(endpoint=diagnose(K,b,selected,C,B,D,rhs,x,r,factor,initial_factor,method), method=method, m=m, status=status,')
namespace = dict(prior.__dict__, diagnose=diagnose)
exec(compile(CAPTURE_SOURCE, '<unchanged-trajectory-endpoint-capture>', 'exec'), namespace)
captured_run = namespace['run']


def exact_run(K, b, J, method):
    started = perf_counter()
    R = np.setdiff1d(np.arange(len(b)), J)
    C = K[np.ix_(J, J)]+1e-5*np.eye(len(J))
    B = K[np.ix_(J, R)]
    D = K[np.ix_(R, R)]+1e-5*np.eye(len(R))
    cf = linalg.cho_factor(C, lower=True)
    solve_core = lambda rhs:linalg.cho_solve(cf, rhs)
    rhs = b[R]-B.T@solve_core(b[J])
    S = D-B.T@solve_core(B) if method == 'preformed_exact_S' else None
    y = np.zeros(len(R))
    r = rhs.copy()
    p = r.copy()
    rr = float(r@r)
    nb = float(np.linalg.norm(b))
    status = 'maxiter'
    for iteration in range(1, 2001):
        ap = S@p if S is not None else D@p-B.T@solve_core(B@p)
        curvature = float(p@ap)
        if curvature <= 0:
            status = 'nonpositive_curvature'
            break
        alpha = rr/curvature
        y += alpha*p
        r -= alpha*ap
        if np.linalg.norm(r) <= .5e-8*nb:
            status = 'recursive_stop'
            break
        new_rr = float(r@r)
        p = r+(new_rr/rr)*p
        rr = new_rr
    endpoint = diagnose(K, b, J, C, B, D, rhs, y, r, None, None, method)
    passed = endpoint['original_relative_norm'] <= 1e-8
    if status == 'recursive_stop':
        status = 'converged' if passed else 'recursive_stop_true_residual_failed'
    return dict(method=method, status=status, converged=passed, iterations=iteration,
                endpoint=endpoint, elapsed_diagnostic_seconds=perf_counter()-started)


def main():
    output = ROOT/'results/airfoil-core-error-diagnosis.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (18, 20))
    started, cpu_started = perf_counter(), process_time()
    old_path = ROOT/'results/core-precision-trajectories.json'
    old = json.loads(old_path.read_text())
    assert all(sha256((ROOT/path).read_bytes()).hexdigest() == h for path, h in old['sources'].items())
    sources = [Path(__file__), old_path, *[ROOT/path for path in old['sources']]]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        captured_function_sha256=sha256(CAPTURE_SOURCE.encode()).hexdigest(),
        protocol=dict(dataset='Airfoil', seed=0, n=1202, core=592, regularization=1e-5,
            methods=['exact_core_actions', 'preformed_exact_S', 'uniform592_H2', 'FIRST8_H1', 'current_four_H2_to_H1'],
            stopping='Unchanged recursive .5e-8 then literal original1e-8; no residual replacement or policy repair.',
            scope='Dense offline source-of-error diagnosis; longdouble only accumulates the same stored original matrix.',
            cpu_cap=18, wall_cap=30), runs=[])

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
            path = ROOT/'data/prepared/airfoil-observed-s0-lambda1e-05.npz'
            record['data_sha256'] = sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
            kernel = SharedKernel(X, bw, Budget([X, b]))
            J = kernel.farthest(592)
            K = kernel.block(X, X)
            helper = prior.precision.Helper(ROOT/'results/mixed_column_factor.so')
            for method in record['protocol']['methods']:
                if method in ['exact_core_actions', 'preformed_exact_S']:
                    row = exact_run(K, b, J, method)
                else:
                    row = captured_run(K, b, J, 1e-5, method, helper)
                    frozen = next(r for r in old['runs'] if r['dataset'] == 'Airfoil' and r['method'] == method)
                    row['frozen_reproduction'] = dict(iterations_match=row['iterations'] == frozen['iterations'],
                        status_match=row['status'] == frozen['status'],
                        original_relative_difference=abs(row['true_relative_residual']-frozen['true_relative_residual']))
                    assert row['frozen_reproduction']['iterations_match'] and row['frozen_reproduction']['status_match']
                record['runs'].append(row)
                save()
                print(json.dumps(dict(method=method, iterations=row['iterations'], status=row['status'],
                    endpoint={k:row['endpoint'][k] for k in ['recursive_relative_norm', 'final_operator_true_relative_norm', 'original_relative_norm', 'initial_g4_minus_exact_g', 'action_order_difference_Sj_minus_S4', 'final_operator_true_minus_recursive']})), flush=True)
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = repr(error)
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
