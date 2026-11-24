#!/usr/bin/env python3
"""Native standard width-two Schur block-CG controls with full correction."""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_schur_block_action_cost as component
import check_native_first8_schur as scalar
from check_core_precision_allocation import Helper
from check_packed_float_allocation import RawPointers
from check_schur_inplace_frontier_v2 import ROOT, LIMIT, RESERVE


class CountedSchur:
    def __init__(self, op):
        self.op = op
        self.calls = 0
        self.rhs = 0
        self.widths = {}
        self.seconds = 0.

    def __matmul__(self, value):
        width = 1 if value.ndim == 1 else value.shape[1]
        self.calls += 1
        self.rhs += width
        self.widths[str(width)] = self.widths.get(str(width), 0)+1
        started = perf_counter()
        if value.ndim == 1:
            result = self.op.apply(value, 1)
        else:
            result = self.op.apply_block(value)
            self.op.counts['Schur_actions'] += 1
            self.op.counts['action_core_products'] += 1
        self.seconds += perf_counter()-started
        return result


def block_cg(op, B, *, rtol, maxiter=2000, deadline=float('inf'), capture=False):
    """Old validated QR block recurrence, identity Schur preconditioner.

Only resource instrumentation, the equal active-prefix B view, CPU censoring,
and bounded tiny-test capture differ from the validated recurrence.
"""
    A = CountedSchur(op)
    budget = op.budget
    active = list(range(B.shape[1]))
    norms = np.linalg.norm(B, axis=0)
    assert B.ndim == 2 and 1 <= B.shape[1] <= 2 and np.all(norms > 0)
    X = np.zeros_like(B)
    R = B.copy()
    P = AP = S = alpha = true_R = Z = P_next = None
    captures = []

    def sync(*extra):
        budget.live = [B, X, R, P, AP, S, alpha, true_R, Z, P_next, *captures, *extra]

    qr_seconds = 0.
    projection_seconds = 0.
    Z = R.copy()
    sync()
    budget.observe('block_initial_QR_bound', temporary_allowance=3*Z.nbytes+4608)
    qr_start = perf_counter()
    P, singular, threshold = component.orthonormal_search(Z)
    qr_seconds += perf_counter()-qr_start
    Z = None
    sync()
    deflations, rank_losses, replacements = [], [], []
    max_local_conjugacy = 0.
    max_small_condition = 1.
    true_checks = 0
    converged = False
    breakdown = None
    iteration = 0
    for iteration in range(1, maxiter+1):
        if process_time() > deadline:
            breakdown = 'cpu_screen_cap'
            iteration -= 1
            break
        if P.shape[1] == 0:
            breakdown = 'zero_search_rank'
            break
        sync()
        AP_new = A@P
        budget.observe('old_and_new_search_images', AP_new)
        AP = AP_new
        del AP_new
        sync()
        S = (P.T@AP+AP.T@P)*.5
        sync()
        try:
            chol = linalg.cho_factor(S, lower=True, check_finite=False)
            max_small_condition = max(max_small_condition, float(np.linalg.cond(S)))
            alpha = linalg.cho_solve(chol, P.T@R, check_finite=False)
        except np.linalg.LinAlgError:
            breakdown = 'nonpositive_search_gram'
            break
        sync()
        temporary = P@alpha
        budget.observe('block_X_update', temporary)
        X += temporary
        del temporary
        temporary = AP@alpha
        budget.observe('block_R_update', temporary)
        R -= temporary
        del temporary
        if capture and iteration <= 2:
            captures.append(X[:, active.index(0)].copy())
        recursive = np.linalg.norm(R, axis=0)/norms[active]
        suspected = recursive <= rtol
        restart = False
        if np.any(suspected) or iteration % 20 == 0 or iteration == maxiter:
            assert active in [[0, 1], [0]]
            sync()
            AX = A@X
            true_R = B[:, :len(active)]-AX
            budget.observe('block_true_residual', AX, true_R)
            del AX
            sync()
            true_checks += 1
            relative = np.linalg.norm(true_R, axis=0)/norms[active]
            if relative[active.index(0)] <= rtol:
                converged = True
                break
            if 1 in active and relative[active.index(1)] <= rtol:
                j = active.index(0)
                deflations.append(dict(iteration=iteration, rhs=1,
                    true_relative_residual=float(relative[active.index(1)])))
                X_new, R_new = X[:, j:j+1].copy(), R[:, j:j+1].copy()
                budget.observe('auxiliary_deflation_X_R', X_new, R_new)
                X, R = X_new, R_new
                del X_new, R_new
                sync()
                new_true = true_R[:, j:j+1].copy()
                budget.observe('auxiliary_deflation_true_R', new_true)
                true_R = new_true
                del new_true
                suspected = suspected[j:j+1]
                relative = relative[j:j+1]
                active = [0]
                sync()
            if np.any(suspected & (relative > rtol)):
                new_R = true_R.copy()
                budget.observe('block_model_residual_replacement', new_R)
                R = new_R
                del new_R
                replacements.append(iteration)
                restart = True
            true_R = None
            sync()
        Z = R.copy()
        sync()
        projection_start = perf_counter()
        if not restart:
            for _ in range(2):
                coefficients = linalg.cho_solve(chol, AP.T@Z, check_finite=False)
                temporary = P@coefficients
                budget.observe('block_local_projection', temporary, coefficients)
                Z -= temporary
                del temporary
        projection_seconds += perf_counter()-projection_start
        budget.observe('block_QR_workspace_bound', temporary_allowance=3*Z.nbytes+4608)
        qr_start = perf_counter()
        P_next, singular, threshold = component.orthonormal_search(Z)
        qr_seconds += perf_counter()-qr_start
        sync()
        budget.observe('block_QR_after', singular, temporary_allowance=2*Z.nbytes+4608)
        if P_next.shape[1] < len(active):
            rank_losses.append(dict(iteration=iteration, search_rank=P_next.shape[1],
                active_rhs=len(active), singular_values=singular.tolist(), threshold=threshold))
        if not restart and P_next.shape[1]:
            numerator = np.linalg.norm(AP.T@P_next)
            denominator = np.linalg.norm(AP)*np.linalg.norm(P_next)
            max_local_conjugacy = max(max_local_conjugacy, float(numerator/max(denominator, 1e-300)))
        P = P_next
        Z = P_next = None
        sync()
    primary = X[:, active.index(0)].copy()
    sync(primary)
    final_action = A@primary
    final_residual = B[:, 0]-final_action
    budget.observe('final_primary_model_verification', primary, final_action, final_residual)
    final_relative = float(np.linalg.norm(final_residual)/norms[0])
    converged = bool(final_relative <= rtol)
    row = dict(iterations=iteration, converged=converged,
        true_relative_Schur_residual=final_relative, breakdown=breakdown,
        auxiliary_deflations=deflations, numerical_rank_losses=rank_losses,
        model_residual_replacement_restarts=replacements, true_block_checks=true_checks,
        max_previous_block_conjugacy_relative=max_local_conjugacy,
        max_small_gram_condition=max_small_condition,
        initial_rhs_width=B.shape[1], final_active_rhs=len(active),
        approximate_Schur_calls=A.calls, approximate_Schur_rhs=A.rhs,
        approximate_Schur_width_histogram=A.widths, approximate_Schur_seconds=A.seconds,
        QR_seconds=qr_seconds, local_projection_seconds=projection_seconds)
    budget.live = []
    return primary, row, captures


def geometric_parameters(X):
    spans = np.ptp(X, axis=0)
    feature = int(np.argmax(spans))
    order = np.argsort(X[:, feature], kind='stable')
    index = int(order[len(X)//2-1])
    return dict(feature=feature, cut_value=float(X[index, feature]), cut_row=index,
                lower_count=len(X)//2, setup_named_workspace_bound=8*len(X)+24*X.shape[1])


def auxiliary_block(op, q, X, kind, cycle, parameters):
    B = np.empty((len(q), 2), order='F')
    with op.budget.hold(q, B):
        B[:, 0] = q
        if kind == 'block_gaussian':
            generator = np.random.default_rng(np.random.SeedSequence([0, 6220, 71, cycle]))
            generator.standard_normal(len(q), out=B[:, 1])
            B[:, 1] *= np.linalg.norm(q)/np.linalg.norm(B[:, 1])
        else:
            values = X[op.R, parameters['feature']]
            mask = values < parameters['cut_value']
            ties = values == parameters['cut_value']
            row_ties = op.R <= parameters['cut_row']
            op.budget.observe('geometric_auxiliary_mask', values, mask, ties, row_ties)
            np.logical_and(ties, row_ties, out=ties)
            np.logical_or(mask, ties, out=mask)
            np.multiply(B[:, 0], mask, out=B[:, 1])
            del values, mask, ties, row_ties
        op.budget.observe('initial_auxiliary_B')
    auxiliary_norm = float(np.linalg.norm(B[:, 1]))
    if auxiliary_norm == 0.:
        one = B[:, :1].copy()
        op.budget.observe('zero_auxiliary_removal', q, B, one)
        return one, dict(initial_width=1, auxiliary_norm=0., zero_auxiliary_removed=True)
    return B, dict(initial_width=2, auxiliary_norm=auxiliary_norm, zero_auxiliary_removed=False)


def solve(X, b, bw, method, helper, *, m=None, cpu_cap=6., deadline=float('inf')):
    started, cpu_started = perf_counter(), process_time()
    stop_cpu = min(deadline, cpu_started+cpu_cap)
    if m is None:
        m = 552 if len(b) == 2048 else 592
    op = None
    full_x = None
    cycles = []
    original_relative = None
    status, error = 'correction_cycle_cap', None
    aux_seconds = block_seconds = partition_seconds = 0.
    parameters = None
    setup_seconds = None
    try:
        if method == 'block_geometric':
            t = perf_counter()
            parameters = geometric_parameters(X)
            partition_seconds = perf_counter()-t
        op = component.BlockOperator(X, b, bw, m, helper)
        full_x, full_r = np.zeros(len(b)), b.copy()
        norm_b = float(np.linalg.norm(b))
        absolute_target = .5e-8*norm_b
        op.budget.live = [full_x, full_r]
        op.budget.observe('block_outer_full_state')
        setup_seconds = perf_counter()-started
        for cycle in range(4):
            counts_before = dict(op.counts)
            initial_relative = float(np.linalg.norm(full_r)/norm_b)
            op.budget.live = [full_x, full_r]
            q = op.reduce(full_r)
            qnorm = float(np.linalg.norm(q))
            aux_info = None
            if qnorm <= absolute_target:
                tail = np.zeros_like(q)
                op.budget.observe('already_small_reduced_rhs', q, tail)
                del q
                inner = dict(iterations=0, converged=True, breakdown=None,
                    true_relative_Schur_residual=0., absolute_target=absolute_target,
                    initial_q_norm=qnorm, skipped_small_q=True,
                    approximate_Schur_calls=0, approximate_Schur_rhs=0,
                    approximate_Schur_width_histogram={}, QR_seconds=0., local_projection_seconds=0.)
            else:
                op.budget.live = [full_x, full_r]
                t = perf_counter()
                B, aux_info = auxiliary_block(op, q, X, method, cycle, parameters)
                aux_seconds += perf_counter()-t
                del q
                with op.budget.hold(full_x, full_r):
                    t = perf_counter()
                    tail, inner, _ = block_cg(op, B, rtol=absolute_target/qnorm,
                                              deadline=stop_cpu)
                    block_seconds += perf_counter()-t
                del B
                inner.update(absolute_target=absolute_target, initial_q_norm=qnorm,
                             skipped_small_q=False)
            op.budget.live = [full_x, full_r, tail]
            op.lift_add(full_r, tail, full_x)
            op.budget.live = [full_x]
            del tail, full_r
            full_r = op.original_residual(full_x, b)
            op.budget.live = [full_x, full_r]
            original_relative = float(np.linalg.norm(full_r)/norm_b)
            op.budget.observe('original_component_norms', temporary_allowance=8*max(m, len(b)-m))
            cycles.append(dict(cycle=cycle, initial_original_relative_residual=initial_relative,
                final_original_relative_residual=original_relative,
                final_landmark_relative_residual=float(np.linalg.norm(full_r[op.S])/norm_b),
                final_tail_relative_residual=float(np.linalg.norm(full_r[op.R])/norm_b),
                auxiliary=aux_info, inner=inner,
                operation_counts={key:op.counts[key]-counts_before[key] for key in op.counts}))
            if original_relative <= 1e-8:
                status = 'converged'
                break
            if inner['breakdown'] is not None:
                status = inner['breakdown']
                break
            if not inner['converged']:
                status = 'inner_model_nonconvergence'
                break
    except (MemoryError, np.linalg.LinAlgError) as caught:
        status = 'memory_failure' if isinstance(caught, MemoryError) else 'numerical_failure'
        error = str(caught)
    row = dict(method=method, m=m, status=status, error=error,
        converged=original_relative is not None and original_relative <= 1e-8,
        final_original_relative_residual=original_relative,
        pre_repair_original_relative_residual=cycles[0]['final_original_relative_residual'] if cycles else None,
        correction_cycles=max(0, len(cycles)-1), cycles=cycles,
        total_block_iterations=sum(c['inner']['iterations'] for c in cycles),
        total_approximate_Schur_calls=sum(c['inner']['approximate_Schur_calls'] for c in cycles),
        total_approximate_Schur_rhs=sum(c['inner']['approximate_Schur_rhs'] for c in cycles),
        total_QR_seconds=sum(c['inner']['QR_seconds'] for c in cycles),
        total_local_projection_seconds=sum(c['inner']['local_projection_seconds'] for c in cycles),
        auxiliary_setup_seconds=aux_seconds, partition_setup_seconds=partition_seconds,
        block_CG_seconds=block_seconds, geometric_parameters=parameters,
        setup_seconds=setup_seconds, total_seconds=perf_counter()-started,
        total_cpu_seconds=process_time()-cpu_started,
        common_immutable_input_bytes=X.nbytes+b.nbytes,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if op is not None:
        row.update(op.report())
        row['paid_core_products'] = op.counts['action_core_products']+op.counts['boundary_core_products']
    return full_x, row


def tiny_check(helper):
    rng = np.random.default_rng(622113)
    X = np.ascontiguousarray(rng.normal(size=(29, 4)))
    b = rng.normal(size=29)
    op = component.BlockOperator(X, b, 1.3, 11, helper)
    q = op.reduce(b)
    B, _ = auxiliary_block(op, q, X, 'block_gaussian', 0, None)
    norm_q = np.linalg.norm(q)
    del q
    _, inner, captures = block_cg(op, B, rtol=.5e-8*np.linalg.norm(b)/norm_q, capture=True)
    K = cdist(X, X, 'sqeuclidean')
    K /= 1.3
    K /= 1.3
    K *= -.5
    np.exp(K, out=K)
    A = K+1e-5*np.eye(len(b))
    C = A[np.ix_(op.S, op.S)]
    cross = A[np.ix_(op.S, op.R)]
    L = np.zeros((op.m, op.m))
    for j in range(op.m):
        source = op.factor.wide if op.factor.high[j] else op.factor.low
        offset = op.factor.offsets[j]
        L[j:, j] = source[offset:offset+op.m-j]
    H0 = linalg.cho_solve((L, True), np.eye(op.m))
    H1 = 2*H0-H0@C@H0
    S1 = A[np.ix_(op.R, op.R)]-cross.T@H1@cross
    powers, raw, errors = [], B.copy(), []
    for captured in captures:
        powers.append(raw)
        Q, _ = linalg.qr(np.column_stack(powers), mode='economic')
        target = Q@linalg.solve(Q.T@S1@Q, Q.T@B, assume_a='pos')
        errors.append(float(np.linalg.norm(captured-target[:, 0])/np.linalg.norm(target[:, 0])))
        raw = S1@raw
    assert inner['converged'] and max(errors) < 1e-8
    actual, row = solve(X, b, 1.3, 'block_gaussian', helper, m=11, cpu_cap=2.)
    exact = linalg.solve(A, b, assume_a='pos')
    solution_error = float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
    assert row['converged'] and solution_error < 1e-7
    return dict(first_two_primary_Galerkin_relative_errors=errors,
                independent_full_solution_relative_error=solution_error,
                original_relative_residual=row['final_original_relative_residual'])


def audit(X, b, bw, method, helper):
    tracemalloc.start()
    try:
        answer, row = solve(X, b, bw, method, helper, cpu_cap=8.)
        peak = tracemalloc.get_traced_memory()[1]
        return dict(method=method, whole_solve_traced_peak_bytes=peak,
                    trace_within_1MiB=peak <= LIMIT, metrics=row)
    finally:
        tracemalloc.stop()


def main():
    output = ROOT/'results/native-block-schur.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    prior_path = ROOT/'results/schur-block-action-cost.json'
    previous = json.loads(prior_path.read_text())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in previous['sources'].items())
    sources = [Path(__file__), prior_path, *[ROOT/p for p in previous['sources']]]
    helper = Helper(ROOT/'results/mixed_column_factor_first8.so')
    raw_helper = RawPointers(ROOT/'results/packed_float_factor.so')
    methods = ['original_exact416', 'FIRST8_H1', 'FIRST8_H2prefix4_then_H1',
               'block_gaussian', 'block_geometric']
    cases = [('smooth', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('mixed', 'synthetic-n2560-mixed-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(cases=cases, methods=methods, regularization=1e-5, seed=0, repeats=1,
            block_core={'2048':552, '824':592, '1202':592}, scalar_core=592, exact_core=416,
            recurrence='Validated QR-normalized standard blockCG with two local A-projections, auxiliary deflation, rank loss monitoring, every20/suspected model checks and final primary model check. Identity Schur preconditioner.',
            Gaussian_auxiliary='SeedSequence[0,6220,71,cycle], independently generated g scaled to||q||.',
            geometric_auxiliary='B=[q,mask*q], fixed median stable split along widest-range training feature. Scalar cutvalue/cutrow retained; transient mask freed before block loop.',
            stopping='Block relative rtol=(.5e-8||originalb||)/||q||. Full original1e-8 verification after each correction; maximum3 additional full-original correction cycles.',
            memory='Same1MiB owned-array plus64KiB reserve model; actual full-loop/IR trace gate. No rank shrink.',
            controls='Unchanged scalar wrapper with the same specializedFIRST8helper; unchanged original_exact416 code.',
            per_run_cpu_cap=6, audit_cpu_cap=8, total_cpu_cap=80, command_wall_cap=120,
            scope='Fixed standard block controls only; no adaptive partition or new principle claim.'),
        tiny_check=None, audits=[], runs=[], failures=[])

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
            record['tiny_check'] = tiny_check(helper)
            save()
            for name, filename, method in [('smooth', cases[0][1], 'block_gaussian'),
                                            ('Airfoil', cases[3][1], 'block_geometric')]:
                with np.load(ROOT/'data/prepared'/filename) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                result = audit(X, b, bw, method, helper)
                result.update(dataset=name, regularization=1e-5)
                record['audits'].append(result)
                save()
                if not result['trace_within_1MiB'] or result['metrics']['status'] == 'memory_failure':
                    record['status'] = 'allocation_gate_failed'
                    save()
                    return
            for case_index, (name, filename) in enumerate(cases):
                path = ROOT/'data/prepared'/filename
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                X.setflags(write=False)
                b.setflags(write=False)
                shift = case_index % len(methods)
                order = methods[shift:]+methods[:shift]
                for position, method in enumerate(order):
                    if process_time()-cpu_started >= 72.:
                        row = dict(status='global_cpu_budget_not_started', converged=False)
                    else:
                        assert not tracemalloc.is_tracing()
                        if method.startswith('block_'):
                            answer, row = solve(X, b, bw, method, helper, deadline=cpu_started+78.)
                        elif method == 'original_exact416':
                            answer, row = scalar.exact_control.run(X, b, bw, 1e-5, method, raw_helper,
                                                                  deadline=cpu_started+78.)
                        else:
                            answer, row = scalar.solve(X, b, bw, 1e-5, method, helper, raw_helper,
                                                       deadline=cpu_started+78.)
                        del answer
                    row.update(dataset=name, n=len(b), regularization=1e-5, method=method,
                               order_position=position, data_sha256=sha256(path.read_bytes()).hexdigest())
                    record['runs'].append(row)
                    save()
                    print(json.dumps({key:row.get(key) for key in ['dataset', 'method', 'status', 'total_seconds', 'total_block_iterations', 'total_inner_iterations', 'iterations', 'correction_cycles', 'final_original_relative_residual', 'true_relative_residual']}), flush=True)
        except Exception as error:
            record['status'] = 'failed'
            record['failures'].append(dict(error=repr(error)))
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
