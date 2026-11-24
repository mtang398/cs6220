#!/usr/bin/env python3
"""Native streamed static FIRST8 factors with original-residual correction.

No dense kernel/core matrix. Common immutable X,b are reported separately.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_limits
import check_compressed_core_schur as old_native
import check_compressed_core_schur_odd as exact_control
from check_packed_float_allocation import RawPointers
from check_core_precision_allocation import Helper, pointer
from check_schur_inplace_frontier_v2 import SharedKernel, ROOT, LIMIT, RESERVE


class StreamedFirst8:
    def __init__(self, X, b, bw, m, lam, helper):
        self.m, self.n, self.lam, self.helper = m, len(b), lam, helper
        self.budget = old_native.ScopedBudget([X, b])
        self.kernel = SharedKernel(X, bw, self.budget)
        started = perf_counter()
        indices = self.kernel.farthest(m)
        self.FPS_seconds = perf_counter()-started
        self.high = np.zeros(m, dtype=np.uint8)
        self.high[:8] = 1
        self.offsets = np.empty(m, dtype=np.uintp)
        lengths = [0, 0]
        for j in range(m):
            kind = int(self.high[j])
            self.offsets[j] = lengths[kind]
            lengths[kind] += m-j
        self.low = np.empty(lengths[0], dtype=np.float32)
        self.wide = np.empty(lengths[1], dtype=np.float64)
        epsilon, pivots = np.empty(m), np.empty(m)
        factor_arrays = [self.high, self.offsets, self.low, self.wide]
        self.budget.fixed = [indices, *factor_arrays, epsilon, pivots]
        points = X[indices]
        self.budget.observe('mixed_stream_setup', points)
        started = perf_counter()
        with self.budget.hold(points):
            for j in range(m):
                work = self.kernel.block(points[j:], points[j:j+1]).reshape(m-j)
                work[0] += lam
                self.budget.observe('mixed_stream_column', work)
                info = helper.column(m, j, pointer(self.high), pointer(self.offsets),
                    pointer(self.low), pointer(self.wide), pointer(work),
                    pointer(epsilon[j:j+1]), pointer(pivots[j:j+1]))
                del work
                if info:
                    raise np.linalg.LinAlgError(f'Mixed pivot {info-1} failed')
        self.factor_seconds = perf_counter()-started
        self.minimum_raw_pivot = float(pivots.min())
        self.initial_epsilon_norm = float(np.linalg.norm(epsilon))
        self.budget.fixed = [indices, *factor_arrays]
        del epsilon, pivots, points
        mask = np.ones(self.n, dtype=bool)
        mask[indices] = False
        tail = np.flatnonzero(mask)
        self.permutation = np.r_[indices, tail]
        self.budget.observe('mixed_permutation', mask, tail, self.permutation)
        self.S, self.R = self.permutation[:m], self.permutation[m:]
        self.budget.fixed = [self.permutation, *factor_arrays]
        del indices, mask, tail
        self.solve_calls = 0
        self.solve_seconds = 0.

    def solve(self, rhs):
        started = perf_counter()
        answer = np.array(rhs, dtype=np.float64, copy=True)
        self.budget.observe('mixed_triangular_solve', rhs, answer)
        info = self.helper.solve(self.m, pointer(self.high), pointer(self.offsets),
            pointer(self.low), pointer(self.wide), pointer(answer))
        if info:
            raise np.linalg.LinAlgError(f'Mixed solve diagonal {info-1} failed')
        self.solve_calls += 1
        self.solve_seconds += perf_counter()-started
        return answer


class Operator:
    def __init__(self, X, b, bw, lam, method, mixed_helper, raw_helper):
        self.method, self.lam = method, lam
        self.m = 600 if method == 'uniform600_H2' else 592
        self.n = len(b)
        self.counts = dict(action_core_products=0, boundary_core_products=0,
            Schur_actions=0, boundary_cross_products=0, original_verifications=0)
        self.times = dict(core_products=0., Schur_actions=0., reduced_rhs=0.,
                          lift_correction=0., original_verification=0.)
        if method == 'uniform600_H2':
            self.owner = old_native.Schur(X, b, bw, self.m, lam, 'fixed2', raw_helper)
            self.factor = self.owner.float
            self.budget, self.kernel = self.owner.budget, self.owner.kernel
            self.permutation, self.S, self.R = self.owner.permutation, self.owner.S, self.owner.R
            self.factor_bytes = self.factor.factor.nbytes
            self.metadata_bytes = 0
            self.FPS_seconds = self.factor.selection_seconds
            self.factor_seconds = self.factor.factor_seconds
            self.minimum_raw_pivot = self.owner.minimum_raw_pivot
        else:
            self.factor = StreamedFirst8(X, b, bw, self.m, lam, mixed_helper)
            self.owner = None
            self.budget, self.kernel = self.factor.budget, self.factor.kernel
            self.permutation, self.S, self.R = self.factor.permutation, self.factor.S, self.factor.R
            self.factor_bytes = self.factor.low.nbytes+self.factor.wide.nbytes
            self.metadata_bytes = self.factor.high.nbytes+self.factor.offsets.nbytes
            self.FPS_seconds = self.factor.FPS_seconds
            self.factor_seconds = self.factor.factor_seconds
            self.minimum_raw_pivot = self.factor.minimum_raw_pivot

    def core_product(self, vector):
        started = perf_counter()
        answer = self.kernel.product(self.S, self.S, vector, label='original_core_product')
        self.budget.observe('original_core_shift', vector, answer, temporary_allowance=vector.nbytes)
        answer += self.lam*vector
        self.times['core_products'] += perf_counter()-started
        return answer

    def refine(self, rhs, corrections, purpose):
        with self.budget.hold(rhs):
            answer = self.factor.solve(rhs)
            with self.budget.hold(answer):
                for _ in range(corrections):
                    residual = self.core_product(answer)
                    self.counts[purpose+'_core_products'] += 1
                    residual *= -1.
                    residual += rhs
                    with self.budget.hold(residual):
                        correction = self.factor.solve(residual)
                        self.budget.observe('refinement_update', correction)
                        answer += correction
                        del correction
                    del residual
        return answer

    def reduce(self, residual):
        started = perf_counter()
        top = self.refine(residual[self.S], 4, 'boundary')
        result = residual[self.R]
        with self.budget.hold(top, result):
            self.kernel.product(self.R, self.S, top, label='reduced_rhs',
                                target=result, subtract=True)
        self.counts['boundary_cross_products'] += 1
        self.times['reduced_rhs'] += perf_counter()-started
        return result

    def apply(self, vector, order):
        started = perf_counter()
        values = self.kernel.product(None, self.R, vector, label='Schur_first_pass')
        t, result = values[self.S], values[self.R]
        self.budget.observe('Schur_split', values, t, result, vector)
        del values
        with self.budget.hold(t, result, vector):
            top = self.refine(t, order, 'action')
            self.budget.observe('Schur_shift', top, temporary_allowance=result.nbytes)
            result += self.lam*vector
        del t
        with self.budget.hold(top, result, vector):
            self.kernel.product(self.R, self.S, top, label='Schur_second_pass',
                                target=result, subtract=True)
        self.counts['Schur_actions'] += 1
        self.times['Schur_actions'] += perf_counter()-started
        return result

    def lift_add(self, residual, tail, full_x):
        started = perf_counter()
        top_rhs = residual[self.S]
        with self.budget.hold(top_rhs):
            self.kernel.product(self.S, self.R, tail, label='correction_cross',
                                target=top_rhs, subtract=True)
            top = self.refine(top_rhs, 4, 'boundary')
            self.budget.observe('full_correction_update', top,
                                temporary_allowance=2*max(top.nbytes, tail.nbytes))
            full_x[self.S] += top
            full_x[self.R] += tail
        self.counts['boundary_cross_products'] += 1
        self.times['lift_correction'] += perf_counter()-started

    def original_residual(self, full_x, b):
        started = perf_counter()
        result = self.kernel.full(full_x, extra=(full_x,))
        self.budget.observe('full_original_update', full_x, result,
                            temporary_allowance=full_x.nbytes)
        result *= -1.
        result += b
        result -= self.lam*full_x
        self.counts['original_verifications'] += 1
        self.times['original_verification'] += perf_counter()-started
        return result

    def report(self):
        return dict(factor_bytes=self.factor_bytes, mixed_flags_offsets_bytes=self.metadata_bytes,
            permutation_bytes=self.permutation.nbytes, epsilon_and_pivots_freed_after_setup=True,
            FPS_seconds=self.FPS_seconds, factor_seconds=self.factor_seconds,
            minimum_raw_pivot=self.minimum_raw_pivot,
            mixed_triangular_solve_calls=self.factor.solve_calls,
            mixed_triangular_solve_seconds=self.factor.solve_seconds,
            phase_seconds=self.times,
            timing_note='Schur actions/reduced RHS/lift include nested core-product and triangular timings; overlapping fields must not be summed.',
            counts=self.counts, kernel_stats=dict(self.kernel.stats), memory=self.budget.report(),
            memory_failure_detail=self.budget.last_failure,
            selected_indices_sha256=sha256(self.S.tobytes()).hexdigest())


def solve(X, b, bw, lam, method, mixed_helper, raw_helper, *, cpu_cap=6., deadline=float('inf')):
    started, cpu_started = perf_counter(), process_time()
    stop_cpu = min(deadline, cpu_started+cpu_cap)
    op = None
    full_x = None
    cycles = []
    actions = []
    total_iterations = 0
    original_relative = None
    status = 'correction_cycle_cap'
    error = None
    setup_seconds = None
    try:
        op = Operator(X, b, bw, lam, method, mixed_helper, raw_helper)
        full_x = np.zeros(len(b))
        full_residual = b.copy()
        op.budget.live = [full_x, full_residual]
        op.budget.observe('outer_full_state')
        norm_b = float(np.linalg.norm(b))
        target = .5e-8*norm_b
        setup_seconds = perf_counter()-started
        for cycle in range(4):
            before = dict(op.counts)
            initial_relative = float(np.linalg.norm(full_residual)/norm_b)
            op.budget.live = [full_x, full_residual]
            rhs = op.reduce(full_residual)
            tail = np.zeros_like(rhs)
            r = rhs.copy()
            p = r.copy()
            rr = float(r@r)
            op.budget.live = [full_x, full_residual, rhs, tail, r, p]
            op.budget.observe('inner_initial_state')
            inner_status = 'already_below_absolute_target' if np.linalg.norm(r) <= target else 'maxiter'
            iterations = 0
            if inner_status == 'maxiter':
                for iterations in range(1, 2001):
                    if process_time() > stop_cpu:
                        inner_status = 'cpu_screen_cap'
                        iterations -= 1
                        break
                    prefix = method == 'FIRST8_H2prefix4_then_H1' and cycle == 0 and iterations <= 4
                    order = 2 if method == 'uniform600_H2' or prefix else 1
                    ap = op.apply(p, order)
                    curvature = float(p@ap)
                    if curvature <= 0 or not np.isfinite(curvature):
                        inner_status = 'nonpositive_curvature'
                        del ap
                        break
                    alpha = rr/curvature
                    op.budget.observe('inner_CG_update', ap, temporary_allowance=2*tail.nbytes)
                    tail += alpha*p
                    r -= alpha*ap
                    del ap
                    actions.append(order)
                    if np.linalg.norm(r) <= target:
                        inner_status = 'absolute_recursive_target'
                        break
                    next_rr = float(r@r)
                    p *= next_rr/rr
                    p += r
                    rr = next_rr
            total_iterations += iterations
            recursive_relative = float(np.linalg.norm(r)/norm_b)
            # The finished inner Krylov vectors are not needed by the lift.
            op.budget.live = [full_x, full_residual, tail]
            del rhs, r, p
            op.lift_add(full_residual, tail, full_x)
            op.budget.live = [full_x]
            del tail, full_residual
            full_residual = op.original_residual(full_x, b)
            op.budget.live = [full_x, full_residual]
            original_relative = float(np.linalg.norm(full_residual)/norm_b)
            op.budget.observe('endpoint_component_norms', temporary_allowance=8*max(op.m, op.n-op.m))
            cycles.append(dict(cycle=cycle, initial_original_relative_residual=initial_relative,
                inner_iterations=iterations, inner_status=inner_status,
                inner_recursive_relative_to_original_b=recursive_relative,
                final_original_relative_residual=original_relative,
                final_landmark_relative_residual=float(np.linalg.norm(full_residual[op.S])/norm_b),
                final_tail_relative_residual=float(np.linalg.norm(full_residual[op.R])/norm_b),
                operation_counts={key:op.counts[key]-before[key] for key in op.counts}))
            if original_relative <= 1e-8:
                status = 'converged'
                break
            if inner_status in ['cpu_screen_cap', 'nonpositive_curvature', 'maxiter']:
                status = inner_status
                break
    except (MemoryError, np.linalg.LinAlgError) as caught:
        status = 'memory_failure' if isinstance(caught, MemoryError) else 'numerical_failure'
        error = str(caught)
    row = dict(method=method, status=status, error=error,
        converged=original_relative is not None and original_relative <= 1e-8,
        m=600 if method == 'uniform600_H2' else 592, total_inner_iterations=total_iterations,
        correction_cycles=max(0, len(cycles)-1), cycles=cycles,
        pre_repair_original_relative_residual=cycles[0]['final_original_relative_residual'] if cycles else None,
        final_original_relative_residual=original_relative, action_orders=actions,
        setup_seconds=setup_seconds, total_seconds=perf_counter()-started,
        total_cpu_seconds=process_time()-cpu_started,
        common_immutable_input_bytes=X.nbytes+b.nbytes,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if op is not None:
        row.update(op.report())
        row['paid_core_products'] = op.counts['action_core_products']+op.counts['boundary_core_products']
    return full_x, row


def audit(X, b, bw, method, mixed_helper, raw_helper):
    tracemalloc.start()
    try:
        answer, row = solve(X, b, bw, 1e-5, method, mixed_helper, raw_helper, cpu_cap=8.)
        current, peak = tracemalloc.get_traced_memory()
        return dict(method=method, whole_solve_traced_peak_bytes=peak,
                    trace_within_1MiB=peak <= LIMIT, metrics=row)
    finally:
        tracemalloc.stop()


def main():
    output = ROOT/'results/native-first8-schur.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    prior_path = ROOT/'results/core-precision-outer-correction.json'
    prior_record = json.loads(prior_path.read_text())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in prior_record['sources'].items())
    source_paths = [Path(__file__), prior_path, *[ROOT/p for p in prior_record['sources']]]
    mixed_helper = Helper(ROOT/'results/mixed_column_factor.so')
    raw_helper = RawPointers(ROOT/'results/packed_float_factor.so')
    methods = ['original_exact416', 'FIRST8_H1', 'FIRST8_H2prefix4_then_H1', 'uniform600_H2']
    cases = [('smooth', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('mixed', 'synthetic-n2560-mixed-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths},
        protocol=dict(cases=cases, methods=methods, regularization=1e-5, seed=0, repeats=1,
            FIRST8_core=592, uniform_core=600, exact_core=416,
            factor='Original streamed Gaussian columns; genuine separate packedFP32/FP64 column stores. FIRST8 fixed before setup. Epsilon and pivots freed after construction.',
            correction='Retain full accumulatedx and originalresidual; boundary4; fixed absolute inner .5e-8||b||; max3additional original-residual correction cycles.',
            baseline='Existing original_exact416 source unchanged, with its original suspected/final verification and residual replacement.',
            memory='Owned arrays +64KiB reserve <=1MiB. Common immutable X,b excluded and reported; independent whole-solve tracemalloc gates.',
            order='Rotate four methods by caseindex.',
            total_cpu_cap=80, command_wall_cap=120, per_attempt_cpu_cap=6.,
            scope='Static native frontier only, no learned allocation or goal-achievement claim.'),
        audits=[], runs=[], failures=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
                      elapsed_cpu_seconds=process_time()-cpu_started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)

    save()
    with threadpool_limits(limits=1):
        audit_specs = [('smooth', cases[0][1], 'FIRST8_H2prefix4_then_H1'),
                       ('Airfoil', cases[3][1], 'FIRST8_H1'),
                       ('smooth', cases[0][1], 'uniform600_H2')]
        for name, filename, method in audit_specs:
            with np.load(ROOT/'data/prepared'/filename) as data:
                X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
            result = audit(X, b, bw, method, mixed_helper, raw_helper)
            result.update(dataset=name, regularization=1e-5)
            record['audits'].append(result)
            save()
            if not result['trace_within_1MiB'] or result['metrics']['status'] == 'memory_failure':
                if method == 'uniform600_H2':
                    methods.remove(method)
                    record['uniform_control_excluded_reason'] = 'Whole-solve allocation gate failed.'
                else:
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
                    if method == 'original_exact416':
                        answer, row = exact_control.run(X, b, bw, 1e-5, method, raw_helper,
                                                       deadline=cpu_started+78.)
                    else:
                        answer, row = solve(X, b, bw, 1e-5, method, mixed_helper, raw_helper,
                                            deadline=cpu_started+78.)
                    del answer
                row.update(dataset=name, regularization=1e-5, method=method, order_position=position,
                           data_sha256=sha256(path.read_bytes()).hexdigest())
                record['runs'].append(row)
                save()
                print(json.dumps({key:row.get(key) for key in ['dataset', 'method', 'status', 'total_inner_iterations', 'iterations', 'correction_cycles', 'total_seconds', 'final_original_relative_residual', 'true_relative_residual']}), flush=True)
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
