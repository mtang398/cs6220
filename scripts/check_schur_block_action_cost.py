#!/usr/bin/env python3
"""Width-two Schur action resource/cost component; not a block solver."""
import ast
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
import check_native_first8_schur as native
from check_core_precision_allocation import Helper
from check_schur_inplace_frontier_v2 import ROOT, LIMIT, RESERVE


QR_PATH = ROOT/'scripts/check_block_pcg_width2.py'
QR_TEXT = QR_PATH.read_text()
QR_NODE = next(node for node in ast.parse(QR_TEXT).body
               if isinstance(node, ast.FunctionDef) and node.name == 'orthonormal_search')
QR_SOURCE = ast.get_source_segment(QR_TEXT, QR_NODE)
qr_namespace = dict(np=np, linalg=linalg)
exec(compile(QR_SOURCE, str(QR_PATH), 'exec'), qr_namespace)
orthonormal_search = qr_namespace['orthonormal_search']


def phase_plan(n, d, m):
    tail, width = n-m, 2
    factor = 2*m*(m+1)+4*(8*m-28)
    fixed = factor+9*m+8*n
    outer = 16*n
    block = 8*tail*width
    core = 8*m*width
    small = 16*width*width*8
    work = 4096
    first_tile4 = 4*8*(tail+width+2)
    phases = dict(
        construction=max(factor+41*m+8*m*d+work,
                         factor+17*m+9*n+8*tail+work),
        # B[:,active] is replaced by its equal contiguous-prefix view. The
        # fixed old active orders are only [0,1] or [0]. No extra RHS is hidden.
        first_pass_four_rows=fixed+outer+5*block+16*n+8*tail*d+first_tile4+work,
        split_first_pass=fixed+outer+6*block+16*n+core,
        core_product=fixed+outer+6*block+4*core+8*m*d+8*(m+d+width+2)+work,
        core_mixed_solves=fixed+outer+6*block+4*core+8*m+work,
        second_pass=fixed+outer+6*block+core+8*m*d+8*(m+d+width+2)+work,
        QR_old_workspace_bound=fixed+outer+9*block+small+work,
        true_block_residual_subtract=fixed+outer+7*block+small,
        auxiliary_deflation_copies=fixed+outer+7*block+small,
        primary_lift_core=fixed+outer+5*block+8*tail+32*m+8*m*d+8*(m+d+2)+work,
        primary_lift_cross=fixed+outer+5*block+8*tail+8*m+8*tail*d+8*(tail+d+2)+work,
        original_verification=fixed+outer+5*block+8*n+8*(n+2)+work,
        original_residual_update=fixed+outer+5*block+16*n)
    peak_phase = max(phases, key=phases.get)
    return dict(m=m, n=n, d=d, factor_bytes=factor, metadata_bytes=9*m,
        permutation_bytes=8*n, outer_full_arrays_bytes=outer,
        width_two_tail_block_bytes=block, phase_bytes=phases,
        limiting_phase=peak_phase, named_peak_bytes=phases[peak_phase],
        peak_with_reserve_bytes=phases[peak_phase]+RESERVE,
        feasible=phases[peak_phase]+RESERVE <= LIMIT)


def capacity(n, d):
    low, high = 8, min(592, n-2)
    assert phase_plan(n, d, low)['feasible']
    while low < high:
        mid = (low+high+1)//2
        if phase_plan(n, d, mid)['feasible']:
            low = mid
        else:
            high = mid-1
    selected = phase_plan(n, d, low)
    selected['next_rank_plan'] = phase_plan(n, d, low+1) if low < min(592, n-2) else None
    return selected


class VariableOperator(native.Operator):
    def __init__(self, X, b, bw, m, helper):
        self.method, self.lam, self.m, self.n = 'FIRST8_H1', 1e-5, m, len(b)
        self.counts = dict(action_core_products=0, boundary_core_products=0,
            Schur_actions=0, boundary_cross_products=0, original_verifications=0)
        self.times = dict(core_products=0., Schur_actions=0., reduced_rhs=0.,
                          lift_correction=0., original_verification=0.)
        self.owner = None
        self.factor = native.StreamedFirst8(X, b, bw, m, self.lam, helper)
        self.budget, self.kernel = self.factor.budget, self.factor.kernel
        self.permutation, self.S, self.R = self.factor.permutation, self.factor.S, self.factor.R
        self.factor_bytes = self.factor.low.nbytes+self.factor.wide.nbytes
        self.metadata_bytes = self.factor.high.nbytes+self.factor.offsets.nbytes
        self.FPS_seconds, self.factor_seconds = self.factor.FPS_seconds, self.factor.factor_seconds
        self.minimum_raw_pivot = self.factor.minimum_raw_pivot


class BlockOperator(VariableOperator):
    def product_block(self, rows, columns, block, *, label, target=None, subtract=False):
        row_count = self.n if rows is None else len(rows)
        col_count = self.n if columns is None else len(columns)
        width = block.shape[1]
        points_columns = self.kernel.X if columns is None else self.kernel.X[columns]
        result = np.empty((row_count, width), order='F') if target is None else target
        available = LIMIT-RESERVE-self.budget.size(points_columns, result, block)-4096
        per_row = 8*(col_count+(self.kernel.d if rows is not None else 0)+width+2)
        row_tile = min(128, int(available//per_row))
        if row_tile < 1:
            raise MemoryError(f'No matrix row tile fits at {label}')
        self.budget.row_sizes.setdefault(label, set()).add(row_tile)
        for start in range(0, row_count, row_tile):
            stop = min(row_count, start+row_tile)
            points = self.kernel.X[start:stop] if rows is None else self.kernel.X[rows[start:stop]]
            tile = self.kernel.block(points, points_columns)
            product = tile@block
            self.budget.observe(label, points_columns, result, block, points, tile, product)
            if subtract:
                result[start:stop] -= product
            else:
                result[start:stop] = product
            del tile, product, points
        return result

    def solve_block(self, rhs):
        answer = np.empty(rhs.shape, dtype=np.float64, order='F')
        with self.budget.hold(rhs, answer):
            for j in range(rhs.shape[1]):
                column = self.factor.solve(rhs[:, j])
                self.budget.observe('block_mixed_column_solve', column)
                answer[:, j] = column
                del column
        return answer

    def apply_block(self, block):
        values = self.product_block(None, self.R, block, label='block_Schur_first_pass')
        t, result = values[self.S, :], values[self.R, :]
        self.budget.observe('block_Schur_split', values, t, result, block)
        del values
        with self.budget.hold(t, result, block):
            top = self.solve_block(t)
            with self.budget.hold(top):
                residual = self.product_block(self.S, self.S, top, label='block_core_product')
                self.budget.observe('block_core_shift', residual, temporary_allowance=residual.nbytes)
                residual += self.lam*top
                residual *= -1.
                residual += t
                with self.budget.hold(residual):
                    correction = self.solve_block(residual)
                    self.budget.observe('block_core_update', correction)
                    top += correction
                    del correction
                del residual
            self.budget.observe('block_Schur_shift', top, temporary_allowance=result.nbytes)
            result += self.lam*block
        del t
        with self.budget.hold(top, result, block):
            self.product_block(self.R, self.S, top, label='block_Schur_second_pass',
                               target=result, subtract=True)
        return result


def live(op, state, *extra):
    op.budget.live = [state[key] for key in ['full_x', 'full_r', 'B', 'X', 'R', 'P', 'AP'] if key in state]+list(extra)


def block_state(op, b):
    size = len(op.R)
    state = dict(full_x=np.zeros(op.n), full_r=b.copy())
    live(op, state)
    state['B'] = np.empty((size, 2), order='F')
    live(op, state)
    q = op.reduce(b)
    op.budget.observe('block_initial_rhs', q)
    state['B'][:, 0] = q
    del q
    np.random.default_rng(622111).standard_normal(size, out=state['B'][:, 1])
    state['X'] = np.zeros_like(state['B'])
    state['R'] = state['B'].copy()
    live(op, state)
    Z = state['R'].copy()
    op.budget.observe('initial_QR_workspace', Z, temporary_allowance=3*Z.nbytes+4608)
    state['P'], _, _ = orthonormal_search(Z)
    del Z
    state['AP'] = np.zeros_like(state['P'])
    live(op, state)
    op.budget.observe('block_retained_state')
    return state


def scalar_state(op, b, direction=None):
    full_x, full_r = np.zeros(op.n), b.copy()
    op.budget.live = [full_x, full_r]
    rhs = op.reduce(b)
    tail, r = np.zeros_like(rhs), rhs.copy()
    p = rhs.copy() if direction is None else direction.copy()
    if direction is None:
        p /= np.linalg.norm(p)
    arrays = [full_x, full_r, rhs, tail, r, p]
    op.budget.live = arrays
    return arrays


def tiny_check(helper):
    rng = np.random.default_rng(622112)
    X = np.ascontiguousarray(rng.normal(size=(29, 4)))
    b = rng.normal(size=29)
    op = BlockOperator(X, b, 1.3, 11, helper)
    # Small independent dense original SPD system, including the implemented
    # H1 action rather than assuming the mixed core inverse is exact.
    K = cdist(X, X, 'sqeuclidean')
    K /= 1.3
    K /= 1.3
    K *= -.5
    np.exp(K, out=K)
    A = K+1e-5*np.eye(len(b))
    C = A[np.ix_(op.S, op.S)]
    B = A[np.ix_(op.S, op.R)]
    L = np.zeros((op.m, op.m))
    for j in range(op.m):
        source = op.factor.wide if op.factor.high[j] else op.factor.low
        offset = op.factor.offsets[j]
        L[j:, j] = source[offset:offset+op.m-j]
    H0 = linalg.cho_solve((L, True), np.eye(op.m))
    H1 = 2*H0-H0@C@H0
    S1 = A[np.ix_(op.R, op.R)]-B.T@H1@B
    vectors = rng.normal(size=(len(op.R), 2))
    block = op.apply_block(vectors)
    scalar = np.column_stack([op.apply(vectors[:, j], 1) for j in range(2)])
    dense = S1@vectors
    dense_error = float(np.linalg.norm(block-dense)/np.linalg.norm(dense))
    scalar_error = float(np.linalg.norm(block-scalar)/np.linalg.norm(scalar))
    assert dense_error < 1e-10 and scalar_error < 1e-10
    return dict(dense_H1_Schur_relative_error=dense_error,
                independent_scalar_columns_relative_error=scalar_error,
                minimum_dense_Schur_eigenvalue=float(linalg.eigvalsh(S1)[0]))


def allocation_audit(X, b, bw, m, helper):
    op = None
    stage_peaks = {}
    tracemalloc.start()
    try:
        op = BlockOperator(X, b, bw, m, helper)
        state = block_state(op, b)
        stage_peaks['setup_and_initial_QR'] = tracemalloc.get_traced_memory()[1]
        AP_new = op.apply_block(state['P'])
        op.budget.observe('old_and_new_AP', AP_new)
        state['AP'] = AP_new
        del AP_new
        live(op, state)
        stage_peaks['matrix_action'] = tracemalloc.get_traced_memory()[1]
        small = (state['P'].T@state['AP']+state['AP'].T@state['P'])*.5
        chol = linalg.cho_factor(small, lower=True, check_finite=False)
        alpha = linalg.cho_solve(chol, state['P'].T@state['R'], check_finite=False)
        for target, source, sign in [('X', 'P', 1.), ('R', 'AP', -1.)]:
            temporary = state[source]@alpha
            op.budget.observe('block_Galerkin_update', temporary, small, alpha)
            if sign > 0:
                state[target] += temporary
            else:
                state[target] -= temporary
            del temporary
        Z = state['R'].copy()
        with op.budget.hold(Z):
            for _ in range(2):
                coefficients = linalg.cho_solve(chol, state['AP'].T@Z, check_finite=False)
                temporary = state['P']@coefficients
                op.budget.observe('local_reprojection', temporary, small, alpha, coefficients)
                Z -= temporary
                del temporary
            op.budget.observe('QR_old_workspace_bound', small, alpha,
                temporary_allowance=3*Z.nbytes+4608)
            P_next, singular, threshold = orthonormal_search(Z)
            op.budget.observe('QR_after', P_next, singular, small, alpha,
                temporary_allowance=2*Z.nbytes+4608)
            rank = P_next.shape[1]
            del P_next
        del Z
        stage_peaks['QR_and_reprojection'] = tracemalloc.get_traced_memory()[1]
        primary = state['X'][:, 0].copy()
        live(op, state, primary)
        op.lift_add(state['full_r'], primary, state['full_x'])
        live(op, state)
        del primary
        original = op.original_residual(state['full_x'], b)
        op.budget.observe('block_with_original_verification', original)
        del original
        stage_peaks['primary_lift_and_original_verification'] = tracemalloc.get_traced_memory()[1]
        # Exact old active orders permit this view instead of fancy indexing.
        active_B = state['B'][:, :2]
        with op.budget.hold(active_B):
            AX = op.apply_block(state['X'])
            true_R = active_B-AX
            op.budget.observe('true_block_residual', AX, true_R)
            del AX
        # Exercise the old auxiliary-deflation copy phases regardless of
        # convergence. This is an allocation simulation, not a solver event.
        live(op, state, true_R)
        X_new, R_new = state['X'][:, :1].copy(), state['R'][:, :1].copy()
        op.budget.observe('auxiliary_deflation_copies', X_new, R_new)
        state['X'], state['R'] = X_new, R_new
        del X_new, R_new
        live(op, state, true_R)
        new_true = true_R[:, :1].copy()
        op.budget.observe('auxiliary_true_residual_copy', new_true)
        del new_true, true_R
        live(op, state)
        stage_peaks['true_check_and_auxiliary_deflation'] = tracemalloc.get_traced_memory()[1]
        peak = tracemalloc.get_traced_memory()[1]
        minimum_rows = min(op.budget.row_sizes['block_Schur_first_pass'])
        return dict(status='passed', m=m, traced_peak_bytes=peak,
            trace_within_1MiB=peak <= LIMIT, first_pass_minimum_rows=minimum_rows,
            minimum_four_rows_passed=minimum_rows >= 4,
            phase_cumulative_traced_peaks=stage_peaks, simulated_QR_rank=rank,
            memory=op.budget.report(), kernel_stats=dict(op.kernel.stats),
            qualification='Actual component action/QR/update/verification/deflation phases, not a complete block solver.')
    except (MemoryError, np.linalg.LinAlgError) as error:
        peak = tracemalloc.get_traced_memory()[1]
        return dict(status='failed', m=m, error=str(error), traced_peak_bytes=peak,
            phase_cumulative_traced_peaks=stage_peaks,
            memory=None if op is None else op.budget.report(),
            live_failure=None if op is None else op.budget.last_failure)
    finally:
        tracemalloc.stop()


def column_validation(X, b, bw, m, helper):
    block_op = BlockOperator(X, b, bw, m, helper)
    state = block_state(block_op, b)
    vectors = state['P'].copy()
    actual = block_op.apply_block(state['P'])
    del block_op, state
    scalar_op = VariableOperator(X, b, bw, m, helper)
    errors = []
    for j in range(2):
        arrays = scalar_state(scalar_op, b, vectors[:, j])
        expected = scalar_op.apply(arrays[-1], 1)
        errors.append(float(np.linalg.norm(actual[:, j]-expected)/np.linalg.norm(expected)))
        del arrays, expected
    assert max(errors) < 5e-8
    return dict(column_relative_errors=errors,
        scope='Independent scalar references retained outside the component allocation/timing claims.')


def time_one(X, b, bw, m, method, helper):
    setup_started = perf_counter()
    if method == 'block_width2':
        op = BlockOperator(X, b, bw, m, helper)
        state = block_state(op, b)
        argument = state['P']
    else:
        op = VariableOperator(X, b, bw, m, helper)
        state = scalar_state(op, b)
        argument = state[-1]
    setup = perf_counter()-setup_started
    stats_before = dict(op.kernel.stats)
    started, cpu = perf_counter(), process_time()
    answer = op.apply_block(argument) if method == 'block_width2' else op.apply(argument, 1)
    elapsed, elapsed_cpu = perf_counter()-started, process_time()-cpu
    result = dict(method=method, m=m, action_seconds=elapsed, action_cpu_seconds=elapsed_cpu,
        separately_measured_setup_seconds=setup,
        action_kernel_stats={key:op.kernel.stats[key]-stats_before[key] for key in stats_before},
        action_output_norm=float(np.linalg.norm(answer)), memory=op.budget.report())
    del answer, argument, state, op
    return result


def main():
    output = ROOT/'results/schur-block-action-cost.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (58, 60))
    started, cpu_started = perf_counter(), process_time()
    prior_path = ROOT/'results/native-first8-schur.json'
    prior = json.loads(prior_path.read_text())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in prior['sources'].items())
    sources = [Path(__file__), QR_PATH, prior_path, *[ROOT/p for p in prior['sources']]]
    specialization_path = ROOT/'results/first8-specialization.json'
    specialization = json.loads(specialization_path.read_text())
    assert specialization['status'] == 'complete'
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in specialization['sources'].items())
    specialized_library = ROOT/'results/mixed_column_factor_first8.so'
    assert sha256(specialized_library.read_bytes()).hexdigest() == specialization['specialized_library_sha256']
    sources.extend([specialization_path, specialized_library, *[ROOT/p for p in specialization['sources']]])
    helper = Helper(specialized_library)
    cases = [('smooth2048', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        orthonormal_search_sha256=sha256(QR_SOURCE.encode()).hexdigest(),
        protocol=dict(cases=cases, regularization=1e-5, width=2, repeats=5,
            capacity='Binary search only on named phase-byte formulas, m<=592; all phases fit1MiB including64KiB reserve and first-pass minimum4rows. No timed rank tuning.',
            states='B=[q,g], X,R,P,AP width2; fullouterx/fullresidual; actual validatedQR, nine-block old workspace bound, reprojection, true-check and auxiliary-deflation copies; core buffers and feature packs.',
            active_B='B[:,active] replaced with exactly equal B[:,:len(active)]view for the old active orders[0,1]and[0]; all other declared QR/deflation copies retained.',
            action='H1 Schur action; each Gaussian tile generated once for both columns, including originalC residual product; per-column mixed triangular solves.',
            helper='Verified bitwise-equivalent FIRST8 specialization used for all block and scalar methods, selected before this gate execution.',
            comparisons=['block_width2 at selectedm', 'scalar_same_core', 'scalar592'],
            timing='Five cyclic single-action repeats. Fresh factor/state rebuilt and freed for each method; setup separately measured, not charged to action ratio. No tracing during timings.',
            scope='Component/phase feasibility only. Previousrank32block failure remains; previous GaussianKernel.matmat already shared tiles. No block iteration or end-to-end benefit claim.',
            cpu_cap=58, wall_cap=90), tiny_check=None, cases=[], failures=[])

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
            for name, filename in cases:
                path = ROOT/'data/prepared'/filename
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                plan = capacity(len(b), X.shape[1])
                case = dict(dataset=name, n=len(b), data_sha256=sha256(path.read_bytes()).hexdigest(),
                            resource_plan=plan, timings=[])
                record['cases'].append(case)
                case['allocation_audit'] = allocation_audit(X, b, bw, plan['m'], helper)
                save()
                audit = case['allocation_audit']
                if audit['status'] != 'passed' or not audit['trace_within_1MiB'] or not audit['minimum_four_rows_passed']:
                    record['status'] = 'allocation_gate_failed'
                    save()
                    return
                case['column_validation'] = column_validation(X, b, bw, plan['m'], helper)
                save()
                methods = ['block_width2', 'scalar_same_core', 'scalar592']
                for repeat in range(5):
                    order = methods[repeat%3:]+methods[:repeat%3]
                    for position, method in enumerate(order):
                        if process_time()-cpu_started > 52.:
                            row = dict(method=method, status='global_cpu_budget_not_started')
                        else:
                            assert not tracemalloc.is_tracing()
                            row = time_one(X, b, bw, 592 if method == 'scalar592' else plan['m'], method, helper)
                            row['status'] = 'complete'
                        row.update(repeat=repeat, order_position=position)
                        case['timings'].append(row)
                        save()
                means = {method:float(np.mean([row['action_seconds'] for row in case['timings']
                    if row['method'] == method and row['status'] == 'complete'])) for method in methods}
                case['action_means_seconds'] = means
                case['block_to_scalar_same_core_cost'] = means['block_width2']/means['scalar_same_core']
                case['block_to_scalar592_cost'] = means['block_width2']/means['scalar592']
                case['required_block_iterations_over_scalar592_for_action_break_even'] = means['scalar592']/means['block_width2']
                case['required_block_iterations_over_same_core_scalar_for_action_break_even'] = means['scalar_same_core']/means['block_width2']
                case['caveat'] = 'Iteration ratios are required thresholds only; smaller block core may change convergence. Setup, QR/updates/deflation and verification are not included in the action-only break-even.'
                save()
                print(json.dumps(dict(dataset=name, selected_m=plan['m'],
                    trace=audit['traced_peak_bytes'], first_pass_rows=audit['first_pass_minimum_rows'],
                    action_means=means, block_to_scalar592=case['block_to_scalar592_cost'])), flush=True)
        except Exception as error:
            record['status'] = 'failed'
            record['failures'].append(dict(error=repr(error)))
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
