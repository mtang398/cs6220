#!/usr/bin/env python3
"""One paid-probe adaptive core-order pilot; original artifacts are immutable."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import check_compressed_core_schur_odd as odd
from check_packed_float_allocation import RawPointers
from check_packed_float_factor import unpack, ROOT


class HalfSolves(RawPointers):
    def __init__(self, original, half_library):
        super().__init__(original)
        self.half_library = ctypes.CDLL(str(half_library))
        self.forward = self.half_library.packed_float_forward
        self.backward = self.half_library.packed_float_backward
        for function in [self.forward, self.backward]:
            function.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int

    def half(self, factor, rhs, *, transpose=False):
        function = self.backward if transpose else self.forward
        result = function(len(rhs), factor.__array_interface__['data'][0],
                          rhs.__array_interface__['data'][0])
        if result:
            raise np.linalg.LinAlgError(f'Invalid half-solve pivot {result-1}')


class Schur(odd.Schur):
    def __init__(self, X, b, bw, m, lam, method, helper):
        self.policy = method
        super().__init__(X, b, bw, m, lam, 'fixed3' if method == 'adaptive' else method, helper)
        self.helper = helper
        self.delta = None
        self.sigma = None
        self.probe_seconds = 0.
        self.probe_kernel_stats = {}
        self.probe_norms = []
        self.remaining_gap = .25e-8*float(np.linalg.norm(b))
        self.initial_gap = self.remaining_gap
        self.spent_gap = 0.
        self.selected_alpha = None
        self.pending_bound = 0.
        self.pending_alpha = None
        self.order_counts = {1:0, 2:0, 3:0}
        self.first_accept_iteration = None
        self.accepted_increment_sum = 0.
        self.maximum_accepted_increment = 0.
        self.alpha_charge_max_difference = 0.
        self.last_order = None
        self.last_energy = None
        self.last_bound = None
        if method == 'adaptive':
            self.setup_probes()

    def setup_probes(self):
        started = perf_counter()
        before = dict(self.kernel.stats)
        multiplier = 10*np.sqrt(2/np.pi)
        for probe in range(6):
            # Recreate one selected column of the frozen (m,6) probe bank.
            # The small bank is transient and counted, never retained.
            bank = np.random.default_rng(622108).normal(size=(self.m, 6))
            g = bank[:, probe].copy()
            self.budget.observe('probe_generation', bank, g)
            del bank
            u = g.copy()
            with self.budget.hold(g, u):
                self.budget.observe('probe_backward', g, u)
                self.helper.half(self.float.factor, u, transpose=True)
                e = self.float.original_product(u)
                self.budget.observe('probe_forward', e)
                self.helper.half(self.float.factor, e)
                e -= g
            del g, u
            e_norm = float(np.linalg.norm(e))
            with self.budget.hold(e):
                self.helper.half(self.float.factor, e, transpose=True)
                z = self.kernel.product(self.R, self.S, e, label='probe_coupling')
                z_norm = float(np.linalg.norm(z))
                del z
            del e
            self.probe_norms.append((e_norm, z_norm))
        self.delta = float(multiplier*max(pair[0] for pair in self.probe_norms))
        self.sigma = float(multiplier*max(pair[1] for pair in self.probe_norms))
        self.probe_kernel_stats = {key:self.kernel.stats[key]-before[key] for key in before}
        self.probe_seconds = perf_counter()-started

    def apply(self, v, *, rr):
        self.selected_alpha = None
        self.pending_alpha = None
        self.pending_bound = 0.
        values = self.kernel.product(None, self.R, v, label='Schur_first_pass')
        t = values[self.S]
        result = values[self.R]
        self.budget.observe('Schur_split', values, t, result, v)
        del values
        with self.budget.hold(t, result, v):
            self.budget.observe('Schur_shift', temporary_allowance=result.nbytes)
            result += self.lam*v
            base_curvature = float(v@result) if self.policy == 'adaptive' else None
            top = self.float.solve(t)
            with self.budget.hold(top):
                residual = self.float.original_product(top)
                self.core_checks += 1
                residual *= -1.
                residual += t
                with self.budget.hold(residual):
                    correction = self.float.solve(residual)
                    self.budget.observe('core_refinement_update', correction)
                    energy = max(float(residual@correction), 0.) if self.policy == 'adaptive' else None
                    top += correction
                    del correction
                del residual
                if self.policy == 'adaptive':
                    denom = base_curvature-float(t@top)
                    alpha1 = rr/denom if denom > 0 else None
                    bound = self.sigma*(1+self.delta)*np.sqrt(energy) if self.delta < 1 else None
                    accept = bool(alpha1 is not None and np.isfinite(alpha1) and bound is not None
                                  and abs(alpha1)*bound <= self.remaining_gap)
                    order = 1 if accept else 3
                    self.last_energy, self.last_bound = energy, bound
                    if accept:
                        self.selected_alpha = alpha1
                        self.pending_alpha = alpha1
                        self.pending_bound = float(bound)
                        if self.first_accept_iteration is None:
                            self.first_accept_iteration = self.calls+1
                else:
                    order = int(self.policy[-1])
                for _ in range(1, order):
                    residual = self.float.original_product(top)
                    self.core_checks += 1
                    residual *= -1.
                    residual += t
                    with self.budget.hold(residual):
                        correction = self.float.solve(residual)
                        self.budget.observe('core_refinement_update', correction)
                        top += correction
                        del correction
                    del residual
        del t
        with self.budget.hold(top, result, v):
            self.kernel.product(self.R, self.S, top, label='Schur_second_pass',
                                target=result, subtract=True)
        self.calls += 1
        self.core_calls += 1
        self.corrections += order
        self.phase_calls['action'] += 1
        self.phase_corrections['action'] += order
        self.stop_reasons['fixed_count'] += 1
        self.correction_counts[order] += 1
        self.order_counts[order] += 1
        self.last_order = order
        return result

    def consume_step(self, alpha):
        if self.pending_alpha is not None:
            self.alpha_charge_max_difference = max(self.alpha_charge_max_difference,
                                                   abs(alpha-self.pending_alpha))
            increment = abs(alpha)*self.pending_bound
            assert increment <= self.remaining_gap*(1+1e-12)
            self.remaining_gap -= increment
            self.spent_gap += increment
            self.accepted_increment_sum += increment
            self.maximum_accepted_increment = max(self.maximum_accepted_increment, increment)

    def report(self):
        return dict(super().report(), policy=self.policy,
            probe_seconds=self.probe_seconds, probe_kernel_stats=self.probe_kernel_stats,
            probe_norms=self.probe_norms, probe_delta=self.delta, probe_sigma=self.sigma,
            initial_gap_budget=self.initial_gap, remaining_gap_budget=self.remaining_gap,
            spent_gap_bound=self.spent_gap, selected_order_counts=self.order_counts,
            first_accept_iteration=self.first_accept_iteration,
            maximum_accepted_increment=self.maximum_accepted_increment,
            alpha_charge_max_difference=self.alpha_charge_max_difference)


SOLVE_SOURCE = odd.WRAPPER_SOURCE
replacements = [
    ("['exact416','fixed1','fixed2','fixed3']", "['adaptive','fixed1','fixed2','fixed3']"),
    ('ap=op.apply(p);curvature=float(p@ap)', 'ap=op.apply(p,rr=rr);curvature=float(p@ap)'),
    ('alpha=rr/curvature',
     'alpha=op.selected_alpha if op.selected_alpha is not None else rr/curvature\n            op.consume_step(alpha)'),
    ('suspected=np.linalg.norm(r)<=rtol*norm_b', 'suspected=np.linalg.norm(r)<=.5*rtol*norm_b')]
for before, after in replacements:
    assert SOLVE_SOURCE.count(before) == 1
    SOLVE_SOURCE = SOLVE_SOURCE.replace(before, after)
solve_namespace = dict(odd.previous.__dict__, Schur=Schur)
exec(compile(SOLVE_SOURCE, '<adaptive-core-order-solve>', 'exec'), solve_namespace)
solve = solve_namespace['solve']


def run(X, b, bw, lam, method, helper, *, m=None, cpu_cap=6., deadline=float('inf')):
    if method == 'original_exact416':
        return odd.run(X, b, bw, lam, method, helper, m=m, cpu_cap=cpu_cap, deadline=deadline)
    return solve(X, b, bw, lam, method, helper, m=m, cpu_cap=cpu_cap,
                 deadline=deadline, verify_every=0)


def tiny_check(helper):
    rng = np.random.default_rng(622109)
    X = np.ascontiguousarray(rng.normal(size=(31, 4)))
    b = rng.normal(size=31)
    bw, lam, m = 1.4, .03, 13
    A = cdist(X, X, 'sqeuclidean')
    A /= bw
    A /= bw
    A *= -.5
    np.exp(A, out=A)
    A.flat[::32] += lam
    reference = linalg.solve(A, b, assume_a='pos')
    op = Schur(X, b, bw, m, lam, 'adaptive', helper)
    L = unpack(op.float.factor, m)
    C = A[np.ix_(op.S, op.S)]
    B = A[np.ix_(op.S, op.R)]
    E = linalg.solve_triangular(L, C, lower=True)
    E = linalg.solve_triangular(L, E.T, lower=True).T-np.eye(m)
    Z = B.T@linalg.solve_triangular(L, E, lower=True, trans='T')
    probes = np.random.default_rng(622108).normal(size=(m, 6))
    expected = np.column_stack([np.linalg.norm(E@probes, axis=0), np.linalg.norm(Z@probes, axis=0)])
    probe_error = float(np.max(np.abs(np.array(op.probe_norms)-expected)))
    assert probe_error < 1e-12
    rhs = rng.normal(size=m)
    half_errors = []
    for transpose in [False, True]:
        actual = rhs.copy()
        helper.half(op.float.factor, actual, transpose=transpose)
        expected_half = linalg.solve_triangular(L, rhs, lower=True, trans='T' if transpose else 'N')
        error = float(np.linalg.norm(actual-expected_half)/np.linalg.norm(expected_half))
        half_errors.append(error)
        assert error < 1e-12
    joined = rhs.copy()
    helper.half(op.float.factor, joined)
    helper.half(op.float.factor, joined, transpose=True)
    joined_bitwise = bool(np.array_equal(joined, op.float.solve(rhs)))
    assert joined_bitwise
    H0 = linalg.cho_solve((L, True), np.eye(m))
    D = A[np.ix_(op.R, op.R)]
    del op
    checks = []
    for method in ['adaptive', 'fixed1', 'fixed2', 'fixed3']:
        action_op = Schur(X, b, bw, m, lam, method, helper)
        direction = rng.normal(size=len(action_op.R))
        observed = action_op.apply(direction, rr=float(direction@direction))
        H = H0.copy()
        for _ in range(action_op.last_order):
            H += H0@(np.eye(m)-C@H)
        expected_action = (D-B.T@H@B)@direction
        action_error = float(np.linalg.norm(observed-expected_action)/np.linalg.norm(expected_action))
        assert action_error < 1e-10
        del action_op
        answer, row = run(X, b, bw, lam, method, helper, m=m, cpu_cap=2.)
        error = float(np.linalg.norm(answer-reference)/np.linalg.norm(reference))
        assert row['converged'] and error < 1e-7
        checks.append(dict(method=method, action_relative_error=action_error, solution_relative_error=error,
                           true_relative_residual=row['true_relative_residual']))
    return dict(half_solve_relative_errors=half_errors, joined_half_solve_bitwise_matches_original=joined_bitwise,
                probe_absolute_error=probe_error,
                full_solve_checks=checks)


def main():
    output = ROOT/'results/compressed-core-adaptive-pilot.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    source = ROOT/'scripts/packed_float_halfsolves.c'
    library = ROOT/'results/packed_float_halfsolves.so'
    compile_command = ['gcc', '-O3', '-march=native', '-ffp-contract=off', '-shared', '-fPIC',
                       str(source), '-o', str(library), '-lm']
    subprocess.run(compile_command, check=True)
    helper = HalfSolves(ROOT/'results/packed_float_factor.so', library)
    old_path = ROOT/'results/compressed-core-schur-odd-pilot.json'
    old = json.loads(old_path.read_text())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in old['sources'].items())
    sources = [Path(__file__), source, library, old_path,
               *[ROOT/p for p in old['sources']]]
    methods = ['original_exact416', 'fixed1', 'fixed2', 'fixed3', 'adaptive']
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        generated_solve_sha256=sha256(SOLVE_SOURCE.encode()).hexdigest(), compile_command=compile_command,
        protocol=dict(datasets=['smooth2048', 'mixed2048'], regularizations=[.001, 1e-5],
            seed=0, repeats=1, methods=methods, candidate_core=600, exact_core=416,
            fixed_boundary_corrections=4, candidate_orders=[1, 3], fixed_controls=[1, 2, 3],
            probe_seed=622108, probe_count=6, probe_multiplier=float(10*np.sqrt(2/np.pi)),
            probes='Sequential Eg=L^-1 C L^-T g-g and Zg=B.T L^-T Eg; no E or Z formed. Transient(m,6) generation counted, only one probe processed at a time.',
            selector='If delta<1, bound=sigma*(1+delta)*sqrt(h0.T H0h0). Accept H1 iff abs(alpha1)*bound<=entire remaining gap budget; otherwise complete H3. Used alpha1 is charged.',
            gap_budget='.25*1e-8*||b||, relative to fixed S3 only; no guarantee for original Schur or floating-point roundoff.',
            API_limit='This diagnostic fixes rtol=1e-8; its setup gap budget is not generalized to other solve rtol values.',
            endpoint='Adaptive and fixed1/2/3 stop recursive residual<=.5e-8||b||, then verify original<=1e-8. Original exact416 retains original1e-8 stopping.',
            verification='Suspected/final only; no candidate residual-replacement fallback.',
            order='Rotate five methods by case_index; one repeat.',
            budget_bytes=odd.previous.LIMIT, reserve_bytes=odd.previous.RESERVE,
            cpu_cap=80, per_attempt_cpu_cap=6., command_wall_cap=120,
            scope='One paid native adaptive pilot. No generic novelty claim.'), tiny_check=None, audits=[], runs=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
                      elapsed_cpu_seconds=process_time()-cpu_started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    save()
    with threadpool_limits(limits=1):
        record['tiny_check'] = tiny_check(helper)
        save()
        path = ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
        with np.load(path) as data:
            X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
        tracemalloc.start()
        try:
            answer, row = run(X, b, bw, 1e-5, 'adaptive', helper, cpu_cap=6.)
            current, peak = tracemalloc.get_traced_memory()
            audit = dict(dataset='smooth2048', regularization=1e-5,
                         traced_peak_bytes=peak, trace_within_1MiB=peak <= odd.previous.LIMIT,
                         metrics=row)
            record['audits'].append(audit)
            del answer
        finally:
            tracemalloc.stop()
        save()
        if not audit['trace_within_1MiB'] or row['status'] == 'memory_failure':
            record['status'] = 'allocation_gate_failed'
            save()
            return
        case_index = 0
        for name in ['smooth', 'mixed']:
            path = ROOT/'data/prepared'/f'synthetic-n2560-{name}-s0-lambda0.001.npz'
            with np.load(path) as data:
                X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
            X.setflags(write=False)
            b.setflags(write=False)
            for lam in [.001, 1e-5]:
                shift = case_index % len(methods)
                order = methods[shift:]+methods[:shift]
                for position, method in enumerate(order):
                    if process_time()-cpu_started >= 72.:
                        row = dict(status='global_cpu_budget_not_started', converged=False)
                    else:
                        assert not tracemalloc.is_tracing()
                        answer, row = run(X, b, bw, lam, method, helper, deadline=cpu_started+78.)
                        del answer
                    row.update(dataset=name+'2048', regularization=lam, method=method,
                               order_position=position, data_sha256=sha256(path.read_bytes()).hexdigest())
                    record['runs'].append(row)
                    save()
                    print(json.dumps({key:row.get(key) for key in ['dataset', 'regularization', 'method', 'status', 'iterations', 'total_seconds', 'true_relative_residual', 'selected_order_counts']}), flush=True)
                case_index += 1
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
