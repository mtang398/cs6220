#!/usr/bin/env python3
"""Odd fixed core-refinement actions; preserved original and wrapper controls.

Static representation diagnostic, not an adaptive mechanism. The original
measured sources are imported unchanged. Only verification cadence is changed
in the original baseline's solve function; its recurrence remains intact.
"""
from collections import Counter
from hashlib import sha256
import inspect
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import check_compressed_core_schur as previous
from check_packed_float_allocation import RawPointers
from check_packed_schur_feasibility import PackedSchur, base, ROOT


class Schur(previous.Schur):
    def __init__(self, *args, **kwargs):
        self.boundary = False
        self.phase_calls = Counter()
        self.phase_corrections = Counter()
        super().__init__(*args, **kwargs)

    def core_solve(self, rhs, *, extra=()):
        if self.float is None:
            return super().core_solve(rhs, extra=extra)
        corrections = 4 if self.boundary else int(self.method[-1])
        phase = 'boundary' if self.boundary else 'action'
        self.core_calls += 1
        self.phase_calls[phase] += 1
        self.phase_corrections[phase] += corrections
        with self.budget.hold(rhs, *extra):
            answer = self.float.solve(rhs)
            with self.budget.hold(answer):
                for _ in range(corrections):
                    residual = self.float.original_product(answer)
                    self.core_checks += 1
                    residual *= -1.
                    residual += rhs
                    with self.budget.hold(residual):
                        delta = self.float.solve(residual)
                        self.budget.observe('core_refinement_update', delta)
                        answer += delta
                        del delta
                    del residual
        self.corrections += corrections
        self.stop_reasons['fixed_count'] += 1
        self.correction_counts[corrections] += 1
        return answer

    def rhs(self, b):
        self.boundary = True
        try:
            return super().rhs(b)
        finally:
            self.boundary = False

    def reconstruct(self, b, tail):
        self.boundary = True
        try:
            return super().reconstruct(b, tail)
        finally:
            self.boundary = False

    def report(self):
        return dict(super().report(), core_calls_by_phase=dict(self.phase_calls),
                    core_corrections_by_phase=dict(self.phase_corrections),
                    boundary_corrections=4 if self.float is not None else None)


# Keep the earlier wrapper recurrence exactly, exposing only method names and
# verification cadence. Record the generated function text hashes below.
WRAPPER_SOURCE = inspect.getsource(previous.solve)
assert WRAPPER_SOURCE.count("deadline=float('inf')):") == 1
WRAPPER_SOURCE = WRAPPER_SOURCE.replace(
    "deadline=float('inf')):", "deadline=float('inf'),verify_every=0):")
assert WRAPPER_SOURCE.count("['exact416','fixed2','fixed4','tol1e-10']") == 1
WRAPPER_SOURCE = WRAPPER_SOURCE.replace(
    "['exact416','fixed2','fixed4','tol1e-10']", "['exact416','fixed1','fixed2','fixed3']")
assert WRAPPER_SOURCE.count('if iteration%4==0 or suspected:') == 1
WRAPPER_SOURCE = WRAPPER_SOURCE.replace(
    'if iteration%4==0 or suspected:',
    'if (verify_every>0 and iteration%verify_every==0) or suspected:')
wrapper_namespace = dict(previous.__dict__, Schur=Schur)
exec(compile(WRAPPER_SOURCE, '<odd-refinement-wrapper>', 'exec'), wrapper_namespace)
solve = wrapper_namespace['solve']


class UpperPacked(PackedSchur):
    last_factor_trace = None

    def __init__(self, kernel, m, lam, *, indices=None):
        selected = kernel.farthest(m) if indices is None else indices
        super().__init__(kernel, m, lam, indices=selected, lower=False)
        UpperPacked.last_factor_trace = self.factor_trace


ORIGINAL_SOURCE = inspect.getsource(base.solve)
assert ORIGINAL_SOURCE.count('if iteration%20==0 or suspected or iteration==maxiter:') == 1
ORIGINAL_SOURCE = ORIGINAL_SOURCE.replace(
    'if iteration%20==0 or suspected or iteration==maxiter:',
    'if suspected or iteration==maxiter:')
original_namespace = dict(base.__dict__, InplaceSchur=UpperPacked)
exec(compile(ORIGINAL_SOURCE, '<original-packed-suspected-only>', 'exec'), original_namespace)
original_exact_solve = original_namespace['solve']


def run(X, b, bw, lam, method, helper, *, m=None, cpu_cap=6., deadline=float('inf')):
    if method == 'original_exact416':
        answer, row = original_exact_solve(
            X, b, bw, 416 if m is None else m, lam,
            maxiter=2000, cpu_cap=cpu_cap, deadline=deadline)
        row.update(method=method, m=416 if m is None else m,
                   factor_trace=UpperPacked.last_factor_trace,
                   baseline_function='Original packed base.solve; periodic checks removed only')
        return answer, row
    return solve(X, b, bw, lam, method, helper, m=m, cpu_cap=cpu_cap,
                 deadline=deadline, verify_every=0)


def tiny_check(helper):
    rng = np.random.default_rng(622107)
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
    checks = []
    for method in ['fixed1', 'fixed3']:
        op = Schur(X, b, bw, m, lam, method, helper)
        C = A[np.ix_(op.S, op.S)]
        B = A[np.ix_(op.R, op.S)]
        D = A[np.ix_(op.R, op.R)]
        exact_S = D - B @ linalg.solve(C, B.T, assume_a='pos')
        v = rng.normal(size=len(op.R))
        action = op.apply(v)
        action_error = float(np.linalg.norm(action - exact_S @ v) / np.linalg.norm(exact_S @ v))
        answer, row = run(X, b, bw, lam, method, helper, m=m, cpu_cap=2.)
        error = float(np.linalg.norm(answer-reference) / np.linalg.norm(reference))
        assert action_error < 1e-10 and row['converged'] and error < 1e-7
        # Independent matrix identity with deliberately imperfect SPD H0,
        # rather than relying only on the almost-exact tiny FP32 factor.
        eig, V = linalg.eigh(C)
        inverse_root = (V / np.sqrt(eig)) @ V.T
        root = (V * np.sqrt(eig)) @ V.T
        H0 = inverse_root @ np.diag(np.linspace(.5, 1.7, m)) @ inverse_root
        H = H0.copy()
        j = int(method[-1])
        for _ in range(j):
            H += H0 @ (np.eye(m)-C@H)
        identity_rhs = inverse_root @ np.linalg.matrix_power(np.eye(m)-root@H0@root, j+1) @ inverse_root
        difference = linalg.solve(C, np.eye(m), assume_a='pos')-H
        identity_error = float(np.linalg.norm(difference-identity_rhs)/np.linalg.norm(difference))
        minimum_difference = float(linalg.eigvalsh((difference+difference.T)*.5)[0])
        minimum_S = float(linalg.eigvalsh(D-B@H@B.T)[0])
        assert identity_error < 1e-10 and minimum_difference > -1e-10 and minimum_S >= lam-1e-10
        checks.append(dict(method=method, action_relative_error=action_error,
                           full_solution_relative_error=error,
                           true_relative_residual=row['true_relative_residual'],
                           odd_identity_relative_error=identity_error,
                           minimum_inverse_difference_eigenvalue=minimum_difference,
                           minimum_approximate_Schur_eigenvalue=minimum_S))
    for method in ['original_exact416', 'exact416']:
        answer, row = run(X, b, bw, lam, method, helper, m=m, cpu_cap=2.)
        error = float(np.linalg.norm(answer-reference)/np.linalg.norm(reference))
        assert row['converged'] and error < 1e-7
        checks.append(dict(method=method, full_solution_relative_error=error,
                           true_relative_residual=row['true_relative_residual']))
    return checks


def allocation_audit(X, b, bw, lam, method, helper):
    tracemalloc.start()
    try:
        answer, row = run(X, b, bw, lam, method, helper, cpu_cap=6.)
        current, peak = tracemalloc.get_traced_memory()
        if row.get('factor_trace'):
            peak = max(peak, row['factor_trace']['earlier_peak'])
        return dict(method=method, traced_peak_bytes=peak,
                    trace_within_1MiB=peak <= previous.LIMIT, metrics=row)
    finally:
        tracemalloc.stop()


def main():
    output = ROOT/'results/compressed-core-schur-odd-pilot.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    helper = RawPointers(ROOT/'results/packed_float_factor.so')
    old_result = ROOT/'results/compressed-core-schur-pilot.json'
    old = json.loads(old_result.read_text())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in old['sources'].items())
    paths = [Path(__file__), old_result, *[ROOT/p for p in old['sources']]]
    methods = ['original_exact416', 'exact416', 'fixed1', 'fixed2', 'fixed3']
    record = dict(status='running', sources={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths},
        generated_function_hashes=dict(wrapper=sha256(WRAPPER_SOURCE.encode()).hexdigest(),
                                       original_baseline=sha256(ORIGINAL_SOURCE.encode()).hexdigest()),
        protocol=dict(datasets=['smooth2048', 'mixed2048'], regularizations=[.001, 1e-5],
            seed=0, repeats=2, methods=methods, candidate_core=600, exact_core=416,
            boundary_corrections=4, action_corrections=[1, 2, 3], verification='Suspected/final only',
            original_baseline='Original packed base.solve recurrence and residual replacement; only periodic verification removed.',
            wrapper='Earlier wrapper recurrence; no fallback/replacement after false recursive convergence.',
            order='Rotate methods by (case_index+repeat) modulo5; all cases for each repeat.',
            fixed2_inclusion='Parent identified the old fixed2 endpoint error as mainly landmark-boundary error before this pilot ran; include fixed2 action with the same fixed4 boundary.',
            tolerance=1e-8, maxiter=2000, per_call_cpu_cap=6., total_cpu_cap=80.,
            budget_bytes=previous.LIMIT, reserve_bytes=previous.RESERVE,
            accounting='FPS/core setup, all boundary/action core products, reconstruction and final original residual charged. Common X,b excluded. No dense native reference.',
            proof='For odd j, C^-1-H_j=C^-1/2(I-C^1/2 H0 C^1/2)^(j+1)C^-1/2 is PSD; hence approximate Schur >= exact Schur in exact arithmetic. H_j itself need not be positive without a spectral condition.',
            scope='Fixed static representation control, no adaptive contribution claim.'),
        tiny_check=[], audits=[], runs=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
                      elapsed_cpu_seconds=process_time()-cpu_started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    with threadpool_limits(limits=1):
        record['tiny_check'] = tiny_check(helper)
        save()
        path = ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
        with np.load(path) as data:
            X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
        for method in ['fixed3', 'original_exact416']:
            audit = allocation_audit(X, b, bw, .001, method, helper)
            record['audits'].append(audit)
            save()
            if not audit['trace_within_1MiB'] or audit['metrics']['status'] == 'memory_failure':
                record['status'] = 'allocation_gate_failed'
                save()
                return
        for repeat in range(2):
            case_index = 0
            for name in ['smooth', 'mixed']:
                path = ROOT/'data/prepared'/f'synthetic-n2560-{name}-s0-lambda0.001.npz'
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                X.setflags(write=False)
                b.setflags(write=False)
                for lam in [.001, 1e-5]:
                    shift = (case_index+repeat) % len(methods)
                    order = methods[shift:]+methods[:shift]
                    for order_position, method in enumerate(order):
                        metadata = dict(dataset=name+'2048', regularization=lam, repeat=repeat,
                                        method=method, order_position=order_position,
                                        data_sha256=sha256(path.read_bytes()).hexdigest())
                        if process_time()-cpu_started >= 72.:
                            row = dict(status='global_cpu_budget_not_started', converged=False)
                        else:
                            assert not tracemalloc.is_tracing()
                            try:
                                answer, row = run(X, b, bw, lam, method, helper,
                                                  deadline=cpu_started+78.)
                                del answer
                            except (MemoryError, np.linalg.LinAlgError) as error:
                                row = dict(status='resource_or_numerical_failure', error=str(error), converged=False)
                        row.update(metadata)
                        record['runs'].append(row)
                        save()
                        print(json.dumps({k:row.get(k) for k in ['dataset', 'regularization', 'repeat', 'method', 'status', 'iterations', 'total_seconds', 'true_relative_residual']}), flush=True)
                    case_index += 1
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
