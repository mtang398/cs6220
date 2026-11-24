"""Isolated matrix-free timing of the established width-two block-CG control.

Imports the frozen offline recurrence unchanged. Every timed solve constructs
its own fixed factor; no dense kernel, cached factor, or previous solution is
available in a timed worker. Two predeclared randomized-order repeats.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import multiprocessing as mp
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
import traceback

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel
from nystrom_project.preconditioners import LowRankPreconditioner
from check_block_pcg_width2 import block_pcg, CountedInverse, CountedOperator, validations
from check_projected_balancing import pcg
from check_power_depth_headroom import refine


def memory_bounds(n, rank, block):
    """Conservative float64 ndarray budgets, including inverse/QR scratch.

    Persistent scalar arrays: b,x,r,z,p,ap,last true_r. Block arrays:
    B,X,R,P,AP, each width2. Small Gram, QR, norm and coefficient arrays
    receive an explicit allowance. Preconditioner-application scratch has at
    most four width-sized n-arrays plus six k-by-width and three k-arrays.
    Block bound also covers live Z and QR input/output. These are source
    allocation bounds, not process RSS or exact LAPACK internal work sizes.
    Kernel tile workspace is common and is disclosed separately.
    """
    width = 2 if block else 1
    factor = 8 * rank * (n+1)
    persistent = 8 * n * (10 if block else 7) + (1024 if block else 128)
    scratch_small = 8*(6*rank*width + 3*rank + 16*width**2)
    live = 8*n*(18 if block else 11) + scratch_small
    return dict(factor_bytes=factor, solver_persistent_array_bound_bytes=persistent,
        solver_live_array_bound_bytes=live,
        factor_plus_persistent_array_bound_bytes=factor+persistent,
        factor_plus_live_array_bound_bytes=factor+live)


def matched_ranks(n):
    target = memory_bounds(n, 32, True)
    return {label: max(k for k in range(1, min(n, 64)+1)
        if memory_bounds(n, k, False)[field] <= target[field])
        for label, field in [('retained', 'factor_plus_persistent_array_bound_bytes'),
                             ('live', 'factor_plus_live_array_bound_bytes')]}


class NativeOperator:
    def __init__(self, kernel, regularization):
        self.kernel, self.regularization = kernel, regularization
        self.calls = self.rhs = 0
        self.widths = {}
    def __matmul__(self, values):
        width = 1 if values.ndim == 1 else values.shape[1]
        self.calls += 1
        self.rhs += width
        self.widths[str(width)] = self.widths.get(str(width), 0)+1
        result = self.kernel.matvec(values) if values.ndim == 1 else self.kernel.matmat(values)
        result += self.regularization*values
        return result


class Identity:
    storage_bytes = rank = 0
    def apply(self, value):
        return value.copy()


def native_refine(kernel, basis, passes, lam):
    Q = basis.copy()
    for step in range(passes):
        Y = kernel.matmat(Q)
        if step+1 < passes:
            Q, _ = linalg.qr(Y, mode='economic', check_finite=False)
    return LowRankPreconditioner.from_sketch(Y, Q.T@Y, lam)


def initialize(kernel, rank, depth, lam):
    omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(kernel.n, rank))
    current = native_refine(kernel, omega, 2, lam)
    del omega
    if depth >= 4:
        current = native_refine(kernel, current.basis, 2, lam)
    if depth >= 8:
        current = native_refine(kernel, current.basis, 4, lam)
    return current


def native_validations():
    checked = validations()
    rng = np.random.default_rng(614)
    X = rng.normal(size=(28, 3))
    kernel = GaussianKernel(X, .8, block_size=7)
    K = kernel.dense()
    b = rng.normal(size=28)
    g = rng.normal(size=28)
    lam = .07
    omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(28, 6))
    dense = refine(K, omega, 2, lam)
    native = initialize(kernel, 6, 2, lam)
    inverse_error = np.linalg.norm(native.apply(b)-dense.apply(b))/np.linalg.norm(dense.apply(b))
    inverse = CountedInverse(native)
    block_inverse = inverse.apply(np.column_stack((b, g)))
    column_inverse = np.column_stack((native.apply(b), native.apply(g)))
    block_inverse_error = np.linalg.norm(block_inverse-column_inverse)/np.linalg.norm(column_inverse)
    native_run, x, _ = block_pcg(NativeOperator(kernel, lam), np.column_stack((b, g)), inverse, rtol=1e-11)
    exact = linalg.solve(K+lam*np.eye(28), b, assume_a='pos')
    error = np.linalg.norm(x-exact)/np.linalg.norm(exact)
    assert inverse_error < 1e-11 and block_inverse_error < 1e-11
    assert native_run['converged'] and error < 1e-9
    checked['native_kernel_vs_dense_factor_inverse_error'] = float(inverse_error)
    checked['native_block_inverse_vs_columnwise_error'] = float(block_inverse_error)
    checked['native_block_solution_vs_dense_relative_error'] = float(error)
    return checked


def forbid_dense():
    raise AssertionError('dense kernel forbidden in timing worker')


def worker(specification, connection, cpu_remaining):
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(math.ceil(cpu_remaining))), max(2, int(math.ceil(cpu_remaining))+1)))
        with threadpool_limits(limits=1):
            data = np.load(ROOT/f"data/prepared/{specification['dataset']}-s0-lambda0.001.npz")
            b = data['y_train']
            X = data['X_train']
            n = len(b)
            lam = specification['regularization']
            block = specification['method'] == 'block_pcg_width2'
            rank, depth = specification['rank'], specification['power_passes']
            # Data loading is shared input preparation and outside solve timing.
            started, cpu_started = perf_counter(), process_time()
            kernel = GaussianKernel(X, float(data['bandwidth']))
            kernel.dense = forbid_dense
            model = Identity() if rank == 0 else initialize(kernel, rank, depth, lam)
            setup_seconds = perf_counter()-started
            setup_stats = dict(kernel.stats)
            inverse = CountedInverse(model)
            A = NativeOperator(kernel, lam)
            if block:
                g = np.random.default_rng(np.random.SeedSequence([0, 6220, 71])).normal(size=n)
                B = np.column_stack((b, g))
                # Release the separate g; only the RHS block is retained.
                del g
                run, _, _ = block_pcg(A, B, inverse)
            else:
                run, _ = pcg(A, b, inverse, maxiter=2500)
            elapsed, cpu_elapsed = perf_counter()-started, process_time()-cpu_started
            stats = dict(kernel.stats)
            result = dict(specification, **run, n=n, total_seconds=elapsed,
                setup_seconds=setup_seconds, solve_seconds=elapsed-setup_seconds,
                timed_cpu_seconds=cpu_elapsed, worker_cpu_seconds=process_time(),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                kernel_stats=stats, setup_kernel_stats=setup_stats,
                solve_kernel_passes=A.calls, solve_scalar_rhs=A.rhs,
                solve_width_histogram=A.widths,
                total_kernel_passes=int(stats['matvecs']+stats.get('matmats', 0)),
                total_scalar_rhs=int(stats['matvecs']+stats.get('matmat_rhs', 0)),
                inverse_calls=inverse.calls, inverse_rhs=inverse.rhs,
                actual_factor_bytes=model.storage_bytes,
                memory_bounds=memory_bounds(n, rank, block),
                input_training_X_bytes=X.nbytes, kernel_X_copy_bytes=kernel.X.nbytes,
                shared_original_b_bytes=b.nbytes,
                kernel_tile_workspace_bound_bytes=min(n, kernel.block_size)*n*8,
                setup_factorization_array_bound_bytes=8*(10*n*rank+12*rank*rank+8*rank),
                factor_sha256=None if rank == 0 else sha256(model.basis.tobytes()+model.eigenvalues.tobytes()).hexdigest())
            connection.send(dict(ok=True, result=result))
    except BaseException:
        connection.send(dict(ok=False, specification=specification, traceback=traceback.format_exc(),
            worker_cpu_seconds=process_time()))
    finally:
        connection.close()


def write_memory_correction():
    source = ROOT/'results/block-pcg-width2-diagnostic.json'
    destination = ROOT/'results/block-pcg-width2-memory-accounting.json'
    if destination.exists():
        raise FileExistsError(destination)
    original = json.loads(source.read_text())
    correction = dict(scope='Accounting correction only; original numerical script/results unchanged',
        original_result_sha256=sha256(source.read_bytes()).hexdigest(),
        reason='Original eight-vector scalar workspace omitted inverse-application scratch; original block small-array bound omitted rank-by-width inverse scratch. Use the conservative corrected bounds here.',
        conventions='Float64 ndarray source bounds, including RHS and inverse/QR work; process/library internals and common kernel tile are separate. Matched ranks are largest fitting these bounds.',
        cases=[])
    for case in original['cases']:
        n = case['n']
        correction['cases'].append(dict(dataset=case['dataset'], regularization=case['regularization'], n=n,
            scalar32=memory_bounds(n, 32, False), block32=memory_bounds(n, 32, True),
            matched_scalar_ranks=matched_ranks(n)))
    destination.write_text(json.dumps(correction, indent=2)+'\n')


def main():
    output = ROOT/'results/block-pcg-native-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    write_memory_correction()
    started, cpu_started = perf_counter(), process_time()
    cpu_budget = 110.
    child_cpu = 0.
    record = dict(status='running',
        scope='Established block CG native cost control; fixed width2 and fixed power factors, no new algorithm or research success claim',
        protocol=dict(datasets=['synthetic-n640-smooth', 'concrete-observed', 'synthetic-n640-mixed'],
            regularizations=[.001, 1e-5], power_passes=[2,4,8], block_rank=32,
            repeats=2, method_order_seed=[0,6220,942], process_isolation='fresh forked child per timed solve',
            rtol=1e-8, maxiter=2500, blas_threads=1, cpu_cap_seconds=110,
            timing='kernel construction, random sketch, power setup and solve included; reading frozen dataset excluded',
            paired_factor='same seed and rank construction per scalar/block method; native arithmetic may differ from dense GEMM rounding',
            memory_matching='scalar same rank32, largest rank fitting corrected persistent-array budget, and largest rank fitting corrected live-array budget',
            kernel_dense_allowed=False,
            setup_memory='separate conservative factorization ndarray bound; no hidden dense K or second solve factor',
            incomplete_policy='save every completed isolated run; do not infer paired timing for incomplete repeats'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),
            ROOT/'scripts/check_block_pcg_width2.py', ROOT/'scripts/check_projected_balancing.py',
            ROOT/'src/nystrom_project/numerics.py', ROOT/'src/nystrom_project/preconditioners.py']}, runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
            total_accounted_cpu_seconds=process_time()-cpu_started+child_cpu)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['validations'] = native_validations()
    generator = np.random.default_rng(np.random.SeedSequence([0,6220,942]))
    tasks = []
    for name in record['protocol']['datasets']:
        data = np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
        ranks = matched_ranks(len(data['y_train']))
        for lam in record['protocol']['regularizations']:
            variants = [dict(method='ordinary_pcg', rank=0, power_passes=0, allocation='unpreconditioned')]
            for depth in [2,4,8]:
                variants.append(dict(method='block_pcg_width2', rank=32, power_passes=depth, allocation='block32'))
                for allocation, rank in [('same_rank',32), ('retained_matched',ranks['retained']), ('live_matched',ranks['live'])]:
                    variants.append(dict(method='ordinary_pcg', rank=rank, power_passes=depth, allocation=allocation))
            for repeat in range(2):
                for index in generator.permutation(len(variants)):
                    tasks.append(dict(variants[index], dataset=name, regularization=lam, repeat=repeat))
    record['predeclared_execution_order'] = tasks
    save()
    context = mp.get_context('fork')
    for index, task in enumerate(tasks):
        remaining = cpu_budget-(process_time()-cpu_started+child_cpu)
        if remaining < 3:
            record['status']='cpu_budget'; break
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=worker, args=(task, send, remaining))
        process.start()
        send.close()
        message = None
        while process.is_alive():
            if receive.poll(.2):
                message = receive.recv()
                break
        process.join()
        if message is None and receive.poll():
            try:
                message = receive.recv()
            except EOFError:
                pass
        receive.close()
        if message is None:
            record['runs'].append(dict(task, status='worker_stopped', exitcode=process.exitcode))
            record['status']='cpu_budget_or_worker_failure'; break
        child_cpu += message.get('result',message).get('worker_cpu_seconds',0.)
        if not message['ok']:
            record['runs'].append(message)
            record['status']='worker_error'; save(); break
        run = message['result']
        record['runs'].append(run)
        save()
        print(json.dumps(dict(index=index+1, of=len(tasks), dataset=task['dataset'], lam=task['regularization'],
            method=task['method'], rank=task['rank'], depth=task['power_passes'], repeat=task['repeat'],
            seconds=run['total_seconds'], iterations=run['iterations'], converged=run['converged'],
            accounted_cpu=record['total_accounted_cpu_seconds'])),flush=True)
    else:
        record['status']='complete'
    record['completed_runs']=sum('total_seconds' in run for run in record['runs'])
    record['planned_runs']=len(tasks)
    save()


if __name__ == '__main__':
    main()
