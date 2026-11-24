#!/usr/bin/env python3
"""Dense diagnostic: one additional K application to learned and static bases.

Only QR-selected branches retain actual kernel columns. Dense products and
simultaneous control states are diagnostic resources, not production timings.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner
from nystrom_project.learning import compress_probe, select_columns_from_factor

OUTPUT = ROOT / 'results/learned-power-refinement-diagnostic.json'
PREFIX, RANK, RTOL = 10, 32, 1e-8
START_CPU, START_WALL = process_time(), perf_counter()
SOURCES = [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py',
           ROOT/'src/nystrom_project/preconditioners.py', ROOT/'src/nystrom_project/learning.py']
record = dict(status='running',
    scope='One-seed dense offline action diagnostic; no production speed or peak-memory advantage claim',
    protocol=dict(datasets=['synthetic-n640-smooth', 'concrete-observed'],
        regularizations=[.001, 1e-5], rank=RANK, prefix=PREFIX, rtol=RTOL,
        max_iterations=4000, blas_threads=1, cpu_limit_seconds=80,
        prefix_solver='Ordinary fixed-RP PCG; shadow compression never changes its active preconditioner',
        learned_basis='Stream first ten p,Kp into learning.compress_probe at fixed rank',
        polishing='Y=K Q; W=Q.T Y; generalized Nystrom, one k-RHS block product',
        gaussian='Y1=K Omega; Q=qr(Y1); Y2=K Q; two k-RHS block products',
        actual_column_restoration='Production QR pivoting on polished factor; query every selected column',
        pairing='Every delayed branch starts at the identical x10 and directly recomputed residual',
        stopping='Original b-Ax checked every20 iterations, on suspected convergence, and at the limit',
        representation_comparison='Compare factor branches with factors and actual-column branches with actual columns',
        dense_work='K@Q is a dense offline surrogate; count block calls/RHS separately from scalar solve products'),
    source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in SOURCES},
    cases=[])


def guard():
    if process_time()-START_CPU > 75:
        raise TimeoutError('75 CPU-second graceful guard reached')


def save():
    record.update(elapsed_cpu_seconds=process_time()-START_CPU,
        elapsed_wall_seconds=perf_counter()-START_WALL,
        offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def pcg(A, b, state, *, x0=None, r0=None, shadow_prefix=False):
    """Count every A product, including true-residual checks and initialization."""
    norm_b = float(np.linalg.norm(b))
    assert norm_b > 0
    x = np.zeros_like(b) if x0 is None else x0.copy()
    initial_products = int(x0 is not None and r0 is None)
    r = (b.copy() if x0 is None else b-A@x) if r0 is None else r0.copy()
    z = state.apply(r)
    p, rho = z.copy(), float(r@z)
    iteration_products = checks = replacements = 0
    saved = None
    shadow = LowRankPreconditioner.from_nystrom(state) if shadow_prefix else None
    events, last_verified = [], -1
    true_relative, stopped = float(np.linalg.norm(r)/norm_b), 'max_iterations'
    state_hash = sha256(state.basis.tobytes()+state.eigenvalues.tobytes()).hexdigest()
    for iteration in range(1, 4001):
        if iteration % 25 == 1:
            guard()
        ap = A@p
        iteration_products += 1
        curvature = float(p@ap)
        if min(rho, curvature) <= 0 or not np.isfinite(rho+curvature):
            stopped = 'nonpositive_inner_product'
            break
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        if shadow_prefix and iteration <= PREFIX:
            shadow, event = compress_probe(shadow, p, ap-state.regularization*p)
            event['iteration'] = iteration
            events.append(event)
            assert shadow.rank == state.rank
        suspected = np.linalg.norm(r) <= RTOL*norm_b
        at_prefix = shadow_prefix and iteration == PREFIX
        if iteration % 20 == 0 or suspected or at_prefix or iteration == 4000:
            true_r = b-A@x
            checks += 1
            last_verified = iteration
            true_relative = float(np.linalg.norm(true_r)/norm_b)
            if at_prefix:
                saved = dict(x=x.copy(), r=true_r.copy(), shadow=shadow,
                    scalar_iteration_products=iteration_products,
                    true_residual_verification_products=checks,
                    operator_products=initial_products+iteration_products+checks,
                    shadow_events=events.copy(), true_relative_residual=true_relative,
                    x_sha256=sha256(x.tobytes()).hexdigest(),
                    residual_sha256=sha256(true_r.tobytes()).hexdigest())
            if true_relative <= RTOL:
                stopped = 'converged'
                break
            if suspected:
                r = true_r.copy()
                z = state.apply(r)
                p, rho = z.copy(), float(r@z)
                replacements += 1
                continue
        z = state.apply(r)
        new_rho = float(r@z)
        p = z+(new_rho/rho)*p
        rho = new_rho
    if last_verified != iteration:
        true_relative = float(np.linalg.norm(b-A@x)/norm_b)
        checks += 1
    assert state_hash == sha256(state.basis.tobytes()+state.eigenvalues.tobytes()).hexdigest()
    return dict(iterations=iteration, converged=bool(true_relative <= RTOL),
        true_relative_residual=true_relative, stop_reason=stopped,
        scalar_iteration_products=iteration_products,
        true_residual_verification_products=checks,
        initial_residual_products=initial_products,
        operator_products=initial_products+iteration_products+checks,
        residual_replacements=replacements), saved


def polish(K, basis, lam):
    Y = K@basis
    core = basis.T@Y
    state = LowRankPreconditioner.from_sketch(Y, core, lam)
    assert state.rank == RANK
    # Independent core Cholesky reconstructs the generalized Nystrom matrix.
    factor = linalg.cho_factor((core+core.T)*.5, lower=True, check_finite=False)
    target = Y@linalg.cho_solve(factor, Y.T, check_finite=False)
    reconstructed = (state.basis*state.eigenvalues)@state.basis.T
    error = float(np.linalg.norm(target-reconstructed)/np.linalg.norm(target))
    assert error < 1e-9
    return state, dict(core_cholesky_spd=True, independent_core_reconstruction_error=error,
        kernel_matmat_calls=1, kernel_matmat_rhs=int(basis.shape[1]),
        input_basis_bytes=int(basis.nbytes), kernel_image_bytes=int(Y.nbytes),
        small_core_bytes=int(core.nbytes), core_min_eigenvalue=float(linalg.eigvalsh(core)[0]))


def audit_state(K, state, *, actual_columns):
    n = len(K)
    gram_error = float(np.linalg.norm(state.basis.T@state.basis-np.eye(state.rank)))
    assert gram_error < 1e-10 and state.rank == RANK and np.all(state.eigenvalues > 0)
    if actual_columns:
        indices = state.indices
        assert len(indices) == RANK and len(np.unique(indices)) == RANK
        column_error = float(np.linalg.norm(state.columns-K[:, indices])/np.linalg.norm(state.columns))
        core_error = float(np.linalg.norm(state.core-K[np.ix_(indices, indices)])/np.linalg.norm(state.core))
        assert column_error < 1e-14 and core_error < 1e-14
        core_factor = linalg.cho_factor(K[np.ix_(indices, indices)], lower=True, check_finite=False)
        Kh = K[:, indices]@linalg.cho_solve(core_factor, K[indices, :], check_finite=False)
    else:
        Kh = (state.basis*state.eigenvalues)@state.basis.T
        column_error = core_error = None
    M = (Kh+Kh.T)*.5+state.regularization*np.eye(n)
    factor = linalg.cho_factor(M, lower=True, check_finite=False)
    rhs = np.random.default_rng(622073).standard_normal((n, 3))
    direct = linalg.cho_solve(factor, rhs, check_finite=False)
    applied = np.column_stack([state.apply(v) for v in rhs.T])
    application_error = float(np.linalg.norm(applied-direct)/np.linalg.norm(direct))
    assert application_error < 2e-7
    return dict(rank=int(state.rank), representation='actual_columns' if actual_columns else 'generalized_factor',
        retained_state_bytes=int(state.storage_bytes), basis_orthogonality_error=gram_error,
        cholesky_spd=True, independent_dense_inverse_application_error=application_error,
        columns_vs_actual_kernel_error=column_error, core_vs_actual_kernel_error=core_error,
        selected_indices=state.indices.tolist() if actual_columns else None,
        trace_approximation=float(state.eigenvalues.sum()))


def select(kernel, K, factor):
    before = int(kernel.stats['column_evaluations'])
    columns = select_columns_from_factor(kernel, factor)
    queries = int(kernel.stats['column_evaluations'])-before
    assert queries == RANK
    return columns, queries


def run():
    for name in record['protocol']['datasets']:
        path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
        with np.load(path) as data:
            X, b, bandwidth = data['X_train'], data['y_train'], float(data['bandwidth'])
            config = json.loads(str(data['config']))
        for lam in record['protocol']['regularizations']:
            guard()
            kernel = GaussianKernel(X, bandwidth)
            K = kernel.dense()
            A = K+lam*np.eye(len(b))
            case = dict(dataset=name, regularization=lam, rank=RANK, n=len(b), status='running',
                prepared_input=str(path.relative_to(ROOT)), prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                preparation_regularization=config['regularization'], actual_operator_regularization=lam,
                stored_reference_solution_ignored=True, states={}, runs=[])
            record['cases'].append(case)
            old = initialize_rpcholesky(kernel, RANK, lam,
                np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
            initial_queries = int(kernel.stats['column_evaluations'])
            assert initial_queries == RANK
            baseline, saved = pcg(A, b, old, shadow_prefix=True)
            assert baseline['converged'] and saved is not None
            shadow = saved['shadow']
            case['prefix'] = {key:value for key,value in saved.items() if key not in ['x','r','shadow']}
            case['prefix'].update(iterations=PREFIX, active_preconditioner_unchanged=True,
                streamed_pairs=PREFIX, archived_direction_image_bytes=0,
                initial_rp_state_bytes=int(old.storage_bytes), shadow_state_bytes=int(shadow.storage_bytes),
                old_plus_shadow_state_bytes=int(old.storage_bytes+shadow.storage_bytes),
                transient_probe_pair_bytes=2*len(b)*8,
                largest_augmented_shadow_factor_bytes=len(b)*(RANK+1)*8,
                common_checkpoint_x_and_r_bytes=int(saved['x'].nbytes+saved['r'].nbytes))
            static, static_setup = polish(K, old.basis, lam)
            learned, learned_setup = polish(K, shadow.basis, lam)
            omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(len(b), RANK))
            Y1 = K@omega
            Q, _ = linalg.qr(Y1, mode='economic', check_finite=False)
            gaussian, gaussian_setup = polish(K, Q, lam)
            gaussian_setup.update(kernel_matmat_calls=2, kernel_matmat_rhs=2*RANK,
                first_gaussian_probe_bytes=int(omega.nbytes), first_gaussian_image_bytes=int(Y1.nbytes),
                gaussian_sha256=sha256(omega.tobytes()).hexdigest())
            static_columns, static_queries = select(kernel, K, static)
            learned_columns, learned_queries = select(kernel, K, learned)
            gaussian_columns, gaussian_queries = select(kernel, K, gaussian)
            states = dict(old=old, rp_power_factor=static, rp_power_columns=static_columns,
                learned_power_factor=learned, learned_power_columns=learned_columns,
                gaussian_power_factor=gaussian, gaussian_power_columns=gaussian_columns)
            for label,state in states.items():
                guard()
                case['states'][label] = audit_state(K, state,
                    actual_columns=label == 'old' or label.endswith('columns'))
            case['construction'] = dict(initial_rp_column_queries=initial_queries,
                static_polish=static_setup, learned_polish=learned_setup, gaussian_two_pass=gaussian_setup,
                qrcp_column_queries=dict(static=static_queries, learned=learned_queries, gaussian=gaussian_queries),
                offline_dense_operator_bytes=int(K.nbytes+A.nbytes),
                scope='Shared diagnostic caches and audit matrices are not peak-memory-matched production states; LAPACK workspace is additional')
            specs = [('old_uninterrupted','old',False), ('old_restart10','old',True),
                ('rp_power_factor_zero','rp_power_factor',False),
                ('rp_power_columns_zero','rp_power_columns',False),
                ('rp_power_factor_after10','rp_power_factor',True),
                ('rp_power_columns_after10','rp_power_columns',True),
                ('learned_power_factor_after10','learned_power_factor',True),
                ('learned_power_columns_after10','learned_power_columns',True),
                ('gaussian_power_factor_zero','gaussian_power_factor',False),
                ('gaussian_power_columns_zero','gaussian_power_columns',False)]
            for label, state_name, delayed in specs:
                guard()
                state = states[state_name]
                result = baseline.copy() if label == 'old_uninterrupted' else pcg(A,b,state,
                    x0=saved['x'] if delayed else None,r0=saved['r'] if delayed else None)[0]
                is_gaussian = state_name.startswith('gaussian')
                is_polished = state_name != 'old'
                result.update(method=label, state=state_name, delayed=delayed,
                    total_iterations=result['iterations']+(PREFIX if delayed else 0),
                    total_scalar_operator_products=result['operator_products']+(saved['operator_products'] if delayed else 0),
                    setup_kernel_matmat_calls=2 if is_gaussian else int(is_polished),
                    setup_kernel_matmat_rhs=2*RANK if is_gaussian else RANK*int(is_polished),
                    setup_actual_column_queries=(0 if is_gaussian else initial_queries)+
                        (RANK if state_name.endswith('columns') else 0),
                    shadow_compression_updates=sum(e['accepted'] for e in saved['shadow_events'])
                        if state_name.startswith('learned') else 0,
                    shadow_compression_attempts=PREFIX if state_name.startswith('learned') else 0,
                    final_retained_state_bytes=int(state.storage_bytes),
                    x10_sha256=saved['x_sha256'] if delayed else None,
                    verification_scope='State audits, offline capture, and shared construction are outside these conceptual solve/setup counts')
                case['runs'].append(result)
                save()
                print(json.dumps(dict(dataset=name,lam=lam,method=label,iterations=result['total_iterations'],
                    converged=result['converged'],relative_residual=result['true_relative_residual'])),flush=True)
            assert all(value['converged'] for value in case['runs'])
            case['status'] = 'complete'
            save()


if __name__ == '__main__':
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_CPU,(79,80))
    try:
        with threadpool_limits(limits=1):
            run()
        for path in SOURCES:
            assert sha256(path.read_bytes()).hexdigest() == record['source_sha256'][str(path.relative_to(ROOT))]
        record['status'] = 'complete'
    except TimeoutError as error:
        record.update(status='cpu_budget_exhausted',error=str(error))
    except Exception as error:
        record.update(status='failed',error=repr(error))
        save()
        raise
    save()
