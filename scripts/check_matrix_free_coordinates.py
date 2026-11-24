#!/usr/bin/env python3
"""Matrix-free cost diagnostic for projected greedy Schur block corrections.

The first coordinate projection uses cached (S P)[j,:].  One joint second
projection restores conjugacy.  All kernel queries and residual verification
are measured; dense references remain outside the solver.
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
sys.path.insert(0, str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from nystrom_project.revisions import revised_solve


def coordinate_solve(kernel, b, lam, rank=16, width=16, retained=16,
                     max_blocks=10000, max_seconds=30., verify_interval=50):
    started = perf_counter()
    n = len(b)
    base = initialize_rpcholesky(kernel, rank, lam,
        np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
    indices = base.indices.copy()
    remaining = np.setdiff1d(np.arange(n), indices, assume_unique=True)
    L = linalg.cholesky(base.columns[indices]+lam*np.eye(rank), lower=True)
    B = linalg.solve_triangular(L, base.columns[remaining].T, lower=True).T.copy()
    del base
    nr = len(remaining)
    diagonal = 1+lam-np.einsum('ij,ij->i', B, B)
    assert np.all(diagonal > 0)
    top_rhs = linalg.solve_triangular(L, b[indices], lower=True)
    r = b[remaining]-B@top_rhs
    x = np.zeros(nr)
    norm_b = np.linalg.norm(b)
    capacity = retained+width
    P = np.empty((nr, capacity), order='F')
    AP = np.empty_like(P)
    curvatures = np.empty(capacity)
    size = 0
    checks = queries = 0
    true_relative = np.linalg.norm(r)/norm_b
    status = 'max_blocks'
    last_verified = -1
    progress = []
    scoring_seconds = projection_seconds = update_seconds = 0.
    maximum_conjugacy = 0.

    def verify(block):
        nonlocal checks, last_verified, true_relative
        full_x = np.empty(n)
        full_x[remaining] = x
        full_x[indices] = linalg.solve_triangular(L.T, top_rhs-B.T@x, lower=False)
        full_r = b-kernel.matvec(full_x)-lam*full_x
        checks += 1
        last_verified = block
        true_relative = float(np.linalg.norm(full_r)/norm_b)
        progress.append(dict(block=block, column_queries=queries, true_relative_residual=true_relative,
            recursive_relative_residual=float(np.linalg.norm(r)/norm_b)))
        return full_x, full_r[remaining]

    setup_seconds = perf_counter()-started
    for block in range(1, max_blocks+1):
        if perf_counter()-started > max_seconds:
            status = 'wall_cap'; block -= 1; break
        old_size = size
        correction = np.zeros(nr)
        combined_image = np.zeros(nr)
        tau = diagonal.copy()
        if size:
            tau -= np.einsum('ij,j,ij->i', AP[:, :size], 1/curvatures[:size], AP[:, :size])
        selected = []
        for _ in range(width):
            t = perf_counter()
            numerator = r.copy()
            if size:
                numerator -= AP[:, :size]@((P[:, :size].T@r)/curvatures[:size])
            valid = tau > 256*np.finfo(float).eps*np.maximum(diagonal, np.abs(diagonal-tau))
            score = np.full(nr, -np.inf)
            score[valid] = numerator[valid]**2/tau[valid]
            score[selected] = -np.inf
            j = int(np.argmax(score))
            if not np.isfinite(score[j]) or score[j] <= 0:
                status = 'no_independent_coordinate'; break
            selected.append(j)
            scoring_seconds += perf_counter()-t
            ap = kernel.entries(remaining, remaining[j:j+1]).reshape(nr)-B@B[j]
            ap[j] += lam
            queries += 1
            t = perf_counter()
            p = np.zeros(nr); p[j] = 1.
            if size:
                coefficients = AP[j, :size]/curvatures[:size]
                p -= P[:, :size]@coefficients
                ap -= AP[:, :size]@coefficients
                coefficients = (P[:, :size].T@ap)/curvatures[:size]
                p -= P[:, :size]@coefficients
                ap -= AP[:, :size]@coefficients
            curvature = float(p@ap)
            if curvature <= 0:
                status = 'nonpositive_curvature'; break
            if size and block % 50 == 0:
                maximum_conjugacy = max(maximum_conjugacy,
                    float(np.max(np.abs(P[:, :size].T@ap)/np.sqrt(curvatures[:size]*curvature))))
            projection_seconds += perf_counter()-t
            t = perf_counter()
            alpha = float(r@p)/curvature
            x += alpha*p; r -= alpha*ap
            correction += alpha*p; combined_image += alpha*ap
            scale = np.sqrt(curvature)
            P[:, size] = p/scale; AP[:, size] = ap/scale
            curvatures[size] = P[:, size]@AP[:, size]
            tau -= AP[:, size]**2/curvatures[size]
            size += 1
            update_seconds += perf_counter()-t
        if status not in ('max_blocks',):
            break
        if retained:
            keep = min(old_size, retained-1)
            first = old_size-keep
            P[:, :keep] = P[:, first:old_size]
            AP[:, :keep] = AP[:, first:old_size]
            curvatures[:keep] = curvatures[first:old_size]
            scale = np.sqrt(float(correction@combined_image))
            P[:, keep] = correction/scale
            AP[:, keep] = combined_image/scale
            curvatures[keep] = P[:, keep]@AP[:, keep]
            size = keep+1
        else:
            size = 0
        suspected = np.linalg.norm(r) <= 1e-8*norm_b
        if block % verify_interval == 0 or suspected or block == max_blocks:
            _, true_r = verify(block)
            if true_relative <= 1e-8:
                status = 'converged'; break
            if suspected:
                r = true_r.copy(); size = 0
    if last_verified != block:
        full_x, _ = verify(block)
    else:
        full_x = np.empty(n); full_x[remaining] = x
        full_x[indices] = linalg.solve_triangular(L.T, top_rhs-B.T@x, lower=False)
    coarse_bytes = L.nbytes+B.nbytes+indices.nbytes+remaining.nbytes
    return full_x, dict(status=status, converged=true_relative <= 1e-8,
        true_relative_residual=true_relative, blocks=block, column_queries=queries,
        verification_matvecs=checks, setup_seconds=setup_seconds,
        total_seconds=perf_counter()-started, scoring_seconds=scoring_seconds,
        projection_seconds=projection_seconds, update_seconds=update_seconds,
        kernel_stats=dict(kernel.stats), width=width, retained_history=retained,
        rank=rank, maximum_checked_history_conjugacy=maximum_conjugacy,
        coarse_state_bytes=coarse_bytes,
        retained_coarse_history_bytes=coarse_bytes+2*nr*retained*8+retained*8,
        coarse_history_block_bytes=coarse_bytes+P.nbytes+AP.nbytes+curvatures.nbytes,
        workspace_note='Additional O(n) score, diagonal, iterate, residual and projection vectors; kernel matvec block buffer during verification.',
        progress=progress)


def main():
    output = ROOT/'results/matrix-free-coordinate-pilot.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (165, 170))
    started, cpu_started = perf_counter(), process_time()
    record = dict(status='running', purpose='Matrix-free setup-inclusive cost pilot; exploratory single repeat',
        protocol=dict(seed=0, regularization=.001, rank=16, block_width=16,
            retained_history=[0,16], blas_threads=1, true_rtol=1e-8,
            reference='prepared independent dense Cholesky; outside measured solver'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(), *sorted((ROOT/'src').rglob('*.py'))]}, cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        for name in ['synthetic-n640-smooth','concrete-observed','airfoil-observed']:
            data=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
            X,b=data['X_train'],data['y_train'];bandwidth=float(data['bandwidth'])
            case=dict(dataset=name,n=len(b),runs=[]);record['cases'].append(case)
            for history in [0,16]:
                kernel=GaussianKernel(X,bandwidth)
                x,metrics=coordinate_solve(kernel,b,.001,retained=history,max_seconds=25)
                error=x-data['alpha_reference']
                metrics['relative_coefficient_error']=float(np.linalg.norm(error)/np.linalg.norm(data['alpha_reference']))
                metrics['method']='sequential_greedy'
                case['runs'].append(metrics);save()
                print(json.dumps(dict(dataset=name,method='greedy',history=history,
                    **{k:metrics[k] for k in ['status','blocks','total_seconds','true_relative_residual']})),flush=True)
            for rank in [48,80]:
                for method in ['flex_compact','flex_power']:
                    kernel=GaussianKernel(X,bandwidth)
                    answer=revised_solve(kernel,b,.001,method=method,rank=rank,seed=0,
                        history_size=1,checkpoint=20,rtol=1e-8,maxiter=2500)
                    metrics=dict(answer.metrics,rank=rank,method=method)
                    case['runs'].append(metrics);save()
                    print(json.dumps(dict(dataset=name,method=method,rank=rank,
                        **{k:metrics[k] for k in ['converged','iterations','total_seconds']})),flush=True)
    record['status']='complete';save()


if __name__=='__main__':
    main()
