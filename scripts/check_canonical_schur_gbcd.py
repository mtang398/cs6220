#!/usr/bin/env python3
"""Bounded all-candidate GBCD control on the fixed-landmark Schur system.

Bo--Sminchisescu, https://arxiv.org/pdf/1206.3238, Eqs. (10)--(13),
Algorithms 1--2. The paper's random candidate subset is deliberately
replaced by all remaining candidates; this is its exhaustive-selection
variant, not exact projected/prefitting selection. No retained history.
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
from nystrom_project.revisions import revised_solve
from check_matrix_free_coordinates import coordinate_solve


def greedy_block(column, diagonal, residual, width=16, audit=None):
    """Choose by r_work[j]^2/S[j,j], refitting the small active set."""
    n = len(residual)
    width = min(width, n)
    C = np.empty((n, width), order='F')
    inverse = np.empty((width, width))
    coefficients = np.zeros(width)
    selected = []
    candidate_residual = residual.copy()
    timings = dict(selection_seconds=0., column_seconds=0.,
                   small_update_seconds=0., candidate_update_seconds=0.)
    for t in range(width):
        start = perf_counter()
        score = candidate_residual**2 / diagonal
        score[selected] = -np.inf
        j = int(np.argmax(score))
        timings['selection_seconds'] += perf_counter()-start
        start = perf_counter()
        C[:, t] = column(j)
        timings['column_seconds'] += perf_counter()-start
        start = perf_counter()
        if t:
            cross = C[selected, t]
            beta = inverse[:t, :t] @ cross
            delta = float(diagonal[j]-cross @ beta)
            if delta <= 128*np.finfo(float).eps*diagonal[j]:
                raise ArithmeticError('nonpositive or numerically unresolved block pivot')
            new_coefficient = float(residual[j]-cross @ coefficients[:t])/delta
            coefficients[:t] -= beta*new_coefficient
            coefficients[t] = new_coefficient
            inverse[:t, :t] += np.outer(beta, beta)/delta
            inverse[:t, t] = inverse[t, :t] = -beta/delta
            inverse[t, t] = 1/delta
        else:
            inverse[0, 0] = 1/diagonal[j]
            coefficients[0] = residual[j]/diagonal[j]
        selected.append(j)
        timings['small_update_seconds'] += perf_counter()-start
        start = perf_counter()
        # Selection-only residual; the persistent gradient changes once/block.
        if t+1 < width:
            candidate_residual = residual-C[:, :t+1] @ coefficients[:t+1]
        timings['candidate_update_seconds'] += perf_counter()-start
        if audit is not None:
            audit(selected, coefficients[:t+1], inverse[:t+1, :t+1])
    return np.asarray(selected), coefficients, C, timings


def independent_small_checks():
    rng = np.random.default_rng(622078)
    n = 29
    Z = rng.normal(size=(n, 11))
    A = Z@Z.T+.4*np.eye(n)
    b = rng.normal(size=n)
    I = np.array([7, 2, 18, 23, 4])
    J = np.setdiff1d(np.arange(n), I)
    L = linalg.cholesky(A[np.ix_(I, I)], lower=True)
    B = linalg.solve_triangular(L, A[np.ix_(I, J)], lower=True).T
    S = A[np.ix_(J, J)]-B@B.T
    q = b[J]-B@linalg.solve_triangular(L, b[I], lower=True)
    x = np.zeros(len(J))
    r = q.copy()
    maximum_inverse_error = maximum_refit_error = maximum_residual_gap = 0.
    selection_matches = True
    for block in range(4):
        old_r = r.copy()
        reference_selection = []
        for _ in range(16):
            if reference_selection:
                G = S[np.ix_(reference_selection, reference_selection)]
                d = linalg.solve(G, old_r[reference_selection], assume_a='pos')
                e = old_r-S[:, reference_selection]@d
            else:
                e = old_r.copy()
            score = e**2/np.diag(S)
            score[reference_selection] = -np.inf
            reference_selection.append(int(np.argmax(score)))
        def audit(active, d, inv):
            nonlocal maximum_inverse_error, maximum_refit_error
            G = S[np.ix_(active, active)]
            expected = linalg.solve(G, old_r[active], assume_a='pos')
            maximum_inverse_error = max(maximum_inverse_error,
                float(np.linalg.norm(G@inv-np.eye(len(active)), ord=np.inf)))
            maximum_refit_error = max(maximum_refit_error,
                float(np.linalg.norm(d-expected)/max(1., np.linalg.norm(expected))))
        active, d, C, _ = greedy_block(lambda j: S[:, j], np.diag(S), old_r, audit=audit)
        selection_matches &= active.tolist() == reference_selection
        x[active] += d
        r = old_r-C@d
        maximum_residual_gap = max(maximum_residual_gap,
            float(np.linalg.norm(q-S@x-r)/np.linalg.norm(b)))
    full = np.empty(n)
    full[J] = x
    full[I] = linalg.solve(A[np.ix_(I, I)], b[I]-A[np.ix_(I, J)]@x, assume_a='pos')
    full_residual = b-A@full
    schur_residual_error = float(np.linalg.norm(full_residual[J]-r)/np.linalg.norm(b))
    assert selection_matches
    assert max(maximum_inverse_error, maximum_refit_error,
               maximum_residual_gap, schur_residual_error) < 1e-10
    return dict(n=n, landmark_count=len(I), blocks=4, block_size=16,
                exact_selection_matches_independent_direct_solves=selection_matches,
                max_inverse_identity_error=maximum_inverse_error,
                max_refit_relative_error=maximum_refit_error,
                max_recursive_residual_gap=maximum_residual_gap,
                reconstructed_full_schur_residual_error=schur_residual_error)


def schur_gbcd(kernel, b, lam=.001, rank=16, width=16, max_seconds=30.):
    wall_start, cpu_start = perf_counter(), process_time()
    base = initialize_rpcholesky(kernel, rank, lam,
        np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
    indices = base.indices.copy()
    remaining = np.setdiff1d(np.arange(len(b)), indices)
    L = linalg.cholesky(base.columns[indices]+lam*np.eye(rank), lower=True)
    B = linalg.solve_triangular(L, base.columns[remaining].T, lower=True).T.copy()
    del base
    top_rhs = linalg.solve_triangular(L, b[indices], lower=True)
    r = b[remaining]-B@top_rhs
    x = np.zeros(len(remaining))
    diagonal = 1+lam-np.einsum('ij,ij->i', B, B)
    norm_b = np.linalg.norm(b)
    setup_seconds = perf_counter()-wall_start
    query_indices = []
    progress = []
    timings = dict(selection_seconds=0., column_seconds=0., small_update_seconds=0.,
                   candidate_update_seconds=0., block_gradient_seconds=0., verification_seconds=0.)
    verifications = replacements = 0
    last_verified = -1
    true_relative = float(np.linalg.norm(r)/norm_b)
    status = 'max_blocks'
    def column(j):
        query_indices.append(int(remaining[j]))
        c = kernel.entries(remaining, remaining[j:j+1]).ravel()-B@B[j]
        c[j] += lam
        return c
    def verify(block):
        nonlocal verifications, last_verified, true_relative
        start = perf_counter()
        full = np.empty(len(b)); full[remaining] = x
        full[indices] = linalg.solve_triangular(L.T, top_rhs-B.T@x, lower=False)
        residual = b-kernel.matvec(full)-lam*full
        true_relative = float(np.linalg.norm(residual)/norm_b)
        verifications += 1; last_verified = block
        progress.append(dict(block=block, columns=len(query_indices), true_relative_residual=true_relative,
            recursive_gap=float(np.linalg.norm(residual[remaining]-r)/norm_b)))
        timings['verification_seconds'] += perf_counter()-start
        return full, residual[remaining]
    for block in range(1, 10001):
        if perf_counter()-wall_start > max_seconds:
            status = 'wall_cap'; block -= 1; break
        active, d, C, local = greedy_block(column, diagonal, r, width)
        for key, value in local.items(): timings[key] += value
        start = perf_counter()
        x[active] += d
        r -= C@d
        timings['block_gradient_seconds'] += perf_counter()-start
        suspected = np.linalg.norm(r) <= 1e-8*norm_b
        if block % 50 == 0 or suspected or block == 10000:
            _, verified_r = verify(block)
            if true_relative <= 1e-8:
                status = 'converged'; break
            if suspected:
                r = verified_r.copy(); replacements += 1
    if last_verified != block: full, _ = verify(block)
    else:
        full = np.empty(len(b)); full[remaining] = x
        full[indices] = linalg.solve_triangular(L.T, top_rhs-B.T@x, lower=False)
    coarse_bytes = L.nbytes+B.nbytes+indices.nbytes+remaining.nbytes
    return full, dict(method='all_candidate_schur_gbcd', status=status,
        converged=true_relative <= 1e-8, true_relative_residual=true_relative,
        blocks=block, column_queries=len(query_indices), verification_matvecs=verifications,
        residual_replacements=replacements, setup_seconds=setup_seconds,
        total_seconds=perf_counter()-wall_start, total_cpu_seconds=process_time()-cpu_start,
        timing_breakdown=timings, kernel_stats=dict(kernel.stats),
        landmark_indices=indices.tolist(), selected_coordinate_trace=query_indices,
        coarse_state_bytes=coarse_bytes, current_block_columns_bytes=len(remaining)*width*8,
        small_block_inverse_bytes=width*width*8,
        memory_note='No retained history; coarse state plus one n_tail-by-16 column block, small inverse, and O(n) vectors; RP initialization/verification have separate temporary workspaces.',
        equivalent_full_entry_passes=len(query_indices)/len(b)+verifications+rank/len(b),
        progress=progress)


def main():
    output = ROOT/'results/canonical-schur-gbcd-diagnostic.json'
    if output.exists(): raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (58, 60))
    started, cpu_started = perf_counter(), process_time()
    data_path = ROOT/'data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz'
    record = dict(status='running', scope='One-case established baseline correction; exploratory single repeat, no novelty claim',
        protocol=dict(dataset='synthetic-n640-smooth', seed=0, regularization=.001,
            landmark_rank=16, block_size=16, retained_history=0, candidate_selection='all unused coordinates',
            subsampling=False, selection_rule='current within-block refit residual squared / original Schur diagonal',
            rtol=1e-8, verification_interval_blocks=50, blas_threads=1, cpu_limit_seconds=60,
            source='Bo--Sminchisescu Eqs10--13 and Algorithms1--2; no candidate subsampling'),
        primary_source='https://arxiv.org/pdf/1206.3238',
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(), ROOT/'scripts/check_matrix_free_coordinates.py',
             *sorted((ROOT/'src').rglob('*.py'))]},
        prepared_sha256=sha256(data_path.read_bytes()).hexdigest(), runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_small_spd_checks'] = independent_small_checks()
        save()
        data = np.load(data_path)
        X, b, bandwidth = data['X_train'], data['y_train'], float(data['bandwidth'])
        kernel = GaussianKernel(X, bandwidth)
        x, metrics = schur_gbcd(kernel, b)
        metrics['relative_coefficient_error'] = float(np.linalg.norm(x-data['alpha_reference'])/np.linalg.norm(data['alpha_reference']))
        record['runs'].append(metrics); save()
        kernel = GaussianKernel(X, bandwidth)
        _, metrics = coordinate_solve(kernel, b, .001, retained=0, max_seconds=15.)
        metrics['method'] = 'exhaustive_projected_prefitting_control'
        record['runs'].append(metrics); save()
        kernel = GaussianKernel(X, bandwidth)
        answer = revised_solve(kernel, b, .001, method='flex_power', rank=48,
            seed=0, history_size=1, checkpoint=20, rtol=1e-8, maxiter=2500)
        record['runs'].append(dict(answer.metrics, method='fixed_power_rank48_control')); save()
    record['status'] = 'complete'; save()
    for row in record['runs']:
        print(json.dumps({key:row[key] for key in ['method','status','converged','blocks','iterations',
            'column_queries','total_seconds','true_relative_residual'] if key in row}), flush=True)


if __name__ == '__main__': main()
