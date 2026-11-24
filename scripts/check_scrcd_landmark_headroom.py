#!/usr/bin/env python3
"""Offline fixed-landmark selection oracle for block Schur coordinate solves.

Dense matrix access, eigenvalues and full-candidate exchanges are diagnostic
oracles. Reported solve work counts are conceptual column/entry counts;
wall/CPU times measure this offline script only, not native kernel solves.
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


def residual_matrix(A, selected):
    C = A[:, selected]
    W = A[np.ix_(selected, selected)]
    factor = linalg.cho_factor(W, lower=True, check_finite=False)
    inverse = linalg.cho_solve(factor, np.eye(len(selected)), check_finite=False)
    E = A-C@linalg.cho_solve(factor, C.T, check_finite=False)
    return (E+E.T)*.5, C, inverse


def swap_scores(A, selected, objective):
    E, C, inverse = residual_matrix(A, selected)
    coefficients = C@inverse
    if objective == 'logdet':
        scores = coefficients**2+np.diag(E)[:, None]*np.diag(inverse)[None, :]
    else:
        removed = coefficients/np.sqrt(np.diag(inverse))[None, :]
        squared = np.sum(removed*removed, axis=0)
        denominator = np.diag(E)[:, None]+removed**2
        numerator = (np.sum(E*E, axis=0)[:, None]
                     +2*removed*(E.T@removed)+removed**2*squared[None, :])
        scores = np.divide(numerator, denominator,
            out=np.full_like(numerator, -np.inf), where=denominator>0)-squared[None, :]
    scores[selected, :] = -np.inf
    return scores, E


def refine(A, initial, objective, max_swaps=20):
    selected = initial.copy()
    events = []
    threshold = 1e-10
    scans = 0
    stopped = 'swap_limit'
    for step in range(max_swaps):
        scores, E = swap_scores(A, selected, objective)
        scans += 1
        j, slot = np.unravel_index(np.argmax(scores), scores.shape)
        best = float(scores[j, slot])
        old_trace = float(np.trace(E))
        if (objective == 'logdet' and best <= 1+threshold) or (
                objective == 'trace' and best <= threshold*max(1., old_trace)):
            stopped = 'no_improving_swap'; break
        old_logdet = float(np.linalg.slogdet(A[np.ix_(selected, selected)])[1])
        old_index = int(selected[slot])
        selected[slot] = j
        new_E, _, _ = residual_matrix(A, selected)
        new_trace = float(np.trace(new_E))
        new_logdet = float(np.linalg.slogdet(A[np.ix_(selected, selected)])[1])
        expected = np.log(best) if objective == 'logdet' else best
        actual = new_logdet-old_logdet if objective == 'logdet' else old_trace-new_trace
        assert actual > 0
        assert abs(expected-actual) < 1e-8*max(1., abs(expected), abs(old_trace))
        events.append(dict(step=step+1, removed=old_index, added=int(j),
            score=best, actual_objective_gain=actual, trace=new_trace,
            core_logdet=new_logdet))
    return selected, dict(objective=objective, initialized_from='matrix_greedy_K',
        accepted_swaps=len(events), all_candidate_scans=scans,
        scored_candidate_removal_pairs=scans*(len(A)-len(initial))*len(initial),
        stop_reason=stopped, events=events,
        access_note='Dense A and all nonselected candidates/removals used offline; no native acquisition cost claim.')


def schur(A, b, selected):
    remaining = np.setdiff1d(np.arange(len(b)), selected)
    L = linalg.cholesky(A[np.ix_(selected, selected)], lower=True, check_finite=False)
    B = linalg.solve_triangular(L, A[np.ix_(selected, remaining)],
        lower=True, check_finite=False).T
    top = linalg.solve_triangular(L, b[selected], lower=True, check_finite=False)
    S = A[np.ix_(remaining, remaining)]-B@B.T
    return remaining, L, B, top, (S+S.T)*.5, b[remaining]-B@top


def assemble(n, selected, remaining, L, B, top, tail):
    full = np.empty(n)
    full[remaining] = tail
    full[selected] = linalg.solve_triangular(L.T, top-B.T@tail,
        lower=False, check_finite=False)
    return full


def small_checks():
    rng = np.random.default_rng(622081)
    n, k = 23, 5
    Z = rng.normal(size=(n, n))
    A = Z@Z.T+.7*np.eye(n)
    b = rng.normal(size=n)
    selected = np.array([7, 2, 18, 4, 11])
    E, C, inverse = residual_matrix(A, selected)
    scores_det, _ = swap_scores(A, selected, 'logdet')
    scores_trace, _ = swap_scores(A, selected, 'trace')
    old_logdet = float(np.linalg.slogdet(A[np.ix_(selected, selected)])[1])
    max_trace_error = max_det_error = max_update_error = 0.
    for j in np.setdiff1d(np.arange(n), selected):
        for slot in range(k):
            replacement = selected.copy(); replacement[slot] = j
            new_E, _, _ = residual_matrix(A, replacement)
            a = C@inverse[:, slot]/np.sqrt(inverse[slot, slot])
            v = E[:, j]+a*a[j]
            updated = E+np.outer(a, a)-np.outer(v, v)/(E[j, j]+a[j]**2)
            max_update_error = max(max_update_error,
                float(np.linalg.norm(updated-new_E)/np.linalg.norm(new_E)))
            actual_trace = np.trace(E)-np.trace(new_E)
            max_trace_error = max(max_trace_error, abs(float(scores_trace[j, slot]-actual_trace)))
            actual_logdet = np.linalg.slogdet(A[np.ix_(replacement, replacement)])[1]-old_logdet
            max_det_error = max(max_det_error, abs(float(np.log(scores_det[j, slot])-actual_logdet)))
    remaining, L, B, top, S, q = schur(A, b, selected)
    exact_tail = linalg.solve(S, q, assume_a='pos')
    exact_full = assemble(n, selected, remaining, L, B, top, exact_tail)
    direct = linalg.solve(A, b, assume_a='pos')
    solution_error = float(np.linalg.norm(exact_full-direct)/np.linalg.norm(direct))
    # A Schur block step must equal an independent full solve on S union J.
    tail = np.zeros(len(remaining))
    before = assemble(n, selected, remaining, L, B, top, tail)
    J = np.array([1, 4, 6, 8, 12, 15])
    tail[J] = linalg.solve(S[np.ix_(J, J)], q[J], assume_a='pos')
    after = assemble(n, selected, remaining, L, B, top, tail)
    union = np.r_[selected, remaining[J]]
    independently = before.copy()
    independently[union] += linalg.solve(A[np.ix_(union, union)],
        (b-A@before)[union], assume_a='pos')
    step_error = float(np.linalg.norm(after-independently)/max(1., np.linalg.norm(after)))
    full_r = b-A@after
    schur_r = q-S@tail
    residual_error = float(np.linalg.norm(full_r[remaining]-schur_r)/np.linalg.norm(b))
    union_residual = float(np.linalg.norm(full_r[union])/np.linalg.norm(b))
    # Reinterpreting a k-subset of the zero-residual union changes neither x nor r.
    replacement = union[-k:]
    R2, L2, B2, top2, _, _ = schur(A, b, replacement)
    reassembled = assemble(n, replacement, R2, L2, B2, top2, after[R2])
    reinterpretation_error = float(np.linalg.norm(after-reassembled)/max(1., np.linalg.norm(after)))
    assert max(max_update_error, max_det_error, max_trace_error,
        solution_error, step_error, residual_error, union_residual, reinterpretation_error) < 1e-10
    return dict(n=n, k=k, swap_pairs_checked=(n-k)*k,
        maximum_trace_gain_absolute_error=max_trace_error,
        maximum_logdet_gain_absolute_error=max_det_error,
        maximum_schur_update_relative_error=max_update_error,
        full_vs_schur_exact_solution_relative_error=solution_error,
        full_union_vs_schur_block_step_relative_error=step_error,
        full_vs_schur_residual_relative_error=residual_error,
        union_relative_residual=union_residual,
        landmark_reinterpretation_relative_error=reinterpretation_error)


def solve_fixed(A, b, selected, cpu_start, width=16, max_blocks=20000):
    n = len(b)
    remaining, L, B, top, S, q = schur(A, b, selected)
    values = linalg.eigvalsh(S, check_finite=False)
    spectrum = dict(condition=float(values[-1]/values[0]),
        lambda_min=float(values[0]), lambda_max=float(values[-1]),
        trace=float(values.sum()), lambda_min_over_trace=float(values[0]/values.sum()),
        core_condition=float(np.linalg.cond(A[np.ix_(selected, selected)])))
    tail = np.zeros(len(remaining)); r = q.copy()
    norm_b = np.linalg.norm(b)
    labels_to_tail = np.full(n, -1, dtype=int); labels_to_tail[remaining] = np.arange(len(remaining))
    rng = np.random.default_rng(622082)
    schedule_hash = sha256()
    chosen_hash = sha256()
    checks = replacements = 0
    last_verified = -1
    progress = []
    status = 'max_blocks'
    true_relative = float(np.linalg.norm(r)/norm_b)
    def verify(block):
        nonlocal checks, last_verified, true_relative
        full = assemble(n, selected, remaining, L, B, top, tail)
        full_r = b-A@full
        true_relative = float(np.linalg.norm(full_r)/norm_b)
        checks += 1; last_verified = block
        progress.append(dict(block=block, true_relative_residual=true_relative,
            full_schur_residual_gap=float(np.linalg.norm(full_r[remaining]-r)/norm_b),
            landmark_relative_residual=float(np.linalg.norm(full_r[selected])/norm_b)))
        return full, full_r[remaining]
    for block in range(1, max_blocks+1):
        if process_time()-cpu_start > 53:
            status = 'global_cpu_guard'; block -= 1; break
        order = rng.permutation(n)
        schedule_hash.update(order.astype(np.int32).tobytes())
        mapped = labels_to_tail[order]
        J = mapped[mapped >= 0][:width]
        chosen_hash.update(remaining[J].astype(np.int32).tobytes())
        delta = linalg.solve(S[np.ix_(J, J)], r[J], assume_a='pos', check_finite=False)
        tail[J] += delta
        r -= S[:, J]@delta
        suspected = np.linalg.norm(r) <= 1e-8*norm_b
        if block % 50 == 0 or suspected or block == max_blocks:
            _, full_tail_r = verify(block)
            if true_relative <= 1e-8:
                status = 'converged'; break
            if suspected:
                r = full_tail_r.copy(); replacements += 1
    if last_verified != block: full, _ = verify(block)
    else: full = assemble(n, selected, remaining, L, B, top, tail)
    queries = block*width
    return full, dict(status=status, converged=true_relative <= 1e-8,
        blocks=block, tail_block_column_queries=queries,
        initial_landmark_column_queries=len(selected), verification_matvecs=checks,
        conceptual_kernel_entry_queries=n*len(selected)+len(remaining)*queries+n*n*checks,
        equivalent_full_entry_passes=len(selected)/n+len(remaining)*queries/(n*n)+checks,
        true_relative_residual=true_relative, residual_replacements=replacements,
        schur_spectrum=spectrum, selected_indices=selected.tolist(),
        original_label_priority_stream_sha256=schedule_hash.hexdigest(),
        accepted_block_label_stream_sha256=chosen_hash.hexdigest(), progress=progress,
        access_note='Solve uses dense S offline; entry/column counts describe a direct matrix-free Schur implementation without a retained tail-column cache.')


def main():
    output = ROOT/'results/scrcd-landmark-headroom-diagnostic.json'
    if output.exists(): raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (58, 60))
    wall_start, cpu_start = perf_counter(), process_time()
    record = dict(status='running', scope='Offline fixed-landmark selection-quality oracle; no native runtime or adaptive-method claim',
        protocol=dict(seed=0, regularization=.001, landmark_rank=16, block_size=16,
            max_blocks=20000, true_rtol=1e-8, verify_every_blocks=50, momentum=False,
            retained_history=0, initial_iterate='exact coordinate Galerkin on the landmark set',
            block_schedule_seed=622082,
            block_schedule='same original-label random permutations; first16 labels outside each fixed landmark set',
            candidate_sets=['rp_K','matrix_greedy_K','max_volume_A_from_greedy','minimum_trace_A_from_greedy'],
            refinement='at most20 best-improving swaps over all nonselected candidates and all removals',
            cpu_limit_seconds=60, blas_threads=1,
            cost_note='Dense K/A, Schur matrices, eigenvalues and all-candidate scans are oracle access; conceptual solver entry counts exclude this extra oracle selection access'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py']}, cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-wall_start,
            elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_small_spd_checks'] = small_checks(); save()
        for name in ['synthetic-n640-smooth','concrete-observed','synthetic-n640-mixed']:
            if process_time()-cpu_start > 52:
                record['status'] = 'cpu_budget_exhausted'; save(); return
            prepared = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data = np.load(prepared)
            X, b = data['X_train'], data['y_train']
            kernel = GaussianKernel(X, float(data['bandwidth']))
            K = kernel.dense(); A = K+.001*np.eye(len(b))
            sets = {}
            for label, pivoting in [('rp_K','random'),('matrix_greedy_K','greedy')]:
                base = initialize_rpcholesky(kernel, 16, .001,
                    np.random.default_rng(np.random.SeedSequence([0,6220,1])), pivoting=pivoting)
                sets[label] = (base.indices.copy(), None)
            for label, objective in [('max_volume_A_from_greedy','logdet'),('minimum_trace_A_from_greedy','trace')]:
                sets[label] = refine(A, sets['matrix_greedy_K'][0], objective)
            row = dict(dataset=name, n=len(b), prepared_sha256=sha256(prepared.read_bytes()).hexdigest(),
                regularization=.001, dense_oracle_entries=len(b)**2, runs=[])
            record['cases'].append(row); save()
            reference = linalg.solve(A, b, assume_a='pos', check_finite=False)
            for label, (selected, refinement) in sets.items():
                full, metrics = solve_fixed(A, b, selected, cpu_start)
                metrics.update(method=label, refinement=refinement,
                    relative_solution_error=float(np.linalg.norm(full-reference)/np.linalg.norm(reference)))
                row['runs'].append(metrics); save()
                print(json.dumps(dict(dataset=name, method=label, blocks=metrics['blocks'],
                    status=metrics['status'], condition=metrics['schur_spectrum']['condition'],
                    trace=metrics['schur_spectrum']['trace'],
                    accepted_swaps=None if refinement is None else refinement['accepted_swaps'])), flush=True)
                if metrics['status'] == 'global_cpu_guard':
                    record['status'] = 'cpu_budget_exhausted'; save(); return
    record['status'] = 'complete'; save()
    print(json.dumps(dict(status='complete', cpu_seconds=record['elapsed_cpu_seconds'])), flush=True)


if __name__ == '__main__': main()
