#!/usr/bin/env python3
"""Offline rebuilt mixed-column precision gate; no native resource claim."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from check_packed_float_factor import PackedFloat, ROOT
from check_packed_float_allocation import RawPointers


def pointer(array):
    return array.__array_interface__['data'][0]


class Helper:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.column = self.library.mixed_column
        self.column.argtypes = [ctypes.c_size_t, ctypes.c_size_t]+[ctypes.c_void_p]*7
        self.column.restype = ctypes.c_int
        self.solve = self.library.mixed_solve
        self.solve.argtypes = [ctypes.c_size_t]+[ctypes.c_void_p]*5
        self.solve.restype = ctypes.c_int


class MixedFactor:
    def __init__(self, C, promoted, helper):
        started = perf_counter()
        self.m = len(C)
        self.helper = helper
        self.high = np.zeros(self.m, dtype=np.uint8)
        self.high[np.asarray(promoted, dtype=np.int64)] = 1
        self.offsets = np.empty(self.m, dtype=np.uintp)
        counts = [0, 0]
        for j in range(self.m):
            kind = int(self.high[j])
            self.offsets[j] = counts[kind]
            counts[kind] += self.m-j
        self.low = np.empty(counts[0], dtype=np.float32)
        self.wide = np.empty(counts[1], dtype=np.float64)
        self.epsilon = np.empty(self.m)
        self.pivots = np.empty(self.m)
        for j in range(self.m):
            work = C[j:, j].copy()
            info = helper.column(self.m, j, pointer(self.high), pointer(self.offsets),
                                 pointer(self.low), pointer(self.wide), pointer(work),
                                 pointer(self.epsilon[j:j+1]), pointer(self.pivots[j:j+1]))
            if info:
                raise np.linalg.LinAlgError(f'Mixed factor pivot {info-1} failed: {self.pivots[j]}')
        self.setup_seconds = perf_counter()-started

    def solve(self, rhs):
        answer = np.array(rhs, dtype=np.float64, copy=True)
        info = self.helper.solve(self.m, pointer(self.high), pointer(self.offsets),
                                 pointer(self.low), pointer(self.wide), pointer(answer))
        if info:
            raise np.linalg.LinAlgError(f'Mixed solve diagonal {info-1} failed')
        return answer

    def dense_L(self):
        result = np.zeros((self.m, self.m))
        for j in range(self.m):
            offset = self.offsets[j]
            source = self.wide if self.high[j] else self.low
            result[j:, j] = source[offset:offset+self.m-j]
        return result

    def metadata(self):
        promoted = np.flatnonzero(self.high)
        return dict(promoted=promoted.tolist(), factor_bytes=self.low.nbytes+self.wide.nbytes,
            formula_factor_bytes=int(2*self.m*(self.m+1)+4*np.sum(self.m-promoted)),
            column_flags_and_offsets_bytes=self.high.nbytes+self.offsets.nbytes,
            epsilon_bytes=self.epsilon.nbytes, raw_pivot_diagnostic_bytes=self.pivots.nbytes,
            construction_working_column_max_bytes=8*self.m,
            setup_seconds=self.setup_seconds, minimum_raw_pivot=float(self.pivots.min()),
            epsilon_norm=float(np.linalg.norm(self.epsilon)))


def tiny_check(helper, original):
    rng = np.random.default_rng(622110)
    m = 17
    matrix = rng.normal(size=(m, m))
    C = matrix@matrix.T+np.eye(m)
    rhs = rng.normal(size=m)
    rows = []
    for name, promoted in [('all32', []), ('all64', list(range(m))), ('mixed', [1, 5, 11])]:
        factor = MixedFactor(C, promoted, helper)
        L = factor.dense_L()
        answer = factor.solve(rhs)
        reference = linalg.solve_triangular(L, rhs, lower=True)
        reference = linalg.solve_triangular(L.T, reference, lower=False)
        solve_error = float(np.linalg.norm(answer-reference)/np.linalg.norm(reference))
        # Each lower F column is the local construction defect in exact
        # arithmetic. A separate dense product checks diagonal inclusion.
        F = C-L@L.T
        eps_reference = np.array([np.linalg.norm(F[j:, j]) for j in range(m)])
        epsilon_error = float(np.max(np.abs(factor.epsilon-eps_reference)))
        assert solve_error < 1e-12 and epsilon_error < 1e-12
        record = dict(name=name, solve_relative_error=solve_error,
                      local_epsilon_max_absolute_error=epsilon_error,
                      minimum_raw_pivot=float(factor.pivots.min()))
        if name == 'all64':
            exact_L = linalg.cholesky(C, lower=True)
            exact_error = float(np.linalg.norm(L-exact_L)/np.linalg.norm(exact_L))
            assert exact_error < 1e-12
            record['factor_vs_exact64_relative_error'] = exact_error
        if name == 'all32':
            packed = np.empty(m*(m+1)//2, dtype=np.float32)
            pivots = np.empty(m)
            for j in range(m):
                work = C[j:, j].copy()
                assert original.packed_float_column(m, j, packed, work, pivots[j:j+1]) == 0
            exact_same = bool(np.array_equal(packed, factor.low))
            original_answer = rhs.copy()
            assert original.packed_float_solve(m, packed, original_answer) == 0
            solve_same = bool(np.array_equal(answer, original_answer))
            assert exact_same and solve_same
            record.update(bitwise_factor_matches_original=exact_same,
                          bitwise_solve_matches_original=solve_same)
        rows.append(record)
    return rows


def trajectory(K, b, selected, lam, *, retain=True):
    n = len(b)
    remaining = np.setdiff1d(np.arange(n), selected)
    C = K[np.ix_(selected, selected)]+lam*np.eye(len(selected))
    B = K[np.ix_(selected, remaining)]
    cf = linalg.cho_factor(C, lower=True)
    S = K[np.ix_(remaining, remaining)]+lam*np.eye(len(remaining))-B.T@linalg.cho_solve(cf, B)
    rhs = b[remaining]-B.T@linalg.cho_solve(cf, b[selected])
    y = np.zeros(len(remaining))
    r = rhs.copy()
    p = r.copy()
    rr = float(r@r)
    norm_b = float(np.linalg.norm(b))
    directions = []
    for iteration in range(2001):
        full = np.empty(n)
        full[remaining] = y
        full[selected] = linalg.cho_solve(cf, b[selected]-B@y)
        relative = float(np.linalg.norm(b-K@full-lam*full)/norm_b)
        if relative <= 1e-8 or iteration == 2000:
            break
        if retain:
            directions.append(p.copy())
        sp = S@p
        alpha = rr/float(p@sp)
        y += alpha*p
        r -= alpha*sp
        new_rr = float(r@r)
        p = r+(new_rr/rr)*p
        rr = new_rr
    assert relative <= 1e-8
    return dict(iterations=iteration, true_relative_residual=relative), C, B, cf, directions


def scores(factor, C, right_sides, F):
    m = len(C)
    bound_score = np.zeros(m)
    signed_derivative = np.zeros(m)
    for rhs in right_sides:
        t = rhs/np.linalg.norm(rhs)
        z = factor.solve(t)
        q = factor.solve(t-C@z)
        for j in range(m):
            vector = z[j]*q[j:]+q[j]*z[j:]
            vector[0] -= z[j]*q[j]
            bound_score[j] += factor.epsilon[j]*np.linalg.norm(vector)/(4*(m-j))
            signed_derivative[j] += float(F[j:, j]@vector)
    return bound_score, np.maximum(-signed_derivative, 0.)/(4*np.arange(m, 0, -1)), signed_derivative


def top8(values):
    return np.argsort(-values, kind='stable')[:8]


def one_case(name, path, helper, original):
    started, cpu_started = perf_counter(), process_time()
    with np.load(path) as data:
        X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
    lam, m = 1e-5, 592
    original600 = PackedFloat(X, b, bw, 600, lam, original)
    selected = original600.indices[:m]
    K = original600.kernel.block(X, X)
    trajectory592, C, B, cf, directions = trajectory(K, b, selected, lam)
    all32 = MixedFactor(C, [], helper)
    L = all32.dense_L()
    F = C-L@L.T
    right_sides = [B@p for p in directions]
    rhs_score, _, _ = scores(all32, C, [b[selected]], F)
    epsilon_score = all32.epsilon/(4*np.arange(m, 0, -1))
    masks = dict(no_promotions=np.array([], dtype=int),
                 epsilon_per_byte=top8(epsilon_score),
                 initial_rhs_score=top8(rhs_score), fixed_last8=np.arange(m-8, m))
    current_score = None
    signed_score = None
    signed_derivative = None
    if len(directions) >= 4:
        current_score, signed_score, signed_derivative = scores(all32, C, right_sides[:4], F)
        masks['current_four_rhs_score'] = top8(current_score)
        masks['signed_benefit_oracle'] = top8(signed_score)
    del F, L
    factors = {'no_promotions':all32}
    for label, mask in masks.items():
        if label != 'no_promotions':
            factors[label] = MixedFactor(C, mask, helper)
    methods = [(label, factor, 1) for label, factor in factors.items()]
    methods.append(('no_promotions_H2', all32, 2))
    observations = {label:[] for label, _, _ in methods}
    reference_core_residuals = []
    for index, t in enumerate(right_sides[4:], 5):
        reference = linalg.cho_solve(cf, t)
        tnorm = float(np.linalg.norm(t))
        reference_core_residuals.append(float(np.linalg.norm(t-C@reference)/tnorm))
        for label, factor, corrections in methods:
            answer = factor.solve(t)
            for _ in range(corrections):
                answer += factor.solve(t-C@answer)
            core_residual = float(np.linalg.norm(t-C@answer)/tnorm)
            global_error = float(np.linalg.norm(B.T@(reference-answer)))
            observations[label].append(dict(trajectory_iteration=index,
                actual_Schur_action_error=global_error, core_relative_residual=core_residual))
    target = np.array([row['actual_Schur_action_error'] for row in observations['no_promotions_H2']])
    all32H1 = np.array([row['actual_Schur_action_error'] for row in observations['no_promotions']])
    summaries = {}
    for label, rows in observations.items():
        errors = np.array([row['actual_Schur_action_error'] for row in rows])
        ratios = errors/np.maximum(target, np.finfo(float).tiny)
        baseline_ratios = errors/np.maximum(all32H1, np.finfo(float).tiny)
        metadata = factors[label if label != 'no_promotions_H2' else 'no_promotions'].metadata()
        assert metadata['factor_bytes'] == metadata['formula_factor_bytes']
        assert metadata['factor_bytes'] <= 721200
        summaries[label] = dict(metadata, corrections=2 if label == 'no_promotions_H2' else 1,
            heldout_count=len(rows),
            at_most_uniform_H2_count=int(np.count_nonzero(errors <= target)),
            median_error_ratio_to_uniform_H2=float(np.median(ratios)) if len(rows) else None,
            max_error_ratio_to_uniform_H2=float(np.max(ratios)) if len(rows) else None,
            median_error_ratio_to_uniform_H1=float(np.median(baseline_ratios)) if len(rows) else None,
            median_core_relative_residual=float(np.median([r['core_relative_residual'] for r in rows])) if rows else None,
            history=rows)
    control_labels = ['epsilon_per_byte', 'initial_rhs_score', 'fixed_last8']
    comparisons = {}
    if 'current_four_rhs_score' in observations:
        current_errors = np.array([row['actual_Schur_action_error'] for row in observations['current_four_rhs_score']])
        for label in control_labels+['signed_benefit_oracle']:
            control_errors = np.array([row['actual_Schur_action_error'] for row in observations[label]])
            comparisons[label] = dict(current_at_most_control_count=int(np.count_nonzero(current_errors <= control_errors)),
                median_current_to_control=float(np.median(current_errors/np.maximum(control_errors, np.finfo(float).tiny))) if len(current_errors) else None,
                overlap_promoted_columns=int(len(np.intersect1d(masks['current_four_rhs_score'], masks[label]))))
    # The 600-core trajectory is a separate headroom context. Its action errors
    # are never compared to a 592-core direction on a different Schur space.
    trajectory600, _, _, _, _ = trajectory(K, b, original600.indices, lam, retain=False)
    concentration = dict(epsilon_top8_density_fraction=float(epsilon_score[top8(epsilon_score)].sum()/epsilon_score.sum()),
                         epsilon_last8_norm_fraction=float(all32.epsilon[-8:].sum()/all32.epsilon.sum()))
    if current_score is not None:
        concentration.update(current_top8_score_fraction=float(current_score[top8(current_score)].sum()/current_score.sum()),
            current_last8_score_fraction=float(current_score[-8:].sum()/current_score.sum()),
            signed_positive_column_count=int(np.count_nonzero(signed_score)),
            signed_top8_positive_score_fraction=float(signed_score[top8(signed_score)].sum()/signed_score.sum()) if signed_score.sum() else 0.,
            current_selected_signed_derivatives=signed_derivative[masks['current_four_rhs_score']].tolist())
    n = len(b)
    return dict(dataset=name, n=n, regularization=lam, data_sha256=sha256(path.read_bytes()).hexdigest(),
        selected592_sha256=sha256(selected.tobytes()).hexdigest(),
        trajectory592=trajectory592, heldout_status='available' if len(directions)>4 else 'no_heldout_directions',
        selection_directions=4, heldout_directions=max(0, len(directions)-4),
        methods=summaries, current_vs_controls=comparisons, score_concentration=concentration,
        reference_precision=dict(dense_Cholesky_median_core_relative_residual=float(np.median(reference_core_residuals)) if reference_core_residuals else None,
            dense_Cholesky_max_core_relative_residual=max(reference_core_residuals, default=None),
            caveat='Action errors use finite-precision dense Cholesky references. Ratios near one or near the reference residual floor do not establish a strict accuracy ordering.'),
        core600_context=dict(trajectory=trajectory600, factor_bytes=original600.factor.nbytes,
            per_action_kernel_entries_fixed2=n*n+600*600,
            per_action_kernel_entries_hybrid592_H1=n*n,
            exact_trajectory_action_work_ratio=(trajectory592['iterations']*n*n)/(trajectory600['iterations']*(n*n+600*600)),
            caveat='Exact trajectory counts only, not mixed/inexact iteration predictions. Formula excludes setup, selection, boundaries, probes and original verification; no native comparison.'),
        resource_metadata=dict(base592_factor_bytes=2*m*(m+1), maximum_eight_promotion_factor_bytes=720944,
            margin_to_core600_factor_at_worst_mask=256,
            mixed_offset_and_flags_bytes=9*m, epsilon_bytes=8*m,
            first_four_rhs_storage_if_retained_bytes=4*8*m,
            dense_C_B_K_trajectory_and_oracle_F='Offline reference allocations, excluded from any native resource claim; no full1MiB feasibility asserted.'),
        elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started)


def main():
    output = ROOT/'results/core-precision-allocation.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    started, cpu_started = perf_counter(), process_time()
    source = ROOT/'scripts/mixed_column_factor.c'
    library = ROOT/'results/mixed_column_factor.so'
    compile_command = ['gcc', '-O3', '-march=native', '-ffp-contract=off', '-shared', '-fPIC',
                       str(source), '-o', str(library), '-lm']
    subprocess.run(compile_command, check=True)
    helper = Helper(library)
    original = RawPointers(ROOT/'results/packed_float_factor.so')
    old_path = ROOT/'results/compressed-core-adaptive-pilot.json'
    old = json.loads(old_path.read_text())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in old['sources'].items())
    sources = [Path(__file__), source, library, old_path, *[ROOT/p for p in old['sources']]]
    cases = [('smooth', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('mixed', 'synthetic-n2560-mixed-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        compile_command=compile_command, protocol=dict(cases=cases, regularization=1e-5,
            core=592, promotion_count=8, selection_prefix=4, heldout='All later directions of the same frozen exact592 Schur trajectory.',
            factor='Rebuilt from original C for every mask; FP64 accumulation, complete columns stored FP32 or FP64. No replacement of columns from an exact factor.',
            epsilon='At initial all32 construction, norm(w-stored_diagonal*stored_column), including diagonal.',
            main_score='Sum over4 normalized current t: epsilon_j*norm(z_j*q[j:]+q_j*z[j:]-z_j*q_j*e1)/(4*(m-j)), z=H0t, q=H0(t-Cz).',
            controls=['epsilon_per_byte', 'initial_bS_same_score', 'fixed_last8', 'uniform592_H1', 'uniform592_H2'],
            signed_oracle='Free dense F=C-LL.T; selecttop8 max(-sum f_j.dot(vector),0)/(4*(m-j)). Rebuild factor. Diagnostic only; reconstruction cost not a native proposal.',
            norm_comparison='All per-direction H1/H2 ratios use identical592 core and frozen direction. 600-core exact trajectory and query formula only separate context.',
            scope='Offline accuracy/headroom diagnostic only. No native timing or full-memory claim.',
            cpu_cap=60, command_wall_cap=90), tiny_check=None, runs=[], failures=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,
                      elapsed_cpu_seconds=process_time()-cpu_started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    save()
    with threadpool_limits(limits=1):
        try:
            record['tiny_check'] = tiny_check(helper, original)
            save()
            for name, filename in cases:
                case_start, case_cpu = perf_counter(), process_time()
                try:
                    row = one_case(name, ROOT/'data/prepared'/filename, helper, original)
                    record['runs'].append(row)
                    save()
                    print(json.dumps(dict(dataset=name, heldout=row['heldout_directions'],
                        H1_at_most_H2={label:method['at_most_uniform_H2_count'] for label,method in row['methods'].items()},
                        elapsed_cpu_seconds=row['elapsed_cpu_seconds'])), flush=True)
                except Exception as error:
                    record['failures'].append(dict(dataset=name, error=repr(error),
                        elapsed_wall_seconds=perf_counter()-case_start,
                        elapsed_cpu_seconds=process_time()-case_cpu))
                    save()
                    raise
        except Exception:
            record['status'] = 'failed'
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
