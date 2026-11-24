#!/usr/bin/env python3
"""Dense core-error subspace gate with matched restart and static controls."""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import check_core_precision_outer_correction as old
from check_core_precision_allocation import MixedFactor, Helper
from check_schur_inplace_frontier_v2 import Budget, SharedKernel, ROOT

PIVOT_TOL = 1e-10
PREFIX = 4


class Archive:
    """Normalized S,W; incremental C-Gram from already computed Cz."""
    def __init__(self, m):
        self.m = m
        self.S, self.W = np.empty((m, 0)), np.empty((m, 0))
        self.T = np.empty((0, 0))
        self.events = []
        self.factor = None

    def append(self, z, delta, Cz):
        energy = float(z@Cz)
        if not np.isfinite(energy) or energy <= 0:
            self.events.append(dict(accepted=False, reason='nonpositive_C_energy', energy=energy))
            return
        scale = np.sqrt(energy)
        s, w, cs = z/scale, delta/scale, Cz/scale
        cross = self.S.T@cs
        diagonal = float(s@cs)
        if len(self.T):
            coefficient = cho_solve(cho_factor(self.T, lower=True), cross)
            pivot = diagonal-float(cross@coefficient)
        else:
            pivot = diagonal
        if not np.isfinite(pivot) or pivot <= PIVOT_TOL*diagonal:
            self.events.append(dict(accepted=False, reason='dependent_C_Gram', normalized_pivot_squared=pivot))
            return
        rank = self.W.shape[1]
        T = np.empty((rank+1, rank+1))
        T[:rank, :rank], T[:rank, rank], T[rank, :rank], T[rank, rank] = self.T, cross, cross, diagonal
        T = (T+T.T)*.5
        cho_factor(T, lower=True)
        self.S = np.column_stack((self.S, s))
        self.W = np.column_stack((self.W, w))
        self.T = T
        self.events.append(dict(accepted=True, normalized_pivot_squared=pivot, raw_C_energy=energy))

    def freeze(self):
        self.condition = float(np.linalg.cond(self.T)) if len(self.T) else None
        self.factor = cho_factor(self.T, lower=True) if len(self.T) else None
        self.S = None

    def apply(self, rhs):
        if self.factor is None:
            return np.zeros_like(rhs)
        return self.W@cho_solve(self.factor, self.W.T@rhs)

    def metadata(self):
        return dict(rank=self.W.shape[1], normalized_T_condition=getattr(self, 'condition', None),
            normalized_T=self.T.tolist(), events=self.events,
            paired_archive_bytes_before_freeze=2*self.W.nbytes+self.T.nbytes,
            retained_W_T_factor_bytes=self.W.nbytes+self.T.nbytes+(0 if self.factor is None else self.factor[0].nbytes),
            S_released=self.S is None, W_sha256=sha256(self.W.tobytes()).hexdigest())


def run(K, b, selected, helper, mode, *, cpu_cap=5., deadline=float('inf')):
    started, cpu_started = perf_counter(), process_time()
    stop_cpu = min(deadline, cpu_started+cpu_cap)
    n, m, lam = len(b), len(selected), 1e-5
    R = np.setdiff1d(np.arange(n), selected)
    C = K[np.ix_(selected, selected)]+lam*np.eye(m)
    B = K[np.ix_(selected, R)]
    D = K[np.ix_(R, R)]+lam*np.eye(len(R))
    factor = MixedFactor(C, np.arange(8), helper)
    counts = dict(action_core_products=0, construction_core_products=0,
        boundary_core_products=0, factor_builds=1, Schur_actions=0,
        boundary_cross_products=0, original_verifications=0, triangular_solves=0,
        transition_Schur_actions=0, correction_applications=0)
    archive = Archive(m)
    active, transition, frozen = False, None, False
    delayed = mode in ['current_after4', 'random_after4', 'matched_restart']

    def base(rhs):
        counts['triangular_solves'] += 1
        return factor.solve(rhs)

    def H1(rhs, kind, collect=False):
        z = base(rhs)
        Cz = C@z
        counts[kind+'_core_products'] += 1
        residual = rhs-Cz
        delta = base(residual)
        if collect:
            archive.append(z, delta, Cz)
        result = z+delta
        if active:
            result += archive.apply(rhs)
            counts['correction_applications'] += 1
        return result

    def boundary(rhs):
        answer = base(rhs)
        for _ in range(4):
            residual = rhs-C@answer
            counts['boundary_core_products'] += 1
            answer += base(residual)
        return answer

    def action(v, collect=False):
        top = H1(B@v, 'action', collect)
        counts['Schur_actions'] += 1
        return D@v-B.T@top

    if mode in ['random_upfront', 'random_after4']:
        rng = np.random.default_rng(np.random.SeedSequence([0, 6220, 79]))
        for _ in range(PREFIX):
            H1(rng.normal(size=m), 'construction', collect=True)
        archive.freeze()
        frozen = True
        active = mode == 'random_upfront'

    full_x, full_residual = np.zeros(n), b.copy()
    norm_b = float(np.linalg.norm(b))
    inner_target = .5e-8*norm_b
    cycles, total_iterations = [], 0
    status, original_relative = 'correction_cycle_cap', 1.
    for cycle in range(4):
        before = dict(counts)
        initial_relative = float(np.linalg.norm(full_residual)/norm_b)
        top = boundary(full_residual[selected])
        rhs = full_residual[R]-B.T@top
        counts['boundary_cross_products'] += 1
        v, r = np.zeros(len(R)), rhs.copy()
        p, rr = r.copy(), float(r@r)
        iterations = 0
        inner_status = 'already_below_absolute_target' if np.linalg.norm(r) <= inner_target else 'maxiter'
        if inner_status == 'maxiter':
            for iterations in range(1, 2001):
                if process_time() > stop_cpu:
                    inner_status = 'cpu_screen_cap'
                    iterations -= 1
                    break
                collect = mode == 'current_after4' and cycle == 0 and iterations <= PREFIX
                ap = action(p, collect)
                curvature = float(p@ap)
                if curvature <= 0 or not np.isfinite(curvature+rr):
                    inner_status = 'nonpositive_curvature'
                    break
                alpha = rr/curvature
                v += alpha*p
                r -= alpha*ap
                if np.linalg.norm(r) <= inner_target:
                    inner_status = 'absolute_recursive_target'
                    break
                if delayed and cycle == 0 and iterations == PREFIX:
                    old_recursive = r.copy()
                    if mode == 'current_after4':
                        archive.freeze()
                        frozen, active = True, True
                    elif mode == 'random_after4':
                        active = True
                    r = rhs-action(v)
                    counts['transition_Schur_actions'] += 1
                    transition = dict(after_inner_iteration=PREFIX, recurrence_restarted=True,
                        new_true_Schur_relative_residual=float(np.linalg.norm(r)/norm_b),
                        old_recursive_relative_residual=float(np.linalg.norm(old_recursive)/norm_b),
                        residual_change_relative_to_original_b=float(np.linalg.norm(r-old_recursive)/norm_b),
                        correction_active=active, correction_rank=archive.W.shape[1])
                    p, rr = r.copy(), float(r@r)
                    if np.linalg.norm(r) <= inner_target:
                        inner_status = 'absolute_true_target_at_transition'
                        break
                    continue
                next_rr = float(r@r)
                p = r+(next_rr/rr)*p
                rr = next_rr
        total_iterations += iterations
        u = boundary(full_residual[selected]-B@v)
        counts['boundary_cross_products'] += 1
        full_x[selected] += u
        full_x[R] += v
        full_residual = b-K@full_x-lam*full_x
        counts['original_verifications'] += 1
        original_relative = float(np.linalg.norm(full_residual)/norm_b)
        cycles.append(dict(cycle=cycle, initial_original_relative_residual=initial_relative,
            inner_iterations=iterations, inner_status=inner_status,
            inner_recursive_relative_to_original_b=float(np.linalg.norm(r)/norm_b),
            final_original_relative_residual=original_relative,
            final_landmark_relative_residual=float(np.linalg.norm(full_residual[selected])/norm_b),
            final_tail_relative_residual=float(np.linalg.norm(full_residual[R])/norm_b),
            operation_counts={key:counts[key]-before[key] for key in counts}))
        if original_relative <= 1e-8:
            status = 'converged'
            break
        if inner_status in ['cpu_screen_cap', 'nonpositive_curvature', 'maxiter']:
            status = inner_status
            break
    paid_core = sum(counts[k] for k in ['action_core_products', 'boundary_core_products', 'construction_core_products'])
    components = dict(FPS_Gaussian_entries=n*m, streamed_factor_entries=m*(m+1)//2,
        Schur_cross_and_tail_entries=counts['Schur_actions']*(n*n-m*m),
        core_product_entries=paid_core*m*m,
        boundary_cross_entries=counts['boundary_cross_products']*m*(n-m),
        original_verification_entries=counts['original_verifications']*n*n)
    return dict(method=mode, m=m, status=status, converged=original_relative <= 1e-8,
        total_inner_iterations=total_iterations, correction_cycles=len(cycles)-1, cycles=cycles,
        pre_repair_original_relative_residual=cycles[0]['final_original_relative_residual'],
        final_original_relative_residual=original_relative, counts=counts, paid_core_products=paid_core,
        entry_components=components, analytical_solver_Gaussian_entries=sum(components.values()),
        FPS_squared_distance_pair_evaluations=n*m, factor=factor.metadata(),
        archive=archive.metadata(), correction_frozen=frozen, transition=transition,
        full_solution_sha256=sha256(full_x.tobytes()).hexdigest(),
        original_residual_sha256=sha256(full_residual.tobytes()).hexdigest(),
        offline_diagnostic_seconds=perf_counter()-started,
        offline_diagnostic_cpu_seconds=process_time()-cpu_started)


def tiny_check():
    rng = np.random.default_rng(622080)
    F = rng.normal(size=(17, 17))
    C = F@F.T+np.eye(17)
    L = np.linalg.cholesky(C)
    L[:, :5] *= 1.08
    H0 = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(17)))
    H1 = 2*H0-H0@C@H0
    archive = Archive(17)
    zs, ds = [], []
    for _ in range(4):
        t = rng.normal(size=17)
        z = H0@t
        Cz = C@z
        delta = H0@(t-Cz)
        archive.append(z, delta, Cz)
        zs.append(z)
        ds.append(delta)
    S, W = np.column_stack(zs), np.column_stack(ds)
    expected = W@np.linalg.solve(S.T@C@S, W.T)
    gram_error = np.linalg.norm(archive.T-archive.S.T@C@archive.S)
    archive.freeze()
    correction = np.column_stack([archive.apply(v) for v in np.eye(17)])
    relative = np.linalg.norm(correction-expected)/np.linalg.norm(expected)
    gap = np.linalg.inv(C)-H1-correction
    minimum = float(np.linalg.eigvalsh((gap+gap.T)*.5)[0])
    assert gram_error < 1e-12 and relative < 1e-11 and minimum > -1e-12
    assert archive.S is None and archive.W.shape[1] == 4
    return dict(C_Gram_error=float(gram_error), correction_identity_relative_error=float(relative),
                minimum_remaining_inverse_gap_eigenvalue=minimum, archive=archive.metadata())


def main():
    output = ROOT/'results/core-error-subspace.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    started, cpu_started = perf_counter(), process_time()
    sources = [Path(__file__).resolve(), Path(old.__file__).resolve(),
               Path(old.prior.__file__).resolve(), Path(old.prior.precision.__file__).resolve(),
               ROOT/'scripts/check_schur_inplace_frontier_v2.py',
               ROOT/'scripts/mixed_column_factor_first8.c',
               ROOT/'results/mixed_column_factor_first8.so']
    modes = ['unchanged', 'matched_restart', 'current_after4', 'random_upfront', 'random_after4']
    cases = [('smooth', 'synthetic-n2560-smooth-s0-lambda0.001.npz'),
             ('mixed', 'synthetic-n2560-mixed-s0-lambda0.001.npz'),
             ('Concrete', 'concrete-observed-s0-lambda1e-05.npz'),
             ('Airfoil', 'airfoil-observed-s0-lambda1e-05.npz')]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(cases=cases, seed=0, regularization=1e-5, modes=modes,
            candidate_core=568, strong_static_core=592, retained_directions=4,
            prefix=PREFIX, normalized_C_Gram_pivot_cutoff=PIVOT_TOL,
            correction='H1+W(S.T C S)^-1 W.T; S=H0t, W=H0(t-C H0t); fixed C,H0 throughout.',
            acquisition='First4 current Schur-action core RHS; existing Cz reused. Random controls pay4 additional core products.',
            restart='One charged recomputation of new Schur residual after4 iterations, including matched restart.',
            stopping='Common base H4 boundary, absolute inner .5e-8||original b||, max3 additional full original-residual corrections, final original1e-8.',
            scope='Dense offline trajectories and analytical entry counts only. Full allocation, native runtime and novelty not established.',
            memory_note='568 factor+metadata+two rank4 archives+Gram706040B versus592 factor+metadata726272B; this is not a complete memory audit.',
            wall_cap_seconds=120, cpu_cap_seconds=80, per_run_cpu_seconds=5),
        tiny_check=None, runs=[], failures=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started,
                      offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp = output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temp.replace(output)
    save()
    helper = Helper(ROOT/'results/mixed_column_factor_first8.so')
    with threadpool_limits(limits=1):
        try:
            record['tiny_check'] = tiny_check()
            save()
            for name, filename in cases:
                path = ROOT/'data/prepared'/filename
                with np.load(path) as data:
                    X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
                kernel = SharedKernel(X, bw, Budget([X, b]))
                selected = kernel.farthest(592)
                K = kernel.block(X, X)
                for method in modes+['FIRST8_H1', 'FIRST8_H2prefix4_then_H1']:
                    if process_time()-cpu_started > 72:
                        row = dict(method=method, status='global_cpu_not_started', converged=False)
                    elif method in modes:
                        row = run(K, b, selected[:568], helper, method, deadline=cpu_started+77)
                    else:
                        row = old.run(K, b, selected, 1e-5, method, helper, cpu_cap=5.)
                    row.update(dataset=name, n=len(b), regularization=1e-5,
                               data_sha256=sha256(path.read_bytes()).hexdigest())
                    record['runs'].append(row)
                    save()
                    print(json.dumps({k:row.get(k) for k in ['dataset','method','status','total_inner_iterations',
                        'correction_cycles','analytical_solver_Gaussian_entries']}), flush=True)
                del K
        except Exception as error:
            record['status'] = 'failed'
            record['failures'].append(repr(error))
            save()
            raise
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
