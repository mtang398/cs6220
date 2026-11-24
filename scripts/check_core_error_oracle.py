#!/usr/bin/env python3
"""Dense coupled inverse-error oracle; original subspace source preserved."""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, eigh
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


def run(K, b, selected, helper, mode, *, oracle=None, cpu_cap=5., deadline=float('inf')):
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
    delayed = mode in ['current_after4', 'random_after4', 'matched_restart', 'oracle_after4']
    exact_factor = cho_factor(C, lower=True) if mode == 'exact_core' else None
    if oracle is not None:
        archive.W = oracle.copy()
        archive.T = np.eye(oracle.shape[1])
        archive.freeze()
        active, frozen = mode == 'oracle_upfront', True

    def base(rhs):
        counts['triangular_solves'] += 1
        return factor.solve(rhs)

    def H1(rhs, kind, collect=False):
        if exact_factor is not None:
            return cho_solve(exact_factor, rhs)
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
                    elif mode in ['random_after4', 'oracle_after4']:
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


def coupled_oracle(C, B, helper):
    factor = MixedFactor(C, np.arange(8), helper)
    L = factor.dense_L()
    left = solve_triangular(L, C, lower=True)
    G = solve_triangular(L, left.T, lower=True).T
    E = (G+G.T)*.5-np.eye(len(C))
    theta, V = eigh(E)
    assert theta[0] > -1
    eta = theta**2/(1+theta)
    Z = solve_triangular(L.T, V*np.sqrt(eta), lower=False)
    Y = B.T@Z
    gram = Y.T@Y
    values, directions = eigh((gram+gram.T)*.5)
    W = Z@directions[:, -4:]
    return W, dict(whitened_error_extremes=[float(theta[0]),float(theta[-1])],
        coupled_gap_eigenvalues_descending=values[::-1].tolist(),
        optimal_rank4_remaining_spectral_fraction=float(values[-5]/values[-1]),
        optimal_rank4_remaining_frobenius_fraction=float(np.linalg.norm(values[:-4])/np.linalg.norm(values)),
        factor_sha256=sha256(factor.low.tobytes()+factor.wide.tobytes()).hexdigest(),
        W_sha256=sha256(W.tobytes()).hexdigest(), stored_correction_bytes=W.nbytes)


def tiny_oracle_check(helper):
    rng = np.random.default_rng(622081)
    F = rng.normal(size=(17,17))
    C = F@F.T+np.eye(17)
    B = rng.normal(size=(17,13))
    W, info = coupled_oracle(C, B, helper)
    factor = MixedFactor(C, np.arange(8), helper)
    H0 = np.column_stack([factor.solve(v) for v in np.eye(17)])
    H1 = 2*H0-H0@C@H0
    gap = np.linalg.inv(C)-H1-W@W.T
    minimum = float(np.linalg.eigvalsh((gap+gap.T)*.5)[0])
    assert minimum > -1e-12
    return dict(minimum_remaining_inverse_gap_eigenvalue=minimum, correction_bytes=W.nbytes)


def main():
    output = ROOT/'results/core-error-oracle.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85))
    started, cpu_started = perf_counter(), process_time()
    prior_path = ROOT/'results/core-error-subspace.json'
    prior = json.loads(prior_path.read_text())
    assert prior['status'] == 'complete'
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items())
    paths = [Path(__file__).resolve(), prior_path, *[ROOT/p for p in prior['sources']]]
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths},
        protocol=dict(core=568, rank=4, regularization=1e-5, seed=0, prefix=4,
            oracle='F=Cinv-H1=ZZ.T; top4 right singular vectors of B.T@Z give W=ZV4. This minimizes spectral/Frobenius Schur-gap approximation error, not solve time or RHS-specific iteration count.',
            modes=['unchanged','matched_restart','oracle_upfront','oracle_after4','exact_core'],
            reference='Exact-core568 action with the same baseH4 boundary is a free-precision reference, not feasible under1MiB.',
            scope='Dense factorization,eigendecomposition,B and oracle construction free; no adaptive/native performance or memory claim.',
            stopping=prior['protocol']['stopping'], wall_cap_seconds=120,cpu_cap_seconds=80),
        tiny_check=None,runs=[],spectra=[],failures=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
        temp=output.with_suffix('.json.tmp');temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temp.replace(output)
    helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
    save()
    with threadpool_limits(limits=1):
        try:
            record['tiny_check']=tiny_oracle_check(helper);save()
            for name,filename in prior['protocol']['cases']:
                path=ROOT/'data/prepared'/filename
                with np.load(path) as data:
                    X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
                kernel=SharedKernel(X,bw,Budget([X,b]));selected=kernel.farthest(592)[:568];K=kernel.block(X,X)
                R=np.setdiff1d(np.arange(len(b)),selected)
                C=K[np.ix_(selected,selected)]+1e-5*np.eye(568);B=K[np.ix_(selected,R)]
                W,spectrum=coupled_oracle(C,B,helper)
                spectrum.update(dataset=name);record['spectra'].append(spectrum)
                for mode in record['protocol']['modes']:
                    if process_time()-cpu_started>72:
                        row=dict(method=mode,status='global_cpu_not_started',converged=False)
                    else:
                        row=run(K,b,selected,helper,mode,oracle=W if mode.startswith('oracle_') else None,
                                deadline=cpu_started+77)
                    row.update(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest())
                    if mode in ['unchanged','matched_restart'] and row['converged']:
                        ref=next(x for x in prior['runs'] if x['dataset']==name and x['method']==mode)
                        assert row['full_solution_sha256']==ref['full_solution_sha256']
                        row['bitwise_matches_previous_control']=True
                    record['runs'].append(row);save()
                    print(json.dumps({k:row.get(k) for k in ['dataset','method','status','total_inner_iterations','correction_cycles']}),flush=True)
        except Exception as error:
            record['status']='failed';record['failures'].append(repr(error));save();raise
    record['status']='complete';save()


if __name__ == '__main__':
    main()

