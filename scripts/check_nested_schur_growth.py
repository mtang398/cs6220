#!/usr/bin/env python3
"""Bounded nested coordinate elimination; fixed total array allowance, not rank.

No n-by-m array is retained. All kernel entry queries, selection, reconstruction,
and true-residual checks are charged. The same-final-set control omits acquisition
and is an attribution oracle, not an end-to-end timing competitor.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.linalg.blas import dger
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CAP = 1024**2
RESERVE = 65536


class Kernel:
    def __init__(self, X, bandwidth):
        assert X.dtype == np.float64 and X.flags.c_contiguous
        self.X = X  # Common immutable input, no additional owned copy.
        self.n, self.d = X.shape
        self.bandwidth = bandwidth
        self.stats = dict(entry_calls=0, entry_evaluations=0, entry_seconds=0.)
        self.peak_named_allowance = 0
        self.minimum_block = self.n

    def base(self, m):
        # Includes current core, solver/index/selection vectors, and library and
        # Python allowance; feature packs and one tile are added separately.
        return 8*m*m+32*self.n*8+RESERVE

    def block_size(self, m, columns):
        available = CAP-self.base(m)-8*columns*self.d
        block = min(128, available//(8*(columns+self.d)))
        if block < 1:
            raise MemoryError('no kernel row fits the array allowance')
        self.minimum_block = min(self.minimum_block, int(block))
        return int(block)

    def entries(self, rows, cols, m=0, extra=0):
        started = perf_counter()
        named = self.base(m)+extra+8*(len(rows)*len(cols)+(len(rows)+len(cols))*self.d)
        self.peak_named_allowance = max(self.peak_named_allowance, named)
        assert named <= CAP, (named, len(rows), len(cols), m, extra)
        packed_rows = self.X[rows]
        packed_cols = self.X[cols]
        tile = cdist(packed_rows, packed_cols, metric='sqeuclidean')
        del packed_rows, packed_cols
        tile *= -.5/self.bandwidth**2
        np.exp(tile, out=tile)
        self.stats['entry_calls'] += 1
        self.stats['entry_evaluations'] += len(rows)*len(cols)
        self.stats['entry_seconds'] += perf_counter()-started
        return tile

    def full_A(self, v, lam, m):
        result = np.empty(self.n)
        columns = np.arange(self.n)
        block = self.block_size(m, self.n)
        for first in range(0, self.n, block):
            rows = columns[first:first+block]
            tile = self.entries(rows, columns, m)
            result[first:first+len(rows)] = tile@v
            del tile  # One tile is dead before the next entries allocation.
        result += lam*v
        return result


def fps(kernel, m):
    selected = np.empty(m, dtype=np.int64)
    similarity = np.zeros(kernel.n)
    rows = np.arange(kernel.n)
    j = 0
    for i in range(m):
        selected[i] = j
        column = kernel.entries(rows, selected[i:i+1]).reshape(kernel.n)
        np.maximum(similarity, column, out=similarity)
        del column
        similarity[selected[:i+1]] = np.inf
        j = int(np.argmin(similarity))
    return selected


class Schur:
    def __init__(self, kernel, selected, lam):
        self.kernel, self.lam = kernel, lam
        self.S = selected.copy()
        self.R = np.setdiff1d(np.arange(kernel.n), selected)
        self.m = len(selected)
        # C-order kernel tile's transpose is Fortran-contiguous; factor in place.
        core = kernel.entries(selected, selected)
        core.flat[::self.m+1] += lam
        L = linalg.cholesky(core.T, lower=True, overwrite_a=True, check_finite=False)
        assert np.shares_memory(L, core), 'Cholesky unexpectedly copied its core'
        self.L = L
        self.calls = 0

    def core_solve(self, rhs):
        return linalg.cho_solve((self.L, True), rhs, check_finite=False)

    def cross(self, v, transpose=False):
        rows, cols = (self.S, self.R) if transpose else (self.R, self.S)
        result = np.empty(len(rows))
        block = self.kernel.block_size(self.m, len(cols))
        for first in range(0, len(rows), block):
            rr = rows[first:first+block]
            tile = self.kernel.entries(rr, cols, self.m)
            result[first:first+len(rr)] = tile@v
            del tile
        return result

    def apply(self, v):
        all_values = np.empty(self.kernel.n)
        rows = np.arange(self.kernel.n)
        block = self.kernel.block_size(self.m, len(self.R))
        for first in range(0, len(rows), block):
            rr = rows[first:first+block]
            tile = self.kernel.entries(rr, self.R, self.m)
            all_values[first:first+len(rr)] = tile@v
            del tile
        q = self.core_solve(all_values[self.S])
        result = all_values[self.R]+self.lam*v
        del all_values
        result -= self.cross(q)
        self.calls += 1
        return result

    def reconstruct(self, b, tail):
        full = np.empty(self.kernel.n)
        full[self.R] = tail
        full[self.S] = self.core_solve(b[self.S]-self.cross(tail, transpose=True))
        return full


def choose_growth(kernel, old, pool, lam, residual, count, objective):
    """f/P or diagonal-pivot selection in the old-landmark Schur complement.

    One square array; BLAS updates its Fortran transpose in place. The supplied
    residual is either current full residual or initial b. Old S is eliminated
    first, including its residual update, then count new pivots are selected.
    """
    union = np.concatenate((old, pool))
    E = kernel.entries(union, union, len(old))
    E.flat[::len(union)+1] += lam
    q = residual[union].copy()
    allowed = np.ones(len(union), dtype=bool)
    chosen = []
    alias = True
    for step in range(len(old)+count):
        if step < len(old):
            j = step
        else:
            diagonal = E.diagonal()
            if objective == 'target':
                score = q*q/np.maximum(diagonal, np.finfo(float).tiny)
            else:
                score = diagonal.copy()
            score[~allowed] = -np.inf
            j = int(np.argmax(score))
            del score, diagonal
            chosen.append(int(union[j]))
        pivot = float(E[j, j])
        assert pivot > 0
        col = E[:, j].copy()
        q -= col*(q[j]/pivot)
        updated = dger(-1./pivot, col, col, a=E.T, overwrite_a=1)
        alias = alias and np.shares_memory(updated, E)
        assert alias, 'dger allocated a second square matrix'
        del updated, col
        allowed[j] = False
        E[j, :] = 0.
        E[:, j] = 0.
        q[j] = 0.
    return np.concatenate((old, np.asarray(chosen, dtype=np.int64))), dict(
        objective=objective, pool_size=len(pool), additions=count,
        dger_transpose_alias_verified=alias,
        selection_square_bytes=E.nbytes, selected_added=chosen)


def solve(kernel, b, lam, method, *, supplied=None, initial=64, final=192,
          pool_size=192, checkpoint=10, rtol=1e-8, maxiter=2500, cpu_cap=20.):
    started, cpu_started = perf_counter(), process_time()
    times = dict(selection=0., setup=0., iteration=0., reconstruction=0., verification=0., transition=0.)
    t = perf_counter()
    m = final if method == 'fps_final' else initial
    selected = supplied.copy() if supplied is not None else fps(kernel, m)
    initial_selected = selected.tolist()
    pool = None
    needs_pool = method in ('delayed_matrix', 'delayed_target', 'upfront_target')
    if needs_pool:
        eligible = np.setdiff1d(np.arange(kernel.n), selected)
        pool = np.random.default_rng(622091).choice(eligible, pool_size, replace=False)
        del eligible
    event = None
    if method == 'upfront_target':
        selected, event = choose_growth(kernel, selected, pool, lam, b, final-initial, 'target')
        del pool
        pool = None
    times['selection'] += perf_counter()-t
    t = perf_counter()
    op = Schur(kernel, selected, lam)
    rhs = b[op.R]-op.cross(op.core_solve(b[op.S]))
    tail = np.zeros(len(rhs)); r = rhs.copy(); p = r.copy()
    del rhs, selected
    rr = float(r@r)
    times['setup'] += perf_counter()-t
    norm_b = float(np.linalg.norm(b))
    checks = replacements = total_products = 0
    status = 'maxiter'; progress = []; iteration = 0
    transition_gap = None
    prefix_transition_seconds = None

    def verify():
        nonlocal checks
        t = perf_counter(); full = op.reconstruct(b, tail)
        times['reconstruction'] += perf_counter()-t
        t = perf_counter(); true = b-kernel.full_A(full, lam, op.m)
        times['verification'] += perf_counter()-t
        checks += 1
        return full, true

    for iteration in range(1, maxiter+1):
        if process_time()-cpu_started >= cpu_cap:
            status = 'cpu_screen_cap'; iteration -= 1; break
        t = perf_counter()
        ap = op.apply(p); total_products += 1
        curv = float(p@ap)
        if curv <= 0 or not np.isfinite(curv):
            status = 'breakdown'; break
        alpha = rr/curv
        tail += alpha*p; r -= alpha*ap
        del ap
        times['iteration'] += perf_counter()-t
        if iteration == checkpoint and method in ('restart_initial', 'delayed_matrix', 'delayed_target'):
            full, true = verify()
            if method == 'restart_initial':
                r = true[op.R].copy()
            else:
                t = perf_counter()
                new_selected, event = choose_growth(kernel, op.S, pool, lam, true,
                    final-initial, 'target' if method == 'delayed_target' else 'matrix')
                times['selection'] += perf_counter()-t
                del pool, op, p, r
                pool = None
                t = perf_counter()
                op = Schur(kernel, new_selected, lam)
                del new_selected
                tail = full[op.R].copy()
                # Reconstructing the new core is the exact correction on S_new.
                del full, true
                times['setup'] += perf_counter()-t
                full, true = verify()
                transition_gap = float(np.linalg.norm(true[op.S])/norm_b)
                r = true[op.R].copy()
            del full, true
            p = r.copy(); rr = float(r@r)
            prefix_transition_seconds = perf_counter()-started
            continue
        suspected = np.linalg.norm(r) <= rtol*norm_b
        if iteration % 20 == 0 or suspected or iteration == maxiter:
            full, true = verify()
            rel = float(np.linalg.norm(true)/norm_b)
            progress.append(dict(iteration=iteration, true_relative_residual=rel))
            if rel <= rtol:
                status = 'converged'; del full, true; break
            if suspected:
                r = true[op.R].copy(); p = r.copy(); rr = float(r@r)
                replacements += 1
                del full, true
                continue
            del full, true
        rr_new = float(r@r)
        p *= rr_new/rr; p += r; rr = rr_new
    full, true = verify()
    rel = float(np.linalg.norm(true)/norm_b)
    metrics = dict(method=method, status=status, converged=rel<=rtol,
        iterations=iteration, schur_products=total_products,
        true_relative_residual=rel,
        landmark_relative_residual=float(np.linalg.norm(true[op.S])/norm_b),
        transition_landmark_relative_residual=transition_gap,
        prefix_and_transition_seconds=prefix_transition_seconds,
        total_seconds=perf_counter()-started, total_cpu_seconds=process_time()-cpu_started,
        phase_seconds=times, original_verifications=checks,
        residual_replacements=replacements, initial_selected_indices=initial_selected,
        final_selected_indices=op.S.tolist(), growth=event,
        named_array_peak_allowance_bytes=int(kernel.peak_named_allowance),
        minimum_kernel_row_block=kernel.minimum_block, array_cap_bytes=CAP,
        common_immutable_input_bytes=kernel.X.nbytes+b.nbytes,
        retained_core_bytes=op.L.nbytes, kernel_stats=dict(kernel.stats), progress=progress,
        timing_role='attribution_oracle_acquisition_omitted' if method=='same_final_oracle' else 'end_to_end')
    return full, metrics


def independent_check():
    rng = np.random.default_rng(622092)
    n = 23
    Z = rng.normal(size=(n,n)); A = Z@Z.T+.7*np.eye(n)
    b = rng.normal(size=n); old = np.array([2,7,15]); new = np.array([2,7,15,4,10,19,21])
    class Dense(Kernel):
        def __init__(self): super().__init__(np.zeros((n,1)), 1.)
        def entries(self, rows, cols, m=0, extra=0): return A[np.ix_(rows,cols)].copy()
    kernel = Dense(); op = Schur(kernel, old, 0.)
    tail = rng.normal(size=len(op.R)); before = op.reconstruct(b, tail)
    residual = b-A@before
    op2 = Schur(kernel, new, 0.); after = op2.reconstruct(b, before[op2.R])
    reference = before.copy()
    reference[new] += linalg.solve(A[np.ix_(new,new)], residual[new], assume_a='pos')
    correction_error = float(np.linalg.norm(after-reference)/np.linalg.norm(reference))
    constraint_error = float(np.linalg.norm((b-A@after)[new])/np.linalg.norm(b))
    v = rng.normal(size=len(op.R))
    dense_S = A[np.ix_(op.R,op.R)]-A[np.ix_(op.R,old)]@linalg.solve(A[np.ix_(old,old)],A[np.ix_(old,op.R)],assume_a='pos')
    action_error = float(np.linalg.norm(op.apply(v)-dense_S@v)/np.linalg.norm(dense_S@v))
    full, result = solve(Dense(), b, 0., 'same_final_oracle', supplied=new,
        rtol=1e-11, maxiter=100, initial=len(new), final=len(new))
    exact = linalg.solve(A,b,assume_a='pos')
    solution_error = float(np.linalg.norm(full-exact)/np.linalg.norm(exact))
    pool = np.setdiff1d(np.arange(n),old)[:12]
    selected,event = choose_growth(Dense(),old,pool,0.,residual,4,'target')
    # Independent rebuilt Schur/residual score at every pivot checks f/P update.
    current = old.copy(); expected=[]
    for _ in range(4):
        remaining = np.setdiff1d(pool,current)
        C=A[np.ix_(remaining,current)]
        E=A[np.ix_(remaining,remaining)]-C@linalg.solve(A[np.ix_(current,current)],C.T,assume_a='pos')
        q=residual[remaining]-C@linalg.solve(A[np.ix_(current,current)],residual[current],assume_a='pos')
        j=int(remaining[np.argmax(q*q/E.diagonal())]);expected.append(j);current=np.append(current,j)
    assert event['selected_added']==expected
    assert max(correction_error,constraint_error,action_error,solution_error)<1e-9 and result['converged']
    return dict(constraint_correction_relative_error=correction_error,
        corrected_constraint_relative_residual=constraint_error,
        reduced_action_relative_error=action_error, full_solution_relative_error=solution_error,
        greedy_selection_matches_independent_rebuild=True,
        dger_transpose_alias_verified=event['dger_transpose_alias_verified'])


def main():
    output=ROOT/'results/nested-schur-growth-diagnostic.json'
    if output.exists(): raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(120,125))
    started,cpu_started=perf_counter(),process_time()
    result=dict(status='running',protocol=dict(seed=0,datasets=['smooth','mixed','Concrete'],
        regularizations=[.001,1e-5], initial_landmarks=64,final_landmarks=192,
        pool_size=192,pool_seed=622091,checkpoint=10,repeat=1,rtol=1e-8,maxiter=2500,
        per_solve_cpu_cap=20.,total_cpu_cap=120.,array_budget_bytes=CAP,
        immutable_input_scope='One shared read-only X and b, excluded from algorithm cap; no kernel.X copy.',
        array_accounting='m² core +32n vector/index allowance +64KiB library/Python reserve +one kernel tile +both packed feature arrays. Selection keeps one256² core plus old64² core; transpose dger and Cholesky aliases asserted.',
        timing_exclusions='Tracemalloc run and independent dense validation are separate; same-final oracle omits acquisition and is not a timing win.',
        source_method='Established Schur elimination/CG and f/P-greedy selection; no novelty claim.'),
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),runs=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,
            elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_check']=independent_check();save()
        for dataset,stem in [('smooth','synthetic-n640-smooth'),('mixed','synthetic-n640-mixed'),('Concrete','concrete-observed')]:
            path=ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:
                X=np.ascontiguousarray(data['X_train']);b=np.ascontiguousarray(data['y_train']);bandwidth=float(data['bandwidth'])
            X.setflags(write=False);b.setflags(write=False)
            for lam in [.001,1e-5]:
                final_set=None
                for method in ['fps_final','fixed_initial','restart_initial','delayed_matrix','upfront_target','delayed_target','same_final_oracle']:
                    if process_time()-cpu_started>110:
                        result['status']='total_cpu_budget';save();return
                    if method=='same_final_oracle' and final_set is None:continue
                    _,metrics=solve(Kernel(X,bandwidth),b,lam,method,
                        supplied=final_set if method=='same_final_oracle' else None,
                        cpu_cap=min(20.,max(1.,112-(process_time()-cpu_started))))
                    metrics.update(dataset=dataset,n=len(b),regularization=lam,
                        data_path=str(path.relative_to(ROOT)),data_sha256=sha256(path.read_bytes()).hexdigest())
                    if method=='delayed_target':final_set=np.asarray(metrics['final_selected_indices'],dtype=np.int64)
                    result['runs'].append(metrics);save()
                    print(json.dumps({k:metrics[k] for k in ['dataset','n','regularization','method','status','iterations','total_seconds','true_relative_residual','named_array_peak_allowance_bytes']}),flush=True)
                if dataset=='smooth' and lam==.001:
                    # Independent traced largest-core run; no traced native timing claim.
                    tracemalloc.start()
                    _,traced=solve(Kernel(X,bandwidth),b,lam,'fps_final',cpu_cap=10.)
                    current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                    result['allocation_audit']=dict(dataset=dataset,n=len(b),landmarks=192,
                        traced_peak_bytes=peak,traced_current_bytes=current,cap_bytes=CAP,
                        under_cap=peak<=CAP,status=traced['status'],timing_excluded=True,
                        source_bound_bytes=traced['named_array_peak_allowance_bytes'])
                    save()
                    if peak>CAP:
                        result['status']='allocation_audit_failed';save();return
    result['status']='complete';save()


if __name__=='__main__':main()
