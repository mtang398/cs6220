"""Bounded shared-kernel Schur-action feasibility; no full solver race.

All candidates use prefixes of the same FPS ordering and one maximum-size
Cholesky factor. Native observations are one CG step, not solve timings.
The parent protocol stops a proposed race only at true convergence, but no
race is run here. Common immutable X/b and independent dense validation
objects are outside the named candidate-array allowance.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.linalg.blas import dtrmv
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from check_schur_inplace_frontier_v2 import Budget, SharedKernel, InplaceSchur, LIMIT, RESERVE

ROOT = Path(__file__).resolve().parents[1]


class SharedAction:
    def __init__(self, kernel, maximum, regularization, indices=None):
        self.base = InplaceSchur(kernel, maximum, regularization, indices=indices)
        self.kernel, self.budget = kernel, kernel.budget
        self.n, self.M, self.lam = kernel.n, maximum, regularization
        self.L, self.S, self.R = self.base.L, self.base.S, self.base.R
        self.permutation = self.base.permutation
        self.aliases = []

    def core_product(self, vector):
        """Full F-contiguous factor only; never pass a strided leading block."""
        before = vector.__array_interface__['data'][0]
        first = dtrmv(self.L, vector, lower=1, trans=1, overwrite_x=1)
        second = dtrmv(self.L, first, lower=1, trans=0, overwrite_x=1)
        self.aliases.append(first.__array_interface__['data'][0] == before
                            and second.__array_interface__['data'][0] == before)
        self.budget.observe('full_core_triangular_product', vector, first, second,
                            temporary_allowance=4096)
        return second

    def leading_solve(self, vector, m):
        """In-place A[:m,:m] inverse, padded to M, using the full L."""
        if m == 0:
            vector.fill(0.)
            return vector
        vector[m:] = 0.
        before = vector.__array_interface__['data'][0]
        first = linalg.solve_triangular(self.L, vector, lower=True,
                                       overwrite_b=True, check_finite=False)
        first[m:] = 0.
        second = linalg.solve_triangular(self.L, first, lower=True, trans='T',
                                        overwrite_b=True, check_finite=False)
        self.aliases.append(first.__array_interface__['data'][0] == before
                            and second.__array_interface__['data'][0] == before)
        self.budget.observe('full_factor_padded_leading_solve', vector, first, second,
                            temporary_allowance=4096)
        return second

    def product(self, rows, columns, vectors, target, label, *, add=False):
        """One live kernel tile, explicit row/column feature packs, q RHS."""
        X = self.kernel.X
        packed_columns = X if columns is None else X[columns]
        width = vectors.shape[1]
        count = len(rows)
        # Account q output scalars per tile row, not the scalar helper's2.
        available = LIMIT - RESERVE - self.budget.size(
            packed_columns, vectors, target) - 4096
        per_row = 8 * (len(packed_columns) + X.shape[1] + width)
        tile_rows = min(128, int(available // per_row))
        if tile_rows < 1:
            raise MemoryError(f'No shared row tile fits at {label}')
        self.budget.row_sizes.setdefault(label, set()).add(tile_rows)
        for start in range(0, count, tile_rows):
            stop = min(count, start + tile_rows)
            points = X[rows[start:stop]]
            tile = self.kernel.block(points, packed_columns)
            values = tile @ vectors
            self.budget.observe(label, packed_columns, vectors, target,
                                points, tile, values)
            if add:
                target[start:stop] += values
            else:
                target[start:stop] = values
            del points, tile, values
        del packed_columns

    def apply(self, ms, directions):
        n, M, q = self.n, self.M, len(ms)
        state_live = self.budget.live.copy()
        Y = np.empty((n, q), order='F')
        Q = np.zeros((M, q), order='F')
        V = np.empty((n-M, q), order='F')
        self.budget.live = [*state_live, Y, Q, V]
        for j, (m, p) in enumerate(zip(ms, directions)):
            work = np.zeros(M)
            work[m:] = p[:M-m]
            Y[:M, j] = self.core_product(work) if m < M else 0.
            del work
            V[:, j] = p[M-m:]
        self.product(self.S, self.R, V, Y[:M], 'shared_top_cross', add=True)
        del V
        self.budget.live = [*state_live, Y, Q]
        for j, m in enumerate(ms):
            Q[:m, j] = Y[:m, j]
            self.leading_solve(Q[:, j], m)
            if 0 < m < M:
                work = Q[:, j].copy()
                Y[:M, j] -= self.core_product(work)
                del work
            Y[:m, j] = 0.
        D = np.zeros((n, q), order='F')
        self.budget.live = [*state_live, Y, Q, D]
        for j, (m, p) in enumerate(zip(ms, directions)):
            D[self.permutation[m:], j] = p
            D[self.S, j] -= Q[:, j]
        self.product(self.R, None, D, Y[M:], 'shared_bottom_lifted')
        for j, (m, p) in enumerate(zip(ms, directions)):
            # No extra lambda for top rows: L L.T already includes it.
            Y[M:, j] += self.lam * p[M-m:]
        self.budget.observe('shared_final_arrays', Y, Q, D,
                            temporary_allowance=8*(n-M))
        del Q, D
        self.budget.live = state_live
        return [Y[m:, j] for j, m in enumerate(ms)]

    def rhs(self, ms, b):
        n, M, q = self.n, self.M, len(ms)
        state_live = self.budget.live.copy()
        Q = np.zeros((M, q), order='F')
        Y = np.empty((n, q), order='F')
        self.budget.live = [*state_live, Q, Y]
        for j, m in enumerate(ms):
            Q[:m, j] = b[self.S[:m]]
            self.leading_solve(Q[:, j], m)
            work = Q[:, j].copy()
            Y[:M, j] = b[self.S] - self.core_product(work)
            del work
        self.product(self.R, self.S, Q, Y[M:], 'shared_initial_rhs')
        Y[M:] *= -1.
        for j in range(q):
            Y[M:, j] += b[self.R]
        result = [Y[m:, j].copy() for j, m in enumerate(ms)]
        self.budget.observe('initial_rhs_copy', Q, Y, *result,
                            temporary_allowance=8*n)
        del Q, Y
        self.budget.live = state_live
        return result


def dense_validation():
    rng = np.random.default_rng(62201124)
    records = []
    # Non-kernel SPD verifies padded leading solve independently.
    G = rng.normal(size=(19, 19))
    A = G @ G.T + .07*np.eye(19)
    L = linalg.cholesky(A, lower=True)
    errors, tail_errors, aliases = [], [], []
    for m in [1, 7, 18, 19]:
        v = rng.normal(size=m); padded = np.zeros(19); padded[:m] = v
        first = linalg.solve_triangular(L, padded, lower=True, overwrite_b=True)
        first[m:] = 0.
        second = linalg.solve_triangular(L, first, lower=True, trans='T', overwrite_b=True)
        errors.append(float(np.linalg.norm(second[:m]-linalg.solve(A[:m,:m],v)) / np.linalg.norm(second[:m])))
        tail_errors.append(float(np.linalg.norm(second[m:])))
        aliases.append(bool(np.shares_memory(first,padded) and np.shares_memory(second,padded)))
    assert max(errors) < 1e-12 and max(tail_errors) == 0. and all(aliases)
    for lam in [.001, 1e-5]:
        X = np.ascontiguousarray(rng.normal(size=(41, 5))); b = rng.normal(size=41)
        bw = 1.9; M = 18; ms = [0,12,18]
        dense = np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw)) + lam*np.eye(len(X))
        budget = Budget([X,b]); kernel = SharedKernel(X,bw,budget)
        op = SharedAction(kernel,M,lam)
        ordered = dense[np.ix_(op.permutation,op.permutation)]
        Ds, rhs_ref = [], []
        for m in ms:
            if m:
                cross = linalg.solve(ordered[:m,:m],ordered[:m,m:],assume_a='pos')
                D = ordered[m:,m:] - ordered[m:,:m]@cross
                rhs = b[op.permutation[m:]] - ordered[m:,:m]@linalg.solve(
                    ordered[:m,:m],b[op.permutation[:m]],assume_a='pos')
            else:
                D = ordered.copy(); rhs = b[op.permutation].copy()
            Ds.append(D); rhs_ref.append(rhs)
        rhs = op.rhs(ms,b)
        rhs_error = max(np.linalg.norm(a-c)/np.linalg.norm(c) for a,c in zip(rhs,rhs_ref))
        ps = [rng.normal(size=41-m) for m in ms]
        before = kernel.stats['entry_evaluations']; actual = op.apply(ms,ps)
        entries = kernel.stats['entry_evaluations']-before
        action_error = max(np.linalg.norm(y-D@p)/np.linalg.norm(D@p) for y,D,p in zip(actual,Ds,ps))
        assert entries == 41**2-M**2 and action_error < 1e-11 and rhs_error < 1e-11
        del actual
        xs = [np.zeros_like(v) for v in rhs]; rs=[v.copy() for v in rhs]; ps=[v.copy() for v in rhs]
        xd=[v.copy() for v in xs]; rd=[v.copy() for v in rhs_ref]; pd=[v.copy() for v in rhs_ref]
        prefixes=[]
        for iteration in range(8):
            budget.live=[*xs,*rs,*ps]
            aps=op.apply(ms,ps)
            for j in range(len(ms)):
                rr=rs[j]@rs[j]; alpha=rr/(ps[j]@aps[j])
                xs[j]+=alpha*ps[j];rs[j]-=alpha*aps[j]
                ps[j]*=(rs[j]@rs[j])/rr;ps[j]+=rs[j]
                ad=Ds[j]@pd[j]; rr=rd[j]@rd[j];alpha=rr/(pd[j]@ad)
                xd[j]+=alpha*pd[j];rd[j]-=alpha*ad
                pd[j]*=(rd[j]@rd[j])/rr;pd[j]+=rd[j]
            prefixes.append(max(float(np.linalg.norm(a-c)/max(1.,np.linalg.norm(c))) for a,c in zip(xs,xd)))
            del aps
        assert max(prefixes)<1e-9 and all(op.aliases)
        records.append(dict(regularization=lam,action_relative_error=float(action_error),
            reduced_rhs_relative_error=float(rhs_error),prefix8_relative_errors=prefixes,
            queried_entries_per_joint_action=entries,formula_entries=41**2-M**2,
            all_full_factor_BLAS_and_solve_buffers_alias=all(op.aliases)))
    return dict(generic_SPD_padded_solve_relative_errors=errors,padded_tail_norms=tail_errors,
                padded_buffer_aliases=aliases,kernel_checks=records)


def state_vectors(rhs):
    return [np.zeros_like(v) for v in rhs], rhs, [v.copy() for v in rhs]


def cg_update(x,r,p,ap):
    rr=float(r@r);alpha=rr/float(p@ap)
    x+=alpha*p;r-=alpha*ap
    p*=float(r@r)/rr;p+=r


def observation(X,b,bw,lam,ms,mode,trace=False):
    if trace:tracemalloc.start()
    wall,cpu=perf_counter(),process_time()
    budget=Budget([X,b]);kernel=SharedKernel(X,bw,budget)
    natural=mode=='natural'
    if natural:
        m=ms[0]
        op=InplaceSchur(kernel,m,lam) if m else None
        rhs=[op.reduced_rhs(b)] if m else [b.copy()]
    else:
        op=SharedAction(kernel,320,lam);rhs=op.rhs(ms,b)
    x,r,p=state_vectors(rhs);del rhs
    budget.live=[*x,*r,*p]
    setup_seconds=perf_counter()-wall;setup_cpu=process_time()-cpu
    before=dict(kernel.stats);step_wall,step_cpu=perf_counter(),process_time()
    if mode=='independent_shared_representation':
        for j,m in enumerate(ms):
            ap=op.apply([m],[p[j]])[0]
            budget.live=[*x,*r,*p,ap]
            budget.observe('CG_scalar_update',temporary_allowance=2*p[j].nbytes)
            cg_update(x[j],r[j],p[j],ap);del ap
            budget.live=[*x,*r,*p]
    else:
        if natural:
            ap=op.apply(p[0]) if op else kernel.full(p[0])
            if not op:ap+=lam*p[0]
            aps=[ap]
        else:aps=op.apply(ms,p)
        budget.live=[*x,*r,*p,*aps]
        for j,ap in enumerate(aps):
            budget.observe('CG_scalar_update',temporary_allowance=2*p[j].nbytes)
            cg_update(x[j],r[j],p[j],ap)
        del ap,aps
        budget.live=[*x,*r,*p]
    step_seconds=perf_counter()-step_wall;step_cpu=process_time()-step_cpu
    row=dict(mode=mode,candidate_sizes=ms,maximum_retained_core=(ms[0] if natural else 320),
        n=len(b),regularization=lam,setup_seconds=setup_seconds,setup_cpu_seconds=setup_cpu,
        step_seconds=step_seconds,step_cpu_seconds=step_cpu,
        step_kernel_stats={k:kernel.stats[k]-before[k] for k in before},
        setup_kernel_stats=before,memory=budget.report(),
        common_immutable_input_bytes=X.nbytes+b.nbytes,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        all_core_buffer_aliases=(all(op.aliases) if not natural else
            all(op.solve_alias_checks) if op else True),
        factor_buffer_alias=(op.base.factor_alias if not natural else op.factor_alias if op else True),
        states_after_step=[dict(x_norm=float(np.linalg.norm(a)),r_norm=float(np.linalg.norm(c))) for a,c in zip(x,r)])
    if op:
        inds=op.S if natural else op.base.S
        row['selected_indices_sha256']=sha256(inds.tobytes()).hexdigest()
    if trace:
        current,peak=tracemalloc.get_traced_memory()
        factor_trace=op.base.factor_trace if not natural else op.factor_trace if op else None
        if factor_trace:peak=max(peak,factor_trace['earlier_peak'])
        tracemalloc.stop()
        row.update(tracemalloc_peak_bytes=peak,tracemalloc_current_bytes=current,
                   trace_within_1MiB=peak<=LIMIT,untimed_allocation_audit=True)
    return row


def main():
    output=ROOT/'results/shared-schur-action-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(10,12))
    wall,cpu=perf_counter(),process_time()
    source=Path(__file__).resolve();dependency=source.with_name('check_schur_inplace_frontier_v2.py')
    record=dict(status='running',protocol=dict(
        purpose='Tiny algebra/trajectory validation and native one-step cost; no full race',
        maximum_core=320,candidate_pairs=[[256,320],[0,320]],native_regularization=.001,
        datasets=['synthetic-n640-smooth','concrete-observed'],
        candidate_states='Independent ordinary reduced-system CG, no cross-trajectory directions or restart',
        future_stopping_rule='First original-system TRUE-converged branch wins; no residual forecast pruning',
        array_budget_bytes=LIMIT,reserve_bytes=RESERVE,blas_threads=1,
        immutable_inputs='Original X/b shared; all temporary feature packs counted',
        maximum_core_setup_charged=True,lock='/tmp/cs6220-project-benchmark.lock',
        scope_limit='No solve-level CPU, convergence, feedback benefit, or novelty claim; full verification schedule is not benchmarked'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [source,dependency]},
        observations=[],allocation_audits=[])
    def save():
        record.update(elapsed_cpu_seconds=process_time()-cpu,elapsed_wall_seconds=perf_counter()-wall)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_validation']=dense_validation();save()
        for name in record['protocol']['datasets']:
            path=ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data=np.load(path);X,b=data['X_train'],data['y_train'];bw=float(data['bandwidth'])
            cases=[([256,320],'joint'),([256,320],'independent_shared_representation'),
                   ([0,320],'joint'),([0,320],'independent_shared_representation'),
                   ([0],'shared_representation_singleton'),([256],'shared_representation_singleton'),
                   ([320],'shared_representation_singleton'),([0],'natural'),([256],'natural'),([320],'natural')]
            for ms,mode in cases:
                try:
                    row=observation(X,b,bw,.001,ms,mode)
                    row.update(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest())
                except (MemoryError,np.linalg.LinAlgError) as error:
                    row=dict(dataset=name,candidate_sizes=ms,mode=mode,status='failure',error=str(error))
                record['observations'].append(row);save()
            for ms in [[256,320],[0,320]]:
                try:
                    row=observation(X,b,bw,.001,ms,'joint',trace=True)
                    row.update(dataset=name)
                except (MemoryError,np.linalg.LinAlgError) as error:
                    if tracemalloc.is_tracing():tracemalloc.stop()
                    row=dict(dataset=name,candidate_sizes=ms,status='failure',error=str(error))
                record['allocation_audits'].append(row);save()
    record['status']='complete';save()
    print(json.dumps(dict(output=str(output),elapsed_cpu_seconds=record['elapsed_cpu_seconds'],
        observations=len(record['observations']),allocation_audits=len(record['allocation_audits']))))


if __name__=='__main__':main()
