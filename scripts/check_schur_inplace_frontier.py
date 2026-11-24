"""Static Schur memory frontier with alias-verified in-place Cholesky.

No immutable training-feature copy, no retained cross columns, and no pair of
simultaneously live kernel tiles. A runtime allocator chooses the largest row
tile fitting named arrays plus a64KiB reserve. tracemalloc runs are separate
from timings. This is an established stronger static control, not a new method.
"""
from hashlib import sha256
import inspect
import json
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import tracemalloc

import numpy as np
import scipy
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
LIMIT=1024*1024
RESERVE=65536


def array_bytes(*arrays):
    roots={}
    for a in arrays:
        if a is None:continue
        while isinstance(a.base,np.ndarray):a=a.base
        roots[id(a)]=a.nbytes
    return int(sum(roots.values()))


class Budget:
    def __init__(self,common):
        self.common={id(a) for a in common}
        self.fixed=[];self.live=[];self.peaks={};self.row_sizes={}
    def size(self,*extra):
        arrays=[]
        for a in [*self.fixed,*self.live,*extra]:
            if a is None:continue
            while isinstance(a.base,np.ndarray):a=a.base
            if id(a) not in self.common:arrays.append(a)
        return array_bytes(*arrays)
    def observe(self,label,*extra,temporary_allowance=0):
        value=self.size(*extra)+temporary_allowance
        self.peaks[label]=max(value,self.peaks.get(label,0))
        if value+RESERVE>LIMIT:raise MemoryError(f'{label}: {value}+{RESERVE}>1MiB')
    def rows(self,label,columns,dimension,*extra,features=False):
        available=LIMIT-RESERVE-self.size(*extra)-4096
        per_row=8*(columns+(dimension if features else 0)+2)
        rows=min(128,int(available//per_row))
        if rows<1:raise MemoryError(f'No row tile fits at {label}')
        self.row_sizes.setdefault(label,set()).add(rows)
        return rows
    def report(self):
        return dict(named_peak_bytes=max(self.peaks.values(),default=0),
            peak_with_reserve_bytes=max(self.peaks.values(),default=0)+RESERVE,
            reserve_bytes=RESERVE,phase_named_peaks=self.peaks,
            phase_row_sizes={k:sorted(v) for k,v in self.row_sizes.items()})


class SharedKernel:
    def __init__(self,X,bandwidth,budget):
        if X.dtype!=np.float64 or not X.flags.c_contiguous:
            raise ValueError('Frozen common X must already be C-contiguous float64; no implicit copy')
        self.X=X;self.n=len(X);self.d=X.shape[1];self.bandwidth=bandwidth;self.budget=budget
        self.stats=dict(column_queries=0,entry_evaluations=0,entry_blocks=0,full_matvecs=0)
    def block(self,rows,columns):
        values=cdist(rows,columns,'sqeuclidean')
        values/=self.bandwidth;values/=self.bandwidth;values*=-.5
        np.exp(values,out=values)
        self.stats['entry_evaluations']+=values.size;self.stats['entry_blocks']+=1
        return values
    def farthest(self,m):
        selected=np.empty(m,dtype=np.int64);similarity=np.zeros(self.n);j=0
        for count in range(m):
            selected[count]=j
            column=self.block(self.X,self.X[j:j+1]).reshape(self.n)
            self.stats['column_queries']+=1
            self.budget.observe('selection',selected,similarity,column)
            np.maximum(similarity,column,out=similarity)
            similarity[selected[:count+1]]=np.inf
            j=int(np.argmin(similarity))
            del column
        return selected
    def product(self,row_indices,column_indices,vector,*,label,extra=(),target=None,subtract=False):
        row_count=self.n if row_indices is None else len(row_indices)
        col_count=self.n if column_indices is None else len(column_indices)
        columns=self.X if column_indices is None else self.X[column_indices]
        result=np.empty(row_count) if target is None else target
        block_rows=self.budget.rows(label,col_count,self.d,*extra,columns,result,vector,
            features=row_indices is not None)
        for start in range(0,row_count,block_rows):
            stop=min(row_count,start+block_rows)
            points=(self.X[start:stop] if row_indices is None else self.X[row_indices[start:stop]])
            tile=self.block(points,columns)
            product=tile@vector
            self.budget.observe(label,*extra,columns,result,vector,points,tile,product)
            if subtract:result[start:stop]-=product
            else:result[start:stop]=product
            del tile,product,points
        del columns
        return result
    def full(self,vector,*,extra=()):
        value=self.product(None,None,vector,label='full_verification',extra=extra)
        self.stats['full_matvecs']+=1
        return value


class InplaceSchur:
    def __init__(self,kernel,m,lam,*,indices=None):
        self.kernel=kernel;self.budget=kernel.budget;self.lam=lam;self.m=m;self.n=kernel.n
        selected=kernel.farthest(m) if indices is None else np.array(indices,dtype=np.int64,copy=True)
        mask=np.ones(self.n,dtype=bool);mask[selected]=False
        remaining=np.flatnonzero(mask)
        self.permutation=np.r_[selected,remaining]
        del selected,remaining,mask
        self.S=self.permutation[:m];self.R=self.permutation[m:]
        core=np.empty((m,m),dtype=np.float64,order='F')
        self.budget.fixed=[self.permutation,core]
        columns=kernel.X[self.S]
        row_count=self.budget.rows('core_assembly',m,kernel.d,columns)
        for start in range(0,m,row_count):
            stop=min(m,start+row_count)
            tile=kernel.block(columns[start:stop],columns)
            self.budget.observe('core_assembly',columns,tile)
            core[start:stop,:]=tile
            del tile
        del columns
        core.flat[::m+1]+=lam
        pointer=core.__array_interface__['data'][0]
        self.factor_trace=None
        if tracemalloc.is_tracing():
            current,prior_peak=tracemalloc.get_traced_memory();tracemalloc.reset_peak()
        self.L=linalg.cholesky(core,lower=True,overwrite_a=True,check_finite=False)
        self.factor_alias=bool(np.shares_memory(self.L,core) and self.L.__array_interface__['data'][0]==pointer)
        self.factor_same_object=self.L is core
        if not self.factor_alias:raise MemoryError('SciPy copied the Fortran core despite overwrite_a=True')
        self.budget.observe('inplace_cholesky',core,self.L,temporary_allowance=4096)
        if tracemalloc.is_tracing():
            after,peak=tracemalloc.get_traced_memory()
            self.factor_trace=dict(traced_bytes_before=current,traced_bytes_after=after,
                peak_extra_during_cholesky=max(0,peak-current),earlier_peak=prior_peak,
                traced_peak_after_reset=peak)
        self.budget.fixed=[self.permutation,self.L]
        del core
        self.calls=0;self.solve_alias_checks=[]
    @property
    def storage_bytes(self):return self.permutation.nbytes+self.L.nbytes
    def core_solve(self,rhs,*,extra=()):
        pointer=rhs.__array_interface__['data'][0]
        solution=linalg.cho_solve((self.L,True),rhs,overwrite_b=True,check_finite=False)
        self.solve_alias_checks.append(bool(np.shares_memory(solution,rhs) and solution.__array_interface__['data'][0]==pointer))
        self.budget.observe('core_rhs_solve',rhs,solution,*extra,temporary_allowance=4096)
        return solution
    def reduced_rhs(self,b):
        top=self.core_solve(b[self.S])
        rhs=b[self.R]
        self.kernel.product(self.R,self.S,top,label='reduced_rhs',extra=(top,rhs),target=rhs,subtract=True)
        return rhs
    def apply(self,v):
        all_values=self.kernel.product(None,self.R,v,label='Schur_first_pass')
        top=self.core_solve(all_values[self.S],extra=(all_values,v))
        result=all_values[self.R]
        self.budget.observe('Schur_vector_update',all_values,top,result,temporary_allowance=result.nbytes)
        result+=self.lam*v
        self.kernel.product(self.R,self.S,top,label='Schur_second_pass',extra=(all_values,top,result),target=result,subtract=True)
        self.calls+=1
        return result
    def reconstruct(self,b,tail):
        top=b[self.S]
        self.kernel.product(self.S,self.R,tail,label='reconstruct_cross',extra=(top,tail),target=top,subtract=True)
        top=self.core_solve(top,extra=(tail,))
        full=np.empty(self.n);full[self.S]=top;full[self.R]=tail
        self.budget.observe('reconstruct_vector',top,tail,full)
        return full


def solve(X,b,bandwidth,m,lam,*,rtol=1e-8,maxiter=2500,cpu_cap=10.,deadline=float('inf'),indices=None):
    started,cpu_start=perf_counter(),process_time()
    budget=Budget([X,b]);kernel=SharedKernel(X,bandwidth,budget)
    op=InplaceSchur(kernel,m,lam,indices=indices)
    rhs=op.reduced_rhs(b)
    x=np.zeros_like(rhs);r=rhs.copy();p=r.copy();rr=float(r@r)
    norm_b=np.linalg.norm(b);checks=0;replacements=0;last_check=-1;progress=[]
    status='maxiter';relative=None;setup=perf_counter()-started
    def verify(iteration):
        nonlocal checks,last_check,relative
        budget.live=[rhs,x,r,p]
        full=op.reconstruct(b,x)
        ax=kernel.full(full,extra=(full,))
        # Reuse the product buffer for b-Ax; no duplicate full residual array.
        ax*=-1.;ax+=b;ax-=lam*full
        relative=float(np.linalg.norm(ax)/norm_b)
        checks+=1;last_check=iteration
        progress.append(dict(iteration=iteration,true_relative_residual=relative,
            landmark_relative_residual=float(np.linalg.norm(ax[op.S])/norm_b)))
        return full,ax
    for iteration in range(1,maxiter+1):
        if process_time()-cpu_start>cpu_cap or process_time()>deadline:
            status='cpu_screen_cap';iteration-=1;break
        budget.live=[rhs,x,r,p]
        ap=op.apply(p);curvature=float(p@ap)
        if curvature<=0 or not np.isfinite(curvature):status='breakdown';break
        alpha=rr/curvature;x+=alpha*p;r-=alpha*ap
        del ap
        suspected=np.linalg.norm(r)<=rtol*norm_b
        if iteration%20==0 or suspected or iteration==maxiter:
            full,true=verify(iteration)
            if relative<=rtol:status='converged';break
            if suspected:
                r=true[op.R];p=r.copy();rr=float(r@r);replacements+=1
                del full,true
                continue
            del full,true
        next_rr=float(r@r);p=r+(next_rr/rr)*p;rr=next_rr
    if last_check!=iteration:full,true=verify(iteration)
    else:
        # full and true still exist only when a successful verification broke.
        if status!='converged':full=op.reconstruct(b,x)
    return full,dict(status=status,converged=relative<=rtol,iterations=iteration,
        true_relative_residual=relative,setup_seconds=setup,total_seconds=perf_counter()-started,
        total_cpu_seconds=process_time()-cpu_start,schur_products=op.calls,original_verifications=checks,
        residual_replacements=replacements,retained_factor_and_indices_bytes=op.storage_bytes,
        factor_alias_verified=op.factor_alias,factor_same_object=op.factor_same_object,
        all_core_rhs_solves_alias=all(op.solve_alias_checks),
        indices=op.S.tolist(),index_sha256=sha256(op.S.tobytes()).hexdigest(),
        memory=budget.report(),kernel_stats=kernel.stats,progress=progress,
        common_immutable_input_bytes=X.nbytes+b.nbytes,kernel_X_is_common_input=kernel.X is X,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def tiny_check():
    rng=np.random.default_rng(622098);X=np.ascontiguousarray(rng.normal(size=(31,4)));b=rng.normal(size=31)
    lam=.03;bandwidth=1.7;S=np.array([7,1,18,23,4,28,11])
    A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bandwidth**2))+lam*np.eye(len(X))
    budget=Budget([X,b]);kernel=SharedKernel(X,bandwidth,budget);op=InplaceSchur(kernel,len(S),lam,indices=S)
    D=A[np.ix_(op.R,op.R)]-A[np.ix_(op.R,S)]@linalg.solve(A[np.ix_(S,S)],A[np.ix_(S,op.R)],assume_a='pos')
    v=rng.normal(size=len(op.R));observed=op.apply(v)
    action_error=float(np.linalg.norm(observed-D@v)/np.linalg.norm(D@v))
    factor_error=float(np.linalg.norm(op.L@op.L.T-A[np.ix_(S,S)])/np.linalg.norm(A[np.ix_(S,S)]))
    full,run=solve(X,b,bandwidth,len(S),lam,rtol=1e-11,maxiter=300,indices=S)
    exact=linalg.solve(A,b,assume_a='pos')
    solution_error=float(np.linalg.norm(full-exact)/np.linalg.norm(exact))
    assert op.factor_alias and run['all_core_rhs_solves_alias']
    assert action_error<1e-12 and factor_error<1e-12 and solution_error<1e-9 and run['converged']
    return dict(action_relative_error=action_error,factor_relative_error=factor_error,
        solution_relative_error=solution_error,factor_alias=op.factor_alias,
        factor_same_object=op.factor_same_object,core_rhs_alias=run['all_core_rhs_solves_alias'])


def allocation_audit(X,b,bandwidth,m,lam):
    """Untimed tracemalloc path: setup, one action and one full verification."""
    tracemalloc.start()
    budget=Budget([X,b]);kernel=SharedKernel(X,bandwidth,budget)
    op=InplaceSchur(kernel,m,lam)
    rhs=op.reduced_rhs(b);x=np.zeros_like(rhs);r=rhs.copy();p=r.copy()
    budget.live=[rhs,x,r,p]
    ap=op.apply(p)
    del ap
    full=op.reconstruct(b,x);ax=kernel.full(full,extra=(full,))
    ax*=-1;ax+=b;ax-=lam*full
    current,peak=tracemalloc.get_traced_memory()
    peak=max(peak,op.factor_trace['earlier_peak'])
    tracemalloc.stop()
    result=dict(named=budget.report(),tracemalloc_current_bytes=current,tracemalloc_peak_bytes=peak,
        factor_trace=op.factor_trace,factor_alias=op.factor_alias,
        factor_same_object=op.factor_same_object,core_rhs_alias=all(op.solve_alias_checks),
        common_input_excluded_bytes=X.nbytes+b.nbytes,
        trace_within_1MiB=peak<=LIMIT,
        audit_scope='Separate untimed setup+one Schur product+one reconstruction/full verification with steady CG vectors; common X/b allocated before trace. Native allocator/library allocations outside tracemalloc are not claimed measured.')
    return result


def main():
    output=ROOT/'results/schur-inplace-frontier-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(58,60))
    started,cpu_start=perf_counter(),process_time()
    from scipy.linalg import _decomp_cholesky
    scipy_source=inspect.getsource(_decomp_cholesky._cholesky)
    record=dict(status='running',scope='Established stronger static Schur controls and memory-frontier audit; no adaptive/goal-success claim',
        protocol=dict(datasets=['synthetic-n640-smooth','concrete-observed'],regularizations=[.001,1e-5],
            landmark_counts=[192,256,320],selection='farthest point from index0, recomputed/charged for every solve',
            array_budget_bytes=LIMIT,reserve_bytes=RESERVE,row_limit=128,
            tile_selection='largest integer row count allowed by phase-specific named live arrays plus64KiB reserve and4KiB tile-expression guard; no timing-based tuning',
            factorization='Fortran core assembled by row blocks; scipy.linalg.cholesky lower=True overwrite_a=True check_finite=False; pointer and sharing verified',
            immutable_inputs='X and b shared without a full training-feature copy; temporary selected/remaining feature packs counted',
            rtol=1e-8,maxiter=2500,per_solve_cpu_cap=10.,global_cpu_cap=60,blas_threads=1,
            timing='single native observation each; allocation trace is a separate untimed path',
            lock='/tmp/cs6220-project-benchmark.lock'),
        scipy_version=scipy.__version__,scipy_cholesky_source_sha256=sha256(scipy_source.encode()).hexdigest(),
        source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__).read_bytes()).hexdigest()},
        runs=[],allocation_audits=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_tiny_check']=tiny_check();save()
        for name in record['protocol']['datasets']:
            path=ROOT/f'data/prepared/{name}-s0-lambda0.001.npz';data=np.load(path)
            X,b=data['X_train'],data['y_train'];bandwidth=float(data['bandwidth'])
            for lam in [.001,1e-5]:
                for m in [192,256,320]:
                    if process_time()-cpu_start>52:record['status']='cpu_budget';save();return
                    try:
                        full,run=solve(X,b,bandwidth,m,lam,deadline=cpu_start+55.)
                        run.update(dataset=name,n=len(b),regularization=lam,m=m,data_sha256=sha256(path.read_bytes()).hexdigest())
                        record['runs'].append(run);save()
                        print(json.dumps({k:run[k] for k in ['dataset','n','m','regularization','status','iterations','total_seconds','true_relative_residual']}),flush=True)
                    except (MemoryError,np.linalg.LinAlgError) as error:
                        record['runs'].append(dict(dataset=name,n=len(b),m=m,regularization=lam,status='failure',error=str(error)));save()
                        continue
                    del full
                    # Trace both shifts; the audit is never included in native
                    # timings and is reported separately for every case.
                    trace_start=perf_counter();trace_cpu=process_time()
                    audit=allocation_audit(X,b,bandwidth,m,lam)
                    audit.update(dataset=name,n=len(b),m=m,regularization=lam,
                        audit_wall_seconds=perf_counter()-trace_start,audit_cpu_seconds=process_time()-trace_cpu)
                    run['resource_validated_by_separate_trace']=audit['trace_within_1MiB']
                    run['tracemalloc_peak_bytes']=audit['tracemalloc_peak_bytes']
                    run['resource_status']='within_declared_cap' if audit['trace_within_1MiB'] else 'allocation_trace_exceeds_cap'
                    record['allocation_audits'].append(audit);save()
    record['status']='complete';save()


if __name__=='__main__':main()
