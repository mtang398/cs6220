#!/usr/bin/env python3
"""Tiny packed-LAPACK feasibility/allocation check; no full solve benchmark.

Established packed Cholesky representation. Existing measured sources remain
unchanged. Numerical checks use independent dense references; allocation paths
perform setup, one Schur action, reconstruction and one full verification only.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import resource
from time import perf_counter,process_time
import tracemalloc

import numpy as np
import scipy
from scipy import linalg
from scipy.linalg import blas,lapack
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/check_schur_inplace_frontier_v2.py'
spec=importlib.util.spec_from_file_location('packed_audited_base',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class PackedSchur(base.InplaceSchur):
    def __init__(self,kernel,m,lam,*,indices,lower=True):
        self.kernel=kernel;self.budget=kernel.budget;self.lam=lam;self.m=m;self.n=kernel.n
        self.lower=bool(lower)
        selected=np.asarray(indices,dtype=np.int64)
        mask=np.ones(self.n,dtype=bool);mask[selected]=False
        remaining=np.flatnonzero(mask)
        self.permutation=np.r_[selected,remaining]
        del remaining,mask
        self.S=self.permutation[:m];self.R=self.permutation[m:]
        packed=np.empty(m*(m+1)//2,dtype=np.float64)
        self.budget.fixed=[self.permutation,packed]
        columns=kernel.X[self.S]
        offset=0
        for j in range(m):
            points=columns[j:] if lower else columns[:j+1]
            tile=kernel.block(points,columns[j:j+1])
            self.budget.observe('packed_core_assembly',selected,columns,points,tile)
            width=len(points)
            packed[offset:offset+width]=tile[:,0]
            packed[offset if lower else offset+width-1]+=lam
            offset+=width
            del tile,points
        del columns
        before=packed.__array_interface__['data'][0]
        if tracemalloc.is_tracing():
            current,earlier_peak=tracemalloc.get_traced_memory();tracemalloc.reset_peak()
        self.L,info=lapack.dpptrf(m,packed,lower=int(lower),overwrite_ap=1)
        if info:raise np.linalg.LinAlgError(f'dpptrf info={info}')
        self.factor_alias=bool(np.shares_memory(self.L,packed) and self.L.__array_interface__['data'][0]==before)
        self.factor_same_object=self.L is packed
        assert self.factor_alias
        self.budget.observe('packed_cholesky',packed,self.L,temporary_allowance=4096)
        self.factor_trace=None
        if tracemalloc.is_tracing():
            after,peak=tracemalloc.get_traced_memory()
            self.factor_trace=dict(earlier_peak=earlier_peak,traced_before=current,traced_after=after,
                peak_extra_during_factorization=max(0,peak-current))
        self.budget.fixed=[self.permutation,self.L]
        self.calls=0;self.solve_alias_checks=[]

    def core_solve(self,rhs,*,extra=()):
        column=rhs.reshape(self.m,1)
        solution,info=lapack.dpptrs(self.m,self.L,column,lower=int(self.lower),overwrite_b=1)
        if info:raise np.linalg.LinAlgError(f'dpptrs info={info}')
        alias=np.shares_memory(solution,rhs)
        self.solve_alias_checks.append(bool(alias))
        assert alias
        self.budget.observe('packed_rhs_solve',rhs,column,solution,*extra,temporary_allowance=4096)
        return solution[:,0]

    def core_product(self,v):
        result=v.copy()
        # Lower A=L L^T; upper A=U^T U.
        for trans in ([1,0] if self.lower else [0,1]):
            updated=blas.dtpmv(self.m,self.L,result,lower=int(self.lower),trans=trans,overwrite_x=1)
            assert np.shares_memory(updated,result)
            result=updated
        self.budget.observe('packed_core_product',v,result)
        return result


def numerical_check():
    rng=np.random.default_rng(622100)
    X=np.ascontiguousarray(rng.normal(size=(29,5)));b=rng.normal(size=29)
    bandwidth=1.4;lam=.04;selected=np.array([2,7,11,19,23,0,4,8,12,17,21])
    A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bandwidth**2))+lam*np.eye(len(X))
    W=A[np.ix_(selected,selected)]
    result={}
    for lower in [True,False]:
        kernel=base.SharedKernel(X,bandwidth,base.Budget([X,b]))
        op=PackedSchur(kernel,len(selected),lam,indices=selected,lower=lower)
        v=rng.normal(size=len(selected))
        inverse=op.core_solve(v.copy());reference=linalg.solve(W,v,assume_a='pos')
        inv_error=float(np.linalg.norm(inverse-reference)/np.linalg.norm(reference))
        product=op.core_product(v)
        product_error=float(np.linalg.norm(product-W@v)/np.linalg.norm(W@v))
        triangular=v.copy()
        for trans in ([0,1] if lower else [1,0]):
            updated=blas.dtpsv(len(selected),op.L,triangular,lower=int(lower),trans=trans,overwrite_x=1)
            assert np.shares_memory(updated,triangular);triangular=updated
        triangular_error=float(np.linalg.norm(triangular-reference)/np.linalg.norm(reference))
        tail=rng.normal(size=len(op.R))
        D=A[np.ix_(op.R,op.R)]-A[np.ix_(op.R,selected)]@linalg.solve(W,A[np.ix_(selected,op.R)],assume_a='pos')
        action=op.apply(tail)
        action_error=float(np.linalg.norm(action-D@tail)/np.linalg.norm(D@tail))
        full=op.reconstruct(b,tail)
        constraint_error=float(np.linalg.norm((b-A@full)[selected])/np.linalg.norm(b))
        prefix_errors=[]
        if not lower:
            for m in [3,7]:
                prefix=op.L[:m*(m+1)//2]
                rhs=np.array(v[:m,None],dtype=np.float64,order='F',copy=True)
                actual,info=lapack.dpptrs(m,prefix,rhs,lower=0,overwrite_b=1)
                expected=linalg.solve(W[:m,:m],v[:m],assume_a='pos')
                error=float(np.linalg.norm(actual[:,0]-expected)/np.linalg.norm(expected))
                assert info==0 and np.shares_memory(prefix,op.L) and np.shares_memory(actual,rhs)
                prefix_errors.append(error)
        values=[inv_error,product_error,triangular_error,action_error,constraint_error,*prefix_errors]
        assert max(values)<1e-11
        result['lower' if lower else 'upper']=dict(inverse_relative_error=inv_error,
            core_product_relative_error=product_error,triangular_solve_relative_error=triangular_error,
            Schur_action_relative_error=action_error,constraint_relative_residual=constraint_error,
            factor_same_object=op.factor_same_object,factor_alias=op.factor_alias,
            RHS_alias_verified=all(op.solve_alias_checks),upper_prefix_relative_errors=prefix_errors,
            packed_bytes=op.L.nbytes,full_core_bytes=W.nbytes)
    # Explicitly demonstrate wrapper copy behavior outside the contiguous contract.
    packed=np.concatenate([W[j:,j] for j in range(len(selected))])
    strided=np.empty(2*len(packed));strided[::2]=packed
    copied,info=lapack.dpptrf(len(selected),strided[::2],lower=1,overwrite_ap=1)
    assert info==0
    B=np.column_stack([v,2*v]) # C-order multiple RHS must be copied by this wrapper.
    solution,info=lapack.dpptrs(len(selected),copied,B,lower=1,overwrite_b=1)
    result['copy_contract']=dict(strided_packed_input_copied=not np.shares_memory(copied,strided),
        C_order_multiple_RHS_copied=not np.shares_memory(solution,B),
        required='Contiguous native float64 packed buffer; one-column reshape or Fortran-contiguous multiple RHS. Assert pointer/sharing at runtime.')
    return result


def allocation_check(X,b,bandwidth,m):
    # Indices are predeclared, not selected/timed: this checks storage, not FPS cost.
    indices=np.arange(m,dtype=np.int64)
    tracemalloc.start()
    budget=base.Budget([X,b]);budget.live=[indices]
    op=None
    try:
        kernel=base.SharedKernel(X,bandwidth,budget)
        op=PackedSchur(kernel,m,.001,indices=indices,lower=True)
        del indices;budget.live=[]
        rhs=op.reduced_rhs(b);x=np.zeros_like(rhs);r=rhs.copy();p=r.copy()
        budget.live=[rhs,x,r,p]
        ap=op.apply(p);del ap
        full=op.reconstruct(b,x);ax=kernel.full(full,extra=(full,))
        ax*=-1;ax+=b;ax-=.001*full
        current,peak=tracemalloc.get_traced_memory()
        peak=max(peak,op.factor_trace['earlier_peak'])
        return dict(status='feasible',n=len(b),m=m,packed_core_bytes=op.L.nbytes,
            full_core_bytes=8*m*m,named=budget.report(),tracemalloc_peak_bytes=peak,
            trace_within_1MiB=peak<=base.LIMIT,factor_alias=op.factor_alias,
            factor_same_object=op.factor_same_object,core_solve_alias=all(op.solve_alias_checks),
            factor_trace=op.factor_trace,kernel_stats=kernel.stats,
            protocol='Untimed setup +one Schur action +reconstruction +one full verification; no CG solve or FPS cost benchmark.')
    except (MemoryError,np.linalg.LinAlgError) as error:
        current,peak=tracemalloc.get_traced_memory()
        if op is not None:peak=max(peak,op.factor_trace['earlier_peak'])
        return dict(status='failure',n=len(b),m=m,error=str(error),named=budget.report(),
            tracemalloc_peak_bytes=peak,packed_core_bytes=4*m*(m+1))
    finally:
        tracemalloc.stop()


def main():
    output=ROOT/'results/packed-schur-feasibility-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(5,6))
    started,cpu_started=perf_counter(),process_time()
    record=dict(status='running',scope='Classical packed Cholesky representation feasibility; no full solve timing or new method claim',
        validation_history='Initial tiny prefix-reference check reused an RHS view that the requested in-place solve overwrote. Corrected test to copy its RHS; no allocation/benchmark path ran in that failed invocation.',
        scipy_version=scipy.__version__,interfaces={name:getattr(module,name).__doc__ for module,names in
            [(lapack,['dpptrf','dpptrs']),(blas,['dtpmv','dtpsv'])] for name in names},
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__).read_bytes()).hexdigest(),
            str(BASE.relative_to(ROOT)):sha256(BASE.read_bytes()).hexdigest()},allocation_checks=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['numerical_check']=numerical_check();save()
        for stem,sizes in [('synthetic-n640-smooth',[480]),('airfoil-observed',[448,480])]:
            with np.load(ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz') as data:
                X=data['X_train'];b=data['y_train'];bandwidth=float(data['bandwidth'])
            X.setflags(write=False);b.setflags(write=False)
            for m in sizes:
                check=allocation_check(X,b,bandwidth,m);check['dataset']=stem
                record['allocation_checks'].append(check);save()
                print(json.dumps({k:check[k] for k in ['dataset','n','m','status','tracemalloc_peak_bytes','packed_core_bytes']}),flush=True)
    record['status']='complete';save()


if __name__=='__main__':main()
