#!/usr/bin/env python3
"""Streaming FP32 packed Cholesky / FP64 solves, primitive feasibility only.

No full core is created during factor construction, mixed triangular solves,
or refinement. Independent dense cores/factors appear afterward, exclusively
for reference checks. This is not a complete Schur/Krylov memory audit.
"""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter,process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_schur_inplace_frontier_v2 import Budget,SharedKernel,ROOT,LIMIT,RESERVE


def load_helper():
    source=Path(__file__).with_name('packed_float_factor.c')
    output=ROOT/'results/packed_float_factor.so'
    command=['gcc','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(output),'-lm']
    subprocess.run(command,check=True)
    library=ctypes.CDLL(str(output))
    f32=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    library.packed_float_column.argtypes=[ctypes.c_size_t,ctypes.c_size_t,f32,f64,f64]
    library.packed_float_column.restype=ctypes.c_int
    library.packed_float_solve.argtypes=[ctypes.c_size_t,f32,f64]
    library.packed_float_solve.restype=ctypes.c_int
    return library,command


def unpack(packed,m):
    """Offline reference only; never used by construction/solve/refinement."""
    lower=np.zeros((m,m));offset=0
    for j in range(m):
        lower[j:,j]=packed[offset:offset+m-j];offset+=m-j
    return lower


class PackedFloat:
    def __init__(self,X,b,bw,m,lam,helper):
        started=perf_counter();cpu_started=process_time()
        self.budget=Budget([X,b]);self.kernel=SharedKernel(X,bw,self.budget)
        self.m=m;self.lam=lam;self.helper=helper
        self.indices=self.kernel.farthest(m)
        self.selection_seconds=perf_counter()-started
        self.factor=np.empty(m*(m+1)//2,dtype=np.float32)
        self.pivots=np.empty(m,dtype=np.float64)
        self.base_arrays=[self.indices,self.factor,self.pivots]
        self.budget.fixed=self.base_arrays.copy()
        points=X[self.indices]
        self.budget.observe('stream_setup',points)
        factor_started=perf_counter()
        self.failed_pivot=None
        for j in range(m):
            column=self.kernel.block(points[j:],points[j:j+1]).reshape(m-j)
            column[0]+=lam
            self.budget.observe('stream_column_update',points,column)
            info=helper.packed_float_column(m,j,self.factor,column,self.pivots[j:j+1])
            del column
            if info:
                self.failed_pivot=int(info-1)
                raise np.linalg.LinAlgError(f'FP32 streaming factor failed at pivot{info-1}; raw={self.pivots[j]}')
        del points
        self.factor_seconds=perf_counter()-factor_started
        self.setup_seconds=perf_counter()-started
        self.setup_cpu_seconds=process_time()-cpu_started
        self.setup_memory=self.budget.report()
        self.setup_kernel_stats=dict(self.kernel.stats)
        self.solve_calls=0;self.solve_seconds=0.;self.products=0
    def solve(self,rhs):
        started=perf_counter();answer=np.array(rhs,dtype=np.float64,copy=True)
        self.budget.observe('mixed_triangular_solve',rhs,answer)
        info=self.helper.packed_float_solve(self.m,self.factor,answer)
        if info:raise np.linalg.LinAlgError(f'Mixed solve invalid diagonal{info-1}')
        self.solve_calls+=1;self.solve_seconds+=perf_counter()-started
        return answer
    def original_product(self,v):
        answer=self.kernel.product(self.indices,self.indices,v,label='exact_core_product')
        self.budget.observe('exact_core_diagonal',v,answer,temporary_allowance=v.nbytes)
        answer+=self.lam*v;self.products+=1
        return answer


def refine(factor,rhs,*,rtol=1e-12,max_corrections=20):
    started=perf_counter();cpu_started=process_time();norm=float(np.linalg.norm(rhs))
    factor.budget.live=[rhs]
    initial=factor.solve(rhs);x=initial.copy();residual=None;history=[]
    factor.budget.live=[rhs,initial,x]
    for iteration in range(max_corrections+1):
        residual=factor.original_product(x)
        residual*=-1;residual+=rhs
        rel=float(np.linalg.norm(residual)/norm);history.append(rel)
        if rel<=rtol or iteration==max_corrections:break
        factor.budget.live=[rhs,initial,x,residual]
        correction=factor.solve(residual)
        factor.budget.observe('refinement_update',correction)
        x+=correction;del correction,residual
        factor.budget.live=[rhs,initial,x]
    factor.budget.live=[]
    return (rhs,initial,x),dict(converged=history[-1]<=rtol,
        corrections=iteration,initial_relative_residual=history[0],
        final_relative_residual=history[-1],minimum_observed_relative_residual=min(history),
        residual_history=history,total_seconds=perf_counter()-started,
        total_cpu_seconds=process_time()-cpu_started,
        target_relative_residual=rtol,max_corrections=max_corrections,
        residual_operator='Original Gaussian core regenerated by FP64 cdist/exp and FP64 matvec; no dense reference used.')


def tiny_check(helper):
    rng=np.random.default_rng(622103);n=13
    matrix=rng.normal(size=(n,n));A=matrix@matrix.T+2*np.eye(n)
    factor=np.empty(n*(n+1)//2,dtype=np.float32);pivots=np.empty(n)
    for j in range(n):
        work=A[j:,j].copy()
        assert helper.packed_float_column(n,j,factor,work,pivots[j:j+1])==0
    L=unpack(factor,n);rhs=rng.normal(size=n)
    actual=rhs.copy();assert helper.packed_float_solve(n,factor,actual)==0
    expected=linalg.solve_triangular(L,rhs,lower=True)
    expected=linalg.solve_triangular(L.T,expected,lower=False)
    solve_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    history=[];x=actual.copy()
    for step in range(8):
        r=rhs-A@x;history.append(float(np.linalg.norm(r)/np.linalg.norm(rhs)))
        if history[-1]<=1e-12:break
        assert helper.packed_float_solve(n,factor,r)==0;x+=r
    exact=linalg.solve(A,rhs,assume_a='pos')
    relative=float(np.linalg.norm(x-exact)/np.linalg.norm(exact))
    assert solve_error<1e-12 and history[-1]<1e-12 and relative<1e-12
    return dict(m=n,mixed_solve_vs_independent_triangular_relative_error=solve_error,
        factor_relative_Frobenius_error=float(np.linalg.norm(L@L.T-A)/np.linalg.norm(A)),
        refinement_residual_history=history,refined_solution_relative_error=relative,
        minimum_raw_pivot=float(pivots.min()))


def offline_reference(factor,saved):
    """Called only after all native algorithm work in this case has ended."""
    started=perf_counter();points=factor.kernel.X[factor.indices];bw=factor.kernel.bandwidth
    C=cdist(points,points,'sqeuclidean');C/=bw;C/=bw;C*=-.5;np.exp(C,out=C)
    C.flat[::factor.m+1]+=factor.lam
    L=unpack(factor.factor,factor.m)
    reconstructed=L@L.T
    values=linalg.eigvalsh(C,check_finite=False)
    factor_error=float(np.linalg.norm(reconstructed-C)/np.linalg.norm(C))
    references=[]
    for rhs,initial,final in saved:
        exact=linalg.solve(C,rhs,assume_a='pos',check_finite=False)
        triangular=linalg.solve_triangular(L,rhs,lower=True,check_finite=False)
        triangular=linalg.solve_triangular(L.T,triangular,lower=False,check_finite=False)
        references.append(dict(initial_solution_relative_error=float(np.linalg.norm(initial-exact)/np.linalg.norm(exact)),
            final_solution_relative_error=float(np.linalg.norm(final-exact)/np.linalg.norm(exact)),
            mixed_solve_vs_dense_triangular_relative_error=float(np.linalg.norm(initial-triangular)/np.linalg.norm(triangular)),
            dense_original_residual_of_refined_solution=float(np.linalg.norm(rhs-C@final)/np.linalg.norm(rhs)),
            dense_direct_reference_relative_residual=float(np.linalg.norm(rhs-C@exact)/np.linalg.norm(rhs))))
    return dict(factor_relative_Frobenius_error=factor_error,
        maximum_factor_reconstruction_entry_error=float(np.max(np.abs(reconstructed-C))),
        original_core_minimum_eigenvalue=float(values[0]),original_core_condition=float(values[-1]/values[0]),
        rhs_references=references,offline_seconds=perf_counter()-started,
        scope='Dense C, unpacked FP64 L and LL^T exist only in this later offline validation; excluded from algorithm allocation/timing.')


def main():
    output=ROOT/'results/packed-float-factor.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(30,35))
    started,cpu_started=perf_counter(),process_time()
    helper,command=load_helper()
    paths=[Path(__file__),Path(__file__).with_name('packed_float_factor.c'),
        ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/packed_float_factor.so']
    record=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths},
        protocol=dict(dataset='synthetic-n2560-smooth-s0',n=2048,core_sizes=[416,600],
            regularizations=[.001,1e-5],selection='Original FP64 FPS from index0; independently recomputed and charged for each case',
            factor='FP32 packed lower; one original FP64 Gaussian column; FP64 subtraction/sqrt/division, then FP32 storage',
            solve='FP64 forward/back substitution with promoted FP32 coefficients',
            refinement='x+=Ltilde^-T Ltilde^-1(rhs-Cx); regenerate exact original C products;20correction cap,1e-12 relative residual',
            rhs=['seeded_normal','actual_selected_training_target'],compiler=command,
            array_budget_bytes=LIMIT,reserve_bytes=RESERVE,
            scope='Primitive only. Setup named-array cap and separate setup trace; no full Schur/Krylov solver memory/performance claim.',
            dense_reference='Only after native construction and both refinement solves per case; never called by those algorithms.',
            cpu_cap_seconds=30,lock='/tmp/cs6220-project-benchmark.lock'),runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['tiny_check']=tiny_check(helper);save()
        path=ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
        with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
        X.setflags(write=False);b.setflags(write=False)
        for m in [416,600]:
            for lam in [.001,1e-5]:
                metadata=dict(m=m,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest())
                try:
                    factor=PackedFloat(X,b,bw,m,lam,helper)
                    row={**metadata,'status':'factorized','factor_bytes':factor.factor.nbytes,
                        'hypothetical_FP64_packed_factor_bytes':2*factor.factor.nbytes,
                        'setup_seconds':factor.setup_seconds,'setup_cpu_seconds':factor.setup_cpu_seconds,
                        'selection_seconds':factor.selection_seconds,'factorization_seconds':factor.factor_seconds,
                        'minimum_raw_pivot':float(factor.pivots.min()),'maximum_raw_pivot':float(factor.pivots.max()),
                        'last_raw_pivot':float(factor.pivots[-1]),'minimum_stored_diagonal':float(np.sqrt(factor.pivots).astype(np.float32).min()),
                        'setup_memory':factor.setup_memory,'setup_kernel_stats':factor.setup_kernel_stats,
                        'selected_indices':factor.indices.tolist(),'rhs_runs':[]}
                    saved=[]
                    for name in ['seeded_normal','actual_selected_training_target']:
                        rhs=(np.random.default_rng(622104).normal(size=m) if name=='seeded_normal' else b[factor.indices])
                        arrays,metrics=refine(factor,rhs);metrics['rhs']=name
                        saved.append(arrays);row['rhs_runs'].append(metrics)
                        factor.budget.fixed=factor.base_arrays+[a for group in saved for a in group]
                    row.update(primitive_memory=factor.budget.report(),mixed_triangular_solves=factor.solve_calls,
                        mixed_triangular_total_seconds=factor.solve_seconds,original_core_products=factor.products,
                        algorithm_kernel_stats=dict(factor.kernel.stats))
                    row['offline_reference']=offline_reference(factor,saved)
                    for metrics,reference in zip(row['rhs_runs'],row['offline_reference']['rhs_references']):metrics.update(reference)
                    record['runs'].append(row);save()
                    print(json.dumps({k:row[k] for k in ['m','regularization','status','factor_bytes','setup_seconds','minimum_raw_pivot']}),flush=True)
                    del factor,saved,arrays,rhs
                except (MemoryError,np.linalg.LinAlgError) as error:
                    record['runs'].append({**metadata,'status':'failure','error':str(error)});save()
        # Separate allocation-only setup; its traced timing is not a native cost.
        tracemalloc.start()
        try:
            audited=PackedFloat(X,b,bw,600,1e-5,helper)
            current,peak=tracemalloc.get_traced_memory()
            record['setup_allocation_audit']=dict(m=600,regularization=1e-5,
                traced_peak_bytes=peak,trace_within_1MiB=peak<=LIMIT,named=audited.setup_memory,
                scope='FPS+streamed construction only, common X/b allocated before trace; no full solver audit.')
            del audited
        finally:tracemalloc.stop()
    record['status']='complete';save()


if __name__=='__main__':main()
