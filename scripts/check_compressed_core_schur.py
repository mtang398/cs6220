#!/usr/bin/env python3
"""One native compressed-core Schur pilot; no dense matrices or hidden fallback."""
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter,process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_schur_inplace_frontier_v2 import Budget,SharedKernel,ROOT,LIMIT,RESERVE
from check_packed_schur_feasibility import PackedSchur
from check_packed_float_factor import PackedFloat
from check_packed_float_allocation import RawPointers


class ScopedBudget(Budget):
    def __init__(self,common):
        super().__init__(common);self.scopes=[];self.last_failure=None
    def size(self,*extra):
        return super().size(*(a for group in self.scopes for a in group),*extra)
    @contextmanager
    def hold(self,*arrays):
        self.scopes.append(arrays)
        try:yield
        finally:self.scopes.pop()
    def snapshot(self,*extra):
        roots={}
        for a in [*self.fixed,*self.live,*(a for group in self.scopes for a in group),*extra]:
            if a is None:continue
            while isinstance(a.base,np.ndarray):a=a.base
            if id(a) not in self.common:
                roots[id(a)]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes)
        return list(roots.values())
    def observe(self,label,*extra,temporary_allowance=0):
        try:super().observe(label,*extra,temporary_allowance=temporary_allowance)
        except MemoryError:
            self.last_failure=dict(phase=label,arrays=self.snapshot(*extra),temporary_allowance=temporary_allowance)
            raise
    def rows(self,label,columns,dimension,*extra,features=False):
        try:return super().rows(label,columns,dimension,*extra,features=features)
        except MemoryError:
            self.last_failure=dict(phase=label,arrays=self.snapshot(*extra),columns=columns,dimension=dimension)
            raise


class Schur:
    def __init__(self,X,b,bw,m,lam,method,helper):
        self.m=m;self.n=len(b);self.lam=lam;self.method=method
        self.calls=0;self.core_calls=0;self.corrections=0;self.core_checks=0
        self.stop_reasons=Counter();self.correction_counts=Counter();self.last_core_relative_max=0.
        self.factor_trace=None;self.exact_alias=True
        if method=='exact416':
            self.budget=ScopedBudget([X,b]);self.kernel=SharedKernel(X,bw,self.budget)
            selected=self.kernel.farthest(m)
            with self.budget.hold(selected):
                self.exact=PackedSchur(self.kernel,m,lam,indices=selected,lower=False)
            del selected
            self.permutation=self.exact.permutation;self.S=self.exact.S;self.R=self.exact.R
            self.factor_bytes=self.exact.L.nbytes;self.factor_trace=self.exact.factor_trace
            self.float=None
        else:
            self.float=PackedFloat(X,b,bw,m,lam,helper)
            previous=self.float.budget
            self.budget=ScopedBudget([X,b]);self.budget.peaks=dict(previous.peaks)
            self.budget.row_sizes={k:set(v) for k,v in previous.row_sizes.items()}
            old_indices=self.float.indices
            mask=np.ones(self.n,dtype=bool);mask[old_indices]=False
            remaining=np.flatnonzero(mask)
            self.permutation=np.r_[old_indices,remaining]
            self.budget.fixed=self.float.base_arrays.copy()
            self.budget.observe('core_adoption',old_indices,mask,remaining,self.permutation)
            self.minimum_raw_pivot=float(self.float.pivots.min())
            self.S=self.permutation[:m];self.R=self.permutation[m:]
            self.float.indices=self.S
            del self.float.pivots
            self.float.base_arrays=[self.permutation,self.float.factor]
            self.float.budget=self.budget;self.float.kernel.budget=self.budget
            self.budget.fixed=self.float.base_arrays.copy()
            self.kernel=self.float.kernel;self.factor_bytes=self.float.factor.nbytes
            del old_indices,mask,remaining,previous
        self.budget.fixed=[self.permutation,self.exact.L if self.float is None else self.float.factor]
    def core_solve(self,rhs,*,extra=()):
        self.core_calls+=1
        with self.budget.hold(rhs,*extra):
            if self.float is None:
                answer=self.exact.core_solve(rhs)
                self.exact_alias &= self.exact.solve_alias_checks.pop()
                return answer
            answer=self.float.solve(rhs)
            fixed={'fixed2':2,'fixed4':4}.get(self.method)
            max_corrections=fixed if fixed is not None else 6
            norm=float(np.linalg.norm(rhs));previous=None;stagnant=0;corrections=0
            with self.budget.hold(answer):
                while True:
                    if fixed is not None and corrections==fixed:
                        reason='fixed_count';break
                    residual=self.float.original_product(answer);self.core_checks+=1
                    residual*=-1.;residual+=rhs
                    relative=float(np.linalg.norm(residual)/max(norm,np.finfo(float).tiny))
                    if fixed is None:
                        self.last_core_relative_max=max(self.last_core_relative_max,relative)
                        if relative<=1e-10:reason='tolerance';del residual;break
                        if corrections==max_corrections:reason='correction_cap';del residual;break
                        stagnant=(stagnant+1 if previous is not None and relative>=.9*previous else 0)
                        if stagnant>=2:reason='stagnation';del residual;break
                        previous=relative
                    with self.budget.hold(residual):
                        delta=self.float.solve(residual)
                        self.budget.observe('core_refinement_update',delta)
                        answer+=delta
                        del delta
                    del residual;corrections+=1
            self.corrections+=corrections;self.stop_reasons[reason]+=1;self.correction_counts[corrections]+=1
            return answer
    def rhs(self,b):
        top=self.core_solve(b[self.S]);result=b[self.R]
        with self.budget.hold(top,result):
            self.kernel.product(self.R,self.S,top,label='reduced_rhs',target=result,subtract=True)
        return result
    def apply(self,v):
        values=self.kernel.product(None,self.R,v,label='Schur_first_pass')
        top=self.core_solve(values[self.S],extra=(values,v))
        result=values[self.R]
        self.budget.observe('Schur_split',values,top,result,v)
        del values
        with self.budget.hold(top,result,v):
            self.budget.observe('Schur_shift',temporary_allowance=result.nbytes)
            result+=self.lam*v
            self.kernel.product(self.R,self.S,top,label='Schur_second_pass',target=result,subtract=True)
        self.calls+=1
        return result
    def reconstruct(self,b,tail):
        top=b[self.S]
        with self.budget.hold(top,tail):
            self.kernel.product(self.S,self.R,tail,label='reconstruct_cross',target=top,subtract=True)
            top=self.core_solve(top)
            full=np.empty(self.n);full[self.S]=top;full[self.R]=tail
            self.budget.observe('reconstruct_vector',top,tail,full)
        return full
    def report(self):
        return dict(core_calls=self.core_calls,core_refinement_checks=self.core_checks,
            core_refinement_corrections=self.corrections,core_stop_reasons=dict(self.stop_reasons),
            core_correction_histogram=dict(self.correction_counts),
            maximum_checked_core_relative_residual=self.last_core_relative_max,
            core_factor_bytes=self.factor_bytes,core_factor_dtype='float64' if self.float is None else 'float32',
            exact_core_alias_verified=self.exact_alias,
            mixed_triangular_solves=0 if self.float is None else self.float.solve_calls,
            mixed_triangular_seconds=0. if self.float is None else self.float.solve_seconds,
            minimum_raw_pivot=None if self.float is None else self.minimum_raw_pivot,
            factor_trace=self.factor_trace)


def solve(X,b,bw,lam,method,helper,*,m=None,rtol=1e-8,maxiter=2000,cpu_cap=6.,deadline=float('inf')):
    started,cpu_started=perf_counter(),process_time();stop_cpu=min(deadline,cpu_started+cpu_cap)
    if m is None:m=416 if method=='exact416' else 600
    if method not in ['exact416','fixed2','fixed4','tol1e-10']:raise ValueError(method)
    op=None;solution=None;iteration=0;checks=0;progress=[];last_check=None;error=None
    status='maxiter';relative=None;landmark_relative=None;setup=None
    try:
        op=Schur(X,b,bw,m,lam,method,helper)
        rhs=op.rhs(b);tail=np.zeros_like(rhs);r=rhs.copy();p=r.copy();rr=float(r@r)
        norm_b=float(np.linalg.norm(b));setup=perf_counter()-started
        def verify(iteration):
            nonlocal checks,relative,landmark_relative,last_check
            op.budget.live=[rhs,tail,r,p]
            full=op.reconstruct(b,tail)
            residual=op.kernel.full(full,extra=(full,))
            op.budget.observe('original_residual_update',full,residual,temporary_allowance=full.nbytes)
            residual*=-1.;residual+=b;residual-=lam*full
            relative=float(np.linalg.norm(residual)/norm_b)
            op.budget.observe('landmark_residual_norm',full,residual,temporary_allowance=8*m)
            landmark_relative=float(np.linalg.norm(residual[op.S])/norm_b)
            checks+=1;last_check=dict(iteration=iteration,true_relative_residual=relative,
                landmark_relative_residual=landmark_relative)
            if len(progress)<8:progress.append(last_check)
            del residual
            return full
        for iteration in range(1,maxiter+1):
            if process_time()>stop_cpu:status='cpu_screen_cap';iteration-=1;break
            op.budget.live=[rhs,tail,r,p]
            ap=op.apply(p);curvature=float(p@ap)
            if curvature<=0 or not np.isfinite(curvature):status='nonpositive_curvature';del ap;break
            alpha=rr/curvature
            op.budget.observe('outer_CG_update',ap,temporary_allowance=2*tail.nbytes)
            tail+=alpha*p;r-=alpha*ap;del ap
            suspected=np.linalg.norm(r)<=rtol*norm_b
            if iteration%4==0 or suspected:
                full=verify(iteration)
                if relative<=rtol:status='converged';solution=full;break
                if suspected:
                    status='recursive_stop_true_residual_failed';solution=full;break
                del full
            next_rr=float(r@r);p*=next_rr/rr;p+=r;rr=next_rr
        if solution is None:solution=verify(iteration)
    except (MemoryError,np.linalg.LinAlgError) as caught:
        status='memory_failure' if isinstance(caught,MemoryError) else 'numerical_failure';error=str(caught)
    run=dict(method=method,m=m,status=status,error=error,
        converged=relative is not None and relative<=rtol,iterations=iteration,
        true_relative_residual=relative,landmark_relative_residual=landmark_relative,
        total_seconds=perf_counter()-started,total_cpu_seconds=process_time()-cpu_started,
        setup_seconds=setup,original_verifications=checks,
        first_verifications=progress,last_verification=last_check,
        common_immutable_input_bytes=X.nbytes+b.nbytes,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if op is not None:
        run.update(op.report(),memory=op.budget.report(),memory_failure_detail=op.budget.last_failure,
            kernel_stats=dict(op.kernel.stats),Schur_products=op.calls,
            selected_indices_sha256=sha256(op.S.tobytes()).hexdigest())
    return solution,run


def tiny_check(helper):
    rng=np.random.default_rng(622106);X=np.ascontiguousarray(rng.normal(size=(31,4)));b=rng.normal(size=31)
    bw=1.4;lam=.03
    A=cdist(X,X,'sqeuclidean');A/=bw;A/=bw;A*=-.5;np.exp(A,out=A);A.flat[::32]+=lam
    exact=linalg.solve(A,b,assume_a='pos');results=[]
    for method in ['fixed2','fixed4','tol1e-10']:
        op=Schur(X,b,bw,13,lam,method,helper);v=rng.normal(size=len(op.R))
        S=A[np.ix_(op.R,op.R)]-A[np.ix_(op.R,op.S)]@linalg.solve(A[np.ix_(op.S,op.S)],A[np.ix_(op.S,op.R)],assume_a='pos')
        action_error=float(np.linalg.norm(op.apply(v)-S@v)/np.linalg.norm(S@v))
        full,metrics=solve(X,b,bw,lam,method,helper,m=13,cpu_cap=2.)
        solution_error=float(np.linalg.norm(full-exact)/np.linalg.norm(exact))
        assert action_error<1e-10 and metrics['converged'] and solution_error<1e-7
        results.append(dict(method=method,action_relative_error=action_error,
            full_solution_relative_error=solution_error,true_relative_residual=metrics['true_relative_residual']))
    return results


def main():
    output=ROOT/'results/compressed-core-schur-pilot.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85))
    started,cpu_started=perf_counter(),process_time()
    helper=RawPointers(ROOT/'results/packed_float_factor.so')
    paths=[Path(__file__),ROOT/'scripts/check_packed_float_factor.py',ROOT/'scripts/check_packed_float_allocation.py',
        ROOT/'scripts/check_packed_schur_feasibility.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',
        ROOT/'scripts/packed_float_factor.c',ROOT/'results/packed_float_factor.so']
    record=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths},
        protocol=dict(datasets=['smooth2048','mixed2048'],seed=0,regularizations=[.001,1e-5],
            methods=['exact416','fixed2','fixed4','tol1e-10'],compressed_core=600,exact_core=416,
            rtol=1e-8,maxiter=2000,per_call_cpu_cap=6.,total_cpu_cap=80.,array_cap_bytes=LIMIT,reserve_bytes=RESERVE,
            fixed_refinement='Initial mixed solve plus exactly2or4 corrections; each correction pays one original core product. No unused final core check.',
            tolerance_refinement='Check original core residual against1e-10; at most6corrections; stop after two consecutive reductions less than10percent. Return truncated action and record reason.',
            original_checks='Every4outer iterations, suspected convergence, and final termination. Failed recursive stopping is reported, no replacement/restart fallback.',
            setup='Charged original FP64 FPS and streamed core factorization for every attempt.',
            scope='Bounded static representation pilot; not adaptive goal achievement. Common immutable X,b separate, no dense matrix during solver/audit.',
            lock='/tmp/cs6220-project-benchmark.lock'),runs=[],audits=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['tiny_check']=tiny_check(helper);save()
        path=ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
        with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
        # Separate complete fixed4 solve tests nested refinement buffers and full
        # original residual verification. Its timings never enter native rows.
        tracemalloc.start()
        try:
            audited,metrics=solve(X,b,bw,.001,'fixed4',helper,cpu_cap=6.)
            current,peak=tracemalloc.get_traced_memory()
            audit=dict(dataset='smooth2048',regularization=.001,method='fixed4',
                traced_peak_bytes=peak,trace_within_1MiB=peak<=LIMIT,metrics=metrics)
            record['audits'].append(audit);del audited
        finally:tracemalloc.stop()
        save()
        if not audit['trace_within_1MiB'] or metrics['status']=='memory_failure':
            record['status']='allocation_gate_failed';save();return
        for name in ['smooth','mixed']:
            path=ROOT/'data/prepared'/f'synthetic-n2560-{name}-s0-lambda0.001.npz'
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            X.setflags(write=False);b.setflags(write=False)
            for lam in [.001,1e-5]:
                for method in record['protocol']['methods']:
                    if process_time()-cpu_started>73:
                        record['status']='campaign_cpu_cap';save();return
                    assert not tracemalloc.is_tracing()
                    full,run=solve(X,b,bw,lam,method,helper,cpu_cap=6.,deadline=cpu_started+77.)
                    run.update(dataset=name+'2048',regularization=lam,
                        data_sha256=sha256(path.read_bytes()).hexdigest())
                    record['runs'].append(run);save();del full
                    print(json.dumps({k:run[k] for k in ['dataset','regularization','method','status','iterations','total_seconds','true_relative_residual']}),flush=True)
    record['status']='complete';save()


if __name__=='__main__':main()
