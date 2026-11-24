#!/usr/bin/env python3
"""Callable mixed-kernel iterative-refinement prototype; no import-time runs.

Fixed inner M = float64(exp(float32(-cdist64/(2*bw**2))))) + lambda I.
Its packed core and every arithmetic accumulation are FP64. Original FP64 FPS
is charged once. The adaptive eta rule is a heuristic, not a contraction bound.
All modes verify the literal original residual before claiming convergence.
"""
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
from pathlib import Path
import resource
from time import perf_counter,process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.linalg import lapack
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
PACKED_SOURCE=ROOT/'scripts/check_packed_schur_feasibility.py'
BASE_SOURCE=ROOT/'scripts/check_schur_inplace_frontier_v2.py'
spec=importlib.util.spec_from_file_location('refinement_packed_primitives',PACKED_SOURCE)
packed=importlib.util.module_from_spec(spec);spec.loader.exec_module(packed)
base=packed.base
SOURCE_SHA256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__),PACKED_SOURCE,BASE_SOURCE]}
MODES={'fixed_0.1':.1,'fixed_0.01':.01,'fixed_0.001':.001,'adaptive':.1}


class NestedBudget(base.Budget):
    """One owner for core, outer state, inner state and nested kernel scratch."""
    def __init__(self,common):
        super().__init__(common)
        self.external=[]
        self.persistent=[]
        self.scopes=[]
    def size(self,*extra):
        scoped=[a for group in self.scopes for a in group]
        return super().size(*self.external,*self.persistent,*scoped,*extra)
    @contextmanager
    def hold(self,*arrays):
        self.scopes.append(arrays)
        try:yield
        finally:self.scopes.pop()
    def tile_rows(self,label,columns,dimension,*extra,low,features,row_limit):
        available=base.LIMIT-base.RESERVE-self.size(*extra)-4096
        per_row=(12 if low else 8)*columns+8*(dimension if features else 0)+16
        rows=min(row_limit,int(available//per_row))
        if rows<1:raise MemoryError(f'No mixed-kernel row tile fits at {label}')
        self.row_sizes.setdefault(label,set()).add(rows)
        return rows


class Kernel(base.SharedKernel):
    def __init__(self,X,bw,budget,*,low,row_limit=128):
        super().__init__(X,bw,budget)
        self.low=bool(low);self.row_limit=row_limit
        self.stats.update(full_products_by_purpose={},kernel_seconds=0.)
    def block(self,rows,columns,*,live=(),label='kernel_block'):
        started=perf_counter()
        distance=cdist(rows,columns,'sqeuclidean')
        if self.low:
            distance*= -.5/(self.bandwidth*self.bandwidth)
            small=distance.astype(np.float32)
            self.budget.observe(label+'_conversion',rows,columns,*live,distance,small)
            del distance
            np.exp(small,out=small)
            tile=small.astype(np.float64)
            self.budget.observe(label+'_promotion',rows,columns,*live,small,tile)
            del small
        else:
            distance/=self.bandwidth;distance/=self.bandwidth;distance*=-.5
            np.exp(distance,out=distance);tile=distance
            self.budget.observe(label,rows,columns,*live,tile)
        self.stats['entry_evaluations']+=tile.size
        self.stats['entry_blocks']+=1
        self.stats['kernel_seconds']+=perf_counter()-started
        return tile
    def farthest(self,m):
        selected=np.empty(m,dtype=np.int64);similarity=np.zeros(self.n);j=0
        for count in range(m):
            selected[count]=j
            tile=self.block(self.X,self.X[j:j+1],live=(selected,similarity),label='FPS64')
            column=tile[:,0]
            self.stats['column_queries']+=1
            self.budget.observe('FPS64_update',selected,similarity,tile,column)
            np.maximum(similarity,column,out=similarity)
            similarity[selected[:count+1]]=np.inf
            j=int(np.argmin(similarity));del column,tile
        return selected
    def product(self,row_indices,column_indices,vector,*,label,extra=(),target=None,subtract=False):
        row_count=self.n if row_indices is None else len(row_indices)
        col_count=self.n if column_indices is None else len(column_indices)
        columns=self.X if column_indices is None else self.X[column_indices]
        result=np.empty(row_count) if target is None else target
        block_rows=self.budget.tile_rows(label,col_count,self.d,*extra,columns,result,vector,
            low=self.low,features=row_indices is not None,row_limit=self.row_limit)
        for start in range(0,row_count,block_rows):
            stop=min(row_count,start+block_rows)
            points=self.X[start:stop] if row_indices is None else self.X[row_indices[start:stop]]
            tile=self.block(points,columns,live=(*extra,result,vector),label=label)
            product=tile@vector
            self.budget.observe(label+'_accumulation',*extra,columns,result,vector,points,tile,product)
            if subtract:result[start:stop]-=product
            else:result[start:stop]=product
            del tile,product,points
        del columns
        return result
    def full(self,vector,*,extra=(),purpose='unspecified'):
        result=self.product(None,None,vector,label=('M32_' if self.low else 'A64_')+purpose,extra=extra)
        self.stats['full_matvecs']+=1
        purposes=self.stats['full_products_by_purpose']
        purposes[purpose]=purposes.get(purpose,0)+1
        return result


class LowSchur(packed.PackedSchur):
    """Packed low-operator core; nested calls expose all live arrays to Budget."""
    def __init__(self,kernel,selected,lam):
        self.kernel=kernel;self.budget=kernel.budget;self.lam=lam
        self.m=len(selected);self.n=kernel.n;self.lower=True
        mask=np.ones(self.n,dtype=bool);mask[selected]=False
        remaining=np.flatnonzero(mask)
        self.permutation=np.r_[selected,remaining]
        self.budget.observe('packed_permutation',selected,mask,remaining,self.permutation)
        del mask,remaining
        self.S=self.permutation[:self.m];self.R=self.permutation[self.m:]
        core=np.empty(self.m*(self.m+1)//2,dtype=np.float64)
        self.budget.fixed=[self.permutation,core]
        self.budget.observe('packed_core_storage',selected)
        columns=kernel.X[self.S]
        offset=0
        with self.budget.hold(selected,columns):
            for j in range(self.m):
                tile=kernel.block(columns[j:],columns[j:j+1],label='low_core')
                width=self.m-j
                core[offset:offset+width]=tile[:,0]
                core[offset]+=lam;offset+=width;del tile
        del columns
        pointer=core.__array_interface__['data'][0]
        self.factor_trace=None
        if tracemalloc.is_tracing():
            before,earlier_peak=tracemalloc.get_traced_memory();tracemalloc.reset_peak()
        self.L,info=lapack.dpptrf(self.m,core,lower=1,overwrite_ap=1)
        if info:raise np.linalg.LinAlgError(f'Low core dpptrf info={info}')
        self.factor_alias=bool(np.shares_memory(core,self.L) and self.L.__array_interface__['data'][0]==pointer)
        self.factor_same_object=self.L is core
        if not self.factor_alias:raise MemoryError('Packed Cholesky copied its core')
        self.budget.observe('packed_factorization',selected,core,self.L,temporary_allowance=4096)
        if tracemalloc.is_tracing():
            after,peak=tracemalloc.get_traced_memory()
            self.factor_trace=dict(earlier_peak=earlier_peak,peak_extra=max(0,peak-before))
        self.budget.fixed=[self.permutation,self.L]
        self.calls=self.core_solves=0;self.all_core_solves_alias=True
    def core_solve(self,rhs,*,extra=()):
        column=rhs.reshape(self.m,1)
        solution,info=lapack.dpptrs(self.m,self.L,column,lower=1,overwrite_b=1)
        if info:raise np.linalg.LinAlgError(f'Low core dpptrs info={info}')
        alias=np.shares_memory(solution,rhs)
        self.core_solves+=1;self.all_core_solves_alias &= bool(alias)
        if not alias:raise MemoryError('Packed core solve copied RHS')
        self.budget.observe('packed_rhs_solve',rhs,column,solution,*extra,temporary_allowance=4096)
        return solution[:,0]
    def reduced_rhs(self,b):
        top=self.core_solve(b[self.S]);result=b[self.R]
        with self.budget.hold(top,result):
            self.kernel.product(self.R,self.S,top,label='inner_rhs',target=result,subtract=True)
        return result
    def apply(self,v):
        values=self.kernel.product(None,self.R,v,label='inner_first')
        top=self.core_solve(values[self.S],extra=(values,v))
        result=values[self.R]
        self.budget.observe('inner_split',values,top,result,v)
        del values
        with self.budget.hold(top,result,v):
            self.budget.observe('inner_diagonal',temporary_allowance=result.nbytes)
            result+=self.lam*v
            self.kernel.product(self.R,self.S,top,label='inner_second',target=result,subtract=True)
        self.calls+=1
        return result
    def reconstruct(self,b,tail):
        top=b[self.S]
        with self.budget.hold(top,tail):
            self.kernel.product(self.S,self.R,tail,label='inner_reconstruct',target=top,subtract=True)
            top=self.core_solve(top,extra=(tail,))
            full=np.empty(self.n);full[self.S]=top;full[self.R]=tail
            self.budget.observe('inner_reconstruct_vector',top,tail,full)
        return full


def inner_cg(op,rhs,eta,*,max_inner,deadline):
    started=perf_counter();budget=op.budget
    tail=None;r=None;p=None
    try:
        r=op.reduced_rhs(rhs);tail=np.zeros_like(r);p=r.copy()
        budget.live=[tail,r,p];budget.observe('inner_initialization')
        rr=float(r@r);target=eta*float(np.linalg.norm(rhs))
        status='inner_converged' if np.sqrt(rr)<=target else 'inner_maxiter'
        iterations=0
        if status!='inner_converged':
            for iterations in range(1,max_inner+1):
                if process_time()>deadline:
                    status='cpu_screen_cap';iterations-=1;break
                budget.live=[tail,r,p]
                ap=op.apply(p);curvature=float(p@ap)
                if curvature<=0 or not np.isfinite(curvature):
                    status='inner_nonpositive_curvature';del ap;break
                alpha=rr/curvature
                budget.observe('inner_CG_update',ap,temporary_allowance=2*tail.nbytes)
                tail+=alpha*p;r-=alpha*ap;del ap
                next_rr=float(r@r)
                if np.sqrt(next_rr)<=target:
                    rr=next_rr;status='inner_converged';break
                p*=next_rr/rr;p+=r;rr=next_rr
        reduced_norm=float(np.linalg.norm(r))
        del r,p
        budget.live=[tail]
        correction=op.reconstruct(rhs,tail)
        metrics=dict(status=status,iterations=iterations,eta=eta,
            reduced_recursive_residual_norm=reduced_norm,target_norm=target,
            reduced_recursive_relative_to_full_rhs=reduced_norm/float(np.linalg.norm(rhs)),
            seconds=perf_counter()-started)
        return correction,metrics
    finally:
        budget.live=[]


def solve(X,b,bw,m,lam,mode='adaptive',*,rtol=1e-8,max_outer=20,max_inner=2000,
          cpu_cap=6.,deadline=float('inf'),verify_every=4):
    """Return (x, metrics); no silent precision fallback or unconstrained retry.

    Modes: adaptive, fixed_0.1, fixed_0.01, fixed_0.001. Fixed modes stop inner CG
    using its recursive reduced residual; only adaptive pays a full Mz query.
    Every mode pays A64z, and literal A64x checks every verify_every outer steps,
    on suspected convergence and at termination if the current x is unchecked.
    """
    if mode not in MODES:raise ValueError(f'mode must be one of {tuple(MODES)}')
    if not (0<m<len(b)) or lam<=0 or verify_every<1:raise ValueError('Invalid m, lambda or verification interval')
    if X.dtype!=np.float64 or b.dtype!=np.float64 or not X.flags.c_contiguous:
        raise ValueError('Common X and b must be float64; X must be C-contiguous')
    started,cpu_started=perf_counter(),process_time()
    stop_cpu=min(deadline,cpu_started+cpu_cap)
    budget=NestedBudget([X,b]);high=Kernel(X,bw,budget,low=False);low=Kernel(X,bw,budget,low=True)
    x=np.zeros_like(b);r=b.copy();budget.external=[x,r]
    norm_b=float(np.linalg.norm(b));eta=MODES[mode]
    status='outer_maxiter';error=None;op=None;events=[]
    updates=0;last_verified=0;checks=0;true_relative=1. if norm_b else 0.
    setup_seconds=None;checks_seconds=0.;Az_seconds=0.;Mz_seconds=0.
    def literal_check():
        nonlocal last_verified,checks,true_relative,checks_seconds
        tick=perf_counter();budget.external=[x,r];budget.live=[]
        literal=high.full(x,purpose='literal_residual')
        budget.observe('literal_residual_update',literal,temporary_allowance=x.nbytes)
        literal*=-1.;literal+=b;literal-=lam*x
        budget.observe('literal_residual_gap',literal,temporary_allowance=r.nbytes)
        gap=float(np.linalg.norm(literal-r)/max(norm_b,np.finfo(float).tiny))
        r[:]=literal;del literal
        true_relative=float(np.linalg.norm(r)/max(norm_b,np.finfo(float).tiny))
        last_verified=updates;checks+=1;checks_seconds+=perf_counter()-tick
        return gap
    try:
        if norm_b==0:
            status='converged';setup_seconds=perf_counter()-started
        else:
            selected=high.farthest(m)
            with budget.hold(selected):op=LowSchur(low,selected,lam)
            del selected
            setup_seconds=perf_counter()-started
            for outer in range(1,max_outer+1):
                if process_time()>stop_cpu:status='cpu_screen_cap';break
                old_norm=float(np.linalg.norm(r))
                z,inner=inner_cg(op,r,eta,max_inner=max_inner,deadline=stop_cpu)
                event=dict(outer=outer,eta=eta,inner=inner)
                if inner['status']!='inner_converged':
                    status=inner['status'];events.append(event);del z;break
                budget.external=[x,r,z]
                tick=perf_counter();Az=high.full(z,purpose='correction')
                budget.observe('A64_diagonal',Az,temporary_allowance=z.nbytes)
                Az+=lam*z;Az_seconds+=perf_counter()-tick
                budget.external=[x,r,z,Az]
                if mode=='adaptive':
                    tick=perf_counter();Mz=low.full(z,purpose='model_error')
                    budget.observe('M32_diagonal',Mz,temporary_allowance=z.nbytes)
                    Mz+=lam*z
                    with budget.hold(Mz):
                        budget.observe('full_M_residual',temporary_allowance=r.nbytes)
                        event['full_M_relative_residual']=float(np.linalg.norm(r-Mz)/old_norm)
                    Mz-=Az
                    ratio=float(np.linalg.norm(Mz)/old_norm)
                    event['operator_error_ratio']=ratio
                    event['next_eta']=float(np.clip(.5*ratio,1e-4,.2))
                    eta=event['next_eta'];del Mz
                    Mz_seconds+=perf_counter()-tick
                x+=z;r-=Az;updates+=1
                budget.external=[x,r];del z,Az
                recursive_relative=float(np.linalg.norm(r)/norm_b)
                event['recursive_original_relative_residual']=recursive_relative
                if updates%verify_every==0 or recursive_relative<=rtol:
                    event['literal_recursive_gap']=literal_check()
                    event['true_relative_residual']=true_relative
                events.append(event)
                if last_verified==updates and true_relative<=rtol:
                    status='converged';break
    except (MemoryError,np.linalg.LinAlgError,FloatingPointError) as caught:
        status='memory_failure' if isinstance(caught,MemoryError) else 'numerical_failure'
        error=str(caught)
    finally:
        # No prior correction or inner vectors are live during the final check.
        budget.live=[];budget.external=[x,r]
        # Variables may still hold arrays when an intermediate allocation failed.
        if 'z' in locals():del z
        if 'Az' in locals():del Az
        if 'Mz' in locals():del Mz
        if updates!=last_verified:
            try:literal_check()
            except (MemoryError,np.linalg.LinAlgError) as caught:
                status='final_verification_failed';error=str(caught);true_relative=None
    if setup_seconds is None:setup_seconds=perf_counter()-started
    verified=last_verified==updates and true_relative is not None
    metrics=dict(mode=mode,status=status,error=error,
        converged=bool(verified and true_relative<=rtol),true_relative_residual=true_relative,
        final_original_residual_verified=verified,outer_updates=updates,
        inner_iterations=sum(e['inner']['iterations'] for e in events),events=events,
        total_seconds=perf_counter()-started,total_cpu_seconds=process_time()-cpu_started,
        setup_seconds=setup_seconds,correction_A64_seconds=Az_seconds,
        adaptive_M32_seconds=Mz_seconds,literal_verification_seconds=checks_seconds,
        literal_original_verifications=checks,verify_every=verify_every,
        high_kernel_stats=high.stats,low_kernel_stats=low.stats,memory=budget.report(),
        common_immutable_input_bytes=X.nbytes+b.nbytes,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=SOURCE_SHA256,
        memory_scope='All owned core/permutation, outer state, inner state, feature gathers, and simultaneous FP64/FP32 conversion tiles; common immutable X,b excluded;64KiB reserve.',
        forcing_scope='Adaptive next eta is heuristic; measured full-M residual recorded. Fixed controls use recursive reduced residual without a full-M query.')
    if op is not None:
        metrics.update(selected_indices=op.S.tolist(),packed_core_bytes=op.L.nbytes,
            factor_alias=op.factor_alias,all_core_solves_alias=op.all_core_solves_alias,
            core_solves=op.core_solves,inner_Schur_products=op.calls,factor_trace=op.factor_trace)
    return x,metrics


def allocation_audit(X,b,bw,m,lam,mode='adaptive'):
    """Untimed full live-state audit, not a convergence/performance experiment."""
    tracemalloc.start();op=None
    try:
        budget=NestedBudget([X,b]);high=Kernel(X,bw,budget,low=False);low=Kernel(X,bw,budget,low=True)
        x=np.zeros_like(b);r=b.copy();budget.external=[x,r]
        selected=high.farthest(m)
        with budget.hold(selected):op=LowSchur(low,selected,lam)
        del selected
        ri=op.reduced_rhs(r);tail=np.zeros_like(ri);p=ri.copy()
        budget.live=[tail,ri,p];ap=op.apply(p)
        budget.observe('audit_inner_update',ap,temporary_allowance=2*tail.nbytes)
        del ap,ri,p;budget.live=[tail]
        z=op.reconstruct(r,tail);del tail;budget.live=[]
        budget.external=[x,r,z]
        Az=high.full(z,purpose='correction')
        budget.observe('audit_A64_diagonal',Az,temporary_allowance=z.nbytes);Az+=lam*z
        budget.external=[x,r,z,Az]
        if mode=='adaptive':
            Mz=low.full(z,purpose='model_error')
            budget.observe('audit_M32_error',Mz,temporary_allowance=2*r.nbytes)
            Mz+=lam*z;Mz-=Az;del Mz
        x+=z;r-=Az;del z,Az;budget.external=[x,r]
        true=high.full(x,purpose='literal_residual')
        budget.observe('audit_literal_update',true,temporary_allowance=r.nbytes)
        true*=-1;true+=b;true-=lam*x
        current,peak=tracemalloc.get_traced_memory()
        peak=max(peak,op.factor_trace['earlier_peak'])
        return dict(status='complete',n=len(b),m=m,mode=mode,memory=budget.report(),
            tracemalloc_peak_bytes=peak,trace_within_1MiB=peak<=base.LIMIT,
            factor_alias=op.factor_alias,core_solve_alias=op.all_core_solves_alias,
            factor_trace=op.factor_trace,high_kernel_stats=high.stats,low_kernel_stats=low.stats,
            scope='Separate setup+steady inner action+reconstruction+Az+optional Mz+literal A residual; no full solve timing.')
    except (MemoryError,np.linalg.LinAlgError) as caught:
        current,peak=tracemalloc.get_traced_memory()
        if op is not None and op.factor_trace:peak=max(peak,op.factor_trace['earlier_peak'])
        return dict(status='failure',error=str(caught),tracemalloc_peak_bytes=peak,memory=budget.report())
    finally:tracemalloc.stop()


def tiny_checks():
    """Independent dense low-operator checks and original-system solves."""
    rng=np.random.default_rng(622101)
    X=np.ascontiguousarray(rng.normal(size=(31,4)));b=rng.normal(size=31)
    bw=1.3;lam=.03;m=9
    distance=cdist(X,X,'sqeuclidean');distance*= -.5/(bw*bw)
    Klow=np.exp(distance.astype(np.float32)).astype(np.float64)
    M=Klow+lam*np.eye(len(b))
    K64=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A=K64+lam*np.eye(len(b))
    selected=np.array([2,7,12,17,22,27,0,4,9])
    budget=NestedBudget([X,b]);kernel=Kernel(X,bw,budget,low=True,row_limit=3)
    op=LowSchur(kernel,selected,lam)
    v=rng.normal(size=len(op.R))
    D=M[np.ix_(op.R,op.R)]-M[np.ix_(op.R,selected)]@linalg.solve(M[np.ix_(selected,selected)],M[np.ix_(selected,op.R)],assume_a='pos')
    action=op.apply(v)
    action_error=float(np.linalg.norm(action-D@v)/np.linalg.norm(D@v))
    blocks=np.empty_like(Klow)
    for start in range(0,len(b),4):blocks[start:start+4]=kernel.block(X[start:start+4],X)
    selected_block=kernel.block(X[selected],X[selected[::-1]])
    assert np.array_equal(Klow,Klow.T) and np.array_equal(blocks,Klow)
    assert np.array_equal(selected_block,Klow[np.ix_(selected,selected[::-1])])
    assert action_error<1e-12
    exact=linalg.solve(A,b,assume_a='pos');runs={}
    for mode in MODES:
        actual,result=solve(X,b,bw,m,lam,mode,rtol=1e-10,cpu_cap=2.)
        error=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
        residual=float(np.linalg.norm(b-A@actual)/np.linalg.norm(b))
        assert result['converged'] and error<1e-8 and residual<1e-10
        assert result['low_kernel_stats']['full_matvecs']==(result['outer_updates'] if mode=='adaptive' else 0)
        assert result['high_kernel_stats']['full_products_by_purpose']['correction']==result['outer_updates']
        runs[mode]=dict(outer_updates=result['outer_updates'],inner_iterations=result['inner_iterations'],
            solution_relative_error=error,independent_original_relative_residual=residual,
            model_error_queries=result['low_kernel_stats']['full_matvecs'],
            literal_original_verifications=result['literal_original_verifications'])
    audit=allocation_audit(X,b,bw,m,lam)
    assert audit['status']=='complete' and audit['trace_within_1MiB']
    return dict(low_kernel_bitwise_symmetric=True,low_kernel_bitwise_block_consistent=True,
        low_Schur_action_relative_error=action_error,original_system_modes=runs,allocation_audit=audit)


tiny_check=tiny_checks


if __name__=='__main__':
    raise SystemExit('Import this module and call solve(), allocation_audit(), or tiny_checks(); no campaign is launched here.')
