#!/usr/bin/env python3
"""Matrix-free cost-only falsifier; no quantized augmented solver is run."""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner

OUTPUT = ROOT/'results/recomputed-deflation-cost-diagnostic.json'
GOLD = ROOT/'results/augcg-swap-oracle-diagnostic.json'
POWER = ROOT/'results/power-depth-headroom-diagnostic.json'
M, LAM, TOL = 20, .001, 1e-8
START_CPU, START_WALL = process_time(), perf_counter()
SOURCE_PATHS = [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py',
                ROOT/'src/nystrom_project/preconditioners.py']
record = dict(status='running',
    scope='Optimistic cost estimate using frozen full-prefix gold counts; no quantized solver, convergence, or speedup claim',
    protocol=dict(seed=0, regularization=LAM, base_ranks=[16,32], stored_prefix_vectors=M,
        stored_basis_dtype='float32', arithmetic_dtype='float64', blas_threads=1,
        native_control_repeats=2, control_order='seeded permutation separately for each repeat',
        micro_samples=7, candidate_summary='minimum sample; deliberately favorable',
        candidate_prefix='Actual ordinary compact-RP PCG20, with normalized p stored as float32',
        candidate_setup='One A[x20,Z] batch supplies true prefix residual, AZ, and Z.T AZ',
        candidate_continuation='Gold remaining iterations times measured A[t,Z] cost; no augmented solve executed',
        omitted_candidate_costs=['orthogonalization/rank repair','coarse factorization','initial coarse correction',
            'continuation preconditioner applications','projection and vector updates','quantization-induced extra iterations'],
        candidate_verification='Charge every verification product recorded by the gold continuation',
        static_controls=['CG','compact_RP','power2','power4','power8'],
        power_construction='Initial two-pass Gaussian, then refine its factor range for2 and4 additional passes',
        matched_rank='Largest compact factor fitting base+float32 Z+float64 coarse core',
        matrix_free=True, full_kernel_never_materialized=True, native_rtol=TOL,
        native_maxiter=2500, cpu_limit_seconds=90),
    source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS},
    evidence_sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [GOLD,POWER]}, cases=[])


def guard():
    if process_time()-START_CPU > 84:
        raise TimeoutError('84 CPU-second graceful limit reached')


def save():
    record.update(elapsed_cpu_seconds=process_time()-START_CPU,
        elapsed_wall_seconds=perf_counter()-START_WALL,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class Operator:
    """One live row-kernel block; direct matmul into preallocated output."""
    def __init__(self,kernel,phase_cap):
        self.kernel,self.n,self.phase_cap=kernel,kernel.n,int(phase_cap)
        self.calls=self.rhs=self.entries=0
        self.max_phase_bytes=0
        self.row_counts=set()

    def rows(self,retained,input_bytes,output_bytes):
        rows=(self.phase_cap-int(retained)-int(input_bytes)-int(output_bytes))//(8*self.n)
        if rows<1:raise MemoryError('No space for one kernel row')
        return min(self.n-1,int(rows))

    def apply(self,v,retained,*,regularized=True,out=None):
        output=np.empty_like(v) if out is None else out
        rows=self.rows(retained,v.nbytes,output.nbytes)
        phase=retained+v.nbytes+output.nbytes+8*rows*self.n
        assert phase<=self.phase_cap and rows<self.n
        self.max_phase_bytes=max(self.max_phase_bytes,int(phase))
        self.row_counts.add(rows)
        for first in range(0,self.n,rows):
            last=min(first+rows,self.n)
            block=self.kernel._block(self.kernel.X[first:last],self.kernel.X)
            np.matmul(block,v,out=output[first:last])
            del block
        if regularized:
            # Both arrays are C contiguous; BLAS axpy avoids a lambda*v array.
            linalg.blas.daxpy(v.ravel(),output.ravel(),a=LAM)
        self.calls+=1
        self.rhs+=1 if v.ndim==1 else v.shape[1]
        self.entries+=self.n*self.n
        return output


def compact_rp(kernel,rank):
    raw=initialize_rpcholesky(kernel,rank,LAM,
        np.random.default_rng(np.random.SeedSequence([0,6220,1])))
    compact=LowRankPreconditioner.from_nystrom(raw)
    del raw
    return compact


def prefix(op,b,state):
    x=np.zeros_like(b);r=b.copy();z=state.apply(r);p=z.copy();rho=float(r@z)
    stored=np.empty((len(b),M),dtype=np.float32)
    retained=state.storage_bytes+stored.nbytes+8*M*M
    for i in range(M):
        stored[:,i]=p/np.linalg.norm(p)
        ap=op.apply(p,retained)
        alpha=rho/float(p@ap);x+=alpha*p;r-=alpha*ap
        z=state.apply(r);newrho=float(r@z);p=z+(newrho/rho)*p;rho=newrho
    return x,r,stored


def native_cg(op,b,state):
    retained=0 if state is None else state.storage_bytes
    x=np.zeros_like(b);r=b.copy();norm_b=float(np.linalg.norm(b))
    z=r.copy() if state is None else state.apply(r)
    p=z.copy();rho=float(r@z);verify=0;replacements=0
    start_calls,start_rhs=op.calls,op.rhs
    for it in range(1,2501):
        if it%25==1:guard()
        ap=op.apply(p,retained);curvature=float(p@ap)
        assert min(curvature,rho)>0 and np.isfinite(curvature+rho)
        alpha=rho/curvature;x+=alpha*p;r-=alpha*ap
        suspected=np.linalg.norm(r)<=TOL*norm_b
        if it%20==0 or suspected or it==2500:
            true_r=b-op.apply(x,retained);verify+=1
            relative=float(np.linalg.norm(true_r)/norm_b)
            if relative<=TOL:break
            if suspected:
                r=true_r.copy();z=r.copy() if state is None else state.apply(r)
                p=z.copy();rho=float(r@z);replacements+=1;continue
        z=r.copy() if state is None else state.apply(r)
        newrho=float(r@z);p=z+(newrho/rho)*p;rho=newrho
    assert relative<=TOL
    return dict(iterations=it,converged=True,true_relative_residual=relative,
        verification_products=verify,scalar_operator_products=op.calls-start_calls,
        scalar_operator_rhs=op.rhs-start_rhs,residual_replacements=replacements)


def power_model(op,rank,depth):
    Q=np.random.default_rng(np.random.SeedSequence([0,6220,3])).normal(size=(op.n,rank))
    for stage in [2]+([2] if depth>=4 else [])+([4] if depth>=8 else []):
        for i in range(stage):
            Y=op.apply(Q,0,regularized=False)
            if i+1<stage:
                Q,_=linalg.qr(Y,mode='economic',check_finite=False)
                Q=np.ascontiguousarray(Q)
                del Y
        core=Q.T@Y
        del Q
        model=LowRankPreconditioner.from_sketch(Y,core,LAM)
        del Y,core
        if (stage==2 and depth==2) or op.calls>=depth:
            return model
        Q=model.basis.copy()
        del model
    return model


def micro(op,stored,vector,retained,*,setup):
    packed=np.empty((op.n,M+1),dtype=np.float64)
    output=np.empty_like(packed)
    core=np.empty((M,M))
    samples=[]
    for _ in range(7):
        guard();start=perf_counter()
        np.copyto(packed[:,0],vector)
        np.copyto(packed[:,1:],stored,casting='unsafe')
        op.apply(packed,retained,out=output)
        if setup:
            np.matmul(packed[:,1:].T,output[:,1:],out=core)
        samples.append(perf_counter()-start)
    return dict(samples_seconds=samples,minimum_seconds=min(samples),median_seconds=float(np.median(samples)),
        input_pack_bytes=packed.nbytes,output_pack_bytes=output.nbytes,
        row_block_size=op.rows(retained,packed.nbytes,output.nbytes)), output[:,0].copy()


def run():
    gold=json.loads(GOLD.read_text(encoding='utf-8'))
    assert gold['status']=='complete'
    for gold_case in gold['cases']:
        guard()
        name,k,n=gold_case['dataset'],gold_case['rank'],gold_case['n']
        path=ROOT/gold_case['prepared_input']
        with np.load(path) as data:
            X,b,bandwidth=data['X_train'],data['y_train'],float(data['bandwidth'])
        assert sha256(path.read_bytes()).hexdigest()==gold_case['prepared_sha256']
        kernel=GaussianKernel(X,bandwidth)
        base_bytes=8*n*k+16*k
        retained_budget=base_bytes+4*n*M+8*M*M
        ranks=dict(compact_RP=retained_budget//(8*n+16),power=retained_budget//(8*n+8))
        maximum_rank=max(ranks.values())
        svd_lwork=int(linalg.lapack.dgesdd_lwork(n,maximum_rank,compute_uv=1,full_matrices=0)[0])
        # Constructor envelope: images, factor, SVD work copy/output, returned
        # basis copy (conservative fifth n*r array), twelve small r*r arrays,
        # queried floating workspace and 8*r LAPACK integer workspace.
        construction_envelope=5*8*n*maximum_rank+12*8*maximum_rank**2+8*svd_lwork+4*8*maximum_rank
        phase_cap=max(retained_budget+8*n*(128+2),construction_envelope)
        row=dict(dataset=name,regularization=LAM,base_rank=k,n=n,status='running',
            prepared_input=str(path.relative_to(ROOT)),prepared_sha256=sha256(path.read_bytes()).hexdigest(),
            gold_total_iterations=gold_case['controls']['old_projected_PCG']['total_iterations'],
            gold_remaining_iterations=gold_case['controls']['old_projected_PCG']['remaining_iterations'],
            gold_verification_products=gold_case['controls']['old_projected_PCG']['verification_matvecs'],
            native_runs=[],memory=dict(compact_base_bytes=base_bytes,single_precision_basis_bytes=4*n*M,
                coarse_core_bytes=8*M*M,candidate_retained_bytes=retained_budget,matched_ranks=ranks,
                common_kernel_phase_cap_bytes=phase_cap,common_other_solver_vector_reserve_bytes=6*8*n,
                common_explicit_array_cap_with_solver_reserve=phase_cap+6*8*n,
                static_constructor_conservative_envelope_bytes=construction_envelope,
                queried_gesdd_lwork_doubles=svd_lwork,
                operator_arrays='retained state + live input + live output + one row_kernel; block is deleted before next allocation',
                candidate_arrays='compact base + float32 Z + float64 E + float64 packed[t,Z] + float64[At,AZ] + row_kernel',
                static_arrays='compact factor (or none) + float64 input_vector + float64 output_vector + row_kernel',
                scope='Exact operator-buffer arrays and conservative constructor envelope; common X/b, Python allocator state and process RSS are not per-method peak certificates'))
        record['cases'].append(row)
        op=Operator(kernel,phase_cap)
        # Warm the kernel/BLAS path before measuring setup and prefix.
        op.apply(np.ascontiguousarray(b),0)
        start=perf_counter();base=compact_rp(kernel,k);base_setup=perf_counter()-start
        assert base.storage_bytes==base_bytes and base.indices.tolist()==gold_case['initial_indices']
        start=perf_counter();x,r_recursive,Z32=prefix(op,b,base);prefix_time=perf_counter()-start
        setup,Ax=micro(op,Z32,x,retained_budget,setup=True)
        true_r=b-Ax
        direction=base.apply(true_r)
        continuation,_=micro(op,Z32,direction,retained_budget,setup=False)
        scalar_samples=[]
        for _ in range(7):
            start=perf_counter();op.apply(x,retained_budget);scalar_samples.append(perf_counter()-start)
        floor=base_setup+prefix_time+setup['minimum_seconds']+\
            row['gold_remaining_iterations']*continuation['minimum_seconds']+\
            row['gold_verification_products']*min(scalar_samples)
        row['candidate_cost_estimate']=dict(base_setup_seconds=base_setup,prefix20_seconds=prefix_time,
            prefix_direction_products=M,setup=setup,continuation=continuation,
            verification_scalar_samples_seconds=scalar_samples,
            true_prefix_relative_residual=float(np.linalg.norm(true_r)/np.linalg.norm(b)),
            optimistic_total_seconds=floor,
            required_kernel_passes=M+1+row['gold_remaining_iterations']+row['gold_verification_products'],
            required_block_rhs=(M+1)*(1+row['gold_remaining_iterations']),
            initial_column_queries=k,post_prefix_basis_rank_assumed=M,
            quantized_convergence_assumed_equal_to_gold=True,quantized_solver_executed=False,
            actual_measured_operator_peak_bytes=op.max_phase_bytes,
            prefix_scalar_row_block_size=op.rows(retained_budget,8*n,8*n),
            verification_scalar_row_block_size=op.rows(retained_budget,8*n,8*n))
        del base,x,r_recursive,Z32,Ax,true_r,direction,op
        save()
        for repeat in range(2):
            order=list(np.random.default_rng(622077+repeat+7*k).permutation(
                ['CG','compact_RP','power2','power4','power8']))
            for method in order:
                guard();op=Operator(kernel,phase_cap)
                before_columns=int(kernel.stats['column_evaluations'])
                begin=perf_counter()
                if method=='CG':state=None;rank=0
                elif method=='compact_RP':
                    rank=int(ranks['compact_RP']);state=compact_rp(kernel,rank)
                else:
                    rank=int(ranks['power']);state=power_model(op,rank,int(method[5:]))
                setup_seconds=perf_counter()-begin
                setup_passes,setup_rhs=op.calls,op.rhs
                state_bytes=0 if state is None else state.storage_bytes
                assert state_bytes<=retained_budget
                solved=native_cg(op,b,state)
                total_seconds=perf_counter()-begin
                solved.update(method=str(method),repeat=repeat,rank=rank,
                    setup_seconds=setup_seconds,total_seconds=total_seconds,
                    solve_seconds=total_seconds-setup_seconds,retained_state_bytes=state_bytes,
                    setup_column_queries=int(kernel.stats['column_evaluations'])-before_columns,
                    setup_kernel_passes=setup_passes,setup_block_rhs=setup_rhs,
                    total_kernel_passes=op.calls,total_kernel_rhs=op.rhs,
                    solve_row_block_size=op.rows(state_bytes,8*n,8*n),
                    all_row_block_sizes=sorted(op.row_counts),actual_operator_peak_bytes=op.max_phase_bytes)
                row['native_runs'].append(solved)
                del state,op
                save()
                print(json.dumps(dict(dataset=name,k=k,method=str(method),repeat=repeat,
                    iterations=solved['iterations'],seconds=total_seconds,candidate_floor=floor)),flush=True)
        summary={}
        for method in ['CG','compact_RP','power2','power4','power8']:
            values=[r['total_seconds'] for r in row['native_runs'] if r['method']==method]
            summary[method]=dict(min_seconds=min(values),median_seconds=float(np.median(values)),max_seconds=max(values),
                optimistic_candidate_to_median=floor/float(np.median(values)),
                optimistic_candidate_to_slower_repeat=floor/max(values))
        best=min(summary,key=lambda key:summary[key]['median_seconds'])
        row.update(status='complete',comparisons=summary,best_native_control=best,
            cost_falsified_vs_best_slower_repeat=bool(floor>summary[best]['max_seconds']))
        save()
        print(json.dumps(dict(case=name,k=k,decision=row['cost_falsified_vs_best_slower_repeat'],
            best=best,ratio=floor/summary[best]['max_seconds'])),flush=True)


if __name__=='__main__':
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_CPU,(89,90))
    try:
        with threadpool_limits(limits=1):run()
        record['status']='complete'
        record['all_cases_cost_falsified']=all(c['cost_falsified_vs_best_slower_repeat'] for c in record['cases'])
        for p in SOURCE_PATHS:
            assert sha256(p.read_bytes()).hexdigest()==record['source_sha256'][str(p.relative_to(ROOT))]
    except TimeoutError as error:record.update(status='cpu_budget_exhausted',error=str(error))
    except Exception as error:
        record.update(status='failed',error=repr(error));save();raise
    save()
