#!/usr/bin/env python3
"""Offline sequential GBCD/history controls and compact static memory controls."""
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner
from check_schur_fsai import schur_blocks
from check_projected_balancing import pcg
from check_projected_coordinate_geometry import PairHistory,verify_score_formula


def sequential(case,memory,cpu_start):
    A,D,b,perm,L,B=[case[k] for k in ('A','D','b','permutation','L','B')]
    n=len(b);nr=len(D);width=16;norm_b=np.linalg.norm(b)
    t=linalg.solve_triangular(L,b[perm[:16]],lower=True,check_finite=False)
    q=b[perm[16:]]-B@t;x=np.zeros(nr);r=q.copy();diagonal=np.diag(D).copy()
    old=PairHistory(nr,memory) if memory else None;local=PairHistory(nr,width)
    budget=case['fixed']['operator_applications']*n
    queries=verifications=0;last_verified=-1;true_rel=float(np.linalg.norm(q)/norm_b)
    best=true_rel;last_improvement=0;progress=[];stop='max_blocks';replacement=0
    timings=dict(score_seconds=0.,old_projection_seconds=0.,withinblock_projection_seconds=0.,
                 history_update_seconds=0.,verification_seconds=0.)
    started=perf_counter();cpu_started=process_time();max_score_error=0.
    def verify(block):
        nonlocal verifications,last_verified,true_rel,best,last_improvement
        start=perf_counter()
        top=linalg.solve_triangular(L.T,t-B.T@x,lower=False,check_finite=False)
        full=np.empty(n);full[perm[:16]]=top;full[perm[16:]]=x
        residual=b-A@full;verifications+=1;last_verified=block;true_rel=float(np.linalg.norm(residual)/norm_b)
        if true_rel<.99*best:best=true_rel;last_improvement=queries
        progress.append(dict(block=block,column_queries=queries,verification_matvecs=verifications,
            true_relative_residual=true_rel,recursive_gap=float(np.linalg.norm(residual[perm[16:]]-r)/norm_b)))
        timings['verification_seconds']+=perf_counter()-start
        return residual[perm[16:]]
    for block in range(1,10001):
        if process_time()-cpu_start>63:stop='global_cpu_guard';block-=1;break
        count=min(width,budget-queries-verifications*n-n)
        if count<=0:stop='entry_query_budget';block-=1;break
        local.clear();step=np.zeros(nr);image=np.zeros(nr);groups=([old] if old is not None else [])+[local]
        for _ in range(count):
            start=perf_counter();eta=r.copy();tau=diagonal.copy()
            for history in groups:
                eta-=history.numerator_correction(r);tau-=history.diagonal
            floor=256*np.finfo(float).eps*np.maximum(np.maximum(diagonal,np.abs(diagonal-tau)),np.finfo(float).tiny)
            good=tau>floor;score=np.full(nr,-np.inf);score[good]=eta[good]**2/tau[good]
            j=int(np.argmax(score));timings['score_seconds']+=perf_counter()-start
            if not np.isfinite(score[j]) or score[j]<=0:stop='no_positive_score';break
            p=np.zeros(nr);p[j]=1.;ap=D[:,j].copy();queries+=1
            # First coefficients are cached A-images of e_j; one reorthogonalization follows.
            for first in (True,False):
                for history in groups:
                    if not history.size:continue
                    begin=perf_counter();P=history.P[:,:history.size];AP=history.AP[:,:history.size]
                    coefficients=(AP[j,:].copy() if first else P.T@ap)/history.curv[:history.size]
                    p-=P@coefficients;ap-=AP@coefficients
                    timings['old_projection_seconds' if history is old else 'withinblock_projection_seconds']+=perf_counter()-begin
            curvature=float(p@ap);numerator=float(r@p)
            if curvature<=0:stop='nonpositive_curvature';break
            actual=numerator*numerator/curvature
            max_score_error=max(max_score_error,abs(score[j]-actual)/max(actual,1e-300))
            alpha=numerator/curvature;x+=alpha*p;r-=alpha*ap;step+=alpha*p;image+=alpha*ap
            begin=perf_counter();local.append(p,ap);timings['history_update_seconds']+=perf_counter()-begin
        if stop not in ('max_blocks',):break
        if old is not None:
            begin=perf_counter();old.append(step,image);timings['history_update_seconds']+=perf_counter()-begin
        suspected=np.linalg.norm(r)<=1e-8*norm_b
        exhausted=queries+verifications*n+n+1>budget
        if block%50==0 or suspected or exhausted or block==10000:
            rr=verify(block)
            if true_rel<=1e-8:stop='converged';break
            if suspected:
                r=rr;replacement+=1
                if old is not None:old.clear()
            if queries-last_improvement>=5000 and true_rel>1e-6:stop='stagnant_verified_residual';break
            if exhausted:stop='entry_query_budget';break
    if last_verified!=block:verify(block)
    assert queries+verifications*n<=budget
    coarse=perm.nbytes+L.nbytes+B.nbytes;history=old.storage_bytes if old is not None else 0
    retained=coarse+history;peak=retained+local.storage_bytes
    return dict(history_pairs=memory,block_size=16,retention='combined_block_step' if memory else 'none_GBCD',
        blocks=block,converged=bool(true_rel<=1e-8),true_relative_residual=true_rel,stop_reason=stop,
        column_queries=queries,verification_matvecs=verifications,column_equivalent_work=queries+verifications*n,
        equivalent_full_entry_passes=float(queries/n+verifications),query_budget=budget,
        retained_coarse_and_history_bytes=int(retained),temporary_withinblock_history_bytes=int(local.storage_bytes),
        coarse_history_withinblock_target_bytes=int(peak),
        retained_matched_rank=int(ceil(retained/(8*n))),peak_target_matched_rank=int(ceil(peak/(8*n))),
        additional_major_vector_workspace_bytes=int(12*nr*8),
        max_projected_score_relative_error=max_score_error,residual_replacements=replacement,
        exploratory_dense_wall_seconds=perf_counter()-started,exploratory_cpu_seconds=process_time()-cpu_started,
        timing_breakdown=timings,progress=progress)


def static_control(case,k,kind):
    start=perf_counter();n=len(case['b'])
    if kind=='compact_rp':
        base=initialize_rpcholesky(case['kernel'],k,.001,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
        state=LowRankPreconditioner.from_nystrom(base);del base
        setup_columns=k;setup_passes=0
    else:
        omega=np.random.default_rng(np.random.SeedSequence([0,6220,3])).normal(size=(n,k))
        first=case['K']@omega;Q,_=linalg.qr(first,mode='economic',check_finite=False)
        images=case['K']@Q;state=LowRankPreconditioner.from_sketch(images,Q.T@images,.001)
        del omega,first,Q,images
        setup_columns=0;setup_passes=2
    setup_seconds=perf_counter()-start
    start=perf_counter();cpu_start=process_time()
    solved,_=pcg(case['A'],case['b'],state,maxiter=2500)
    return dict(kind=kind,requested_rank=k,effective_rank=state.rank,**solved,
        retained_factor_bytes=int(state.storage_bytes),setup_kernel_column_queries=setup_columns,
        setup_full_kernel_matmat_passes=setup_passes,
        equivalent_full_entry_passes=float(solved['operator_applications']+setup_passes+setup_columns/n),
        exploratory_dense_solve_seconds=perf_counter()-start,exploratory_solve_cpu_seconds=process_time()-cpu_start,
        offline_setup_seconds=setup_seconds,
        setup_note='RP initialization uses real kernel columns; Gaussian power products use cached dense K; setup times are not production-comparable')


def main():
    output=root/'results/coordinate-static-controls-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(67,69));started=perf_counter();cpu_start=process_time()
    result=dict(status='running',scope='Dense exploratory mechanism/memory controls, no production speed claim',
        protocol=dict(regularization=.001,landmarks=16,block_size=16,history_pairs=[0,8,16],
            withinblock_pairs=16,selection='exact projected scalar greedy',projection='cached first projection plus one reorthogonalization',
            retention='combined block step',rtol=1e-8,max_blocks=10000,verification_interval=50,
            query_budget='ordinary fixed RP16 operator_applications*n; includes full verification cost',
            memory_rank_rule='ceil(target_bytes/(8*n)); targets retained-only and retained+withinblock pairs',
            memory_scope='actual static factor bytes recorded; static setup workspaces and ordinary work vectors not peak-matched',
            blas_threads=1,cpu_limit_seconds=67),
        source_sha256={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),root/'scripts/check_projected_coordinate_geometry.py',root/'scripts/check_schur_fsai.py',
             root/'scripts/check_projected_balancing.py',root/'src/nystrom_project/preconditioners.py',root/'src/nystrom_project/numerics.py']},cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['score_check']=verify_score_formula()
        for name in ('synthetic-n640-smooth','concrete-observed','airfoil-observed'):
            data=np.load(root/f'data/prepared/{name}-s0-lambda0.001.npz');b=data['y_train']
            kernel=GaussianKernel(data['X_train'],float(data['bandwidth']));K=kernel.dense();A=K+.001*np.eye(len(b))
            base=initialize_rpcholesky(kernel,16,.001,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
            fixed,_=pcg(A,b,base,maxiter=2500);perm,L,B,D=schur_blocks(A,base.indices);del base
            case=dict(name=name,kernel=kernel,K=K,A=A,D=D,b=b,permutation=perm,L=L,B=B,fixed=fixed)
            row=dict(dataset=name,n=len(b),fixed_rp16=fixed,sequential=[],static_controls=[]);result['cases'].append(row)
            for history in (0,8,16):
                if process_time()-cpu_start>60:result['status']='budget_exhausted';save();return
                value=sequential(case,history,cpu_start);row['sequential'].append(value);save()
                print(json.dumps(dict(dataset=name,history=history,**{k:value[k] for k in
                    ('blocks','converged','true_relative_residual','stop_reason','equivalent_full_entry_passes',
                     'retained_matched_rank','peak_target_matched_rank','exploratory_cpu_seconds')})),flush=True)
            ranks=sorted(set(v[k] for v in row['sequential'] for k in ('retained_matched_rank','peak_target_matched_rank')))
            for rank in ranks:
                for kind in ('compact_rp','power_gaussian'):
                    if process_time()-cpu_start>63:result['status']='budget_exhausted';save();return
                    value=static_control(case,rank,kind);row['static_controls'].append(value);save()
                    print(json.dumps(dict(dataset=name,**{k:value[k] for k in ('kind','requested_rank','iterations','converged',
                        'equivalent_full_entry_passes','retained_factor_bytes','exploratory_dense_solve_seconds')})),flush=True)
        result['status']='complete';save();print(json.dumps(dict(status=result['status'],cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__=='__main__':main()
