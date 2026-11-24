#!/usr/bin/env python3
"""Dense offline test of greedy Schur coordinate-block corrections.

Iteration operators use selected cached Schur columns, not Schur matvecs.
Reported kernel-column query counts describe a future uncached implementation;
they are not timings. Original-system verification matvecs are counted exactly.
History stores one normalized combined correction per block, at most m pairs.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
from nystrom_project.flexible import AConjugateHistory
from check_schur_fsai import schur_blocks
from check_projected_balancing import pcg


def project_inplace(Z,AZ,history,scratch):
    for _ in range(2):
        for p,ap,curvature in history._pairs:
            coefficients=(p@AZ)/curvature
            np.multiply(p[:,None],coefficients[None,:],out=scratch);Z-=scratch
            np.multiply(ap[:,None],coefficients[None,:],out=scratch);AZ-=scratch


def block_coefficients(Z,AZ,r):
    gram=Z.T@AZ;gram=(gram+gram.T)*.5
    values,vectors=linalg.eigh(gram,check_finite=False)
    threshold=64*np.finfo(float).eps*len(values)*max(float(values[-1]),np.finfo(float).tiny)
    keep=values>threshold
    if not np.any(keep) or values[0]<-threshold:
        return None,dict(reason='indefinite_or_empty_projected_block',minimum_eigenvalue=float(values[0]))
    rhs=Z.T@r
    coefficients=vectors[:,keep]@((vectors[:,keep].T@rhs)/values[keep])
    return coefficients,dict(effective_rank=int(np.count_nonzero(keep)),
        energy_decrease=float(2*(rhs@coefficients)-coefficients@gram@coefficients))


def solve_blocks(A,D,b,permutation,L,B,block_size,history_size,max_blocks=2000):
    n=len(b);k=len(L);m=len(D);width=min(block_size,m)
    norm_b=np.linalg.norm(b);top_rhs=linalg.solve_triangular(L,b[permutation[:k]],lower=True,check_finite=False)
    q=b[permutation[k:]]-B@top_rhs
    x=np.zeros(m);r=q.copy();history=AConjugateHistory(m,history_size)
    diagonal=np.diag(D).copy();assert np.all(diagonal>0)
    Z=np.empty((m,width));AZ=np.empty_like(Z);scratch=np.empty_like(Z);scores=np.empty(m)
    queries=verifications=replacements=0;maximum_history=0;breakdown=None;converged=False
    progress=[];rank_deficient_blocks=0;minimum_decrease=np.inf;max_conjugacy=0.
    last_verified=-1;true_relative=float(np.linalg.norm(q)/norm_b);top_residual=0.
    def verify(block):
        nonlocal verifications,last_verified,true_relative,top_residual
        top=linalg.solve_triangular(L.T,top_rhs-B.T@x,lower=False,check_finite=False)
        full_x=np.empty(n);full_x[permutation[:k]]=top;full_x[permutation[k:]]=x
        full_residual=b-A@full_x;verifications+=1;last_verified=block
        true_relative=float(np.linalg.norm(full_residual)/norm_b)
        top_residual=float(np.linalg.norm(full_residual[permutation[:k]])/norm_b)
        return full_residual[permutation[k:]]
    for block in range(1,max_blocks+1):
        np.multiply(r,r,out=scores);scores/=diagonal
        if width==m:chosen=np.arange(m)
        else:
            trial=np.argpartition(scores,m-width)[m-width:];boundary=scores[trial].min()
            above=np.flatnonzero(scores>boundary);tied=np.flatnonzero(scores==boundary)
            chosen=np.sort(np.r_[above,tied[:width-len(above)]])
        Z.fill(0)
        for column,index in enumerate(chosen):
            Z[index,column]=1
            AZ[:,column]=D[:,index]
        queries+=width
        project_inplace(Z,AZ,history,scratch)
        coefficients,details=block_coefficients(Z,AZ,r)
        if coefficients is None:
            breakdown=details;block-=1;break
        correction=Z@coefficients;image=AZ@coefficients
        curvature=float(correction@image)
        if curvature<=0 or details['energy_decrease']<=0:
            breakdown=dict(reason='nonpositive_block_correction');block-=1;break
        if history.size:
            max_conjugacy=max(max_conjugacy,max(abs(p@image)/np.sqrt(curvature*c) for p,_,c in history._pairs))
        minimum_decrease=min(minimum_decrease,details['energy_decrease'])
        rank_deficient_blocks+=details['effective_rank']<width
        x+=correction;r-=image;history.append(correction,image)
        maximum_history=max(maximum_history,history.storage_bytes)
        recursive_relative=float(np.linalg.norm(r)/norm_b)
        suspected=recursive_relative<=1e-8
        if block%50==0 or suspected or block==max_blocks:
            true_r=verify(block)
            progress.append(dict(block=block,recursive_relative_residual=recursive_relative,
                true_original_relative_residual=true_relative,
                residual_gap_relative_original_rhs=float(np.linalg.norm(r-true_r)/norm_b),
                kernel_column_queries=queries,verification_matvecs=verifications))
            if true_relative<=1e-8:converged=True;break
            if suspected:
                r=true_r;history.clear();replacements+=1
    if last_verified!=block:verify(block)
    coarse_bytes=int(permutation.nbytes+L.nbytes+B.nbytes)
    # Three n_R-by-b arrays (Z,AZ,projection scratch) and the combined step/image.
    major_block_workspace_bytes=int(3*m*width*8+2*m*8)
    return dict(blocks=block,converged=bool(true_relative<=1e-8),true_relative_residual=true_relative,
        eliminated_equations_relative_residual=top_residual,breakdown=breakdown,
        block_size=width,history_pairs=history_size,history_mode='one_combined_step_per_block',
        equivalent_kernel_column_queries=queries,landmark_column_queries=k,
        original_system_verification_matvecs=verifications,
        equivalent_full_kernel_entry_passes=float((queries+k)/n+verifications),
        rank_deficient_blocks=int(rank_deficient_blocks),residual_replacements=replacements,
        minimum_predicted_energy_decrease=float(minimum_decrease) if np.isfinite(minimum_decrease) else None,
        maximum_new_step_history_A_conjugacy=max_conjugacy,
        coarse_factor_state_bytes=coarse_bytes,maximum_retained_history_bytes=maximum_history,
        maximum_retained_factor_and_history_bytes=coarse_bytes+maximum_history,
        major_block_workspace_bytes=major_block_workspace_bytes,
        remaining_iterate_residual_rhs_score_diagonal_bytes=int(5*m*8),
        workspace_note='Small b-by-b eigensolver arrays and full-system verification vectors are additional; dense diagnostic A/D excluded',
        progress=progress)


def small_checks():
    rng=np.random.default_rng(634);n=19
    C=rng.normal(size=(n,n));S=C@C.T+.2*np.eye(n)
    L=linalg.cholesky(S,lower=True);Q,_=linalg.qr(rng.normal(size=(n,3)),mode='economic')
    P=linalg.solve_triangular(L.T,Q,lower=False);AP=S@P
    history=AConjugateHistory(n,3)
    for p,ap in zip(P.T,AP.T):history.append(p,ap)
    r=rng.normal(size=n);r-=AP@(P.T@r)
    chosen=np.array([1,4,8,13]);Z=np.eye(n)[:,chosen];AZ=S[:,chosen].copy()
    project_inplace(Z,AZ,history,np.empty_like(Z))
    coefficients,details=block_coefficients(Z,AZ,r)
    step=Z@coefficients;image=AZ@coefficients
    actual_energy_decrease=float(2*r@step-step@S@step)
    errors=dict(projected_A_image=float(np.linalg.norm(image-S@step)/np.linalg.norm(image)),
        exact_block_optimality=float(np.linalg.norm(Z.T@(r-S@step))),
        retained_A_conjugacy=float(np.max(np.abs(P.T@(S@step)))),
        relative_energy_decrease_error=abs(actual_energy_decrease-details['energy_decrease'])/actual_energy_decrease)
    assert max(errors.values())<1e-10
    kernel=GaussianKernel(rng.normal(size=(17,3)),1.1);K=kernel.dense();A=K+.03*np.eye(17)
    permutation,L,B,D=schur_blocks(A,np.array([9,2,15,0]))
    solved=solve_blocks(A,D,rng.normal(size=17),permutation,L,B,13,0,max_blocks=2)
    assert solved['converged'] and solved['blocks']==1
    errors['full_remaining_block_true_residual']=solved['true_relative_residual']
    return errors


def main(lam,max_blocks,output):
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    started,cpu_started=perf_counter(),process_time()
    result=dict(status='running',scope='Dense offline direct-correction study; equivalent query counts are not production measurements',
        protocol=dict(seed=0,regularization=lam,landmarks=16,block_sizes=[8,16],history_pairs=[0,8,16],
            selection='simultaneous largest recursive Schur residual_i squared / Schur diagonal_i',
            history='one combined A-normalized correction per block, FIFO at requested pair cap',
            rtol=1e-8,stopping_denominator='norm(original full-system rhs)',max_blocks=max_blocks,
            verification_interval=50,verification_operator='original full A, with recovered eliminated coordinates',
            production_column_model='K[:,J] minus coarse product and regularization diagonal',
            iteration_dense_access='Schur columns only; no full Schur matvec',blas_threads=1,cpu_limit_seconds=110),
        source_sha256={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),root/'scripts/check_schur_fsai.py',root/'scripts/check_projected_balancing.py',
             root/'src/nystrom_project/flexible.py',root/'src/nystrom_project/numerics.py']},workloads=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['small_independent_checks']=small_checks()
        for name in ('synthetic-n640-smooth','concrete-observed','airfoil-observed'):
            prepared=root/f'data/prepared/{name}-s0-lambda0.001.npz'
            data=np.load(prepared);b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']))
            K=kernel.dense();A=K+lam*np.eye(len(b))
            base=initialize_rpcholesky(kernel,16,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
            fixed,_=pcg(A,b,base,maxiter=2500)
            permutation,L,B,D=schur_blocks(A,base.indices)
            workload=dict(dataset=name,n=len(b),prepared_feature_sha256=sha256(prepared.read_bytes()).hexdigest(),
                fixed_rp=dict(**fixed,landmark_column_queries=16,equivalent_full_kernel_entry_passes=float(fixed['operator_applications']+16/len(b))),variants=[])
            result['workloads'].append(workload)
            del base
            for block_size in (8,16):
                for history_size in (0,8,16):
                    if process_time()-cpu_started>95:
                        result['status']='budget_exhausted';save();return
                    solved=solve_blocks(A,D,b,permutation,L,B,block_size,history_size,max_blocks=max_blocks)
                    workload['variants'].append(solved);save()
                    print(json.dumps(dict(dataset=name,b=block_size,m=history_size,
                        **{key:solved[key] for key in ('blocks','converged','true_relative_residual','equivalent_kernel_column_queries',
                            'original_system_verification_matvecs','equivalent_full_kernel_entry_passes')},
                        fixed_iterations=fixed['iterations'],fixed_matvecs=fixed['operator_applications'])),flush=True)
        result['status']='complete';save();print(json.dumps(dict(status=result['status'],cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--regularization',type=float,default=.001)
    parser.add_argument('--max-blocks',type=int,default=2000)
    parser.add_argument('--output',type=Path,default=root/'results/block-correction-diagnostic.json')
    args=parser.parse_args();main(args.regularization,args.max_blocks,args.output)
