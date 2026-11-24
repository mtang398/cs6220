#!/usr/bin/env python3
"""Offline projected-greedy and sequential Schur correction controls.

Column counts are equivalent uncached kernel queries; dense Schur storage is
diagnostic only. Every convergence claim uses the original full-system residual.
"""
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
from check_schur_fsai import schur_blocks
from check_projected_balancing import pcg
from check_block_corrections import block_coefficients


class PairHistory:
    def __init__(self,n,capacity):
        self.P=np.empty((n,capacity),order='F');self.AP=np.empty_like(self.P)
        self.curv=np.empty(capacity);self.diagonal=np.zeros(n)
        self.capacity=capacity;self.size=0;self.next=0;self.insertions=0
    @property
    def storage_bytes(self):
        return self.P.nbytes+self.AP.nbytes+self.curv.nbytes+self.diagonal.nbytes
    def clear(self):
        self.size=self.next=0;self.diagonal.fill(0)
    def append(self,p,ap):
        if not self.capacity:return
        curvature=float(p@ap)
        if curvature<=0:raise np.linalg.LinAlgError('nonpositive history direction')
        slot=self.size if self.size<self.capacity else self.next
        if self.size==self.capacity:
            self.diagonal-=self.AP[:,slot]**2/self.curv[slot]
            self.next=(self.next+1)%self.capacity
        else:self.size+=1
        scale=np.sqrt(curvature);self.P[:,slot]=p/scale;self.AP[:,slot]=ap/scale
        self.curv[slot]=self.P[:,slot]@self.AP[:,slot]
        self.diagonal+=self.AP[:,slot]**2/self.curv[slot]
        self.insertions+=1
        if self.insertions%128==0:
            self.diagonal[:]=np.einsum('ij,j,ij->i',self.AP[:,:self.size],1/self.curv[:self.size],self.AP[:,:self.size])
    def numerator_correction(self,r):
        if not self.size:return np.zeros_like(r)
        return self.AP[:,:self.size]@((self.P[:,:self.size].T@r)/self.curv[:self.size])
    def project_inplace(self,p,ap):
        if not self.size:return
        for _ in range(2):
            c=(self.P[:,:self.size].T@ap)
            if p.ndim==1:c/=self.curv[:self.size]
            else:c/=self.curv[:self.size,None]
            p-=self.P[:,:self.size]@c;ap-=self.AP[:,:self.size]@c


def projected_scores(r,diagonal,histories):
    numerator=r.copy();tau=diagonal.copy()
    for history in histories:
        numerator-=history.numerator_correction(r);tau-=history.diagonal
    floor=256*np.finfo(float).eps*np.maximum(np.maximum(diagonal,np.abs(diagonal-tau)),np.finfo(float).tiny)
    good=tau>floor
    scores=np.full(len(r),-np.inf);scores[good]=numerator[good]**2/tau[good]
    return scores,tau,numerator


def project_groups(p,ap,histories):
    # Two passes through both mutually conjugate groups. Each group's own
    # two-pass projector adds reorthogonalization without any operator query.
    for _ in range(2):
        for history in histories:history.project_inplace(p,ap)


def verify_score_formula():
    rng=np.random.default_rng(635);C=rng.normal(size=(21,21));S=C@C.T+.4*np.eye(21)
    Q,_=linalg.qr(rng.normal(size=(21,4)),mode='economic')
    L=linalg.cholesky(S,lower=True);P=linalg.solve_triangular(L.T,Q,lower=False)
    history=PairHistory(21,4)
    for p in P.T:history.append(p,S@p)
    r=rng.normal(size=21)  # Deliberately not orthogonal to history.
    scores,_,_=projected_scores(r,np.diag(S),(history,))
    direct=[]
    for j in range(21):
        e=np.eye(21)[:,j];p=e-P@(P.T@S[:,j])
        direct.append(float((r@p)**2/(p@S@p)))
    error=float(np.max(np.abs(scores-direct))/max(direct))
    assert error<1e-12
    return dict(projected_scalar_score_max_relative_error=error)


def run(case,block_size,history_size,mode,global_cpu_start):
    A,D,b,permutation,L,B=[case[k] for k in ('A','D','b','permutation','L','B')]
    n=len(b);nr=len(D);k=len(L);norm_b=np.linalg.norm(b);diagonal=np.diag(D).copy()
    top_rhs=linalg.solve_triangular(L,b[permutation[:k]],lower=True,check_finite=False)
    q=b[permutation[k:]]-B@top_rhs;x=np.zeros(nr);r=q.copy()
    history=PairHistory(nr,history_size)
    temporary=PairHistory(nr,block_size) if mode.startswith('sequential') else None
    query_budget=case['fixed']['operator_applications']*n
    queries=checks=steps=0;stop='max_blocks';true_rel=np.linalg.norm(q)/norm_b;last_verified=-1
    best=true_rel;last_improvement=0;progress=[];replacements=0;top_residual=0.
    rank_deficient=0;max_geometry_error=0.;minimum_decrease=np.inf
    started=perf_counter();cpu_started=process_time()
    def verify(block):
        nonlocal checks,true_rel,last_verified,top_residual,best,last_improvement
        xS=linalg.solve_triangular(L.T,top_rhs-B.T@x,lower=False,check_finite=False)
        whole=np.empty(n);whole[permutation[:k]]=xS;whole[permutation[k:]]=x
        residual=b-A@whole;checks+=1;last_verified=block
        true_rel=float(np.linalg.norm(residual)/norm_b)
        top_residual=float(np.linalg.norm(residual[permutation[:k]])/norm_b)
        if true_rel<.99*best:best=true_rel;last_improvement=queries
        progress.append(dict(block=block,coordinate_steps=steps,column_queries=queries,
            verification_matvecs=checks,true_relative_residual=true_rel,
            recursive_residual_gap=float(np.linalg.norm(residual[permutation[k:]]-r)/norm_b)))
        return residual[permutation[k:]]
    for block in range(1,10001):
        if process_time()-global_cpu_start>140:stop='global_cpu_guard';block-=1;break
        available=query_budget-queries-checks*n-n  # Reserve final verification.
        width=min(block_size,available)
        if width<=0:stop='entry_query_budget';block-=1;break
        if mode=='simultaneous':
            scores,_,_=projected_scores(r,diagonal,(history,))
            valid=np.flatnonzero(np.isfinite(scores));width=min(width,len(valid))
            if not width:stop='no_independent_coordinate';block-=1;break
            order=np.argsort(scores[valid],kind='stable')[-width:];chosen=np.sort(valid[order])
            Z=np.zeros((nr,width));AZ=np.empty_like(Z)
            for column,j in enumerate(chosen):Z[j,column]=1;AZ[:,column]=D[:,j]
            queries+=width;steps+=width;history.project_inplace(Z,AZ)
            delta,details=block_coefficients(Z,AZ,r)
            if delta is None:stop='degenerate_block';block-=1;break
            correction=Z@delta;image=AZ@delta
            if correction@image<=0:stop='nonpositive_correction';block-=1;break
            minimum_decrease=min(minimum_decrease,details['energy_decrease'])
            rank_deficient+=details['effective_rank']<width
            x+=correction;r-=image;history.append(correction,image)
        else:
            if temporary is not None:temporary.clear()
            correction=np.zeros(nr);combined_image=np.zeros(nr)
            success=True
            for local in range(width):
                groups=(history,temporary) if temporary is not None else (history,)
                scores,tau,numerator=projected_scores(r,diagonal,groups)
                j=int(np.argmax(scores))
                if not np.isfinite(scores[j]) or scores[j]<=0:
                    stop='no_positive_projected_score';success=False;break
                p=np.zeros(nr);p[j]=1;ap=D[:,j].copy();queries+=1;steps+=1
                project_groups(p,ap,groups)
                curvature=float(p@ap);num=float(r@p)
                if curvature<=0:stop='nonpositive_coordinate_curvature';success=False;break
                actual_score=num*num/curvature
                max_geometry_error=max(max_geometry_error,abs(actual_score-scores[j])/max(actual_score,1e-300))
                alpha=num/curvature;x+=alpha*p;r-=alpha*ap
                minimum_decrease=min(minimum_decrease,actual_score)
                correction+=alpha*p;combined_image+=alpha*ap
                if temporary is None:history.append(p,ap)
                else:temporary.append(p,ap)
            if temporary is not None and success:
                if mode=='sequential_all':
                    for i in range(temporary.size):history.append(temporary.P[:,i],temporary.AP[:,i])
                else:history.append(correction,combined_image)
            if not success:break
        suspected=np.linalg.norm(r)<=1e-8*norm_b
        budget_exhausted=queries+checks*n+n+1>query_budget
        if block%50==0 or suspected or block==10000 or budget_exhausted:
            true_r=verify(block)
            if true_rel<=1e-8:stop='converged';break
            if suspected:r=true_r;history.clear();replacements+=1
            if queries-last_improvement>=5000 and true_rel>1e-6:
                stop='stagnant_verified_residual';break
            if budget_exhausted:stop='entry_query_budget';break
    if last_verified!=block:verify(block)
    assert queries+checks*n<=query_budget
    coarse_bytes=permutation.nbytes+L.nbytes+B.nbytes
    temporary_bytes=temporary.storage_bytes if temporary is not None else 0
    if mode=='simultaneous':major_workspace=3*nr*block_size*8+2*nr*8
    else:major_workspace=4*nr*8
    return dict(mode=mode,block_size=block_size,history_pairs=history_size,blocks=block,coordinate_steps=steps,
        stop_reason=stop,converged=bool(true_rel<=1e-8),true_relative_residual=true_rel,
        eliminated_equation_relative_residual=top_residual,column_queries=queries,
        full_original_verification_matvecs=checks,column_equivalent_work=queries+checks*n,
        column_equivalent_budget=query_budget,equivalent_full_entry_passes=float(queries/n+checks),
        exploratory_dense_wall_seconds=perf_counter()-started,exploratory_cpu_seconds=process_time()-cpu_started,
        retained_history_allocation_bytes=int(history.storage_bytes),coarse_factor_bytes=int(coarse_bytes),
        retained_coarse_and_history_bytes=int(coarse_bytes+history.storage_bytes),
        temporary_withinblock_history_bytes=int(temporary_bytes),major_direction_workspace_bytes=int(major_workspace),
        other_remaining_vector_workspace_bytes=int(8*nr*8),
        workspace_note='Small block eigensolver buffers and full-system verification vectors additional; dense A/D diagnostic storage excluded',
        max_projected_score_relative_error=max_geometry_error,
        minimum_energy_decrease=float(minimum_decrease) if np.isfinite(minimum_decrease) else None,
        rank_deficient_blocks=int(rank_deficient),residual_replacements=replacements,progress=progress)


def main():
    output=root/'results/projected-coordinate-geometry-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(145,149));started=perf_counter();cpu_start=process_time()
    result=dict(status='running',scope='Dense offline coordinate-geometry controls; no production runtime claim',
        protocol=dict(regularization=.001,landmarks=16,max_blocks=10000,rtol=1e-8,
            stopping_denominator='norm(original rhs)',verification_interval_blocks=50,
            budget='fixed-RP original A matvec count*n; selected column and each full verification=n columns',
            stagnation='5000 column queries without1percent improvement inbestverifiedresidual and residual>1e-6',
            selection='exact projected scalar score, including numerator correction',blas_threads=1,cpu_limit_seconds=145),
        source_sha256={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),root/'scripts/check_schur_fsai.py',root/'scripts/check_projected_balancing.py',
             root/'scripts/check_block_corrections.py',root/'src/nystrom_project/numerics.py']},baselines=[],runs=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_score_check']=verify_score_formula();cases=[]
        for name in ('synthetic-n640-smooth','concrete-observed','airfoil-observed'):
            prepared=root/f'data/prepared/{name}-s0-lambda0.001.npz';data=np.load(prepared);b=data['y_train']
            kernel=GaussianKernel(data['X_train'],float(data['bandwidth']));K=kernel.dense();A=K+.001*np.eye(len(b))
            base=initialize_rpcholesky(kernel,16,.001,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
            fixed,_=pcg(A,b,base,maxiter=2500);permutation,L,B,D=schur_blocks(A,base.indices)
            cases.append(dict(name=name,A=A,D=D,b=b,permutation=permutation,L=L,B=B,fixed=fixed))
            result['baselines'].append(dict(dataset=name,n=len(b),**fixed))
        variants=[('scalar',1,16),('scalar',1,32),('simultaneous',8,16),('simultaneous',16,16),
                  ('sequential_combined',8,16),('sequential_combined',16,16),('sequential_all',8,16),('sequential_all',16,16)]
        for mode,width,history in variants:
            for case in cases:
                if process_time()-cpu_start>135:result['status']='budget_exhausted';save();return
                solved=run(case,width,history,mode,cpu_start);solved['dataset']=case['name'];result['runs'].append(solved);save()
                print(json.dumps({key:solved[key] for key in ('dataset','mode','block_size','history_pairs','blocks','coordinate_steps',
                    'converged','true_relative_residual','stop_reason','equivalent_full_entry_passes','exploratory_cpu_seconds')}),flush=True)
        result['status']='complete';save();print(json.dumps(dict(status=result['status'],cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__=='__main__':main()
