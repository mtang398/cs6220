"""Offline truncated multipreconditioned-CG test with fixed Nyström columns.

Separating the Nyström range and its orthogonal complement yields two
directions whose sum is the usual inverse-preconditioned residual. A joint
Galerkin step can share one kernel evaluation pass for their two images.
This is an established multipreconditioning construction, not a novelty claim.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner, initialize_gaussian
from nystrom_project.flexible import AConjugateHistory
from check_projected_balancing import pcg


def solve(A,b,base,history_size,maxiter=2500):
    x=np.zeros_like(b);r=b.copy();norm_b=np.linalg.norm(b)
    history=AConjugateHistory(len(b),history_size)
    blocks=rhs_count=checks=replacements=0
    trace=[];ranks=[];reason='max_iterations'
    started=perf_counter();max_history=0
    for iteration in range(1,maxiter+1):
        c=base.basis.T@r
        perp=r-base.basis@c
        correction=base.basis.T@perp
        c+=correction;perp-=base.basis@correction
        coarse=base.basis@(c/(base.eigenvalues+base.regularization))
        Z=np.column_stack((perp/base.regularization,coarse))
        norms=np.linalg.norm(Z,axis=0)
        keep=norms>np.finfo(float).tiny
        Z=Z[:,keep]/norms[keep]
        AZ=A@Z;blocks+=1;rhs_count+=len(norms[keep])
        for _ in range(2):
            for p,ap,curvature in history._pairs:
                coeff=(p@AZ)/curvature
                Z-=p[:,None]*coeff;AZ-=ap[:,None]*coeff
        gram=(Z.T@AZ+AZ.T@Z)*.5
        values,V=linalg.eigh(gram,check_finite=False)
        threshold=256*np.finfo(float).eps*max(float(values[-1]),np.finfo(float).tiny)
        if values[0]<-threshold or values[-1]<=threshold:
            reason='indefinite_or_empty_block';break
        keep=values>threshold
        P=Z@(V[:,keep]/np.sqrt(values[keep]))
        AP=AZ@(V[:,keep]/np.sqrt(values[keep]))
        alpha=P.T@r
        x+=P@alpha;r-=AP@alpha
        ranks.append(int(np.count_nonzero(keep)))
        for p,ap in zip(P.T,AP.T):history.append(p,ap)
        max_history=max(max_history,history.storage_bytes)
        suspected=np.linalg.norm(r)<=1e-8*norm_b
        if iteration%20==0 or suspected or iteration==maxiter:
            true_r=b-A@x;checks+=1
            rel=float(np.linalg.norm(true_r)/norm_b)
            trace.append(dict(iteration=iteration,true_relative_residual=rel))
            if rel<=1e-8:reason='converged';break
            if suspected:r=true_r;history.clear();replacements+=1
    true_rel=float(np.linalg.norm(b-A@x)/norm_b);checks+=1
    return dict(converged=bool(true_rel<=1e-8),stop_reason=reason,iterations=iteration,
        true_relative_residual=true_rel,block_operator_calls=blocks,block_rhs=rhs_count,
        verification_matvecs=checks,equivalent_kernel_evaluation_passes=blocks+checks,
        history_size=history_size,max_history_bytes=max_history,
        retained_state_bytes=base.storage_bytes+max_history,
        temporary_direction_and_image_bytes=4*len(b)*8,
        rank_one_blocks=ranks.count(1),residual_replacements=replacements,
        exploratory_dense_seconds=perf_counter()-started,progress=trace)


def main():
    output=ROOT/'results/two-component-preconditioning-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(85,89))
    started,cpu_started=perf_counter(),process_time()
    record=dict(status='running',scope='Dense offline multipreconditioning headroom test; no CPU-speed or novelty claim',
        protocol=dict(regularization=.001,rank=32,seed=0,history_sizes=[2,8],
            directions='usual Nystrom inverse action split into range and orthogonal-complement components',
            baseline_ranks=[32,36,48],rtol=1e-8,maxiter=2500,blas_threads=1),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),ROOT/'src/nystrom_project/flexible.py',
             ROOT/'src/nystrom_project/preconditioners.py',ROOT/'scripts/check_projected_balancing.py']},cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(645)
        small=GaussianKernel(rng.normal(size=(35,3)),1.3)
        small_A=small.dense()+.03*np.eye(35)
        small_base=LowRankPreconditioner.from_nystrom(initialize_rpcholesky(small,8,.03,rng))
        check=solve(small_A,rng.normal(size=35),small_base,8,300)
        assert check['converged']
        record['independent_small_spd_check']=check
        for name in ['synthetic-n640-smooth','synthetic-n640-random','synthetic-n640-mixed','concrete-observed','airfoil-observed']:
            if process_time()-cpu_started>65:record['status']='cpu_budget';save();return
            d=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
            b=d['y_train'];kernel=GaussianKernel(d['X_train'],float(d['bandwidth']))
            K=kernel.dense();A=K+.001*np.eye(len(b))
            base=LowRankPreconditioner.from_nystrom(initialize_rpcholesky(kernel,32,.001,
                np.random.default_rng(np.random.SeedSequence([0,6220,1]))))
            case=dict(dataset=name,n=len(b),runs=[]);record['cases'].append(case)
            for history in [2,8]:
                result=solve(A,b,base,history);result['method']='two_components'
                case['runs'].append(result);save()
                print(json.dumps(dict(dataset=name,history=history,iterations=result['iterations'],
                    passes=result['equivalent_kernel_evaluation_passes'],converged=result['converged'])),flush=True)
            for rank in [32,36,48]:
                for method in ['compact','power']:
                    state=(LowRankPreconditioner.from_nystrom(initialize_rpcholesky(kernel,rank,.001,
                        np.random.default_rng(np.random.SeedSequence([0,6220,1])))) if method=='compact'
                        else initialize_gaussian(kernel,rank,.001,
                            np.random.default_rng(np.random.SeedSequence([0,6220,3])),power=True))
                    result,_=pcg(A,b,state,maxiter=2500)
                    result.update(method=method,rank=rank,retained_state_bytes=state.storage_bytes,
                        equivalent_kernel_evaluation_passes=result['operator_applications']+(2 if method=='power' else 0),
                        setup_kernel_columns=rank if method=='compact' else 0,
                        setup_block_rhs=2*rank if method=='power' else 0)
                    case['runs'].append(result);save()
            print(json.dumps(dict(dataset=name,static=[(v['method'],v['rank'],v['iterations'])
                for v in case['runs'] if v['method']!='two_components'])),flush=True)
    record['status']='complete';save()


if __name__=='__main__':main()
