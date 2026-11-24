"""Established target-dependent pivoting before the main fixed-PCG recurrence.

Dense operator action is an offline diagnostic, not a timing claim. Pivot
construction accesses only the diagonal and selected columns of A. The selected
actual kernel columns define the ordinary Nystrom preconditioner.
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
from nystrom_project.numerics import GaussianKernel,NystromPreconditioner,initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner
from check_projected_balancing import pcg
from check_power_depth_headroom import refine,Identity


def select(A,b,k,*,target=True,check=False):
    n=len(b);F=np.zeros((n,k));r=b.copy();diagonal=np.diag(A).copy()
    selected=[];steps=[]
    for j in range(k):
        score=r*r/np.maximum(diagonal,np.finfo(float).tiny) if target else diagonal.copy()
        score[selected]=-np.inf
        p=int(np.argmax(score));pivot=float(diagonal[p])
        assert pivot>0
        column=(A[:,p]-F[:,:j]@F[p,:j])/np.sqrt(pivot)
        coefficient=r[p]/np.sqrt(pivot)
        if check:
            old=0. if not selected else float(b[selected]@linalg.solve(A[np.ix_(selected,selected)],b[selected],assume_a='pos'))
            enlarged=selected+[p]
            new=float(b[enlarged]@linalg.solve(A[np.ix_(enlarged,enlarged)],b[enlarged],assume_a='pos'))
            assert abs((new-old)-coefficient**2)<1e-9*max(abs(new),1)
        F[:,j]=column;r-=column*coefficient
        diagonal=np.maximum(diagonal-column*column,0)
        selected.append(p)
        steps.append(dict(index=p,schur_diagonal=pivot,energy_decrease=float(coefficient**2)))
    idx=np.array(selected,dtype=int)
    x=np.zeros(n);x[idx]=linalg.solve(A[np.ix_(idx,idx)],b[idx],assume_a='pos')
    residual_error=float(np.linalg.norm(r-(b-A@x))/max(np.linalg.norm(b),1))
    assert residual_error<1e-9
    return idx,x,dict(steps=steps,schur_residual_relative_error=residual_error,
                      retained_column_count=k,setup_column_rhs=k,
                      partial_factor_workspace_bytes=F.nbytes)


def main():
    output=ROOT/'results/target-greedy-initialization-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    protocol_path=ROOT/'configs/revision-target-greedy-initialization.json'
    protocol=json.loads(protocol_path.read_text())
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    started,cpu_started=perf_counter(),process_time()
    record=dict(status='running',scope='Offline iteration/column-quality diagnostic; no end-to-end time claim',
        protocol=protocol,source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
         [Path(__file__).resolve(),protocol_path,ROOT/'scripts/check_projected_balancing.py',
          ROOT/'scripts/check_power_depth_headroom.py',ROOT/'src/nystrom_project/numerics.py',
          ROOT/'src/nystrom_project/preconditioners.py']},cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(62209);G=rng.normal(size=(16,16));small=G@G.T+.2*np.eye(16)
        idx,x,check=select(small,rng.normal(size=16),8,check=True)
        record['independent_check']=check
        for name in protocol['workloads']:
            path=ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data=np.load(path);b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']))
            K=kernel.dense();n=len(b);k=protocol['rank']
            omega=np.random.default_rng(np.random.SeedSequence([0,6220,3])).normal(size=(n,k))
            first=refine(K,omega,2,.001);fourth=refine(K,first.basis,2,.001);eighth=refine(K,fourth.basis,4,.001)
            for lam in protocol['regularizations']:
                if process_time()-cpu_started>100:record['status']='cpu_budget';save();return
                A=K+lam*np.eye(n)
                case=dict(dataset=name,regularization=lam,n=n,rank=k,prepared_sha256=sha256(path.read_bytes()).hexdigest(),runs=[])
                record['cases'].append(case)
                target_idx,target_x,checks=select(A,b,k)
                matrix_idx,matrix_x,matrix_checks=select(A,b,k,target=False)
                rp=initialize_rpcholesky(kernel,k,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
                rp_x=np.zeros(n);rp_x[rp.indices]=linalg.solve(A[np.ix_(rp.indices,rp.indices)],b[rp.indices],assume_a='pos')
                for method,idx,x,diagnostic in [('target_f_over_P',target_idx,target_x,checks),
                    ('matrix_diagonal_greedy_A',matrix_idx,matrix_x,matrix_checks),('rp_K',rp.indices,rp_x,{})]:
                    base=LowRankPreconditioner.from_nystrom(NystromPreconditioner(idx,K[:,idx],lam))
                    for initial in ['zero','coordinate_Galerkin']:
                        run,_=pcg(A,b,base,x0=None if initial=='zero' else x,maxiter=2500)
                        run.update(method=method,initial_guess=initial,indices=idx.tolist(),construction_checks=diagnostic,
                            factor_bytes=base.storage_bytes,setup_columns=k,main_operator_passes=run['operator_applications']+1)
                        case['runs'].append(run)
                for method,model,passes in [('CG',Identity(),0)]+[
                    (f'power{q}',LowRankPreconditioner(s.basis,s.eigenvalues,lam),q) for q,s in [(2,first),(4,fourth),(8,eighth)]]:
                    run,_=pcg(A,b,model,maxiter=2500)
                    run.update(method=method,initial_guess='zero',factor_bytes=model.storage_bytes,setup_block_passes=passes,
                        setup_block_rhs=k*passes,total_kernel_entry_passes=passes+run['operator_applications']+1)
                    case['runs'].append(run)
                save()
                print(json.dumps(dict(dataset=name,lam=lam,runs=[(r['method'],r['initial_guess'],r['iterations'],r['converged']) for r in case['runs']])),flush=True)
    record['status']='complete';save()


if __name__=='__main__':main()
