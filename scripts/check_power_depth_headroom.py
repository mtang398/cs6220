"""Stronger fixed-rank subspace-iteration controls and delayed refinement.

Exact leading eigenvectors are an offline reference, not a timed method.
Every delayed branch uses exactly the same refined factor as its upfront
control. No residual-based trigger is proposed in this diagnostic.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel
from nystrom_project.preconditioners import LowRankPreconditioner
from check_projected_balancing import pcg


class Identity:
    storage_bytes=0
    def apply(self,r):return r.copy()


def refine(K,basis,passes,lam):
    Q=basis.copy()
    for step in range(passes):
        Y=K@Q
        if step+1<passes:Q,_=linalg.qr(Y,mode='economic',check_finite=False)
    return LowRankPreconditioner.from_sketch(Y,Q.T@Y,lam)


def main():
    output=ROOT/'results/power-depth-headroom-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(105,110))
    started,cpu_started=perf_counter(),process_time()
    record=dict(status='running',scope='Dense offline action and baseline headroom; no production CPU claim',
        protocol=dict(seed=0,rank=32,regularizations=[.001,1e-5],power_kernel_passes=[2,4,8],
            checkpoint=20,rtol=1e-8,maxiter=2500,blas_threads=1,
            construction='Two initial passes; refine the factor range for two and then four additional passes.',
            delayed_state='exactly same stored matrix as upfront control, no changed factor or sketch',
            oracle='exact leading32 eigenspace; dense decomposition disclosed, no setup-time comparison'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),ROOT/'src/nystrom_project/numerics.py',
             ROOT/'src/nystrom_project/preconditioners.py',ROOT/'scripts/check_projected_balancing.py']},cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        for name in ['synthetic-n640-smooth','synthetic-n640-random','synthetic-n640-mixed','concrete-observed','airfoil-observed']:
            data=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
            b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']))
            K=kernel.dense();w,U=linalg.eigh(K,subset_by_index=[len(b)-32,len(b)-1],check_finite=False)
            w,U=w[::-1],U[:,::-1]
            omega=np.random.default_rng(np.random.SeedSequence([0,6220,3])).normal(size=(len(b),32))
            first=refine(K,omega,2,.001)
            fourth=refine(K,first.basis,2,.001)
            eighth=refine(K,fourth.basis,4,.001)
            for lam in [.001,1e-5]:
                if process_time()-cpu_started>95:record['status']='cpu_budget';save();return
                A=K+lam*np.eye(len(b))
                models={depth:LowRankPreconditioner(s.basis,s.eigenvalues,lam)
                    for depth,s in [(2,first),(4,fourth),(8,eighth)]}
                old,captured=pcg(A,b,models[2],capture=True,maxiter=2500)
                case=dict(dataset=name,rank=32,regularization=lam,n=len(b),runs=[],
                    factor_state_bytes=models[2].storage_bytes,
                    reference_exact_spectrum=w.tolist(),
                    exact_eigenbasis_residual=float(np.linalg.norm(K@U-U*w)/np.linalg.norm(U*w)),
                    prefix_available='x' in captured)
                record['cases'].append(case)
                for depth,model in models.items():
                    run=old.copy() if depth==2 else pcg(A,b,model,maxiter=2500)[0]
                    run.update(method='power_upfront',power_passes=depth,total_iterations=run['iterations'],
                        total_kernel_entry_passes=depth+run['operator_applications']+1,
                        setup_block_rhs=32*depth,source_final_verification_products=1,
                        factor_sha256=sha256(model.basis.tobytes()+model.eigenvalues.tobytes()).hexdigest())
                    case['runs'].append(run)
                    if 'x' in captured:
                        delayed,_=pcg(A,b,model,x0=captured['x'],maxiter=2480)
                        delayed.update(method='power_after20',power_passes=depth,
                            total_iterations=20+delayed['iterations'],
                            total_kernel_entry_passes=depth+21+delayed['operator_applications']+1,
                            prefix_operator_products=21,setup_block_rhs=32*depth,
                            factor_sha256=run['factor_sha256'],source_final_verification_products=1)
                        case['runs'].append(delayed)
                for method,model in [('unpreconditioned',Identity()),
                    ('exact_top32_oracle',LowRankPreconditioner(U,w,lam))]:
                    run,_=pcg(A,b,model,maxiter=2500)
                    run.update(method=method,total_iterations=run['iterations'],
                        total_kernel_entry_passes=run['operator_applications']+1 if method=='unpreconditioned' else None,
                        source_final_verification_products=1,
                        construction_included=False if method=='exact_top32_oracle' else True)
                    case['runs'].append(run)
                save()
                print(json.dumps(dict(dataset=name,lam=lam,runs=[
                    (r['method'],r.get('power_passes'),r['total_iterations'],r['converged'])
                    for r in case['runs']])),flush=True)
    record['status']='complete';save()


if __name__=='__main__':main()
