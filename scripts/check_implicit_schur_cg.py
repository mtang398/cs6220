"""Implicit Schur-CG baseline without retained cross-kernel columns.

This is established elimination, not a proposed adaptive method. The memory
allowance includes named setup/application arrays plus a LAPACK allowance;
the process high-water RSS and common immutable data are also reported.
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


def farthest_indices(kernel,m):
    """Farthest-point selection retains only a similarity vector and indices."""
    indices=[];similarity=np.zeros(kernel.n);j=0
    for _ in range(m):
        indices.append(j);c=kernel.column(j)
        np.maximum(similarity,c,out=similarity);similarity[indices]=np.inf
        j=int(np.argmin(similarity))
    return np.asarray(indices,dtype=np.int64)


class ImplicitSchur:
    def __init__(self,kernel,indices,lam,row_block=128):
        self.kernel=kernel;self.indices=indices.copy();self.lam=lam;self.row_block=row_block
        self.remaining=np.setdiff1d(np.arange(kernel.n),indices)
        core=kernel.entries(indices,indices)
        core.flat[::len(indices)+1]+=lam
        self.L=linalg.cholesky(core,lower=True,check_finite=False)
        self.calls=0
        self.retained_bytes=self.L.nbytes+self.indices.nbytes+self.remaining.nbytes
        # Named simultaneous arrays plus conservative opaque-library allowance.
        n,m=kernel.n,len(indices);r=n-m;q=min(row_block,max(m,r))
        self.setup_allowance_bytes=3*m*m*8+5*n*8+65536
        self.apply_allowance_bytes=self.retained_bytes+12*n*8+q*max(r,m)*8+65536

    def solve_core(self,rhs):
        return linalg.cho_solve((self.L,True),rhs,check_finite=False)

    def cross(self,v,transpose=False):
        rows,cols=(self.indices,self.remaining) if transpose else (self.remaining,self.indices)
        result=np.empty(len(rows))
        for start in range(0,len(rows),self.row_block):
            rr=rows[start:start+self.row_block]
            result[start:start+len(rr)]=self.kernel.entries(rr,cols)@v
        return result

    def apply(self,v):
        n=self.kernel.n;all_values=np.empty(n)
        # The same first pass yields A_RR v and A_SR v.
        for start in range(0,n,self.row_block):
            rr=np.arange(start,min(n,start+self.row_block))
            all_values[rr]=self.kernel.entries(rr,self.remaining)@v
        q=self.solve_core(all_values[self.indices])
        result=all_values[self.remaining]+self.lam*v
        result-=self.cross(q)
        self.calls+=1
        return result

    def reconstruct(self,b,x_tail):
        x=np.empty(self.kernel.n);x[self.remaining]=x_tail
        x[self.indices]=self.solve_core(b[self.indices]-self.cross(x_tail,transpose=True))
        return x


def solve(kernel,b,lam,m,*,indices=None,rtol=1e-8,maxiter=2500,cpu_cap=30.):
    started,cpu_started=perf_counter(),process_time()
    if indices is None:indices=farthest_indices(kernel,m)
    op=ImplicitSchur(kernel,indices,lam)
    budget=1024**2
    assert max(op.setup_allowance_bytes,op.apply_allowance_bytes)<=budget
    rhs=b[op.remaining]-op.cross(op.solve_core(b[indices]))
    x=np.zeros(len(rhs));r=rhs.copy();p=r.copy();rr=float(r@r)
    norm_b=np.linalg.norm(b);checks=0;replacements=0;status='maxiter';progress=[]
    setup=perf_counter()-started
    for iteration in range(1,maxiter+1):
        if process_time()-cpu_started>cpu_cap:
            status='cpu_screen_cap';iteration-=1;break
        ap=op.apply(p);curv=float(p@ap)
        if curv<=0 or not np.isfinite(curv):status='breakdown';break
        alpha=rr/curv;x+=alpha*p;r-=alpha*ap
        suspected=np.linalg.norm(r)<=rtol*norm_b
        if iteration%20==0 or suspected or iteration==maxiter:
            full=op.reconstruct(b,x)
            true=b-kernel.matvec(full)-lam*full;checks+=1
            rel=float(np.linalg.norm(true)/norm_b)
            progress.append(dict(iteration=iteration,true_relative_residual=rel,
                landmark_residual=float(np.linalg.norm(true[indices])/norm_b)))
            if rel<=rtol:status='converged';break
            if suspected:
                r=true[op.remaining].copy();p=r.copy();rr=float(r@r);replacements+=1;continue
        rr_new=float(r@r);p=r+(rr_new/rr)*p;rr=rr_new
    full=op.reconstruct(b,x)
    residual=b-kernel.matvec(full)-lam*full;checks+=1
    rel=float(np.linalg.norm(residual)/norm_b)
    return full,dict(status=status,converged=rel<=rtol,iterations=iteration,
        true_relative_residual=rel,landmark_residual=float(np.linalg.norm(residual[indices])/norm_b),
        total_seconds=perf_counter()-started,total_cpu_seconds=process_time()-cpu_started,
        setup_seconds=setup,schur_products=op.calls,original_verifications=checks,
        residual_replacements=replacements,selected_indices=indices.tolist(),
        retained_bytes=op.retained_bytes,setup_array_allowance_bytes=op.setup_allowance_bytes,
        application_array_allowance_bytes=op.apply_allowance_bytes,algorithm_array_budget_bytes=budget,
        common_input_bytes=kernel.X.nbytes+b.nbytes,
        kernel_stats=dict(kernel.stats),progress=progress)


def independent_check():
    rng=np.random.default_rng(622082);X=rng.normal(size=(29,4));kernel=GaussianKernel(X,1.7)
    lam=.03;K=kernel.dense();A=K+lam*np.eye(len(X));b=rng.normal(size=len(X))
    S=np.array([2,7,13,18,23]);op=ImplicitSchur(kernel,S,lam,row_block=7)
    R=op.remaining;D=A[np.ix_(R,R)]-A[np.ix_(R,S)]@linalg.solve(A[np.ix_(S,S)],A[np.ix_(S,R)],assume_a='pos')
    v=rng.normal(size=len(R));expected=D@v;actual=op.apply(v)
    action_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    full,metrics=solve(GaussianKernel(X,1.7),b,lam,len(S),indices=S,rtol=1e-11,maxiter=300)
    reference=linalg.solve(A,b,assume_a='pos')
    error=float(np.linalg.norm(full-reference)/np.linalg.norm(reference))
    assert action_error<1e-12 and error<1e-9 and metrics['converged']
    return dict(action_relative_error=action_error,solution_relative_error=error,
        expected_entries_per_action=(len(R)**2+2*len(R)*len(S)),
        observed_action_entries=kernel.stats['entry_evaluations']-len(S)**2)


def main():
    output=ROOT/'results/implicit-schur-cg-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(105,110))
    started,cpu_started=perf_counter(),process_time()
    result=dict(status='running',scope='Established implicit Schur-CG baseline under a declared array allowance; not adaptive success',
        protocol=dict(dataset='synthetic-n640-smooth',seed=0,landmark_counts=[16,64,128,192],
            regularizations=[.001,1e-5],selection='farthest-point from index0',
            row_block=128,rtol=1e-8,maxiter=2500,per_solve_cpu_cap=20.,blas_threads=1,
            array_budget_bytes=1024**2,lapack_extra_allowance_bytes=65536,
            resource_scope='All named algorithm arrays and declared library allowance; common immutable input and interpreter reported separately. No universal RSS-cap claim.',
            setup_peak='3*m² doubles plus5n doubles and library allowance',
            application_peak='retained core and indices plus12n doubles, one kernel block, library allowance'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),ROOT/'src/nystrom_project/numerics.py']},runs=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_check']=independent_check();save()
        path=ROOT/'data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz'
        data=np.load(path);X,b,bandwidth=data['X_train'],data['y_train'],float(data['bandwidth'])
        for lam in result['protocol']['regularizations']:
            for m in result['protocol']['landmark_counts']:
                if process_time()-cpu_started>80:result['status']='cpu_budget';save();return
                _,metrics=solve(GaussianKernel(X,bandwidth),b,lam,m,cpu_cap=20.)
                metrics.update(landmark_count=m,regularization=lam)
                result['runs'].append(metrics);save()
                print(json.dumps({k:metrics[k] for k in ['landmark_count','regularization','status','iterations','total_seconds','true_relative_residual']}),flush=True)
    result['status']='complete';save()


if __name__=='__main__':main()
