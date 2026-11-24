"""Exact original-system PCG with compressed block inverse; dense cost gate."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import eigvalsh,solve_triangular
from threadpoolctl import threadpool_limits
from check_core_precision_allocation import MixedFactor,Helper
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def run(A,b,S,factor,order):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];calls=0
 def H(r):
  z=factor.solve(r)
  if order:z+=factor.solve(r-C@z)
  return z
 def apply(r):
  nonlocal calls
  calls+=1;u=H(r[S]);v=(r[R]-B.T@u)/1e-5
  z=np.empty_like(r);z[S]=u-H(B@v);z[R]=v;return z
 x=np.zeros(n);r=b.copy();z=apply(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);start=process_time();status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-start>4:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;x+=alpha*p;r-=alpha*ap
  # Do not apply preconditioner after an accepted endpoint.
  if np.linalg.norm(b-A@x)<=1e-8*nb:
   k+=1;rel=float(np.linalg.norm(b-A@x)/nb);status='converged';break
  z=apply(r);new=float(r@z);p=z+new/rho*p;rho=new
 components=dict(FPS=n*m,factor=m*(m+1)//2,operator=k*n*n,cross=calls*2*m*(n-m),core=calls*2*order*m*m,final_verification=n*n)
 return dict(method='H'+str(order),iterations=k,preconditioner_calls=calls,status=status,converged=status=='converged',original_relative_residual=rel,entry_components=components,analytical_entries=sum(components.values()))

def main():
 out=ROOT/'results/exact-outer-mixed-core.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();control_path=ROOT/'results/core-energy-selector.json';controls=json.loads(control_path.read_text())
 sources=[Path(__file__).resolve(),ROOT/'scripts/check_core_precision_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so',control_path]
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},protocol=dict(core=592,shift=1e-5,question='Can exact original-system PCG remove approximate Schur correction costs enough to justify its larger per-step action?',inverse='u=H rS,v=(rR-BT u)/lambda,zS=u-HBv,zR=v. H0=stored factor inverse; H1=2H0-H0 C H0. SPD block inverse if H SPD.',counts='FullA periteration; two cross products and two core inverse actions per preconditioning. H1 adds two C products. Dense construction/verification during iterations free except final verification. No native timing/full-memory claim.',controls='Saved converged fixed FIRST8_H1 and H2-prefix Schur policies on identical data; current selector excluded.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;C=A[np.ix_(S,S)];factor=MixedFactor(C,range(8),helper)
   L=factor.dense_L();T=solve_triangular(L,C,lower=True);W=solve_triangular(L,T.T,lower=True).T;largest=float(eigvalsh((W+W.T)*.5,subset_by_index=[591,591])[0]);assert largest<2
   matched=[r for r in controls['runs'] if r['dataset']==name and r['method'] in ['FIRST8_H1','FIRST8_H2prefix4_then_H1'] and r['converged']];assert matched and all(r['data_sha256']==sha256(path.read_bytes()).hexdigest() for r in matched)
   best=min(matched,key=lambda r:r['analytical_solver_Gaussian_entries']);row=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),largest_whitened_core_eigenvalue=largest,reference=dict(method=best['method'],entries=best['analytical_solver_Gaussian_entries']),runs=[])
   for order in [0,1]:
    result=run(A,b,S,factor,order);result['entry_ratio']=result['analytical_entries']/row['reference']['entries'];row['runs'].append(result);print(json.dumps(dict(dataset=name,**result)),flush=True)
   rec['cases'].append(row);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
