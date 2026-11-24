"""Residual-weighted Fourier-feature resampling diagnostic, dense and free setup."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

class FeatureInverse:
 def __init__(self,Z,lam):
  self.Z=Z;self.lam=lam;G=Z.T@Z;G.flat[::len(G)+1]+=lam;self.F=cho_factor(G,lower=True)
 def apply(self,r):return (r-self.Z@cho_solve(self.F,self.Z.T@r))/self.lam

def solve(A,b,H,x0=None,capture=False):
 x=np.zeros_like(b) if x0 is None else x0.copy();r=b-A@x;z=H.apply(r);p=z.copy();rho=float(r@z);norm=np.linalg.norm(b);start=process_time();saved=None;status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/norm)
  if capture and k==10:saved=x.copy()
  if rel<=1e-8:status='converged';break
  if process_time()-start>3:status='cpu_cap';break
  if k==2000:break
  Ap=A@p;den=float(p@Ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;x+=alpha*p;r-=alpha*Ap;z=H.apply(r);new=float(r@z);p=z+new/rho*p;rho=new
 return dict(iterations=k,status=status,converged=status=='converged',original_relative_residual=rel),saved

def main():
 out=ROOT/'results/residual-fourier-preconditioning.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time()
 sources=[Path(__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},protocol=dict(m=416,pool=832,seed=622091,lambda_value=1e-5,checkpoint=10,representation='P=lambda I+ZZ^T; Gaussian frequencies omega~N(0,I/bw²), random uniform phases, raw features sqrt(2)cos(Xomega+phase). Initial Z uses first416 divided by sqrt416.',selection='Pool832, q_j=.5/832+.5*(phi_j^T direction)^2/sum scores. Sample416 with replacement; Z_j=phi_j/sqrt(416*832*q_j). Expectation equals uniform finite-pool covariance. Uniform and initial-b controls use the same categorical inverse-CDF random numbers.',cost='Dense feature/kernel matrices, pool scoring, factorization and reconstruction free. No native memory/time claim. Each run3CPU/2000steps; literal original1e-8 everyiterate. Static and changed preconditioners are fixed within each branch.',scope='Established RFF preconditioner with a diagnostic residual-based sampling rule; not a novelty claim or reproduction of a specific literature implementation.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  rng=np.random.default_rng(11);Z=rng.normal(size=(23,7));r=rng.normal(size=23);h=FeatureInverse(Z,.3);err=float(np.linalg.norm((Z@Z.T+.3*np.eye(23))@h.apply(r)-r)/np.linalg.norm(r));assert err<1e-10;rec['tiny_inverse_error']=err
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   if process_time()-cpu>63:rec['status']='cpu_budget';save();return
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   n=len(b);A=SharedKernel(X,bw,Budget([X,b])).block(X,X);A.flat[::n+1]+=1e-5
   rng=np.random.default_rng(622091);omega=rng.normal(size=(X.shape[1],832))/bw;phase=rng.uniform(0,2*np.pi,832);Phi=np.sqrt(2)*np.cos(X@omega+phase)
   H=FeatureInverse(Phi[:,:416]/np.sqrt(416),1e-5);base,x=solve(A,b,H,capture=True);row=dict(dataset=name,n=n,data_sha256=sha256(path.read_bytes()).hexdigest(),baseline=base,branches=[]);rec['cases'].append(row);save()
   if x is None:row['branch_status']='no_checkpoint';save();continue
   uniforms=rng.random(416)
   for mode in ['unchanged_restart','uniform_resample','initial_target_resample','current_residual_resample']:
    if mode=='unchanged_restart':newH=H;distinct=416;probability=None
    else:
     probability=np.full(832,1/832)
     if mode!='uniform_resample':
      direction=b if mode=='initial_target_resample' else b-A@x;score=(Phi.T@direction)**2
      if score.sum()>0:probability=.5*probability+.5*score/score.sum()
     indices=np.searchsorted(np.cumsum(probability),uniforms);indices=np.minimum(indices,831);distinct=len(np.unique(indices));newH=FeatureInverse(Phi[:,indices]/np.sqrt(416*832*probability[indices])[None,:],1e-5)
    result,_=solve(A,b,newH,x0=x);result.update(method=mode,prefix_plus_iterations=10+result['iterations'],distinct_frequencies=distinct);row['branches'].append(result);save();print(json.dumps(dict(dataset=name,**result)),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
