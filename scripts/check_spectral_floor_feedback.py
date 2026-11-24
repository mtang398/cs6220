"""Frozen-basis spectral-floor feedback diagnostic; no full-budget performance claim."""
import json,resource
from pathlib import Path
from time import perf_counter,process_time
from hashlib import sha256
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_taylor_inexact_cg_diagnostic import ROOT,fps_nystrom

def run(A,b,U,theta,lam,mu,mode):
 nb=np.linalg.norm(b);x=np.zeros_like(b);r=b.copy();p=None;ap=None;denom=None;history=[];fits=[];status='maxiter'
 for iteration in range(2000):
  if np.linalg.norm(b-A@x)<=1e-8*nb:status='converged';break
  c=U.T@r;z=(r-U@c)/mu+U@(c/(theta+lam))
  if p is None:p=z.copy()
  else:p=z-(float(z@ap)/denom)*p
  ap=A@p;denom=float(p@ap);alpha=float(r@p)/denom
  if denom<=0:status='curvature_failure';break
  projection=U.T@p;q=p-U@projection;d=float(q@q)
  fit=max(lam,float((denom-np.dot(projection*(theta+lam),projection))/d)) if d>1e-28*float(p@p) else mu
  x+=alpha*p;r-=alpha*ap;history.append(mu);fits.append(fit)
  if mode=='adaptive' or (mode=='first_direction_frozen' and len(history)==1):mu=fit
 else:iteration=2000
 res=float(np.linalg.norm(b-A@x)/nb)
 return dict(mode=mode,initial_mu=history[0] if history else mu,iterations=len(history),converged=res<=1e-8,true_relative_residual=res,status=status,mu_history=history,fits=fits)

def main():
 out=ROOT/'results/spectral-floor-feedback.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(100,110));t,c=perf_counter(),process_time();rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(rank=64,rtol=1e-8,maxiter=2000,recurrence='one previous A-conjugate direction for all methods; literal original residual stop',fixed_grid='7 logarithmic mu values between lambda and theta_min+lambda, plus RHS curvature-fit floor',fit='mu=(p^TAp−(U^Tp)^Tdiag(theta+lambda)(U^Tp))/||p−UU^Tp||²',free='dense kernels/setup and every verification; no totalmemory/runtime claim',first_rhs_fit='computed from p=b using one A product; charge one extra product'),runs=[])
 with threadpool_limits(limits=1):
  for ds,stem in [('smooth512','synthetic-n640-smooth'),('mixed512','synthetic-n640-mixed'),('Concrete824','concrete-observed')]:
   path=ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz'
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));theta,U,_=fps_nystrom(K)
   for lam in [.001,1e-5]:
    A=K+lam*np.eye(len(K));ftu=float(theta.min()+lam);proj=U.T@b;q=b-U@proj
    initialfit=max(lam,float((b@(A@b)-np.dot(proj*(theta+lam),proj))/(q@q)))
    controls=[('fixed',float(mu)) for mu in np.geomspace(lam,ftu,7)]+[('rhs_fit_frozen',initialfit),('first_direction_frozen',ftu),('adaptive',ftu)]
    for mode,mu in controls:
     result=run(A,b,U,theta,lam,mu,mode);result.update(dataset=ds,regularization=lam,ftu_mu=ftu,solver_plus_probe_products=result['iterations']+(mode=='rhs_fit_frozen'),data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(result)
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
