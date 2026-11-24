"""Dense diagnostic of recurrence-based preconditioner energy in degree selection."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_taylor_inexact_cg_diagnostic import ROOT,DenseTaylor,fps_nystrom,load_polynomial

def main():
 out=ROOT/'results/taylor-energy-cg-diagnostic.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60));t,c=perf_counter(),process_time();lib,fun=load_polynomial()
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(rtol=1e-8,maxiter=2000,rank=64,rule='alpha_upper(q)=rho/max(lambda*p2,sigma-beta_q*p2); same uniform per-step gap budget',energy_update='sigma_new=rho_new+2*beta*r_new^T p_old+beta^2*sigma_old',audit='Independent P action every step only to audit recurrence; not used by selector. Dense setup/cache free. No native memory/time claim.'),runs=[])
 with threadpool_limits(limits=1):
  for ds,stem in [('smooth512','synthetic-n640-smooth'),('mixed512','synthetic-n640-mixed'),('Concrete824','concrete-observed')]:
   path=ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz'
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));val,Q,idx=fps_nystrom(K);tay=DenseTaylor(X,bw,fun)
   for lam in [.001,1e-5]:
    A=K+lam*np.eye(len(K));coeff=val/(val+lam);nb=np.linalg.norm(b);tol=1e-8*nb;step=tol/8000
    def H(v):return (v-Q@(coeff*(Q.T@v)))/lam
    x=np.zeros_like(b);r=b.copy();z=H(r);p=z.copy();rho=float(r@z);sigma=rho;gap=0.;hist=[];status='maxiter';maxenergy=0.;maxratio=0.
    for it in range(2000):
     p2=float(p@p);energy=float(lam*p2+(Q.T@p)@(val*(Q.T@p)));maxenergy=max(maxenergy,abs(sigma-energy)/max(energy,1e-300))
     bounds=tay.bounds(p);lower=np.maximum(lam*p2,sigma-tay.beta*p2);upper=rho/lower
     eligible=np.flatnonzero(upper*bounds<=step)
     if not len(eligible):status='degree_cap';break
     q=int(eligible[0]);ap=tay.matrix(q)@p+lam*p;alpha=rho/float(p@ap);maxratio=max(maxratio,float(alpha/upper[q]));gap+=abs(alpha)*bounds[q]
     x+=alpha*p;r-=alpha*ap;hist.append(q)
     if np.linalg.norm(r)+gap<=tol:status='stopped';break
     z=H(r);newrho=float(r@z);beta=newrho/rho
     sigma=newrho+2*beta*float(r@p)+beta*beta*sigma
     p=z+beta*p;rho=newrho
    residual=float(np.linalg.norm(b-A@x)/nb)
    rec['runs'].append(dict(dataset=ds,regularization=lam,status=status,converged=residual<=1e-8,iterations=len(hist),degrees=hist,mean_degree=float(np.mean(hist)),true_relative_residual=residual,truncation_gap_over_rhs=float(gap/nb),maximum_energy_relative_error=maxenergy,maximum_alpha_over_bound=maxratio,data_sha256=sha256(path.read_bytes()).hexdigest()))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
