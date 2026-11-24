"""Exact Schur trajectories bound allowable core-refinement work; dense oracle."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
from check_large_core_exchange_oracle import ROOT,build

def solve_verified(A,b,J,op):
 R,F,B,V,D,rhs=op;y=np.zeros(len(R));r=rhs.copy();p=r.copy();rho=float(r@r);nb=np.linalg.norm(b)
 for it in range(2001):
  full=np.empty(len(b));full[R]=y;full[J]=cho_solve(F,b[J]-B@y)
  residual=float(np.linalg.norm(b-A@full)/nb)
  if residual<=1e-8 or it==2000:return dict(iterations=it,converged=residual<=1e-8,true_relative_residual=residual)
  ap=D@p;alpha=rho/float(p@ap);y+=alpha*p;r-=alpha*ap;new=float(r@r);p=r+(new/rho)*p;rho=new


def main():
 out=ROOT/'results/larger-core-refinement-headroom.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60));t,c=perf_counter(),process_time()
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(cores=[416,600],regularizations=[.001,1e-5],rtol=1e-8,core_refinement_products=[1,2,3,4],endpoint='Literal original full residual checked each step, including reconstruction; initial reduced-only endpoint attempt failed assert and produced no result file',scope='Exact dense core solves and setup free; only kernel-entry headroom, no native-memory/time/approximateconvergence claim',entry_model='iterations*(n²−m²+j*m²), j regenerated core products eachSchur action; ignore extra RHS/reconstruction refinement'),runs=[])
 with threadpool_limits(limits=1):
  for ds in ['smooth','random','mixed']:
   path=ROOT/'data/prepared'/f'synthetic-n2560-{ds}-s0-lambda0.001.npz'
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   n=len(b);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));indices=[0];distance=2-2*K[:,0]
   for _ in range(599):
    j=int(np.argmax(distance));indices.append(j);distance=np.minimum(distance,2-2*K[:,j])
   indices=np.asarray(indices)
   for lam in [.001,1e-5]:
    A=K+lam*np.eye(n);results={}
    for m in [416,600]:
     op=build(A,b,indices[:m]);result=solve_verified(A,b,indices[:m],op);results[m]=result
    ref=results[416]['iterations']*(n*n-416*416);nc=results[600]['iterations'];break_even=(ref/nc-(n*n-600*600))/(600*600)
    rec['runs'].append(dict(dataset=ds,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest(),exact_core416=results[416],exact_core600=results[600],break_even_core_products_per_action=break_even,query_ratios={str(j):nc*(n*n-600*600+j*600*600)/ref for j in [1,2,3,4]},retained_packed_fp32_core600_bytes=4*600*601//2,retained_packed_fp64_core416_bytes=8*416*417//2))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
