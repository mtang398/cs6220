"""Free dense setup/inverse oracle for degree selection. No runtime/memory claim."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import gammainc
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]

def main():
 out=ROOT/'results/taylor-degree-oracle.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(100,110));start,cpu=perf_counter(),process_time()
 cases=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),('mixed512','synthetic-n640-mixed-s0-lambda0.001.npz'),('Concrete824','concrete-observed-s0-lambda0.001.npz')]
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(cases=[n for n,_ in cases],regularizations=[.001,1e-5],rtol=1e-8,max_outer=30,contraction_threshold=.2,degree_selection='least global degree meeting current exact-correction bound, or actual-defect oracle',static_controls='every degree 0 through conservative beta<=lambda/2 degree plus 3',excluded_costs='all dense kernel construction, all Cholesky factors, every selection probe, exact inner inverses; this diagnoses degree changes only',scope='No memory or full-work superiority claim'),runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2)+'\n')
 with threadpool_limits(limits=1):
  for name,stem in cases:
   path=ROOT/'data/prepared'/stem
   with np.load(path) as q:X,b,bw=q['X_train'],q['y_train'],float(q['bandwidth'])
   n=len(b);Z=X/bw;s=np.einsum('ij,ij->i',Z,Z);D=np.exp(-s/2);G=Z@Z.T;K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));di=np.diag_indices(n)
   for lam in [.001,1e-5]:
    for pg in range(100):
     roots=np.sqrt(gammainc(pg+1,s));beta=float(np.max(roots*(roots.sum()-roots)))
     if beta<=lam/2:break
    pmax=pg+3; factors=[];tails=[];term=np.ones_like(G);poly=np.ones_like(G)
    A=K.copy();A[di]+=lam
    for p in range(pmax+1):
     if p:term*=G/p;poly+=term
     e=gammainc(p+1,s);M=D[:,None]*poly*D[None,:];M[di]+=e+lam
     factors.append(cho_factor(M,lower=True,check_finite=False));tails.append(np.sqrt(e))
    normb=np.linalg.norm(b)
    def solve(mode,degree=None):
     x=np.zeros(n);r=b.copy();history=[];probes=0
     for step in range(30):
      normr=np.linalg.norm(r)
      if normr<=1e-8*normb:break
      candidates=[degree] if degree is not None else range(pmax+1)
      for p in candidates:
       z=cho_solve(factors[p],r,check_finite=False);probes+=1
       root=tails[p];w=root*(root@np.abs(z)-root*np.abs(z));cert=float(np.linalg.norm(w)/normr)
       actual=float(np.linalg.norm(r-A@z)/normr)
       if degree is not None or (cert if mode=='certificate' else actual)<=.2:break
      else:raise RuntimeError('no accepted degree')
      x+=z;r=b-A@x
      history.append(dict(degree=p,certificate=cert,actual_contraction=actual,true_relative_residual=float(np.linalg.norm(r)/normb)))
      if np.linalg.norm(r)>1000*normb:break
     return dict(dataset=name,regularization=lam,mode=mode,degree=degree,global_degree=pg,max_degree=pmax,converged=bool(np.linalg.norm(r)<=1e-8*normb),true_relative_residual=float(np.linalg.norm(r)/normb),outer_steps=len(history),degree_sum=sum(h['degree'] for h in history),free_inverse_probes=probes,history=history,data_sha256=sha256(path.read_bytes()).hexdigest())
    rec['runs'].extend(solve('fixed',p) for p in range(pmax+1))
    rec['runs'].extend(solve(mode) for mode in ['certificate','actual_defect'])
    save()
 rec['status']='complete';save();print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
