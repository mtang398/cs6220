"""Free-factor streaming block-Jacobi capacity gate for original kernel system."""
import json, resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import check_full_aggregation_schur as agg
from check_schur_inplace_frontier_v2 import ROOT, SharedKernel, Budget

class Permutation:
 def __init__(self,selected,n):self.m=len(selected);self.order=np.r_[selected,np.setdiff1d(np.arange(n),selected)]
 def to_rotated(self,v):return v[self.order]
 def to_original(self,v):
  out=np.empty_like(v);out[self.order]=v;return out
 def matrix(self,A):return A[np.ix_(self.order,self.order)]

def pcg(A,b,groups,factors):
 def apply(r):
  out=np.empty_like(r)
  for g,F in zip(groups,factors):out[g]=cho_solve(F,r[g])
  return out
 x=np.zeros_like(b);r=b.copy();z=apply(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);start=process_time();status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-start>6:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if den<=0:status='nonpositive_curvature';break
  alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=apply(r);new=float(r@z);p=z+(new/rho)*p;rho=new
 return dict(iterations=k,status=status,converged=status=='converged',original_relative_residual=rel,free_factor_kernel_entries=(k+1)*len(b)**2)

def main():
 out=ROOT/'results/streamed-block-jacobi-headroom.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time()
 files=[Path(__file__).resolve(),Path(agg.__file__).resolve(),Path(agg.pair.__file__).resolve(),Path(agg.pair.previous.__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(question='Can large blocks, enabled by sequential factor recomputation, beat a retained exact principal core even when all factor work is free?',blocks='Balanced geometric partition with each block at most320. Exact inverse of original A diagonal blocks.',memory_planner='One dense320x320 FP64 buffer + six n-vectors +64KiB reserve, at most1015808B for n2048. Diagnostic caches all factors and A; native peak memory not audited.',cost='Candidate construction, factor work, and verification during iterations free. Charge one n-squared operator per step plus final verification. Compare exact416 Schur core with FPS, factor, boundary, iteration, final entries. Entry counts are not timings.',scope='Static headroom gate only. Original relative residual1e-8;2000iterations;6CPU per solve. Does not implement BLOKS or adaptive block selection.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   n=len(b);kernel=SharedKernel(X,bw,Budget([X,b]));A=kernel.block(X,X);A.flat[::n+1]+=1e-5
   groups=agg.partition(X,(n+319)//320);factors=[cho_factor(A[np.ix_(g,g)],lower=True) for g in groups]
   candidate=pcg(A,b,groups,factors)
   selected=kernel.farthest(416);rot=Permutation(selected,n);Ar,br,op=agg.prepare(A,b,rot)
   control,_,_=agg.pair.solve(A,b,rot,Ar,br,op,deadline=process_time()+6)
   m=416;control['Gaussian_entries']=n*m+m*(m+1)//2+2*m*(n-m)+control['iterations']*(n*n-m*m)+n*n
   row=dict(dataset=name,n=n,data_sha256=sha256(path.read_bytes()).hexdigest(),group_sizes=list(map(len,groups)),candidate=candidate,control=control,free_factor_entry_ratio=candidate['free_factor_kernel_entries']/control['Gaussian_entries'])
   rec['cases'].append(row);save();print(json.dumps(dict(dataset=name,candidate=candidate,control_steps=control['iterations'],ratio=row['free_factor_entry_ratio'])),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
