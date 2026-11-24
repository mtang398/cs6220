"""Offline continuous rank truncation with optional protected current-solve secants."""
import json,resource,time,hashlib
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
root=Path(__file__).resolve().parents[1];out=root/'results/probe-compression-diagnostic.json'
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(150,155));started=time.perf_counter()
record=dict(purpose='Dense offline rank truncation diagnostic; no CPU speedup claim',status='running',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=[])
def save():
 record['elapsed_wall_seconds']=time.perf_counter()-started;out.write_text(json.dumps(record,indent=2)+'\n')
class State:
 def __init__(self,base,protected):
  self.U=base.basis.copy();self.s=np.sqrt(base.eigenvalues);self.lam=base.regularization;self.protected=protected;self.coords=[];self.updates=0;self.skips=0
 def apply(self,r):
  c=self.U.T@r;t=r-self.U@c;e=self.U.T@t;t-=self.U@e;c+=e
  return t/self.lam+self.U@(c/(self.s*self.s+self.lam))
 def update(self,p,Kp):
  F=self.U*self.s;cross=F.T@p;delta=p@Kp-cross@cross
  if delta<1e-11*max(abs(p@Kp),abs(cross@cross),1e-300):self.skips+=1;return
  u=(Kp-F@cross)/np.sqrt(delta);aug=np.column_stack([F,u]);G=aug.T@aug
  coords=[np.r_[c,0.] for c in self.coords]+[np.r_[cross,np.sqrt(delta)]]
  coords=coords[-self.protected:] if self.protected else []
  if coords:
   C=np.column_stack(coords);Q,R=linalg.qr(C,mode='full',check_finite=False)
   rank=np.linalg.matrix_rank(C,tol=1e-10*np.linalg.norm(C))
   N=Q[:,rank:]
  else:N=np.eye(len(self.s)+1)
  eig,V=linalg.eigh(N.T@G@N,check_finite=False);discard=N@V[:,0]
  Q=linalg.null_space(discard[None,:]);new=aug@Q
  U,s,Vh=linalg.svd(new,full_matrices=False,check_finite=False)
  oldtrace=float(self.s@self.s);newtrace=float(s@s)
  if not self.protected and newtrace<oldtrace*(1+1e-9):self.skips+=1;return
  self.coords=[Vh@(Q.T@c) for c in coords]
  self.U=U;self.s=s;self.updates+=1

def solve(A,b,state,limit,interval):
 x=np.zeros_like(b);r=b.copy();last=None;maxiter=2500
 for it in range(1,maxiter+1):
  p=state.apply(r);ap=A@p
  if last:
   old,aold=last
   for _ in range(2):
    beta=(old@ap)/(old@aold);p-=beta*old;ap-=beta*aold
  alpha=(r@p)/(p@ap);x+=alpha*p;r-=alpha*ap
  if np.linalg.norm(r)<=1e-8*np.linalg.norm(b):
   rr=b-A@x
   if np.linalg.norm(rr)<=1e-8*np.linalg.norm(b):break
   r=rr;last=None;continue
  if limit and it<=limit and it%interval==0:state.update(p,ap-state.lam*p)
  last=(p,ap)
 return dict(iterations=it,relative_residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),updates=state.updates,skipped=state.skips)
with threadpool_limits(limits=1):
 for name in ['synthetic-n640-smooth','concrete-observed']:
  d=np.load(root/f'data/prepared/{name}-s0-lambda0.001.npz');b=d['y_train'];kernel=GaussianKernel(d['X_train'],float(d['bandwidth']));K=kernel.dense()
  for lam in [.001,1e-5]:
   A=K+lam*np.eye(len(b))
   for k in [16,32]:
    base=initialize_rpcholesky(kernel,k,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
    for protect,limit,interval in [(0,0,1),(0,20,1),(1,20,1),(4,20,1),(0,2500,5),(1,2500,5),(4,2500,5)]:
     result=solve(A,b,State(base,protect),limit,interval);result.update(dataset=name,regularization=lam,rank=k,protected=protect,limit=limit,interval=interval)
     record['runs'].append(result);save();print(json.dumps(result),flush=True)
record['status']='complete';save()
