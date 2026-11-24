"""Offline fixed-column exchange using a reused one-sided matrix-residual sketch."""
import json,resource,time,hashlib
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from nystrom_project.numerics import GaussianKernel,NystromPreconditioner,initialize_rpcholesky
root=Path(__file__).resolve().parents[1];out=root/'results/operator-sketch-exchange-diagnostic.json'
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(145,150));t0=time.perf_counter()
result=dict(status='running',purpose='Dense offline diagnostic of current-solve operator images versus independent Gaussian sketch for actual-column swaps',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=[])
def save():
 result['elapsed_wall_seconds']=time.perf_counter()-t0;out.write_text(json.dumps(result,indent=2)+'\n')
def solve(A,b,M,x=None,prefix=False):
 x=np.zeros_like(b) if x is None else x.copy();r=b-A@x;z=M.apply(r);p=z.copy();rho=r@z;P=[];AP=[]
 for it in range(1,2501):
  ap=A@p;alpha=rho/(p@ap)
  if prefix:P.append(p.copy());AP.append(ap.copy())
  x+=alpha*p;r-=alpha*ap
  if prefix and it==8:return x,np.column_stack(P),np.column_stack(AP)
  if np.linalg.norm(r)<1e-8*np.linalg.norm(b):
   r=b-A@x
   if np.linalg.norm(r)<1e-8*np.linalg.norm(b):break
   z=M.apply(r);p=z.copy();rho=r@z;continue
  z=M.apply(r);rho2=r@z;p=z+(rho2/rho)*p;rho=rho2
 return dict(iterations=it,relative_residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b)))
def score(M,Q,KQ):return float(np.linalg.norm(KQ-M.basis@(M.eigenvalues[:,None]*(M.basis.T@Q)))**2)
def exchange(K,M,Q,KQ):
 oldscore=score(M,Q,KQ);events=[];visited=set()
 for _ in range(4):
  error=KQ-M.basis@(M.eigenvalues[:,None]*(M.basis.T@Q));weights=np.sum(error*error,axis=1)
  weights[M.indices]=-np.inf
  for j in visited:weights[j]=-np.inf
  j=int(np.argmax(weights));visited.add(j);inds=np.r_[M.indices,j];C=K[:,inds]
  best=M;bestscore=oldscore;removed=None
  for i in range(len(M.indices)):
   take=np.arange(len(inds))!=i
   trial=NystromPreconditioner(inds[take],C[:,take],M.regularization)
   v=score(trial,Q,KQ)
   if v<bestscore*(1-1e-9):best,bestscore,removed=trial,v,int(inds[i])
  events.append(dict(candidate=j,removed=removed,old_score=oldscore,new_score=bestscore));M=best;oldscore=bestscore
 return M,events
with threadpool_limits(limits=1):
 for name in ['synthetic-n640-smooth','concrete-observed']:
  data=np.load(root/f'data/prepared/{name}-s0-lambda0.001.npz');b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']));K=kernel.dense()
  for lam in [.001,1e-5]:
   A=K+lam*np.eye(len(b))
   for k in [16,32]:
    M=initialize_rpcholesky(kernel,k,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
    x,P,AP=solve(A,b,M,prefix=True);Q,R=linalg.qr(P,mode='economic');KQ=linalg.solve_triangular(R.T,(AP-lam*P).T,lower=True).T
    np.testing.assert_allclose(KQ,K@Q,rtol=1e-8,atol=1e-9)
    learned,events=exchange(K,M,Q,KQ)
    gaussian,_=linalg.qr(np.random.default_rng(73).normal(size=(len(b),8)),mode='economic')
    static,static_events=exchange(K,M,gaussian,K@gaussian)
    old=solve(A,b,M);restart=solve(A,b,M,x);new=solve(A,b,learned,x);control=solve(A,b,static)
    restart['iterations']+=8;new['iterations']+=8
    row=dict(dataset=name,regularization=lam,rank=k,fixed=old,restart=restart,solve_sketch=new,gaussian_sketch=control,events=events,static_events=static_events)
    result['cases'].append(row);save();print(json.dumps({a:v for a,v in row.items() if 'events' not in a}),flush=True)
result['status']='complete';save()
