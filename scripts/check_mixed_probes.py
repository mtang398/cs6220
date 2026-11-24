"""Dense offline diagnostic of fixed-rank generalized Nyström probe replacement."""
import json,resource,time,sys
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky

root=Path(__file__).resolve().parents[1]
forced='--forced' in sys.argv
lam=1e-5 if '--small-lambda' in sys.argv else .001
out=root/('results/forced-probes-diagnostic.json' if forced else 'results/mixed-probes-diagnostic.json')
if lam==1e-5:out=out.with_name(out.stem+'-lambda1e-5.json')
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(150,155))
t0=time.perf_counter()
record=dict(purpose='Dense diagnostic, not production timing; fixed rank mixed coordinate and current-solve probes',status='running',regularization=lam,runs=[])
def save():
 record['elapsed_wall_seconds']=time.perf_counter()-t0
 out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
class Model:
 def __init__(self,Y,W,lam,labels):
  self.Y=Y;self.W=(W+W.T)*.5;self.lam=lam;self.labels=labels
  w,V=linalg.eigh(self.W,check_finite=False);good=w>32*np.finfo(float).eps*len(w)*max(w[-1],1e-300)
  F=(Y@V[:,good])/np.sqrt(w[good]);self.U,s,_=linalg.svd(F,full_matrices=False,check_finite=False);self.s=s*s
 def apply(self,r):
  t=self.U.T@r;c=r-self.U@t;dt=self.U.T@c;c-=self.U@dt;t+=dt
  return c/self.lam+self.U@(t/(self.s+self.lam))
 def score(self,mode):
  if mode=='trace':return float(sum(self.s))
  return float(np.log1p(self.s/self.lam).sum())
 def update(self,p,Kp,mode,it):
  v=float(p@Kp)
  if v<=1e-12*np.linalg.norm(p)*np.linalg.norm(Kp):return self,False
  p=p/np.sqrt(v);Kp=Kp/np.sqrt(v)
  cross=self.Y.T@p;Y=np.column_stack([self.Y,Kp]);W=np.block([[self.W,cross[:,None]],[cross[None,:],np.ones((1,1))]])
  k=len(self.labels);G=Y.T@Y
  try:
   w=linalg.eigvalsh(W,check_finite=False)
   if w[0]<=1e-11*w[-1]:return self,False
   inv=linalg.cho_solve(linalg.cho_factor(W,lower=True,check_finite=False),np.eye(k+1),check_finite=False)
   if mode=='trace':
    V=inv@G; scores=np.trace(V)-np.diag(V@inv)/np.diag(inv)
   else:
    H=W+G/self.lam;Hi=linalg.cho_solve(linalg.cho_factor(H,lower=True,check_finite=False),np.eye(k+1),check_finite=False)
    scores=np.linalg.slogdet(H)[1]-np.linalg.slogdet(W)[1]+np.log(np.diag(Hi))-np.log(np.diag(inv))
   if forced:
    allowed=np.array([v<0 for v in self.labels]+[False])
    if not np.any(allowed):allowed[:-1]=True
    scores[~allowed]=-np.inf
   best=int(np.argmax(scores))
   if best==k:return self,False
   take=np.arange(k+1)!=best
   new=Model(Y[:,take],W[np.ix_(take,take)],self.lam,(self.labels+[it])[0:best]+(self.labels+[it])[best+1:])
   if not forced and new.score(mode)<=self.score(mode)+1e-8:return self,False
   return new,True
  except linalg.LinAlgError:return self,False

def solve(A,K,b,M,mode,interval=1,recurrence='fcg',maxiter=2000,update_limit=None):
 x=np.zeros_like(b);r=b.copy();prev=None;updates=0;rows=[]
 for it in range(1,maxiter+1):
  z=M.apply(r);az=A@z;p=z.copy();ap=az.copy()
  if prev is not None and recurrence=='fcg':
   p0,ap0=prev
   for _ in range(2):
    beta=(p0@ap)/(p0@ap0);p-=beta*p0;ap-=beta*ap0
  curv=p@ap;num=r@p
  if curv<=0 or num<=0:break
  step=num/curv;x+=step*p;r-=step*ap
  if np.linalg.norm(r)<=1e-8*np.linalg.norm(b):
   r=b-A@x
   if np.linalg.norm(r)<=1e-8*np.linalg.norm(b):break
   prev=None;continue
  if mode and it%interval==0 and (update_limit is None or it<=update_limit):
   M,accepted=M.update(p,ap-M.lam*p,mode,it);updates+=accepted
  prev=(p,ap)
 return dict(iterations=it,true_relative_residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),updates=updates,learned_probes=sum(v>0 for v in M.labels),spectrum=list(map(float,M.s)),labels=M.labels)
with threadpool_limits(limits=1):
 for name in ['synthetic-n640-smooth','concrete-observed']:
  data=np.load(root/f'data/prepared/{name}-s0-lambda0.001.npz');b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']));K=kernel.dense();A=K+lam*np.eye(len(b))
  for k in [16,32]:
   base=initialize_rpcholesky(kernel,k,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
   variants=[(None,1,'fcg',None)]+[(m,1,'fcg',limit) for m in ['trace','logdet'] for limit in [k//4,k//2,k]] if forced else [(m,i,r,None) for m,i,r in [(None,1,'fcg'),('trace',1,'fcg'),('trace',5,'fcg'),('logdet',1,'fcg'),('logdet',5,'fcg')]]
   for mode,interval,rec,limit in variants:
    M=Model(base.columns,base.core,lam,[-int(j)-1 for j in base.indices])
    result=solve(A,K,b,M,mode,interval,rec,update_limit=limit);result.update(update_limit=limit,dataset=name,rank=k,mode=mode,interval=interval,recurrence=rec)
    record['runs'].append(result);save()
    print(json.dumps({a:v for a,v in result.items() if a not in ('spectrum','labels')}),flush=True)
record['status']='complete';save()
