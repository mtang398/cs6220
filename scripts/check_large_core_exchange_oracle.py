"""Changed-regime exact principal-core exchange oracle; all search/rebuild costs free."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]

def build(A,b,J):
 R=np.setdiff1d(np.arange(len(b)),J);C=A[np.ix_(J,J)];F=cho_factor(C,lower=True);B=A[np.ix_(J,R)];V=cho_solve(F,B);D=A[np.ix_(R,R)]-B.T@V;rhs=b[R]-B.T@cho_solve(F,b[J])
 return R,F,B,V,D,rhs

def solve(A,b,J,op,x0=None,checkpoint=None):
 R,F,B,V,D,rhs=op;y=np.zeros(len(R)) if x0 is None else x0[R].copy();r=rhs-D@y;p=r.copy();rho=float(r@r);nb=np.linalg.norm(b);saved=None;iterations=0
 for it in range(2000):
  # Literal reduced residual equals full residual after exact-core reconstruction.
  true=rhs-D@y
  if np.linalg.norm(true)<=1e-8*nb:break
  ap=D@p;alpha=rho/float(p@ap);y+=alpha*p;r-=alpha*ap;new=float(r@r);p=r+(new/rho)*p;rho=new;iterations=it+1
  if checkpoint is not None and iterations==checkpoint:
   full=np.empty(len(b));full[R]=y;full[J]=cho_solve(F,b[J]-B@y);saved=full.copy()
 full=np.empty(len(b));full[R]=y;full[J]=cho_solve(F,b[J]-B@y);res=float(np.linalg.norm(b-A@full)/nb)
 return dict(iterations=iterations,converged=res<=1.00001e-8,true_relative_residual=res),saved

def main():
 out=ROOT/'results/large-core-exchange-oracle.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(110,115));t,c=perf_counter(),process_time()
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(n=2048,core=416,checkpoint=10,rtol=1e-8,pool_per_selector=4,selectors=['current squared residual / Schur diagonal','matrix-only Schur diagonal'],removal='maximize exact principal determinant ratio h_ii*S_jj+w_i² for fixed j',free='Dense kernels, full candidate scores, selection, core rebuilds, all diagnostic verification; not native memory/time claim',controls=['uninterrupted same core','identical core restarted at checkpoint'],scope='Changed large-core regime only; earlier full scope preserved'),runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n')
 with threadpool_limits(limits=1):
  for ds in ['smooth','mixed']:
   path=ROOT/'data/prepared'/f'synthetic-n2560-{ds}-s0-lambda0.001.npz'
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));J=[0];dist=2-2*K[:,0]
   for _ in range(415):
    j=int(np.argmax(dist));J.append(j);dist=np.minimum(dist,2-2*K[:,j])
   J=np.asarray(J)
   for lam in [.001,1e-5]:
    A=K+lam*np.eye(len(K));op=build(A,b,J);base,state=solve(A,b,J,op,checkpoint=10);assert state is not None
    restart,_=solve(A,b,J,op,state);R,F,B,V,D,rhs=op;r=(b-A@state)[R];diag=np.diag(D);invdiag=np.diag(cho_solve(F,np.eye(len(J))))
    pools={'current':np.argsort(r*r/diag)[-4:][::-1], 'matrix':np.argsort(diag)[-4:][::-1]}
    row=dict(dataset=ds,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest(),baseline=base,restart=restart,candidates=[])
    for rule,pool in pools.items():
     for rank,jpos in enumerate(pool):
      w=V[:,jpos];ratios=invdiag*diag[jpos]+w*w;i=int(np.argmax(ratios));newJ=J.copy();newJ[i]=R[jpos];newop=build(A,b,newJ);result,_=solve(A,b,newJ,newop,state)
      row['candidates'].append(dict(selector=rule,pool_rank=rank,removed=int(J[i]),added=int(R[jpos]),determinant_ratio=float(ratios[i]),current_residual_score=float(r[jpos]**2/diag[jpos]),prefix_plus_iterations=10+result['iterations'],**result))
    rec['runs'].append(row);save()
 rec['status']='complete';save();print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
