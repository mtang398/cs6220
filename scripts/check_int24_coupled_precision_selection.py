"""Dense optimistic core-factor precision selection with matched restart controls."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from threadpoolctl import threadpool_limits
from check_int24_core_capacity import quantize
from check_fused_congruence_core import setup
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def run(A,b,S,Q,x0=None,stop_steps=None):
 n=len(b);m=len(S);nb=np.linalg.norm(b);x0=np.zeros(n) if x0 is None else x0
 residual=b-A@x0;H=lambda v:cho_solve((Q,True),v,check_finite=False)
 action,precondition,original,rhs=setup(A,residual,S,H,1e-5)
 y=np.zeros(n);r=rhs.copy();z=precondition(r);p=z.copy();rho=float(r@z);status='maxiter';t=process_time()
 for k in range(2001):
  x=x0+original(y);rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if stop_steps is not None and k==stop_steps:status='prefix';break
  if process_time()-t>4:status='cpu_cap';break
  if k==2000:break
  ap=action(p);den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap;z=precondition(r);new=float(r@z);p=z+new/rho*p;rho=new
 entries=n*m+m*(m+1)//2+k*n*n+2*m*(n-m)+n*n
 return x,dict(status=status,iterations=k,original_relative_residual=rel,analytical_entries=entries)

def select(scores,m):
 base=(3*m*m+109*m)//2-180+64+9*m # actual duplicate diag plus promotion flags/offset slots
 selected=[];used=base
 costs=5*(m-np.arange(m)-1)
 candidates=list(range(8,m-1));candidates.sort(key=lambda j:(-scores[j]/costs[j],j))
 for j in candidates:
  if scores[j]>0 and used+costs[j]<=721200:selected.append(j);used+=int(costs[j])
 return selected,used

def main():
 out=ROOT/'results/int24-coupled-precision-selection.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();files=[Path(__file__).resolve(),ROOT/'scripts/check_int24_core_capacity.py',ROOT/'scripts/check_fused_congruence_core.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(core=660,static_full_core=675,prefix_steps=4,payload_budget=721200,selection='Greedy squared first-order full block-preconditioner-error score per extra byte. Matrix-only Frobenius score, initial full b action score, current full original residual action score. FP64 column promotions, diagonal alreadyFP64. Common4-step exact-fusion prefix, matched original iterate and explicit restart.',scope='Dense optimistic headroom. Full FP64 reference factor, inverse derivative matrices, mask scoring and factor regeneration are free. Payload includes promotion flags/offsets; full workspace not feasible. Common prefixsetup charged once; restart regeneration free; both boundary/verification entries charged; per-iterate true stopping free. No native/adaptive-success claim.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));allS=kernel.farthest(675);S=allS[:660];A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5
   L=cholesky(A[np.ix_(S,S)],lower=True);Q=quantize(L,'int24');Delta=L-Q
   prefix,pr=run(A,b,S,Q,stop_steps=4);assert pr['status']=='prefix'
   residual=b-A@prefix;P=cho_solve((Q,True),Delta,check_finite=False);V=solve_triangular(Q.T,np.eye(660),lower=False,check_finite=False)
   R=np.setdiff1d(np.arange(len(b)),S);B=A[np.ix_(S,R)]
   H=lambda v:cho_solve((Q,True),v,check_finite=False)
   BP=B.T@P;BV=B.T@V
   UP=np.vstack([P,np.zeros((len(R),660))]);UV=np.vstack([V,np.zeros((len(R),660))])
   WP=np.vstack([.5*P+H(B@BP)/1e-5,-BP/1e-5]);WV=np.vstack([.5*V+H(B@BV)/1e-5,-BV/1e-5])
   F=np.stack([UP,UV,WP,WV]);perm=[3,2,1,0]
   gram=np.einsum('anm,bnm->abm',F,F);static=np.sum(gram*gram[perm][:,perm],axis=(0,1))
   scores={'matrix':static}
   for label,rhs in [('initial',b),('current',residual)]:
    coeff=np.einsum('anm,n->am',F,rhs[np.r_[S,R]])
    terms=np.einsum('anm,am->nm',F,coeff[perm]);scores[label]=np.sum(terms*terms,axis=0)
   masks={key:select(value,660) for key,value in scores.items()};case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),prefix=pr,masks={key:dict(columns=v[0],payload_bytes=v[1]) for key,v in masks.items()},runs=[])
   for mode in ['full675_upfront','unchanged_restart','matrix_upfront','matrix_restart','initial_restart','current_restart']:
    SS=S;QQ=Q.copy();start=prefix if mode.endswith('restart') else None
    if mode=='full675_upfront':SS=allS;QQ=quantize(cholesky(A[np.ix_(SS,SS)],lower=True),'int24')
    elif mode!='unchanged_restart':
     key=mode.split('_')[0];mask=masks[key][0];QQ[:,mask]=L[:,mask]
    x,row=run(A,b,SS,QQ,x0=start);del x
    if start is not None:
     row['analytical_entries']+=pr['analytical_entries']-(len(b)*660+660*661//2);row['iterations']+=4
    row['method']=mode;case['runs'].append(row);print(json.dumps(dict(dataset=name,**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
