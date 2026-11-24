"""Optimistic exact-error-triggered refresh after factor-column promotion."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve
from threadpoolctl import threadpool_limits
from check_int24_precision_selection import run,quantize,ROOT,SharedKernel,Budget

def cheap(A,b,S,Q,x0):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];D=A[np.ix_(R,R)];E=C-Q@Q.T;H=lambda v:cho_solve((Q,True),v,check_finite=False)
 nb=np.linalg.norm(b);rr=b-A@x0;y=np.zeros(n);r=np.r_[rr[S],rr[R]-B.T@H(rr[S])];z=np.r_[H(r[:m]),r[m:]/1e-5];p=z.copy();rho=float(r@z);s=np.zeros(m);refresh=[];status='maxiter';t=process_time()
 for k in range(2001):
  x=np.empty(n);x[S]=y[:m]-H(B@y[m:]);x[R]=y[m:];x+=x0;rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-t>4:status='cpu_cap';break
  if k==2000:break
  h=H(B@p[m:]);q=p[:m]-h;ap=np.r_[Q@(Q.T@p[:m]),D@p[m:]-B.T@h];den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap;s+=alpha*q
  e=E@s # exact oracle, free scoring; charge a core pass if a correction is taken
  if np.linalg.norm(e)>.1*np.linalg.norm(r):r[:m]-=e;r[m:]+=B.T@H(e);s.fill(0);refresh.append(k+1)
  z=np.r_[H(r[:m]),r[m:]/1e-5];new=float(r@z);p=z+new/rho*p;rho=new
 entries=k*(n*n-m*m)+len(refresh)*n*m+2*m*(n-m)+n*n
 return dict(status=status,iterations=k,original_relative_residual=rel,refresh_steps=refresh,analytical_entries=entries)

def main():
 out=ROOT/'results/int24-promoted-refresh.json';priorpath=ROOT/'results/int24-coupled-precision-selection.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();prior=json.loads(priorpath.read_text());files=[Path(__file__).resolve(),priorpath,ROOT/'scripts/check_int24_precision_selection.py',ROOT/'scripts/check_int24_core_capacity.py']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(choice='Reuse coupled column masks and common4step prefix. Oracle norm(E s)>.1norm(transformed r) triggers cheap residual correction; current/initial/matrix masks, unchanged660 and full675upfront controls.',scope='All E applications/selection/factor regeneration and per-step true stopping free; correction kernel entries, commonprefix/boundaries/finalverification charged. Tests opportunity for cheaper approximate actions, not native runtime or feasible memory.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for case,fn in zip(prior['cases'],['synthetic-n2560-smooth-s0-lambda0.001.npz','synthetic-n2560-mixed-s0-lambda0.001.npz','concrete-observed-s0-lambda1e-05.npz','airfoil-observed-s0-lambda1e-05.npz']):
   path=ROOT/'data/prepared'/fn;assert sha256(path.read_bytes()).hexdigest()==case['data_sha256']
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));allS=kernel.farthest(675);S=allS[:660];A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;L=cholesky(A[np.ix_(S,S)],lower=True);Q=quantize(L,'int24');prefix,pr=run(A,b,S,Q,stop_steps=4);rows=[]
   for mode in ['full675','unchanged','matrix','initial','current']:
    SS=S;QQ=Q.copy();xx=prefix
    if mode=='full675':SS=allS;QQ=quantize(cholesky(A[np.ix_(SS,SS)],lower=True),'int24');xx=np.zeros(len(b))
    elif mode!='unchanged':mask=case['masks'][mode]['columns'];QQ[:,mask]=L[:,mask]
    row=cheap(A,b,SS,QQ,xx);row['method']=mode
    if mode=='full675':row['analytical_entries']+=len(b)*675+675*676//2
    else:row['analytical_entries']+=pr['analytical_entries'];row['iterations']+=4
    rows.append(row);print(json.dumps(dict(dataset=case['dataset'],**row)),flush=True)
   rec['cases'].append(dict(dataset=case['dataset'],runs=rows,exact_control=case['runs'][0]));save()
 rec['status']='complete';save()
if __name__=='__main__':main()
