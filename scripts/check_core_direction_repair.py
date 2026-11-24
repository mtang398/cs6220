"""Dense gate for residual plus last-direction repair with core-supported errors."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
from check_core_precision_allocation import MixedFactor,Helper
from check_fused_congruence_core import solve as exact_solve
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def run(A,b,S,factor,mode):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];D=A[np.ix_(R,R)];L=factor.dense_L();E=C-L@L.T;H=factor.solve
 nb=np.linalg.norm(b);threshold=.5e-8*nb;y=np.zeros(n);r=np.r_[b[S],b[R]-B.T@H(b[S])];z=np.r_[H(r[:m]),r[m:]/1e-5];p=z.copy();rho=float(r@z);s=np.zeros(m);refresh=[];identities=[];status='maxiter';start=process_time()
 for k in range(2001):
  x=np.empty(n);x[S]=y[:m]-H(B@y[m:]);x[R]=y[m:];rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-start>4:status='cpu_cap';break
  if k==2000:break
  h=H(B@p[m:]);q=p[:m]-h;ap=np.r_[L@(L.T@p[:m]),D@p[m:]-B.T@h];den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap;s+=alpha*q
  trigger=(k+1)%8==0 if mode.startswith('periodic') else np.linalg.norm(r)<=threshold
  repaired=False
  if trigger:
   e=E@s;r[:m]-=e;r[m:]+=B.T@H(e);s.fill(0);refresh.append(k+1)
   if mode.endswith('repair'):
    eq=E@q;ap[:m]+=eq;ap[m:]-=B.T@H(eq);curvature=float(p@ap)
    if curvature<=0:status='nonpositive_repair_curvature';break
    gamma=float(p@r)/curvature;y+=gamma*p;r-=gamma*ap
    z=np.r_[H(r[:m]),r[m:]/1e-5];beta=-float(ap@z)/curvature;direction=z+beta*p
    identities.append(dict(residual_orthogonality=float(abs(p@r)/max(np.linalg.norm(p)*np.linalg.norm(r),1e-300)),last_conjugacy=float(abs(ap@direction)/max(np.linalg.norm(ap)*np.linalg.norm(direction),1e-300))))
    p=direction;rho=float(r@z);repaired=True
  if not repaired:
   z=np.r_[H(r[:m]),r[m:]/1e-5];new=float(r@z)
   p=z.copy() if trigger and mode.endswith('restart') else z+new/rho*p;rho=new
 entries=n*m+m*(m+1)//2+k*(n*n-m*m)+len(refresh)*n*m+2*m*(n-m)+n*n
 return dict(method=mode,status=status,iterations=k,original_relative_residual=rel,refresh_steps=refresh,repair_checks=identities,analytical_entries=entries)

def main():
 out=ROOT/'results/core-direction-repair.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();helper=Helper(ROOT/'results/mixed_column_factor_first8.so');files=[Path(__file__).resolve(),ROOT/'scripts/check_core_precision_allocation.py',ROOT/'scripts/check_fused_congruence_core.py',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(core=592,shift=1e-5,mechanism='On refresh compute Es and Eq, repair residual and preceding direction image; exact line correction gamma=(pTr)/(pTGp), then one-direction conjugation beta=−(Gp)Tz/(pTGp). No full-history conjugacy claim.',cost='Optimistic dense gate: Es/Eq share one two-RHS kernel core traversal and two-RHS cross traversal, priced nm entries per refresh. Extra arithmetic, factor construction, per-step true stopping and history storage free; native memory/cost unproven.',controls='Exact fusion, terminal restart, periodic8 FR; no threshold sweep.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;factor=MixedFactor(A[np.ix_(S,S)],range(8),helper);control,_=exact_solve(A,b,S,factor.solve,1e-5);control['method']='exact';case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),runs=[control])
   for mode in ['terminal_restart','terminal_repair','periodic_fr','periodic_repair']:
    row=run(A,b,S,factor,mode);case['runs'].append(row);print(json.dumps(dict(dataset=name,**{k:v for k,v in row.items() if k!='repair_checks'})),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
