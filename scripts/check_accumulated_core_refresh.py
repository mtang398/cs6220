"""Streamed column-defect estimate for direction-dependent core-product omission."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_fused_congruence_core as prior
from check_core_precision_allocation import MixedFactor,Helper
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def run(A,b,S,factor,mode):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];D=A[np.ix_(R,R)];L=factor.dense_L();E=C-L@L.T;u=np.finfo(float).eps/2;gamma=2*m*u/(1-2*m*u);eta=1/((1-u)**2*(1-gamma));trace=m*(1+1e-5);bound=float((np.sqrt(2)*eta*np.linalg.norm(factor.epsilon)+2*gamma*trace)/(1-gamma*np.sqrt(m)));H=factor.solve
 y=np.zeros(n);r=np.r_[b[S],b[R]-B.T@H(b[S])];z=np.r_[H(r[:m]),r[m:]/1e-5];p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);s=np.zeros(m);refresh=[];omissions=0;status='maxiter';start=process_time();max_defect=0.
 for k in range(2001):
  x=np.empty(n);x[S]=y[:m]-H(B@y[m:]);x[R]=y[m:];rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-start>6:status='cpu_cap';break
  if k==2000:break
  v=p[m:];t=B@v;d=D@v;q=p[:m]-H(t)
  a=(C@q if mode=='exact' else L@(L.T@q))+t;ap=np.r_[a,d+B.T@(q-H(a))];den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap
  do=False
  if mode!='exact':
   omissions+=1;s+=alpha*q
   do=(k+1)%8==0 if mode=='periodic8' else bound*np.linalg.norm(s)>.1*np.linalg.norm(r)
   if do:
    e=C@s-L@(L.T@s);r[:m]-=e;r[m:]+=B.T@H(e);s.fill(0);refresh.append(k+1)
    # Diagnostic original residual identity; no diagnostic value controls decisions.
    xx=np.empty(n);xx[S]=y[:m]-H(B@y[m:]);xx[R]=y[m:]
    mapped=np.r_[r[:m],r[m:]+B.T@H(r[:m])]
    actual=(b-A@xx)[np.r_[S,R]];max_defect=max(max_defect,float(np.linalg.norm(actual-mapped)/nb))
  z=np.r_[H(r[:m]),r[m:]/1e-5];new=float(r@z)
  p=z.copy() if do and mode=='adaptive_restart' else z+new/rho*p
  rho=new
 components=dict(FPS=n*m,factor=m*(m+1)//2,actions=k*n*n-omissions*m*m,refresh=len(refresh)*(m*m+m*(n-m)),boundary_cross=2*m*(n-m),final_verification=n*n)
 return dict(method=mode,iterations=k,status=status,converged=status=='converged',original_relative_residual=rel,omitted=omissions,refresh_steps=refresh,refresh_identity_defect=max_defect,entry_components=components,analytical_entries=sum(components.values()))

def main():
 out=ROOT/'results/accumulated-core-error-refresh.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
 files=[Path(__file__).resolve(),Path(prior.__file__).resolve(),ROOT/'scripts/check_core_precision_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},protocol=dict(core=592,shift=1e-5,choice='Approximate every core product, accumulate s=sum alpha*q; replace residual by [-Es,+BT H Es] at refresh. Adaptive trigger bound*norm(s)>.1*norm(transformed recursive residual); periodic8 control; adaptive_restart isolates restart cost.',cost='Dense gate: factors, selection, vector arithmetic and per-step diagnostic true residual free. Kernel entries charge actions, each correction core+cross, setup, boundaries and final verification. Original 1e-8 acceptance; no native memory/speed claim.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;factor=MixedFactor(A[np.ix_(S,S)],range(8),helper);case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),runs=[])
   for mode in ['exact','periodic8','adaptive','adaptive_restart']:
    row=run(A,b,S,factor,mode);case['runs'].append(row);print(json.dumps(dict(dataset=name,**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
