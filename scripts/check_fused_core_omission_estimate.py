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
 y=np.zeros(n);r=np.r_[b[S],b[R]-B.T@H(b[S])];z=np.r_[H(r[:m]),r[m:]/1e-5];p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);budget=.25e-8*nb;used=0.;omissions=[];status='maxiter';start=process_time();gap=np.zeros(m)
 for k in range(2001):
  x=np.empty(n);x[S]=y[:m]-H(B@y[m:]);x[R]=y[m:];rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-start>4:status='cpu_cap';break
  if k==2000:break
  v=p[m:];t=B@v;d=D@v;q=p[:m]-H(t);cheap=L@(L.T@q)
  approx_den=float(q@cheap+2*q@t+v@d);take=False;alpha0=None;errornorm=None
  if mode!='exact' and approx_den>0:
   alpha0=rho/approx_den;errornorm=float(np.linalg.norm(E@q)) if mode=='direction_oracle' else bound*float(np.linalg.norm(q))
   take=abs(alpha0)*errornorm<=budget-used
  a=(cheap if take else C@q)+t;ap=np.r_[a,d+B.T@(q-H(a))];den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den
  if take:
   # Recheck the budget with the actually used floating-point step length.
   if abs(alpha)*errornorm>budget-used:
    a=C@q+t;ap=np.r_[a,d+B.T@(q-H(a))];alpha=rho/float(p@ap);take=False
   else:used+=abs(alpha)*errornorm;gap+=alpha*(E@q);omissions.append(k+1)
  y+=alpha*p;r-=alpha*ap;z=np.r_[H(r[:m]),r[m:]/1e-5];new=float(r@z);p=z+new/rho*p;rho=new
 # For computed recursive transformed residual, original residual mapping has +BT Hrtop.
 mapped=np.empty(n);mapped[S]=r[:m];mapped[R]=r[m:]+B.T@H(r[:m]);actual=b-A@x
 # actual - recursive = -sum alpha*(A-Atilde)*original_direction, supported on core.
 predicted=np.zeros(n);predicted[S]=-gap
 gap_defect=float(np.linalg.norm(actual-mapped-predicted)/nb)
 components=dict(FPS=n*m,factor=m*(m+1)//2,actions=k*n*n-len(omissions)*m*m,boundary_cross=2*m*(n-m),final_verification=n*n)
 return dict(method=mode,iterations=k,status=status,converged=status=='converged',original_relative_residual=rel,omitted_steps=omissions,certificate_used_over_norm_b=used/nb,actual_gap_over_norm_b=float(np.linalg.norm(gap)/nb),gap_identity_defect_over_norm_b=gap_defect,core_error_estimate=bound,offline_core_error_frobenius=float(np.linalg.norm(E,'fro')),entry_components=components,analytical_entries=sum(components.values()))

def main():
 out=ROOT/'results/fused-core-omission-streamed-estimate.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
 files=[Path(__file__).resolve(),Path(prior.__file__).resolve(),ROOT/'scripts/check_core_precision_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},protocol=dict(core=592,shift=1e-5,choice='Replace Cq with Ltilde LtildeT q when accumulated sum|alpha|*error_bound fits .25e-8||b||. Use stored column-defect norms plus FP64 summation allowance; original residual verification mandatory.',identity='T^-T transforms [e,-BT H e] into [e,0], so omitted-core residual gap is core-supported in original coordinates.',cost='Dense arithmetic/verification and selector costs free in this trajectory test. E computed only for diagnostic gap, never for decisions. Bound estimate uses factor.epsilon available during streamed setup, m and shift. Floating-point model/kernel/application error prevents full residual-certificate claim; no native speed claim.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;factor=MixedFactor(A[np.ix_(S,S)],range(8),helper);case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),runs=[])
   for mode in ['exact','column_defect_estimate']:
    row=run(A,b,S,factor,mode);case['runs'].append(row);print(json.dumps(dict(dataset=name,**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
