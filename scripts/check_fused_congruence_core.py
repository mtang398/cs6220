"""Exact congruence fusion for compressed-core PCG; dense validation."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from check_core_precision_allocation import MixedFactor,Helper
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def setup(A,b,S,H,lam):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];D=A[np.ix_(R,R)]
 def action(p):
  w,v=p[:m],p[m:];t=B@v;q=w-H(t);a=C@q+t
  return np.r_[a,D@v+B.T@(q-H(a))]
 def precondition(r):return np.r_[H(r[:m]),r[m:]/lam]
 def original(y):
  x=np.empty(n);x[S]=y[:m]-H(B@y[m:]);x[R]=y[m:];return x
 rhs=np.r_[b[S],b[R]-B.T@H(b[S])]
 return action,precondition,original,rhs

def solve(A,b,S,H,lam,keep=False):
 action,precondition,original,rhs=setup(A,b,S,H,lam);y=np.zeros(len(b));r=rhs.copy();z=precondition(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);start=process_time();status='maxiter';trajectory=[]
 for k in range(2001):
  x=original(y);rel=float(np.linalg.norm(b-A@x)/nb)
  if keep:trajectory.append(x.copy())
  if rel<=1e-8:status='converged';break
  if process_time()-start>4:status='cpu_cap';break
  if k==2000:break
  ap=action(p);den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap;z=precondition(r);new=float(r@z);p=z+(new/rho)*p;rho=new
 n=len(b);m=len(S);components=dict(FPS=n*m,factor=m*(m+1)//2,actions=k*n*n,boundary_cross=2*m*(n-m),final_verification=n*n)
 return dict(iterations=k,status=status,converged=status=='converged',original_relative_residual=rel,analytical_entries=sum(components.values()),entry_components=components),trajectory

def tiny():
 rng=np.random.default_rng(622092);n,m=29,9;F=rng.normal(size=(n,n));A=F@F.T+np.eye(n);b=rng.normal(size=n);S=np.arange(m);cf=cho_factor(A[:m,:m]+.2*np.eye(m),lower=True);H=lambda v:cho_solve(cf,v);B=A[:m,m:];T=np.eye(n);T[:m,m:]=-H(B);action,precond,original,rhs=setup(A,b,S,H,.4)
 G=T.T@A@T;v=rng.normal(size=n);action_error=float(np.linalg.norm(action(v)-G@v)/np.linalg.norm(G@v));rhs_error=float(np.linalg.norm(rhs-T.T@b)/np.linalg.norm(b))
 row,ys=solve(A,b,S,H,.4,keep=True);M=T@np.block([[H(np.eye(m)),np.zeros((m,n-m))],[np.zeros((n-m,m)),np.eye(n-m)/.4]])@T.T
 x=np.zeros(n);r=b.copy();z=M@r;p=z.copy();rho=float(r@z);errors=[]
 for k,ref in enumerate(ys):
  errors.append(float(np.linalg.norm(x-ref)/max(np.linalg.norm(ref),1.)))
  if k==len(ys)-1:break
  ap=A@p;alpha=rho/(p@ap);x+=alpha*p;r-=alpha*ap;z=M@r;new=float(r@z);p=z+new/rho*p;rho=new
 # Floating-point arithmetic does not preserve exact later CG trajectories.
 reference=cho_solve(cho_factor(A,lower=True),b)
 endpoint_error=float(np.linalg.norm(ys[-1]-reference)/np.linalg.norm(reference))
 direct_endpoint_error=float(np.linalg.norm(x-reference)/np.linalg.norm(reference))
 assert action_error<1e-12 and rhs_error<1e-12 and max(errors[:10])<1e-12 and row['converged']
 assert max(endpoint_error,direct_endpoint_error)<1e-7
 return dict(action_relative_error=action_error,rhs_relative_error=rhs_error,max_original_iterate_relative_difference=max(errors),first_ten_iterate_max_difference=max(errors[:10]),endpoint_relative_solution_error=endpoint_error,direct_endpoint_relative_solution_error=direct_endpoint_error,iterations=row['iterations'])

def main():
 out=ROOT/'results/fused-congruence-core.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();path=ROOT/'results/exact-outer-mixed-core.json';prior=json.loads(path.read_text());helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
 files=[Path(__file__).resolve(),ROOT/'scripts/check_core_precision_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so',path]
 rec=dict(status='running',sources={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in files},protocol=dict(core=592,shift=1e-5,mechanism='Exact G=T^T A T with T=[[I,-HB],[0,I]], fixed preconditioner diag(H,lambda^-1 I). H is compressed stored-factor inverse.',count='One full n² worth of entries per fused action; one initial and final cross pass; literal final original verification. All per-iterate reconstruction/verification granted free for trajectory gate.',scope='Dense algebra/cost gate; native memory/runtime and solve-informed novelty unestablished.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  rec['tiny_check']=tiny();save()
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   data_path=ROOT/'data/prepared'/fn
   with np.load(data_path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;factor=MixedFactor(A[np.ix_(S,S)],range(8),helper)
   row,_=solve(A,b,S,factor.solve,1e-5);control=next(c for c in prior['cases'] if c['dataset']==name);assert control['data_sha256']==sha256(data_path.read_bytes()).hexdigest()
   row.update(dataset=name,data_sha256=control['data_sha256'],reference=control['reference'],direct_original_pcg_iterations=control['runs'][0]['iterations'],entry_ratio=row['analytical_entries']/control['reference']['entries']);rec['cases'].append(row);save();print(json.dumps(row),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
