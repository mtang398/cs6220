"""Distinguish core conditioning from errors in the congruence coupling."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve
from threadpoolctl import threadpool_limits
from check_int16_core_capacity import quantize
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def run(A,b,S,Q,L,mode):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];D=A[np.ix_(R,R)];H=lambda v:cho_solve((Q,True),v,check_finite=False)
 def J(v):
  if mode=='exact_transform_oracle':return cho_solve((L,True),v,check_finite=False)
  h=H(v);return h+H(v-C@h)
 y=np.zeros(n);r=np.r_[b[S],b[R]-B.T@J(b[S])];z=np.r_[H(r[:m]),r[m:]/1e-5];p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);status='maxiter';t=process_time()
 for k in range(2001):
  x=np.empty(n);x[S]=y[:m]-J(B@y[m:]);x[R]=y[m:];rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-t>4:status='cpu_cap';break
  if k==2000:break
  v=p[m:];t0=B@v;q=p[:m]-J(t0);a=C@q+t0;ap=np.r_[a,D@v+B.T@(q-J(a))];den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap;z=np.r_[H(r[:m]),r[m:]/1e-5];new=float(r@z);p=z+new/rho*p;rho=new
 extra=(2*k+2)*m*m if mode=='refined_transform' else 0
 entries=n*m+m*(m+1)//2+k*n*n+extra+2*m*(n-m)+n*n
 return dict(method=mode,status=status,iterations=k,original_relative_residual=rel,extra_core_products=2*k+2 if mode=='refined_transform' else 0,analytical_entries=entries)

def main():
 out=ROOT/'results/refined-int16-congruence.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();priorpath=ROOT/'results/int16-core-capacity.json';prior=json.loads(priorpath.read_text());files=[Path(__file__).resolve(),ROOT/'scripts/check_int16_core_capacity.py',priorpath]
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(core=818,shift=1e-5,mechanism='Use J=2H−HCH only in T=[[I,-JB],[0,I]], keep diag(H,lambda^-1I) as preconditioner; exact G=T^TAT. Same postquantized int16 factor. Exact-transform reference free oracle.',cost='Charge2additionalcoreproducts peraction plus2boundarycoreproducts forJ. Dense original stopping, all factorization/decoding and inverse arithmetic free. Postquantized exact factor unavailable in1MiB via current streamed construction. Static diagnostic, not adaptive success.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for old,filename in zip(prior['cases'],['synthetic-n2560-smooth-s0-lambda0.001.npz','synthetic-n2560-mixed-s0-lambda0.001.npz','concrete-observed-s0-lambda1e-05.npz','airfoil-observed-s0-lambda1e-05.npz']):
   path=ROOT/'data/prepared'/filename;assert sha256(path.read_bytes()).hexdigest()==old['data_sha256']
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(818);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;L=cholesky(A[np.ix_(S,S)],lower=True);Q=quantize(L,'int16');case=dict(dataset=old['dataset'],prior_controls=old['runs'],runs=[])
   for mode in ['refined_transform','exact_transform_oracle']:
    row=run(A,b,S,Q,L,mode);case['runs'].append(row);print(json.dumps(dict(dataset=old['dataset'],**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
