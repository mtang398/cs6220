"""Optimistic retained-byte gate for low-rank correction of inverse cross coupling."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve,svd,solve_triangular
from threadpoolctl import threadpool_limits
from check_int16_core_capacity import quantize
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def run(A,b,S,Q,Z):
 n=len(b);m=len(S);R=np.setdiff1d(np.arange(n),S);C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];D=A[np.ix_(R,R)];H=lambda v:cho_solve((Q,True),v,check_finite=False)
 y=np.zeros(n);r=np.r_[b[S],b[R]-Z.T@b[S]];z=np.r_[H(r[:m]),r[m:]/1e-5];p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);status='maxiter';t=process_time()
 for k in range(2001):
  x=np.empty(n);x[S]=y[:m]-Z@y[m:];x[R]=y[m:];rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if process_time()-t>4:status='cpu_cap';break
  if k==2000:break
  v=p[m:];q=p[:m]-Z@v;a=C@q+B@v;ap=np.r_[a,D@v+B.T@q-Z.T@a];den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;y+=alpha*p;r-=alpha*ap;z=np.r_[H(r[:m]),r[m:]/1e-5];new=float(r@z);p=z+new/rho*p;rho=new
 return dict(status=status,iterations=k,original_relative_residual=rel,analytical_entries=n*m+m*(m+1)//2+k*n*n+2*m*(n-m)+n*n)

def main():
 out=ROOT/'results/cached-coupling-error.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();priorpath=ROOT/'results/int16-core-capacity.json';prior=json.loads(priorpath.read_text());files=[Path(__file__).resolve(),ROOT/'scripts/check_int16_core_capacity.py',priorpath]
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(rank=8,payload_budget=721200,precision='Postquantized int16, first8columns/diagonal/scalesFP64; duplicate first8diagonals included. Core reduced so factor+8*n*rank <=721200.',mechanism='Z=HB+UV^T approximates C^-1B, exact congruence T=[[I,-Z],[0,I]], diag(H,lambda^-1I). Raw SVD and C-energy weighted SVD via L^T deltaZ.',cost='Denseconstruction, reference inverse, SVD, decodedfactor/cache and stopping free. n²entries/action models streamed HB and BT H applications plus lowrankarithmetic; no native memory/runtime or current-solve benefit. Accuratecoupling is overbudget oracle.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for old,filename in zip(prior['cases'],['synthetic-n2560-smooth-s0-lambda0.001.npz','synthetic-n2560-mixed-s0-lambda0.001.npz','concrete-observed-s0-lambda1e-05.npz','airfoil-observed-s0-lambda1e-05.npz']):
   path=ROOT/'data/prepared'/filename;assert sha256(path.read_bytes()).hexdigest()==old['data_sha256']
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   n=len(b);m=max(k for k in range(8,n) if k*k+63*k-152+64*n<=721200);payload=m*m+63*m-152+64*n
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(m);R=np.setdiff1d(np.arange(n),S);A=kernel.block(X,X);A.flat[::n+1]+=1e-5;C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];L=cholesky(C,lower=True);Q=quantize(L,'int16');H=lambda v:cho_solve((Q,True),v,check_finite=False);base=H(B);truth=cho_solve((L,True),B,check_finite=False);delta=truth-base
   U,s,V=svd(delta,full_matrices=False,check_finite=False);raw=(U[:,:8]*s[:8])@V[:8];Uw,sw,Vw=svd(L.T@delta,full_matrices=False,check_finite=False);weighted=solve_triangular(L.T,Uw[:,:8]*sw[:8],lower=False,check_finite=False)@Vw[:8]
   norm=float(np.linalg.norm(L.T@delta,'fro'));case=dict(dataset=old['dataset'],core=m,payload_bytes=payload,weighted_error_fraction_after_rank8=float(np.linalg.norm(L.T@(delta-weighted),'fro')/norm),weighted_spectral_error_fraction=float(sw[8]/sw[0]),prior_controls=old['runs'],runs=[])
   for mode,correction in [('uncorrected',0.),('raw_rank8',raw),('weighted_rank8',weighted),('exact_coupling_oracle',delta)]:
    row=run(A,b,S,Q,base+correction);row['method']=mode;case['runs'].append(row);print(json.dumps(dict(dataset=old['dataset'],core=m,**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
