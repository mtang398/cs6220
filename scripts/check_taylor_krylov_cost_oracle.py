"""Exact-trajectory replay with free future knowledge; not an inexact solver."""
import ctypes,json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.special import gammainc
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_taylor_kernel_cost import Taylor,ROOT
from check_kernel_precision_cost import Product

def main():
 out=ROOT/'results/taylor-krylov-cost-oracle.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(110,115));t,c=perf_counter(),process_time()
 lib=ctypes.CDLL(str(ROOT/'results/taylor_kernel_tile.so'));fun=lib.taylor_tile
 arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fun.argtypes=[arr,arr,arr,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int];fun.restype=None
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(rank=64,rtol=1e-8,degree_cap=40,repeats=7,trajectory='Exact Gaussian PCG with fixed FPS Nystrom preconditioner',selection='Free exact future iteration count and exact alpha; equal per-step allocation of half final residual tolerance',static='Least common degree whose summed exact-trajectory action-error bounds fit same gap allowance',timings='Standalone matched128KiB tiles; weighted by trajectory degree counts; setup, degree selection and all preconditioning free',scope='Optimistic action-cost replay; no inexact convergence, total-memory or end-to-end claim'),runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n')
 cases=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),('mixed512','synthetic-n640-mixed-s0-lambda0.001.npz'),('Concrete824','concrete-observed-s0-lambda0.001.npz')]
 with threadpool_limits(limits=1):
  for ds,stem in cases:
   path=ROOT/'data/prepared'/stem
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   n=len(b);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));s=np.einsum('ij,ij->i',X,X)/(bw*bw)
   roots=np.sqrt(gammainc(np.arange(41)[:,None]+1,s[None,:]))
   selected=[0];distance=2-2*K[:,0]
   for _ in range(63):
    j=int(np.argmax(distance));selected.append(j);distance=np.minimum(distance,2-2*K[:,j])
   C=K[:,selected];W=K[np.ix_(selected,selected)];W.flat[::65]+=1e-12
   N=C@cho_solve(cho_factor(W,lower=True),C.T);val,Q=eigh(N,subset_by_index=[n-64,n-1]);val=np.maximum(val,0)
   for lam in [.001,1e-5]:
    A=K+lam*np.eye(n);nb=np.linalg.norm(b)
    def H(v):return (v-Q@((val/(val+lam))*(Q.T@v)))/lam
    x=np.zeros(n);r=b.copy();z=H(r);p=z.copy();rho=r@z;bounds=[];alphas=[]
    for it in range(2000):
     ap=A@p;alpha=rho/(p@ap)
     inner=roots@np.abs(p);w=roots*(inner[:,None]-roots*np.abs(p)[None,:]);bounds.append(np.linalg.norm(w,axis=1)*abs(alpha));alphas.append(float(alpha))
     x+=alpha*p;r-=alpha*ap
     if np.linalg.norm(b-A@x)<=1e-8*nb:break
     z=H(r);newrho=r@z;p=z+(newrho/rho)*p;rho=newrho
    else:raise RuntimeError('exact trajectory failed')
    bounds=np.asarray(bounds);Nstep=len(bounds);allowance=.5e-8*nb
    ok=bounds<=allowance/Nstep
    if not np.all(np.any(ok,axis=1)):raise RuntimeError('degree cap')
    degrees=np.argmax(ok,axis=1);static=int(np.flatnonzero(bounds.sum(axis=0)<=allowance)[0]);modes=['cdist64','gram64']+[f'taylor{j}' for j in sorted(set(degrees.tolist()+[static]))]
    ops={m:Product(X,b,bw,m) if m in ['cdist64','gram64'] else Taylor(X,b,bw,int(m[6:]),True,fun) for m in modes}
    for op in ops.values():op.apply()
    times={m:[] for m in modes}
    for rep in range(7):
     offset=rep%len(modes)
     for m in modes[offset:]+modes[:offset]:
      now=perf_counter();ops[m].apply();times[m].append(perf_counter()-now)
    med={m:float(np.median(v)) for m,v in times.items()};base=min(med['cdist64'],med['gram64'])*Nstep;adapt=sum(med[f'taylor{p}'] for p in degrees);fixed=med[f'taylor{static}']*Nstep
    rec['runs'].append(dict(dataset=ds,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest(),exact_iterations=Nstep,true_relative_residual=float(np.linalg.norm(b-A@x)/nb),adaptive_degrees=degrees.tolist(),static_degree=static,adaptive_bound_sum=float(sum(bounds[i,p] for i,p in enumerate(degrees))),gap_allowance=float(allowance),static_bound_sum=float(bounds[:,static].sum()),seconds=times,median_action_seconds=med,weighted_exact_action_seconds=base,weighted_adaptive_action_seconds=adapt,weighted_static_action_seconds=fixed,adaptive_speedup_over_exact=base/adapt,adaptive_speedup_over_static=fixed/adapt))
    save()
 rec['status']='complete';save();print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
