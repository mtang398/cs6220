"""Optimistic factor precision/capacity gate, not a feasible-memory implementation."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve
from threadpoolctl import threadpool_limits
import check_fused_congruence_core as fused
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def quantize(L,mode):
 m=len(L)
 if mode=='float32':return L.astype(np.float32).astype(float)
 if mode=='exact':return L.copy()
 Q=L.astype(np.float16).astype(float);Q[:,:8]=L[:,:8];np.fill_diagonal(Q,np.diag(L));return Q

def streamed(C):
 m=len(C);L=np.zeros_like(C)
 for j in range(m):
  v=C[j:,j]-L[j:,:j]@L[j,:j]
  if v[0]<=0:return None,dict(failed_pivot=j,value=float(v[0]))
  L[j,j]=np.sqrt(v[0]);v[1:]/=L[j,j]
  L[j+1:,j]=v[1:] if j<8 else v[1:].astype(np.float16).astype(float)
 return L,dict(failed_pivot=None)

def main():
 out=ROOT/'results/half-core-capacity.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time()
 files=[Path(__file__).resolve(),Path(fused.__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(shift=1e-5,factor_byte_allowance=721200,cores={'float32':600,'half':822},half='First8 columns and diagonal FP64; all other entries FP16. First quantize full FP64 factor, then test sequential quantized column construction.',scope='Optimistic dense gate only: dense construction, decoded factors, selection and per-step true residual checks free. Same-rank exact822 exceeds byte allowance and is only a capacity oracle. No native-memory or adaptive novelty claim.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));Sall=kernel.farthest(822);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),runs=[])
   for mode,m in [('float32',600),('half',822),('exact',822),('streamed_half',822)]:
    S=Sall[:m];C=A[np.ix_(S,S)];info={}
    if mode=='streamed_half':L,info=streamed(C)
    else:L=quantize(cholesky(C,lower=True),mode)
    if L is None:row=dict(method=mode,status='nonpositive_factor_pivot',converged=False,**info)
    else:
     H=lambda v:cho_solve((L,True),v,check_finite=False)
     row,_=fused.solve(A,b,S,H,1e-5);row.update(method=mode,core=m,factor_payload_bytes=4*m*(m+1)//2 if mode=='float32' else (m*m+55*m-216 if mode!='exact' else 8*m*(m+1)//2),**info)
    case['runs'].append(row);print(json.dumps(dict(dataset=name,**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
