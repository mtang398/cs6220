"""Compile and validate actual packed24 factor operations."""
import json,subprocess,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
from check_packed_int24_factor import Helper24,Streamed24,ROOT,pointer

def main():
 out=ROOT/'results/int24-native-primitives.json';lib=ROOT/'results/packed_int24_factor.so';source=ROOT/'scripts/packed_int24_factor.c'
 if out.exists() or lib.exists():raise FileExistsError('Preserve artifacts')
 resource.setrlimit(resource.RLIMIT_CPU,(50,55));t,cpu=perf_counter(),process_time()
 command=['gcc','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(source),'-lm','-o',str(lib)];subprocess.run(command,check=True,capture_output=True)
 helper=Helper24(lib);rng=np.random.default_rng(622099);checks=[]
 with threadpool_limits(limits=1):
  for name in ['signed','zero_scales']:
   m=17;V=rng.normal(size=(m,m));C=V@V.T+np.eye(m) if name=='signed' else np.eye(m)
   obj=object.__new__(Streamed24);obj.m=m;obj.low=np.empty(3*(m-8)*(m-9)//2,dtype=np.uint8);obj.wide=np.empty(8*m-28);obj.diag=np.empty(m);obj.scales=np.empty(m)
   for j in range(m):
    work=C[j:,j].copy();eps=np.empty(1);pivot=np.empty(1)
    assert helper.column(m,j,pointer(obj.low),pointer(obj.wide),pointer(obj.diag),pointer(obj.scales),pointer(work),pointer(eps),pointer(pivot))==0
   L=obj.dense_reference();q=rng.normal(size=m);v=q.copy()
   assert helper.solve(m,pointer(obj.low),pointer(obj.wide),pointer(obj.diag),pointer(obj.scales),pointer(v))==0
   ref=cho_solve((L,True),q);err=float(np.linalg.norm(v-ref)/np.linalg.norm(ref));assert err<1e-12
   tmp=np.empty(m);got=np.empty(m);assert helper.product(m,pointer(obj.low),pointer(obj.wide),pointer(obj.diag),pointer(obj.scales),pointer(q),pointer(tmp),pointer(got))==0
   pe=float(np.linalg.norm(got-L@(L.T@q))/np.linalg.norm(got));assert pe<1e-12
   if name=='signed':assert np.any(L[9:,8]<0)
   else:assert np.all(obj.scales[8:]==0)
   checks.append(dict(case=name,solve_relative_error=err,product_relative_error=pe))
  path=ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
  with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
  factor=Streamed24(X,b,bw,675,1e-5,helper);L=factor.dense_reference();q=rng.normal(size=675);v=factor.solve(q);ref=cho_solve((L,True),q);err=float(np.linalg.norm(v-ref)/np.linalg.norm(ref));assert err<1e-8
  got=factor.product(q);pe=float(np.linalg.norm(got-L@(L.T@q))/np.linalg.norm(got));assert pe<1e-12
  checks.append(dict(case='native_core675',solve_relative_error=err,product_relative_error=pe,factor_bytes=factor.factor_bytes,factor_seconds=factor.factor_seconds,minimum_raw_pivot=factor.minimum_raw_pivot))
 files=[Path(__file__).resolve(),source,lib,ROOT/'scripts/check_packed_int24_factor.py']
 rec=dict(status='complete',compile_command=command,sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},checks=checks,scope='Primitive correctness and native construction; dense references outside production path. No total-solver memory/runtime claim.',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu)
 out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
