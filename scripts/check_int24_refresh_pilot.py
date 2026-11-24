"""Native uniform-FP32 exact-congruence representation comparison."""
import json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_int24_refresh as candidate
import check_native_int24_fused as exact24
import check_native_uniform_fused as uniform
import check_native_fused_residual_recurrence as control
ROOT=candidate.ROOT

def main():
 out=ROOT/'results/native-int24-refresh.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();raw=control.RawPointers(ROOT/'results/packed_float_factor.so');mixed=candidate.Helper24(ROOT/'results/packed_int24_factor.so')
 files=[Path(__file__).resolve(),Path(candidate.__file__).resolve(),Path(control.__file__).resolve(),Path(exact24.__file__).resolve(),Path(uniform.__file__).resolve(),ROOT/'scripts/check_native_first8_schur.py',ROOT/'scripts/check_native_adaptive_fused.py',ROOT/'scripts/check_packed_int24_factor.py',ROOT/'scripts/packed_int24_factor.c',ROOT/'results/packed_int24_factor.so',ROOT/'scripts/check_compressed_core_schur.py',ROOT/'scripts/check_packed_float_factor.py',ROOT/'scripts/check_packed_float_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/packed_float_factor.so',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(seed=0,repeats=2,regularization=1e-5,candidate='Packed signed24bit core675; column FP64 scales, first8columns/diagonal FP64. Sequential streamed factorization. Approximate diagonal congruence actions with accumulated gap correction; adaptive .1 residual trigger or periodic8, continued heuristic directions. Exactint24 controls included.',controls=['uniform FP32core600 exactcongruence','original exact416'],order='Rotatebycase, reverse onsecondrepeat.',scope='Static representation optimization, not adaptive novelty. Complete native cost and original1e-8 acceptance; newmemoryaudit.'),audits=[],runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
 with threadpool_limits(limits=1):
  for name,fn in [cases[0],cases[3]]:
   with np.load(ROOT/'data/prepared'/fn) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   tracemalloc.start()
   try:x,row=candidate.solve(X,b,bw,mixed,cpu_cap=8);_,peak=tracemalloc.get_traced_memory();del x
   finally:tracemalloc.stop()
   rec['audits'].append(dict(dataset=name,traced_peak_bytes=peak,metrics=row));save()
   if peak>candidate.LIMIT or row['status']=='memory_failure':rec['status']='allocation_gate_failed';save();return
  for repeat in range(2):
   for index,(name,fn) in enumerate(cases):
    path=ROOT/'data/prepared'/fn
    with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
    methods=['adaptive','periodic8','int24','uniform600','exact416'];methods=methods[index%5:]+methods[:index%5]
    if repeat:methods.reverse()
    for method in methods:
     if process_time()-cpu>71:rec['status']='cpu_budget';save();return
     if method in ['adaptive','periodic8']:x,row=candidate.solve(X,b,bw,mixed,mode=method)
     elif method=='int24':x,row=exact24.solve(X,b,bw,mixed)
     elif method=='uniform600':x,row=uniform.solve(X,b,bw,raw)
     else:x,row=control.scalar.exact_control.run(X,b,bw,1e-5,'original_exact416',raw,deadline=cpu+78)
     del x;row.update(dataset=name,repeat=repeat,method=method,data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(row);save();print(json.dumps({k:row.get(k) for k in ['dataset','repeat','method','status','iterations','total_seconds','refresh_steps']}),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
