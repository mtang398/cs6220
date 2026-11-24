"""Native uniform-FP32 exact-congruence representation comparison."""
import json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_uniform_fused as candidate
import check_native_fused_residual_recurrence as control
ROOT=candidate.ROOT

def main():
 out=ROOT/'results/native-uniform-fused.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();raw=control.RawPointers(ROOT/'results/packed_float_factor.so');mixed=control.Helper(ROOT/'results/mixed_column_factor_first8.so')
 files=[Path(__file__).resolve(),Path(candidate.__file__).resolve(),Path(control.__file__).resolve(),ROOT/'scripts/check_compressed_core_schur.py',ROOT/'scripts/check_packed_float_factor.py',ROOT/'scripts/check_packed_float_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/packed_float_factor.so',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(seed=0,repeats=2,regularization=1e-5,candidate='UniformFP32 core600; exactcongruence and freshHr. No core iterative refinement or promoted columns.',controls=['FIRST8 core592 exactcongruence','original exact416'],order='Rotatebycase, reverse onsecondrepeat.',scope='Static representation optimization, not adaptive novelty. Complete native cost and original1e-8 acceptance; newmemoryaudit.'),audits=[],runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
 with threadpool_limits(limits=1):
  for name,fn in [cases[0],cases[3]]:
   with np.load(ROOT/'data/prepared'/fn) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   tracemalloc.start()
   try:x,row=candidate.solve(X,b,bw,raw,cpu_cap=8);_,peak=tracemalloc.get_traced_memory();del x
   finally:tracemalloc.stop()
   rec['audits'].append(dict(dataset=name,traced_peak_bytes=peak,metrics=row));save()
   if peak>candidate.LIMIT or row['status']=='memory_failure':rec['status']='allocation_gate_failed';save();return
  for repeat in range(2):
   for index,(name,fn) in enumerate(cases):
    path=ROOT/'data/prepared'/fn
    with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
    methods=['uniform600','FIRST8','exact416'];methods=methods[index%3:]+methods[:index%3]
    if repeat:methods.reverse()
    for method in methods:
     if process_time()-cpu>71:rec['status']='cpu_budget';save();return
     if method=='uniform600':x,row=candidate.solve(X,b,bw,raw)
     elif method=='FIRST8':x,row=control.solve(X,b,bw,mixed)
     else:x,row=control.scalar.exact_control.run(X,b,bw,1e-5,'original_exact416',raw,deadline=cpu+78)
     del x;row.update(dataset=name,repeat=repeat,method=method,data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(row);save();print(json.dumps({k:row.get(k) for k in ['dataset','repeat','method','status','iterations','total_seconds']}),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
