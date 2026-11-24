"""Memory and complete-cost gate for recursive preconditioned top residual."""
import json,tracemalloc,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_adaptive_fused as new
ROOT=new.ROOT

def main():
 out=ROOT/'results/native-adaptive-fused.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();helper=new.Helper(ROOT/'results/mixed_column_factor_first8.so')
 files=[Path(__file__).resolve(),Path(new.__file__).resolve(),ROOT/'scripts/check_factor_product_cost.py',ROOT/'scripts/first8_factor_product.c',ROOT/'results/first8_factor_product.so',ROOT/'scripts/check_native_first8_schur.py',ROOT/'scripts/check_compressed_core_schur.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(repeats=2,seed=0,core=592,shift=1e-5,update='Current-direction core omission; streamed defect estimate; cumulative truncation allowance .25e-8||b||. Cheap trials always charged; actual alpha rechecked.',controls='Repaired fresh-Hr recurrence, identical stop and original acceptance. Alternate method order by case and repetition.',scope='Native adaptive vs exact fused gate with full costs. Original 1e-8 acceptance. No broad success claim; exact416 comparison follows only if mechanism improves.'),audits=[],runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
 with threadpool_limits(limits=1):
  # An actual early cap must return the current iterate and its current residual.
  with np.load(ROOT/'data/prepared'/cases[3][1]) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
  x,row=new.solve(X,b,bw,helper,cpu_cap=0,adaptive=True)
  assert x is not None and np.linalg.norm(x)==0 and row['status']=='cpu_cap' and row['original_relative_residual']==1
  rec['terminal_check']=row;del x
  for name,fn in [cases[0],cases[3]]:
   with np.load(ROOT/'data/prepared'/fn) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   tracemalloc.start()
   try:x,row=new.solve(X,b,bw,helper,cpu_cap=8,adaptive=True);_,peak=tracemalloc.get_traced_memory();del x
   finally:tracemalloc.stop()
   rec['audits'].append(dict(dataset=name,traced_peak_bytes=peak,metrics=row));save()
   if peak>new.LIMIT or row['status']=='memory_failure':rec['status']='allocation_gate_failed';save();return
  for repeat in range(2):
   for index,(name,fn) in enumerate(cases):
    path=ROOT/'data/prepared'/fn
    with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
    order=[False,True] if (repeat+index)%2==0 else [True,False]
    for adaptive in order:
     if process_time()-cpu>70:rec['status']='cpu_budget';save();return
     x,row=new.solve(X,b,bw,helper,adaptive=adaptive);del x
     row.update(dataset=name,repeat=repeat,data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(row);save();print(json.dumps({k:row.get(k) for k in ['dataset','repeat','method','status','iterations','total_seconds','omitted','rejected','original_relative_residual']}),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
