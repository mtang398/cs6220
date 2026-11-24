"""Same-matrix RHS dependence of two native static representations."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_first8_schur as native
from check_core_precision_allocation import Helper
from check_packed_float_allocation import RawPointers
ROOT=native.ROOT

def main():
 out=ROOT/'results/same-matrix-representation-frontier.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time()
 # Include all project Python sources to preserve transitive solver identity.
 sources=sorted((ROOT/'scripts').glob('*.py'))+[ROOT/'scripts/mixed_column_factor_first8.c',ROOT/'results/mixed_column_factor_first8.so',ROOT/'results/packed_float_factor.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},protocol=dict(question='Does preferred representation change with RHS for the same kernel matrix?',datasets=['smooth','random','mixed'],seed=0,n=2048,regularization=1e-5,repeats=2,methods=['original_exact416','FIRST8_H1'],order='Reverse method order on second repetition; rotate by target.',cost='Native complete solve including setup, original residual correction and verification. Existing original solvers and allocation policies unchanged. Specialized FIRST8 helper used.',scope='Development static portfolio opportunity check, no learned selector or success claim. Same matrix verified bitwise; reserved confirmation seeds untouched.'),runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 helper=Helper(ROOT/'results/mixed_column_factor_first8.so');raw=RawPointers(ROOT/'results/packed_float_factor.so');cases=[];matrix_identity=None
 for name in ['smooth','random','mixed']:
  path=ROOT/'data/prepared'/f'synthetic-n2560-{name}-s0-lambda0.001.npz'
  with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
  identity=(sha256(X.tobytes()).hexdigest(),bw)
  if matrix_identity is None:matrix_identity=identity
  assert identity==matrix_identity
  X.setflags(write=False);b.setflags(write=False);cases.append((name,path,X,b,bw))
 rec['matrix_identity']=matrix_identity;save()
 with threadpool_limits(limits=1):
  for repeat in range(2):
   for index,(name,path,X,b,bw) in enumerate(cases):
    order=['original_exact416','FIRST8_H1']
    if (index+repeat)%2:order.reverse()
    for method in order:
     if process_time()-cpu>71:rec['status']='cpu_budget';save();return
     if method=='original_exact416':answer,row=native.exact_control.run(X,b,bw,1e-5,method,raw,deadline=cpu+78)
     else:answer,row=native.solve(X,b,bw,1e-5,method,helper,raw,deadline=cpu+78)
     del answer
     row.update(dataset=name,repeat=repeat,method=method,data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(row);save()
     print(json.dumps({k:row.get(k) for k in ['dataset','repeat','method','status','converged','total_seconds','iterations','total_inner_iterations']}),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
