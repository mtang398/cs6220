"""Counterbalanced repetitions of unchanged native fused congruence pilot."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_fused_congruence as fused
ROOT=fused.ROOT

def main():
 out=ROOT/'results/native-fused-congruence-repeats.json'
 if out.exists():raise FileExistsError(out)
 prior_path=ROOT/'results/native-fused-congruence.json';prior=json.loads(prior_path.read_text());assert prior['status']=='complete'
 for f,h in prior['sources'].items():assert sha256((ROOT/f).read_bytes()).hexdigest()==h
 assert all(a['traced_peak_bytes']<=fused.LIMIT and a['metrics']['converged'] for a in prior['audits'])
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();helper=fused.Helper(ROOT/'results/mixed_column_factor_first8.so');raw=fused.RawPointers(ROOT/'results/packed_float_factor.so')
 rec=dict(status='running',sources={**prior['sources'],str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__).read_bytes()).hexdigest(),str(prior_path.relative_to(ROOT)):sha256(prior_path.read_bytes()).hexdigest()},protocol=dict(repeats=2,seed=0,regularization=1e-5,order='Reverse pilot rotated order on repeat1, cyclically shift that order on repeat2.',scope='Unchanged native solvers; allocation audit reused. Complete cost. Timing repeats not independent data seeds; no adaptive success claim.'),runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
 with threadpool_limits(limits=1):
  for repeat in [1,2]:
   for index,(name,fn) in enumerate(cases):
    path=ROOT/'data/prepared'/fn
    with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
    order=['fused_H0','original_exact416','FIRST8_H1' if index<2 else 'FIRST8_H2prefix4_then_H1'];order=order[index%3:]+order[:index%3];order.reverse()
    if repeat==2:order=order[1:]+order[:1]
    for method in order:
     if process_time()-cpu>71:rec['status']='cpu_budget';save();return
     if method=='fused_H0':answer,row=fused.solve(X,b,bw,helper)
     elif method=='original_exact416':answer,row=fused.scalar.exact_control.run(X,b,bw,1e-5,method,raw,deadline=cpu+78)
     else:answer,row=fused.scalar.solve(X,b,bw,1e-5,method,helper,raw,deadline=cpu+78)
     del answer;row.update(dataset=name,repeat=repeat,method=method,data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(row);save();print(json.dumps({k:row.get(k) for k in ['dataset','repeat','method','status','total_seconds']}),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
