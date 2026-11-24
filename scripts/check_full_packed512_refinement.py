"""Original n512 scope: streamed full FP32 factor and FP64 residual refinement."""
import json,resource,tracemalloc,gc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
from check_packed_float_factor import PackedFloat,refine,ROOT,LIMIT
from check_packed_float_allocation import RawPointers
import check_kernel_accounting_hoist as hoist
import check_refresh_regularization_scope as scope

def sha(p):return sha256(p.read_bytes()).hexdigest()
def run(X,b,bw,lam,helper):
 start=perf_counter();cpu=process_time();factor=PackedFloat(X,b,bw,len(b),lam,helper);rhs=b[factor.indices];arrays,info=refine(factor,rhs,rtol=1e-8)
 factor.budget.live=list(arrays);x=np.empty_like(b);x[factor.indices]=arrays[2];factor.budget.live=list(arrays)+[x];factor.budget.observe('full_original_coordinate_reconstruction',x)
 residual=factor.kernel.full(x,extra=(x,));factor.budget.observe('full_original_coordinate_residual',residual,temporary_allowance=x.nbytes);residual+=lam*x;residual-=b;relative=float(np.linalg.norm(residual)/np.linalg.norm(b));factor.budget.live=[]
 return dict(converged=bool(info['converged'] and relative<=1e-8),original_relative_residual=relative,refinement=info,setup_seconds=factor.setup_seconds,total_seconds=perf_counter()-start,total_cpu_seconds=process_time()-cpu,m=len(b),regularization=lam,kernel_stats=dict(factor.kernel.stats),factor_solve_calls=factor.solve_calls,refinement_products=factor.products,memory=factor.budget.report(),permutation_is_complete=bool(np.array_equal(np.sort(factor.indices),np.arange(len(b)))))
def main():
 out=ROOT/'results/full-packed512-refinement.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(50,55));t=perf_counter();cpu=process_time();helper=RawPointers(ROOT/'results/packed_float_factor.so')
 files=scope.source_paths()+[Path(__file__).resolve()];result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol='Nine original n512 seed0 cases smooth/random/mixed ×lambda .1/.001/1e-5. Existing PackedFloat and RawPointers, m=n; existing refine target1e-8. Original full FPS order/setup retained and charged despite avoidable all-point selection cost. Common accounting hoist all runs. Two setup-inclusive timing repeats plus separate whole-trace audit percase. Final streamed full original-coordinate verification, readonly input storage excluded consistently. Baseline engineering only, no optimality or scientific goal success claim. No recompilation, dense kernel, or heldout data.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1),hoist.variant('hoisted'):
  for name in ['smooth','random','mixed']:
   path=ROOT/f'data/prepared/synthetic-n640-{name}-s0-lambda0.001.npz'
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   assert len(b)==512;X.setflags(write=False);b.setflags(write=False);hashes=dict(X=sha256(X.tobytes()).hexdigest(),b=sha256(b.tobytes()).hexdigest())
   for lam in [.1,.001,1e-5]:
    case=dict(dataset=name,data_sha256=sha(path),input_hashes=hashes,regularization=lam,audit=None,runs=[]);result['cases'].append(case);gc.collect();tracemalloc.start()
    try:row=run(X,b,bw,lam,helper);_,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    case['audit']=dict(traced_peak_bytes=peak,within_1MiB=peak<=LIMIT,run=row);save();assert peak<=LIMIT and row['memory']['peak_with_reserve_bytes']<=LIMIT and row['converged'] and row['permutation_is_complete']
    for repeat in range(2):
     row=run(X,b,bw,lam,helper);row['repeat']=repeat;case['runs'].append(row);save();assert row['converged'] and row['permutation_is_complete'] and row['memory']['peak_with_reserve_bytes']<=LIMIT
    first,second=case['runs'];checks={k:first[k]==second[k] for k in ['kernel_stats','factor_solve_calls','refinement_products','original_relative_residual','memory']};case['repeat_work_identity']=checks;assert all(checks.values());print(name,lam,first['refinement']['corrections'],first['original_relative_residual'],peak,flush=True);save()
   assert hashes==dict(X=sha256(X.tobytes()).hexdigest(),b=sha256(b.tobytes()).hexdigest())
 result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
