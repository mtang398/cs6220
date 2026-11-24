"""Frozen packed-race pilot, same-representation controls, no pruning."""
import json,resource
from pathlib import Path
from time import perf_counter,process_time
from hashlib import sha256
import numpy as np
from threadpoolctl import threadpool_limits
from check_packed_shared_schur_race import run_packed,sources
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/packed-shared-race-pilot.json'
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(112,118))
start=perf_counter();cpu=process_time()
record=dict(status='running',protocol=dict(datasets=['smooth','mixed','Concrete'],regularizations=[.001,1e-5],maximum_sizes=[128,160],modes=['shared_schur','race','shared_plain'],budget_bytes=262144,repeats=1,seed=0,rtol=1e-8,per_run_cpu_cap=8.,total_cpu_cap=112,order='Cyclic mode order across six cases; both maximum sizes included',interpretation='Complete-cost pilot; preserve dense-race and1MiB negatives; no confirmation claim'),source_sha256=sources(),runs=[])
record['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__).read_bytes()).hexdigest()
def save():
 record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
 out.write_text(json.dumps(record,indent=2)+'\n')
with threadpool_limits(limits=1):
 case=0
 for name in record['protocol']['datasets']:
  stem=f'synthetic-n640-{name}' if name!='Concrete' else 'concrete-observed'
  path=ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
  with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
  for lam in [.001,1e-5]:
   for m in [128,160]:
    order=['shared_schur','race','shared_plain'];shift=case%3;order=order[shift:]+order[:shift]
    for mode in order:
     if process_time()-cpu>103:record['status']='cpu_budget';save();raise SystemExit
     try:
      row,x=run_packed(X,b,bw,lam,mode,maximum=m,budget_bytes=262144,per_run_cpu=8.,deadline=cpu+105)
      del x
     except (MemoryError,np.linalg.LinAlgError) as e:row=dict(mode=mode,status='resource_or_numerical_failure',error=str(e))
     row.update(dataset=name,regularization=lam,maximum=m,data_sha256=sha256(path.read_bytes()).hexdigest());record['runs'].append(row);save()
   case+=1
record['status']='complete';save()
