"""Predeclared fixed-memory core/workspace tradeoff; static diagnostic only."""
import json, resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from threadpoolctl import threadpool_limits
from check_schur_inplace_frontier_v2 import solve, allocation_audit, ROOT

out=ROOT/'results/schur-workspace-tradeoff.json'
if out.exists(): raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(65,70))
start=perf_counter(); cpu=process_time()
record=dict(status='running',scope='Test whether a larger core trades away enough kernel workspace to reverse total-time gains. No adaptive method or novelty claim.',protocol=dict(counts=[320,336,344],repeats=3,regularizations=[.001,1e-5],datasets=['synthetic-n640-smooth','concrete-observed'],owned_array_budget=1048576,common_inputs='shared and separately reported',rtol=1e-8,blas_threads=1,cpu_cap=65,order='cyclic core-count order across repeats'),source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/check_schur_inplace_frontier_v2.py']},runs=[],audits=[])
def save():
 record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
 out.write_text(json.dumps(record,indent=2)+'\n')
with threadpool_limits(limits=1):
 for dataset in record['protocol']['datasets']:
  path=ROOT/f'data/prepared/{dataset}-s0-lambda0.001.npz'
  with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
  for lam in [.001,1e-5]:
   for rep in range(3):
    sizes=[320,336,344];sizes=sizes[rep:]+sizes[:rep]
    for m in sizes:
     if process_time()-cpu>58:record['status']='cpu_budget';save();raise SystemExit
     row=dict(dataset=dataset,regularization=lam,m=m,repeat=rep,data_sha256=sha256(path.read_bytes()).hexdigest())
     try:
      x,metrics=solve(X,b,bw,m,lam,cpu_cap=8.,deadline=cpu+60.)
      row.update(metrics);del x
     except (MemoryError,np.linalg.LinAlgError) as e:row.update(status='resource_or_numerical_failure',error=str(e))
     record['runs'].append(row);save()
   for m in [320,336,344]:
    try:
     a=allocation_audit(X,b,bw,m,lam);a.update(dataset=dataset,regularization=lam,m=m);record['audits'].append(a)
    except (MemoryError,np.linalg.LinAlgError) as e:record['audits'].append(dict(dataset=dataset,regularization=lam,m=m,error=str(e)))
    save()
record['status']='complete';save()
