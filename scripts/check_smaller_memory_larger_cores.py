"""Additional larger-core controls at256KiB; predeclared112/128, development only."""
import json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
from check_static_schur_scope import base,plain_cg,ROOT
out=ROOT/'results/smaller-memory-larger-core-controls.json'
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(112,118))
started=perf_counter();cpu=process_time()
record=dict(status='running',protocol=dict(budgets_and_counts={'262144':[112,128]},datasets=['smooth','mixed','Concrete'],regularizations=[.001,1e-5],seed=0,repeats=1,per_solve_cpu_cap=8.,total_cpu_cap=112,rtol=1e-8,maxiter=2500,common_inputs='X,b additional identical input bytes;64KiB reserve remains unchanged',interpretation='Initial multi-budget headroom screen; retain1MiB negative results. No adaptive method or all-scope validation claim.'),runs=[],audits=[],source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/check_static_schur_scope.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']})
def save():
 record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu)
 out.write_text(json.dumps(record,indent=2)+'\n')
with threadpool_limits(limits=1):
 for B,counts in [(262144,[112,128])]:
  base.LIMIT=B
  for name in ['smooth','mixed','Concrete']:
   stem=f'synthetic-n640-{name}' if name!='Concrete' else 'concrete-observed'
   path=ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   audited=set()
   for lam in [.001,1e-5]:
    for m in counts:
     if process_time()-cpu>103:record['status']='cpu_budget';save();raise SystemExit
     row=dict(dataset=name,n=len(b),budget=B,m=m,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest())
     assert not tracemalloc.is_tracing()
     try:
      if m:x,r=base.solve(X,b,bw,m,lam,cpu_cap=8.,deadline=cpu+105)
      else:x,r=plain_cg(X,b,bw,lam,cpu_cap=8.,deadline=cpu+105)
      row.update(r);del x
     except (MemoryError,np.linalg.LinAlgError) as error:row.update(status='resource_or_numerical_failure',error=str(error))
     record['runs'].append(row);save()
     if m not in audited and row.get('converged'):
      try:
       if m:a=base.allocation_audit(X,b,bw,m,lam)
       else:
        tracemalloc.start();x,r=plain_cg(X,b,bw,lam,maxiter=2);_,peak=tracemalloc.get_traced_memory();del x
        a=dict(tracemalloc_peak_bytes=peak)
       a.update(dataset=name,budget=B,m=m,within_budget=a['tracemalloc_peak_bytes']<=B)
       record['audits'].append(a)
      except (MemoryError,np.linalg.LinAlgError) as e:record['audits'].append(dict(dataset=name,budget=B,m=m,error=str(e)))
      finally:tracemalloc.stop()
      audited.add(m);save()
record['status']='complete';save()
