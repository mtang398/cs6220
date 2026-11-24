"""Full packed-upper static controls, including construction and certification."""
import json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_packed_schur_feasibility import PackedSchur,base,ROOT
class UpperPacked(PackedSchur):
 def __init__(self,kernel,m,lam,*,indices=None):
  selected=kernel.farthest(m) if indices is None else indices
  super().__init__(kernel,m,lam,indices=selected,lower=False)
base.InplaceSchur=UpperPacked
out=ROOT/'results/packed-schur-solves.json'
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(112,118))
start=perf_counter();cpu=process_time()
record=dict(status='running',protocol=dict(budgets_and_counts={'262144':[96,128,160],'1048576':[320,448,480]},datasets=['smooth','mixed','Concrete'],regularizations=[.001,1e-5],repeats=1,seed=0,rtol=1e-8,per_solve_cpu_cap=8.,total_cpu_cap=112,common_inputs='SharedX,b additional; unchanged64KiB reserve',scope='Static representation controls only; no adaptive benefit claim'),source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/check_packed_schur_feasibility.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']},runs=[],audits=[])
def save():
 record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
 out.write_text(json.dumps(record,indent=2)+'\n')
with threadpool_limits(limits=1):
 rng=np.random.default_rng(622104);X=rng.normal(size=(27,4));b=rng.normal(size=27)
 A=np.exp(-cdist(X,X,'sqeuclidean')/2)+.02*np.eye(27)
 x,r=base.solve(X,b,1.,12,.02,rtol=1e-11,maxiter=200)
 err=float(np.linalg.norm(x-np.linalg.solve(A,b))/np.linalg.norm(np.linalg.solve(A,b)))
 assert r['converged'] and err<1e-9
 record['independent_full_solve_error']=err;save()
 for B,counts in [(262144,[96,128,160]),(1048576,[320,448,480])]:
  base.LIMIT=B
  for name in ['smooth','mixed','Concrete']:
   stem=f'synthetic-n640-{name}' if name!='Concrete' else 'concrete-observed'
   path=ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   audited=set()
   for lam in [.001,1e-5]:
    for m in counts:
     if process_time()-cpu>103:record['status']='cpu_budget';save();raise SystemExit
     row=dict(dataset=name,budget=B,m=m,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest())
     assert not tracemalloc.is_tracing()
     try:
      x,r=base.solve(X,b,bw,m,lam,cpu_cap=8.,deadline=cpu+105);row.update(r);del x
     except (MemoryError,np.linalg.LinAlgError) as e:row.update(status='resource_or_numerical_failure',error=str(e))
     record['runs'].append(row);save()
     if m not in audited and row.get('converged'):
      try:
       a=base.allocation_audit(X,b,bw,m,lam);a.update(dataset=name,budget=B,m=m,within_budget=a['tracemalloc_peak_bytes']<=B);record['audits'].append(a)
      except (MemoryError,np.linalg.LinAlgError) as e:record['audits'].append(dict(dataset=name,budget=B,m=m,error=str(e)))
      finally:tracemalloc.stop()
      audited.add(m);save()
record['status']='complete';save()
