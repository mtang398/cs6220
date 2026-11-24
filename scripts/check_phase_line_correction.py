"""Phase line correction followed by inherited FR: causal diagnostic, not exact CG."""
import inspect,json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_phase_direction_repair as phase
old=phase.old;ROOT=old.ROOT
START="    z=np.r_[H(r[:m]),r[m:]/1e-5];beta=-float(gd@z)/curvature;direction=z+beta*phase"
END="    p=direction;rho=float(r@z);repaired=True;phase.fill(0);r_start=r.copy()"
text=phase.MODIFIED;lo=text.index(START);hi=text.index(END,lo)+len(END)
replacement="""    identities.append(dict(guard='passed',gamma=gamma,curvature=curvature,correction_norm=abs(gamma)*displacement_norm,residual_norm_before=before_norm,residual_norm_after=float(np.linalg.norm(r)),residual_orthogonality=float(abs(phase@r)/max(np.linalg.norm(phase)*np.linalg.norm(r),1e-300))))
    phase.fill(0);r_start=r.copy()"""
MODIFIED=text[:lo]+replacement+text[hi:];namespace=dict(old.run.__globals__);exec(compile(MODIFIED,'<phase-line-only-inherited-fr>','exec'),namespace);run=namespace['run']

def main():
 out=ROOT/'results/core-phase-line-correction.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time()
 priorpath=ROOT/'results/core-phase-direction-repair.json';prior=json.loads(priorpath.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items())
 files=[Path(__file__).resolve(),priorpath]+[ROOT/p for p in prior['sources']]
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol='Periodic8 only. Phase line correction changes y,r but preserves inherited p and old rho for ordinary FR update. No projection-only run or exact-CG claim. Same optimistic query counts, two additional n-vectors free, all reference work free; native feasibility unproven.',tiny=phase.tiny(),cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save();helper=old.Helper(ROOT/'results/mixed_column_factor_first8.so')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn;previous=next(c for c in prior['cases'] if c['dataset']==name);assert sha256(path.read_bytes()).hexdigest()==previous['data_sha256']
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   kernel=old.SharedKernel(X,bw,old.Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;factor=old.MixedFactor(A[np.ix_(S,S)],range(8),helper)
   row=run(A,b,S,factor,'periodic_phase');row['method']='periodic_phase_line_only'
   controls=[r for r in previous['runs'] if r['method'].startswith('periodic')]
   result['cases'].append(dict(dataset=name,data_sha256=previous['data_sha256'],run=row,controls=controls))
   if row['status']=='converged':assert row['original_relative_residual']<=1e-8
   print(name,row['status'],row['iterations'],row['analytical_entries'],flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
