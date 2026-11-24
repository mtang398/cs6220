"""Whole-phase displacement repair; optimistic dense gate with fixed refresh triggers."""
import inspect,json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_core_direction_repair as old
ROOT=old.ROOT
TEXT=inspect.getsource(old.run)
MODIFIED=TEXT.replace("refresh=[];identities=[];status='maxiter'","refresh=[];identities=[];phase=np.zeros(n);r_start=r.copy();status='maxiter'",1)
MODIFIED=MODIFIED.replace('alpha=rho/den;y+=alpha*p;r-=alpha*ap;s+=alpha*q','alpha=rho/den;y+=alpha*p;r-=alpha*ap;s+=alpha*q\n  if mode.endswith(\'phase\'):phase+=alpha*p',1)
BEFORE="   if mode.endswith('repair'):"
AFTER="""   if mode.endswith('phase'):
    gd=r_start-r;curvature=float(phase@gd)
    if not np.isfinite(curvature) or curvature<=0:
     identities.append(dict(guard='nonpositive_or_nonfinite_phase_curvature',curvature=curvature if np.isfinite(curvature) else None));status='invalid_phase_curvature';break
    gamma=float(phase@r)/curvature
    if not np.isfinite(gamma):status='nonfinite_phase_gamma';break
    before_norm=float(np.linalg.norm(r));displacement_norm=float(np.linalg.norm(phase));y+=gamma*phase;r-=gamma*gd
    z=np.r_[H(r[:m]),r[m:]/1e-5];beta=-float(gd@z)/curvature;direction=z+beta*phase
    identities.append(dict(guard='passed',gamma=gamma,curvature=curvature,correction_norm=abs(gamma)*displacement_norm,residual_norm_before=before_norm,residual_norm_after=float(np.linalg.norm(r)),residual_orthogonality=float(abs(phase@r)/max(np.linalg.norm(phase)*np.linalg.norm(r),1e-300)),phase_conjugacy=float(abs(gd@direction)/max(np.linalg.norm(gd)*np.linalg.norm(direction),1e-300))))
    p=direction;rho=float(r@z);repaired=True;phase.fill(0);r_start=r.copy()
   if mode.endswith('repair'):"""
assert MODIFIED.count(BEFORE)==1;MODIFIED=MODIFIED.replace(BEFORE,AFTER,1)
PRIVATE=dict(old.run.__globals__);exec(compile(MODIFIED,'<whole-phase-direction-repair>','exec'),PRIVATE);run=PRIVATE['run']

def tiny():
 rng=np.random.default_rng(91);n=13;F=rng.normal(size=(n,n));G=F@F.T+np.eye(n);r0=rng.normal(size=n);d=rng.normal(size=n);r=r0-G@d;gd=r0-r
 assert np.allclose(gd,G@d,rtol=1e-13,atol=1e-13)
 curvature=float(d@gd);gamma=float(d@r)/curvature;r-=gamma*gd;z=r.copy();p=z-d*float(gd@z)/curvature
 checks=dict(image_relative_error=float(np.linalg.norm(gd-G@d)/np.linalg.norm(G@d)),line_orthogonality=float(abs(d@r)/(np.linalg.norm(d)*np.linalg.norm(r))),projected_conjugacy=float(abs(gd@p)/(np.linalg.norm(gd)*np.linalg.norm(p))))
 assert max(checks.values())<1e-12;return checks

def main():
 out=ROOT/'results/core-phase-direction-repair.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
 priorpath=ROOT/'results/core-direction-repair.json';prior=json.loads(priorpath.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items())
 files=[Path(__file__).resolve(),Path(old.__file__).resolve(),priorpath]+[ROOT/p for p in prior['sources']]
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(mechanism='At existing refresh, gd=r_start-r after exact Es correction, repair along whole phase displacement and conjugate next direction against gd. Reset phase and residual start after correction.',cost='Same optimistic analytical kernel-entry accounting as prior gate. Two extra n-vectors free; dense construction, reference stopping, history and additional arithmetic free. Fixed-memory native feasibility unproven.',controls='Unchanged FR, last-direction repair, whole-phase repair with periodic8 and terminal triggers. No threshold search. CPU4 per solve,80/85 process,120wall/flock,BLAS1.'),tiny=tiny(),cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save();helper=old.Helper(ROOT/'results/mixed_column_factor_first8.so')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn;previous=next(c for c in prior['cases'] if c['dataset']==name);assert sha256(path.read_bytes()).hexdigest()==previous['data_sha256']
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   kernel=old.SharedKernel(X,bw,old.Budget([X,b]));S=kernel.farthest(592);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;factor=old.MixedFactor(A[np.ix_(S,S)],range(8),helper)
   case=dict(dataset=name,data_sha256=previous['data_sha256'],runs=[],replay_checks=[]);result['cases'].append(case)
   for mode in ['periodic_fr','periodic_repair','periodic_phase','terminal_fr','terminal_repair','terminal_phase']:
    row=run(A,b,S,factor,mode);case['runs'].append(row)
    controls=[r for r in previous['runs'] if r['method']==mode]
    if controls:
     expected=controls[0]
     if expected['status']=='converged' and row['status']=='converged':
      checks={k:row[k]==expected[k] for k in ['iterations','original_relative_residual','refresh_steps','analytical_entries']};case['replay_checks'].append(dict(method=mode,checks=checks));assert all(checks.values())
     else:case['replay_checks'].append(dict(method=mode,note='CPU-capped trajectories have machine-dependent lengths; no exact completed-work comparison.',prior_status=expected['status'],status=row['status']))
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    print(name,mode,row['status'],row['iterations'],row['analytical_entries'],flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
