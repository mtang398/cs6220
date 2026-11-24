"""Dense core-RHS replay: can learned compression-error secants save refinement?"""
import json,resource
from pathlib import Path
from time import perf_counter,process_time
from hashlib import sha256
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from check_packed_float_factor import PackedFloat,ROOT
from check_packed_float_allocation import RawPointers

def main():
 out=ROOT/'results/core-error-feedback.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60));t,c=perf_counter(),process_time();helper=RawPointers(ROOT/'results/packed_float_factor.so')
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(core=600,outer_directions=16,rtol=1e-9,max_corrections=8,methods=['base','initial_rhs_pair','random_pair','rolling_current_pair'],scope='Dense exact outer trajectory and core products; pair setup reference inverse free. Fixed core size, extra pair memory NOT matched. Counts only, no native/adaptive goalclaim.',pair='s=H0*t−refined_solution; y=final_residual−initial_base_residual; retain s,w=H0*y andBFGScoefficients'),runs=[])
 with threadpool_limits(limits=1):
  path=ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
  with np.load(path) as a:X,b,bw=a['X_train'],a['y_train'],float(a['bandwidth'])
  for lam in [.001,1e-5]:
   f=PackedFloat(X,b,bw,600,lam,helper);J=f.indices;R=np.setdiff1d(np.arange(len(b)),J);K=f.kernel.block(X,X);C=K[np.ix_(J,J)]+lam*np.eye(len(J));B=K[np.ix_(J,R)];cf=cho_factor(C,lower=True);D=K[np.ix_(R,R)]+lam*np.eye(len(R))-B.T@cho_solve(cf,B)
   rhs=b[R]-B.T@cho_solve(cf,b[J]);r=rhs.copy();p=r.copy();rho=float(r@r);rhs_sequence=[]
   for it in range(16):
    rhs_sequence.append(B@p);ap=D@p;alpha=rho/float(p@ap);r-=alpha*ap;new=float(r@r);p=r+(new/rho)*p;rho=new
   def pair(s,y):
    den=float(s@y)
    if den<=0 or not np.isfinite(den):return None
    w=f.solve(y);inv=1/den;gamma=inv+inv*inv*float(y@w)
    return s.copy(),w,inv,gamma
   def make_static(v):
    z0=f.solve(v);ref=cho_solve(cf,v);s=z0-ref;return pair(s,C@s)
   static={'initial_rhs_pair':make_static(b[J]),'random_pair':make_static(np.random.default_rng(622107).normal(size=600))}
   for mode in ['base','initial_rhs_pair','random_pair','rolling_current_pair']:
    correction=static.get(mode);rows=[];rejects=0
    for number,target in enumerate(rhs_sequence):
     targetnorm=np.linalg.norm(target);initial=f.solve(target);base_residual=target-C@initial
     def apply(v,base=None):
      z=f.solve(v) if base is None else base.copy()
      if correction is not None:
       s,w,inv,gamma=correction;sv=float(s@v);wv=float(w@v);z+=s*(gamma*sv-inv*wv)-w*(inv*sv)
      return z
     z=apply(target,initial);history=[]
     for step in range(9):
      residual=target-C@z;rel=float(np.linalg.norm(residual)/targetnorm);history.append(rel)
      if rel<=1e-9 or step==8:break
      z+=apply(residual)
     rows.append(dict(rhs_index=number,core_products=len(history),extra_base_residual_product=1,converged=history[-1]<=1e-9,residual_history=history))
     if mode=='rolling_current_pair':
      newpair=pair(initial-z,residual-base_residual)
      if newpair is None:rejects+=1
      else:correction=newpair
    rec['runs'].append(dict(regularization=lam,method=mode,rhs_runs=rows,core_products=sum(x['core_products'] for x in rows),extra_base_residual_products=len(rows) if mode=='rolling_current_pair' else 0,rejected_updates=rejects,converged_count=sum(x['converged'] for x in rows)))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
