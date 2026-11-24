"""Degree-three global feature action error on exact congruence directions."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
import check_kernel_accounting_hoist as hoist
from check_localized_gaussian_moments import polynomial
ROOT=scope.ROOT

def sha(p):return sha256(p.read_bytes()).hexdigest()
def tiny():
 rng=np.random.default_rng(42);n,m=9,4;F=rng.normal(size=(n,n));A=F@F.T+np.eye(n);B=A[:m,m:];H=np.linalg.inv(A[:m,:m]);T=np.eye(n);T[:m,m:]=-H@B;p=rng.normal(size=n);E=rng.normal(size=(n,n));E=(E+E.T)/2;u=T@p;e=E@u;g=np.r_[e[:m],e[m:]-B.T@H@e[:m]];err=float(np.linalg.norm(g-T.T@E@T@p));assert err<1e-12;return err

def main():
 out=ROOT/'results/global-moment-congruence-error.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(50,55));start=perf_counter();cpu=process_time()
 paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(__file__).resolve()]+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},tiny_absolute_identity_error=tiny(),protocol='Exact native packed24core675 under shared hoisted product; copies of p,Gp,current r,rho are free observer data. Offline degree3 Gaussian expansion uses ONE mean over ALL scaled X, unlike prior product-specific centers. g=T^T(P-K)Tp. No inexact trajectory, memory or speed claim. Global P+lambdaI is SPD in exact arithmetic. Four development hard cases, no heldout/smalln validation.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,priorpath in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(priorpath.read_text());assert all(sha(ROOT/p)==h for p,h in prior['sources'].items());path=ROOT/'data/prepared'/scope.DATASETS[name];assert sha(path)==prior['data_sha256']
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   X.setflags(write=False);b.setflags(write=False);history=[];state={};cls=scope.packed.Fused;oldapply=cls.apply;oldpre=cls.precondition
   def pre(self,r):
    z=oldpre(self,r);state.update(op=self,r=r.copy(),rho=float(r@z));return z
   def apply(self,p,return_ha=False):
    value=oldapply(self,p,return_ha);assert not return_ha;history.append(dict(p=p.copy(),gp=value.copy(),r=state['r'].copy(),rho=state['rho']));return value
   cls.apply=apply;cls.precondition=pre
   try:
    with hoist.variant('hoisted'):answer,row=scope.packed.solve(X,b,bw,scope.Helper24(ROOT/'results/packed_int24_factor.so'),cpu_cap=7)
    del answer
   finally:cls.apply=oldapply;cls.precondition=oldpre
   controls=[r for r in prior['runs'] if r['method']=='packed24'];checks={k:all(row[k]==r[k] for r in controls) for k in ['iterations','kernel_stats','original_relative_residual','converged']};assert all(checks.values())
   op=state['op'];S,R=op.S,op.R;m=op.m;H=op.factor.solve;scaled=X/bw;K=np.exp(-cdist(scaled,scaled,'sqeuclidean')/2);P=polynomial(scaled,scaled,scaled.mean(0),3);B=K[np.ix_(S,R)];nb=float(np.linalg.norm(b));samples=[]
   for index in sorted(set([0,len(history)//2,len(history)-1])):
    record=history[index];p,gp,r=record['p'],record['gp'],record['r'];u=np.empty(len(b));u[S]=p[:m]-H(B@p[m:]);u[R]=p[m:];e=(P-K)@u;g=np.r_[e[S],e[R]-B.T@H(e[S])];Au=K@u+1e-5*u;exact=np.r_[Au[S],Au[R]-B.T@H(Au[S])];alpha=record['rho']/float(p@gp)
    samples.append(dict(capture_index_zero_based=index,capture_count=len(history),alpha=alpha,rho=record['rho'],r_norm=float(np.linalg.norm(r)),exact_gp_norm=float(np.linalg.norm(gp)),physical_error_norm=float(np.linalg.norm(e)),transformed_error_norm=float(np.linalg.norm(g)),relative_transformed_action_error=float(np.linalg.norm(g)/np.linalg.norm(gp)),step_error_over_recursive_residual=float(abs(alpha)*np.linalg.norm(g)/np.linalg.norm(r)),step_original_error_over_rhs=float(abs(alpha)*np.linalg.norm(e)/nb),step_original_error_over_requested_tolerance=float(abs(alpha)*np.linalg.norm(e)/(1e-8*nb)),offline_exact_native_relative_difference=float(np.linalg.norm(exact-gp)/np.linalg.norm(gp))))
   result['cases'].append(dict(dataset=name,data_sha256=sha(path),baseline=row,baseline_checks=checks,samples=samples));save();print(name,[(s['relative_transformed_action_error'],s['step_error_over_recursive_residual'],s['step_original_error_over_rhs']) for s in samples],flush=True)
 result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
