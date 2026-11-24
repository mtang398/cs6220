"""Frozen positive diagonal metric in symmetrically preconditioned coordinates."""
import json,resource,sys
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_cached_kernel_static_frontier as static
import check_refresh_regularization_scope as scope
ROOT=scope.ROOT

def half(M,v,inverse=False):
 U=M.basis;lam=M.regularization;c=U.T@v;perp=v-U@c;fix=U.T@perp;c+=fix;perp-=U@fix
 complement=np.sqrt(lam) if inverse else 1/np.sqrt(lam);values=np.sqrt(M.eigenvalues+lam) if inverse else 1/np.sqrt(M.eigenvalues+lam)
 return complement*perp+U@(values[:,None]*c if v.ndim==2 else values*c)
def fit(u,v):
 norms=np.linalg.norm(u,axis=0)
 if np.any(norms==0) or not np.all(np.isfinite(norms)):return None,dict(valid=False,reason='invalid_pair_norm')
 u=u/norms;v=v/norms;s=np.sum(u*u,axis=1);t=np.sum(v*v,axis=1);both=(s==0)&(t==0);bad=((s==0)^(t==0))|~np.isfinite(s+t)
 if np.any(bad):return None,dict(valid=False,reason='inconsistent_zero_or_nonfinite_statistics',bad_coordinates=int(np.sum(bad)))
 d=np.ones(len(s));d[~both]=np.sqrt(s[~both]/t[~both]);valid=bool(np.all(np.isfinite(d)) and np.all(d>0))
 return d,dict(valid=valid,min_d=float(d.min()),max_d=float(d.max()),zerozero_coordinates=int(sum(both)))
def cdiag(A,M):
 U=M.basis;a=1/np.sqrt(M.regularization);c=1/np.sqrt(M.eigenvalues+M.regularization)-a;AU=A@U;small=U.T@AU
 return a*a*np.diag(A)+2*a*np.sum((U*c)*AU,axis=1)+np.sum(((U*c)@small)*(U*c),axis=1)
def solve(A,b,H,x0=None,prefix=False):
 x=np.zeros(len(b)) if x0 is None else x0.copy();r=b-A@x;z=H(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);t=process_time();P=[];AP=[];status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if prefix and k==8:status='prefix';break
  if process_time()-t>2:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if not np.isfinite(den+rho) or min(den,rho)<=0:status='invalid_curvature';break
  if prefix:P.append(p.copy());AP.append(ap.copy())
  alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=H(r);new=float(r@z);p=z+(new/rho)*p;rho=new
 return x,dict(status=status,iterations=k,original_relative_residual=rel),P,AP

def tiny():
 rng=np.random.default_rng(622131);n=23;U=np.linalg.qr(rng.normal(size=(n,6)))[0];M=static.LowRankPreconditioner(U,np.arange(1,7),.02);F=rng.normal(size=(n,n));A=F@F.T+np.eye(n);S=half(M,np.eye(n));inv=half(M,np.eye(n),True);err=float(np.linalg.norm(S@inv-np.eye(n)));diagerr=float(np.linalg.norm(cdiag(A,M)-np.diag(S@A@S))/np.linalg.norm(np.diag(S@A@S)));d=np.exp(rng.normal(size=n));B=S@np.diag(d)@S;ev=float(eigh((B+B.T)*.5,eigvals_only=True)[0]);assert err<1e-10 and diagerr<1e-12 and ev>0
 return dict(inverse_half_identity_error=err,Cdiag_relative_error=diagerr,metric_minimum_eigenvalue=ev)
def main():
 out=ROOT/'results/whitened-diagonal-metric.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time();paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(static.__file__).resolve(),Path(__file__).resolve(),ROOT/'src/nystrom_project/preconditioners.py',ROOT/'src/nystrom_project/numerics.py']+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},tiny=tiny(),protocol='FixedFPS128 Nyström H, eightPCGprefix paidP/AP. u=H^-half p,v=HhalfAp; normalize each paired column bynormu; d=sqrt(sum u²/sum v²). No clipping; zerozero1/inconsistentzero invalid. Freeze Hhalf diag(d) Hhalf; current8/firstpair/random8/exactdiagC oracle/unchangedrestart/uninterrupted/currentwarm thenunchanged. DenseK/setup/everyiterate true checks/metric arithmetic free oracle. Random8 Cq requires8extra ARHS; exactdiagC usesrank128 Aimages and is explicitlyfreeoracle. No native memory/time/novelty claim;2CPUbranch/max2000.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(path.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha256(datafile.read_bytes()).hexdigest()==prior['data_sha256']
   with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   n=len(b);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A=K+1e-5*np.eye(n);indices=static.fps(K,128);M=static.LowRankPreconditioner.from_sketch(K[:,indices],K[np.ix_(indices,indices)],1e-5);x8,prefix,P,AP=solve(A,b,M.apply,prefix=True);assert prefix['status']=='prefix';P=np.column_stack(P);AP=np.column_stack(AP);u=half(M,P,True);v=half(M,AP);rng=np.random.default_rng(622132);q=rng.normal(size=(n,8));cq=half(M,A@half(M,q));fits={};metrics={}
   for key,uu,vv in [('current',u,v),('first_pair',u[:,:1],v[:,:1]),('random',q,cq)]:metrics[key],fits[key]=fit(uu,vv)
   diagonal=cdiag(A,M);valid=bool(np.all(np.isfinite(diagonal)) and np.all(diagonal>0));fits['oracle_jacobi']=dict(valid=valid,min_Cdiag=float(diagonal.min()),max_Cdiag=float(diagonal.max()));metrics['oracle_jacobi']=1/diagonal if valid else None
   case=dict(dataset=name,data_sha256=prior['data_sha256'],prefix=prefix,fits=fits,runs=[],strong_controls=[r for r in prior['runs'] if r['method'] in ['uniform600','packed24','original_exact416']]);result['cases'].append(case)
   for mode in ['current','first_pair','random','oracle_jacobi','unchanged_restart','unchanged_uninterrupted','current_warm']:
    key=mode if mode in metrics else 'current' if mode=='current_warm' else None
    if key and not fits[key]['valid']:case['runs'].append(dict(method=mode,status='invalid_metric'));save();continue
    d=metrics.get(key);H=(lambda r,d=d:half(M,d*half(M,r))) if key else M.apply;x0=x8;pref=8;warm=0
    if mode=='unchanged_uninterrupted':x0=None;pref=0
    if mode=='current_warm':r=b-A@x8;z=H(r);az=A@z;x0=x8+z*float(r@z)/float(z@az);H=M.apply;warm=1
    _,row,_,_=solve(A,b,H,x0);row.update(method=mode,prefix_products=pref,warm_products=warm,extra_probe_rhs=8 if mode=='random' else 0,oracle_matrix_rhs=128 if mode=='oracle_jacobi' else 0,operator_products_without_free_acquisition=pref+warm+row['iterations']+int(x0 is not None)+1)
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    case['runs'].append(row);print(name,mode,row['status'],row['iterations'],row['original_relative_residual'],flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
