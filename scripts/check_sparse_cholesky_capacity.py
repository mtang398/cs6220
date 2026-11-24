"""Sparse full Cholesky capacity gate; exact factor/actions free information oracle."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
from check_scaled_hss_headroom import partition
ROOT=scope.ROOT

def inverse(T,r):return solve_triangular(T.T,solve_triangular(T,r,lower=True,check_finite=False),lower=False,check_finite=False)
def score(L,r):
 v=solve_triangular(L,r,lower=True,check_finite=False);w=solve_triangular(L.T,v,lower=False,check_finite=False)
 return L*L*(v[None,:]**2/max(float(r@r),1e-300)+w[:,None]**2/max(float(v@v),1e-300))
def sparsify(L,scores):
 n=len(L);fixed=dict(reserve=65536,kernel_workspace=65536,permutation=8*n,sixteen_vectors=128*n,CSR_row_pointers=4*(n+1));available=1048576-sum(fixed.values());nnz=min(n*(n+1)//2,available//12);assert nnz>=n
 rows,cols=np.tril_indices(n,-1);take=nnz-n;values=scores[rows,cols];order=np.lexsort((cols,rows,-values))[:take];T=np.diag(np.diag(L));T[rows[order],cols[order]]=L[rows[order],cols[order]];assert np.all(np.isfinite(T)) and np.all(np.diag(T)>0)
 return T,dict(fixed=fixed,retained_nnz=nnz,FP64_int32_entry_bytes=12*nnz,total_bytes=sum(fixed.values())+12*nnz,qualification='Single CSR factor, transpose view shares values. Logical compact payload only; dense oracle factor/masks/construction/exact inverse actions excluded, native peak unproven.')
def solve(A,b,T=None,x0=None,prefix=False):
 H=(lambda r:r) if T is None else (lambda r:inverse(T,r));x=np.zeros(len(b)) if x0 is None else x0.copy();r=b-A@x;z=H(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);t=process_time();status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if prefix and k==4:status='prefix';break
  if process_time()-t>2:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if not np.isfinite(den+rho) or min(den,rho)<=0:status='invalid_curvature';break
  alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=H(r);new=float(r@z);p=z+(new/rho)*p;rho=new
 return x,dict(status=status,iterations=k,original_relative_residual=rel)
def tiny():
 rng=np.random.default_rng(622099);F=rng.normal(size=(13,13));A=F@F.T+np.eye(13);L=cholesky(A,lower=True);r=rng.normal(size=13);E=np.tril(rng.normal(size=(13,13)),-1);v=solve_triangular(L,r,lower=True);w=inverse(L,r);derivative=-inverse(L,E@v)-solve_triangular(L.T,E.T@w,lower=False);eps=1e-6;finite=(inverse(L+eps*E,r)-inverse(L-eps*E,r))/(2*eps);err=float(np.linalg.norm(finite-derivative)/np.linalg.norm(derivative));assert err<1e-7
 T=np.diag(np.diag(L))+np.tril(L,-1)*(np.tril(rng.random((13,13)),-1)>.5);H=inverse(T,np.eye(13));sym=float(np.linalg.norm(H-H.T)/np.linalg.norm(H));mineig=float(np.linalg.eigvalsh((H+H.T)*.5)[0]);actionerr=float(np.linalg.norm((T@T.T)@H-np.eye(13)));assert sym<1e-12 and mineig>0 and actionerr<1e-10
 return dict(inverse_derivative_relative_error=err,SPD_min_eigenvalue=mineig,inverse_symmetry_relative_error=sym,inverse_action_identity_error=actionerr)
def main():
 out=ROOT/'results/sparse-cholesky-capacity.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time();paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(__file__).resolve(),ROOT/"scripts/check_scaled_hss_headroom.py"]+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},tiny=tiny(),protocol='Exact free Cholesky after common geometric ordering. Keep every diagonal and globaltopoffdiagonal scores to fixed1MiB logical CSR payload including16vectors,permutation,64KiBworkspace/reserve. FP64values/int32indices12bytes perentry. Magnitude Lij² versus current/initial/random normalizedtwohalfsolve scores. Currentr from4plainCGprefix. Scores measure summed single-entry effects, not joint objective/optimality; cancellations and inverse amplification omitted. DenseT triangular solves and all exact L/inverse actions free oracle; no full-cost/native memory/novelty claim. Literal1e-8,max2000,2CPUbranch.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(path.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha256(datafile.read_bytes()).hexdigest()==prior['data_sha256']
   with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   perm,_=partition(X);Xp=X[perm];bp=b[perm];A=np.exp(-cdist(Xp,Xp,'sqeuclidean')/(2*bw*bw));A.flat[::len(b)+1]+=1e-5;L=cholesky(A,lower=True);x4,prefix=solve(A,bp,prefix=True);assert prefix['status']=='prefix';r4=bp-A@x4;rng=np.random.default_rng(622100)
   case=dict(dataset=name,data_sha256=prior['data_sha256'],prefix=prefix,permutation_sha256=sha256(perm.tobytes()).hexdigest(),runs=[],strong_controls=[r for r in prior['runs'] if r['method'] in ['uniform600','packed24','original_exact416']]);result['cases'].append(case);currentT=None
   for mode in ['magnitude','current','initial','random','current_warm','plain_restart','plain_uninterrupted']:
    planner=None;warm=0
    if mode in ['magnitude','current','initial','random']:
     scores=L*L if mode=='magnitude' else score(L,r4 if mode=='current' else bp if mode=='initial' else rng.normal(size=len(b)));T,planner=sparsify(L,scores);del scores
     if mode=='current':currentT=T.copy()
     x0=x4;active=T
    elif mode=='current_warm':
     z=inverse(currentT,r4);az=A@z;den=float(z@az)
     if not np.isfinite(den) or den<=0:case['runs'].append(dict(method=mode,status='invalid_warm_curvature'));save();continue
     x0=x4+z*float(r4@z)/den;active=None;warm=1
    elif mode=='plain_restart':x0=x4;active=None
    else:x0=None;active=None
    _,row=solve(A,bp,active,x0);pref=0 if mode=='plain_uninterrupted' else 4;row.update(method=mode,planner=planner,prefix_products=pref,warm_products=warm,optimistic_operator_products=pref+warm+row['iterations']+int(x0 is not None)+1)
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    case['runs'].append(row);print(name,mode,row['status'],row['iterations'],row['original_relative_residual'],flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
