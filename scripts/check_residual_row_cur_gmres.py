"""Residual-weighted CUR row selection; fixed right-GMRES20 capacity gate."""
import json,resource,warnings
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import qr,lu_factor,lu_solve,solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_cached_kernel_static_frontier as static
import check_refresh_regularization_scope as scope
import check_whitened_diagonal_metric as pcg_helper
ROOT=scope.ROOT

def cur_inverse(K,J,I,lam):
 C=K[:,J];R=K[I,:];W=K[np.ix_(I,J)];Q,T=qr(C,mode='economic');B=solve(W.T,T.T,assume_a='gen',check_finite=False).T@R;small=lam*np.eye(len(J))+B@Q
 with warnings.catch_warnings():
  warnings.simplefilter('error');lu=lu_factor(small,check_finite=False)
 def apply(r):
  c=Q.T@r;perp=r-Q@c;fix=Q.T@perp;c+=fix;perp-=Q@fix
  return perp/lam+Q@lu_solve(lu,c-(B@perp)/lam,check_finite=False)
 apply.matrix_apply=lambda v:lam*v+Q@(B@v)
 return apply,dict(W_condition=float(np.linalg.cond(W)),small_condition=float(np.linalg.cond(small)),C_condition=float(np.linalg.cond(T))), (C,R,W)

def prefix(A,b):
 x=np.zeros(len(b));r=b.copy();p=r.copy();rr=float(r@r)
 for _ in range(4):ap=A@p;alpha=rr/float(p@ap);x+=alpha*p;r-=alpha*ap;new=float(r@r);p=r+(new/rr)*p;rr=new
 return x,b-A@x

def gmres(A,b,H,x0):
 start=process_time();x=x0.copy();r=b-A@x;nb=np.linalg.norm(b);steps=0;h_calls=0;op=1;checks=1;status='maxiter';orthmax=0.;restart_count=0
 while steps<2000:
  relative=float(np.linalg.norm(r)/nb)
  if relative<=1e-8:status='converged';break
  if process_time()-start>2:status='cpu_cap';break
  beta=np.linalg.norm(r);V=np.zeros((len(b),21));V[:,0]=r/beta;hess=np.zeros((21,20));base=x.copy();candidate=x.copy();rr=r.copy();restart_count+=1
  for j in range(min(20,2000-steps)):
   if process_time()-start>2:status='cpu_cap';break
   z=H(V[:,j]);h_calls+=1;w=A@z;op+=1
   for _ in range(2):coefs=V[:,:j+1].T@w;hess[:j+1,j]+=coefs;w-=V[:,:j+1]@coefs
   hn=float(np.linalg.norm(w));hess[j+1,j]=hn
   if hn>0:V[:,j+1]=w/hn
   rhs=np.zeros(j+2);rhs[0]=beta;y=np.linalg.lstsq(hess[:j+2,:j+1],rhs,rcond=None)[0];candidate=base+H(V[:,:j+1]@y);h_calls+=1;rr=b-A@candidate;checks+=1;steps+=1;relative=float(np.linalg.norm(rr)/nb)
   if not np.isfinite(relative):status='nonfinite_residual';break
   orthmax=max(orthmax,float(np.linalg.norm(V[:,:j+1].T@V[:,:j+1]-np.eye(j+1))))
   if relative<=1e-8:status='converged';break
   if hn==0:status='arnoldi_breakdown';break
  x=candidate;r=rr
  if status!='maxiter':break
 return dict(status=status,iterations=steps,original_relative_residual=float(np.linalg.norm(b-A@x)/nb),right_preconditioner_applications=h_calls,arnoldi_and_initial_A_products=op,literal_residual_checks=checks+1,restart_cycles=restart_count,max_basis_orthogonality_error=orthmax)

def tiny():
 rng=np.random.default_rng(622151);F=rng.normal(size=(19,19));K=F@F.T+.1*np.eye(19);J=np.arange(4);I=np.array([0,2,5,8]);lam=.2;H,meta,(C,R,W)=cur_inverse(K,J,I,lam);r=rng.normal(size=19);M=lam*np.eye(19)+C@solve(W,R);direct=solve(M,r);stable=H(r);naive=(r-C@solve(lam*W+R@C,R@r))/lam;err=float(np.linalg.norm(stable-direct)/np.linalg.norm(direct));naiveerr=float(np.linalg.norm(naive-direct)/np.linalg.norm(direct));assert err<1e-9 and naiveerr<1e-9
 return dict(stable_inverse_relative_error=err,woodbury_relative_error=naiveerr,conditioning=meta)

def main():
 out=ROOT/'results/residual-row-cur-gmres.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time();paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(static.__file__).resolve(),Path(pcg_helper.__file__).resolve(),Path(__file__).resolve(),ROOT/'src/nystrom_project/preconditioners.py',ROOT/'src/nystrom_project/numerics.py']+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},tiny=tiny(),protocol='FixedFPS J64, QRCP of weightedC.T selects64distinctrows; weightsabs(currentr4)/normr4, abs(initialb)/normb, abs(seedGaussian)/normprobe, orones. Same4plainCGprefix. FixedrightGMRES20, two-passMGS, literaltrue1e-8/max2000/2CPUbranch; everycandidateupdate appliesfixedH once (counted). CURstableQRWoodbury; no SPDclaim. SameGMRES Nyström64samecolumns andFPS128. DenseK/construction/conditioning/perstepverification freecapacityoracle, not fullcost/native memory/novelty. Basis21n and retainedCUR2nk+k² countedinlowerbound; actualdenseworkspace unresolved.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(path.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha256(datafile.read_bytes()).hexdigest()==prior['data_sha256']
   with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   n=len(b);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A=K+1e-5*np.eye(n);J=static.fps(K,64);C=K[:,J];x4,r4=prefix(A,b);rng=np.random.default_rng(622152);random=rng.normal(size=n);weights=dict(current=np.abs(r4)/np.linalg.norm(r4),matrix=np.ones(n),initial=np.abs(b)/np.linalg.norm(b),random=np.abs(random)/np.linalg.norm(random));case=dict(dataset=name,data_sha256=prior['data_sha256'],runs=[],retained_lower_bound_bytes=8*(2*n*64+64*64+21*n)+8*20*21+4*128,strong_controls=[r for r in prior['runs'] if r['method'] in ['uniform600','packed24','original_exact416']]);result['cases'].append(case)
   for mode in ['current','matrix','initial','random','nystrom64','nystrom128','pcg128']:
    info={}
    try:
     if mode in weights:
      _,R,piv=qr((weights[mode][:,None]*C).T,mode='economic',pivoting=True,check_finite=False);I=piv[:64];H,info,_=cur_inverse(K,J,I,1e-5);info['row_indices']=I.tolist();inverse_check=float(np.linalg.norm(H.matrix_apply(H(r4))-r4)/np.linalg.norm(r4));info['checkpoint_inverse_action_relative_error']=inverse_check
      if not np.isfinite(inverse_check) or inverse_check>1e-7:
       case['runs'].append(dict(method=mode,status='invalid_inverse_action',conditioning=info));save();continue
     else:
      selected=J if mode=='nystrom64' else static.fps(K,128);M=static.LowRankPreconditioner.from_sketch(K[:,selected],K[np.ix_(selected,selected)],1e-5);H=M.apply
     if mode=='pcg128':_,row,_,_=pcg_helper.solve(A,b,H,x4)
     else:row=gmres(A,b,H,x4)
     row.update(method=mode,conditioning=info,prefix_products=4)
    except (np.linalg.LinAlgError,RuntimeWarning) as exc:row=dict(method=mode,status='invalid_preconditioner',error=str(exc),conditioning=info)
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    case['runs'].append(row);print(name,mode,row['status'],row.get('iterations'),row.get('original_relative_residual'),flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
