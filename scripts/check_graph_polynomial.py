"""Bounded positive Bernstein graph-polynomial preconditioner gate."""
import json,resource,math
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
ROOT=scope.ROOT
TRANS=np.zeros((9,9))
for j in range(9):
 for k in range(j,9):TRANS[k,j]=math.comb(8,j)*math.comb(8-j,k-j)*(-1)**(k-j)
def graph(X,bw):
 dist=cdist(X,X,'sqeuclidean');np.fill_diagonal(dist,np.inf);neighbors=np.argsort(dist,axis=1,kind='stable')[:,:8];rows=np.repeat(np.arange(len(X)),8);cols=neighbors.ravel();weights=np.exp(-dist[rows,cols]/(2*bw*bw));W=sparse.csr_matrix((weights,(rows,cols)),shape=(len(X),len(X)));W=W.maximum(W.T);degree=np.asarray(W.sum(1)).ravel();S=sparse.diags(1/np.sqrt(degree));J=(sparse.eye(len(X),format='csr')-S@W@S)*.5;return J.tocsr()
def Qmatrix(J,r):
 powers=[r.copy()]
 for _ in range(8):powers.append(J@powers[-1])
 return np.column_stack(powers)@TRANS
def action(J,a,v):
 coeff=TRANS@a;out=coeff[-1]*v
 for k in range(7,-1,-1):out=J@out+coeff[k]*v
 return out
def fit(J,A,r):
 n=len(r);Q=Qmatrix(J,r);G=Q.T@A@Q;G=(G+G.T)*.5;h=Q.T@r;lo=1/(n+1e-5);hi=1e5
 # Solve exactly the nine-variable convex quadratic in bounded t=a/hi.
 d=np.sqrt(np.maximum(np.diag(G),1e-300));C=G/(d[:,None]*d[None,:]);hh=h/d;w,U=np.linalg.eigh(C);tol=100*np.finfo(float).eps*max(float(w[-1]),1e-300);positive=w>tol
 if np.min(w)<-tol:return None,dict(valid=False,reason='indefinite computed normalized energy Gram',min_eigenvalue=float(w[0]))
 R=np.sqrt(w[positive])[:,None]*U[:,positive].T;target=(U[:,positive].T@hh)/np.sqrt(w[positive]);answer=lsq_linear(R,target,bounds=(d*lo,d*hi),tol=1e-13,max_iter=1000,lsmr_tol=1e-13)
 a=answer.x/d;gradient=(G@a-h)/d;normscale=max(np.linalg.norm(h/d,np.inf),np.linalg.norm((G@a)/d,np.inf),1e-300);normalized=gradient/normscale;projected=answer.x-np.clip(answer.x-normalized,d*lo,d*hi);kkt=float(np.linalg.norm(projected,np.inf))
 scalar=float(np.clip((r@r)/(r@A@r),lo,hi));ones=np.ones(9)*scalar;obj=lambda v:float(.5*v@G@v-v@h);value=obj(a);baseline=obj(ones);nullh=float(np.linalg.norm(hh-U[:,positive]@(U[:,positive].T@hh))/max(np.linalg.norm(hh),1e-300));valid=bool(answer.success and kkt<=1e-6 and value<=baseline+1e-8*max(abs(baseline),1) and nullh<1e-8)
 return a,dict(valid=valid,optimizer_success=bool(answer.success),message=answer.message,objective=value,scalar_objective=baseline,normalized_projected_gradient=kkt,nullspace_linear_fraction=nullh,gram_rank=int(sum(positive)),a=a.tolist(),bounds=[lo,hi])
def cg(A,b,H,x0=None,cap=2.5,stop4=False):
 x=np.zeros(len(b)) if x0 is None else x0.copy();r=b-A@x;z=H(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);t=process_time();status='maxiter';checks=0
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/nb);checks+=1
  if rel<=1e-8:status='converged';break
  if stop4 and k==4:status='prefix';break
  if process_time()-t>cap:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=H(r);new=float(r@z);p=z+new/rho*p;rho=new
 return x,dict(status=status,iterations=k,original_relative_residual=rel,true_checks=checks)
def tiny():
 rng=np.random.default_rng(81);X=rng.normal(size=(19,3));J=graph(X,1).toarray();a=np.exp(rng.normal(size=9));
 H=action(sparse.csr_matrix(J),a,np.eye(19));ev,V=np.linalg.eigh(J);f=sum(a[j]*math.comb(8,j)*ev**j*(1-ev)**(8-j) for j in range(9));ref=(V*f)@V.T;err=float(np.linalg.norm(H-ref)/np.linalg.norm(ref));assert err<1e-10 and np.min(np.linalg.eigvalsh((H+H.T)*.5))>0;return dict(relative_horner_error=err,min_H_eigenvalue=float(np.linalg.eigvalsh((H+H.T)*.5)[0]))
def main():
 out=ROOT/'results/graph-polynomial.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
 paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(__file__).resolve()]+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol='Symmetric union8NN Gaussian graph, normalized Laplacian/2 J. Degree8 positive Bernstein polynomial, bounded coefficients. One9RHS fit kernelpass plus explicit scalarRHS count. Dense kernel, graph construction, fit arithmetic and periterate truechecks free oracle; no native runtime/memory claim. Current/initial/random persistent versus same current-action warmstart/plain controls at common4CGprefix. No ridge; rank-truncated numerical Gram factor permitted only with recorded negligible nullspace linear component and KKT checks.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  result['tiny']=tiny();save()
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(path.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha256(datafile.read_bytes()).hexdigest()==prior['data_sha256']
   with np.load(datafile) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   n=len(b);A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A.flat[::n+1]+=1e-5;J=graph(X,bw);x4,prefix=cg(A,b,lambda r:r,stop4=True);r4=b-A@x4;assert prefix['status']=='prefix';rng=np.random.default_rng(622018);targets={'current':r4,'initial':b,'random':rng.normal(size=n)};fits={};case=dict(dataset=name,data_sha256=prior['data_sha256'],prefix=prefix,graph_nnz=J.nnz,memory_lower_bound=dict(CSR=J.data.nbytes+J.indices.nbytes+J.indptr.nbytes,coefficients=72,Q_AQ=144*n,eight_work_vectors=64*n,qualification='Dense graph construction/distances/kernel and eig/fit workspace excluded; incomplete lower bound.'),fits={},runs=[],strong_static_controls=[r for r in prior['runs'] if r['method'] in ['uniform600','packed24','original_exact416']]);result['cases'].append(case)
   for key,target in targets.items():a,info=fit(J,A,target);case['fits'][key]=info;fits[key]=a;save()
   for mode in ['current','initial','random','current_warm','plain_restart','plain_uninterrupted']:
    selected=mode if mode in fits else 'current' if mode=='current_warm' else None
    if selected is not None and not case['fits'][selected]['valid']:case['runs'].append(dict(method=mode,status='invalid_fit'));save();continue
    x0=x4.copy();prefixcount=4;fitcount=int(selected is not None);warm=0
    if mode in fits:H=lambda v,a=fits[mode]:action(J,a,v)
    else:
     H=lambda v:v
     if mode=='plain_uninterrupted':x0=None;prefixcount=0
     if mode=='current_warm':z=action(J,fits['current'],r4);az=A@z;x0+=z*float(r4@z)/float(z@az);warm=1
    _,row=cg(A,b,H,x0);row.update(method=mode,fit_rhs_count=9*fitcount,fit_kernel_passes=fitcount,prefix_products=prefixcount,warm_products=warm,operator_products=prefixcount+fitcount+warm+row['iterations']+int(x0 is not None)+1,all_true_check_products=row['true_checks'],analytical_entries=(prefixcount+fitcount+warm+row['iterations']+int(x0 is not None)+1)*n*n)
    case['runs'].append(row);print(name,mode,row['status'],row['iterations'],flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
