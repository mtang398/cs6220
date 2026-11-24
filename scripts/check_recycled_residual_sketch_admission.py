"""Recycled residual-sketch coordinate admission, fixed64→96; dense diagnostic."""
import json,resource,sys
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import qr,solve_triangular,svd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.preconditioners import LowRankPreconditioner
import check_refresh_regularization_scope as scope

def sha(p):return sha256(p.read_bytes()).hexdigest()
def fps(K,count):
 inds=[];sim=np.zeros(len(K));j=0
 for _ in range(count):inds.append(j);np.maximum(sim,K[:,j],out=sim);sim[inds]=np.inf;j=int(np.argmin(sim))
 return np.array(inds,dtype=int)
def append(K,F,indices,requested):
 events=[]
 for j in requested:
  if int(j) in indices:raise ValueError('duplicate admitted column')
  col=K[:,j]-F@F[j];pivot=float(col[j]);tol=64*np.finfo(float).eps*max(len(K),1)
  if pivot<=tol:return F,indices,events,False
  F=np.column_stack((F,col/np.sqrt(pivot)));indices=indices+[int(j)];events.append(dict(index=int(j),pivot=pivot))
 return F,indices,events,True

def preconditioner(F):
 U,s,_=svd(F,full_matrices=False,check_finite=False);return LowRankPreconditioner(U,s*s,1e-5)
def solve(A,b,M,x0=None,prefix=False):
 x=np.zeros(len(b)) if x0 is None else x0.copy();r=b-A@x;z=M.apply(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);t=process_time();P=[];AP=[];status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/nb)
  if rel<=1e-8:status='converged';break
  if prefix and k==8:status='prefix';break
  if process_time()-t>2:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  if prefix:P.append(p.copy());AP.append(ap.copy())
  alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=M.apply(r);new=float(r@z);p=z+new/rho*p;rho=new
 return x,dict(status=status,iterations=k,original_relative_residual=rel),P,AP

def recycle(K,F,indices,Q,KQ):
 F=F.copy();indices=list(indices);rounds=[]
 for cycle in range(4):
  Y=(KQ-F@(F.T@Q)).T;Y[:,indices]=0
  _,R,piv=qr(Y,mode='economic',pivoting=True,check_finite=False);diag=np.abs(np.diag(R));tol=max(Y.shape)*np.finfo(float).eps*max(np.linalg.norm(Y,2),1e-300);rank=int(np.sum(diag>tol))
  if rank<8:return F,indices,rounds,False,'residual_sketch_rank_depleted'
  selected=piv[:8];F,indices,events,valid=append(K,F,indices,selected);rounds.append(dict(round=cycle,sketch_rank=rank,sketch_norm=float(np.linalg.norm(Y)),pivots=events))
  if not valid:return F,indices,rounds,False,'nonpositive_residual_pivot'
 return F,indices,rounds,True,None

def tiny():
 rng=np.random.default_rng(622046);Z=rng.normal(size=(23,12));K=Z@Z.T+.1*np.eye(23);F,indices,_,ok=append(K,np.empty((23,0)),[],[0,4,8]);assert ok;Q,_=qr(rng.normal(size=(23,8)),mode='economic');KQ=K@Q
 Y=(KQ-F@(F.T@Q)).T;direct=Q.T@(K-F@F.T);relative=float(np.linalg.norm(Y-direct)/np.linalg.norm(direct));_,_,piv=qr(Y,mode='economic',pivoting=True);new=[int(j) for j in piv if int(j) not in indices][:8];F,indices,_,ok=append(K,F,indices,new);assert ok;minimum=float(np.linalg.eigvalsh((K-F@F.T+K.T-F@F.T)*.5)[0]);assert relative<1e-12 and minimum>-1e-10
 return dict(recycled_sketch_relative_error=relative,residual_minimum_eigenvalue=minimum)

def main():
 out=ROOT/'results/recycled-residual-sketch-admission.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time()
 paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(__file__).resolve(),ROOT/'src/nystrom_project/preconditioners.py',ROOT/'src/nystrom_project/numerics.py']+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol=dict(rank_initial=64,rank_final=96,prefix=8,sketch_width=8,rounds=4,mechanism='QRCP of recycled Q^T(K−FF^T) selects8 actual coordinates per round. Append residual Cholesky columns; update sketch from SAME cached KQ, no deletion. Rank depletion explicitly invalidates branch; no random fill.',controls='Independent Gaussian8 sketch sameprefix, greedy residual-diagonal growth sameprefix, current finalset upfront hindsight, staticFPS96 upfront, initial64 uninterrupted/restart.',cost='Dense kernel/construction/SVD/free everyiterate true stopping; record prefix, candidate column queries, Gaussian8RHS acquisition, restart/final checks separately. No native full-cost or novelty claim. 2CPU/solve,2000cap. Stored P/AP and coordinate admissions may acquire off-Krylov columns.'),tiny=tiny(),cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   old=json.loads(path.read_text());assert all(sha(ROOT/p)==h for p,h in old['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha(datafile)==old['data_sha256']
   with np.load(datafile) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   n=len(b);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A=K+1e-5*np.eye(n);initial=fps(K,64);F,indices,_,ok=append(K,np.empty((n,0)),[],initial);assert ok;M=preconditioner(F);x8,prefix,P,AP=solve(A,b,M,prefix=True);assert prefix['status']=='prefix'
   P=np.column_stack(P);AP=np.column_stack(AP);norms=np.linalg.norm(P,axis=0);Pn=P/norms;APn=AP/norms;Q,R=qr(Pn,mode='economic');KQ=solve_triangular(R.T,(APn-1e-5*Pn).T,lower=True,check_finite=False).T;image_error=float(np.linalg.norm(KQ-K@Q)/max(np.linalg.norm(K@Q),1e-300))
   random,_=qr(np.random.default_rng(622045).normal(size=(n,8)),mode='economic');case=dict(dataset=name,data_sha256=old['data_sha256'],prefix=prefix,paid_image_relative_error=image_error,normalized_prefix_R_condition=float(np.linalg.cond(R)),sketches={},runs=[],strong_controls=[r for r in old['runs'] if r['method'] in ['uniform600','packed24','original_exact416']]);result['cases'].append(case)
   variants={}
   for mode,q,kq in [('current',Q,KQ),('gaussian',random,K@random)]:
    FF,ii,rounds,valid,reason=recycle(K,F,indices,q,kq);case['sketches'][mode]=dict(valid=valid,reason=reason,rounds=rounds,final_indices=ii)
    if mode=='current' and image_error>1e-7:case['sketches'][mode].update(valid=False,reason='paid_image_reconstruction_error');valid=False
    if valid:variants[mode]=FF
   FF=F.copy();ii=list(indices)
   for _ in range(32):
    diagonal=np.diag(K)-np.sum(FF*FF,axis=1);diagonal[ii]=-np.inf;j=int(np.argmax(diagonal));FF,ii,_,valid=append(K,FF,ii,[j]);assert valid
   variants['diagonal']=FF;static,_,_,valid=append(K,np.empty((n,0)),[],fps(K,96));assert valid
   branches=[('initial_uninterrupted',F,None,0,0,64),('initial_restart',F,x8,8,0,64),('diagonal_growth',variants['diagonal'],x8,8,0,96),('static96_upfront',static,None,0,0,96)]
   if 'current' in variants:branches += [('current_growth',variants['current'],x8,8,0,96),('current_finalset_upfront_hindsight',variants['current'],None,0,0,96)]
   if 'gaussian'in variants:branches += [('gaussian_growth',variants['gaussian'],x8,8,8,96)]
   for mode,FF,start,pref,probe,columns in branches:
    _,row,_,_=solve(A,b,preconditioner(FF),start);row.update(method=mode,prefix_products=pref,independent_probe_rhs_products=probe,actual_columns=columns,analytical_entries=(pref+row['iterations']+int(start is not None)+1+(1 if probe else 0))*n*n+n*columns,oracle_note='Denseconstruction/repeated literal stopping excluded except one final verification; Gaussian probe counted one width8 batch and8scalarRHS. Same-finalset upfront is hindsight selection oracle.')
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    case['runs'].append(row);print(name,mode,row['status'],row['iterations'],flush=True);save()
 result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
