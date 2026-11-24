"""Nested sparse orthogonal rotations and core-diagonal MMF capacity oracle."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve,eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
import check_scaled_hss_headroom as ordering
ROOT=scope.ROOT

def planner(n):
 def parts(m):return dict(packed_core=4*m*(m+1),rotation_blocks=40*(n-m),diagonals=8*n,core_indices=4*m,ordering=8*n,sixteen_vectors=128*n,kernel_workspace=65536,reserve=65536)
 m=max(m for m in range(1,n+1) if sum(parts(m).values())<=1048576)
 return m,dict(parts=parts(m),total_bytes=sum(parts(m).values()),core=m,qualification='One general2x2 rotation stores2int32 indices+4FP64 values=40B; layers require no extra permutation, bounded offsets counted within reserve. Logical compact planner only, dense construction excluded.')

def construct(A,target,m,mode):
 n=len(A);W=A.copy();z=target.copy();active=list(range(n));layers=[];retired=[];alive=np.ones(n,dtype=bool);local_loss=[]
 while len(active)>m:
  layer=[];nextactive=[];pairs=min(len(active)//2,len(active)-m)
  for k in range(pairs):
   i,j=active[2*k:2*k+2];ids=np.array([i,j]);rows=W[np.ix_(ids,np.array(active))];gram=rows@rows.T;_,vectors=eigh((gram+gram.T)*.5,check_finite=False);R=vectors.T
   # Explicit general orthogonal matrix retained; reflection is allowed.
   W[ids,:]=R@W[ids,:];W[:,ids]=W[:,ids]@R.T;z[ids]=R@z[ids]
   remaining=np.asarray(active,dtype=int);remaining=remaining[alive[remaining]]
   costs=[]
   for e in [i,j]:
    rest=remaining[remaining!=e];g=W[e,rest]
    costs.append(float(g@g) if mode=='static' else float((g@z[rest])**2+z[e]**2*(g@g)))
   which=int(costs[1]<costs[0]);e=[i,j][which];keep=[j,i][which]
   assert np.isfinite(W[e,e]) and W[e,e]>0
   saved_diagonal=float(W[e,e]);W[e,:]=0;W[:,e]=0;W[e,e]=saved_diagonal
   retired.append(e);alive[e]=False;nextactive.append(keep);layer.append((i,j,R));local_loss.append(costs[which])
  nextactive.extend(active[2*pairs:]);active=nextactive;layers.append(layer)
 return W,z,active,layers,retired,local_loss

def rotate(v,layers,transpose=False):
 out=v.copy()
 for ii,jj,R in (reversed(layers) if transpose else layers):
  a=out[ii].copy();b=out[jj].copy()
  if not transpose:out[ii]=R[:,0,0]*a+R[:,0,1]*b;out[jj]=R[:,1,0]*a+R[:,1,1]*b
  else:out[ii]=R[:,0,0]*a+R[:,1,0]*b;out[jj]=R[:,0,1]*a+R[:,1,1]*b
 return out

def model(A,target,m,mode):
 W,z,core,layers,retired,loss=construct(A,target,m,mode);layers=[(np.array([r[0] for r in layer],dtype=np.int32),np.array([r[1] for r in layer],dtype=np.int32),np.array([r[2] for r in layer])) for layer in layers];core=np.array(core);retired=np.array(retired,dtype=int);diagonal=np.diag(W).copy();L=cholesky(W[np.ix_(core,core)],lower=True)
 def apply(r):
  v=rotate(r,layers);v[retired]/=diagonal[retired];v[core]=cho_solve((L,True),v[core],check_finite=False);return rotate(v,layers,True)
 return apply,dict(rotation_count=sum(len(layer[0]) for layer in layers),layers=len(layers),minimum_retired_diagonal=float(diagonal[retired].min()) if len(retired) else None,local_loss_sum=float(sum(loss)),final_action_error=float(np.linalg.norm((A@target)-rotate(W@rotate(target,layers),layers,True)))),(W,core,layers)

def solve(A,b,H,x0=None,prefix=False):
 x=np.zeros(len(b)) if x0 is None else x0.copy();r=b-A@x;z=H(r);p=z.copy();rho=float(r@z);nb=np.linalg.norm(b);t=process_time();status='maxiter'
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
 rng=np.random.default_rng(622141);F=rng.normal(size=(17,17));A=F@F.T+np.eye(17);r=rng.normal(size=17);H,metadata,(W,core,layers)=model(A,r,5,'current');Q=np.column_stack([rotate(np.eye(17)[:,j],layers) for j in range(17)]);orth=float(np.linalg.norm(Q.T@Q-np.eye(17)));B=Q.T@W@Q;inverse=np.column_stack([H(np.eye(17)[:,j]) for j in range(17)]);identity=float(np.linalg.norm(B@inverse-np.eye(17)));mineig=float(np.linalg.eigvalsh(B)[0]);g=rng.normal(size=16);v=rng.normal(size=17);E=np.zeros((17,17));E[0,1:]=g;E[1:,0]=g;drop=abs(float(np.linalg.norm(E@v)**2-((g@v[1:])**2+v[0]**2*(g@g))));assert orth<1e-12 and identity<1e-10 and mineig>0 and drop<1e-10
 return dict(orthogonality_error=orth,representation_inverse_error=identity,minimum_eigenvalue=mineig,local_drop_identity_error=drop)

def main():
 out=ROOT/'results/mmf-capacity.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(100,105));t=perf_counter();cpu=process_time();paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(__file__).resolve(),Path(ordering.__file__).resolve()]+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},tiny=tiny(),protocol='Common geometric permutation; nested disjoint adjacent active pairs, matrix-only2rowGram eigenrotation, sequential retirement. Static minimizes squaredcrossnorm; current/initial/random/exacterror minimizes localdrop action norm. Crosses zeroed, positive diagonal retained, denseactivecore. Exacterror=A^-1 currentr is free informationoracle. No global optimality: successive dropped terms interact. Perlayer vectorized apply, compact logical40B rotations; denseconstruction/referencework free. Same4plainCGprefix, literal1e-8,max2000,2CPUbranch; no native memory/runtime or novelty claim.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(path.read_text());assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha256(datafile.read_bytes()).hexdigest()==prior['data_sha256']
   with np.load(datafile) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   perm,_=ordering.partition(X);X=X[perm];b=b[perm];n=len(b);A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A.flat[::n+1]+=1e-5;m,plan=planner(n);x4,prefix=solve(A,b,lambda r:r,prefix=True);assert prefix['status']=='prefix';r4=b-A@x4;L=cholesky(A,lower=True);error=cho_solve((L,True),r4);del L;rng=np.random.default_rng(622142);targets=dict(static=r4,current=r4,initial=b,random=rng.normal(size=n),oracle_error=error);case=dict(dataset=name,data_sha256=prior['data_sha256'],planner=plan,prefix=prefix,runs=[],strong_controls=[r for r in prior['runs'] if r['method'] in ['uniform600','packed24','original_exact416']]);result['cases'].append(case);currentH=None
   for mode in ['static','current','initial','random','oracle_error','current_warm','plain_restart','plain_uninterrupted']:
    metadata=None;warm=0;pref=4;x0=x4
    if mode in targets:
     H,metadata,_=model(A,targets[mode],m,'static' if mode=='static' else 'action')
     if mode=='current':currentH=H
    else:
     H=lambda r:r
     if mode=='plain_uninterrupted':x0=None;pref=0
     if mode=='current_warm':z=currentH(r4);az=A@z;x0=x4+z*float(r4@z)/float(z@az);warm=1
    _,row=solve(A,b,H,x0);row.update(method=mode,construction=metadata,prefix_products=pref,warm_products=warm,optimistic_operator_products=pref+warm+row['iterations']+int(x0 is not None)+1)
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    case['runs'].append(row);print(name,mode,row['status'],row['iterations'],row['original_relative_residual'],flush=True);save()
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
