"""Preconditioned-residual aggregation diagnostic; dense work is free."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import eigh,qr,cho_factor,cho_solve
from threadpoolctl import threadpool_limits
import check_pair_rotation_schur as pair
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget


def partition(X,count):
 def split(indices,k):
  if k==1:return [indices]
  axis=int(np.argmax(np.ptp(X[indices],axis=0)));order=indices[np.argsort(X[indices,axis],kind='stable')]
  left=k//2;mid=len(indices)*left//k
  return split(order[:mid],left)+split(order[mid:],k-left)
 return split(np.arange(len(X)),count)

class Rotation:
 def __init__(self,groups,contrasts):
  self.groups=groups;self.contrasts=contrasts;self.n=sum(map(len,groups));self.m=2*len(groups);self.bases=[]
  for group,v in zip(groups,contrasts):
   u=np.ones(len(group))/np.sqrt(len(group));Q,_=qr(np.column_stack([u,v]),mode='full')
   if Q[:,0]@u<0:Q[:,0]*=-1
   if Q[:,1]@v<0:Q[:,1]*=-1
   self.bases.append(Q)
  self.permutation=np.r_[np.concatenate([g[:2] for g in groups]),np.concatenate([g[2:] for g in groups])]
  assert len(np.unique(self.permutation))==self.n
 def to_rotated(self,v):
  out=np.empty_like(v)
  for g,Q in zip(self.groups,self.bases):out[g]=Q.T@v[g]
  return out[self.permutation]
 def to_original(self,v):
  local=np.empty_like(v);local[self.permutation]=v;out=np.empty_like(v)
  for g,Q in zip(self.groups,self.bases):out[g]=Q@local[g]
  return out
 def matrix(self,A):
  out=A.copy()
  for g,Q in zip(self.groups,self.bases):out[:,g]=out[:,g]@Q
  for g,Q in zip(self.groups,self.bases):out[g,:]=Q.T@out[g,:]
  return out[np.ix_(self.permutation,self.permutation)]

def initial(K,groups):
 vs=[]
 for g in groups:
  u=np.ones(len(g))/np.sqrt(len(g));P=np.eye(len(g))-np.outer(u,u);G=P@K[np.ix_(g,g)]@P
  _,V=eigh((G+G.T)*.5);v=V[:,-1];v-=u*(u@v)
  if np.linalg.norm(v)<1e-12:v=P[:,0]
  v/=np.linalg.norm(v)
  if v[np.argmax(abs(v))]<0:v*=-1
  vs.append(v)
 return Rotation(groups,vs)

def adapt(old,direction):
 vs=[];fallback=0
 for g,v in zip(old.groups,old.contrasts):
  u=np.ones(len(g))/np.sqrt(len(g));local=direction[g];centered=local-u*(u@local)
  norm=np.linalg.norm(centered)
  if norm<=64*np.finfo(float).eps*max(np.linalg.norm(local),np.finfo(float).tiny):vs.append(v.copy());fallback+=1
  else:vs.append(centered/norm)
 return Rotation(old.groups,vs),fallback

def prepare(A,b,rotation):
 Ar=rotation.matrix(A);br=rotation.to_rotated(b)
 return Ar,br,pair.previous.build(Ar,br,np.arange(rotation.m))

def main():
 out=ROOT/'results/preconditioned-aggregation-schur.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
 files=[Path(__file__).resolve(),Path(pair.__file__).resolve(),Path(pair.previous.__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(
  question='Does applying the old block inverse before local aggregation improve the direction relative to raw residual?' ,
  groups=200,coarse=400,checkpoint=10,grouping='Balanced recursive longest-feature-axis splits; every row belongs to exactly one group.',
  static='Keep constant and dominant centered local-kernel eigenvector in each group; full local QR defines fine coordinates.',
  learned='Replace contrast by locally centered old block-inverse times current original residual; lambda times identity is the fine Schur approximation. Preserve means.',
  controls=['static uninterrupted','static restart10','current raw residual'],
  scope='Dense rotation/exactcoarsefactor/construction/refactor grantedfree. Full original1e-8 checked everyiterate. No native/querycost orfullmemoryclaim. Exacterrororacle is coveragecheck, not learnedmethod evidence.'),tiny_check=None,cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
  temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');temp.replace(out)
 save()
 with threadpool_limits(limits=1):
  rng=np.random.default_rng(622088);F=rng.normal(size=(31,31));A=F@F.T+np.eye(31);b=rng.normal(size=31);groups=partition(rng.normal(size=(31,3)),5)
  old=initial(A,groups);exact=cho_solve(cho_factor(A,lower=True),b);x0=rng.normal(size=31);changed,_=adapt(old,exact-x0)
  Q=np.column_stack([changed.to_original(np.eye(31)[:,j]) for j in range(31)])
  Ar,br,op=prepare(A,b,changed);row,_,_=pair.solve(A,b,changed,Ar,br,op,x0=x0)
  orth=float(np.linalg.norm(Q.T@Q-np.eye(31)));transform=float(np.linalg.norm(Ar-Q.T@A@Q)/np.linalg.norm(A));fine=float(np.linalg.norm(changed.to_rotated(exact-x0)[changed.m:])/np.linalg.norm(exact-x0))
  assert max(orth,transform,fine)<1e-12 and row['converged'] and row['iterations']==0
  R,F,B,V,D,rhs=op;rr=changed.to_rotated(b);u=cho_solve(F,rr[:changed.m]);v=(rr[R]-B.T@u)/.3
  zz=np.r_[u-V@v,v];M=Ar.copy();M[np.ix_(R,R)]=B.T@V+.3*np.eye(len(R))
  inverse_error=float(np.linalg.norm(M@zz-rr)/np.linalg.norm(rr));assert inverse_error<1e-10
  rec['tiny_check']=dict(block_inverse_relative_residual=inverse_error,orthogonality=orth,transformed_matrix_error=transform,exact_error_fine_fraction=fine,oracle_fine_iterations=row['iterations']);save()
  cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
  for name,filename in cases:
   if process_time()-cpu>68:rec['status']='cpu_budget';save();return
   path=ROOT/'data/prepared'/filename
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   K=SharedKernel(X,bw,Budget([X,b])).block(X,X);A=K.copy();A.flat[::len(b)+1]+=1e-5
   groups=partition(X,200);old=initial(K,groups);Ar,br,op=prepare(A,b,old)
   baseline,saved,_=pair.solve(A,b,old,Ar,br,op,capture_prefix=True,deadline=min(cpu+77,process_time()+4))
   case=dict(dataset=name,n=len(b),data_sha256=sha256(path.read_bytes()).hexdigest(),group_sizes=[len(g) for g in groups],baseline=baseline,branches=[])
   rec['cases'].append(case);save()
   if 10 not in saved:case['branch_status']='baseline_ended_before_checkpoint';save();continue
   x=saved[10];reference=cho_solve(cho_factor(A,lower=True),b)
   r=b-A@x;rr=old.to_rotated(r);R,F,B,V,D,rhs=op
   u=cho_solve(F,rr[:old.m]);v=(rr[R]-B.T@u)/1e-5
   zr=np.empty_like(r);zr[:old.m]=u-V@v;zr[R]=v;z=old.to_original(zr)
   directions={'current_residual':r,'preconditioned_residual':z}
   case['direction_metrics']={name:dict(cosine_with_error=float(direction@(reference-x)/(np.linalg.norm(direction)*np.linalg.norm(reference-x)))) for name,direction in directions.items()}
   for mode in ['unchanged_restart','current_residual','preconditioned_residual']:
    rot,fallback=(old,0) if mode=='unchanged_restart' else adapt(old,directions[mode])
    transformed,rb,newop=(Ar,br,op) if mode=='unchanged_restart' else prepare(A,b,rot)
    row,_,_=pair.solve(A,b,rot,transformed,rb,newop,x0=None if mode=='initial_target' else x,deadline=min(cpu+77,process_time()+4))
    row.update(method=mode,prefix_plus_iterations=row['iterations']+(0 if mode=='initial_target' else 10),fallback_groups=fallback,
               exact_remaining_error_fine_fraction=float(np.linalg.norm(rot.to_rotated(reference-x)[rot.m:])/np.linalg.norm(reference-x)))
    case['branches'].append(row);save();print(json.dumps(dict(dataset=name,method=mode,status=row['status'],total_iterations=row['prefix_plus_iterations'])),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
