"""New additive multilevel capacity/acquisition gate; dense oracle construction."""
import json,resource,math
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve,eigh
from threadpoolctl import threadpool_limits
from check_full_aggregation_schur import partition
import check_refresh_regularization_scope as scope
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def sha(p):return sha256(p.read_bytes()).hexdigest()
def blocks(A,groups):return [(g,cholesky(A[np.ix_(g,g)],lower=True)) for g in groups]
def local_apply(factors,v):
 z=np.empty_like(v)
 for g,L in factors:z[g]=cho_solve((L,True),v[g],check_finite=False)
 return z

def basis(groups,V,n):
 P=np.zeros((n,sum(min(8,len(g)) for g in groups)));offset=0;payload=0;coarsegroups=[]
 for g in groups:
  width=min(8,len(g));Q=np.linalg.qr(V[g],mode='reduced')[0][:,:width];P[np.ix_(g,np.arange(offset,offset+width))]=Q;coarsegroups.append(np.arange(offset,offset+width));offset+=width;payload+=8*len(g)*width
 return P,payload,coarsegroups

def hierarchy(A,groups,V):
 n=len(A);P1,p1bytes,leafcoarse=basis(groups,V,n);A1=P1.T@A@P1
 midgroups=[np.concatenate(leafcoarse[i:i+4]) for i in range(0,len(leafcoarse),4)]
 f1=blocks(A1,midgroups);V1=P1.T@V;P2,p2bytes,_=basis(midgroups,V1,len(A1));A2=P2.T@A1@P2;L2=cholesky((A2+A2.T)*.5,lower=True)
 f0=blocks(A,groups)
 def action(v):
  coarse=P1.T@v;return local_apply(f0,v)+P1@(local_apply(f1,coarse)+P2@cho_solve((L2,True),P2.T@coarse,check_finite=False))
 parts=dict(leaf_packed_factors=sum(4*len(g)*(len(g)+1) for g in groups),middle_packed_factors=sum(4*len(g)*(len(g)+1) for g in midgroups),top_packed_factor=4*len(A2)*(len(A2)+1),P1_local_values=p1bytes,P2_local_values=p2bytes,permutation=8*n,group_offsets=8*(len(groups)+len(midgroups)+4),eight_fine_work_vectors=64*n,four_middle_work_vectors=32*len(A1),four_top_work_vectors=32*len(A2),kernel_tile=65536,reserve=65536)
 return action,dict(parts=parts,total_bytes=sum(parts.values()),within_1MiB=sum(parts.values())<=1048576,coarse1=len(A1),coarse2=len(A2),qualification='Packed factor/local basis component planner only. Dense construction, candidate storage during assembly and native peak allocations unresolved.')

def pcg(A,b,M,x0=None,cap=4,prefix=False):
 n=len(b);x=np.zeros(n) if x0 is None else x0.copy();r=b-A@x;z=M(r);p=z.copy();rho=float(r@z);normb=np.linalg.norm(b);start=process_time();history=[];status='maxiter'
 for k in range(2001):
  rel=float(np.linalg.norm(b-A@x)/normb)
  if rel<=1e-8:status='converged';break
  if prefix and k==4:status='prefix';break
  if process_time()-start>cap:status='cpu_cap';break
  if k==2000:break
  ap=A@p;den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  x+=(rho/den)*p;r-=(rho/den)*ap;z=M(r)
  if prefix:history.append(z.copy())
  new=float(r@z);p=z+(new/rho)*p;rho=new
 return x,dict(status=status,iterations=k,original_relative_residual=rel),history

def tiny():
 rng=np.random.default_rng(62);F=rng.normal(size=(48,48));A=F@F.T+np.eye(48);groups=[np.arange(i,min(i+16,48)) for i in range(0,48,16)];V=np.column_stack((np.ones(48),rng.normal(size=(48,7))));M,_=hierarchy(A,groups,V);H=M(np.eye(48));err=float(np.linalg.norm(H-H.T)/np.linalg.norm(H));ev=float(np.linalg.eigvalsh((H+H.T)*.5)[0]);assert err<1e-12 and ev>0;return dict(relative_symmetry_error=err,smallest_eigenvalue=ev)

def main():
 out=ROOT/'results/bootstrap-multilevel.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time()
 paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(__file__).resolve()]+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol=dict(architecture='Leaf32 block inverse + P1 (group-of-four-leaves inverse + P2 top inverse P2T) P1T. Local QR width8. Entire principal-core architecture replaced.',candidates='Constant plus7 vectors. Current uses last3 B0-preconditioned residuals after common4-step prefix + last4 of fixed7 Gaussian vectors; random uses all7 same Gaussian vectors. Filter applies3 steps I−omega B0 A to7 candidates, constant unfiltered. P2 restricts the same8 candidate vectors through P1T.',omega='Free exact largest eigenvalue of generalized(A,blockdiag(Aleaf)); omega=1/largest, identical branches. Not a paid production estimator.',cost='Optimistic kernel passes: leaf setup, prefix4, filter3 width7 passes, one full kernel assembly pass, postrestart scalariterations, reconstruction residual andfinal verification charged. Dense eig/assembly arithmetic/free periteration true checks excluded; RHS work counted separately. No native CPU/memory claim.',limits='CPU4 perbranch,80/85 process,external120wall/flock,BLAS1.'),tiny=tiny(),cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   prior=json.loads(path.read_text());assert all(sha(ROOT/p)==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha(datafile)==prior['data_sha256']
   with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   n=len(b);kernel=SharedKernel(X,bw,Budget([X,b]));A=kernel.block(X,X);A.flat[::n+1]+=1e-5;groups=partition(X,math.ceil(n/32));f0=blocks(A,groups);M0=lambda v:local_apply(f0,v)
   x4,prefix,history=pcg(A,b,M0,prefix=True);assert prefix['status']=='prefix' and len(history)==4
   diagonal=np.zeros_like(A)
   for g in groups:diagonal[np.ix_(g,g)]=A[np.ix_(g,g)]
   largest=float(eigh(A,diagonal,subset_by_index=[n-1,n-1],eigvals_only=True,check_finite=False)[0]);omega=1/largest
   rng=np.random.default_rng(622089);random=rng.normal(size=(n,7));current=np.column_stack(history[-3:]+[random[:,i] for i in range(3,7)])
   case=dict(dataset=name,data_sha256=prior['data_sha256'],prefix=prefix,largest_generalized_eigenvalue=largest,omega=omega,strong_static_controls=[r for r in prior['runs'] if r['method'] in ['uniform600','packed24','original_exact416']],runs=[]);result['cases'].append(case)
   for mode in ['B0_restart','current_unfiltered','current_filtered','random_filtered']:
    filters=0;planner=None
    if mode=='B0_restart':M=M0;assembly=0
    else:
     V=(random if mode=='random_filtered' else current).copy()
     if mode.endswith('filtered') and mode!='current_unfiltered':
      for step in range(3):V-=omega*M0(A@V)
      filters=3
     V/=np.maximum(np.linalg.norm(V,axis=0),1e-300);V=np.column_stack((np.ones(n)/np.sqrt(n),V));M,planner=hierarchy(A,groups,V);assembly=1
    _,row,_=pcg(A,b,M,x4);row.update(method=mode,planner=planner)
    counts=dict(leaf_setup=sum(len(g)**2 for g in groups),prefix=4*n*n,filter=filters*n*n,assembly=assembly*n*n,restart_residual=n*n,iterations=row['iterations']*n*n,final_verification=n*n)
    row.update(kernel_entry_components=counts,analytical_entries=sum(counts.values()),filter_scalar_rhs_products=7*filters,total_scalar_rhs_products=4+7*filters+assembly+row['iterations']+2)
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    case['runs'].append(row);print(name,mode,row['status'],row['iterations'],row['analytical_entries'],None if planner is None else planner['total_bytes'],flush=True);save()
 result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
