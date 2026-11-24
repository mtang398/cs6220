"""Free block-error oracle on exact Schur-CG trajectories; not native timing."""
import itertools,json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve,svd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]

def order(X,indices,depth=2):
 if depth==0:return [indices]
 axis=int(np.argmax(np.var(X[indices],axis=0)));ii=indices[np.argsort(X[indices,axis],kind='stable')];mid=len(ii)//2
 return order(X,ii[:mid],depth-1)+order(X,ii[mid:],depth-1)
def trajectory(K,b,lam,selected):
 n=len(b);mask=np.ones(n,bool);mask[selected]=False;tail=np.flatnonzero(mask)
 C=K[np.ix_(selected,selected)]+lam*np.eye(len(selected));B=K[np.ix_(selected,tail)];L=cho_factor(C,lower=True)
 S=K[np.ix_(tail,tail)]+lam*np.eye(len(tail))-B.T@cho_solve(L,B)
 rhs=b[tail]-B.T@cho_solve(L,b[selected]);x=np.zeros(len(tail));r=rhs.copy();p=r.copy();rho=r@r;steps=[]
 for it in range(2000):
  ap=S@p;alpha=rho/(p@ap);steps.append((p.copy(),float(alpha)));x+=alpha*p;r-=alpha*ap
  if np.linalg.norm(r)<=.5e-8*np.linalg.norm(b):break
  new=r@r;p=r+(new/rho)*p;rho=new
 full=np.empty(n);full[tail]=x;full[selected]=cho_solve(L,b[selected]-B@x)
 residual=float(np.linalg.norm(b-(K@full+lam*full))/np.linalg.norm(b));assert residual<1e-8
 return tail,steps,residual

def main():
 out=ROOT/'results/block-cache-oracle.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(100,110));t,c=perf_counter(),process_time()
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(core_candidate=320,core_control=416,tail_clusters=4,offdiagonal_pair_rank=6,rtol=1e-8,gap_allowance_fraction=.25,free='Dense kernels, full SVD, all candidate errors, exact directions, alpha and future step count; no setup/native-memory claim',selection='Enumerate all64 subsets of six symmetric offdiagonal pairs, maximize skipped exact entries while actual summed action error fits equal per-step gap allowance',resource_model='Candidate packedcore+SVD factors must fit within controlpackedcore; remaining application workspace not certified'),runs=[])
 with threadpool_limits(limits=1):
  for ds in ['smooth','mixed']:
   path=ROOT/'data/prepared'/f'synthetic-n2560-{ds}-s0-lambda0.001.npz'
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   n=len(b);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));selected=[0];distance=2-2*K[:,0]
   for _ in range(415):
    j=int(np.argmax(distance));selected.append(j);distance=np.minimum(distance,2-2*K[:,j])
   selected=np.asarray(selected);tail=np.setdiff1d(np.arange(n),selected[:320]);groups=order(X,tail);lookup=np.empty(n,int);lookup[tail]=np.arange(len(tail));blocks=[lookup[g] for g in groups]
   errors=[];entries=[];factorbytes=0
   for i,j in itertools.combinations(range(4),2):
    bi,bj=blocks[i],blocks[j];G=K[np.ix_(tail[bi],tail[bj])];U,s,V=svd(G,full_matrices=False,check_finite=False);rank=6
    errors.append((bi,bj,G-(U[:,:rank]*s[:rank])@V[:rank]));entries.append(2*len(bi)*len(bj));factorbytes+=8*rank*(len(bi)+len(bj)+1)
   candidate_bytes=8*320*321//2+factorbytes;control_bytes=8*416*417//2
   assert candidate_bytes<=control_bytes
   masks=np.asarray(list(itertools.product([0.,1.],repeat=6)));savings=masks@np.array(entries)
   for lam in [.001,1e-5]:
    _,steps,residual=trajectory(K,b,lam,selected[:320]);_,control,controlres=trajectory(K,b,lam,selected)
    allowance=.25e-8*np.linalg.norm(b)/len(steps);hist=[];total_skipped=0;error_sum=0
    for p,alpha in steps:
     E=np.zeros((6,len(tail)))
     for k,(bi,bj,G) in enumerate(errors):E[k,bi]=G@p[bj];E[k,bj]=G.T@p[bi]
     gram=E@E.T;norms=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',masks,gram,masks),0))
     eligible=np.flatnonzero(abs(alpha)*norms<=allowance);best=int(eligible[np.argmax(savings[eligible])]);skip=int(savings[best]);total_skipped+=skip;error_sum+=abs(alpha)*norms[best]
     hist.append(dict(approximate_pairs=int(masks[best].sum()),skipped_entries=skip,actual_error_increment=float(abs(alpha)*norms[best])))
    cand=len(steps)*(n*n-320*320)-total_skipped;ctrl=len(control)*(n*n-416*416)
    rec['runs'].append(dict(dataset=ds,regularization=lam,data_sha256=sha256(path.read_bytes()).hexdigest(),candidate_iterations=len(steps),control_iterations=len(control),true_relative_residual=residual,control_true_relative_residual=controlres,candidate_retained_factor_bytes=candidate_bytes,control_retained_factor_bytes=control_bytes,cache_factor_bytes=factorbytes,skipped_kernel_entries=total_skipped,candidate_kernel_entries=cand,control_kernel_entries=ctrl,kernel_entry_ratio=cand/ctrl,accumulated_action_error=float(error_sum),history=hist))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
