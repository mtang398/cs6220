"""Static time-to-accuracy frontier with independently cached K per solve."""
import argparse,json,resource,sys,tracemalloc,gc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.preconditioners import LowRankPreconditioner
METHODS=['direct','plain','nystrom32','nystrom128','nystrom512'];LIMIT=128*1024**2

def fps(K,m):
 similarity=np.zeros(len(K));chosen=np.empty(m,dtype=np.int64);j=0
 for k in range(m):chosen[k]=j;np.maximum(similarity,K[:,j],out=similarity);similarity[chosen[:k+1]]=np.inf;j=int(np.argmin(similarity))
 return chosen

def run(X,b,bw,lam,method):
 t=perf_counter();cpu=process_time();n=len(b);rank=min(int(method[7:]),n-1) if method.startswith('nystrom') else 0
 ledger=dict(cached_kernel=8*n*n,rank=rank,conservative_setup_workspace=(32*n*rank+48*rank*rank+120*n if rank else (8*n*n+120*n if method=='direct' else 120*n)),reserve=65536);ledger['sum_bound']=ledger['cached_kernel']+ledger['conservative_setup_workspace']+ledger['reserve'];assert ledger['sum_bound']<=LIMIT
 K=cdist(X,X,'sqeuclidean');K*=(-.5/(bw*bw));np.exp(K,out=K);kernel_seconds=perf_counter()-t;products=0;verifications=0;iterations=0;replacements=0;effective_rank=None
 def mv(v):
  nonlocal products
  products+=1;return K@v+lam*v
 def verify(x):
  nonlocal verifications
  verifications+=1;rr=b-mv(x);return rr,float(np.linalg.norm(rr)/np.linalg.norm(b))
 selection_seconds=0.;precondition_seconds=0.;status='maxiter';x=np.zeros(n)
 try:
  if method=='direct':
   start=perf_counter();Ac=np.array(K,order='F',copy=True);Ac.flat[::n+1]+=lam;L=cholesky(Ac,lower=True,overwrite_a=True,check_finite=False);precondition_seconds=perf_counter()-start;x=cho_solve((L,True),b,check_finite=False);r,rel=verify(x);status='converged' if rel<=1e-8 else 'accuracy_failure';setup=kernel_seconds+precondition_seconds
  else:
   if rank:
    start=perf_counter();indices=fps(K,rank);selection_seconds=perf_counter()-start;start=perf_counter();images=K[:,indices];core=K[np.ix_(indices,indices)];pre=LowRankPreconditioner.from_sketch(images,core,lam);effective_rank=pre.rank;del images,core;precondition_seconds=perf_counter()-start;H=pre.apply
   else:H=lambda v:v
   setup=perf_counter()-t;r=b.copy();z=H(r);p=z.copy();rho=float(r@z);normb=np.linalg.norm(b)
   for iterations in range(2001):
    if np.linalg.norm(r)<=1e-8*normb:
     true,rel=verify(x)
     if rel<=1e-8:status='converged';break
     r=true;z=H(r);p=z.copy();rho=float(r@z);replacements+=1
    if process_time()-cpu>2:status='cpu_cap';break
    if iterations==2000:break
    ap=mv(p);den=float(p@ap)
    if den<=0 or rho<=0:status='nonpositive_curvature';break
    alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=H(r);new=float(r@z);p=z+new/rho*p;rho=new
   _,rel=verify(x)
   if rel<=1e-8:status='converged'
  if process_time()-cpu>2:status='cpu_cap'
 except (np.linalg.LinAlgError,MemoryError) as exc:
  status='numerical_failure';rel=None;setup=perf_counter()-t
 return dict(method=method,status=status,converged=status=='converged',n=n,requested_rank=rank,effective_rank=effective_rank,iterations=iterations,residual_replacements=replacements,original_relative_residual=rel,total_seconds=perf_counter()-t,total_cpu_seconds=process_time()-cpu,kernel_setup_seconds=kernel_seconds,selection_seconds=selection_seconds,precondition_setup_seconds=precondition_seconds,setup_seconds=setup,operator_products=products,original_verifications=verifications,byte_ledger=ledger)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--lam',required=True,type=float);args=parser.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(20,22));t=perf_counter();cpu=process_time()
 names={'Concrete':'concrete-observed-s0-lambda1e-05.npz','Airfoil':'airfoil-observed-s0-lambda1e-05.npz'}
 if args.dataset in names:filename=names[args.dataset]
 else:
  n,name=args.dataset.split('-');filename=f'synthetic-n{640 if n=="512" else 2560}-{name}-s0-lambda0.001.npz'
 path=ROOT/'data/prepared'/filename
 with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
 X.setflags(write=False);b.setflags(write=False);hashes=dict(X=sha256(X.tobytes()).hexdigest(),b=sha256(b.tobytes()).hexdigest());folder=ROOT/'results/cached-kernel-static-frontier';folder.mkdir(exist_ok=True);out=folder/f'{args.dataset}-lambda{args.lam:g}.json';assert not out.exists()
 files=[Path(__file__).resolve(),ROOT/'src/nystrom_project/preconditioners.py',ROOT/'src/nystrom_project/numerics.py'];result=dict(status='running',dataset=args.dataset,regularization=args.lam,n=len(b),data_sha256=sha256(path.read_bytes()).hexdigest(),sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol='Each solve independently constructs and retains dense Gaussian K, charged in setup. Common128MiB algorithm-owned ceiling, readonly inputs excluded, not process RSS. Deterministic cached FPS start0; Nyström N+lambdaI stable existing LowRankPreconditioner. Rank512 clamps to n−1. Native CPU2/solve; failures not runtime wins. Recursive stopping with paid true verification and restart when required. Two reversed method orders; only largest smooth2048 hardcase has separate all-method wholetrace audits. No hidden K reuse.',runs=[],audits=[])
 def save():result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  if args.dataset=='2048-smooth' and args.lam==1e-5:
   for method in METHODS:
    gc.collect();tracemalloc.start()
    try:row=run(X,b,bw,args.lam,method);_,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    result['audits'].append(dict(method=method,traced_peak_bytes=peak,within_ceiling=peak<=LIMIT,run=row));save();assert peak<=LIMIT
  order=METHODS if args.lam!=.001 else METHODS[2:]+METHODS[:2]
  for repeat in range(2):
   for method in (order if repeat==0 else list(reversed(order))):
    row=run(X,b,bw,args.lam,method);row['repeat']=repeat;result['runs'].append(row);save();print(method,repeat,row['status'],row['iterations'],row['total_seconds'],flush=True)
 result['sources_unchanged']=all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items());result['inputs_unchanged']=hashes==dict(X=sha256(X.tobytes()).hexdigest(),b=sha256(b.tobytes()).hexdigest());assert result['sources_unchanged'] and result['inputs_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
