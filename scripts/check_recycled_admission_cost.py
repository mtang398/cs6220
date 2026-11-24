"""Setup-inclusive cached-K cost of recycled residual-sketch column admission."""
import json,resource,gc,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import qr,solve_triangular
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_recycled_residual_sketch_admission as helper
import check_cached_kernel_static_frontier as static
import check_refresh_regularization_scope as scope
ROOT=scope.ROOT;METHODS=['current','gaussian','nystrom96','nystrom128','nystrom512','direct'];LIMIT=128*1024**2

def sha(p):return sha256(p.read_bytes()).hexdigest()
def candidate(X,b,bw,mode):
 start=perf_counter();cpu=process_time();n=len(b);nb=np.linalg.norm(b);times={};products=0;verifications=0;replacements=0;sketch_rhs=0
 ledger=dict(cached_K=8*n*n,construction_factors_SVD_work=8*n*96*10+8*96*96*10,prefix_images_Q_sketch=8*n*8*7,work_vectors=8*n*20,reserve=65536);ledger['sum_bound']=sum(ledger.values());assert ledger['sum_bound']<=LIMIT
 t=perf_counter();K=cdist(X,X,'sqeuclidean');K*=(-.5/(bw*bw));np.exp(K,out=K);times['kernel']=perf_counter()-t
 def mv(v):
  nonlocal products
  products+=1;return K@v+1e-5*v
 def verify(x):
  nonlocal verifications
  verifications+=1;r=b-mv(x);return r,float(np.linalg.norm(r)/nb)
 t=perf_counter();initial=helper.fps(K,64);F,indices,_,ok=helper.append(K,np.empty((n,0)),[],initial);assert ok;M=helper.preconditioner(F);times['initial_selection_factor']=perf_counter()-t
 x=np.zeros(n);r=b.copy();z=M.apply(r);p=z.copy();rho=float(r@z);P=[];AP=[];status='maxiter';prefix_steps=0
 t=perf_counter()
 for k in range(8):
  if process_time()-cpu>2:status='cpu_cap';break
  if np.linalg.norm(r)<=1e-8*nb:
   r,relative=verify(x)
   if relative<=1e-8:status='converged_prefix';break
   z=M.apply(r);p=z.copy();rho=float(r@z);replacements+=1
  ap=mv(p);den=float(p@ap)
  if den<=0 or rho<=0:status='nonpositive_curvature';break
  P.append(p.copy());AP.append(ap.copy());alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=M.apply(r);new=float(r@z);p=z+new/rho*p;rho=new;prefix_steps+=1
 times['prefix']=perf_counter()-t;rounds=[];condition=None;rank=64;continuation_steps=0
 if prefix_steps==8:
  t=perf_counter()
  if mode=='current':
   PP=np.column_stack(P);AAP=np.column_stack(AP);norms=np.linalg.norm(PP,axis=0);PP/=norms;AAP/=norms;Q,R=qr(PP,mode='economic');condition=float(np.linalg.cond(R));KQ=solve_triangular(R.T,(AAP-1e-5*PP).T,lower=True,check_finite=False).T
  else:
   Q,_=qr(np.random.default_rng(622045).normal(size=(n,8)),mode='economic');KQ=K@Q;sketch_rhs=8
  FF,indices,rounds,valid,reason=helper.recycle(K,F,indices,Q,KQ)
  if valid:M=helper.preconditioner(FF);rank=96
  else:status='invalid_sketch_'+str(reason)
  times['sketch_admission_factor']=perf_counter()-t
  if valid:
   t=perf_counter();r,relative=verify(x);z=M.apply(r);p=z.copy();rho=float(r@z)
   for k in range(2001):
    continuation_steps=k
    if np.linalg.norm(r)<=1e-8*nb:
     rtrue,relative=verify(x)
     if relative<=1e-8:status='converged';break
     r=rtrue;z=M.apply(r);p=z.copy();rho=float(r@z);replacements+=1
    if process_time()-cpu>2:status='cpu_cap';break
    if k==2000:break
    ap=mv(p);den=float(p@ap)
    if den<=0 or rho<=0:status='nonpositive_curvature';break
    alpha=rho/den;x+=alpha*p;r-=alpha*ap;z=M.apply(r);new=float(r@z);p=z+new/rho*p;rho=new
   times['continuation']=perf_counter()-t
 t=perf_counter();_,relative=verify(x);times['final_verification']=perf_counter()-t
 if relative<=1e-8 and not status.startswith('invalid'):status='converged'
 if process_time()-cpu>2:status='cpu_cap'
 return dict(method=mode,status=status,converged=status=='converged',original_relative_residual=relative,total_seconds=perf_counter()-start,total_cpu_seconds=process_time()-cpu,stage_seconds=times,prefix_iterations=prefix_steps,continuation_iterations=continuation_steps,iterations=prefix_steps+continuation_steps,operator_products=products,independent_sketch_rhs_products=sketch_rhs,independent_sketch_kernel_batches=int(sketch_rhs>0),original_verifications=verifications,residual_replacements=replacements,final_rank=rank,normalized_prefix_R_condition=condition,admission_rounds=rounds,byte_ledger=ledger)

def main():
 out=ROOT/'results/recycled-admission-cost.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
 paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']];files=scope.source_paths()+[Path(helper.__file__).resolve(),Path(static.__file__).resolve(),Path(__file__).resolve(),ROOT/'src/nystrom_project/preconditioners.py',ROOT/'src/nystrom_project/numerics.py']+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol='Independent cached K per solve, all construction/setup/prefix/QR/SVD/recycled admissions timed. Paid8direction prefix for both current/Gaussian; Gaussian adds K@Q width8. No directKQ validation in timing. Recursive stopping with literal residual acceptance/restart; CPU2 includes setup, max2000 continuation. StaticFPS96/128/512 and direct use existing cached-frontier run. Two reversed orders. Common128MiB diagnostic ceiling, not processRSS/scope revision. Separate largest current/Gaussian traces, allcase conservative byteledger. Original staged64→96 mechanism unchanged.',cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for caseindex,(name,path) in enumerate(zip(['smooth','mixed','Concrete','Airfoil'],paths)):
   prior=json.loads(path.read_text());assert all(sha(ROOT/p)==h for p,h in prior['sources'].items());datafile=ROOT/'data/prepared'/scope.DATASETS[name];assert sha(datafile)==prior['data_sha256']
   with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   X.setflags(write=False);b.setflags(write=False);hashes=dict(X=sha256(X.tobytes()).hexdigest(),b=sha256(b.tobytes()).hexdigest());case=dict(dataset=name,data_sha256=prior['data_sha256'],input_hashes=hashes,audits=[],runs=[]);result['cases'].append(case)
   if name=='smooth':
    for mode in ['current','gaussian']:
     gc.collect();tracemalloc.start()
     try:row=candidate(X,b,bw,mode);_,peak=tracemalloc.get_traced_memory()
     finally:tracemalloc.stop()
     case['audits'].append(dict(method=mode,traced_peak_bytes=peak,within_ceiling=peak<=LIMIT,run=row));save();assert peak<=LIMIT
   order=METHODS[caseindex:]+METHODS[:caseindex]
   for repeat in range(2):
    for mode in (order if repeat==0 else list(reversed(order))):
     row=candidate(X,b,bw,mode) if mode in ['current','gaussian'] else static.run(X,b,bw,1e-5,mode);row['repeat']=repeat
     if row['converged']:assert row['original_relative_residual']<=1e-8
     case['runs'].append(row);print(name,mode,repeat,row['status'],row['iterations'],row['total_seconds'],flush=True);save()
   case['input_hashes_unchanged']=hashes==dict(X=sha256(X.tobytes()).hexdigest(),b=sha256(b.tobytes()).hexdigest());assert case['input_hashes_unchanged']
 result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
