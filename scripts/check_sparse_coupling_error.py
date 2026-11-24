"""Fixed-eight-entry sparse coupling capacity gate; free dense acquisition."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve,svd,solve_triangular
from threadpoolctl import threadpool_limits
import check_cached_coupling_error as priorcode
ROOT=priorcode.ROOT

def omp(C,F,steps=8):
 m,t=F.shape;support=np.empty((t,steps),dtype=np.uint16);coeff=np.zeros((t,steps));correlation=F.copy();used=np.zeros((m,t),dtype=bool);columns=np.arange(t);diagonal=np.sqrt(np.diag(C))
 for k in range(steps):
  scores=np.abs(correlation)/diagonal[:,None];scores[used]=-1;chosen=np.argmax(scores,axis=0);support[:,k]=chosen;used[chosen,columns]=True
  ids=support[:,:k+1].astype(np.intp);small=C[ids[:,:,None],ids[:,None,:]];rhs=F[ids,columns[:,None]]
  values=np.linalg.solve(small,rhs[...,None])[...,0];coeff[:,:k+1]=values
  correlation=F-np.einsum('mtk,tk->mt',C[:,ids],values)
 D=np.zeros_like(F);D[support.astype(np.intp),columns[:,None]]=coeff
 residual=F-C@D;selected=residual[support.astype(np.intp),columns[:,None]]
 return D,support,coeff,float(np.linalg.norm(selected)/max(np.linalg.norm(F),1e-300))
def tiny():
 rng=np.random.default_rng(512);L=np.tril(rng.normal(size=(11,11)));L.flat[::12]+=5;C=L@L.T;delta=rng.normal(size=(11,4));F=C@delta;D,ids,val,orth=omp(C,F,3)
 lhs=float(np.sum((L.T@(delta-D))**2));rhs=float(np.sum((L.T@delta)**2)-np.sum(D*F));relative=abs(lhs-rhs)/max(abs(lhs),1);assert relative<1e-12 and orth<1e-12
 return dict(weighted_projection_identity_relative_error=relative,selected_correlation_relative_norm=orth)
def digest(p):return sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'results/sparse-coupling-error.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time()
 controlpath=ROOT/'results/int16-core-capacity.json';control=json.loads(controlpath.read_text());paths=[ROOT/f'results/refresh-scope/{n}-lambda1e-05.json' for n in ['smooth','mixed','concrete','airfoil']]
 files=[Path(__file__).resolve(),Path(priorcode.__file__).resolve(),ROOT/'scripts/check_int16_core_capacity.py',controlpath]+paths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):digest(p) for p in files},tiny=tiny(),protocol=dict(support='Exactly8 entries per tail column, distinct core row indices uint16, coefficientsFP64; fixed-width arrays no pointers.',payload='postquantized int16 factor m²+63m−152 plus80(n−m)<=721200; largest m<n. Payload only, not total working-memory proof.',acquisition='Dense C,F=B−CHB, eight C-energy normalized OMP steps and all refits free. Exact coupling/SVD/decodedfactor/free original residual stopping oracle. Not native timing or adaptive-method evidence.',cost='Existing exact-congruence run counts n²/kernel action, setup/boundaries/final verification; sparse arithmetic/acquisition excluded.',controls='Samecore unchanged, sparseOMP, exactcoupling, weightedrank8 with separately disclosed payload. Original strong static controls retained. No parameter sweep.'),cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,path in zip(['smooth','mixed','Concrete','Airfoil'],paths):
   ref=json.loads(path.read_text());assert all(digest(ROOT/p)==h for p,h in ref['sources'].items())
   filenames={'smooth':'synthetic-n2560-smooth-s0-lambda0.001.npz','mixed':'synthetic-n2560-mixed-s0-lambda0.001.npz','Concrete':'concrete-observed-s0-lambda1e-05.npz','Airfoil':'airfoil-observed-s0-lambda1e-05.npz'};datafile=ROOT/'data/prepared'/filenames[name];assert digest(datafile)==ref['data_sha256']
   with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   n=len(b);m=max(k for k in range(8,n) if k*k+63*k-152+80*(n-k)<=721200);assert m<65536
   kernel=priorcode.SharedKernel(X,bw,priorcode.Budget([X,b]));S=kernel.farthest(m);R=np.setdiff1d(np.arange(n),S);A=kernel.block(X,X);A.flat[::n+1]+=1e-5;C=A[np.ix_(S,S)];B=A[np.ix_(S,R)];L=cholesky(C,lower=True);Q=priorcode.quantize(L,'int16');base=cho_solve((Q,True),B,check_finite=False);truth=cho_solve((L,True),B,check_finite=False);delta=truth-base;F=B-C@base
   D,indices,coeff,orth=omp(C,F);norm=float(np.linalg.norm(L.T@delta));weightederror=float(np.linalg.norm(L.T@(delta-D))/norm)
   U,s,V=svd(L.T@delta,full_matrices=False,check_finite=False);rank8=solve_triangular(L.T,U[:,:8]*s[:8],lower=False,check_finite=False)@V[:8]
   factorbytes=m*m+63*m-152;sparsebytes=indices.nbytes+coeff.nbytes;assert sparsebytes==80*(n-m) and factorbytes+sparsebytes<=721200
   case=dict(dataset=name,data_sha256=ref['data_sha256'],core=m,factor_payload_bytes=factorbytes,sparse_payload_bytes=sparsebytes,total_payload_bytes=factorbytes+sparsebytes,weightedrank8_payload_bytes=factorbytes+64*n,weightedrank8_within_payload_budget=factorbytes+64*n<=721200,selected_correlation_relative_norm=orth,weighted_error_fraction=weightederror,weightedrank8_error_fraction=float(np.linalg.norm(L.T@(delta-rank8))/norm),strong_static_controls=[r for r in ref['runs'] if r['method'] in ['uniform600','packed24','original_exact416']],runs=[]);result['cases'].append(case)
   for mode,correction in [('unchanged',0.),('sparse8',D),('exact_coupling',delta),('weightedrank8',rank8)]:
    row=priorcode.run(A,b,S,Q,base+correction);row['method']=mode;case['runs'].append(row)
    if row['status']=='converged':assert row['original_relative_residual']<=1e-8
    print(name,m,mode,row['status'],row['iterations'],row['analytical_entries'],flush=True);save()
 result['sources_unchanged']=all(digest(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
