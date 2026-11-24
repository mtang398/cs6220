"""Free dense-Cholesky/HODLR-factor capacity screen; no native claim."""
import json,resource,heapq
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,svd
from threadpoolctl import threadpool_limits
from check_scaled_hss_headroom import partition,pcg,ROOT,SharedKernel,Budget


def blocks_and_leaf(L,levels):
    blocks=[]
    for groups in levels[:-1]:
        for a,b in groups:
            mid=(a+b)//2
            U,s,Vt=svd(L[mid:b,a:mid],full_matrices=False)
            blocks.append(dict(a=a,b=b,mid=mid,U=U,s=s,Vt=Vt,cost=8*(b-a)))
    fixed=dict(packed_leaf_triangles=8*sum((b-a)*(b-a+1)//2 for a,b in levels[-1]),
               permutation_five_vectors=48*len(L),kernel_workspace=65536,reserve=65536)
    return blocks,fixed


def ranks_for(blocks,fixed,mode):
    remaining=1048576-sum(fixed.values())
    if remaining<0:raise MemoryError('leaf triangles alone exceed allowance')
    if mode=='uniform':
        uniform=0
        while sum(t['cost']*min(uniform+1,len(t['s'])) for t in blocks)<=remaining:
            uniform+=1
            if all(uniform>=len(t['s']) for t in blocks):break
        return [min(uniform,len(t['s'])) for t in blocks]
    ranks=[0]*len(blocks);heap=[]
    for i,t in enumerate(blocks):heapq.heappush(heap,(-float(t['s'][0]**2/t['cost']),i))
    while heap:
        _,i=heapq.heappop(heap);t=blocks[i];k=ranks[i]
        if t['cost']>remaining:continue
        ranks[i]+=1;remaining-=t['cost']
        if k+1<len(t['s']):heapq.heappush(heap,(-float(t['s'][k+1]**2/t['cost']),i))
    return ranks


def compress(L,levels,blocks,ranks):
    T=np.zeros_like(L)
    for a,b in levels[-1]:T[a:b,a:b]=L[a:b,a:b]
    for t,r in zip(blocks,ranks):
        T[t['mid']:t['b'],t['a']:t['mid']]=(t['U'][:,:r]*t['s'][:r])@t['Vt'][:r,:]
    assert np.array_equal(np.diag(T),np.diag(L))
    return T


def main():
    out=ROOT/'results/hodlr-cholesky-headroom.json'
    if out.exists():raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
    files=[Path(__file__).resolve(),ROOT/'scripts/check_scaled_hss_headroom.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(
        mechanism='Compress off-diagonal sibling blocks of exact Cholesky L independently; retain exact leaf triangles. Use Ltilde Ltilde.T as SPD preconditioner.',
        modes=['largest uniform rank','greedy next singularvalue-squared per stored byte'],leaf_max=32,
        planner='Packed leaf triangles + rectangular U,V factors + permutation/five vectors +64KiB kernel workspace +64KiB reserve <=1MiB.',
        scope='Full exact Cholesky/SVD construction and dense factor applications are free oracle assistance. No compact factorization/native timing/full allocation audit/current-solve claim.',
        stopping='Literal original relative residual1e-8, max2000 iterations,6CPU per solve.',limits='120wall/80CPU/BLAS1'),tiny_check=None,runs=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(622086);Z=rng.normal(size=(64,64));A=Z@Z.T+np.eye(64);L=cholesky(A,lower=True)
        _,levels=partition(rng.normal(size=(64,3)),8);blocks,fixed=blocks_and_leaf(L,levels)
        full=compress(L,levels,blocks,[len(t['s']) for t in blocks]);err=float(np.linalg.norm(full-L)/np.linalg.norm(L));assert err<1e-12
        rec['tiny_check']=dict(full_rank_factor_relative_error=err);save()
        cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
        for name,filename in cases:
            if process_time()-cpu>62:rec['status']='cpu_budget';save();return
            path=ROOT/'data/prepared'/filename
            with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
            perm,levels=partition(X);A=SharedKernel(X,bw,Budget([X,b])).block(X,X);A.flat[::len(b)+1]+=1e-5
            A=A[np.ix_(perm,perm)];b=b[perm];L=cholesky(A,lower=True);blocks,fixed=blocks_and_leaf(L,levels)
            for mode in ['uniform','greedy']:
                ranks=ranks_for(blocks,fixed,mode);T=compress(L,levels,blocks,ranks)
                planner_bytes=sum(fixed.values())+sum(t['cost']*r for t,r in zip(blocks,ranks));assert planner_bytes<=1048576
                row=dict(dataset=name,mode=mode,n=len(b),data_sha256=sha256(path.read_bytes()).hexdigest(),
                    planner_bytes=planner_bytes,fixed_bytes=fixed,ranks=ranks,
                    relative_factor_error=float(np.linalg.norm(T-L)/np.linalg.norm(L)),minimum_diagonal=float(np.diag(T).min()),**pcg(A,b,T))
                rec['runs'].append(row);save();print(json.dumps({k:row[k] for k in ['dataset','mode','planner_bytes','status','iterations','original_relative_residual']}),flush=True)
    rec['status']='complete';save()
if __name__=='__main__':main()
