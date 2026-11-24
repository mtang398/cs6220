"""Unavailable exact-error oracle for signed-cancellation Schur partitions.

The core and block cardinalities are fixed. After twenty PCG steps, one
deterministic swap sweep minimize e.T blockdiag(S_BB) e using exact remaining
error. Score gains must translate to complete remaining convergence. Dense
matrices, exact error and partition search are free here: this is neither a
memory-feasible algorithm nor timing evidence.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits
import check_schur_block_headroom as base
from check_schur_block_matched_factors import balanced


def optimize(S, error, groups):
    e = error / np.linalg.norm(error)
    H = S*(e[:,None]*e[None,:] if e.ndim == 1 else e@e.T)
    hnorm = np.linalg.norm(H, ord="fro")
    labels = np.empty(len(S), dtype=int)
    for k,q in enumerate(groups): labels[q] = k
    membership = np.eye(len(groups))[labels]
    sums = H@membership
    def score(lab):
        return float(sum(np.sum(H[np.ix_(np.flatnonzero(lab==k),np.flatnonzero(lab==k))])
                         for k in range(len(groups))))
    before = score(labels)
    # Independent check of the swap difference before using its fast formula.
    i,j = int(groups[0][0]), int(groups[1][0])
    a,b = labels[i],labels[j]
    delta = 2*(sums[i,b]-sums[i,a]+sums[j,a]-sums[j,b]+H[i,i]+H[j,j]-2*H[i,j])
    swapped = labels.copy();swapped[i],swapped[j]=b,a
    err = abs((score(swapped)-before)-delta)
    assert err < 1e-11*max(1.,np.linalg.norm(H))
    changes=[]
    for sweep in range(1):
        count=0
        for i in range(len(S)):
            a=labels[i]
            delta=2*(sums[i,labels]-sums[i,a]+sums[:,a]-sums[np.arange(len(S)),labels]
                     +H[i,i]+np.diag(H)-2*H[i,:])
            delta[labels==a]=np.inf
            j=int(np.argmin(delta));b=labels[j]
            if delta[j] < -1e-14*max(hnorm,1e-30):
                update=H[:,j]-H[:,i]
                sums[:,a]+=update;sums[:,b]-=update
                labels[i],labels[j]=b,a
                count+=1
        changes.append(count)
    new=[np.flatnonzero(labels==k) for k in range(len(groups))]
    assert [len(q) for q in new]==[len(q) for q in groups]
    after=score(labels)
    assert after<=before+1e-12*max(1.,abs(before))
    return new,dict(score_before=before,score_after=after,swaps_per_sweep=changes,
                    independent_swap_delta_error=err)


def make_preconditioner(S, groups):
    factors=[(q,cho_factor(S[np.ix_(q,q)],lower=True)) for q in groups]
    def apply(v):
        z=np.empty_like(v)
        for q,f in factors:z[q]=cho_solve(f,v[q])
        return z
    return LinearOperator(S.shape,matvec=apply)


VERIFY = None


def verifier(A,b,W,C,R,Cidx):
    def verify(y):
        x=np.empty_like(b)
        x[R]=y
        x[Cidx]=cho_solve(W,b[Cidx]-C.T@y)
        return float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
    return verify


def solve(S,g,M,x0=None,maxiter=2500):
    count=0
    saved=None
    class Finished(Exception): pass
    def callback(x):
        nonlocal count,saved
        count+=1
        if VERIFY(x)<=1e-8:
            saved=x.copy()
            raise Finished
    try:
        x,info=cg(S,g,M=M,x0=x0,rtol=0.,atol=0.,maxiter=maxiter,callback=callback)
    except Finished:
        return saved,count,0
    return x,count,int(info)


def main():
    global VERIFY
    output=base.ROOT/'results/schur-cancellation-oracle-v2.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    reference_path=base.ROOT/'results/schur-block-matched-factors.json'
    reference=json.loads(reference_path.read_text())
    record=dict(status='running',protocol=dict(checkpoint=20,core_count=128,swap_sweeps=1,
        factor_allowance_bytes=103040,oracle='Exact current error; two smallest generalized eigenvectors as matrix-only control',
        scope='Free dense construction, true error and repartition; no end-to-end or memory claim',
        stopping='First literal original residual <=1e-8, checked every iterate; original verification products counted',
        comparisons=['uninterrupted','unchanged_restart','exact_error_partition','matrix_only_partition','larger_core160_recomputed_control'],
        primary_decision='Full verified solve action count, not partition score'),
        sources={str(p.relative_to(base.ROOT)):sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__),Path(base.__file__),Path(__file__).with_name('check_schur_block_matched_factors.py'),reference_path]},runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for row in reference['runs']:
            if row['core_count']!=128:continue
            name,lam=row['dataset'],row['regularization']
            stem='concrete-observed' if name=='Concrete' else f'synthetic-n640-{name}'
            path=base.ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
            assert sha256(path.read_bytes()).hexdigest()==row['data_sha256']
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A=K+lam*np.eye(len(K))
            Cidx=base.selected(K,128);R=np.setdiff1d(np.arange(len(K)),Cidx)
            W=cho_factor(A[np.ix_(Cidx,Cidx)],lower=True);C=A[np.ix_(R,Cidx)]
            S=A[np.ix_(R,R)]-C@cho_solve(W,C.T);S=(S+S.T)*.5
            g=b[R]-C@cho_solve(W,b[Cidx])
            groups=balanced(X[R],np.arange(len(R)),row['group_count'])
            M=make_preconditioner(S,groups)
            VERIFY=verifier(A,b,W,C,R,Cidx)
            prefix,steps,info=solve(S,g,M,maxiter=20)
            exact=cho_solve(cho_factor(S,lower=True),g)
            new,score=optimize(S,exact-prefix,groups)
            Dold=np.zeros_like(S)
            for q in groups:Dold[np.ix_(q,q)]=S[np.ix_(q,q)]
            theta,V=eigh(S,Dold,subset_by_index=[0,1])
            V/=np.sqrt(np.sum(V*(S@V),axis=0))[None,:]
            matrix_groups,matrix_score=optimize(S,V,groups)
            item=dict(dataset=name,regularization=lam,score=score,matrix_score=matrix_score,smallest_eigenvalues=theta.tolist(),branches=[],prefix_iterations=steps,
                      exact_schur_solution_residual=float(np.linalg.norm(g-S@exact)/np.linalg.norm(g)))
            for mode,preconditioner,x0 in [('uninterrupted',M,None),('unchanged_restart',M,prefix),
                                          ('exact_error_partition',make_preconditioner(S,new),prefix),
                                          ('matrix_only_partition',make_preconditioner(S,matrix_groups),prefix)]:
                y,count,status=solve(S,g,preconditioner,x0=x0)
                x=np.empty_like(b);x[R]=y;x[Cidx]=cho_solve(W,b[Cidx]-C.T@y)
                rel=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
                total=count+(steps if x0 is not None else 0)
                item['branches'].append(dict(mode=mode,iterations=total,solver_info=status,
                    true_relative_residual=rel,converged=bool(status==0 and rel<=1e-8),
                    optimistic_action_kernel_entries=total*(len(A)**2-128**2),
                    original_verification_products=total+1,
                    action_and_verification_entries=total*(len(A)**2-128**2)+(total+1)*len(A)**2,
                    free_diagonal_block_reuse_entries=total*(len(A)**2-128**2-sum(len(q)**2 for q in groups))))
            Cidx160=base.selected(K,160);R160=np.setdiff1d(np.arange(len(K)),Cidx160)
            W160=cho_factor(A[np.ix_(Cidx160,Cidx160)],lower=True);C160=A[np.ix_(R160,Cidx160)]
            S160=A[np.ix_(R160,R160)]-C160@cho_solve(W160,C160.T);S160=(S160+S160.T)*.5
            g160=b[R160]-C160@cho_solve(W160,b[Cidx160])
            VERIFY=verifier(A,b,W160,C160,R160,Cidx160)
            y160,count160,status160=solve(S160,g160,None)
            rel160=VERIFY(y160)
            item['larger_core_control']=dict(iterations=count160,solver_info=status160,
                true_relative_residual=rel160,converged=bool(status160==0 and rel160<=1e-8),
                optimistic_action_kernel_entries=count160*(len(A)**2-160**2),
                original_verification_products=count160+1,
                action_and_verification_entries=count160*(len(A)**2-160**2)+(count160+1)*len(A)**2)
            record['runs'].append(item);save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
