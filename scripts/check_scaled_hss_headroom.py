"""Dense SPD scaling/projection HSS representation gate, construction free."""
import json, resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh, qr, block_diag, cho_solve
from threadpoolctl import threadpool_limits
from check_schur_inplace_frontier_v2 import SharedKernel, Budget, ROOT


def partition(X, leaf_max=32):
    depth=max(1,int(np.ceil(np.log2(len(X)/leaf_max))))
    def split(indices,d):
        if d==0:return indices
        axis=int(np.argmax(np.ptp(X[indices],axis=0)))
        order=indices[np.argsort(X[indices,axis],kind='stable')]
        mid=len(order)//2
        return np.r_[split(order[:mid],d-1),split(order[mid:],d-1)]
    perm=split(np.arange(len(X)),depth)
    levels=[[(0,len(X))]]
    for _ in range(depth):
        levels.append([(l,u) for a,b in levels[-1] for l,u in [(a,(a+b)//2),((a+b)//2,b)]])
    return perm,levels


def planner(n,levels):
    leaves=levels[-1];count=len(leaves)
    leaf_bytes=sum((b-a)*(b-a+1)//2*8 for a,b in leaves)
    rows=[]
    for r in range(1,min(b-a for a,b in leaves)+1):
        parts=dict(packed_leaf_factors=leaf_bytes,leaf_bases=8*n*r,
            nonroot_transfers=8*(count-2)*2*r*r,sibling_couplings=8*(count-1)*r*r,
            dense_reduced_factor_allowance=8*(count-1)*4*r*r,
            permutation_and_five_vectors=48*n,kernel_workspace=65536,reserve=65536)
        rows.append(dict(rank=r,bytes=sum(parts.values()),parts=parts))
    feasible=[x for x in rows if x['bytes']<=1048576]
    if not feasible:raise MemoryError('No rank fits declared component planner')
    return feasible[-1],rows


def construct(A,levels,rank):
    M=A.copy();previous_basis=None;previous_groups=None;history=[]
    for groups in levels[:0:-1]:
        factors=[cholesky(M[a:b,a:b],lower=True) for a,b in groups]
        scaled=M.copy()
        for (a,b),L in zip(groups,factors):
            scaled[a:b,:]=solve_triangular(L,scaled[a:b,:],lower=True)
        for (a,b),L in zip(groups,factors):
            scaled[:,a:b]=solve_triangular(L,scaled[:,a:b].T,lower=True).T
        bases=[];projections=[];nested=[];orth=[]
        for i,((a,b),L) in enumerate(zip(groups,factors)):
            if previous_basis is None:
                Q=np.eye(b-a);children=None
            else:
                children=block_diag(previous_basis[2*i],previous_basis[2*i+1])
                Q,_=qr(solve_triangular(L,children,lower=True),mode='economic')
            row=Q.T@scaled[a:b,:];row[:,a:b]=0
            gram=row@row.T
            values,V=eigh((gram+gram.T)*.5)
            V=Q@V[:,-min(rank,len(values)):]
            U=L@V
            Wi=solve_triangular(L.T,V,lower=False).T
            bases.append(U);projections.append(U@Wi)
            orth.append(float(np.linalg.norm(V.T@V-np.eye(V.shape[1]))))
            if children is not None:
                transfer=np.linalg.lstsq(children,U,rcond=None)[0]
                nested.append(float(np.linalg.norm(children@transfer-U)/np.linalg.norm(U)))
        left=np.empty_like(M)
        for (a,b),P in zip(groups,projections):left[a:b,:]=P@M[a:b,:]
        new=np.empty_like(M)
        for (a,b),P in zip(groups,projections):new[:,a:b]=left[:,a:b]@P.T
        for a,b in groups:new[a:b,a:b]=M[a:b,a:b]
        asym=float(np.linalg.norm(new-new.T)/np.linalg.norm(new))
        M=(new+new.T)*.5
        history.append(dict(block_count=len(groups),largest_block=max(b-a for a,b in groups),
            ranks=[u.shape[1] for u in bases],maximum_nested_relative_error=max(nested,default=0),
            maximum_basis_orthogonality_error=max(orth),relative_asymmetry_before_symmetrizing=asym))
        previous_basis,previous_groups=bases,groups
    factor=cholesky(M,lower=True)
    return M,factor,history


def pcg(A,b,L,cap=6.):
    cpu=process_time();x=np.zeros_like(b);r=b.copy();z=cho_solve((L,True),r);p=z.copy();rho=float(r@z)
    target=1e-8*np.linalg.norm(b);status='maxiter';iterations=0
    for k in range(2001):
        true=b-A@x
        if np.linalg.norm(true)<=target:status='converged';break
        if process_time()-cpu>cap:status='cpu_cap';break
        if k==2000:break
        ap=A@p;den=float(p@ap)
        if min(rho,den)<=0:status='nonpositive_curvature';break
        x+=(rho/den)*p;r-=(rho/den)*ap
        z=cho_solve((L,True),r);new=float(r@z);p=z+(new/rho)*p;rho=new;iterations+=1
    return dict(status=status,converged=status=='converged',iterations=iterations,
        original_relative_residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),
        exact_matvec_kernel_entries=iterations*len(b)**2,preconditioner_applications=iterations+1,
        diagnostic_cpu_seconds=process_time()-cpu)


def main():
    out=ROOT/'results/scaled-hss-headroom.json'
    if out.exists():raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
    files=[Path(__file__).resolve(),ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},
        protocol=dict(source='Xing-Huang-Chow SIMAX2021 Sections3-4 https://faculty.cc.gatech.edu/~echow/pubs/xing-huang-chow-simax-2021.pdf',
            method='Bottom-up SPD scaling/projection; parent bases restricted to scaled child-basis span; complete interblock couplings retained through low-rank nested bases.',
            leaf_max=32,rank='largest uniform rank fitting declared component planner, no convergence tuning',
            scope='Dense construction/full dense Cholesky inverse granted free. Component planner is not a native allocation audit or implemented compact factorization. Static representation gate only; no solve-adaptive claim.',
            reference='Existing four hardshift development cases, seed0; exact original residual1e-8 every iteration.',
            regularization=1e-5,wall_cap_seconds=120,cpu_cap_seconds=80,per_solve_cpu_cap=6),
        tiny_check=None,cases=[],failures=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    with threadpool_limits(limits=1):
        try:
            rng=np.random.default_rng(622082);Z=rng.normal(size=(64,64));A=Z@Z.T+np.eye(64)
            _,levels=partition(rng.normal(size=(64,3)),8)
            full,_,_=construct(A,levels,32);compressed,L,h=construct(A,levels,3)
            err=float(np.linalg.norm(full-A)/np.linalg.norm(A));assert err<1e-10
            rec['tiny_check']=dict(full_rank_identity_relative_error=err,compressed_minimum_eigenvalue=float(eigh(compressed,eigvals_only=True)[0]),
                max_nested_error=max(x['maximum_nested_relative_error'] for x in h));save()
            cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),
                ('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
            for name,filename in cases:
                if process_time()-cpu>65:rec['status']='cpu_budget';save();return
                path=ROOT/'data/prepared'/filename
                with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
                perm,levels=partition(X);plan,allplans=planner(len(b),levels)
                kernel=SharedKernel(X,bw,Budget([X,b]));A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5
                A=A[np.ix_(perm,perm)];b=b[perm]
                M,L,history=construct(A,levels,plan['rank'])
                row=dict(dataset=name,n=len(b),planner=plan,planner_frontier=allplans,
                    data_sha256=sha256(path.read_bytes()).hexdigest(),history=history,
                    relative_matrix_error=float(np.linalg.norm(M-A)/np.linalg.norm(A)),
                    minimum_Cholesky_diagonal=float(np.diag(L).min()),**pcg(A,b,L))
                rec['cases'].append(row);save();print(json.dumps({k:row[k] for k in ['dataset','status','iterations','original_relative_residual']}),flush=True)
        except Exception as e:rec['status']='failed';rec['failures'].append(repr(e));save();raise
    rec['status']='complete';save()


if __name__=='__main__':main()
