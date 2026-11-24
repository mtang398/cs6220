#!/usr/bin/env python3
"""Dense offline Nyström/Schur plus fixed-pattern FSAI diagnostic.

The Schur complement is assembled densely only to test the mechanism. No
production assembly, runtime advantage, or adaptive sparsity is claimed.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import numpy as np
from scipy import linalg,sparse
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
from check_projected_balancing import pcg


def schur_blocks(A,indices):
    remaining=np.setdiff1d(np.arange(len(A)),indices)
    permutation=np.r_[indices,remaining]
    L=linalg.cholesky(A[np.ix_(indices,indices)],lower=True,check_finite=False)
    B=linalg.solve_triangular(L,A[np.ix_(remaining,indices)].T,lower=True,check_finite=False).T
    D=A[np.ix_(remaining,remaining)]-B@B.T
    return permutation,L,B,(D+D.T)*.5


def block_apply(permutation,L,B,r,tail_inverse):
    r=np.asarray(r);permuted=r[permutation];k=len(L)
    top=linalg.solve_triangular(L,permuted[:k],lower=True,check_finite=False)
    tail=tail_inverse(permuted[k:]-B@top)
    top=linalg.solve_triangular(L.T,top-B.T@tail,lower=False,check_finite=False)
    value=np.empty_like(r,dtype=float);value[permutation]=np.concatenate((top,tail),axis=0)
    return value


def build_fsai(D,points,row_budget):
    distances=cdist(points,points,'sqeuclidean')
    values=[];columns=[];indptr=[0];normalization_errors=[]
    for i in range(len(D)):
        count=min(row_budget-1,i)
        neighbors=np.argsort(distances[i,:i],kind='stable')[:count]
        J=np.r_[np.sort(neighbors),i].astype(int)
        block=D[np.ix_(J,J)];rhs=np.zeros(len(J));rhs[-1]=1
        y=linalg.cho_solve(linalg.cho_factor(block,lower=True,check_finite=False),rhs,check_finite=False)
        assert y[-1]>0
        g=y/np.sqrt(y[-1])
        normalization_errors.append(abs(float(g@block@g)-1))
        values.extend(g.tolist());columns.extend(J.tolist());indptr.append(len(values))
    G=sparse.csr_matrix((np.asarray(values),np.asarray(columns,dtype=np.int32),np.asarray(indptr,dtype=np.int32)),shape=D.shape)
    assert np.all(G.diagonal()>0)
    assert np.all(G.indices<=np.repeat(np.arange(len(D)),np.diff(G.indptr)))
    return G,dict(row_budget_including_diagonal=row_budget,nnz=int(G.nnz),
        minimum_positive_diagonal=float(G.diagonal().min()),
        maximum_row_D_energy_normalization_error=float(max(normalization_errors)),
        spd_certificate='G is lower triangular with positive diagonal; H_D=G.T G is SPD',
        sparse_factor_bytes=int(G.data.nbytes+G.indices.nbytes+G.indptr.nbytes))


class SchurFSAI:
    def __init__(self,permutation,L,B,G):
        self.permutation,self.L,self.B,self.G=permutation,L,B,G
        self.GT=G.transpose(copy=False)  # CSC view shares its numerical buffers.
        assert np.shares_memory(G.data,self.GT.data)
    def apply(self,r):
        return block_apply(self.permutation,self.L,self.B,r,lambda v:self.GT@(self.G@v))
    @property
    def storage_bytes(self):
        return (self.permutation.nbytes+self.L.nbytes+self.B.nbytes
                +self.G.data.nbytes+self.G.indices.nbytes+self.G.indptr.nbytes)


def independent_small_checks():
    rng=np.random.default_rng(631);n=25;k=5
    Z=rng.normal(size=(n,n));A=Z@Z.T+.3*np.eye(n)
    X=rng.normal(size=(n,4));indices=np.array([7,1,12,20,3])
    permutation,L,B,D=schur_blocks(A,indices)
    checks=[]
    rhs=rng.normal(size=(n,3))
    for budget in (4,8,n-k):
        G,stats=build_fsai(D,X[permutation[k:]],budget)
        model=SchurFSAI(permutation,L,B,G)
        H=model.apply(np.eye(n));mineig=float(linalg.eigvalsh((H+H.T)*.5)[0])
        assert mineig>0
        Gi=linalg.solve_triangular(G.toarray(),np.eye(n-k),lower=True)
        approximate_A=np.block([[L@L.T,L@B.T],[B@L.T,B@B.T+Gi@Gi.T]])
        expected_permuted=linalg.cho_solve(linalg.cho_factor(approximate_A,lower=True),rhs[permutation])
        expected=np.empty_like(rhs);expected[permutation]=expected_permuted
        observed=model.apply(rhs)
        error=float(np.linalg.norm(expected-observed)/np.linalg.norm(expected))
        assert error<1e-11
        item=dict(row_budget=budget,min_eigenvalue_H=mineig,
            symmetry_relative_error=float(np.linalg.norm(H-H.T)/np.linalg.norm(H)),
            relative_error_against_independent_block_matrix_inverse=error,
            max_row_normalization_error=stats['maximum_row_D_energy_normalization_error'])
        if budget==n-k:
            direct=linalg.solve(A,rhs,assume_a='pos')
            item['full_pattern_relative_error_against_A_inverse']=float(np.linalg.norm(observed-direct)/np.linalg.norm(direct))
            assert item['full_pattern_relative_error_against_A_inverse']<1e-11
        checks.append(item)
    return checks


def main():
    output=root/'results/schur-fsai-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(85,89))
    started,cpu_started=perf_counter(),process_time()
    result=dict(status='running',scope='Dense offline scientific diagnostic; no production assembly or runtime claim',
        protocol=dict(seed=0,regularization=[.001,1e-5],column_ranks=[16,32],row_budgets=[4,8],
            row_budget_includes_diagonal=True,pattern='row diagonal plus nearest earlier residual points in standardized X',
            residual_order='increasing original training index after removing selected indices',
            selected_order='RPCholesky pivot order',recurrence='ordinary fixed PCG',maxiter=2500,rtol=1e-8,
            blas_threads=1,cpu_limit_seconds=85,
            retained_arrays='permutation,L,B and CSR G; transpose shares G buffers; dense A/K/D excluded'),
        source_sha256={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),root/'src/nystrom_project/numerics.py',root/'scripts/check_projected_balancing.py']},runs=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_small_checks']=independent_small_checks()
        for name in ('synthetic-n640-smooth','concrete-observed','airfoil-observed'):
            prepared=root/f'data/prepared/{name}-s0-lambda0.001.npz'
            data=np.load(prepared);X,b=data['X_train'],data['y_train']
            kernel=GaussianKernel(X,float(data['bandwidth']));K=kernel.dense()
            config=json.loads(str(data['config']))
            for lam in (.001,1e-5):
                A=K+lam*np.eye(len(b))
                for k in (16,32):
                    if process_time()-cpu_started>75:
                        result['status']='budget_exhausted';save();return
                    base=initialize_rpcholesky(kernel,k,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
                    fixed,_=pcg(A,b,base,maxiter=2500)
                    permutation,L,B,D=schur_blocks(A,base.indices)
                    # Exact tail inverse verifies the block formula independently
                    # of the sparse approximation on every actual test matrix.
                    D_factor=linalg.cho_factor(D,lower=True,check_finite=False)
                    probes=np.random.default_rng(632).normal(size=(len(b),3))
                    exact=block_apply(permutation,L,B,probes,lambda v:linalg.cho_solve(D_factor,v,check_finite=False))
                    exact_error=float(np.linalg.norm(A@exact-probes)/np.linalg.norm(probes))
                    assert exact_error<1e-7
                    row=dict(dataset=name,n=len(b),regularization=lam,rank=k,indices=base.indices.tolist(),
                        prepared_features=str(prepared.relative_to(root)),prepared_sha256=sha256(prepared.read_bytes()).hexdigest(),
                        dataset_sha256=config['dataset_sha256'],source_reference_ignored=True,
                        fixed_rp=dict(**fixed,retained_state_bytes=int(base.storage_bytes)),
                        exact_schur_block_inverse_relative_residual=exact_error,
                        factor_L_bytes=int(L.nbytes),factor_B_bytes=int(B.nbytes),permutation_bytes=int(permutation.nbytes),
                        variants=[])
                    for row_budget in (4,8):
                        G,stats=build_fsai(D,X[permutation[k:]],row_budget)
                        model=SchurFSAI(permutation,L,B,G)
                        solved,_=pcg(A,b,model,maxiter=2500)
                        item=dict(**solved,**stats,retained_state_bytes=int(model.storage_bytes),
                            byte_ratio_to_fixed_rp=float(model.storage_bytes/base.storage_bytes),
                            iteration_ratio_to_fixed_rp=(float(solved['iterations']/fixed['iterations'])
                                if solved['converged'] and fixed['converged'] else None))
                        row['variants'].append(item)
                    result['runs'].append(row);save()
                    print(json.dumps(dict(dataset=name,lam=lam,k=k,fixed=fixed['iterations'],fixed_converged=fixed['converged'],
                        fsai={str(v['row_budget_including_diagonal']):dict(iterations=v['iterations'],converged=v['converged'],
                            true_residual=v['true_relative_residual'],nnz=v['nnz'],state_bytes=v['retained_state_bytes']) for v in row['variants']},
                        fixed_bytes=base.storage_bytes)),flush=True)
        result['status']='complete';save()
        print(json.dumps(dict(status=result['status'],cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__=='__main__':main()
