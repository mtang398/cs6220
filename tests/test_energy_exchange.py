import numpy as np
from scipy.linalg import solve

from nystrom_project.numerics import GaussianKernel,NystromPreconditioner,initialize_rpcholesky
from nystrom_project.exchange import removal_directions,energy_exchange


def test_matrix_application_and_every_removal_against_dense_inverse():
    rng=np.random.default_rng(91)
    kernel=GaussianKernel(rng.normal(size=(35,4)),1.5,block_size=8)
    K=kernel.dense();R=rng.normal(size=(35,7))
    np.testing.assert_allclose(kernel.matmat(R),K@R,rtol=1e-13,atol=1e-13)
    assert kernel.stats['matmats']==1 and kernel.stats['matmat_rhs']==7
    for lam in [0.1,0.001,1e-5]:
        state=initialize_rpcholesky(kernel,5,lam,np.random.default_rng(8))
        j=next(i for i in range(35) if i not in state.indices)
        r=rng.normal(size=35)
        Z,indices,C,_,_=removal_directions(state,j,kernel.column(j),r)
        for removal in range(6):
            take=np.arange(6)!=removal
            W=C[indices,:][np.ix_(take,take)]
            M=C[:,take]@solve(W,C[:,take].T,assume_a='pos')+lam*np.eye(35)
            expected=solve(M,r,assume_a='pos')
            np.testing.assert_allclose(Z[:,removal],expected,rtol=2e-6,atol=1e-7)


def test_energy_acceptance_matches_actual_one_step_error_reduction():
    rng=np.random.default_rng(92)
    kernel=GaussianKernel(rng.normal(size=(40,4)),1.4)
    lam=.03;A=kernel.dense()+lam*np.eye(40)
    rhs=rng.normal(size=40);truth=solve(A,rhs,assume_a='pos')
    old=initialize_rpcholesky(kernel,5,lam,np.random.default_rng(3))
    new,event=energy_exchange(kernel,old,rhs,np.random.default_rng(5),candidate_count=4)
    def energy_after(model):
        z=model.apply(rhs);x=z*(rhs@z)/(z@A@z);e=x-truth
        return e@A@e
    assert energy_after(new)<=energy_after(old)+1e-10
    assert len(set(new.indices))==5
    for row in event['details']:
        if row['accepted']:assert row['energy_decrease_after']>row['energy_decrease_before']
