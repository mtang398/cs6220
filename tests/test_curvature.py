import numpy as np
from scipy.linalg import solve
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
from nystrom_project.flexible import AConjugateHistory
from nystrom_project.curvature import curvature_exchange,logdet_projected


def test_history_curvature_is_preserved_and_no_kernel_product_is_hidden():
    rng=np.random.default_rng(123)
    kernel=GaussianKernel(rng.normal(size=(45,5)),1.8)
    lam=.01;A=kernel.dense()+lam*np.eye(45)
    history=AConjugateHistory(45,5)
    for _ in range(5):
        z=rng.normal(size=45);p,ap=history.project(z,A@z);history.append(p,ap)
    P=np.column_stack([p for p,_,_ in history.snapshot().pairs])
    old=initialize_rpcholesky(kernel,6,lam,np.random.default_rng(12))
    before=logdet_projected(old,P);stats=dict(kernel.stats)
    new,event=curvature_exchange(kernel,old,rng.normal(size=45),np.random.default_rng(8),history=history,candidate_count=4)
    after=logdet_projected(new,P)
    assert after>=before-1e-9
    assert kernel.stats['matvecs']==stats['matvecs']
    assert len(set(new.indices))==6 and history.size==5
    C=kernel.dense()[:,new.indices]
    M=C@solve(C[new.indices,:],C.T,assume_a='pos')+lam*np.eye(45)
    sign,exact=np.linalg.slogdet(P.T@M@P)
    assert sign>0
    np.testing.assert_allclose(after,exact,rtol=1e-9,atol=1e-9)
