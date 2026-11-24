import numpy as np
from scipy import linalg
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner
from nystrom_project.learning import compress_probe
from nystrom_project.revisions import revised_solve


def test_compression_matches_independent_augmented_nystrom():
    rng=np.random.default_rng(89)
    kernel=GaussianKernel(rng.normal(size=(50,4)),1.1);K=kernel.dense()
    old=initialize_rpcholesky(kernel,6,.01,rng)
    state=LowRankPreconditioner.from_nystrom(old)
    p=rng.normal(size=50)
    new,event=compress_probe(state,p,K@p)
    assert event['accepted']
    omega=np.column_stack((np.eye(50)[:,old.indices],p))
    Y=K@omega;aug=Y@np.linalg.solve(omega.T@Y,Y.T)
    w,V=linalg.eigh((aug+aug.T)*.5)
    expected=(V[:,-6:]*w[-6:])@V[:,-6:].T
    actual=(new.basis*new.eigenvalues)@new.basis.T
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
    assert linalg.eigvalsh(K-actual)[0]>-1e-10


def test_learning_solves_original_system_and_restores_columns():
    rng=np.random.default_rng(90);X=rng.normal(size=(90,5));b=rng.normal(size=90)
    for method in ['flex_compact','flex_compress','flex_compress_columns','flex_restart','flex_gaussian_qr','flex_fsai']:
        kernel=GaussianKernel(X,1.8)
        result=revised_solve(kernel,b,.01,method=method,rank=12,history_size=1,warmup_steps=5,checkpoint=10)
        relative=np.linalg.norm(b-(kernel.dense()+.01*np.eye(90))@result.solution)/np.linalg.norm(b)
        assert result.metrics['converged'] and relative<1e-8
        if method in ('flex_compress','flex_compress_columns','flex_restart'):
            assert result.metrics['restarts']==1
            assert any(e.get('restart') and e['iteration']==5 for e in result.events)
        if method=='flex_compress_columns':
            assert len(set(result.metrics['final_indices']))==12
            assert result.metrics['column_evaluations']==24
