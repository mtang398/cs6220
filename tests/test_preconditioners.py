import unittest
import numpy as np
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
from nystrom_project.preconditioners import SpectralPreconditioner,DiagonalPreconditioner,initialize_gaussian,LowRankPreconditioner


class FixedControls(unittest.TestCase):
    def test_generalized_nystrom_and_compact_inverse(self):
        rng=np.random.default_rng(31)
        kernel=GaussianKernel(rng.normal(size=(40,3)),1.2)
        K=kernel.dense();omega=np.random.default_rng(45).normal(size=(40,7))
        model=initialize_gaussian(kernel,7,.01,np.random.default_rng(45))
        exact=(K@omega)@np.linalg.solve(omega.T@K@omega,omega.T@K)
        v=rng.normal(size=40)
        np.testing.assert_allclose((model.basis*model.eigenvalues)@model.basis.T,exact,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(model.apply(v),np.linalg.solve(exact+.01*np.eye(40),v),rtol=1e-9,atol=1e-9)
        self.assertEqual(kernel.stats['matmats'],1)
        self.assertEqual(kernel.stats['matmat_rhs'],7)
        power=initialize_gaussian(kernel,7,.01,np.random.default_rng(45),power=True)
        self.assertEqual(kernel.stats['matmats'],3)
        self.assertEqual(kernel.stats['matmat_rhs'],21)
        Q,_=np.linalg.qr(K@omega)
        exact_power=(K@Q)@np.linalg.solve(Q.T@K@Q,Q.T@K)
        np.testing.assert_allclose((power.basis*power.eigenvalues)@power.basis.T,exact_power,rtol=1e-9,atol=1e-10)
        base=initialize_rpcholesky(kernel,7,.01,rng)
        compact=LowRankPreconditioner.from_nystrom(base)
        np.testing.assert_array_equal(base.apply(v),compact.apply(v))
        self.assertLess(compact.storage_bytes,base.storage_bytes)

    def test_nonconstant_kernel_diagonal_and_greedy_pivots(self):
        rng=np.random.default_rng(23)
        factor=rng.normal(size=(25,4))
        matrix=factor @ factor.T
        class Oracle:
            n=25
            diagonal=np.diag(matrix)
            def column(self,j):return matrix[:,j].copy()
        model=initialize_rpcholesky(Oracle(),8,.01,rng,pivoting='greedy')
        self.assertEqual(model.indices[0],np.argmax(np.diag(matrix)))
        self.assertEqual(len(np.unique(model.indices)),8)
        self.assertEqual(model.rank,4)
        np.testing.assert_allclose((model.basis*model.eigenvalues) @ model.basis.T,matrix,rtol=1e-10,atol=1e-10)

    def test_dense_inverse(self):
        rng=np.random.default_rng(18)
        kernel=GaussianKernel(rng.normal(size=(40,3)),1.2)
        base=initialize_rpcholesky(kernel,8,.01,rng)
        U,s=base.basis,base.eigenvalues
        Khat=(U*s) @ U.T
        v=rng.normal(size=40)
        for cls in (SpectralPreconditioner,DiagonalPreconditioner):
            with self.subTest(kind=cls.__name__):
                model=cls(base)
                if cls is SpectralPreconditioner:
                    M=(U*(s+.01)) @ U.T+model.floor*(np.eye(40)-U @ U.T)
                else:M=Khat+np.diag(1-np.diag(Khat)+.01)
                np.testing.assert_allclose(model.apply(v),np.linalg.solve(M,v),rtol=1e-10,atol=1e-10)


if __name__=='__main__':unittest.main()
