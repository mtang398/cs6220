"""Independent algebra, indexing, sparse storage, and matrix-free setup tests."""
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import LinearOperator,cg

from nystrom_project.numerics import GaussianKernel
from nystrom_project.schur import SchurFSAIPreconditioner,initialize_schur_fsai
import nystrom_project.schur as schur_module


class SchurTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(633)

    def kernel(self,n=25):
        return GaussianKernel(self.rng.normal(size=(n,4)),1.3,block_size=6)

    def test_local_index_mapping_patterns_counters_and_no_dense_setup(self):
        kernel=self.kernel(27);K=kernel.dense();lam=.025;k=6;sparsity=4
        kernel.column(0)  # Setup must preserve existing counters.
        kernel.dense=lambda:self.fail('dense kernel requested')
        kernel.matvec=lambda v:self.fail('kernel matvec requested during setup/application')
        kernel.matmat=lambda v:self.fail('kernel matmat requested during setup/application')
        entry_calls=[];distance_shapes=[];selection_refs=[]
        entries=kernel.entries;distance=schur_module.cdist;initialize=schur_module.initialize_rpcholesky
        def queried(rows,columns):
            entry_calls.append((rows.copy(),columns.copy()))
            self.assertLessEqual(len(rows),sparsity)
            return entries(rows,columns)
        def row_distance(a,b,*args,**kwargs):
            distance_shapes.append((a.shape,b.shape))
            self.assertEqual(a.shape[0],1)
            return distance(a,b,*args,**kwargs)
        def selection(*args,**kwargs):
            value=initialize(*args,**kwargs);selection_refs.append(weakref.ref(value));return value
        kernel.entries=queried
        with patch.object(schur_module,'cdist',side_effect=row_distance),patch.object(schur_module,'initialize_rpcholesky',side_effect=selection):
            state=initialize_schur_fsai(kernel,k,lam,self.rng,sparsity=sparsity)
        gc.collect();self.assertTrue(all(ref() is None for ref in selection_refs))
        self.assertTrue(distance_shapes)
        self.assertEqual(kernel.stats['column_evaluations'],k+1)
        self.assertEqual(kernel.stats['entry_blocks'],kernel.n-k)
        self.assertEqual(kernel.stats['entry_evaluations'],sum(min(i+1,sparsity)**2 for i in range(kernel.n-k)))
        self.assertEqual(kernel.stats['matvecs'],0)
        self.assertNotIn('matmats',kernel.stats)
        A=K+lam*np.eye(kernel.n);R=state.remaining_indices;S=state.indices
        D=A[np.ix_(R,R)]-A[np.ix_(R,S)]@linalg.solve(A[np.ix_(S,S)],A[np.ix_(S,R)],assume_a='pos')
        nonconsecutive=0
        for i,(rows,columns) in enumerate(entry_calls):
            distances=np.sum((kernel.X[R[:i]]-kernel.X[R[i]])**2,axis=1)
            nearest=np.argsort(distances,kind='stable')[:min(i,sparsity-1)]
            expected_pattern=np.r_[np.sort(nearest),i]
            np.testing.assert_array_equal(state.row_pattern(i),expected_pattern)
            np.testing.assert_array_equal(rows,R[expected_pattern]);np.testing.assert_array_equal(columns,rows)
            nonconsecutive+=not np.array_equal(rows,np.arange(len(rows)))
            local=D[np.ix_(expected_pattern,expected_pattern)];e=np.eye(len(local))[:,-1]
            y=np.linalg.solve(local,e)
            np.testing.assert_allclose(state.row_values(i),y/np.sqrt(y[-1]),rtol=2e-12,atol=2e-12)
        self.assertGreater(nonconsecutive,10)
        before=dict(kernel.stats);state.apply(self.rng.normal(size=kernel.n));self.assertEqual(kernel.stats,before)

    def test_inverse_agrees_with_independent_approximate_block_matrix_and_is_spd(self):
        kernel=self.kernel(22);K=kernel.dense();lam=.01
        state=initialize_schur_fsai(kernel,5,lam,self.rng,sparsity=4)
        A=K+lam*np.eye(kernel.n);S,R=state.indices,state.remaining_indices
        A11=A[np.ix_(S,S)];A21=A[np.ix_(R,S)]
        G=state.G.toarray();Dhat=np.linalg.inv(G.T@G)
        approximate=np.block([[A11,A21.T],[A21,A21@np.linalg.solve(A11,A21.T)+Dhat]])
        H=np.column_stack([state.apply(v) for v in np.eye(kernel.n)])
        np.testing.assert_allclose(H,H.T,rtol=2e-12,atol=2e-12)
        self.assertGreater(linalg.eigvalsh(H)[0],0)
        rhs=self.rng.normal(size=kernel.n)
        expected_permuted=np.linalg.solve(approximate,rhs[state.permutation])
        expected=np.empty_like(rhs);expected[state.permutation]=expected_permuted
        np.testing.assert_allclose(state.apply(rhs),expected,rtol=2e-11,atol=2e-11)

    def test_full_local_patterns_and_full_landmarks_give_exact_inverse(self):
        kernel=self.kernel(14);K=kernel.dense();lam=.02;v=self.rng.normal(size=14)
        expected=np.linalg.solve(K+lam*np.eye(14),v)
        for rank in (0,4,14):
            with self.subTest(rank=rank):
                kernel.reset_stats()
                state=initialize_schur_fsai(kernel,rank,lam,self.rng,sparsity=14)
                np.testing.assert_allclose(state.apply(v),expected,rtol=2e-12,atol=2e-12)
                self.assertEqual(state.rank,rank)
                self.assertEqual(state.G.nnz,(14-rank)*(15-rank)//2)
                self.assertEqual(kernel.stats.get('entry_blocks',0),14-rank)

    def test_duplicate_points_have_deterministic_ties_and_regularized_spd_rows(self):
        kernel=GaussianKernel(np.zeros((13,2)),1.0);K=kernel.dense()
        state=initialize_schur_fsai(kernel,4,1e-4,self.rng,sparsity=3,initialization='greedy')
        self.assertEqual(state.rank,4)
        self.assertEqual(state.diagnostics['landmark_kernel_rank'],1)
        for i in range(9):
            np.testing.assert_array_equal(state.row_pattern(i),np.r_[np.arange(min(i,2)),i])
        H=np.column_stack([state.apply(v) for v in np.eye(13)])
        self.assertGreater(linalg.eigvalsh((H+H.T)*.5)[0],0)
        self.assertTrue(np.all(np.isfinite(H)))

    def test_retained_storage_counts_unique_buffers_and_exposes_readonly_rows(self):
        state=initialize_schur_fsai(self.kernel(20),5,.02,self.rng,sparsity=4)
        arrays=(state.permutation,state.L,state.B,state.G.data,state.G.indices,state.G.indptr)
        self.assertEqual(state.storage_bytes,sum(a.nbytes for a in arrays))
        self.assertTrue(all(not a.flags.writeable for a in arrays))
        self.assertTrue(np.shares_memory(state._GT.data,state.G.data))
        self.assertTrue(np.shares_memory(state._GT.indices,state.G.indices))
        self.assertTrue(np.shares_memory(state._GT.indptr,state.G.indptr))
        self.assertTrue(np.shares_memory(state.indices,state.permutation))
        self.assertTrue(np.shares_memory(state.remaining_indices,state.permutation))
        self.assertTrue(np.shares_memory(state.row_pattern(8),state.G.indices))
        self.assertTrue(np.shares_memory(state.row_values(8),state.G.data))
        self.assertFalse(state.row_values(8).flags.writeable)
        self.assertFalse(any(hasattr(state,name) for name in ('columns','basis','kernel','D','X')))
        replacement=SchurFSAIPreconditioner(state.permutation,state.L,state.B,state.G.copy(),.02)
        self.assertIs(replacement.L,state.L);self.assertIs(replacement.B,state.B)
        self.assertIs(replacement.permutation,state.permutation)

    def test_static_state_works_in_independent_pcg_with_true_residual(self):
        kernel=self.kernel(31);K=kernel.dense();lam=1e-3;A=K+lam*np.eye(31)
        state=initialize_schur_fsai(kernel,7,lam,self.rng,sparsity=5)
        b=self.rng.normal(size=31)
        preconditioner=LinearOperator((31,31),matvec=state.apply,dtype=np.float64)
        x,info=cg(A,b,M=preconditioner,rtol=1e-10,atol=0,maxiter=300)
        self.assertEqual(info,0)
        self.assertLess(np.linalg.norm(b-A@x)/np.linalg.norm(b),1e-9)
        np.testing.assert_allclose(x,np.linalg.solve(A,b),rtol=2e-9,atol=2e-9)

    def test_invalid_parameters_and_non_spd_sparse_factor_are_rejected(self):
        kernel=self.kernel(12)
        for kwargs in (dict(rank=-1),dict(rank=13),dict(rank=2.5),dict(sparsity=0),
                       dict(sparsity=1.5),dict(regularization=0),dict(regularization=np.nan),dict(initialization='invalid')):
            params=dict(rank=3,regularization=.1,sparsity=3);params.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                initialize_schur_fsai(kernel,rng=self.rng,**params)
        state=initialize_schur_fsai(kernel,3,.1,self.rng,sparsity=3)
        for vector in (np.zeros(11),np.full(12,np.nan),np.zeros((12,1))):
            with self.assertRaises(ValueError):state.apply(vector)
        with self.assertRaises(IndexError):state.row_pattern(-1)
        bad=sparse.eye(9,format='csr');bad.data[0]=0
        with self.assertRaises(ValueError):
            SchurFSAIPreconditioner(state.permutation,state.L,state.B,bad,.1)
        invalid_indices=state.G.copy()
        invalid_indices.indices[invalid_indices.indptr[1]]=-1
        with self.assertRaises(ValueError):
            SchurFSAIPreconditioner(state.permutation,state.L,state.B,invalid_indices,.1)


if __name__=='__main__':unittest.main()
