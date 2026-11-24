"""Algebra/robustness tests, not timing or convergence benchmark evidence."""

import json
import unittest

import numpy as np
from scipy import linalg

from nystrom_project.numerics import (
    GaussianKernel, NystromPreconditioner, _all_removal_scores,
    exchange_columns, initialize_rpcholesky,
)


def dense_preconditioner(K, indices, regularization):
    columns = K[:, indices]
    core = K[np.ix_(indices, indices)]
    return columns @ np.linalg.pinv(core, rcond=1e-12) @ columns.T + regularization*np.eye(len(K))


class FixedCandidate:
    def __init__(self, candidate):
        self.candidate = candidate

    def choice(self, choices, p=None):
        if self.candidate not in choices:
            raise AssertionError("forced candidate is already selected")
        return self.candidate


class KernelAndNystromTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(6220)

    def test_blockwise_kernel_matches_independent_distances_and_counts(self):
        X = self.rng.normal(size=(31, 4))
        X[4] = X[3]
        kernel = GaussianKernel(X, 0.9, block_size=7)
        differences = X[:, None, :] - X[None, :, :]
        K = np.exp(-np.sum(differences**2, axis=2)/(2*0.9**2))
        np.testing.assert_allclose(kernel.dense(), K, rtol=2e-15, atol=2e-15)
        kernel.dense = lambda: self.fail("performance path requested a dense kernel")
        for j in (0, 3, 30):
            np.testing.assert_allclose(kernel.column(j), K[:, j], atol=2e-15)
        v = self.rng.normal(size=31)
        np.testing.assert_allclose(kernel.matvec(v), K @ v, rtol=2e-14, atol=2e-14)
        self.assertEqual(kernel.stats["column_evaluations"], 3)
        self.assertEqual(kernel.stats["matvecs"], 1)
        self.assertGreaterEqual(kernel.stats["column_seconds"], 0)
        self.assertGreaterEqual(kernel.stats["matvec_seconds"], 0)
        kernel.reset_stats()
        self.assertEqual(kernel.stats, dict(matvecs=0, column_evaluations=0,
                                           matvec_seconds=0.0, column_seconds=0.0))

    def test_rpcholesky_reuses_factor_and_obeys_nystrom_psd_order(self):
        kernel = GaussianKernel(self.rng.normal(size=(45, 5)), 1.1)
        K = kernel.dense()
        kernel.dense = lambda: self.fail("RPCholesky requested dense K")
        state = initialize_rpcholesky(kernel, 8, 1e-3, self.rng)
        self.assertEqual(len(np.unique(state.indices)), 8)
        self.assertEqual(state.rank, 8)
        self.assertTrue(state.diagnostics["reused_factor"])
        self.assertEqual(kernel.stats["column_evaluations"], 8)
        approximation = (state.basis*state.eigenvalues) @ state.basis.T
        C = K[:, state.indices]
        exact_nystrom = C @ linalg.solve(K[np.ix_(state.indices, state.indices)], C.T,
                                       assume_a="pos")
        np.testing.assert_allclose(approximation, exact_nystrom, rtol=1e-11, atol=1e-12)
        self.assertGreaterEqual(linalg.eigvalsh(K-approximation)[0], -1e-11)
        np.testing.assert_allclose(state.gram, C.T @ C, rtol=1e-14, atol=1e-14)

    def test_stable_inverse_and_score_match_independent_dense_solves(self):
        kernel = GaussianKernel(self.rng.normal(size=(35, 4)), 1.2)
        K = kernel.dense()
        indices = np.array([0, 3, 9, 10, 18, 27])
        for regularization in (0.1, 1e-3, 1e-5):
            with self.subTest(regularization=regularization):
                state = NystromPreconditioner(indices, K[:, indices], regularization)
                M = dense_preconditioner(K, indices, regularization)
                for v in (self.rng.normal(size=len(K)), K[:, indices[0]]):
                    direct = linalg.solve(M, v, assume_a="pos")
                    np.testing.assert_allclose(state.apply(v), direct, rtol=2e-8, atol=2e-9)
                    self.assertAlmostEqual(state.score(v)/(v @ direct), 1.0, delta=1e-9)
                    self.assertAlmostEqual(state.score(v)/(v @ state.apply(v)), 1.0, delta=1e-9)
                    error = v @ linalg.solve(K+regularization*np.eye(len(K)), v, assume_a="pos")
                    self.assertGreaterEqual(state.score(v), error*(1-1e-10))

    def test_duplicate_inputs_preserve_budget_and_lower_effective_rank(self):
        X = np.repeat(np.array([[0., 0.], [1., -1.], [2., 1.]]), 4, axis=0)
        kernel = GaussianKernel(X, 1.0)
        K = kernel.dense()
        state = initialize_rpcholesky(kernel, 9, 1e-4, self.rng)
        self.assertEqual(len(np.unique(state.indices)), 9)
        self.assertEqual(state.rank, 3)
        self.assertTrue(state.diagnostics["core_truncated"])
        self.assertGreater(state.diagnostics["rp_uniform_fill"], 0)
        M = dense_preconditioner(K, state.indices, state.regularization)
        v = self.rng.normal(size=len(K))
        direct = linalg.solve(M, v, assume_a="pos")
        np.testing.assert_allclose(state.apply(v), direct, rtol=1e-8, atol=1e-8)
        self.assertAlmostEqual(state.score(v)/(v @ direct), 1., delta=1e-9)

    def test_near_dependent_columns_remain_finite_and_psd(self):
        X = np.array([[0.], [1e-10], [2e-10], [1.], [1.+1e-10], [2.]])
        kernel = GaussianKernel(X, 1.0)
        K = kernel.dense()
        state = initialize_rpcholesky(kernel, 5, 1e-5, self.rng)
        self.assertEqual(len(state.indices), 5)
        self.assertLess(state.rank, 5)
        v = self.rng.normal(size=len(X))
        self.assertTrue(np.all(np.isfinite(state.apply(v))))
        self.assertGreater(state.score(v), 0)
        approximation = (state.basis*state.eigenvalues) @ state.basis.T
        self.assertGreater(linalg.eigvalsh(K-approximation)[0], -1e-10)

    def test_storage_counts_basis_and_arrays_are_immutable(self):
        kernel = GaussianKernel(self.rng.normal(size=(20, 3)), 0.8)
        state = initialize_rpcholesky(kernel, 4, 0.1, self.rng)
        arrays = [v for v in vars(state).values() if isinstance(v, np.ndarray)]
        self.assertEqual(state.storage_bytes, sum(array.nbytes for array in arrays))
        self.assertGreater(state.storage_bytes, state.columns.nbytes+state.basis.nbytes)
        self.assertTrue(all(not array.flags.writeable for array in arrays))

    def test_zero_budget_is_regularization_inverse_and_full_budget_has_no_candidates(self):
        kernel = GaussianKernel(self.rng.normal(size=(10, 2)), 0.5)
        v = self.rng.normal(size=10)
        zero = initialize_rpcholesky(kernel, 0, 0.2, self.rng)
        np.testing.assert_allclose(zero.apply(v), v/0.2)
        self.assertEqual(zero.rank, 0)
        full = initialize_rpcholesky(kernel, 10, 0.2, self.rng)
        for state in (zero, full):
            new, event = exchange_columns(kernel, state, v, self.rng)
            self.assertIs(new, state)
            self.assertFalse(event["accepted"])
            self.assertEqual(event["candidates"], 0)


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)

    def test_all_removal_identity_against_independent_dense_solves(self):
        Z = self.rng.normal(size=(28, 12))
        K = Z @ Z.T+0.2*np.eye(28)
        augmented = np.array([1, 4, 7, 12, 18, 21])
        C = K[:, augmented]
        W = K[np.ix_(augmented, augmented)]
        v = self.rng.normal(size=28)
        for lam in (0.1, 1e-3):
            scores, _ = _all_removal_scores(W, C.T @ C, C.T @ v, v @ v, lam)
            direct = []
            for removal in range(len(augmented)):
                indices = np.delete(augmented, removal)
                M = dense_preconditioner(K, indices, lam)
                direct.append(v @ linalg.solve(M, v, assume_a="pos"))
            np.testing.assert_allclose(scores, direct, rtol=2e-9, atol=2e-9)
            self.assertEqual(int(np.argmin(scores)), int(np.argmin(direct)))

    def test_exchange_chooses_best_removal_and_stably_verifies_acceptance(self):
        kernel = GaussianKernel(self.rng.normal(size=(26, 4)), 1.0)
        K = kernel.dense()
        indices = np.array([0, 1, 2, 3])
        state = NystromPreconditioner(indices, K[:, indices], 0.01)
        candidate = 12
        v = K[:, candidate]
        scores = []
        for removal in range(len(indices)):
            proposal = indices.copy()
            proposal[removal] = candidate
            M = dense_preconditioner(K, proposal, state.regularization)
            scores.append(v @ linalg.solve(M, v, assume_a="pos"))
        scores.append(state.score(v))
        best = int(np.argmin(scores))
        self.assertLess(best, len(indices))
        kernel.dense = lambda: self.fail("exchange requested dense K")
        new, event = exchange_columns(kernel, state, v, FixedCandidate(candidate),
                                      candidate_count=1, sampling="uniform")
        self.assertTrue(event["accepted"])
        self.assertEqual(event["exchanges"], 1)
        self.assertEqual(event["candidates"], 1)
        self.assertEqual(event["details"][0]["removed"], int(indices[best]))
        self.assertEqual(len(np.unique(new.indices)), 4)
        self.assertLess(new.score(v), state.score(v)*(1-1e-12))
        self.assertAlmostEqual(event["score_after"], min(scores), delta=1e-8)
        np.testing.assert_allclose(new.gram, new.columns.T @ new.columns, atol=1e-13)
        self.assertEqual(kernel.stats["column_evaluations"], 1)
        json.dumps(event, allow_nan=False)

    def test_residual_sampling_uses_supplied_vector_and_stops_at_zero_unselected_mass(self):
        kernel = GaussianKernel(self.rng.normal(size=(18, 3)), 0.8)
        state = initialize_rpcholesky(kernel, 3, 0.1, self.rng)
        v = np.zeros(kernel.n)
        v[state.indices[0]] = 2
        unchanged, event = exchange_columns(kernel, state, v, self.rng)
        self.assertIs(unchanged, state)
        self.assertEqual(event["candidates"], 0)
        candidate = int(np.setdiff1d(np.arange(kernel.n), state.indices)[0])
        v[:] = 0
        v[candidate] = 3
        _, event = exchange_columns(kernel, state, v, self.rng, candidate_count=1)
        self.assertEqual(event["details"][0]["candidate"], candidate)
        self.assertAlmostEqual(event["score_before"], state.score(v))

    def test_rank_deficient_candidate_uses_stable_fallback_and_can_improve(self):
        kernel = GaussianKernel(np.array([[0.], [0.], [0.], [2.], [3.], [4.]]), 1.0)
        K = kernel.dense()
        indices = np.array([0, 1, 2])
        state = NystromPreconditioner(indices, K[:, indices], 1e-3)
        v = K[:, 3]
        new, event = exchange_columns(kernel, state, v, FixedCandidate(3),
                                      candidate_count=1, sampling="uniform")
        self.assertTrue(event["accepted"])
        self.assertEqual(event["numerical_fallbacks"], 1)
        self.assertEqual(state.rank, 1)
        self.assertEqual(new.rank, 2)
        self.assertEqual(len(np.unique(new.indices)), 3)
        M = dense_preconditioner(K, new.indices, new.regularization)
        direct_score = v @ linalg.solve(M, v, assume_a="pos")
        self.assertAlmostEqual(new.score(v), direct_score, delta=1e-9)
        json.dumps(event, allow_nan=False)

    def test_equal_rank_deficient_exchange_returns_identical_state(self):
        kernel = GaussianKernel(np.zeros((8, 2)), 1.0)
        state = initialize_rpcholesky(kernel, 3, 1e-5, self.rng)
        candidate = int(np.setdiff1d(np.arange(8), state.indices)[0])
        v = self.rng.normal(size=8)
        new, event = exchange_columns(kernel, state, v, FixedCandidate(candidate),
                                      candidate_count=1, sampling="uniform")
        self.assertIs(new, state)
        self.assertFalse(event["accepted"])
        self.assertEqual(event["exchanges"], 0)
        self.assertEqual(event["score_after"], event["score_before"])


if __name__ == "__main__":
    unittest.main()
