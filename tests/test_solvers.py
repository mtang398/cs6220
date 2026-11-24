"""Solver correctness and control tests; these are not performance results."""

import unittest

import numpy as np
from scipy import linalg

from nystrom_project.numerics import GaussianKernel
from nystrom_project.solvers import METHODS, LimitedMemoryBFGS, solve


class RecordedKernel(GaussianKernel):
    """Detect dense shortcuts and retain the last actual matvec input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_matvec_input = None

    def matvec(self, v):
        self.last_matvec_input = v.copy()
        return super().matvec(v)

    def dense(self):
        raise AssertionError("a performance solver called kernel.dense()")


def problem():
    rng = np.random.default_rng(62202026)
    X = rng.normal(size=(32, 4))
    bandwidth = 1.4
    differences = X[:, None, :] - X[None, :, :]
    K = np.exp(-np.sum(differences**2, axis=2)/(2*bandwidth**2))
    y = np.sin(X[:, 0])+0.3*rng.normal(size=len(X))
    return X, bandwidth, K, y


class SolverTests(unittest.TestCase):
    def test_all_methods_solve_original_system_and_share_initial_landmarks(self):
        X, bandwidth, K, y = problem()
        regularization = 0.2
        A = K+regularization*np.eye(len(K))
        reference = linalg.solve(A, y, assume_a="pos")
        norm_y = np.linalg.norm(y)
        states = {}
        for method in METHODS:
            with self.subTest(method=method):
                kernel = RecordedKernel(X, bandwidth, block_size=9)
                kwargs = {"restart_iterations": [5, 10, 15]} if method == "matched_restart" else {}
                result = solve(kernel, y, regularization, method=method, rank=6,
                               seed=17, checkpoint=5, candidate_count=2,
                               rtol=1e-9, maxiter=400, **kwargs)
                metrics = result.metrics
                self.assertTrue(metrics["converged"], metrics)
                self.assertIsNone(metrics["breakdown"])
                measured = np.linalg.norm(y-A @ result.solution)/norm_y
                self.assertLessEqual(measured, 1.01e-9)
                self.assertAlmostEqual(metrics["true_relative_residual"], measured, delta=2e-14)
                self.assertAlmostEqual(result.history[-1]["true_relative_residual"], measured,
                                       delta=2e-14)
                np.testing.assert_allclose(result.solution, reference, rtol=2e-7, atol=2e-8)
                # A directly checked final residual must actually evaluate Kx.
                np.testing.assert_array_equal(kernel.last_matvec_input, result.solution)
                self.assertEqual(metrics["matvecs"], kernel.stats["matvecs"])
                self.assertEqual(metrics["matvecs"],
                                 metrics["iterations"]+metrics["verification_matvecs"])
                self.assertGreaterEqual(metrics["verification_matvecs"], 1)
                for row in result.history:
                    if row["iteration"] % 5 == 0:
                        self.assertIn("true_relative_residual", row)
                states[method] = metrics["initial_indices"]
        shared = states["fixed"]
        self.assertEqual(len(set(shared)), 6)
        for method in ("residual_exchange", "target_exchange", "uniform_exchange",
                       "matched_restart", "periodic_rebuild"):
            self.assertEqual(states[method], shared, method)
        self.assertEqual(states["cg"], [])
        self.assertEqual(states["lbfgs"], [])

    def test_rejected_exchanges_preserve_the_fixed_pcg_recurrence(self):
        X, bandwidth, _, y = problem()
        kwargs = dict(rank=5, seed=31, checkpoint=3, rtol=1e-10, maxiter=200)
        fixed = solve(RecordedKernel(X, bandwidth, block_size=7), y, 0.15,
                      method="fixed", **kwargs)
        self.assertTrue(fixed.metrics["converged"])
        self.assertGreater(fixed.metrics["iterations"], kwargs["checkpoint"])
        for method in ("residual_exchange", "target_exchange", "uniform_exchange"):
            with self.subTest(method=method):
                result = solve(RecordedKernel(X, bandwidth, block_size=7), y, 0.15,
                               method=method, candidate_count=0, **kwargs)
                self.assertTrue(result.metrics["converged"])
                self.assertEqual(result.metrics["restarts"], 0)
                self.assertEqual(result.metrics["iterations"], fixed.metrics["iterations"])
                self.assertEqual(result.metrics["matvecs"], fixed.metrics["matvecs"])
                self.assertEqual(result.metrics["column_evaluations"],
                                 fixed.metrics["column_evaluations"])
                self.assertEqual(result.metrics["initial_indices"], fixed.metrics["initial_indices"])
                self.assertEqual(result.metrics["final_indices"], fixed.metrics["initial_indices"])
                self.assertTrue(result.events)
                self.assertTrue(all(not event["accepted"] and event["candidates"] == 0
                                    for event in result.events))
                np.testing.assert_array_equal(result.solution, fixed.solution)
                self.assertEqual(result.history, fixed.history)

    def test_matched_restart_schedule_stops_at_earlier_convergence(self):
        X, bandwidth, _, y = problem()
        schedule = [4, 8, 100, 104]
        result = solve(RecordedKernel(X, bandwidth, block_size=8), y, 0.15,
                       method="matched_restart", rank=5, seed=21, checkpoint=4,
                       restart_iterations=schedule, rtol=1e-10, maxiter=160)
        self.assertTrue(result.metrics["converged"])
        self.assertGreater(result.metrics["iterations"], 8)
        self.assertLess(result.metrics["iterations"], 100)
        actual = [row["iteration"] for row in result.history if row.get("restarted")]
        expected = [iteration for iteration in schedule if iteration < result.metrics["iterations"]]
        self.assertEqual(actual, expected)
        self.assertEqual(result.metrics["restarts"], len(expected))
        self.assertFalse(result.history[-1].get("restarted", False))
        # With no requested restart, this control must reproduce fixed PCG.
        kwargs = dict(rank=5, seed=21, checkpoint=4, rtol=1e-10, maxiter=160)
        fixed = solve(RecordedKernel(X, bandwidth, block_size=8), y, 0.15,
                      method="fixed", **kwargs)
        unchanged = solve(RecordedKernel(X, bandwidth, block_size=8), y, 0.15,
                          method="matched_restart", restart_iterations=[], **kwargs)
        np.testing.assert_array_equal(unchanged.solution, fixed.solution)
        self.assertEqual(unchanged.history, fixed.history)
        self.assertEqual(unchanged.metrics["restarts"], 0)

    def test_iteration_limit_returns_measured_residual_and_nonconvergence(self):
        X, bandwidth, K, y = problem()
        for limit in (1, 2):
            with self.subTest(maxiter=limit):
                kernel = RecordedKernel(X, bandwidth, block_size=8)
                result = solve(kernel, y, 0.1, method="fixed", rank=4, checkpoint=4,
                               maxiter=limit, rtol=1e-12, seed=9)
                self.assertFalse(result.metrics["converged"])
                self.assertEqual(result.metrics["iterations"], limit)
                measured = np.linalg.norm(y-(K @ result.solution+0.1*result.solution))/np.linalg.norm(y)
                self.assertGreater(measured, 1e-12)
                self.assertAlmostEqual(result.metrics["true_relative_residual"], measured, delta=2e-14)
                np.testing.assert_array_equal(kernel.last_matvec_input, result.solution)
                self.assertEqual(result.metrics["verification_matvecs"], 1)
                self.assertEqual(result.metrics["matvecs"], limit+1)

    def test_zero_rhs_returns_exact_solution_without_setup_or_matvec(self):
        X, bandwidth, _, _ = problem()
        kernel = RecordedKernel(X, bandwidth)
        result = solve(kernel, np.zeros(len(X)), 0.1, method="residual_exchange", rank=5)
        np.testing.assert_array_equal(result.solution, np.zeros(len(X)))
        self.assertTrue(result.metrics["converged"])
        self.assertEqual(result.metrics["iterations"], 0)
        self.assertEqual(result.metrics["true_relative_residual"], 0)
        self.assertEqual(kernel.stats["matvecs"], 0)
        self.assertEqual(kernel.stats["column_evaluations"], 0)
        self.assertEqual(result.history, [])
        self.assertEqual(result.events, [])


class LimitedMemoryBFGSTests(unittest.TestCase):
    def test_two_loop_matches_explicit_inverse_bfgs_with_memory_eviction(self):
        rng = np.random.default_rng(6220)
        n, max_pairs = 7, 3
        Z = rng.normal(size=(n, n))
        A = Z.T @ Z+0.5*np.eye(n)
        preconditioner = LimitedMemoryBFGS(max_pairs, initial_scale=0.4)
        probe = rng.normal(size=n)
        np.testing.assert_allclose(preconditioner.apply(probe), 0.4*probe)
        pairs = []
        for _ in range(6):
            s = rng.normal(size=n)
            z = A @ s
            self.assertTrue(preconditioner.update(s, z))
            pairs.append((s.copy(), z.copy()))
            pairs = pairs[-max_pairs:]
            last_s, last_z = pairs[-1]
            H = (last_s @ last_z)/(last_z @ last_z)*np.eye(n)
            for retained_s, retained_z in pairs:
                rho = 1/(retained_s @ retained_z)
                left = np.eye(n)-rho*np.outer(retained_s, retained_z)
                H = left @ H @ left.T+rho*np.outer(retained_s, retained_s)
            observed = np.column_stack([preconditioner.apply(e) for e in np.eye(n)])
            np.testing.assert_allclose(observed, H, rtol=2e-13, atol=2e-14)
            np.testing.assert_allclose(observed, observed.T, atol=2e-14)
            self.assertGreater(linalg.eigvalsh(observed)[0], 0)
            self.assertEqual(len(preconditioner.pairs), len(pairs))
            self.assertEqual(preconditioner.storage_bytes, len(pairs)*(2*n*8+8))

    def test_unreliable_secant_pairs_leave_inverse_unchanged(self):
        preconditioner = LimitedMemoryBFGS(2, initial_scale=0.75)
        s = np.array([1., 2., 3.])
        self.assertTrue(preconditioner.update(s, np.array([2., 5., 9.])))
        before = preconditioner.apply(np.array([2., -1., 3.]))
        count = len(preconditioner.pairs)
        for bad_s, bad_z in ((s, -s), (np.zeros(3), np.zeros(3)),
                             (s, np.array([np.nan, 1., 1.]))):
            self.assertFalse(preconditioner.update(bad_s, bad_z))
            self.assertEqual(len(preconditioner.pairs), count)
            np.testing.assert_array_equal(preconditioner.apply(np.array([2., -1., 3.])), before)
        empty = LimitedMemoryBFGS(0, initial_scale=0.75)
        self.assertFalse(empty.update(s, s))
        np.testing.assert_array_equal(empty.apply(s), 0.75*s)


if __name__ == "__main__":
    unittest.main()
