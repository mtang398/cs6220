"""Micro tests for flexible recurrences, projections, and cost accounting."""

import unittest
import weakref

import numpy as np
from scipy import linalg

from nystrom_project.flexible import AConjugateHistory, flexible_solve
from nystrom_project.numerics import GaussianKernel, NystromPreconditioner, initialize_rpcholesky


class DenseKernel:
    """Independent tiny dense operator; dense() remains forbidden to the solver."""

    def __init__(self, K):
        self.K = np.asarray(K)
        self.n = len(K)
        self.stats = dict(matvecs=0, column_evaluations=0,
                          matvec_seconds=0.0, column_seconds=0.0)

    def matvec(self, v):
        self.stats["matvecs"] += 1
        return self.K @ v

    def dense(self):
        raise AssertionError("flexible solver requested a dense kernel")


class DiagonalInverse:
    def __init__(self, diagonal):
        self.diagonal = np.asarray(diagonal)

    def apply(self, v):
        return self.diagonal*v

    @property
    def storage_bytes(self):
        return self.diagonal.nbytes


class HistoryTests(unittest.TestCase):
    def test_vector_and_matrix_projection_match_independent_dense_projection(self):
        rng = np.random.default_rng(6220)
        n, count = 13, 4
        V = rng.normal(size=(n, n))
        A = V.T @ V+np.eye(n)
        values, vectors = linalg.eigh(A)
        Q = vectors[:, :count]/np.sqrt(values[:count])
        history = AConjugateHistory(n, count)
        for q in Q.T:
            history.append(q, A @ q)
        Z = rng.normal(size=(n, 5))
        AZ = A @ Z
        expected = Z-Q @ linalg.solve(Q.T @ A @ Q, Q.T @ AZ, assume_a="pos")
        p, ap = history.project(Z, AZ)
        np.testing.assert_allclose(p, expected, rtol=1e-12, atol=2e-13)
        np.testing.assert_allclose(ap, A @ p, rtol=1e-12, atol=2e-13)
        np.testing.assert_allclose(Q.T @ ap, 0, atol=2e-13)
        np.testing.assert_array_equal(AZ, A @ Z)
        for j in range(Z.shape[1]):
            p_single, ap_single = history.project(Z[:, j], AZ[:, j])
            np.testing.assert_allclose(p_single, p[:, j], atol=2e-14)
            np.testing.assert_allclose(ap_single, ap[:, j], atol=2e-13)
        calls = []
        def matvec(v):
            calls.append(v.copy())
            return A @ v
        history.form_direction(Z[:, 0], matvec)
        self.assertEqual(len(calls), 1)
        self.assertEqual(history.storage_bytes, count*(2*n*8+8))

    def test_sliding_window_snapshot_is_independent_and_preserves_projection(self):
        n = 8
        A = np.diag(np.arange(1., n+1))
        history = AConjugateHistory(n, 3)
        for i in range(5):
            p = np.eye(n)[:, i]
            history.append(p, A @ p)
        self.assertEqual(history.size, 3)
        snapshot = history.snapshot()
        restored = AConjugateHistory.from_snapshot(snapshot)
        z = np.arange(1., n+1)
        old_p, old_ap = history.project(z, A @ z)
        copy_p, copy_ap = restored.project(z, A @ z)
        np.testing.assert_array_equal(copy_p, old_p)
        np.testing.assert_array_equal(copy_ap, old_ap)
        self.assertEqual(snapshot.storage_bytes, history.storage_bytes)
        for p, ap, _ in snapshot.pairs:
            self.assertFalse(p.flags.writeable)
            self.assertFalse(ap.flags.writeable)
        history.clear()
        self.assertEqual(history.size, 0)
        self.assertEqual(restored.size, 3)
        self.assertEqual(len(snapshot.pairs), 3)


class FlexibleSolverTests(unittest.TestCase):
    def test_one_direction_matches_flexible_short_recurrence(self):
        rng = np.random.default_rng(501)
        n, steps, lam = 16, 7, 0.1
        V = rng.normal(size=(n, n))
        A = V.T @ V+2*np.eye(n)
        b = rng.normal(size=n)
        preconditioners = [DiagonalInverse(np.exp(rng.normal(scale=0.3, size=n)))
                           for _ in range(steps)]
        observed = []
        def update(x, residual, old, iteration, history):
            return preconditioners[iteration], {"accepted": True}
        result = flexible_solve(
            DenseKernel(A-lam*np.eye(n)), b, lam, preconditioner=preconditioners[0],
            history_size=1, checkpoint=1, maxiter=steps, rtol=1e-14,
            update_callback=update,
            step_callback=lambda x, r, i, h: observed.append(x))
        # Independent beta=(z_k.T (r_k-r_{k-1}))/(z_{k-1}.T r_{k-1})
        # implementation, rather than calling the history projection code.
        x, r = np.zeros(n), b.copy()
        previous_r = previous_p = None
        previous_rho = None
        for iteration in range(steps):
            z = preconditioners[iteration].apply(r)
            rho = float(r @ z)
            if iteration:
                beta = float(z @ (r-previous_r))/previous_rho
                p = z+beta*previous_p
            else:
                p = z.copy()
            ap = A @ p
            alpha = rho/(p @ ap)
            previous_r, previous_p, previous_rho = r.copy(), p.copy(), rho
            x += alpha*p
            r -= alpha*ap
            np.testing.assert_allclose(observed[iteration], x, rtol=2e-11, atol=2e-12)
        self.assertEqual(result.metrics["restarts"], 0)
        self.assertEqual(result.metrics["accepted_iterations"], list(range(1, steps)))
        self.assertTrue(all(e["history_preserved"] for e in result.events))

    def test_full_history_with_changing_spd_preconditioners_converges_and_is_conjugate(self):
        rng = np.random.default_rng(920)
        n, lam = 17, 0.2
        V = rng.normal(size=(n, n))
        A = V.T @ V+np.eye(n)
        b = rng.normal(size=n)
        reference = linalg.solve(A, b, assume_a="pos")
        preconditioners = [DiagonalInverse(np.exp(rng.normal(scale=0.4, size=n)))
                           for _ in range(n+1)]
        energies, histories = [], []
        def inspect(x, residual, iteration, history):
            error = x-reference
            energies.append(float(error @ (A @ error)))
            snapshot = history.snapshot()
            P = np.column_stack([p for p, _, _ in snapshot.pairs])
            AP = np.column_stack([ap for _, ap, _ in snapshot.pairs])
            np.testing.assert_allclose(P.T @ AP, np.eye(history.size), atol=2e-12)
            np.testing.assert_allclose(AP, A @ P, rtol=2e-11, atol=2e-12)
            histories.append(snapshot)
        def update(x, residual, old, iteration, history):
            return preconditioners[iteration], {"accepted": True}
        result = flexible_solve(
            DenseKernel(A-lam*np.eye(n)), b, lam, preconditioner=preconditioners[0],
            history_size=n, checkpoint=1, maxiter=n, rtol=1e-10,
            update_callback=update, step_callback=inspect)
        self.assertTrue(result.metrics["converged"], result.metrics)
        self.assertLessEqual(result.metrics["iterations"], n)
        self.assertEqual(result.metrics["restarts"], 0)
        np.testing.assert_allclose(result.solution, reference, rtol=2e-10, atol=2e-11)
        self.assertTrue(np.all(np.diff(energies) <= 1e-12))
        initial_energy = float(reference @ (A @ reference))
        measured = -np.diff([initial_energy]+energies)
        predicted = [row["predicted_energy_decrease"] for row in result.history[1:]]
        np.testing.assert_allclose(measured, predicted, rtol=1e-10, atol=1e-12)
        self.assertEqual(result.metrics["matvecs"],
                         result.metrics["direction_matvecs"]+result.metrics["verification_matvecs"])

    def test_fixed_and_rejected_update_controls_have_identical_history_budget(self):
        rng = np.random.default_rng(22)
        X = rng.normal(size=(28, 3))
        kernel = GaussianKernel(X, 1.1, block_size=7)
        preconditioner = initialize_rpcholesky(kernel, 5, 0.2, rng)
        setup_columns = kernel.stats["column_evaluations"]
        b = rng.normal(size=28)
        fixed = flexible_solve(kernel, b, 0.2, preconditioner=preconditioner,
                               history_size=4, checkpoint=3, rtol=1e-10)
        self.assertTrue(fixed.metrics["converged"])
        self.assertEqual(fixed.metrics["column_evaluations"], 0)
        self.assertEqual(kernel.stats["column_evaluations"], setup_columns)
        other = GaussianKernel(X, 1.1, block_size=7)
        calls = []
        def rejected(x, residual, old, iteration, history):
            z = old.apply(residual)
            az = other.matvec(z)+0.2*z
            history.project(z, az)
            calls.append(iteration)
            return old, {"accepted": False, "exchanges": 0}
        unchanged = flexible_solve(other, b, 0.2, preconditioner=preconditioner,
                                   history_size=4, checkpoint=3, rtol=1e-10,
                                   update_callback=rejected)
        np.testing.assert_array_equal(unchanged.solution, fixed.solution)
        self.assertEqual(unchanged.history, fixed.history)
        self.assertTrue(calls)
        self.assertEqual(unchanged.metrics["update_matvecs"], len(calls))
        self.assertEqual(unchanged.metrics["matvecs"], fixed.metrics["matvecs"]+len(calls))
        self.assertEqual(unchanged.metrics["max_history_vectors"], 8)
        self.assertEqual(unchanged.metrics["max_history_bytes"], 4*(2*len(X)*8+8))
        self.assertEqual(fixed.metrics["max_history_bytes"], unchanged.metrics["max_history_bytes"])

    def test_bounded_history_allows_actual_nystrom_changes_without_restart(self):
        rng = np.random.default_rng(82)
        X = rng.normal(size=(30, 4))
        kernel = GaussianKernel(X, 1.0, block_size=8)
        K = kernel.dense()
        lam = 0.3
        preconditioners = []
        for indices in (np.arange(5), np.arange(5, 10), np.arange(10, 15)):
            preconditioners.append(NystromPreconditioner(indices, K[:, indices], lam))
        b = rng.normal(size=len(X))
        calls = []
        def update(x, residual, old, iteration, history):
            self.assertGreater(history.size, 0)
            new = preconditioners[(iteration//3) % len(preconditioners)]
            calls.append(iteration)
            return new, {"accepted": new is not old, "exchanges": int(new is not old)}
        kernel.dense = lambda: self.fail("performance solve called dense()")
        result = flexible_solve(kernel, b, lam, preconditioner=preconditioners[0],
                                history_size=3, checkpoint=3, maxiter=200, rtol=1e-9,
                                update_callback=update)
        self.assertTrue(result.metrics["converged"], result.metrics)
        self.assertGreater(len(calls), 1)
        self.assertEqual(result.metrics["restarts"], 0)
        self.assertTrue(all(e["history_preserved"] for e in result.events))
        self.assertLessEqual(result.metrics["max_retained_directions"], 3)
        reference = linalg.solve(K+lam*np.eye(len(X)), b, assume_a="pos")
        np.testing.assert_allclose(result.solution, reference, rtol=3e-8, atol=3e-8)
        self.assertEqual(result.metrics["direction_matvecs"], result.metrics["iterations"])
        measured = np.linalg.norm(b-(K @ result.solution+lam*result.solution))/np.linalg.norm(b)
        self.assertAlmostEqual(result.metrics["true_relative_residual"], measured, delta=2e-14)

    def test_zero_history_is_steepest_descent_and_uses_no_retained_vectors(self):
        n, lam = 10, 0.1
        A = np.diag(np.linspace(1., 2., n))
        b = np.arange(1., n+1)
        path = []
        result = flexible_solve(DenseKernel(A-lam*np.eye(n)), b, lam,
                                history_size=0, checkpoint=5, maxiter=100, rtol=1e-10,
                                step_callback=lambda x, r, i, h: path.append(x))
        expected_first = b*(b @ b)/(b @ A @ b)
        np.testing.assert_allclose(path[0], expected_first, atol=2e-15)
        self.assertTrue(result.metrics["converged"])
        self.assertEqual(result.metrics["max_history_vectors"], 0)
        self.assertEqual(result.metrics["max_history_bytes"], 0)
        np.testing.assert_allclose(result.solution, b/np.diag(A), rtol=1e-9)

    def test_factory_counts_setup_and_does_not_retain_initial_preconditioner(self):
        rng = np.random.default_rng(721)
        kernel = GaussianKernel(rng.normal(size=(28, 3)), 1.0, block_size=8)
        b = rng.normal(size=kernel.n)
        created = []
        def factory():
            state = initialize_rpcholesky(kernel, 4, .2, np.random.default_rng(1))
            created.append(weakref.ref(state))
            return state
        checkpoints = []
        def update(x, residual, old, iteration, history):
            checkpoints.append(iteration)
            if len(checkpoints) == 1:
                return initialize_rpcholesky(kernel, 4, .2, np.random.default_rng(2)), {"accepted": True}
            self.assertIsNone(created[0](), "solver retained its initial preconditioner")
            return old, {"accepted": False}
        result = flexible_solve(kernel, b, .2, preconditioner_factory=factory,
                                history_size=3, checkpoint=2, rtol=1e-9,
                                update_callback=update)
        self.assertTrue(result.metrics["converged"])
        self.assertGreater(len(checkpoints), 1)
        self.assertTrue(result.metrics["initialization_included"])
        self.assertGreater(result.metrics["setup_seconds"], 0)
        self.assertEqual(result.metrics["column_evaluations"], 8)
        self.assertEqual(kernel.stats["column_evaluations"], 8)
        self.assertIsNone(created[0]())
        with self.assertRaises(ValueError):
            flexible_solve(kernel, b, .2, preconditioner=DiagonalInverse(np.ones(kernel.n)),
                           preconditioner_factory=factory)

    def test_initial_guess_zero_rhs_and_iteration_cap_counts(self):
        A = np.diag([1., 2., 4., 8.])
        b = np.ones(4)
        for x0, expected in ((None, 2), (np.full(4, .1), 3)):
            result = flexible_solve(DenseKernel(A-.1*np.eye(4)), b, .1,
                                    history_size=2, x0=x0, maxiter=1, rtol=1e-14)
            self.assertFalse(result.metrics["converged"])
            self.assertEqual(result.metrics["matvecs"], expected)
            self.assertEqual(result.metrics["verification_matvecs"], 1)
            self.assertEqual(result.metrics["direction_matvecs"], 1)
        exact = flexible_solve(DenseKernel(A-.1*np.eye(4)), b, .1, x0=1/np.diag(A))
        self.assertTrue(exact.metrics["converged"])
        self.assertEqual(exact.metrics["iterations"], 0)
        self.assertEqual(exact.metrics["matvecs"], 1)
        zero = flexible_solve(DenseKernel(A-.1*np.eye(4)), np.zeros(4), .1)
        self.assertTrue(zero.metrics["converged"])
        self.assertEqual(zero.metrics["matvecs"], 0)
        self.assertEqual(zero.metrics["iterations"], 0)


if __name__ == "__main__":
    unittest.main()
