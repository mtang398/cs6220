"""Float64 kernel, Nyström inverse, and fixed-budget column exchanges.

No performance operation constructs or retains an n-by-n kernel.  A kernel
matvec uses O(block_size*n) workspace.  An ordinary candidate score costs
O(n*k + k**3), using the cached column Gram matrix.  An accepted state is
rebuilt stably in O(n*k**2 + k**3), rather than claiming a rank-one update.
If the small-matrix removal identity is unreliable, the explicitly recorded
fallback builds each of the k proposed states in turn: O(n*k**3 + k**4).
All these paths use O(n*k + k**2) memory, plus the kernel matvec workspace.
Only ``GaussianKernel.dense`` is intended for small independent validations.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


_EPS = np.finfo(np.float64).eps
_ACCEPTANCE_RTOL = 1e-12


class NumericalStabilityError(np.linalg.LinAlgError):
    """A factorization or score cannot be used reliably in float64."""


def _regularization(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("regularization must be finite and positive")
    return value


def _vector(value: np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.all(np.isfinite(value)):
        raise ValueError(f"expected a finite vector with shape ({n},)")
    return value


def _rank_threshold(values: np.ndarray) -> float:
    scale = max(float(values[-1]), np.finfo(float).tiny) if values.size else 0.0
    return 32 * _EPS * max(values.size, 1) * scale


class GaussianKernel:
    """K[i,j] = exp(-||X[i]-X[j]||² / (2*bandwidth²))."""

    def __init__(self, X: np.ndarray, bandwidth: float, block_size: int = 128):
        X = np.array(X, dtype=np.float64, order="C", copy=True)
        if X.ndim != 2 or min(X.shape) < 1 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a nonempty finite two-dimensional array")
        bandwidth = float(bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be finite and positive")
        if not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.X = X
        self.X.setflags(write=False)
        self.bandwidth = bandwidth
        self.block_size = int(block_size)
        self.n = X.shape[0]
        self.stats: dict[str, int | float] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats.clear()
        self.stats.update(matvecs=0, column_evaluations=0,
                          matvec_seconds=0.0, column_seconds=0.0)

    def _block(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        # cdist avoids cancellation from ||x||² + ||y||² - 2*x.T*y.
        distances = cdist(rows, columns, metric="sqeuclidean")
        distances /= self.bandwidth
        distances /= self.bandwidth
        distances *= -0.5
        np.exp(distances, out=distances)
        return distances

    def column(self, j: int) -> np.ndarray:
        if not isinstance(j, (int, np.integer)) or not 0 <= j < self.n:
            raise IndexError("kernel column index is out of range")
        start = perf_counter()
        result = self._block(self.X, self.X[j:j+1]).reshape(self.n)
        result[j] = 1.0
        self.stats["column_evaluations"] += 1
        self.stats["column_seconds"] += perf_counter() - start
        return result

    def matvec(self, v: np.ndarray) -> np.ndarray:
        v = _vector(v, self.n)
        start = perf_counter()
        result = np.empty(self.n, dtype=np.float64)
        for first in range(0, self.n, self.block_size):
            last = min(first+self.block_size, self.n)
            block = self._block(self.X[first:last], self.X)
            result[first:last] = block @ v
        self.stats["matvecs"] += 1
        self.stats["matvec_seconds"] += perf_counter() - start
        return result

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Evaluate a requested kernel submatrix, counting every queried entry."""
        rows, columns = np.asarray(rows), np.asarray(columns)
        for indices in (rows, columns):
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= self.n)):
                raise IndexError('kernel entry indices must be one-dimensional integers in range')
        started = perf_counter()
        block = self._block(self.X[rows], self.X[columns])
        self.stats['entry_blocks'] = self.stats.get('entry_blocks', 0)+1
        self.stats['entry_evaluations'] = self.stats.get('entry_evaluations', 0)+rows.size*columns.size
        self.stats['entry_seconds'] = self.stats.get('entry_seconds', 0.)+perf_counter()-started
        return block

    def dense(self) -> np.ndarray:
        """Construct the dense kernel for validation; does not count as matvecs."""
        result = self._block(self.X, self.X)
        np.fill_diagonal(result, 1.0)
        return result

    def matmat(self, vectors: np.ndarray) -> np.ndarray:
        """Apply K to several vectors, sharing kernel evaluation in each block.

        Matrix applications and their number of right-hand sides are recorded
        separately from scalar matvecs. No n-by-n kernel is retained.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != self.n or not np.all(np.isfinite(vectors)):
            raise ValueError("expected a finite n-by-r matrix")
        start = perf_counter()
        result = np.empty_like(vectors)
        for first in range(0, self.n, self.block_size):
            last = min(first+self.block_size, self.n)
            result[first:last] = self._block(self.X[first:last], self.X) @ vectors
        self.stats["matmats"] = self.stats.get("matmats", 0)+1
        self.stats["matmat_rhs"] = self.stats.get("matmat_rhs", 0)+vectors.shape[1]
        self.stats["matmat_seconds"] = self.stats.get("matmat_seconds", 0.0)+perf_counter()-start
        return result


class NystromPreconditioner:
    """Stable inverse of C W† C.T + regularization*I.

    Eigenvalues of W below a documented relative float64 threshold are
    truncated; ``rank`` is the numerical rank of the retained factor, while
    ``indices`` always contains the full requested column budget.  Read-only
    arrays prevent accidental invalidation of the cached Gram matrix.

    ``factor`` is a trusted existing Nyström factor, used internally to reuse
    RPCholesky's work.  It is reused when its column count matches W's numerical
    rank; otherwise the rank-revealing W decomposition rebuilds the factor.
    """

    def __init__(self, indices: np.ndarray, columns: np.ndarray,
                 regularization: float, *, factor: np.ndarray | None = None,
                 gram: np.ndarray | None = None,
                 diagnostics: dict[str, Any] | None = None):
        self.regularization = _regularization(regularization)
        raw_indices = np.asarray(indices)
        if raw_indices.ndim != 1 or (raw_indices.size and not np.issubdtype(
                raw_indices.dtype, np.integer)):
            raise ValueError("indices must be a one-dimensional integer array")
        self.indices = np.array(raw_indices, dtype=np.int64, copy=True)
        self.columns = np.array(columns, dtype=np.float64, order="C", copy=True)
        if (self.columns.ndim != 2 or self.columns.shape[0] < 1
                or self.columns.shape[1] != self.indices.size
                or not np.all(np.isfinite(self.columns))):
            raise ValueError("columns must be a finite n-by-k array")
        self.n = self.columns.shape[0]
        k = self.indices.size
        if (np.unique(self.indices).size != k or np.any(self.indices < 0)
                or np.any(self.indices >= self.n)):
            raise ValueError("retained indices must be unique and in range")
        core = self.columns[self.indices, :]
        if not np.allclose(core, core.T, rtol=1e-12, atol=1e-14):
            raise ValueError("selected kernel core must be symmetric")
        self.core = (core+core.T)*0.5
        self._gram = None
        if gram is not None:
            self._gram = np.array(gram, dtype=np.float64, copy=True)
            if self._gram.shape != (k, k) or not np.all(np.isfinite(self._gram)):
                raise ValueError("gram must be a finite k-by-k array")
            self._gram = (self._gram+self._gram.T)*0.5
            self._gram.setflags(write=False)
        self.diagnostics = dict(diagnostics or {})
        values, vectors = linalg.eigh(self.core, check_finite=False)
        threshold = _rank_threshold(values)
        if values.size and values[0] < -max(threshold, 64*_EPS):
            raise NumericalStabilityError("selected kernel core is not numerically PSD")
        keep = values > threshold
        core_rank = int(np.count_nonzero(keep))
        if factor is not None:
            factor = np.asarray(factor, dtype=np.float64)
            if (factor.ndim != 2 or factor.shape[0] != self.n
                    or not np.all(np.isfinite(factor))):
                raise ValueError("factor must be a finite n-by-r array")
        reuse = factor is not None and factor.shape[1] == core_rank
        if not reuse:
            factor = (self.columns @ vectors[:, keep]) / np.sqrt(values[keep])
        assert factor is not None
        if factor.shape[1]:
            basis, singular_values, _ = linalg.svd(
                factor, full_matrices=False, check_finite=False)
            factor_threshold = 32*_EPS*max(factor.shape)*singular_values[0]
            retain = singular_values > factor_threshold
            self.basis = np.ascontiguousarray(basis[:, retain])
            self.eigenvalues = singular_values[retain]**2
        else:
            factor_threshold = 0.0
            self.basis = np.empty((self.n, 0), dtype=np.float64)
            self.eigenvalues = np.empty(0, dtype=np.float64)
        self.rank = self.eigenvalues.size
        self.diagnostics.update(
            requested_columns=int(k), core_rank=core_rank,
            effective_rank=int(self.rank), core_eigenvalue_threshold=float(threshold),
            factor_singular_value_threshold=float(factor_threshold),
            core_truncated=bool(core_rank < k), reused_factor=bool(reuse),
            factor_source="provided_factor" if reuse else "core_eigendecomposition")
        for array in (self.indices, self.columns, self.core,
                      self.basis, self.eigenvalues):
            array.setflags(write=False)

    def _components(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected = self.basis.T @ v
        complement = v - self.basis @ projected
        # A second projection reduces leakage along the low-rank span when
        # 1/lambda amplifies small orthogonality errors.
        correction = self.basis.T @ complement
        projected = projected + correction
        complement = complement - self.basis @ correction
        return projected, complement

    def apply(self, v: np.ndarray) -> np.ndarray:
        v = _vector(v, self.n)
        projected, complement = self._components(v)
        return (complement/self.regularization
                + self.basis @ (projected/(self.eigenvalues+self.regularization)))

    def score(self, v: np.ndarray) -> float:
        """Positive span/complement expression for v.T M^{-1} v."""
        v = _vector(v, self.n)
        projected, complement = self._components(v)
        return float(complement @ complement/self.regularization
                     + np.sum(projected**2/(self.eigenvalues+self.regularization)))

    @property
    def gram(self) -> np.ndarray:
        """Exchange-only cache; its first construction is charged to updates."""
        if self._gram is None:
            self._gram = self.columns.T @ self.columns
            self._gram.setflags(write=False)
        return self._gram

    @property
    def storage_bytes(self) -> int:
        """Bytes of all arrays retained by this state, including its basis."""
        return sum(array.nbytes for array in (self.indices, self.columns, self.core,
                                              self.basis, self.eigenvalues)) + (
                                                  0 if self._gram is None else self._gram.nbytes)


def initialize_rpcholesky(kernel: GaussianKernel, k: int, regularization: float,
                         rng: np.random.Generator, *, pivoting="random") -> NystromPreconditioner:
    """Select exactly k unique columns; permit smaller numerical factor rank.

    Pivots are sampled proportionally to the Cholesky residual diagonal.  If
    its remaining mass is numerically zero, unused indices fill the column
    budget uniformly without adding spurious directions to the factor.
    """
    regularization = _regularization(regularization)
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= kernel.n:
        raise ValueError("k must be an integer between zero and kernel.n")
    if pivoting not in ("random", "greedy"):
        raise ValueError("pivoting must be random or greedy")
    n = kernel.n
    indices = np.empty(k, dtype=np.int64)
    columns = np.empty((n, k), dtype=np.float64)
    factor = np.zeros((n, k), dtype=np.float64)
    diagonal = np.array(getattr(kernel, "diagonal", np.ones(n)), dtype=np.float64, copy=True)
    if diagonal.shape != (n,) or not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0):
        raise ValueError("kernel diagonal must be finite and nonnegative")
    selected = np.zeros(n, dtype=bool)
    rank, uniform_fill, skipped = 0, 0, 0
    tolerance = 64*_EPS*max(k, 1)*max(float(diagonal.max()), np.finfo(float).tiny)
    for step in range(k):
        diagonal[selected] = 0.0
        mass = float(np.sum(diagonal))
        if mass > tolerance:
            j = int(np.argmax(diagonal)) if pivoting == "greedy" else int(rng.choice(n, p=diagonal/mass))
        else:
            j = int(rng.choice(np.flatnonzero(~selected)))
            uniform_fill += 1
        indices[step] = j
        selected[j] = True
        column = kernel.column(j)
        columns[:, step] = column
        remainder = column - factor[:, :rank] @ factor[j, :rank]
        pivot = float(remainder[j])
        if pivot > tolerance:
            factor[:, rank] = remainder/np.sqrt(pivot)
            diagonal = np.maximum(diagonal-factor[:, rank]**2, 0.0)
            rank += 1
        else:
            skipped += 1
        diagonal[selected] = 0.0
    return NystromPreconditioner(
        indices, columns, regularization, factor=factor[:, :rank],
        diagnostics=dict(initialization="rpcholesky" if pivoting == "random" else "pivoted_cholesky", rp_factor_rank=rank,
                         rp_uniform_fill=uniform_fill, rp_skipped_pivots=skipped,
                         rp_pivot_tolerance=float(tolerance)))


def _all_removal_scores(core: np.ndarray, gram: np.ndarray, projections: np.ndarray,
                        norm_squared: float, regularization: float
                        ) -> tuple[np.ndarray, dict[str, float]]:
    """Score each principal deletion using one H factorization and solves.

    The last deletion is the unchanged set when the candidate is last.
    Reject singular/poorly conditioned systems; a pseudoinverse cannot simply
    replace H^{-1} in the principal-deletion identity.
    """
    lam = _regularization(regularization)
    size = core.shape[0]
    core_values = linalg.eigvalsh(core, check_finite=False)
    if not core_values.size or core_values[0] <= _rank_threshold(core_values):
        raise NumericalStabilityError("augmented kernel core is numerically rank deficient")
    H = core + gram/lam
    H = (H+H.T)*0.5
    values = linalg.eigvalsh(H, check_finite=False)
    if values[0] <= 0 or values[0]/values[-1] < 1e-12:
        raise NumericalStabilityError("augmented score system is ill conditioned")
    lower = linalg.cholesky(H, lower=True, check_finite=False)
    z = linalg.solve_triangular(lower, projections, lower=True, check_finite=False)
    u = linalg.solve_triangular(lower.T, z, lower=False, check_finite=False)
    # The squared column norms of L^{-1} give diag(H^{-1}); no dense inverse
    # is formed or applied in the solver.
    lower_inverse = linalg.solve_triangular(
        lower, np.eye(size), lower=True, check_finite=False)
    inverse_diagonal = np.sum(lower_inverse**2, axis=0)
    scores = norm_squared/lam - (z @ z-u*u/inverse_diagonal)/lam**2
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise NumericalStabilityError("removal score lost positivity or finiteness")
    condition = float(values[-1]/values[0])
    # A diagnostic bound for loss of absolute precision through cancellation.
    cancellation_scale = float(norm_squared/lam)
    return scores, dict(condition=condition, cancellation_scale=cancellation_scale)


def _replacement(state: NystromPreconditioner, candidate: int, column: np.ndarray,
                 removal: int, gram_augmented: np.ndarray) -> NystromPreconditioner:
    indices = state.indices.copy()
    indices[removal] = candidate
    columns = state.columns.copy()
    columns[:, removal] = column
    # Order the augmented Gram to match replacement in the old slot.
    order = np.arange(state.indices.size)
    order[removal] = state.indices.size
    gram = gram_augmented[np.ix_(order, order)]
    return NystromPreconditioner(indices, columns, state.regularization, gram=gram)


def _strict_decrease(before: float, after: float) -> bool:
    return bool(np.isfinite(after) and after >= 0
                and after < before-_ACCEPTANCE_RTOL*abs(before))


def exchange_columns(kernel: GaussianKernel, state: NystromPreconditioner,
                     vector: np.ndarray, rng: np.random.Generator,
                     candidate_count: int = 4, sampling: str = "residual"
                     ) -> tuple[NystromPreconditioner, dict[str, Any]]:
    """Evaluate sequential candidates, including every deletion and the no-op.

    ``vector`` supplies BOTH sampling weights and the fixed score at this
    checkpoint. Residual sampling stops if all unselected weights vanish.
    Each sampling draw is over the current unselected set; a rejected column
    can be drawn again.  An accepted replacement is verified with the stable
    factor-based inverse that the solver will use.  No accepted replacement
    returns a distinct state object unless its verified score strictly drops.
    """
    if kernel.n != state.n:
        raise ValueError("kernel and preconditioner sizes differ")
    if sampling not in ("residual", "uniform"):
        raise ValueError("sampling must be 'residual' or 'uniform'")
    if (not isinstance(candidate_count, (int, np.integer)) or candidate_count < 0):
        raise ValueError("candidate_count must be a nonnegative integer")
    vector = _vector(vector, kernel.n)
    before = state.score(vector)
    if not np.isfinite(before):
        raise ValueError("the scoring vector produces a nonfinite quadratic form")
    event: dict[str, Any] = dict(
        accepted=False, exchanges=0, score_before=before, score_after=before,
        candidates=0, numerical_rejections=0, numerical_fallbacks=0, details=[])
    k = state.indices.size
    if not k or k == kernel.n or not candidate_count or before == 0:
        return state, event
    current = state
    score = before
    norm_squared = float(vector @ vector)
    for _ in range(candidate_count):
        available = np.ones(kernel.n, dtype=bool)
        available[current.indices] = False
        choices = np.flatnonzero(available)
        if sampling == "residual":
            weights = np.abs(vector[choices])
            scale = float(np.max(weights))
            if scale == 0:
                break
            weights = (weights/scale)**2
            candidate = int(rng.choice(choices, p=weights/np.sum(weights)))
        else:
            candidate = int(rng.choice(choices))
        column = kernel.column(candidate)
        cross = current.columns.T @ column
        gram = np.empty((k+1, k+1), dtype=np.float64)
        gram[:k, :k] = current.gram
        gram[:k, k] = gram[k, :k] = cross
        gram[k, k] = column @ column
        core = np.empty_like(gram)
        core[:k, :k] = current.core
        core[:k, k] = core[k, :k] = column[current.indices]
        core[k, k] = column[candidate]
        projections = np.empty(k+1)
        projections[:k] = current.columns.T @ vector
        projections[k] = column @ vector
        event["candidates"] += 1
        detail: dict[str, Any] = dict(candidate=candidate, score_before=score,
                                      accepted=False, scoring="identity")
        fallback_reason = None
        try:
            scores, diagnostic = _all_removal_scores(
                core, gram, projections, norm_squared, current.regularization)
            detail.update(diagnostic)
            # The no-op is an independent stable value already available.
            discrepancy = abs(float(scores[k])-score)
            tolerance = 1e-8*max(score, np.finfo(float).tiny)
            if discrepancy > tolerance:
                raise NumericalStabilityError("identity no-op disagrees with stable score")
            scores[k] = score
            removal = int(np.argmin(scores))
            replacement = None
            if removal < k and _strict_decrease(score, float(scores[removal])):
                replacement = _replacement(current, candidate, column, removal, gram)
                verified = replacement.score(vector)
                detail.update(predicted_score=float(scores[removal]), verified_score=verified)
                if abs(float(scores[removal])-verified) > 1e-8*max(
                        verified, np.finfo(float).tiny):
                    raise NumericalStabilityError(
                        "predicted replacement disagrees with stable score")
                if not _strict_decrease(score, verified):
                    event["numerical_rejections"] += 1
                    detail["reason"] = "predicted decrease failed stable verification"
                    replacement = None
            elif removal < k:
                removal = k
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            fallback_reason = str(error)
            replacement = None
            removal = k
        if fallback_reason is not None:
            event["numerical_fallbacks"] += 1
            detail.update(scoring="direct_fallback", reason=fallback_reason)
            best_score = score
            invalid = 0
            for trial_removal in range(k):
                try:
                    trial = _replacement(current, candidate, column, trial_removal, gram)
                    trial_score = trial.score(vector)
                    if _strict_decrease(best_score, trial_score):
                        replacement, removal, best_score = trial, trial_removal, trial_score
                except (np.linalg.LinAlgError, FloatingPointError):
                    invalid += 1
            detail["invalid_removals"] = invalid
            if invalid:
                event["numerical_rejections"] += 1
            if replacement is not None:
                detail["verified_score"] = best_score
        if replacement is not None:
            previous_score = score
            score = replacement.score(vector)
            if not _strict_decrease(previous_score, score):
                raise NumericalStabilityError("accepted state failed repeated score verification")
            detail.update(accepted=True, removed=int(current.indices[removal]),
                          score_after=score, effective_rank=int(replacement.rank))
            current = replacement
            event["exchanges"] += 1
        else:
            detail["score_after"] = score
        event["details"].append(detail)
    event.update(accepted=bool(event["exchanges"]), score_after=score)
    return current, event
