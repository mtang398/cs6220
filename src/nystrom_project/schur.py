"""Landmark Schur complement with a sparse factorized approximate inverse.

For selected indices S first in the permutation, A=K+lambda I has blocks

    A = [L 0; B I] diag(I,D) [L.T B.T; 0 I],
    L L.T = A_SS, B = A_RS L^{-T}, D = A_RR-B B.T.

Each lower-triangular row of G uses its diagonal and the nearest earlier
remaining points. On its pattern J, solve D_JJ y=e_last and set g=y/sqrt(y_last).
Replacing D^{-1} by G.T G preserves positive definiteness. Only local kernel
blocks are queried; neither D nor the full kernel is formed.

Retained numerical storage is O(n*k+n*b+k**2), excluding the kernel's input
features and ordinary solver work vectors. Setup uses rowwise neighbor searches
(O(n**2*d) distance arithmetic, O(n*d) feature/row workspace), RPCholesky, and
local b-by-b solves. An application costs O(n*k+n*b+k**2) without a kernel call.
The factory calling initialization owns its setup timer and kernel counters.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.spatial.distance import cdist

from .numerics import NumericalStabilityError, initialize_rpcholesky


class SchurFSAIPreconditioner:
    """Fixed SPD inverse using L, B, and a lower-triangular CSR factor G.

    The constructor takes ownership of the provided numerical buffers and marks
    them read-only. It shares already prepared factors rather than copying them;
    callers must not mutate aliases. Construct a new state when replacing G.
    ``indices`` and ``remaining_indices`` are views into ``permutation``.
    CSR row patterns index the remaining block, not original training rows.
    ``rank`` is the number of landmark indices, including duplicate data points.
    """

    def __init__(self, permutation, L, B, G, regularization, *, diagnostics=None):
        lam = float(regularization)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("regularization must be finite and positive")
        raw_permutation = np.asarray(permutation)
        if (raw_permutation.ndim != 1 or raw_permutation.size == 0
                or not np.issubdtype(raw_permutation.dtype, np.integer)):
            raise ValueError("permutation must be a nonempty integer vector")
        self.permutation = np.asarray(raw_permutation, dtype=np.int64)
        self.n = self.permutation.size
        if not np.array_equal(np.sort(self.permutation), np.arange(self.n)):
            raise ValueError("permutation must contain every training index exactly once")
        self.L = np.asarray(L, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        if self.L.ndim != 2 or self.L.shape[0] != self.L.shape[1]:
            raise ValueError("L must be square")
        self.rank = self.L.shape[0]
        remaining = self.n-self.rank
        if remaining < 0 or self.B.shape != (remaining, self.rank):
            raise ValueError("B must have shape (n-rank,rank)")
        if (not np.all(np.isfinite(self.L)) or not np.all(np.isfinite(self.B))
                or np.any(np.diag(self.L) <= 0) or np.any(np.triu(self.L, 1) != 0)):
            raise ValueError("L must be finite lower triangular with positive diagonal; B must be finite")
        if not sparse.issparse(G):
            raise ValueError("G must be a sparse matrix")
        self.G = G.tocsr(copy=False).astype(np.float64, copy=False)
        if (self.G.shape != (remaining, remaining) or not self.G.has_canonical_format
                or not np.all(np.isfinite(self.G.data))):
            raise ValueError("G must be finite canonical CSR of shape (n-rank,n-rank)")
        for row in range(remaining):
            first, last = self.G.indptr[row:row+2]
            columns = self.G.indices[first:last]
            if (columns.size == 0 or columns[-1] != row or np.any(columns < 0) or np.any(columns > row)
                    or self.G.data[last-1] <= 0):
                raise ValueError("G must be lower triangular with a positive diagonal in every row")
        self.regularization = lam
        self.diagnostics = dict(diagnostics or {})
        for array in (self.permutation, self.L, self.B,
                      self.G.data, self.G.indices, self.G.indptr):
            array.setflags(write=False)
        # CSR transpose is a CSC view with the same three numerical buffers.
        self._GT = self.G.transpose(copy=False)

    @property
    def indices(self):
        return self.permutation[:self.rank]

    @property
    def remaining_indices(self):
        return self.permutation[self.rank:]

    def _row_bounds(self, row):
        if not isinstance(row, (int, np.integer)) or not 0 <= row < self.n-self.rank:
            raise IndexError("remaining-block row index is out of range")
        return self.G.indptr[row:row+2]

    def row_pattern(self, row):
        """Read-only CSR column-index view in remaining-block coordinates."""
        first, last = self._row_bounds(row)
        return self.G.indices[first:last]

    def row_values(self, row):
        """Read-only view of the matching nonzero row coefficients."""
        first, last = self._row_bounds(row)
        return self.G.data[first:last]

    def apply(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.n,) or not np.all(np.isfinite(vector)):
            raise ValueError("expected a finite vector with n entries")
        permuted = vector[self.permutation]
        top = (linalg.solve_triangular(self.L, permuted[:self.rank], lower=True, check_finite=False)
               if self.rank else np.empty(0, dtype=np.float64))
        q = permuted[self.rank:]-self.B @ top
        tail = self._GT @ (self.G @ q)
        if self.rank:
            top = linalg.solve_triangular(self.L.T, top-self.B.T @ tail,
                                          lower=False, check_finite=False)
        output = np.empty_like(vector)
        output[self.permutation[:self.rank]] = top
        output[self.permutation[self.rank:]] = tail
        return output

    @property
    def storage_bytes(self):
        """Exact retained array payload; transpose/views share counted buffers."""
        return sum(array.nbytes for array in (self.permutation, self.L, self.B,
                                              self.G.data, self.G.indices, self.G.indptr))


def _nearest_previous(points, row, count):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == row:
        return np.arange(row, dtype=np.int64)
    distances = cdist(points[row:row+1], points[:row], metric="sqeuclidean").ravel()
    selected = np.argpartition(distances, count-1)[:count]
    boundary = distances[selected].max()
    # Resolve equal-distance ties by earlier residual index, without sorting
    # all preceding points or retaining a pairwise distance matrix.
    inside = np.flatnonzero(distances < boundary)
    tied = np.flatnonzero(distances == boundary)
    return np.sort(np.r_[inside, tied[:count-inside.size]])


def initialize_schur_fsai(kernel, rank, regularization, rng, sparsity=8,
                          initialization="random"):
    """Build a fixed Schur-FSAI state using actual landmark kernel columns.

    ``sparsity`` includes the row diagonal. ``initialization`` is random
    RPCholesky or greedy pivoted Cholesky. Kernel statistics are not reset.
    The kernel must expose X, n, column(j), and entries(rows,columns), where
    entries returns the outer-indexed kernel submatrix and accounts for it.
    """
    lam = float(regularization)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("regularization must be finite and positive")
    if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= kernel.n:
        raise ValueError("rank must be an integer between zero and n")
    if not isinstance(sparsity, (int, np.integer)) or sparsity < 1:
        raise ValueError("sparsity must be a positive integer")
    if initialization not in ("random", "greedy"):
        raise ValueError("initialization must be random or greedy")
    features = np.asarray(kernel.X)
    if features.ndim != 2 or features.shape[0] != kernel.n or not np.all(np.isfinite(features)):
        raise ValueError("kernel.X must contain finite feature rows")
    selected = initialize_rpcholesky(kernel, rank, lam, rng, pivoting=initialization)
    mask = np.ones(kernel.n, dtype=bool)
    mask[selected.indices] = False
    permutation = np.r_[selected.indices, np.flatnonzero(mask)]
    remaining = permutation[rank:]
    landmark_kernel_rank = int(selected.rank)
    if rank:
        core = selected.columns[selected.indices, :].copy()
        core.flat[::rank+1] += lam
        L = linalg.cholesky((core+core.T)*.5, lower=True, check_finite=False)
        B = linalg.solve_triangular(L, selected.columns[remaining, :].T,
                                    lower=True, check_finite=False).T
    else:
        L = np.empty((0, 0), dtype=np.float64)
        B = np.empty((kernel.n, 0), dtype=np.float64)
    # No selected columns or Nyström basis survive while local rows are built.
    del selected
    points = features[remaining]
    m = remaining.size
    counts = np.minimum(np.arange(1, m+1, dtype=np.int64), sparsity)
    indptr = np.empty(m+1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    column_indices = np.empty(int(indptr[-1]), dtype=np.int64)
    coefficients = np.empty(int(indptr[-1]), dtype=np.float64)
    maximum_row_error = 0.0
    for row in range(m):
        neighbors = _nearest_previous(points, row, int(counts[row])-1)
        pattern = np.r_[neighbors, row]
        actual_indices = remaining[pattern]
        local = np.asarray(kernel.entries(actual_indices, actual_indices), dtype=np.float64)
        if local.shape != (len(pattern), len(pattern)) or not np.all(np.isfinite(local)):
            raise ValueError("kernel.entries returned an invalid local block")
        local = local.copy()
        local.flat[::len(pattern)+1] += lam
        block_B = B[pattern, :]
        local -= block_B @ block_B.T
        local = (local+local.T)*.5
        rhs = np.zeros(len(pattern), dtype=np.float64)
        rhs[-1] = 1.0
        try:
            solution = linalg.cho_solve(linalg.cho_factor(local, lower=True, check_finite=False),
                                        rhs, check_finite=False)
        except linalg.LinAlgError as error:
            raise NumericalStabilityError(f"nonpositive local Schur block at remaining row {row}") from error
        if solution[-1] <= 0 or not np.all(np.isfinite(solution)):
            raise NumericalStabilityError(f"invalid local inverse factor at remaining row {row}")
        values = solution/np.sqrt(solution[-1])
        first, last = indptr[row:row+2]
        column_indices[first:last] = pattern
        coefficients[first:last] = values
        maximum_row_error = max(maximum_row_error, abs(float(values @ local @ values)-1))
    G = sparse.csr_matrix((coefficients, column_indices, indptr), shape=(m, m))
    return SchurFSAIPreconditioner(permutation, L, B, G, lam, diagnostics=dict(
        initialization=initialization, landmark_kernel_rank=landmark_kernel_rank,
        sparsity=int(sparsity), local_kernel_blocks=int(m), nnz=int(G.nnz),
        maximum_row_D_energy_normalization_error=maximum_row_error))
