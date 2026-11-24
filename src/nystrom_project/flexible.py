"""Flexible conjugate directions with explicit bounded A-orthogonalization.

For z_k = M_k^{-1} r_k and retained A-conjugate directions p_j,

    p_k = z_k - sum_j p_j * (p_j.T A z_k)/(p_j.T A p_j),
    alpha_k = (r_k.T p_k)/(p_k.T A p_k),
    x_{k+1} = x_k + alpha_k p_k,  r_{k+1} = r_k - alpha_k A p_k.

The window retains min(k,m) previous directions.  m=0 gives preconditioned
steepest descent; m=1 is the usual one-direction flexible CG recurrence;
retaining every previous direction gives full flexible conjugate directions.
This is an existing solver mechanism, not a column-selection contribution.

Primary sources: Knyazev--Lashuk (2007), equations (1), (2), and (14),
https://arxiv.org/html/math/0605767 ; Notay (2000),
https://doi.org/10.1137/S1064827599362314 .  PETSc's FCG source also explicitly
stores directions and their A images: https://petsc.org/release/src/ksp/ksp/impls/fcg/fcg.c.html .

Retaining m directions costs 2*m*n float64 entries plus m curvature scalars,
in addition to the preconditioner and ordinary solver work vectors. Projection
costs O(m*n) per vector (two modified Gram--Schmidt passes by default). Every
iteration uses one A matvec; verification and update-callback matvecs are
counted separately. Fixed and changing preconditioners must use the same
history window/reorthogonalization for the principal controlled comparison.

Three-branch snapshots for conventional PCG should be taken immediately after
a completed step and before the next preconditioner application/beta update.
Continuation needs x, recursive r, the just-used p, and rho from that step,
plus the immutable old preconditioner and iteration/checkpoint counters.
At this boundary rho is the OLD r.T M^{-1}r, not the new one.  Restart branches
need x and a shared true residual; next p and rho are recomputed under old/new
M. Alternatively snapshot after beta formation: x,r,next_p,next_rho suffice.
An FCG continuation instead needs x,r and the retained direction/A-image
history; a snapshot below stores those arrays independently when requested.
Snapshots are diagnostic allocations, not hidden production storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .solvers import SolveResult


def _array(value: np.ndarray, n: int, *, matrix: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim not in ((1, 2) if matrix else (1,)) or result.shape[0] != n
            or not np.all(np.isfinite(result))):
        raise ValueError("expected finite float64-compatible vector(s) with n rows")
    return result


@dataclass(frozen=True)
class HistorySnapshot:
    n: int
    max_history: int
    reorthogonalize: bool
    pairs: tuple[tuple[np.ndarray, np.ndarray, float], ...]

    @property
    def storage_bytes(self) -> int:
        return sum(p.nbytes+ap.nbytes+8 for p, ap, _ in self.pairs)


class AConjugateHistory:
    """A-normalized direction/image pairs in a sliding window.

    Stored arrays are read-only. ``project`` does not change the stored
    directions and accepts a vector or a matrix of candidate directions. Its
    second argument must be the corresponding A image. No operator call is
    made by ``project``. ``form_direction`` adds exactly one operator call.
    """

    def __init__(self, n: int, max_history: int, reorthogonalize: bool = True):
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(max_history, (int, np.integer)) or max_history < 0:
            raise ValueError("max_history must be a nonnegative integer")
        self.n = int(n)
        self.max_history = int(max_history)
        self.reorthogonalize = bool(reorthogonalize)
        self._pairs: deque[tuple[np.ndarray, np.ndarray, float]] = deque()
        self.projection_seconds = 0.0
        self.projected_vectors = 0
        self.projection_inner_products = 0

    @property
    def size(self) -> int:
        return len(self._pairs)

    @property
    def storage_bytes(self) -> int:
        """Retained numerical payload; Python container overhead is excluded."""
        return sum(p.nbytes+ap.nbytes+8 for p, ap, _ in self._pairs)

    def clear(self) -> None:
        self._pairs.clear()

    def project(self, z: np.ndarray, az: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = _array(z, self.n, matrix=True)
        az = _array(az, self.n, matrix=True)
        if z.shape != az.shape:
            raise ValueError("z and az must have the same shape")
        started = perf_counter()
        p, ap = z.copy(), az.copy()
        passes = 2 if self.reorthogonalize else 1
        for _ in range(passes):
            for previous, a_previous, curvature in self._pairs:
                coefficient = (previous @ ap)/curvature
                if p.ndim == 1:
                    p -= coefficient*previous
                    ap -= coefficient*a_previous
                else:
                    p -= previous[:, None]*coefficient[None, :]
                    ap -= a_previous[:, None]*coefficient[None, :]
        count = 1 if p.ndim == 1 else p.shape[1]
        self.projected_vectors += count
        self.projection_inner_products += passes*self.size*count
        self.projection_seconds += perf_counter()-started
        return p, ap

    def form_direction(self, z: np.ndarray, matvec: Callable) -> tuple[np.ndarray, np.ndarray]:
        z = _array(z, self.n)
        return self.project(z, matvec(z))

    def append(self, p: np.ndarray, ap: np.ndarray) -> None:
        p, ap = _array(p, self.n), _array(ap, self.n)
        curvature = float(p @ ap)
        if not np.isfinite(curvature) or curvature <= 0:
            raise np.linalg.LinAlgError("direction has nonpositive or nonfinite A-curvature")
        if self.max_history == 0:
            return
        scale = np.sqrt(curvature)
        normalized, a_normalized = p/scale, ap/scale
        stored_curvature = float(normalized @ a_normalized)
        if not np.isfinite(stored_curvature) or stored_curvature <= 0:
            raise np.linalg.LinAlgError("A-normalization failed")
        normalized.setflags(write=False)
        a_normalized.setflags(write=False)
        if self.size == self.max_history:
            self._pairs.popleft()
        self._pairs.append((normalized, a_normalized, stored_curvature))

    def snapshot(self) -> HistorySnapshot:
        """Copy only on explicit diagnostic request; no matvec is needed."""
        pairs = []
        for p, ap, curvature in self._pairs:
            p, ap = p.copy(), ap.copy()
            p.setflags(write=False)
            ap.setflags(write=False)
            pairs.append((p, ap, curvature))
        return HistorySnapshot(self.n, self.max_history, self.reorthogonalize, tuple(pairs))

    @classmethod
    def from_snapshot(cls, snapshot: HistorySnapshot) -> "AConjugateHistory":
        result = cls(snapshot.n, snapshot.max_history, snapshot.reorthogonalize)
        if len(snapshot.pairs) > snapshot.max_history:
            raise ValueError("snapshot exceeds its history budget")
        for p, ap, curvature in snapshot.pairs:
            p, ap = _array(p, snapshot.n).copy(), _array(ap, snapshot.n).copy()
            if not np.isfinite(curvature) or curvature <= 0:
                raise ValueError("invalid snapshot curvature")
            p.setflags(write=False)
            ap.setflags(write=False)
            result._pairs.append((p, ap, float(curvature)))
        return result


def flexible_solve(kernel, y, regularization, *, preconditioner=None, preconditioner_factory=None,
                   history_size=8, reorthogonalize=True, checkpoint=20,
                   rtol=1e-8, maxiter=2000, x0=None, update_callback=None,
                   step_callback=None, direction_update=None) -> SolveResult:
    """Solve with fixed or changing SPD inverse applications and no update restart.

    ``preconditioner`` has .apply(v), or is callable; None is the identity.
    At a checked, nonconverged checkpoint, call
    ``update_callback(x, true_r, old, iteration, history) -> (new, event)``.
    The callback may use history.project(), but must not append/clear history.
    A changed object must have event['accepted']=True. It becomes active on the
    next step without discarding directions or replacing the recursive residual.

    ``step_callback(x, recursive_r, iteration, history)`` is a diagnostic hook
    after each completed step, before any update. x and r are copies. Copies of
    retained directions are made only if the callback requests snapshot().

    Use ``preconditioner_factory()`` to construct a fresh initial state inside
    the timed solve, including its kernel calls. The factory should capture
    only kernel/configuration, not an existing state; the caller then retains
    no hidden initial-preconditioner copy during adaptive solves. It cannot
    be combined with a supplied preconditioner. A zero RHS skips construction.
    Alternatively, initialization of a supplied preconditioner is outside this
    routine's timer/counters and must be added by the caller to an end-to-end
    comparison. Kernel counters are deltas; existing setup counts are preserved.
    Projection timing includes projections requested by update callbacks and
    overlaps update_seconds, so those timing components must not be added.
    """
    lam = float(regularization)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("regularization must be finite and positive")
    for name, value in (("checkpoint", checkpoint), ("maxiter", maxiter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must lie strictly between zero and one")
    if preconditioner_factory is not None and preconditioner is not None:
        raise ValueError("supply a preconditioner or a factory, not both")
    y = _array(y, kernel.n)
    x = np.zeros_like(y) if x0 is None else _array(x0, kernel.n).copy()
    directions = AConjugateHistory(kernel.n, history_size, reorthogonalize)
    initial_stats = dict(kernel.stats)
    started = perf_counter()
    norm_y = float(np.linalg.norm(y))
    setup_seconds = 0.0
    if preconditioner_factory is not None and norm_y:
        setup_started = perf_counter()
        preconditioner = preconditioner_factory()
        setup_seconds = perf_counter()-setup_started
    initial_indices = getattr(preconditioner, "indices", np.empty(0, dtype=int)).tolist()
    preconditioner_seconds = update_seconds = diagnostic_seconds = 0.0
    preconditioner_applications = direction_matvecs = verification_matvecs = 0
    initial_residual_matvecs = update_matvecs = restarts = 0
    max_preconditioner_bytes = int(getattr(preconditioner, "storage_bytes", 0))
    max_history_bytes = max_retained_directions = 0
    max_combined_bytes = max_preconditioner_bytes
    rows, events = [], []
    iteration = 0
    breakdown = None
    converged = False

    def matvec(v):
        return kernel.matvec(v)+lam*v

    def apply(v):
        nonlocal preconditioner_seconds, preconditioner_applications
        start = perf_counter()
        if preconditioner is None:
            output = v.copy()
        elif hasattr(preconditioner, "apply"):
            output = preconditioner.apply(v)
        else:
            output = preconditioner(v)
        output = _array(output, kernel.n)
        preconditioner_seconds += perf_counter()-start
        preconditioner_applications += 1
        return output

    if norm_y == 0:
        x.fill(0)
        r = true_r = np.zeros_like(y)
        converged = True
    elif x0 is None or not np.any(x):
        r = true_r = y.copy()
    else:
        r = y-matvec(x)
        true_r = r.copy()
        initial_residual_matvecs += 1
        converged = np.linalg.norm(r) <= rtol*norm_y
    true_current = True
    if norm_y:
        initial_relative = float(np.linalg.norm(r)/norm_y)
        rows.append(dict(iteration=0, recursive_relative_residual=initial_relative,
                         true_relative_residual=initial_relative, history_size=0))

    for next_iteration in range(1, maxiter+1):
        if converged:
            break
        z = apply(r)
        if not np.isfinite(r @ z) or r @ z <= 0:
            breakdown = "nonpositive_or_nonfinite_preconditioned_residual"
            break
        used = directions.size
        p, ap = directions.form_direction(z, matvec)
        direction_matvecs += 1
        curvature, numerator = float(p @ ap), float(r @ p)
        if (not np.isfinite(curvature) or curvature <= 0
                or not np.isfinite(numerator) or numerator <= 0):
            breakdown = "nonpositive_or_nonfinite_flexible_direction"
            break
        alpha = numerator/curvature
        x += alpha*p
        r -= alpha*ap
        iteration = next_iteration
        true_current = False
        directions.append(p, ap)
        max_history_bytes = max(max_history_bytes, directions.storage_bytes)
        max_retained_directions = max(max_retained_directions, directions.size)
        max_combined_bytes = max(max_combined_bytes,
                                 int(getattr(preconditioner, "storage_bytes", 0))+directions.storage_bytes)
        recursive_relative = float(np.linalg.norm(r)/norm_y)
        row = dict(iteration=iteration, recursive_relative_residual=recursive_relative,
                   directions_used=used, history_size=directions.size,
                   predicted_energy_decrease=float(numerator*numerator/curvature))
        checkpoint_now = iteration % checkpoint == 0
        suspected = recursive_relative <= rtol
        if checkpoint_now or suspected or iteration == maxiter:
            true_r = y-matvec(x)
            verification_matvecs += 1
            true_current = True
            row["true_relative_residual"] = float(np.linalg.norm(true_r)/norm_y)
            converged = row["true_relative_residual"] <= rtol
        if step_callback is not None:
            start = perf_counter()
            step_callback(x.copy(), r.copy(), iteration, directions)
            diagnostic_seconds += perf_counter()-start
        if converged:
            rows.append(row)
            break
        if direction_update is not None:
            start = perf_counter()
            before_matvecs = kernel.stats['matvecs']
            change = direction_update(p, ap, r, preconditioner, iteration)
            if change is not None:
                preconditioner, event = change
                event = dict(event, iteration=iteration)
                if event.get('restart', False):
                    if not true_current:
                        true_r = y-matvec(x)
                        true_current = True
                        verification_matvecs += 1
                    r = true_r.copy()
                    directions.clear()
                    restarts += 1
                    row['restarted'] = True
                event['history_preserved'] = bool(directions.size)
                events.append(event)
                max_preconditioner_bytes = max(max_preconditioner_bytes, preconditioner.storage_bytes)
                max_combined_bytes = max(max_combined_bytes, preconditioner.storage_bytes+directions.storage_bytes)
            update_seconds += perf_counter()-start
            update_matvecs += kernel.stats['matvecs']-before_matvecs
        if suspected:
            # A failed recursive stopping check is a numerical residual
            # replacement, not a preconditioner-update restart.
            r = true_r.copy()
            directions.clear()
            restarts += 1
            row["restarted"] = True
            events.append(dict(kind="residual_replacement", iteration=iteration))
        if checkpoint_now and iteration < maxiter and update_callback is not None:
            start = perf_counter()
            before_matvecs = kernel.stats["matvecs"]
            old = preconditioner
            new, event = update_callback(x.copy(), true_r.copy(), old, iteration, directions)
            event = dict(event)
            accepted = bool(event.get("accepted", new is not old))
            if not accepted and new is not old:
                raise ValueError("a rejected update must return the old preconditioner object")
            event.update(accepted=accepted, iteration=iteration,
                         kind=event.get("kind", "preconditioner_update"),
                         history_preserved=bool(directions.size))
            events.append(event)
            preconditioner = new
            del old
            update_seconds += perf_counter()-start
            update_matvecs += kernel.stats["matvecs"]-before_matvecs
            max_preconditioner_bytes = max(max_preconditioner_bytes,
                                           int(getattr(preconditioner, "storage_bytes", 0)))
            max_combined_bytes = max(max_combined_bytes,
                                     int(getattr(preconditioner, "storage_bytes", 0))+directions.storage_bytes)
        rows.append(row)

    if not true_current:
        true_r = y-matvec(x)
        verification_matvecs += 1
    true_relative = float(np.linalg.norm(true_r)/norm_y) if norm_y else 0.0
    converged = bool(true_relative <= rtol)
    stats = {key: value-initial_stats.get(key, 0) for key, value in kernel.stats.items()}
    metrics: dict[str, Any] = dict(
        method="flexible", converged=converged, iterations=iteration,
        true_relative_residual=true_relative, breakdown=breakdown,
        total_seconds=perf_counter()-started, setup_seconds=setup_seconds,
        initialization_included=preconditioner_factory is not None,
        preconditioner_seconds=preconditioner_seconds,
        preconditioner_applications=preconditioner_applications,
        update_seconds=update_seconds, projection_seconds=directions.projection_seconds,
        diagnostic_seconds=diagnostic_seconds, diagnostic_run=step_callback is not None,
        direction_matvecs=direction_matvecs, verification_matvecs=verification_matvecs,
        initial_residual_matvecs=initial_residual_matvecs, update_matvecs=update_matvecs,
        restarts=restarts, history_budget=int(history_size),
        reorthogonalize=bool(reorthogonalize), max_retained_directions=max_retained_directions,
        max_history_vectors=2*max_retained_directions, max_history_bytes=max_history_bytes,
        max_preconditioner_state_bytes=max_preconditioner_bytes,
        max_combined_retained_bytes=max_combined_bytes,
        projection_inner_products=directions.projection_inner_products,
        initial_indices=initial_indices,
        final_indices=getattr(preconditioner, "indices", np.empty(0, dtype=int)).tolist(),
        effective_rank=getattr(preconditioner, "rank", None),
        preconditioner_diagnostics=dict(getattr(preconditioner, "diagnostics", {})),
        accepted_exchanges=sum(event.get("exchanges", int(event.get("accepted", False)))
                               for event in events),
        accepted_iterations=[event["iteration"] for event in events if event.get("accepted")],
        numerical_fallbacks=sum(event.get("numerical_fallbacks", 0) for event in events),
        numerical_rejections=sum(event.get("numerical_rejections", 0) for event in events),
        **stats)
    return SolveResult(x, metrics, rows, events)
