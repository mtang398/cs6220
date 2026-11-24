"""PCG with fixed preconditioners inside each block and explicit restart rules."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .numerics import initialize_rpcholesky, exchange_columns


METHODS = ("cg", "fixed", "residual_exchange", "matched_restart",
           "target_exchange", "uniform_exchange", "periodic_rebuild", "lbfgs")


class LimitedMemoryBFGS:
    """Positive definite two-loop inverse using at most m block secant pairs.

    Pairs are block increments (sum of PCG steps, sum of A times those steps).
    The pairs are frozen during a block. This is a specified L-BFGS baseline,
    not a reproduction of Morales--Nocedal's experiments on sequences of systems.
    """
    def __init__(self, max_pairs, initial_scale):
        self.max_pairs = int(max_pairs)
        self.initial_scale = float(initial_scale)
        self.pairs = []

    def update(self, s, z):
        curvature = float(s @ z)
        if self.max_pairs == 0 or curvature <= 1e-12*np.linalg.norm(s)*np.linalg.norm(z):
            return False
        if not np.isfinite(curvature) or not np.all(np.isfinite(z)):
            return False
        self.pairs.append((s.copy(), z.copy(), 1.0/curvature))
        self.pairs = self.pairs[-self.max_pairs:]
        return True

    def apply(self, v):
        q = v.copy()
        alphas = []
        for s, z, rho in reversed(self.pairs):
            alpha = rho*float(s @ q)
            alphas.append(alpha)
            q -= alpha*z
        gamma = self.initial_scale
        if self.pairs:
            s, z, _ = self.pairs[-1]
            gamma = float(s @ z)/float(z @ z)
        q *= gamma
        for (s, z, rho), alpha in zip(self.pairs, reversed(alphas)):
            q += s*(alpha-rho*float(z @ q))
        return q

    @property
    def storage_bytes(self):
        return sum(s.nbytes+z.nbytes+8 for s, z, _ in self.pairs)


@dataclass
class SolveResult:
    solution: np.ndarray
    metrics: dict
    history: list
    events: list


def solve(kernel, y, regularization, *, method="fixed", rank=16, seed=0,
          checkpoint=20, candidate_count=4, rtol=1e-8, maxiter=2000,
          restart_iterations=None, exchange_callback=None):
    """Solve the original regularized system, counting all verification matvecs.

    Rejected exchanges preserve the existing PCG recurrence. Accepted changes
    restart with the explicitly recomputed residual. All methods verify the
    true residual on the same checkpoint schedule and before returning success.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if regularization <= 0 or checkpoint < 1 or maxiter < 1 or rank < 1:
        raise ValueError("Positive regularization, checkpoint, maxiter and rank required")
    if method == "matched_restart" and restart_iterations is None:
        raise ValueError("matched_restart requires the accepted-update iteration list")
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (kernel.n,) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid right-hand side")
    kernel.reset_stats()
    started = perf_counter()
    x = np.zeros_like(y)
    norm_y = float(np.linalg.norm(y))
    if norm_y == 0:
        return SolveResult(x, dict(method=method, converged=True, iterations=0,
                                   true_relative_residual=0.0, breakdown=None,
                                   total_seconds=perf_counter()-started,
                                   setup_seconds=0.0, update_seconds=0.0,
                                   preconditioner_seconds=0.0, restarts=0,
                                   verification_matvecs=0, initial_indices=[], final_indices=[],
                                   effective_rank=0, max_preconditioner_state_bytes=0,
                                   accepted_exchanges=0, accepted_iterations=[],
                                   numerical_fallbacks=0, numerical_rejections=0,
                                   diagnostic_run=exchange_callback is not None,
                                   **kernel.stats), [], [])
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 6220, 1]))
    update_rng = np.random.default_rng(np.random.SeedSequence([seed, 6220, 2]))
    init_start = perf_counter()
    if method == "cg":
        preconditioner = None
    elif method == "lbfgs":
        preconditioner = LimitedMemoryBFGS(rank//2, 1/(1+regularization))
    else:
        preconditioner = initialize_rpcholesky(kernel, rank, regularization, init_rng)
    setup_seconds = perf_counter()-init_start
    initial_indices = (preconditioner.indices.tolist()
                       if hasattr(preconditioner, "indices") else [])
    preconditioner_seconds = 0.0
    update_seconds = 0.0
    max_preconditioner_state_bytes = getattr(preconditioner, "storage_bytes", 0)

    def apply(v):
        nonlocal preconditioner_seconds
        start = perf_counter()
        z = v.copy() if preconditioner is None else preconditioner.apply(v)
        preconditioner_seconds += perf_counter()-start
        return z

    def matvec(v):
        return kernel.matvec(v)+regularization*v

    r = y.copy()
    z = apply(r)
    rho = float(r @ z)
    p = z.copy()
    history = [{"iteration": 0, "recursive_relative_residual": 1.0,
                "true_relative_residual": 1.0}]
    events = []
    restarts = 0
    verification_matvecs = 0
    matched = set(restart_iterations or [])
    block_s = np.zeros_like(y) if method == "lbfgs" else None
    block_z = np.zeros_like(y) if method == "lbfgs" else None
    converged = False
    breakdown = None
    true_r = r.copy()
    true_residual_current = True
    iteration = 0
    for iteration in range(1, maxiter+1):
        ap = matvec(p)
        curvature = float(p @ ap)
        if not np.isfinite(curvature) or curvature <= 0 or not np.isfinite(rho) or rho <= 0:
            breakdown = "nonpositive_or_nonfinite_pcg_inner_product"
            break
        step = rho/curvature
        x += step*p
        true_residual_current = False
        r -= step*ap
        if method == "lbfgs":
            block_s += step*p
            block_z += step*ap
        recursive_relative = float(np.linalg.norm(r)/norm_y)
        row = {"iteration": iteration, "recursive_relative_residual": recursive_relative}
        checkpoint_now = iteration % checkpoint == 0
        suspected_convergence = recursive_relative <= rtol
        restart = False
        if checkpoint_now or suspected_convergence or iteration == maxiter:
            true_r = y-matvec(x)
            true_residual_current = True
            verification_matvecs += 1
            relative = float(np.linalg.norm(true_r)/norm_y)
            row["true_relative_residual"] = relative
            if relative <= rtol:
                converged = True
                history.append(row)
                break
            if suspected_convergence:
                restart = True  # Explicit residual replacement after a false recursive stop.
                events.append({"iteration": iteration, "kind": "residual_replacement"})
        if checkpoint_now and iteration < maxiter:
            update_start = perf_counter()
            if method in ("residual_exchange", "uniform_exchange", "target_exchange"):
                vector = y if method == "target_exchange" else true_r
                old = preconditioner
                preconditioner, event = exchange_columns(
                    kernel, old, vector, update_rng, candidate_count=candidate_count,
                    sampling="uniform" if method == "uniform_exchange" else "residual")
                event = dict(event, iteration=iteration, kind="exchange")
                events.append(event)
                restart = restart or bool(event["accepted"])
                update_seconds += perf_counter()-update_start
                if event["accepted"] and exchange_callback is not None:
                    # Diagnostics use a separate run; callback work is not a benchmark result.
                    exchange_callback(x.copy(), old, preconditioner, iteration)
                del old
            elif method == "periodic_rebuild":
                preconditioner = initialize_rpcholesky(
                    kernel, rank, regularization, update_rng)
                restart = True
                events.append({"iteration": iteration, "kind": "rebuild"})
                update_seconds += perf_counter()-update_start
            elif method == "matched_restart":
                restart = restart or iteration in matched
            elif method == "lbfgs":
                changed = preconditioner.update(block_s, block_z)
                events.append({"iteration": iteration, "kind": "lbfgs", "accepted": changed})
                block_s.fill(0)
                block_z.fill(0)
                restart = restart or changed
                update_seconds += perf_counter()-update_start
            max_preconditioner_state_bytes = max(
                max_preconditioner_state_bytes, getattr(preconditioner, "storage_bytes", 0))
        if restart:
            r = true_r.copy()
            z = apply(r)
            rho = float(r @ z)
            p = z.copy()
            restarts += 1
            row["restarted"] = True
        else:
            z = apply(r)
            new_rho = float(r @ z)
            p = z+(new_rho/rho)*p
            rho = new_rho
        history.append(row)
    if not true_residual_current:
        true_r = y-matvec(x)
        verification_matvecs += 1
    true_relative = float(np.linalg.norm(true_r)/norm_y)
    elapsed = perf_counter()-started
    metrics = dict(method=method, converged=bool(converged), iterations=iteration,
                   true_relative_residual=true_relative, breakdown=breakdown,
                   total_seconds=elapsed, setup_seconds=setup_seconds,
                   update_seconds=update_seconds, preconditioner_seconds=preconditioner_seconds,
                   restarts=restarts, verification_matvecs=verification_matvecs,
                   initial_indices=initial_indices,
                   final_indices=(preconditioner.indices.tolist()
                                  if hasattr(preconditioner, "indices") else []),
                   effective_rank=getattr(preconditioner, "rank", None),
                   max_preconditioner_state_bytes=int(max_preconditioner_state_bytes),
                   accepted_exchanges=sum(e.get("exchanges", 0) for e in events),
                   numerical_fallbacks=sum(e.get("numerical_fallbacks", 0) for e in events),
                   numerical_rejections=sum(e.get("numerical_rejections", 0) for e in events),
                   accepted_iterations=[e["iteration"] for e in events
                                        if e.get("kind") == "exchange" and e.get("accepted")],
                   diagnostic_run=exchange_callback is not None,
                   **kernel.stats)
    return SolveResult(x, metrics, history, events)


def fixed_steps(kernel, rhs, regularization, x, preconditioner, steps):
    """Fresh PCG branch from x for mechanism diagnostics, outside timing runs."""
    x = x.copy()
    r = rhs-kernel.matvec(x)-regularization*x
    z = preconditioner.apply(r)
    p = z.copy()
    rho = float(r @ z)
    outputs = []
    for i in range(1, steps+1):
        ap = kernel.matvec(p)+regularization*p
        denom = float(p @ ap)
        if denom <= 0 or rho <= 0:
            break
        a = rho/denom
        x += a*p
        r -= a*ap
        outputs.append(x.copy())
        if np.linalg.norm(r) < 1e-14*np.linalg.norm(rhs):
            break
        z = preconditioner.apply(r)
        rho_new = float(r @ z)
        p = z+(rho_new/rho)*p
        rho = rho_new
    return outputs
