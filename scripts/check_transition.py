#!/usr/bin/env python3
"""Bounded offline diagnosis of a prescribed Nyström change at iteration 20.

Shared-checkpoint continuations retain its original eight valid pairs; longer
windows retain more subsequent directions. Genuine longer-history production
runs also start from zero. Dense A is
used only to measure accuracy/conjugacy, never as the solver operator.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
from nystrom_project.flexible import AConjugateHistory, flexible_solve
from nystrom_project.numerics import GaussianKernel, NystromPreconditioner, initialize_rpcholesky


def main(root, output):
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic: {output}")
    resource.setrlimit(resource.RLIMIT_CPU, (55, 56))
    started, cpu_started = perf_counter(), process_time()
    source = root/"results/revision-curvature-001/runs/synthetic-n640-smooth-s0-lambda0.001-curvature4once-r0.json"
    record = json.loads(source.read_text())
    prepared = root/"data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz"
    with np.load(prepared) as data:
        X, y = data["X_train"], data["y_train"]
        bandwidth = float(data["bandwidth"])
        reference = data["alpha_reference"]
        config = json.loads(str(data["config"]))
    assert config["dataset_sha256"] == record["case"]["dataset_sha256"]
    lam = float(record["case"]["regularization"])
    initial = np.asarray(record["metrics"]["initial_indices"], dtype=int)
    final = np.asarray(record["metrics"]["final_indices"], dtype=int)
    n, norm_y, tolerance = len(y), np.linalg.norm(y), 1e-8
    result = dict(
        status="running", scope="Offline accuracy diagnostic on one development case; not a timing benchmark",
        protocol=dict(checkpoint=20, switch_iteration=20, history_size=8, rtol=tolerance,
                      maxiter=2000, blas_threads=1, cpu_limit_seconds=55,
                      prescribed_final_indices=final.tolist(), initial_indices=initial.tolist(),
                      shared_checkpoint=True, archived_pairs_used_only_for_diagnostics=True),
        case=record["case"],
        source_run=dict(path=str(source.relative_to(root)), sha256=sha256(source.read_bytes()).hexdigest(),
                        adaptive_iterations=record["metrics"]["iterations"]),
        prepared_sha256=sha256(prepared.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                       for path in [root/"src/nystrom_project/flexible.py", root/"src/nystrom_project/numerics.py",
                                    Path(__file__).resolve()]},
        branches={})

    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,
                      elapsed_cpu_seconds=process_time()-cpu_started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    def make_kernel():
        return GaussianKernel(X, bandwidth, block_size=128)

    def make_old(kernel):
        state = initialize_rpcholesky(kernel, len(initial), lam,
                                     np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
        np.testing.assert_array_equal(state.indices, initial)
        return state

    def make_new(kernel):
        return NystromPreconditioner(final, np.column_stack([kernel.column(int(j)) for j in final]), lam)

    with threadpool_limits(limits=1):
        dense_kernel = make_kernel()
        A = dense_kernel.dense()+lam*np.eye(n)
        reference_error = np.linalg.norm(y-A @ reference)/norm_y
        result["reference_relative_residual"] = float(reference_error)
        checkpoint = {}

        class Trace:
            def __init__(self, prefix=()):
                self.pairs = list(prefix)
                self.rows = []
                self.projector_checks = []

            def record(self, x, r, iteration, history, before_pairs=None, z=None, p=None):
                pair = history._pairs[-1]
                newest, anewest, curvature = pair
                true_ap = A @ newest
                true_r = y-A @ x
                error = reference-x
                energy = float(error @ (A @ error))
                energy_scale = np.sqrt(max(energy, np.finfo(float).tiny))
                if before_pairs is None:
                    # Production hook is after append; history size is already
                    # bounded. The oldest just-projected pair is in our archive.
                    before_pairs = self.pairs[-min(iteration-1, history.max_history):] if iteration > 1 else []
                projected_count = len(before_pairs)
                previous = self.pairs
                retained_ids = {id(pair[0]) for pair in before_pairs}
                # Archives use the original immutable arrays, hence identities
                # identify exactly the directions used before append.
                forgotten = [pair for pair in previous if id(pair[0]) not in retained_ids]

                def maximum_dot(pairs, vector):
                    if not pairs:
                        return 0.0
                    P = np.column_stack([q for q, _, _ in pairs])
                    return float(np.max(np.abs(P.T @ vector)))

                retained_after = list(history._pairs)
                row = dict(
                    iteration=int(iteration), energy_error=energy,
                    true_relative_residual=float(np.linalg.norm(true_r)/norm_y),
                    recursive_relative_residual=float(np.linalg.norm(r)/norm_y),
                    residual_gap_relative_rhs=float(np.linalg.norm(r-true_r)/norm_y),
                    projected_directions=projected_count,
                    retained_history_gram_error=float(maximum_dot(before_pairs, true_ap)/np.sqrt(curvature)),
                    forgotten_history_conjugacy_error=float(maximum_dot(forgotten, true_ap)/np.sqrt(curvature)),
                    cached_ap_relative_error=float(np.linalg.norm(anewest-true_ap)/np.linalg.norm(true_ap)),
                    retained_recursive_residual_orthogonality=float(maximum_dot(retained_after, r)/energy_scale),
                    retained_true_residual_orthogonality=float(maximum_dot(retained_after, true_r)/energy_scale),
                    first20_true_residual_orthogonality=float(maximum_dot(previous[:20], true_r)/energy_scale),
                    history_bytes=int(history.storage_bytes))
                self.rows.append(row)
                self.pairs.append(pair)
                if z is not None and iteration in (21, 22, 25, 30, 40):
                    az = A @ z
                    if before_pairs:
                        P = np.column_stack([q for q, _, _ in before_pairs])
                        AP = A @ P
                        gram = (P.T @ AP+AP.T @ P)*.5
                        coefficients = linalg.solve(gram, P.T @ az, assume_a="pos")
                        independent_p = z-P @ coefficients
                    else:
                        independent_p = z
                    difference = independent_p-p
                    denominator = float(p @ (A @ p))
                    relative = np.sqrt(max(float(difference @ (A @ difference)), 0)/denominator)
                    self.projector_checks.append(dict(iteration=iteration,
                        relative_A_norm_difference_from_dense_gram_projection=float(relative)))

            def summary(self):
                post = [row for row in self.rows if row["iteration"] > 20]
                early = [row for row in post if row["iteration"] <= 60]
                fields = ("retained_history_gram_error", "forgotten_history_conjugacy_error",
                          "cached_ap_relative_error", "residual_gap_relative_rhs",
                          "retained_recursive_residual_orthogonality", "retained_true_residual_orthogonality")
                return dict(maximum_post20={key: max(row[key] for row in post) for key in fields},
                            maximum_iterations21_to60={key: max(row[key] for row in early) for key in fields},
                            selected_steps=[row for row in self.rows if row["iteration"] in
                                            (20, 21, 22, 25, 28, 30, 32, 40, 50, 60, 80, 100, 140, 200, 300)
                                            or row is self.rows[-1]],
                            dense_projector_checks=self.projector_checks)

        production_trace = Trace()
        kernel = make_kernel()

        def hook(x, r, iteration, history):
            production_trace.record(x, r, iteration, history)
            if iteration == 20:
                checkpoint.update(x=x.copy(), r=r.copy(), true_r=y-A @ x,
                                  history=history.snapshot(), pairs=tuple(production_trace.pairs))

        def update(x, true_r, old, iteration, history):
            if iteration != 20:
                return old, dict(accepted=False, exchanges=0)
            checkpoint["production_true_r"] = true_r.copy()
            return make_new(kernel), dict(accepted=True, exchanges=4, kind="prescribed_update")

        production = flexible_solve(kernel, y, lam, preconditioner_factory=lambda: make_old(kernel),
                                    history_size=8, checkpoint=20, rtol=tolerance, maxiter=2000,
                                    update_callback=update, step_callback=hook)
        result["branches"]["production_preserve8"] = dict(
            iterations=production.metrics["iterations"], converged=production.metrics["converged"],
            true_relative_residual=production.metrics["true_relative_residual"],
            diagnostic=production_trace.summary())
        P20 = np.column_stack([p for p, _, _ in checkpoint["pairs"]])
        result["shared_checkpoint"] = dict(
            iteration=20, recursive_relative_residual=float(np.linalg.norm(checkpoint["r"])/norm_y),
            true_relative_residual=float(np.linalg.norm(checkpoint["true_r"])/norm_y),
            recursive_true_residual_gap=float(np.linalg.norm(checkpoint["r"]-checkpoint["true_r"])/norm_y),
            all20_actual_A_gram_error=float(np.max(np.abs(P20.T @ (A @ P20)-np.eye(20)))),
            production_vs_dense_true_residual_gap=float(np.linalg.norm(checkpoint["production_true_r"]-checkpoint["true_r"])/norm_y))
        save()

        def continue_branch(label, budget, *, restart=False, use_old=False, replace_residual=False,
                            invalid_restore_all20=False):
            kernel = make_kernel()
            state = make_old(kernel) if use_old else make_new(kernel)
            x = checkpoint["x"].copy()
            # Use the production operator's verified residual for an exact
            # comparison with flexible_solve(x0=x20), not dense BLAS ordering.
            r = checkpoint["production_true_r"].copy() if replace_residual else checkpoint["r"].copy()
            prefix = checkpoint["pairs"]
            if restart:
                history = AConjugateHistory(n, budget)
                trace = Trace(prefix)
            else:
                count = min(budget, len(prefix)) if invalid_restore_all20 else min(budget, 8)
                pairs = prefix[-count:]
                # Do not clone immutable arrays: diagnostic identities track
                # whether a prior pair was among the actually projected ones.
                history = AConjugateHistory(n, budget)
                history._pairs.extend(pairs)
                trace = Trace(prefix)
            final_iteration = 20
            breakdown = None
            for iteration in range(21, 2001):
                z = state.apply(r)
                before_pairs = list(history._pairs)
                p, ap = history.form_direction(z, lambda v: kernel.matvec(v)+lam*v)
                numerator, denominator = float(r @ p), float(p @ ap)
                if numerator <= 0 or denominator <= 0:
                    breakdown = "nonpositive_flexible_direction"
                    break
                alpha = numerator/denominator
                x += alpha*p
                r -= alpha*ap
                history.append(p, ap)
                trace.record(x, r, iteration, history, before_pairs, z, p)
                final_iteration = iteration
                suspected = np.linalg.norm(r)/norm_y <= tolerance
                if iteration % 20 == 0 or suspected or iteration == 2000:
                    true_r = y-(kernel.matvec(x)+lam*x)
                    if np.linalg.norm(true_r)/norm_y <= tolerance:
                        break
                    if suspected:
                        r = true_r.copy()
                        history.clear()
            true_relative = float(np.linalg.norm(y-A @ x)/norm_y)
            row = dict(iterations=final_iteration, iterations_after20=final_iteration-20,
                       converged=bool(true_relative <= tolerance), true_relative_residual=true_relative,
                       history_budget=budget, residual_replaced_at20=replace_residual,
                       history_cleared_at20=restart, breakdown=breakdown,
                       diagnostic=trace.summary())
            result["branches"][label] = row
            save()
            return x, trace

        x_replayed, trace_replayed = continue_branch("shared_preserve8", 8)
        result["production_replay"] = dict(
            identical_solution=bool(np.array_equal(x_replayed, production.solution)),
            relative_solution_difference=float(np.linalg.norm(x_replayed-production.solution)/np.linalg.norm(production.solution)),
            identical_iteration_count=bool(result["branches"]["shared_preserve8"]["iterations"] == production.metrics["iterations"]))
        assert result["production_replay"]["identical_solution"]
        x_restart, _ = continue_branch("shared_restart8_true_residual", 8, restart=True, replace_residual=True)
        continue_branch("shared_restart8_recursive_residual", 8, restart=True)
        continue_branch("shared_preserve8_true_residual", 8, replace_residual=True)
        continue_branch("shared_preserve32_existing8", 32)
        continue_branch("shared_preserve_full_existing8", n)
        continue_branch("shared_old_continuation8", 8, use_old=True)

        # A tempting but invalid reconstruction: the archived discarded pairs
        # are no longer jointly A-conjugate/residual-orthogonal. Keep the failed
        # probe explicit so it cannot be mistaken for a full-history result.
        continue_branch("invalid_restore_all20_probe", n, invalid_restore_all20=True)
        result["invalid_control"] = result["branches"].pop("invalid_restore_all20_probe")
        result["invalid_control"]["reason"] = (
            "Reinserting directions previously discarded by history8 violates both global "
            "A-conjugacy and residual orthogonality; this is not a valid full-history comparison.")

        for budget, label in [(32, "production_preserve32_from_zero"), (n, "production_preserve_full_from_zero")]:
            long_kernel = make_kernel()
            long_trace = Trace()
            long_x20 = {}

            def long_hook(x, r, iteration, history):
                long_trace.record(x, r, iteration, history)
                if iteration == 20:
                    long_x20["relative_difference_from_shared_x20"] = float(
                        np.linalg.norm(x-checkpoint["x"])/np.linalg.norm(checkpoint["x"]))

            def long_update(x, true_r, old, iteration, history):
                if iteration != 20:
                    return old, dict(accepted=False, exchanges=0)
                return make_new(long_kernel), dict(accepted=True, exchanges=4, kind="prescribed_update")

            long_solve = flexible_solve(long_kernel, y, lam,
                preconditioner_factory=lambda: make_old(long_kernel), history_size=budget,
                checkpoint=20, rtol=tolerance, maxiter=2000,
                update_callback=long_update, step_callback=long_hook)
            result["branches"][label] = dict(iterations=long_solve.metrics["iterations"],
                converged=long_solve.metrics["converged"], true_relative_residual=long_solve.metrics["true_relative_residual"],
                history_budget=budget, max_history_bytes=long_solve.metrics["max_history_bytes"],
                breakdown=long_solve.metrics["breakdown"], checkpoint_comparison=long_x20,
                diagnostic=long_trace.summary())
            save()

        warm_kernel = make_kernel()
        warm = flexible_solve(warm_kernel, y, lam, preconditioner_factory=lambda: make_new(warm_kernel),
                              history_size=8, checkpoint=20, rtol=tolerance, maxiter=1980,
                              x0=checkpoint["x"])
        result["branches"]["public_fixed_warm_start_new8"] = dict(
            iterations=20+warm.metrics["iterations"], iterations_after20=warm.metrics["iterations"],
            converged=warm.metrics["converged"], true_relative_residual=warm.metrics["true_relative_residual"],
            identical_to_manual_restart=bool(np.array_equal(warm.solution, x_restart)),
            relative_solution_difference_from_manual_restart=float(np.linalg.norm(warm.solution-x_restart)/np.linalg.norm(x_restart)))
        assert np.array_equal(warm.solution, x_restart)
        result["status"] = "complete"
        save()
        print(json.dumps({label: {key: value for key, value in branch.items() if key != "diagnostic"}
                          for label, branch in result["branches"].items()}, indent=2))
        print(json.dumps(dict(production_replay=result["production_replay"],
                              checkpoint=result["shared_checkpoint"], cpu_seconds=result["elapsed_cpu_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), (args.output or args.root/"results/transition-diagnostic.json").resolve())
