#!/usr/bin/env python3
"""Offline diagnosis of selected sets; dense spectral work is not solver timing.

Rebuild recorded initial/final Nyström sets and solve from zero with a fixed
preconditioner and history-8 FCG. Also replay RP factor reuse for the initial
set, to expose finite-precision differences caused solely by reconstruction.
The only output is a new diagnostic JSON; recorded experiments are read-only.
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
from nystrom_project.flexible import flexible_solve
from nystrom_project.numerics import GaussianKernel, NystromPreconditioner, initialize_rpcholesky


def main(root: Path, output: Path, cpu_limit: int = 55) -> None:
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic: {output}")
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit+1))
    wall_started, cpu_started = perf_counter(), process_time()
    run_prefix = "synthetic-n640-smooth-s0-lambda0.001-"
    sources = {
        "curvature4once_final": root/"results/revision-curvature-001/runs"/(run_prefix+"curvature4once-r0.json"),
        "energy1_final": root/"results/revision-screen-001/runs"/(run_prefix+"fcg_energy1-r0.json"),
        "direction1_final": root/"results/revision-screen-001/runs"/(run_prefix+"fcg_direction1-r0.json"),
    }
    records = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in sources.items()}
    first = records["energy1_final"]
    initial_indices = first["metrics"]["initial_indices"]
    for record in records.values():
        assert record["metrics"]["initial_indices"] == initial_indices
        assert record["case"]["dataset_sha256"] == first["case"]["dataset_sha256"]
        assert record["case"]["train_indices_sha256"] == first["case"]["train_indices_sha256"]
    prepared = root/"data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz"
    with np.load(prepared) as data:
        X, y = data["X_train"], data["y_train"]
        bandwidth = float(data["bandwidth"])
        reference = data["alpha_reference"]
        prepared_config = json.loads(str(data["config"]))
    assert prepared_config["dataset_sha256"] == first["case"]["dataset_sha256"]
    assert prepared_config["train_indices_sha256"] == first["case"]["train_indices_sha256"]
    lam = first["case"]["regularization"]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "purpose": "Separate fixed-set preconditioner quality from changing-preconditioner history loss",
        "scope": "One development case; offline dense accuracy/spectral diagnostic, not a timing benchmark",
        "status": "running", "case": first["case"],
        "protocol": dict(history_size=8, checkpoint=20, rtol=1e-8, maxiter=2000,
                         start="zero", preconditioner_updates=False,
                         blas_threads=1, cpu_limit_seconds=cpu_limit),
        "source_runs": {label: dict(path=str(path.relative_to(root)), sha256=sha256(path.read_bytes()).hexdigest())
                        for label, path in sources.items()},
        "prepared_sha256": sha256(prepared.read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                          for path in [root/"src/nystrom_project/numerics.py",
                                       root/"src/nystrom_project/flexible.py", Path(__file__).resolve()]},
        "sets": [],
    }

    def save():
        result["elapsed_wall_seconds"] = perf_counter()-wall_started
        result["elapsed_cpu_seconds"] = process_time()-cpu_started
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    with threadpool_limits(limits=1):
        diagnostic_kernel = GaussianKernel(X, bandwidth)
        K = diagnostic_kernel.dense()
        A = K+lam*np.eye(len(y))
        logdet_A = float(2*np.log(np.diag(linalg.cholesky(A, lower=True))).sum())
        result["reference"] = dict(
            trace_K=float(np.trace(K)), logdet_A=logdet_A,
            relative_residual=float(np.linalg.norm(y-A @ reference)/np.linalg.norm(y)))
        choices = [("initial_rebuilt", initial_indices, None, False),
                   ("initial_rp_factor", initial_indices, None, True)]
        choices.extend((label, record["metrics"]["final_indices"], record, False)
                       for label, record in records.items())
        for label, indices, source_record, reuse_rp in choices:
            if process_time()-cpu_started > cpu_limit-5:
                result["status"] = "budget_exhausted"
                save()
                return
            indices = np.asarray(indices, dtype=np.int64)
            kernel = GaussianKernel(X, bandwidth, block_size=128)

            def factory():
                if reuse_rp:
                    state = initialize_rpcholesky(
                        kernel, len(indices), lam,
                        np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                    np.testing.assert_array_equal(state.indices, indices)
                    return state
                columns = np.column_stack([kernel.column(int(j)) for j in indices])
                return NystromPreconditioner(indices, columns, lam)

            state = factory()
            approximation = (state.basis*state.eigenvalues) @ state.basis.T
            M = approximation+lam*np.eye(len(y))
            M = (M+M.T)*0.5
            spectrum = linalg.eigvalsh(A, M, check_finite=False, driver="gvd")
            logdet_M = float((len(y)-state.rank)*np.log(lam)
                            + np.log(lam+state.eigenvalues).sum())
            logdet_direct = float(2*np.log(np.diag(linalg.cholesky(M, lower=True))).sum())
            assert abs(logdet_direct-logdet_M) < 1e-7
            row = dict(
                label=label, indices=indices.tolist(), effective_rank=int(state.rank),
                factor_source=state.diagnostics["factor_source"],
                trace_Khat=float(state.eigenvalues.sum()),
                trace_K_minus_Khat=float(np.trace(K)-state.eigenvalues.sum()),
                logdet_M=logdet_M, logdet_A_minus_logdet_M=logdet_A-logdet_M,
                logdet_dense_check_absolute_error=abs(logdet_direct-logdet_M),
                generalized_eigenvalue_min=float(spectrum[0]),
                generalized_eigenvalue_max=float(spectrum[-1]),
                generalized_condition_number=float(spectrum[-1]/spectrum[0]),
                generalized_eigenvalue_median=float(np.median(spectrum)),
                generalized_eigenvalues_largest_10=spectrum[-10:].tolist(),
                generalized_eigenvalues_above_10=int(np.count_nonzero(spectrum > 10)),
                generalized_eigenvalues_above_100=int(np.count_nonzero(spectrum > 100)),
                initial_rhs_score=float(state.score(y)))
            del state, approximation, M, spectrum
            # Factory construction inside the solve retains no stale caller-owned
            # initial state; its setup work is included in the solver's metrics.
            solved = flexible_solve(kernel, y, lam, preconditioner_factory=factory,
                                    history_size=8, checkpoint=20, rtol=1e-8, maxiter=2000)
            measured = float(np.linalg.norm(y-A @ solved.solution)/np.linalg.norm(y))
            assert abs(measured-solved.metrics["true_relative_residual"]) < 1e-10
            row["frozen_solve"] = dict(
                converged=solved.metrics["converged"], iterations=solved.metrics["iterations"],
                true_relative_residual=measured,
                relative_solution_error=float(np.linalg.norm(solved.solution-reference)/np.linalg.norm(reference)),
                matvecs=solved.metrics["matvecs"], breakdown=solved.metrics["breakdown"],
                accepted_exchanges=solved.metrics["accepted_exchanges"],
                max_history_bytes=solved.metrics["max_history_bytes"])
            if source_record is not None:
                row["source_adaptive_solve"] = dict(
                    iterations=source_record["metrics"]["iterations"],
                    accepted_exchanges=source_record["metrics"]["accepted_exchanges"],
                    accepted_iterations=source_record["metrics"]["accepted_iterations"])
            result["sets"].append(row)
            save()
            print(json.dumps({key: row[key] for key in
                              ["label", "trace_Khat", "logdet_M", "generalized_condition_number", "frozen_solve"]}), flush=True)
        baseline = result["sets"][0]
        for row in result["sets"]:
            row["relative_to_initial_rebuilt"] = dict(
                iteration_ratio=row["frozen_solve"]["iterations"]/baseline["frozen_solve"]["iterations"],
                condition_ratio=row["generalized_condition_number"]/baseline["generalized_condition_number"],
                trace_Khat_change=row["trace_Khat"]-baseline["trace_Khat"],
                logdet_M_change=row["logdet_M"]-baseline["logdet_M"])
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-limit", type=int, default=55)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    main(root, (arguments.output or root/"results/frozen-set-diagnostic.json").resolve(), arguments.cpu_limit)
