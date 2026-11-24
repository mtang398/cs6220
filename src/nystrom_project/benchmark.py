"""Reproducible paired experiments, raw records, and timing summaries."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

from .datasets import prepare
from .numerics import GaussianKernel
from .solvers import solve


def summarize(destination):
    import csv
    from collections import defaultdict
    groups = defaultdict(list)
    incomplete = []
    for file in sorted((destination/"runs").glob("*.json")):
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            incomplete.append(dict(file=file.name, error=str(error)))
            continue
        if "metrics" not in record:
            continue
        c, o = record["case"], record["options"]
        groups[(c["dataset"], c["target"], c["seed"], c["regularization"], o["rank"], o["method"])].append(record)
    if incomplete:
        (destination/"incomplete-records.json").write_text(json.dumps(incomplete, indent=2)+"\n", encoding="utf-8")
    rows = []
    for key, records in sorted(groups.items()):
        metrics = [r["metrics"] for r in records]
        times = [m["total_seconds"] for m in metrics]
        row = dict(zip(["dataset", "target", "seed", "regularization", "rank", "method"], key))
        successful_times = [m["total_seconds"] for m in metrics if m["converged"]]
        row.update(repetitions=len(records), converged_repetitions=len(successful_times),
                   all_converged=all(m["converged"] for m in metrics),
                   median_time_to_accuracy_seconds=(float(np.median(successful_times))
                                                    if len(successful_times) == len(times) else None),
                   median_seconds=float(np.median(times)), min_seconds=min(times), max_seconds=max(times),
                   iterations=int(np.median([m["iterations"] for m in metrics])),
                   matvecs=int(np.median([m["matvecs"] for m in metrics])),
                   median_update_seconds=float(np.median([m["update_seconds"] for m in metrics])),
                   max_true_relative_residual=max(m["true_relative_residual"] for m in metrics),
                   max_relative_solution_error=max(m["relative_solution_error"] for m in metrics),
                   max_relative_prediction_difference=max(m["relative_prediction_difference"] for m in metrics),
                   accepted_exchanges=int(np.median([m["accepted_exchanges"] for m in metrics])),
                   numerical_fallbacks=int(np.median([m["numerical_fallbacks"] for m in metrics])),
                   max_peak_rss_bytes=max(m["peak_rss_bytes"] for m in metrics),
                   max_preconditioner_state_bytes=max(m["max_preconditioner_state_bytes"] for m in metrics))
        rows.append(row)
    if rows:
        with (destination/"summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def run(root, config_path, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Preserve existing measurements: {destination}")
    (destination/"runs").mkdir(parents=True, exist_ok=True)
    source = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
              for p in sorted((root/"src").rglob("*.py"))}
    with ZipFile(destination/"source.zip", "x", ZIP_DEFLATED) as snapshot:
        for relative in source:
            snapshot.write(root/relative, relative)
        for name in ("pyproject.toml", "requirements-lock.txt"):
            snapshot.write(root/name, name)
        snapshot.writestr("protocol.json", json.dumps(config, indent=2)+"\n")
    manifest = dict(protocol=config, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), cpu=platform.processor(),
                    source_sha256=source, status="running", completed_runs=0,
                    memory_definition="Linux process peak RSS through the end of solve; includes interpreter/libraries and input arrays. Dense references run in parent process.",
                    excluded_from_timing="Dataset download/preprocessing, bandwidth calculation, dense reference, untimed restart-schedule calibration, worker startup, post-solve accuracy checks",
                    restart_schedule_calibrations=[], run_order=[])
    manifest_path = destination/"manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    save()
    started = perf_counter()
    budget = config.get("max_wall_seconds", 900)
    failures = []
    completed = 0
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    try:
        with threadpool_limits(limits=1):
            manifest["parent_threadpools"] = threadpool_info()
            for workload in config["workloads"]:
                for seed in config["seeds"]:
                    for regularization in config["regularization"]:
                        case_path, case = prepare(root, workload["dataset"], seed, regularization,
                                                  target=workload.get("target", "observed"),
                                                  synthetic_n=config.get("synthetic_n", 640))
                        for rank in config["ranks"]:
                            accepted_iterations = None
                            if "matched_restart" in config["methods"]:
                                # Derive the control schedule separately, then randomize all timed runs.
                                with np.load(case_path) as data:
                                    calibration = solve(
                                        GaussianKernel(data["X_train"], float(data["bandwidth"]),
                                                       block_size=config.get("block_size", 128)),
                                        data["y_train"], regularization, method="residual_exchange",
                                        rank=rank, seed=seed, checkpoint=config["checkpoint"],
                                        candidate_count=config["candidates"], rtol=config["rtol"],
                                        maxiter=config["maxiter"])
                                accepted_iterations = calibration.metrics["accepted_iterations"]
                                manifest["restart_schedule_calibrations"].append(dict(
                                    case=case_path.stem, rank=rank, accepted_iterations=accepted_iterations,
                                    excluded_seconds=calibration.metrics["total_seconds"]))
                            for repeat in range(config["repetitions"]):
                                methods = list(config["methods"])
                                np.random.default_rng(seed+repeat*100).shuffle(methods)
                                for method in methods:
                                    remaining = budget-(perf_counter()-started)
                                    if remaining <= 0:
                                        manifest["status"] = "budget_exhausted"
                                        return
                                    options = dict(method=method, rank=rank, seed=seed,
                                                   checkpoint=config["checkpoint"], candidate_count=config["candidates"],
                                                   rtol=config["rtol"], maxiter=config["maxiter"],
                                                   block_size=config.get("block_size", 128))
                                    if method == "matched_restart":
                                        if accepted_iterations is None:
                                            raise RuntimeError("Adaptive run required before matched restart")
                                        options["restart_iterations"] = accepted_iterations
                                    stem = f"{case_path.stem}-k{rank}-{method}-r{repeat}"
                                    output = destination/"runs"/f"{stem}.json"
                                    command = [sys.executable, "-m", "nystrom_project.worker", "--case", str(case_path),
                                               "--options", json.dumps(options), "--output", str(output)]
                                    try:
                                        proc = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                                              text=True, encoding="utf-8", check=True,
                                                              timeout=min(remaining, config.get("run_timeout_seconds", 120)))
                                        record = json.loads(output.read_text(encoding="utf-8"))
                                        if method == "residual_exchange" and accepted_iterations is not None:
                                            if record["metrics"]["accepted_iterations"] != accepted_iterations:
                                                raise RuntimeError("Timed adaptive run differs from calibrated restart schedule")
                                        completed += 1
                                        manifest["run_order"].append(stem)
                                        print(json.dumps(dict(run=stem, converged=record["metrics"]["converged"],
                                                              seconds=round(record["metrics"]["total_seconds"], 4),
                                                              iterations=record["metrics"]["iterations"])), flush=True)
                                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                                        message = error.stderr or str(error)
                                        if isinstance(message, bytes): message = message.decode("utf-8")
                                        failures.append(dict(run=stem, error=message))
                                        (destination/"failures.json").write_text(json.dumps(failures, indent=2)+"\n", encoding="utf-8")
                                        if isinstance(error, subprocess.TimeoutExpired) and remaining <= config.get("run_timeout_seconds", 120):
                                            manifest["status"] = "budget_exhausted"
                                            return
                                        raise
                                    manifest["completed_runs"] = completed
                                    manifest["elapsed_wall_seconds"] = perf_counter()-started
                                    save()
            manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["completed_runs"] = completed
        manifest["elapsed_wall_seconds"] = perf_counter()-started
        manifest["failures"] = failures
        for relative, digest in source.items():
            if sha256((root/relative).read_bytes()).hexdigest() != digest:
                manifest["status"] = "source_changed_during_run"
        save()
        rows = summarize(destination)
        print(json.dumps(dict(status=manifest["status"], completed_runs=completed,
                              summary_rows=len(rows), elapsed=manifest["elapsed_wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.config, args.output)
