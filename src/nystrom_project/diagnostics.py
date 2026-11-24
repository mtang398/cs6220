"""Compare old and exchanged preconditioners from exactly the same iterate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from threadpoolctl import threadpool_limits

from .datasets import prepare
from .numerics import GaussianKernel
from .solvers import solve, fixed_steps


def run(root, config_path, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if destination.exists():
        raise FileExistsError(f"Preserve existing diagnostics: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    output = dict(protocol=config, python=sys.version, status="running", cases=[],
                  source_sha256={str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((root/"src").rglob("*.py"))},
                  timing_use="Diagnostic branches and dense references are excluded from benchmark results.")
    with ZipFile(destination.with_name(destination.stem+"-source.zip"), "x", ZIP_DEFLATED) as snapshot:
        for relative in output["source_sha256"]:
            snapshot.write(root/relative, relative)
        for name in ("pyproject.toml", "requirements-lock.txt"):
            snapshot.write(root/name, name)
        snapshot.writestr("protocol.json", json.dumps(config, indent=2)+"\n")

    def save():
        output["elapsed_wall_seconds"] = perf_counter()-started
        destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    with threadpool_limits(limits=1):
        for target in config["targets"]:
            for seed in config["seeds"]:
                for lam in config["regularization"]:
                    if perf_counter()-started > config["max_wall_seconds"]:
                        output["status"] = "budget_exhausted"
                        save()
                        return
                    case_path, metadata = prepare(root, "synthetic", seed, lam, target,
                                                  config["synthetic_n"])
                    with np.load(case_path) as data:
                        kernel = GaussianKernel(data["X_train"], float(data["bandwidth"]))
                        rhs, reference = data["y_train"], data["alpha_reference"]
                    A = kernel.dense()+lam*np.eye(kernel.n)
                    norm_rhs = np.linalg.norm(rhs)
                    norm_reference = np.linalg.norm(reference)

                    def errors(x):
                        e = x-reference
                        return dict(energy_error_squared=float(e @ (A @ e)),
                                    relative_residual=float(np.linalg.norm(rhs-A @ x)/norm_rhs),
                                    relative_coefficient_error=float(np.linalg.norm(e)/norm_reference))

                    checkpoints = []

                    def compare(x, old, new, iteration):
                        if len(checkpoints) >= config["max_checkpoints_per_solve"]:
                            return
                        residual = rhs-A @ x
                        before, after = old.score(residual), new.score(residual)
                        initial = errors(x)
                        checkpoint = dict(iteration=iteration, old_indices=old.indices.tolist(),
                                          new_indices=new.indices.tolist(), score_old=before,
                                          score_new=after, score_ratio=after/before,
                                          initial=initial, horizons=[])
                        max_steps = max(config["horizons"])
                        old_path = fixed_steps(kernel, rhs, lam, x, old, max_steps)
                        new_path = fixed_steps(kernel, rhs, lam, x, new, max_steps)
                        for horizon in config["horizons"]:
                            old_steps, new_steps = min(horizon, len(old_path)), min(horizon, len(new_path))
                            old_error = errors(old_path[old_steps-1] if old_steps else x)
                            new_error = errors(new_path[new_steps-1] if new_steps else x)
                            old_energy = old_error["energy_error_squared"]
                            new_energy = new_error["energy_error_squared"]
                            checkpoint["horizons"].append(dict(
                                requested_steps=horizon, old_steps=old_steps, new_steps=new_steps,
                                old=old_error, new=new_error,
                                energy_ratio_new_over_old=new_energy/old_energy if old_energy > 0 else None,
                                old_contraction=old_energy/initial["energy_error_squared"],
                                new_contraction=new_energy/initial["energy_error_squared"]))
                        checkpoints.append(checkpoint)

                    result = solve(kernel, rhs, lam, method="residual_exchange", rank=config["rank"],
                                   seed=seed, checkpoint=config["checkpoint"],
                                   candidate_count=config["candidates"], rtol=config["rtol"],
                                   maxiter=config["maxiter"], exchange_callback=compare)
                    output["cases"].append(dict(case=metadata, checkpoints=checkpoints,
                                                solve_converged=result.metrics["converged"],
                                                solve_iterations=result.metrics["iterations"],
                                                accepted_iterations=result.metrics["accepted_iterations"]))
                    save()
                    print(json.dumps(dict(target=target, seed=seed, regularization=lam,
                                          diagnostic_checkpoints=len(checkpoints))), flush=True)
    output["status"] = "complete"
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.config, args.output)
