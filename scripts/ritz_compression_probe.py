"""Bounded diagnostic of Ritz compression of FCG history; not a benchmark.

The fixed Nyström factor never changes. At a full-history append, retain a
few largest generalized Ritz vectors of (A,M) from the older directions,
optionally keep the latest old directions, and then append the newest p,Ap.
The original failed variants have recent_old=0 and already preserve newest p.
This is a local experiment, not a reproduction of Lan-DR or eigCG.

Run from repository root with PYTHONPATH=project/src, one BLAS thread, and
an unused output path. Existing result files are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy import linalg
try:
    from threadpoolctl import threadpool_limits, threadpool_info
except ImportError:
    from contextlib import nullcontext
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("Without threadpoolctl, set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1")
    def threadpool_limits(limits):
        return nullcontext()
    def threadpool_info():
        return {"threadpoolctl": "unavailable", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

import nystrom_project.flexible as flexible
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    project = Path(__file__).resolve().parents[1]
    case = project / "data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz"
    with np.load(case) as data:
        X, y, bandwidth = data["X_train"], data["y_train"], float(data["bandwidth"])
    source_files = [Path(__file__).resolve()] + sorted((project / "src").rglob("*.py"))
    payload = {
        "purpose": "untimed fixed-factor history-compression mechanism diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(case.relative_to(project)),
        "input_sha256": sha256(case.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(project)): sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__,
        "settings": {"rank": 16, "regularization": .001, "history_size": 8,
                     "checkpoint": 20, "rtol": 1e-8, "maxiter": 1000, "seed": 0},
        "initial_in_memory_observation": [
            {"ritz": 0, "recent_old": 0, "iterations": 138},
            {"ritz": 2, "recent_old": 0, "iterations": 185},
            {"ritz": 3, "recent_old": 0, "iterations": 192},
            {"ritz": 4, "recent_old": 0, "iterations": 187}],
        "runs": [],
    }
    original = flexible.AConjugateHistory
    variants = [(0, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 4), (3, 2), (1, 4)]
    started = perf_counter()
    try:
        with threadpool_limits(limits=1):
            payload["threadpools"] = threadpool_info()
            for keep, recent_old in variants:
                kernel = GaussianKernel(X, bandwidth, block_size=128)
                model = initialize_rpcholesky(
                    kernel, 16, .001,
                    np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                diagnostics = {"compressions": 0, "max_history_gram_error": 0.0}

                class CompressedHistory(original):
                    def append(self, p, ap):
                        if self.size == self.max_history:
                            pairs = list(self._pairs)
                            retained = pairs[-recent_old:] if recent_old else []
                            older = pairs[:-recent_old] if recent_old else pairs
                            V = np.column_stack([v for v, _, _ in older])
                            AV = np.column_stack([v for _, v, _ in older])
                            MV = (.001 * V + model.basis @ (
                                model.eigenvalues[:, None] * (model.basis.T @ V)))
                            HA, HM = V.T @ AV, V.T @ MV
                            # Small nu of (M,A) equals large theta of (A,M).
                            _, C = linalg.eigh((HM + HM.T) * .5, (HA + HA.T) * .5,
                                               check_finite=False)
                            C = C[:, :keep]
                            U, AU = V @ C, AV @ C
                            self._pairs = deque()
                            for j in range(keep):
                                original.append(self, U[:, j], AU[:, j])
                            for v, av, _ in retained:
                                original.append(self, v, av)
                            diagnostics["compressions"] += 1
                        # p was formed against the full old history and is kept.
                        original.append(self, p, ap)
                        V = np.column_stack([v for v, _, _ in self._pairs])
                        AV = np.column_stack([v for _, v, _ in self._pairs])
                        error = float(np.linalg.norm(V.T @ AV - np.eye(self.size), 2))
                        diagnostics["max_history_gram_error"] = max(
                            diagnostics["max_history_gram_error"], error)

                flexible.AConjugateHistory = CompressedHistory if keep else original
                result = flexible.flexible_solve(
                    kernel, y, .001, preconditioner=model, history_size=8,
                    checkpoint=20, rtol=1e-8, maxiter=1000)
                metrics = result.metrics
                row = {"ritz": keep, "recent_old": recent_old,
                       "newest_direction_preserved": True,
                       "iterations": metrics["iterations"],
                       "converged": metrics["converged"],
                       "true_relative_residual": metrics["true_relative_residual"],
                       "breakdown": metrics["breakdown"],
                       "direction_matvecs": metrics["direction_matvecs"],
                       "verification_matvecs": metrics["verification_matvecs"],
                       **diagnostics}
                payload["runs"].append(row)
                print(json.dumps(row), flush=True)
    finally:
        flexible.AConjugateHistory = original
    payload["diagnostic_wall_seconds"] = perf_counter() - started
    payload["timing_warning"] = "Shared-process diagnostic with extra Gram checks; no speed claim."
    payload["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
