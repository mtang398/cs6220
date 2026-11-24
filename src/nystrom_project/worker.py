"""One isolated timed solve. Dense reference construction runs in the parent."""
import argparse
import json
from pathlib import Path
import resource

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits, threadpool_info

from .numerics import GaussianKernel
from .solvers import solve
from .revisions import revised_solve


def run(case_path, options):
    with np.load(case_path) as data:
        X = data["X_train"]
        Xtest = data["X_test"]
        y = data["y_train"]
        ytest = data["y_test"]
        reference = data["alpha_reference"]
        pred_reference = data["prediction_reference"]
        bandwidth = float(data["bandwidth"])
        ystd = float(data["y_std"])
        case = json.loads(str(data["config"]))
    with threadpool_limits(limits=1):
        kernel = GaussianKernel(X, bandwidth, block_size=options.get("block_size", 128))
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        solver = revised_solve if options["method"].startswith("flex_") else solve
        result = solver(kernel, y, case["regularization"], **{
            key: options[key] for key in ["method", "rank", "seed", "checkpoint",
                "candidate_count", "rtol", "maxiter", "restart_iterations", "history_size",
                "update_limit", "min_improvement", "warmup_steps", "sparsity"] if key in options})
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        # These accuracy computations occur after the solve timer and RSS capture.
        difference = result.solution-reference
        relative_solution_error = float(np.linalg.norm(difference)/max(np.linalg.norm(reference), 1e-300))
        energy_error_squared = float(difference @ (
            kernel.matvec(difference)+case["regularization"]*difference))
        pred = np.empty(len(ytest))
        for start in range(0, len(ytest), 128):
            block = Xtest[start:start+128]
            cross = np.exp(-cdist(block, X, "sqeuclidean")/(2*bandwidth**2))
            pred[start:start+len(block)] = cross @ result.solution
        pred_difference = float(np.linalg.norm(pred-pred_reference)/max(np.linalg.norm(pred_reference), 1e-300))
        result.metrics.update(
            relative_solution_error=relative_solution_error,
            energy_error_squared=max(0.0, energy_error_squared),
            relative_prediction_difference=pred_difference,
            test_rmse=float(np.linalg.norm(pred-ytest)/np.sqrt(len(pred))*ystd),
            reference_test_rmse=float(np.linalg.norm(pred_reference-ytest)/np.sqrt(len(pred))*ystd),
            peak_rss_before_bytes=rss_before, peak_rss_bytes=peak_rss,
            incremental_peak_rss_bytes=max(0, peak_rss-rss_before),
            blas_threads=[p["num_threads"] for p in threadpool_info()],
        )
    return {"case": case, "options": options, "metrics": result.metrics,
            "history": result.history, "events": result.events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--options", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = run(args.case, json.loads(args.options))
    # Interrupted workers must never leave a truncated file named as a finished run.
    temporary = args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(record["metrics"], allow_nan=False))

