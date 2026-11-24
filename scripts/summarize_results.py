"""Summarize paired observations without treating checkpoints as independent trials."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def summarize(root, campaign, diagnostic, destination, continuation=None):
    manifest = json.loads((campaign/"manifest.json").read_text())
    manifests=[manifest]
    campaigns=[campaign]
    if continuation is not None:
        continuation_manifest=json.loads((continuation/"manifest.json").read_text())
        if continuation_manifest["protocol"] != manifest["protocol"] or continuation_manifest["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Continuation must use the original protocol and source")
        manifests.append(continuation_manifest)
        campaigns.append(continuation)
    groups = defaultdict(list)
    candidates=[p for directory in campaigns for p in sorted((directory/"runs").glob("*.json"))]
    files=[]
    incomplete_records=[]
    for p in candidates:
        try:
            json.loads(p.read_text())
            files.append(p)
        except json.JSONDecodeError as error:
            incomplete_records.append(dict(path=str(p),error=str(error)))
    if len({p.name for p in files}) != len(files):
        raise ValueError("Duplicate measurements across campaigns")
    for path in files:
        record = json.loads(path.read_text())
        c, o = record["case"], record["options"]
        key = (c["dataset"], c["target"], c["seed"], c["regularization"], o["rank"])
        groups[key].append(record)
    pairs = []
    for key, records in sorted(groups.items()):
        methods = defaultdict(list)
        for record in records:
            methods[record["options"]["method"]].append(record)
        row = dict(zip(("dataset", "target", "seed", "regularization", "rank"), key))
        row["methods"] = {}
        for method, runs in methods.items():
            metrics = [r["metrics"] for r in runs]
            valid = len(runs) == manifest["protocol"]["repetitions"] and all(m["converged"] for m in metrics)
            row["methods"][method] = dict(
                repetitions=len(runs), complete_and_converged=valid,
                median_attempt_seconds=median(m["total_seconds"] for m in metrics),
                median_time_to_accuracy_seconds=median(m["total_seconds"] for m in metrics) if valid else None,
                iterations=[m["iterations"] for m in metrics],
                median_update_seconds=median(m["update_seconds"] for m in metrics),
                median_restarts=median(m["restarts"] for m in metrics),
                median_matvecs=median(m["matvecs"] for m in metrics),
                max_true_relative_residual=max(m["true_relative_residual"] for m in metrics),
                max_relative_solution_error=max(m["relative_solution_error"] for m in metrics),
                max_relative_prediction_difference=max(m["relative_prediction_difference"] for m in metrics),
                max_numerical_fallbacks=max(m["numerical_fallbacks"] for m in metrics))
        if "fixed" in methods and "residual_exchange" in methods:
            a, b = methods["fixed"][0], methods["residual_exchange"][0]
            if a["case"]["dataset_sha256"] != b["case"]["dataset_sha256"]:
                raise ValueError("Unpaired data")
            if a["case"]["train_indices_sha256"] != b["case"]["train_indices_sha256"]:
                raise ValueError("Unpaired split")
            if a["metrics"]["initial_indices"] != b["metrics"]["initial_indices"]:
                raise ValueError("Unpaired initial columns")
            f, a = row["methods"]["fixed"], row["methods"]["residual_exchange"]
            if f["complete_and_converged"] and a["complete_and_converged"]:
                row["adaptive_over_fixed_time"] = a["median_time_to_accuracy_seconds"]/f["median_time_to_accuracy_seconds"]
                row["adaptive_over_fixed_iterations"] = median(a["iterations"])/median(f["iterations"])
        pairs.append(row)
    workload_groups = defaultdict(list)
    for row in pairs:
        workload_groups[(row["dataset"], row["target"])].append(row)
    workloads = []
    for (dataset, target), rows in sorted(workload_groups.items()):
        eligible = [r for r in rows if "adaptive_over_fixed_time" in r]
        methods = defaultdict(list)
        for row in rows:
            for method, data in row["methods"].items():
                if data["complete_and_converged"]:
                    methods[method].append(data)
        workloads.append(dict(
            dataset=dataset, target=target, fully_paired_seeds=len(eligible),
            adaptive_faster_seeds=sum(r["adaptive_over_fixed_time"] < 1 for r in eligible),
            median_paired_time_ratio=median(r["adaptive_over_fixed_time"] for r in eligible) if eligible else None,
            min_paired_time_ratio=min((r["adaptive_over_fixed_time"] for r in eligible), default=None),
            max_paired_time_ratio=max((r["adaptive_over_fixed_time"] for r in eligible), default=None),
            methods={m:dict(converged_seeds=len(v),
                            median_seconds=median(x["median_time_to_accuracy_seconds"] for x in v),
                            median_iterations=median(median(x["iterations"]) for x in v),
                            median_update_seconds=median(x["median_update_seconds"] for x in v))
                     for m,v in methods.items()}))

    d = json.loads(diagnostic.read_text())
    diagnostics = []
    for lam in d["protocol"]["regularization"]:
        cases = [c for c in d["cases"] if c["case"]["regularization"] == lam]
        checkpoints = [p for c in cases for p in c["checkpoints"]]
        horizons = []
        for horizon in d["protocol"]["horizons"]:
            values = [h["energy_ratio_new_over_old"] for p in checkpoints for h in p["horizons"]
                      if h["requested_steps"] == horizon and h["energy_ratio_new_over_old"] is not None]
            horizons.append(dict(steps=horizon, checkpoints=len(values), new_better=sum(v < 1 for v in values),
                                 median_energy_ratio=median(values)))
        diagnostics.append(dict(regularization=lam, solves=len(cases),
                                converged_solves=sum(c["solve_converged"] for c in cases),
                                checkpoints=len(checkpoints),
                                strictly_reduced_scores=sum(p["score_ratio"] < 1 for p in checkpoints),
                                median_score_ratio=median(p["score_ratio"] for p in checkpoints),
                                horizons=horizons))
    expected=(len(manifest["protocol"]["workloads"])*len(manifest["protocol"]["seeds"])*
              len(manifest["protocol"]["regularization"])*len(manifest["protocol"]["ranks"])*
              len(manifest["protocol"]["methods"])*manifest["protocol"]["repetitions"])
    status="complete" if len(files)==expected and manifests[-1]["status"]=="complete" else manifests[-1]["status"]
    output = dict(campaign_status=status, completed_runs=len(files),
                  incomplete_records_preserved=incomplete_records,
                  source_campaigns=[dict(path=str(p),status=m["status"],completed_runs=m["completed_runs"],
                                         elapsed_wall_seconds=m["elapsed_wall_seconds"])
                                    for p,m in zip(campaigns,manifests)],
                  elapsed_wall_seconds=sum(m["elapsed_wall_seconds"] for m in manifests),
                  analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  campaign=str(campaign), diagnostic=str(diagnostic), workloads=workloads,
                  paired_cases=pairs, diagnostics=diagnostics,
                  interpretation="Descriptive paired observations. Timing repetitions share randomness; checkpoints within a solve are dependent. No significance tests or population claims.")
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=output["campaign_status"], completed_runs=output["completed_runs"],
                          workloads=workloads, diagnostics=diagnostics), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.campaign, args.diagnostic, args.output, args.continuation)
