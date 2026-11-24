"""Run only records missing after a bounded campaign stops; preserve its directory."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,required=True)
parser.add_argument("--campaign",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--max-wall-seconds",type=float,default=120)
args=parser.parse_args()
root=args.root.resolve(); base=args.campaign.resolve(); out=args.output.resolve()
original=json.loads((base/"manifest.json").read_text())
if original["status"] == "running":
    raise RuntimeError("Wait for the original campaign to stop")
if original["status"] == "complete":
    raise RuntimeError("The original campaign is already complete")
for path,digest in original["source_sha256"].items():
    if sha256((root/path).read_bytes()).hexdigest() != digest:
        raise RuntimeError("Continuation requires unchanged numerical and solver source")
if out.exists():
    raise FileExistsError(out)
(out/"runs").mkdir(parents=True)
config=original["protocol"]
schedules={(r["case"],r["rank"]):r["accepted_iterations"]
           for r in original["restart_schedule_calibrations"]}
manifest=dict(protocol=config,status="running",completed_runs=0,
              source_sha256=original["source_sha256"], continuation_of=str(base),
              max_wall_seconds=args.max_wall_seconds,run_order=[],
              reason="Complete only missing measurements after the original execution budget ended.")
start=perf_counter()
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
try:
    for workload in config["workloads"]:
        name=workload["dataset"]
        size=f"-n{config['synthetic_n']}" if name == "synthetic" else ""
        target=workload.get("target","observed")
        for seed in config["seeds"]:
            for lam in config["regularization"]:
                case=f"{name}{size}-{target}-s{seed}-lambda{lam:g}"
                path=root/"data/prepared"/f"{case}.npz"
                for rank in config["ranks"]:
                    for repeat in range(config["repetitions"]):
                        import numpy as np
                        methods=list(config["methods"])
                        np.random.default_rng(seed+repeat*100).shuffle(methods)
                        for method in methods:
                            stem=f"{case}-k{rank}-{method}-r{repeat}"
                            existing=base/"runs"/f"{stem}.json"
                            if existing.exists():
                                try:
                                    prior=json.loads(existing.read_text())
                                    if "metrics" in prior:
                                        continue
                                except json.JSONDecodeError:
                                    pass  # Preserve the interrupted record and complete it separately.
                            options=dict(method=method,rank=rank,seed=seed,checkpoint=config["checkpoint"],
                                         candidate_count=config["candidates"],rtol=config["rtol"],
                                         maxiter=config["maxiter"],block_size=config["block_size"])
                            if method == "matched_restart":
                                options["restart_iterations"]=schedules[(case,rank)]
                            remaining=args.max_wall_seconds-(perf_counter()-start)
                            if remaining <= 0:
                                raise TimeoutError("Continuation budget exhausted")
                            destination=out/"runs"/f"{stem}.json"
                            subprocess.run([sys.executable,"-m","nystrom_project.worker",
                                            "--case",str(path),"--options",json.dumps(options),
                                            "--output",str(destination)],cwd=root,env=env,
                                           check=True,capture_output=True,text=True,
                                           timeout=min(remaining,config["run_timeout_seconds"]))
                            record=json.loads(destination.read_text())
                            if method == "residual_exchange" and record["metrics"]["accepted_iterations"] != schedules[(case,rank)]:
                                raise RuntimeError("Adaptive restart schedule changed")
                            manifest["completed_runs"]+=1
                            manifest["run_order"].append(stem)
                            print(stem,record["metrics"]["converged"],flush=True)
    manifest["status"]="complete"
except BaseException as error:
    manifest["status"]="failed"
    manifest["failure"]=str(error)
    raise
finally:
    manifest["elapsed_wall_seconds"]=perf_counter()-start
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
