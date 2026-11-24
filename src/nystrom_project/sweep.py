"""Bounded, configuration-driven comparisons for method revision."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
from threadpoolctl import threadpool_limits
from .datasets import prepare


def run(root,config_path,out):
    root=Path(root).resolve();out=Path(out).resolve()
    config=json.loads(Path(config_path).read_text())
    if out.exists():raise FileExistsError(out)
    (out/"runs").mkdir(parents=True)
    source={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in sorted((root/"src").rglob("*.py"))}
    manifest=dict(protocol=config,status="running",source_sha256=source,completed_runs=0,
                  python=sys.version,platform=platform.platform(),order=[],failures=[],
                  timing="All setup, updates, kernel products and verification are inside worker solve time. Parent references and worker startup are outside.")
    with ZipFile(out/"source.zip","x",ZIP_DEFLATED) as z:
        for p in source:z.write(root/p,p)
        z.writestr("protocol.json",json.dumps(config,indent=2))
        z.write(root/"requirements-lock.txt","requirements-lock.txt")
    start=perf_counter()
    def save():
        manifest["elapsed_wall_seconds"]=perf_counter()-start
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")
    save()
    try:
        with threadpool_limits(limits=1):
            for workload in config["workloads"]:
                for seed in config["seeds"]:
                    for lam in config["regularization"]:
                        name=workload["dataset"];target=workload.get("target","observed")
                        size=f"-n{config.get('synthetic_n',640)}" if name=="synthetic" else ""
                        case=root/"data/prepared"/f"{name}{size}-{target}-s{seed}-lambda{lam:g}.npz"
                        if not case.exists():
                            case,_=prepare(root,name,seed,lam,target,config.get("synthetic_n",640))
                        for repeat in range(config.get("repetitions",1)):
                            variants=list(config["variants"])
                            np.random.default_rng(seed+100*repeat).shuffle(variants)
                            for variant in variants:
                                remaining=config["max_wall_seconds"]-(perf_counter()-start)
                                if remaining<=0:
                                    manifest["status"]="budget_exhausted";return
                                options=dict(rank=16,seed=seed,checkpoint=20,candidate_count=4,
                                             rtol=1e-8,maxiter=2000,block_size=128)
                                options.update(config.get("solver_defaults",{}))
                                options.update({k:v for k,v in variant.items() if k!="id"})
                                stem=f"{case.stem}-{variant['id']}-r{repeat}"
                                path=out/"runs"/f"{stem}.json"
                                command=[sys.executable,"-m","nystrom_project.worker","--case",str(case),
                                         "--options",json.dumps(options),"--output",str(path)]
                                try:
                                    subprocess.run(command,cwd=root,env=env,check=True,capture_output=True,
                                                   text=True,timeout=min(remaining,config.get("run_timeout_seconds",180)))
                                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                                    message=error.stderr or str(error)
                                    if isinstance(message,bytes):message=message.decode()
                                    manifest["failures"].append(dict(run=stem,error=message))
                                    if isinstance(error,subprocess.TimeoutExpired) and remaining<=config.get("run_timeout_seconds",180):
                                        manifest["status"]="budget_exhausted";return
                                    raise
                                record=json.loads(path.read_text());m=record["metrics"]
                                record["variant_id"]=variant["id"]
                                path.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
                                manifest["order"].append(stem);manifest["completed_runs"]+=1
                                print(json.dumps(dict(run=stem,converged=m["converged"],iterations=m["iterations"],
                                                     seconds=round(m["total_seconds"],4),update=round(m["update_seconds"],4),
                                                     exchanges=m["accepted_exchanges"])),flush=True)
                                save()
            manifest["status"]="complete"
    except BaseException:
        manifest["status"]="failed";raise
    finally:
        if any(sha256((root/p).read_bytes()).hexdigest()!=h for p,h in source.items()):
            manifest["status"]="source_changed_during_run"
        save()
        contract=root/"docs/method-revision.json"
        if contract.exists():
            c=json.loads(contract.read_text())
            c["consumed_command_seconds"]+=manifest["elapsed_wall_seconds"]
            c["events"].append(dict(experiment=str(out.relative_to(root)),status=manifest["status"],
                                    elapsed_seconds=manifest["elapsed_wall_seconds"],completed_runs=manifest["completed_runs"]))
            contract.write_text(json.dumps(c,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.root,args.config,args.output)
