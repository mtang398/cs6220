#!/usr/bin/env python3
"""Dense offline PCG study with a fixed retained-vector budget.

Use rank b-d Nyström and archive only the first d PCG direction/image pairs.
Their complete numerical span becomes a fixed coarse space; no Ritz selection
or sliding-history recurrence is used. Retained-array bytes include every
Nyström array and U/AU/coarse factor. Dense diagnostic matrices and temporary
factorization workspaces are excluded, so this is not a peak-memory benchmark.
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
from nystrom_project.numerics import GaussianKernel, NystromPreconditioner, initialize_rpcholesky
from check_projected_balancing import BalancedInverse


def pcg(A, b, inverse, *, capture_steps=0, x0=None, maxiter=2000, rtol=1e-8):
    """Ordinary fixed-preconditioner PCG; optional early diagnostic snapshot."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    norm_b = np.linalg.norm(b)
    r = b.copy() if x0 is None else b-A @ x
    z = inverse.apply(r)
    p, rho = z.copy(), float(r @ z)
    captured = dict(directions=[], images=[])
    matvecs, replacements = int(x0 is not None), 0
    converged, breakdown = False, None
    for iteration in range(1, maxiter+1):
        ap = A @ p
        matvecs += 1
        curvature = float(p @ ap)
        if rho <= 0 or curvature <= 0 or not np.isfinite(rho+curvature):
            breakdown = "nonpositive_or_nonfinite_inner_product"
            break
        if iteration <= capture_steps:
            scale = np.linalg.norm(p)
            captured["directions"].append(p/scale)
            captured["images"].append(ap/scale)
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        suspected = np.linalg.norm(r) <= rtol*norm_b
        if iteration % 20 == 0 or suspected or iteration == maxiter or iteration == capture_steps:
            true_r = b-A @ x
            matvecs += 1
            if iteration == capture_steps:
                captured.update(x=x.copy(), true_r=true_r.copy(), prefix_matvecs=matvecs,
                    recursive_true_gap_relative_rhs=float(np.linalg.norm(r-true_r)/norm_b))
            if np.linalg.norm(true_r) <= rtol*norm_b:
                converged = True
                break
            if suspected:
                r = true_r.copy()
                z = inverse.apply(r)
                rho, p = float(r @ z), z.copy()
                replacements += 1
                continue
        z = inverse.apply(r)
        new_rho = float(r @ z)
        p = z+(new_rho/rho)*p
        rho = new_rho
    true_relative = float(np.linalg.norm(b-A @ x)/norm_b)
    return dict(iterations=iteration, converged=converged, true_relative_residual=true_relative,
                residual_replacements=replacements, operator_applications=matvecs, breakdown=breakdown), captured


def main(root, output):
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic: {output}")
    resource.setrlimit(resource.RLIMIT_CPU, (55, 56))
    started, cpu_started = perf_counter(), process_time()
    result = dict(status="running", scope="One-seed dense offline accuracy study; no runtime or peak-memory claims",
        protocol=dict(regularization=.001, budgets=[16, 32, 64], coarse_dimensions=[4, 8, 16],
                      initial_rank="b-d", coarse_span="all first d ordinary PCG directions; numerical SVD basis",
                      capture_steps="exactly d", rtol=1e-8, maxiter=2000, blas_threads=1,
                      initial_coarse_correction=False, no_ritz_selection=True,
                      state_byte_definition="Nyström indices,columns,core,basis,eigenvalues plus U,AU,Cholesky(E); no Gram cache",
                      excluded_allocations="common PCG working vectors, temporary setup workspaces and all dense diagnostic arrays",
                      cpu_limit_seconds=55),
        source_sha256={str(path.relative_to(root)):sha256(path.read_bytes()).hexdigest()
                       for path in [root/"src/nystrom_project/numerics.py",
                                    root/"scripts/check_projected_balancing.py", Path(__file__).resolve()]},
        workloads=[])

    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    def initial_state(kernel, k):
        return initialize_rpcholesky(kernel, k, .001,
            np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))

    names=("synthetic-n640-smooth-s0-lambda0.001", "synthetic-n640-random-s0-lambda0.001",
           "concrete-observed-s0-lambda0.001", "airfoil-observed-s0-lambda0.001")
    with threadpool_limits(limits=1):
        for name in names:
            prepared=root/"data/prepared"/(name+".npz")
            with np.load(prepared) as data:
                X,y,bandwidth=data["X_train"],data["y_train"],float(data["bandwidth"])
                config=json.loads(str(data["config"]))
            kernel=GaussianKernel(X,bandwidth)
            K=kernel.dense()
            A=K+.001*np.eye(len(y))
            workload=dict(name=name,n=len(y),config=config,prepared_sha256=sha256(prepared.read_bytes()).hexdigest(),budgets=[])
            result["workloads"].append(workload)
            for b in (16,32,64):
                if process_time()-cpu_started > 45:
                    result["status"]="budget_exhausted";save();return
                full=initial_state(kernel,b)
                fixed_full,_=pcg(A,y,full)
                assert fixed_full["converged"]
                budget=dict(b=b,fixed_rank_b=dict(**fixed_full,indices=full.indices.tolist(),
                    state_bytes=int(full.storage_bytes)),allocations=[])
                workload["budgets"].append(budget)
                for d in (4,8,16):
                    if d>=b:continue
                    old=initial_state(kernel,b-d)
                    np.testing.assert_array_equal(old.indices,full.indices[:b-d])
                    base,captured=pcg(A,y,old,capture_steps=d)
                    assert base["converged"] and "x" in captured
                    P=np.column_stack(captured["directions"])
                    AP=np.column_stack(captured["images"])
                    left,singular,right=linalg.svd(P,full_matrices=False,check_finite=False)
                    keep=singular>1e-12*singular[0]
                    U=left[:,keep]
                    AU=(AP @ right[keep].T)/singular[keep]
                    cached_error=float(np.linalg.norm(AU-A @ U)/np.linalg.norm(A @ U))
                    assert cached_error<1e-9
                    E=(U.T @ AU+AU.T @ U)*.5
                    factor=linalg.cho_factor(E,lower=True,check_finite=False)
                    B=K+.001*(U @ U.T)-AU @ linalg.cho_solve(factor,AU.T,check_finite=False)
                    B=(B+B.T)*.5
                    schur=NystromPreconditioner(old.indices,B[:,old.indices],.001)
                    allocation=dict(d=d,initial_rank=b-d,coarse_numerical_rank=int(U.shape[1]),
                        singular_values=singular.tolist(),cached_AU_relative_error=cached_error,
                        prefix_direction_image_bytes=int(P.nbytes+AP.nbytes),
                        prefix_total_retained_bytes=int(old.storage_bytes+P.nbytes+AP.nbytes),
                        prefix_operator_applications=int(captured["prefix_matvecs"]),
                        fixed_base_uninterrupted=dict(**base,state_bytes=int(old.storage_bytes),indices=old.indices.tolist()),
                        prefix_recursive_true_residual_gap=captured["recursive_true_gap_relative_rhs"],branches={})
                    for label,state in (("balanced_old",old),("balanced_schur_old_indices",schur)):
                        inverse=BalancedInverse(state,U,AU)
                        checks=inverse.dense_check()
                        continued,_=pcg(A,y,inverse,x0=captured["x"])
                        assert continued["converged"]
                        assert inverse.storage_bytes <= full.storage_bytes
                        allocation["branches"][label]=dict(**continued,
                            total_iterations=d+continued["iterations"],
                            total_operator_applications=captured["prefix_matvecs"]+continued["operator_applications"],
                            frozen_retained_state_bytes=int(inverse.storage_bytes),
                            base_state_bytes=int(state.storage_bytes),
                            U_AU_bytes=int(U.nbytes+AU.nbytes),coarse_factor_bytes=int(inverse.L.nbytes),
                            byte_ratio_to_rank_b=float(inverse.storage_bytes/full.storage_bytes),
                            iteration_ratio_to_rank_b=float((d+continued["iterations"])/fixed_full["iterations"]),checks=checks)
                    budget["allocations"].append(allocation)
                    save()
                    print(json.dumps(dict(workload=name,b=b,d=d,fixed_b=fixed_full["iterations"],
                        fixed_base=base["iterations"],balanced=allocation["branches"]["balanced_old"]["total_iterations"],
                        schur=allocation["branches"]["balanced_schur_old_indices"]["total_iterations"],
                        full_bytes=full.storage_bytes,balanced_bytes=allocation["branches"]["balanced_old"]["frozen_retained_state_bytes"])),flush=True)
        result["status"]="complete";save()
        print(json.dumps(dict(status=result["status"],cpu_seconds=result["elapsed_cpu_seconds"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=PROJECT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    main(args.root.resolve(),(args.output or args.root/"results/balanced-budget-diagnostic.json").resolve())
