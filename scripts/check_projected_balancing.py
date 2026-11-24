#!/usr/bin/env python3
"""Offline ordinary-PCG diagnostic for balanced complement-kernel Nyström.

Dense kernels and archived search directions are allowed only in this accuracy
diagnostic. No runtime comparison or matrix-free implementation is claimed.
Every branch uses ordinary fixed-SPD PCG, restarted at the same iteration-20 x.
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


def apply_columns(state, vectors):
    return np.column_stack([state.apply(v) for v in vectors.T])


def model_apply(state, vectors):
    return state.regularization*vectors+state.basis @ (state.eigenvalues[:, None]*(state.basis.T @ vectors))


def projected_rpcholesky(K, k, lam, rng):
    """Generic PSD RPCholesky: projected kernel diagonals are not all ones."""
    n = len(K)
    diagonal = np.maximum(np.diag(K).copy(), 0)
    selected = np.zeros(n, dtype=bool)
    F = np.zeros((n, k))
    indices = []
    rank = 0
    threshold = 64*np.finfo(float).eps*max(k, 1)*max(float(diagonal.max()), 1)
    for _ in range(k):
        diagonal[selected] = 0
        total = diagonal.sum()
        j = int(rng.choice(n, p=diagonal/total)) if total > threshold else int(rng.choice(np.flatnonzero(~selected)))
        indices.append(j)
        selected[j] = True
        remainder = K[:, j]-F[:, :rank] @ F[j, :rank]
        if remainder[j] > threshold:
            F[:, rank] = remainder/np.sqrt(remainder[j])
            diagonal = np.maximum(diagonal-F[:, rank]**2, 0)
            rank += 1
    indices = np.asarray(indices, dtype=int)
    return NystromPreconditioner(indices, K[:, indices], lam, factor=F[:, :rank],
                                 diagnostics=dict(initialization="projected_rpcholesky", rp_factor_rank=rank))


class BalancedInverse:
    def __init__(self, state, U, AU):
        self.state, self.U, self.AU = state, U, AU
        E = (U.T @ AU+AU.T @ U)*.5
        self.L = linalg.cholesky(E, lower=True, check_finite=False)

    def small_solve(self, value):
        return linalg.cho_solve((self.L, True), value, check_finite=False)

    def apply(self, r):
        coarse = self.small_solve(self.U.T @ r)
        z = self.state.apply(r-self.AU @ coarse)
        return z+self.U @ (coarse-self.small_solve(self.AU.T @ z))

    @property
    def storage_bytes(self):
        return self.state.storage_bytes+self.U.nbytes+self.AU.nbytes+self.L.nbytes

    def dense_check(self):
        n = self.state.n
        B, values, lam = self.state.basis, self.state.eigenvalues, self.state.regularization
        H0 = np.eye(n)/lam+(B*(1/(lam+values)-1/lam)) @ B.T
        E_inverse = self.small_solve(np.eye(self.L.shape[0]))
        F = self.AU @ E_inverse
        H0F = apply_columns(self.state, F)
        H = H0-self.U @ H0F.T-H0F @ self.U.T+self.U @ (F.T @ H0F+E_inverse) @ self.U.T
        symmetry_error = np.linalg.norm(H-H.T)/np.linalg.norm(H)
        H = (H+H.T)*.5
        chol = linalg.cholesky(H, lower=True, check_finite=False)
        probes = np.random.default_rng(625).standard_normal((n, 3))
        direct = np.column_stack([self.apply(v) for v in probes.T])
        application_error = np.linalg.norm(direct-H @ probes)/np.linalg.norm(direct)
        interpolation_error = np.linalg.norm(np.column_stack([self.apply(v) for v in self.AU.T])-self.U)/np.linalg.norm(self.U)
        assert application_error < 1e-10
        assert interpolation_error < 1e-9
        return dict(cholesky_spd=True, minimum_cholesky_diagonal=float(np.diag(chol).min()),
                    symmetry_relative_error=float(symmetry_error),
                    application_vs_dense_relative_error=float(application_error),
                    HA_U_equals_U_relative_error=float(interpolation_error))


def pcg(A, b, inverse, *, x0=None, capture=False, maxiter=2000, rtol=1e-8):
    """Independent conventional PCG with true-residual stopping checks."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    norm_b = np.linalg.norm(b)
    r = b.copy() if x0 is None else b-A @ x
    z = inverse.apply(r)
    p, rho = z.copy(), float(r @ z)
    saved = dict(directions=[], images=[])
    replacements, matvecs = 0, int(x0 is not None)
    converged, breakdown = False, None
    for iteration in range(1, maxiter+1):
        ap = A @ p
        matvecs += 1
        curvature = float(p @ ap)
        if rho <= 0 or curvature <= 0 or not np.isfinite(rho+curvature):
            breakdown = "nonpositive_or_nonfinite_inner_product"
            break
        if capture and iteration <= 20:
            saved["directions"].append(p.copy())
            saved["images"].append(ap.copy())
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        suspected = np.linalg.norm(r) <= rtol*norm_b
        if iteration % 20 == 0 or suspected or iteration == maxiter:
            true_r = b-A @ x
            matvecs += 1
            if capture and iteration == 20:
                saved.update(x=x.copy(), true_r=true_r.copy())
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
                residual_replacements=replacements, operator_applications=matvecs,
                breakdown=breakdown), saved


def main(root, output):
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic: {output}")
    resource.setrlimit(resource.RLIMIT_CPU, (55, 56))
    started, cpu_started = perf_counter(), process_time()
    result = dict(status="running", scope="One-seed offline accuracy diagnostic; no runtime benefit claimed",
        protocol=dict(operator="dense A=K+lambda I, identical across all ordinary-PCG branches",
                      regularization=.001, ranks=[16, 32], coarse_dimensions=[4, 8],
                      capture_iteration=20, rtol=1e-8, maxiter=2000, blas_threads=1,
                      same_x20_all_branches=True, initial_coarse_correction=False,
                      coarse_selection="largest generalized Ritz values of (A,M_old) in first20 PCG search span",
                      projected_initial_diagonal="actual diag(Q K Q) or diag(B), not ones",
                      schur_kernel="B=K+lambda U U.T-AU (U.T A U)^-1 AU.T", cpu_limit_seconds=55),
        source_sha256={str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                       for path in [root/"src/nystrom_project/numerics.py", Path(__file__).resolve()]}, cases=[])

    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    with threadpool_limits(limits=1):
        for name in ("synthetic-n640-smooth-s0-lambda0.001", "concrete-observed-s0-lambda0.001"):
            prepared = root/"data/prepared"/(name+".npz")
            with np.load(prepared) as data:
                X, y, bandwidth = data["X_train"], data["y_train"], float(data["bandwidth"])
                config = json.loads(str(data["config"]))
            kernel = GaussianKernel(X, bandwidth)
            K, lam = kernel.dense(), .001
            A = K+lam*np.eye(len(y))
            for k in (16, 32):
                if process_time()-cpu_started > 45:
                    result["status"] = "budget_exhausted"
                    save()
                    return
                old = initialize_rpcholesky(kernel, k, lam,
                    np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                baseline, captured = pcg(A, y, old, capture=True)
                assert baseline["converged"] and "x" in captured
                x20 = captured["x"]
                directions = np.column_stack(captured["directions"])
                # Euclidean SVD isolates the observed search span despite loss
                # of global A-conjugacy in ordinary finite-precision PCG.
                directions /= np.linalg.norm(directions, axis=0)[None, :]
                left, singular, _ = linalg.svd(directions, full_matrices=False, check_finite=False)
                retained = singular > 1e-12*singular[0]
                V = left[:, retained]
                AV = A @ V
                Asmall, Msmall = V.T @ AV, V.T @ model_apply(old, V)
                values, vectors = linalg.eigh((Asmall+Asmall.T)*.5, (Msmall+Msmall.T)*.5, check_finite=False)
                restart_old, _ = pcg(A, y, old, x0=x20)
                case = dict(name=name, config=config, n=len(y), rank=k,
                    prepared_sha256=sha256(prepared.read_bytes()).hexdigest(),
                    initial_indices=old.indices.tolist(), ordinary_fixed_from_zero=baseline,
                    ordinary_old_restart=restart_old,
                    search_span_dimension=int(V.shape[1]),
                    search_directions_singular_values=singular.tolist(),
                    first20_direction_and_image_archive_bytes=int(2*len(y)*20*8),
                    coarse_dimensions=[])
                result["cases"].append(case)
                for d in (4, 8):
                    U, _ = linalg.qr(V @ vectors[:, -d:], mode="economic", check_finite=False)
                    AU = A @ U
                    KU = K @ U
                    small_K = U.T @ KU
                    projected_K = K-U @ KU.T-KU @ U.T+U @ small_K @ U.T
                    projected_K = (projected_K+projected_K.T)*.5
                    projected_old = NystromPreconditioner(old.indices, projected_K[:, old.indices], lam)
                    projected_new = projected_rpcholesky(projected_K, k, lam,
                        np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                    E = (U.T @ AU+AU.T @ U)*.5
                    E_factor = linalg.cho_factor(E, lower=True, check_finite=False)
                    schur_B = K+lam*(U @ U.T)-AU @ linalg.cho_solve(E_factor, AU.T, check_finite=False)
                    schur_B = (schur_B+schur_B.T)*.5
                    schur_old = NystromPreconditioner(old.indices, schur_B[:, old.indices], lam)
                    schur_new = projected_rpcholesky(schur_B, k, lam,
                        np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                    minimum_schur_eigenvalue = float(linalg.eigvalsh(schur_B, subset_by_index=[0, 0], check_finite=False)[0])
                    assert minimum_schur_eigenvalue > -1e-10*np.linalg.norm(K, ord="fro")
                    # Independent exact-complement check: balancing the full
                    # Schur base B+lambda I must invert A, unlike QKQ+lambda I.
                    full_factor = linalg.cho_factor(schur_B+lam*np.eye(len(y)), lower=True, check_finite=False)
                    probes = np.random.default_rng(626).standard_normal((len(y), 3))
                    rhs = A @ probes
                    coarse = linalg.cho_solve(E_factor, U.T @ rhs, check_finite=False)
                    z = linalg.cho_solve(full_factor, rhs-AU @ coarse, check_finite=False)
                    exact = z+U @ (coarse-linalg.cho_solve(E_factor, AU.T @ z, check_finite=False))
                    exact_error = float(np.linalg.norm(exact-probes)/np.linalg.norm(probes))
                    assert exact_error < 1e-7
                    group = dict(d=d, largest_generalized_ritz_values=values[-d:].tolist(),
                        U_orthogonality_error=float(np.max(np.abs(U.T @ U-np.eye(d)))),
                        projected_kernel_U_relative_error=float(np.linalg.norm(projected_K @ U)/np.linalg.norm(projected_K)),
                        trace_K=float(np.trace(K)), trace_projected_K=float(np.trace(projected_K)),
                        minimum_projected_diagonal=float(np.diag(projected_K).min()),
                        schur_kernel_U_relative_error=float(np.linalg.norm(schur_B @ U)/np.linalg.norm(schur_B)),
                        minimum_schur_eigenvalue=minimum_schur_eigenvalue,
                        trace_schur_B=float(np.trace(schur_B)),
                        full_schur_balanced_inverse_relative_error=exact_error,
                        extra_U_AU_storage_bytes=int(U.nbytes+AU.nbytes),
                        extra_coarse_factor_storage_bytes=int(d*d*8), branches={})
                    for label, state in (("old_balanced", old), ("projected_old_indices_balanced", projected_old),
                                         ("projected_new_rp_indices_balanced", projected_new),
                                         ("schur_old_indices_balanced", schur_old),
                                         ("schur_new_rp_indices_balanced", schur_new)):
                        inverse = BalancedInverse(state, U, AU)
                        checks = inverse.dense_check()
                        solve, _ = pcg(A, y, inverse, x0=x20)
                        assert solve["converged"], (name, k, d, label, solve)
                        group["branches"][label] = dict(**solve, total_iterations_including_prefix=20+solve["iterations"],
                            indices=state.indices.tolist(), retained_old_indices=int(np.intersect1d(old.indices, state.indices).size),
                            effective_rank=int(state.rank), preconditioner_storage_bytes=int(inverse.storage_bytes),
                            base_preconditioner_storage_bytes=int(state.storage_bytes),
                            projected_trace_approximation=float(state.eigenvalues.sum()), checks=checks)
                    case["coarse_dimensions"].append(group)
                    save()
                    print(json.dumps(dict(case=name, k=k, d=d, old_from_zero=baseline["iterations"],
                        old_restart_total=20+restart_old["iterations"],
                        balanced_totals={key:value["total_iterations_including_prefix"] for key,value in group["branches"].items()})), flush=True)
        result["status"] = "complete"
        save()
        print(json.dumps(dict(status=result["status"], cpu_seconds=result["elapsed_cpu_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), (args.output or args.root/"results/projected-balancing-diagnostic.json").resolve())
