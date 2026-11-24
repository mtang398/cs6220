"""Bounded dense diagnostic of secant-fitted PSD pair residual corrections.

This is an offline mechanism test, not a production timing benchmark. All
branches start at the same 12-step fixed-PCG iterate. No dense inverse is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/paired-residual-fit-diagnostic.json"
PREFIX, TRAIN, NEIGHBORS = 12, 8, 8
CPU_START, WALL_START = time.process_time(), time.perf_counter()
CPU_BUDGET = 54.0
record = dict(
    purpose="Dense offline mechanism test; no kernel-matvec runtime claim",
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    prefix=PREFIX, training_secants=TRAIN, validation_secants=PREFIX-TRAIN,
    candidate_neighbors=NEIGHBORS, validation_gain_margin=0.01,
    status="running", cases=[],
)


def save():
    record["cpu_seconds"] = time.process_time()-CPU_START
    record["wall_seconds"] = time.perf_counter()-WALL_START
    record["offline_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    OUTPUT.write_text(json.dumps(record, indent=2)+"\n")


def budget():
    if time.process_time()-CPU_START >= CPU_BUDGET:
        raise TimeoutError("54 CPU-second diagnostic budget reached")


def pcg(K, lam, b, preconditioner, x0=None, capture=False):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    products = 0
    if x0 is None:
        r = b.copy()
    else:
        r = b-K@x-lam*x
        products += 1
    target = 1e-8*np.linalg.norm(b)
    p = None
    previous_rho = None
    secants, images, checkpoint = [], [], None
    replacements = 0
    start = time.perf_counter()
    for iteration in range(1, 2501):
        if iteration % 16 == 1:
            budget()
        z = preconditioner.apply(r)
        rho = float(r@z)
        p = z.copy() if p is None else z+(rho/previous_rho)*p
        ap = K@p+lam*p
        products += 1
        curvature = float(p@ap)
        if curvature <= 0 or rho <= 0:
            raise ArithmeticError("PCG lost positive curvature")
        if capture and iteration <= PREFIX:
            scale = np.sqrt(curvature)
            secants.append(p.copy()/scale)
            images.append(ap.copy()/scale)
        x += (rho/curvature)*p
        r -= (rho/curvature)*ap
        if capture and iteration == PREFIX:
            checkpoint = x.copy()
        previous_rho = rho
        if np.linalg.norm(r) <= target:
            rtrue = b-K@x-lam*x
            products += 1
            if np.linalg.norm(rtrue) <= target:
                break
            r = rtrue
            p, previous_rho = None, None
            replacements += 1
    else:
        rtrue = b-K@x-lam*x
        products += 1
    result = dict(
        iterations=iteration, relative_residual=float(np.linalg.norm(rtrue)/np.linalg.norm(b)),
        converged=bool(np.linalg.norm(rtrue) <= target), operator_products=products,
        residual_restarts=replacements, dense_solve_seconds=time.perf_counter()-start,
    )
    if capture:
        return result, checkpoint, np.column_stack(secants), np.column_stack(images)
    return result


def action(model, X):
    d, left, right, off = model
    out = d[:, None]*X if X.ndim == 2 else d*X
    if left.size:
        if X.ndim == 2:
            out[left] += off[:, None]*X[right]
            out[right] += off[:, None]*X[left]
        else:
            out[left] += off*X[right]
            out[right] += off*X[left]
    return out


class BlockInverse:
    def __init__(self, F, lam, model):
        d, self.left, self.right, off = model
        self.tdiag = 1/np.sqrt(lam+d)
        self.toff = np.empty_like(off)
        mineig = float(np.min(lam+d))
        for q, (i, j) in enumerate(zip(self.left, self.right)):
            values, vectors = linalg.eigh(np.array([[lam+d[i], off[q]],
                                                  [off[q], lam+d[j]]]))
            mineig = min(mineig, float(values[0]))
            if values[0] <= 0:
                raise ArithmeticError("Block base is not SPD")
            T = (vectors/np.sqrt(values))@vectors.T
            self.tdiag[i], self.tdiag[j], self.toff[q] = T[0, 0], T[1, 1], T[0, 1]
        self.spd_lower_bound = mineig
        self.basis, singular, _ = linalg.svd(self.whiten(F), full_matrices=False,
                                            check_finite=False)
        self.values = singular*singular

    def whiten(self, r):
        return action((self.tdiag, self.left, self.right, self.toff), r)

    def apply(self, r):
        q = self.whiten(r)
        c = self.basis.T@q
        perpendicular = q-self.basis@c
        correction = self.basis.T@perpendicular
        perpendicular -= self.basis@correction
        c += correction
        return self.whiten(perpendicular+self.basis@(c/(1+self.values)))


def models(X, K, F, P, Y):
    n = len(X)
    e = np.maximum(0.0, 1-np.sum(F*F, axis=1))
    c = np.sum(P*Y, axis=1)
    g = np.sum(P*P, axis=1)
    d = np.clip(np.divide(c, g, out=np.zeros_like(c), where=g>0), 0, e)
    gain = 2*d*c-d*d*g
    standardized = (X-X.mean(axis=0))/np.maximum(X.std(axis=0), 1e-14)
    distances, neighbors = cKDTree(standardized).query(standardized,
                                                     k=min(NEIGHBORS+1, n))
    edges = sorted({(min(i, int(j)), max(i, int(j))) for i in range(n)
                    for j in neighbors[i, 1:] if i != j})
    scored = []
    max_gain_error = 0.0
    for i, j in edges:
        pb, yb = P[[i, j]], Y[[i, j]]
        _, vectors = linalg.eigh(yb@yb.T, check_finite=False)
        v = vectors[:, -1]
        t = v@pb
        gc, gg = float(t@(v@yb)), float(t@t)
        nonzero = v*v > 1e-28
        cap = float(np.min(e[[i, j]][nonzero]/(v*v)[nonzero]))
        amplitude = float(np.clip(gc/gg, 0, cap)) if gg>1e-30 else 0.0
        db = amplitude*np.outer(v, v)
        pair_gain = 2*amplitude*gc-amplitude*amplitude*gg
        direct = np.sum(yb*yb)-np.sum((yb-db@pb)**2)
        max_gain_error = max(max_gain_error, abs(pair_gain-direct))
        benefit = pair_gain-gain[i]-gain[j]
        if benefit > 1e-14*max(float(np.sum(yb*yb)), 1e-30):
            scored.append((benefit, i, j, db))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    used = set()
    selected = []
    pair_diag = d.copy()
    pair_off = []
    for benefit, i, j, block in scored:
        if i in used or j in used:
            continue
        used.update((i, j))
        selected.append((i, j))
        pair_diag[i], pair_diag[j] = block[0, 0], block[1, 1]
        pair_off.append(block[0, 1])
    left = np.array([i for i, j in selected], dtype=int)
    right = np.array([j for i, j in selected], dtype=int)
    empty = np.empty(0, dtype=int)
    empty_float = np.empty(0)
    fitted = (pair_diag, left, right, np.array(pair_off))
    static_off = K[left, right]-np.sum(F[left]*F[right], axis=1)
    return {
        "zero_restart": (np.zeros(n), empty, empty, empty_float),
        "learned_diagonal": (d, empty, empty, empty_float),
        "learned_pairs": fitted,
        "static_pitc_same_pairs": (e, left, right, static_off),
    }, dict(candidate_edges=len(edges), selected_pairs=len(selected),
            training_gain_identity_max_abs_error=max_gain_error,
            learned_pair_diagonal_cap_max_excess=float(np.max(pair_diag-e)))


def run():
    for name in ["synthetic-n640-smooth", "concrete-observed", "airfoil-observed"]:
        data = np.load(ROOT/f"data/prepared/{name}-s0-lambda0.001.npz")
        X, b, lam = data["X_train"], data["y_train"], .001
        kernel = GaussianKernel(X, float(data["bandwidth"]))
        K = kernel.dense()
        for rank in [16, 32]:
            budget()
            case = dict(dataset=name, seed=0, rank=rank, regularization=lam,
                        n=len(b), runs=[], status="running")
            record["cases"].append(case)
            base = initialize_rpcholesky(kernel, rank, lam,
                                        np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
            old, checkpoint, P, AP = pcg(K, lam, b, base, capture=True)
            old.update(method="fixed_uninterrupted", total_iterations=old["iterations"])
            case["runs"].append(old)
            if checkpoint is None:
                raise ArithmeticError("Fixed solve ended before the common prefix")
            F = base.basis*np.sqrt(base.eigenvalues)
            Y = AP-lam*P-F@(F.T@P)
            del AP
            fit_start = time.perf_counter()
            candidates, info = models(X, K, F, P[:, :TRAIN], Y[:, :TRAIN])
            case.update(info)
            case["fit_dense_seconds"] = time.perf_counter()-fit_start
            case["construction_secant_bytes"] = int(2*P.nbytes)
            case["base_two_n_by_k_arrays_bytes"] = int(2*len(b)*rank*8)
            losses = {}
            for label, model in candidates.items():
                losses[label] = {}
                for part, indices in [("training", slice(0, TRAIN)),
                                      ("validation", slice(TRAIN, PREFIX))]:
                    numerator = np.linalg.norm(Y[:, indices]-action(model, P[:, indices]))**2
                    denominator = np.linalg.norm(Y[:, indices])**2
                    losses[label][part] = float(numerator/denominator) if denominator else 0.0
            case["secant_relative_squared_losses"] = losses
            case["learned_pair_validation_gate_pass"] = bool(
                losses["learned_pairs"]["validation"] < .99*min(
                    1.0, losses["learned_diagonal"]["validation"]))
            for label, model in candidates.items():
                budget()
                build_start = time.perf_counter()
                preconditioner = base if label == "zero_restart" else BlockInverse(F, lam, model)
                build_seconds = time.perf_counter()-build_start
                trial = np.random.default_rng(91).normal(size=len(b))
                answer = preconditioner.apply(trial)
                applied = lam*answer+action(model, answer)+F@(F.T@answer)
                inverse_error = float(np.linalg.norm(applied-trial)/np.linalg.norm(trial))
                result = pcg(K, lam, b, preconditioner, checkpoint)
                pair_count = len(model[1])
                result.update(method=label, total_iterations=PREFIX+result["iterations"],
                              build_dense_seconds=build_seconds,
                              preconditioner_inverse_action_relative_error=inverse_error,
                              certified_spd_lower_bound=(lam if label == "zero_restart"
                                  else preconditioner.spd_lower_bound),
                              application_extra_float_bytes=(0 if label == "zero_restart"
                                  else int(8*(len(b)+pair_count))),
                              application_extra_index_bytes=int(16*pair_count),
                              candidate_model_bytes=int(sum(a.nbytes for a in model)))
                case["runs"].append(result)
                save()
                print(json.dumps(dict(dataset=name, rank=rank, **result)), flush=True)
            case["status"] = "complete"
            save()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_CPU, (58, 59))
    try:
        with threadpool_limits(limits=1):
            run()
        record["status"] = "complete"
    except TimeoutError as error:
        record["status"] = "cpu_budget_exhausted"
        record["error"] = str(error)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        save()
        raise
    save()
