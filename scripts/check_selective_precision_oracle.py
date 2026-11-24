"""Selective factor precision oracle; construction and selection are uncharged.

The exact dense eigenbasis, decoded factors, selection workspace, and kernel
matrix are deliberately outside the 128-KiB retained-representation allowance.
This is a full-trajectory diagnostic, not a feasible-memory method or CPU claim.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 131072
PROMOTIONS = 8
RTOL = 1e-8


def action(Q, theta, lam, v):
    """Symmetric SPD inverse for arbitrary, possibly nonorthogonal Q."""
    a = Q.T @ v
    w = v - Q @ a
    weights = 1. / (theta + lam)
    term = weights * a if v.ndim == 1 else weights[:, None] * a
    return w / (theta[-1] + lam) + Q @ (term - (Q.T @ w) / (theta[-1] + lam))


def normalized_probes(*vectors):
    return np.column_stack([v / np.linalg.norm(v) for v in vectors])


def discrepancy(Q, reference, theta, lam, probes):
    delta = action(Q, theta, lam, probes) - action(reference, theta, lam, probes)
    return float(np.sum(delta * delta))


def greedy_mask(half, single, reference, theta, lam, probes):
    """Reevaluate the complete operator action after each possible promotion."""
    current = half.copy()
    target = action(reference, theta, lam, probes)
    selected = []
    scores = [float(np.sum((action(current, theta, lam, probes) - target) ** 2))]
    for _ in range(PROMOTIONS):
        best, best_score = None, np.inf
        for j in range(current.shape[1]):
            if j in selected:
                continue
            current[:, j] = single[:, j]
            difference = action(current, theta, lam, probes) - target
            score = float(np.sum(difference * difference))
            current[:, j] = half[:, j]
            if score < best_score:
                best, best_score = j, score
        assert best is not None
        current[:, best] = single[:, best]
        selected.append(best)
        scores.append(best_score)
    return np.asarray(selected, dtype=np.int64), scores


def hybrid(half, single, mask):
    Q = half.copy()
    Q[:, mask] = single[:, mask]
    return Q


def solve(A, b, Q, theta, lam, x0=None):
    operator_products = 0
    verification_products = 0
    iterations = 0
    saved = None
    first = None
    norm_b = np.linalg.norm(b)

    def product(v):
        nonlocal operator_products
        operator_products += 1
        return A @ v

    class Finished(Exception):
        pass

    def callback(x):
        nonlocal iterations, verification_products, saved, first
        iterations += 1
        if iterations == 1:
            first = x.copy()
        verification_products += 1
        if np.linalg.norm(b - A @ x) <= RTOL * norm_b:
            saved = x.copy()
            raise Finished

    inverse = LinearOperator(A.shape, matvec=lambda v: action(Q, theta, lam, v), dtype=np.float64)
    operator = LinearOperator(A.shape, matvec=product, dtype=np.float64)
    try:
        x, info = cg(operator, b, x0=x0, M=inverse, rtol=0., atol=0.,
                     maxiter=2500, callback=callback)
    except Finished:
        x, info = saved, 0
    verification_products += 1
    residual = float(np.linalg.norm(b - A @ x) / norm_b)
    return dict(iterations=iterations, solver_info=int(info),
                converged=bool(info == 0 and residual <= RTOL),
                true_relative_residual=residual,
                operator_products=operator_products,
                original_verification_products=verification_products), first


def independent_checks():
    rng = np.random.default_rng(622119)
    q = rng.normal(size=(11, 4)) * .7  # Deliberately nonorthogonal.
    theta, lam = np.array([3., 1.2, .5, .2]), .03
    p = np.eye(11) - q @ q.T
    explicit = p @ p / (theta[-1] + lam) + (q / (theta + lam)) @ q.T
    v = rng.normal(size=(11, 2))
    error = float(np.linalg.norm(action(q, theta, lam, v) - explicit @ v) /
                  np.linalg.norm(explicit @ v))
    smallest = float(np.linalg.eigvalsh(explicit)[0])
    assert error < 1e-12 and smallest > 0.
    q, _ = np.linalg.qr(rng.normal(size=(11, 4)))
    a = q.T @ v
    expected = (v - q @ a) / (theta[-1] + lam) + q @ (a / (theta + lam)[:, None])
    exact_error = float(np.linalg.norm(action(q, theta, lam, v) - expected) /
                        np.linalg.norm(expected))
    assert exact_error < 1e-12
    return dict(nonorthogonal_action_relative_error=error,
                nonorthogonal_minimum_eigenvalue=smallest,
                orthogonal_spectral_action_relative_error=exact_error)


def main():
    output = ROOT / 'results/selective-precision-oracle.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (112, 118))
    start, cpu = perf_counter(), process_time()
    baseline_path = ROOT / 'results/quantized-spectral-headroom.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['status'] == 'complete'
    record = dict(status='running',
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_source=dict(path=str(baseline_path.relative_to(ROOT)),
                             sha256=sha256(baseline_path.read_bytes()).hexdigest()),
        protocol=dict(seed=0, datasets=['smooth', 'mixed', 'Concrete'], shifts=[.001, 1e-5],
            rank_by_train_size={'512': 119, '824': 71}, promotions=PROMOTIONS, checkpoint=1,
            factor_byte_allowance=BUDGET, stopping='First literal original residual<=1e-8',
            inverse='(I-QQ^T)^2/(theta_min+lambda)+Q diag(1/(theta+lambda))Q^T',
            selector='Eight greedy promotions; sum squared exact H-action discrepancy on unit-2-norm probes, recomputed after every promotion',
            static_selector='Descending ||q32_j-q16_j||_2/(theta_j+lambda); heuristic, not a reproduced prior algorithm',
            representation='FP16 unpromoted columns, FP32 promoted columns, FP64 eigenvalues, eight int64 promotion indices',
            excluded='Exact dense eigendecomposition, dense A, all decoded factors, construction, promotion regeneration, selection and application workspace/cost',
            claim='Oracle trajectory comparison only; neither total-memory feasibility nor native CPU performance',
            lock='/tmp/cs6220-project-benchmark.lock', blas_threads=1, cpu_cap_seconds=112),
        independent_checks=independent_checks(), selections=[], runs=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,
                      elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record, indent=2) + '\n')

    save()
    with threadpool_limits(limits=1):
        for name in record['protocol']['datasets']:
            stem = 'concrete-observed' if name == 'Concrete' else f'synthetic-n640-{name}'
            path = ROOT / f'data/prepared/{stem}-s0-lambda0.001.npz'
            digest = sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
            n = len(b)
            rank = int(record['protocol']['rank_by_train_size'][str(n)])
            K = np.exp(-cdist(X, X, 'sqeuclidean') / (2*bw*bw))
            values, vectors = eigh(K)
            theta = np.maximum(values[::-1][:rank], 0.)
            reference = vectors[:, ::-1][:, :rank]
            half = reference.astype(np.float16).astype(float)
            single = reference.astype(np.float32).astype(float)
            retained = 2*n*(rank-PROMOTIONS) + 4*n*PROMOTIONS + 8*rank + 8*PROMOTIONS
            assert retained <= BUDGET
            assert 2*n*(rank+1-PROMOTIONS)+4*n*PROMOTIONS+8*(rank+1)+8*PROMOTIONS > BUDGET
            for lam in record['protocol']['shifts']:
                A = K + lam*np.eye(n)
                z0 = action(half, theta, lam, b)
                az0 = A @ z0
                alpha0 = float(b @ z0 / (z0 @ az0))
                assert alpha0 > 0.
                x1 = alpha0*z0
                r1 = b - A @ x1
                probes0, probes01 = normalized_probes(b), normalized_probes(b, r1)
                delta_norms = np.linalg.norm(single-half, axis=0)
                weights = delta_norms / (theta + lam)
                static_mask = np.argsort(-weights, kind='stable')[:PROMOTIONS]
                mask0, scores0 = greedy_mask(half, single, reference, theta, lam, probes0)
                mask01, scores01 = greedy_mask(half, single, reference, theta, lam, probes01)
                for mask in (static_mask, mask0, mask01):
                    assert len(mask) == PROMOTIONS and len(np.unique(mask)) == PROMOTIONS
                record['selections'].append(dict(dataset=name, regularization=lam,
                    rank=rank, prefix_true_relative_residual=float(np.linalg.norm(r1)/np.linalg.norm(b)),
                    prefix_x_sha256=sha256(x1.tobytes()).hexdigest(),
                    prefix_r_sha256=sha256(r1.tobytes()).hexdigest(),
                    static_mask=static_mask.tolist(), r0_mask=mask0.tolist(), r01_mask=mask01.tolist(),
                    r0_scores=scores0, r01_scores=scores01,
                    r0_r01_mask_intersection=len(set(mask0)&set(mask01))))
                cases = [
                    ('half_uninterrupted', half, None, None),
                    ('half_restart1', half, None, x1),
                    ('static_hybrid_upfront', hybrid(half, single, static_mask), static_mask, None),
                    ('r0_hybrid_upfront', hybrid(half, single, mask0), mask0, None),
                    ('r0_hybrid_delayed1', hybrid(half, single, mask0), mask0, x1),
                    ('r01_hybrid_delayed1', hybrid(half, single, mask01), mask01, x1),
                    ('fp64_same_rank_oracle', reference, None, None),
                ]
                for method, Q, mask, initial in cases:
                    if process_time()-cpu > 108:
                        record['status'] = 'cpu_budget'; save(); return
                    stats, first = solve(A, b, Q, theta, lam, initial)
                    prefix = 1 if initial is not None else 0
                    stats['continuation_iterations'] = stats['iterations']
                    stats['iterations'] += prefix
                    stats['operator_products'] += prefix
                    stats['original_verification_products'] += prefix
                    if method == 'half_uninterrupted':
                        prefix_error = float(np.linalg.norm(first-x1)/max(np.linalg.norm(x1), np.finfo(float).tiny))
                        assert prefix_error < 1e-13
                    else:
                        prefix_error = None
                    action_errors = []
                    for column in range(2):
                        probe = probes01[:, column]
                        target = action(reference, theta, lam, probe)
                        action_errors.append(float(np.linalg.norm(action(Q, theta, lam, probe)-target)/np.linalg.norm(target)))
                    factor_bytes = retained if mask is not None else n*rank*(8 if method=='fp64_same_rank_oracle' else 2)+8*rank
                    row = dict(dataset=name, regularization=lam, method=method, n=n, rank=rank,
                        data_sha256=digest, retained_bytes=factor_bytes,
                        retained_within_allowance=factor_bytes<=BUDGET,
                        promoted_indices=[] if mask is None else mask.tolist(), prefix_iterations=prefix,
                        shared_prefix_error=prefix_error,
                        checkpoint_x_sha256=None if initial is None else sha256(initial.tobytes()).hexdigest(),
                        action_relative_errors_r0_r1=action_errors,
                        action_discrepancy_r0_r1=discrepancy(Q, reference, theta, lam, probes01),
                        complement_scale=float(theta[-1]+lam), **stats)
                    record['runs'].append(row); save()
                controls = [r for r in baseline['runs'] if r['dataset']==name and
                            r['regularization']==lam and r['precision']=='float32' and
                            r['role']=='matched_factor_bytes']
                assert len(controls)==1 and controls[0]['data_sha256']==digest
                record.setdefault('reused_fp32_controls', []).append(controls[0])
                save()
    record['status'] = 'complete'
    save()
    print(json.dumps({key:record[key] for key in ('status','elapsed_wall_seconds','elapsed_cpu_seconds')}))


if __name__ == '__main__':
    main()
