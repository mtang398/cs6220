"""Conditional inducing directional derivatives: one frozen-update capacity gate.

Dense construction is diagnostic assistance, not a native cost/memory claim.
The original Gaussian system and literal 1e-8 stopping remain unchanged.
"""
import json
import resource
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nystrom_project.preconditioners import LowRankPreconditioner
from check_cached_kernel_static_frontier import fps


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def kernel(X, Z, h):
    return np.exp(-cdist(X, Z, 'sqeuclidean') / (2 * h * h))


def joint_features(X, Z, U, h):
    """Inducing values followed by h times their directional derivatives."""
    C = kernel(X, Z, h)
    W = kernel(Z, Z, h)
    zu = np.sum(Z * U, axis=1)
    Cd = C * ((X @ U.T - zu[None, :]) / h)
    right = (Z @ U.T - zu[None, :]) / h
    left = (zu[:, None] - U @ Z.T) / h
    Wpd = W * right
    Wdd = W * (U @ U.T - left * right)
    return np.column_stack((C, Cd)), np.block([[W, Wpd], [Wpd.T, Wdd]])


def choose_directions(X, Z, C, W, h, probes):
    """Independent centerwise conditional-covariance choices, not joint optimality."""
    w, V = eigh(W, check_finite=False)
    keep = w > 32 * np.finfo(float).eps * len(w) * w[-1]
    Winv = (V[:, keep] / w[keep]) @ V[:, keep].T
    directions = {name: [] for name in list(probes) + ['matrix']}
    diagnostics = []
    for j, z in enumerate(Z):
        D = C[:, j, None] * ((X - z) / h)
        B = W[:, j, None] * ((Z - z) / h)
        WB = Winv @ B
        E = D - C @ WB
        T = np.eye(X.shape[1]) - B.T @ WB
        T = (T + T.T) * .5
        tw, tv = eigh(T, check_finite=False)
        cutoff = 64 * np.finfo(float).eps * X.shape[1] * max(tw[-1], 1.)
        assert tw[0] > -cutoff, (j, tw[0], cutoff)
        active = tw > cutoff
        assert np.any(active)
        R = tv[:, active] / np.sqrt(tw[active])
        ER = E @ R
        _, ev = eigh(ER.T @ ER, check_finite=False)
        matrix_u = R @ ev[:, -1]
        matrix_u /= np.linalg.norm(matrix_u)
        directions['matrix'].append(matrix_u)
        row = dict(center=j, conditional_min_eigenvalue=float(tw[0]),
                   conditional_max_eigenvalue=float(tw[-1]), conditional_cutoff=float(cutoff),
                   retained_dimension=int(np.sum(active)),
                   conditional_condition=float(tw[-1] / tw[active][0]),
                   choices={})
        for name, v in probes.items():
            g = E.T @ v
            u = R @ (R.T @ g)
            magnitude = float(np.linalg.norm(u))
            fallback = bool(magnitude <= np.finfo(float).tiny)
            u = matrix_u.copy() if fallback else u / magnitude
            directions[name].append(u)
            null_leakage = float(np.linalg.norm(g-tv[:, active]@(tv[:, active].T@g)) / max(np.linalg.norm(g), 1e-300))
            optimum = float(np.linalg.norm(R.T @ g)**2)
            attained = float((g @ u)**2 / (u @ T @ u))
            row['choices'][name] = dict(fallback=fallback,
                conditional_correlation_squared=attained, optimal_correlation_squared=optimum,
                relative_nullspace_leakage=null_leakage,
                projected_gradient_norm=float(np.linalg.norm(R.T @ g)))
        diagnostics.append(row)
    return {name: np.asarray(value) for name, value in directions.items()}, diagnostics


def solve(K, b, H, lam, x0=None, prefix=False):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    normb = np.linalg.norm(b)
    products = checks = 0
    start = process_time()
    def mv(v):
        nonlocal products
        products += 1
        return K @ v + lam * v
    r = b.copy() if x0 is None else b - mv(x)
    z = H.apply(r)
    p = z.copy()
    rho = float(r @ z)
    status = 'maxiter'
    replacements = 0
    maxiter = 8 if prefix else 2000
    for iteration in range(maxiter + 1):
        true = b - mv(x)
        checks += 1
        relative = float(np.linalg.norm(true) / normb)
        if relative <= 1e-8:
            status = 'converged'
            break
        if prefix and iteration == 8:
            status = 'prefix'
            break
        if process_time() - start > 2.0:
            status = 'cpu_cap'
            break
        if iteration == maxiter:
            break
        if np.linalg.norm(r) <= 1e-8 * normb:
            r = true
            z = H.apply(r)
            p = z.copy()
            rho = float(r @ z)
            replacements += 1
        Ap = mv(p)
        den = float(p @ Ap)
        if den <= 0 or rho <= 0 or not np.isfinite(den + rho):
            status = 'invalid_curvature'
            break
        alpha = rho / den
        x += alpha * p
        r -= alpha * Ap
        z = H.apply(r)
        newrho = float(r @ z)
        p = z + (newrho / rho) * p
        rho = newrho
    return x, dict(status=status, converged=status == 'converged', iterations=iteration,
        original_relative_residual=relative, operator_products=products,
        verification_products=checks, iteration_products=iteration,
        boundary_products=int(x0 is not None), residual_replacements=replacements,
        diagnostic_continuation_cpu_seconds=process_time() - start)


def tiny():
    rng = np.random.default_rng(622061)
    X = rng.normal(size=(19, 3))
    Z = X[:4].copy()
    U = rng.normal(size=Z.shape)
    U /= np.linalg.norm(U, axis=1)[:, None]
    h = 1.7
    C, W = joint_features(X, Z, U, h)
    eps = 1e-4
    fd = (kernel(X, Z + eps * h * U, h) - kernel(X, Z - eps * h * U, h)) / (2 * eps)
    fdd = (kernel(Z + eps*h*U, Z + eps*h*U, h)
        - kernel(Z + eps*h*U, Z - eps*h*U, h)
        - kernel(Z - eps*h*U, Z + eps*h*U, h)
        + kernel(Z - eps*h*U, Z - eps*h*U, h)) / (4*eps*eps)
    first_error = float(np.linalg.norm(C[:, 4:] - fd) / np.linalg.norm(C[:, 4:]))
    mixed_error = float(np.linalg.norm(W[4:, 4:] - fdd) / np.linalg.norm(W[4:, 4:]))
    H = LowRankPreconditioner.from_sketch(C, W, .03)
    F = H.basis * np.sqrt(H.eigenvalues)
    residual_min = float(eigh(kernel(X, X, h) - F @ F.T, eigvals_only=True)[0])
    r = rng.normal(size=len(X))
    z = H.apply(r)
    inverse_error = float(np.linalg.norm(F @ (F.T @ z) + .03*z - r) / np.linalg.norm(r))
    assert first_error < 1e-7 and mixed_error < 1e-6 and residual_min > -1e-10 and inverse_error < 1e-10
    directions, _ = choose_directions(X, Z, C[:, :4], W[:4, :4], h, {'current': r})
    assert all(np.allclose(np.linalg.norm(u, axis=1), 1) for u in directions.values())
    return dict(first_derivative_relative_error=first_error, mixed_derivative_relative_error=mixed_error,
                approximation_residual_minimum_eigenvalue=residual_min, inverse_relative_error=inverse_error)


def main():
    out = ROOT / 'results/directional-derivative-preconditioning-v2.json'
    assert not out.exists()
    resource.setrlimit(resource.RLIMIT_CPU, (110, 115))
    start, cpu = perf_counter(), process_time()
    sources = [Path(__file__).resolve(), ROOT/'scripts/check_cached_kernel_static_frontier.py',
               ROOT/'src/nystrom_project/preconditioners.py', ROOT/'src/nystrom_project/numerics.py']
    record = dict(status='running', sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        protocol=dict(initial_points=64, derivative_directions=64, prefix_steps=8, regularization=1e-5,
            choice='Condition each normalized h*derivative family on the 64 initial point values. E=D-C Wpp^dagger B, T=I-B^T Wpp^dagger B. For probe v choose u proportional to T^dagger E^T v. Matrix control maximizes ||Eu||^2/(u^T T u). Choices at all centers condition only on the original point set, not on each other.',
            probes='current literal r8; initial b; independent seeded Gaussian; matrix-only conditional derivative energy. Combine point and derivative features using the FULL joint covariance, including offdiagonal cross-center derivative covariance.',
            comparison='Common x8 restart for all derivative controls and point64/128/512. Initial-b and matrix derivatives also upfront; current-final-set-upfront is explicitly hindsight. Static point128/512 also upfront.',
            cost='Dense cached K, derivatives, selection, eigendecomposition and factors are free diagnostic assistance. Every iterate is checked against the original system. All products reported, including separate verification/boundary counts. Prefix cost recorded separately. NOT complete-cost, fixed-memory or native-runtime evidence. No resource contract adopted.',
            decision='If current directional selection does not improve subsequent convergence relative to initial/random/matrix/point controls enough to justify pricing, close this selector without rank/checkpoint tuning. No full-scope success or heldout use.'), cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start, elapsed_cpu_seconds=process_time()-cpu)
        out.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['tiny'] = tiny()
        save()
        cases = [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),
                 ('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),
                 ('Concrete','concrete-observed-s0-lambda1e-05.npz'),
                 ('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
        for name, filename in cases:
            path = ROOT/'data/prepared'/filename
            with np.load(path) as data:
                X, b, h = data['X_train'], data['y_train'], float(data['bandwidth'])
            X.setflags(write=False)
            b.setflags(write=False)
            input_hashes = [sha256(a.tobytes()).hexdigest() for a in (X, b)]
            K = kernel(X, X, h)
            indices = fps(K, 512)
            Z = X[indices[:64]]
            C = K[:, indices[:64]]
            W = K[np.ix_(indices[:64], indices[:64])]
            H = LowRankPreconditioner.from_sketch(C, W, 1e-5)
            x8, prefix = solve(K, b, H, 1e-5, prefix=True)
            case = dict(dataset=name, n=len(b), data_file=str(path.relative_to(ROOT)),
                        data_sha256=digest(path), input_hashes=input_hashes, prefix=prefix, runs=[])
            record['cases'].append(case)
            save()
            assert prefix['status'] == 'prefix'
            probes = {'current': b-K@x8-1e-5*x8, 'initial': b,
                      'random': np.random.default_rng(622062).normal(size=len(b))}
            directions, diagnostics = choose_directions(X, Z, C, W, h, probes)
            case['direction_diagnostics'] = diagnostics
            matrices = {}
            case['factor_checks'] = {}
            for mode, U in directions.items():
                Cj, Wj = joint_features(X, Z, U, h)
                M = LowRankPreconditioner.from_sketch(Cj, Wj, 1e-5)
                matrices[mode] = M
                q = M.apply(probes['random'])
                F = M.basis * np.sqrt(M.eigenvalues)
                defect = float(np.linalg.norm(F@(F.T@q)+1e-5*q-probes['random']) / np.linalg.norm(probes['random']))
                eigenvalues = eigh(Wj, eigvals_only=True)
                cutoff = float(32*np.finfo(float).eps*len(eigenvalues)*eigenvalues[-1])
                kept = eigenvalues[eigenvalues > cutoff]
                case['factor_checks'][mode] = dict(effective_rank=M.rank, inverse_relative_error=defect,
                    joint_minimum_eigenvalue=float(eigenvalues[0]), joint_maximum_eigenvalue=float(eigenvalues[-1]),
                    joint_cutoff=cutoff, joint_retained_condition=float(kept[-1]/kept[0]),
                    basis_bytes=M.storage_bytes, direction_metadata_bytes=U.nbytes+Z.nbytes,
                    directions=U.tolist())
                assert defect < 1e-7
            for m in [128, 512]:
                subset = indices[:m]
                matrices[f'point{m}'] = LowRankPreconditioner.from_sketch(K[:, subset], K[np.ix_(subset, subset)], 1e-5)
            matrices['point64'] = H
            branches = [(mode, mode, x8, 8, False) for mode in ['current','initial','random','matrix','point64','point128','point512']]
            branches += [(mode+'_upfront', mode, None, 0, mode=='current') for mode in ['initial','matrix','point128','point512','current']]
            for label, key, x0, pref, hindsight in branches:
                _, row = solve(K, b, matrices[key], 1e-5, x0=x0)
                row.update(method=label, prefix_iterations=pref, prefix_plus_iterations=pref+row['iterations'],
                    prefix_operator_products=prefix['operator_products'] if pref else 0,
                    hindsight=hindsight, effective_rank=matrices[key].rank)
                case['runs'].append(row)
                print(name, label, row['status'], row['iterations'], row['original_relative_residual'], flush=True)
                save()
            case['inputs_unchanged'] = input_hashes == [sha256(a.tobytes()).hexdigest() for a in (X, b)]
            assert case['inputs_unchanged']
            save()
    record['sources_unchanged'] = all(digest(ROOT/p)==h for p,h in record['sources'].items())
    assert record['sources_unchanged']
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
