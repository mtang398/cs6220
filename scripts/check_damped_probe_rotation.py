#!/usr/bin/env python3
"""Dense oracle for bounded relative changes of generalized Nyström factors.

The representation is a rank-k generalized factor, not k actual columns.
The cap is fixed at two; no tuning or production speed claim is performed.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel, NystromPreconditioner, initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner
from nystrom_project.learning import compress_probe


def unique_payload(arrays):
    roots = {}
    for a in arrays:
        if not isinstance(a, np.ndarray):
            continue
        while isinstance(a.base, np.ndarray):
            a = a.base
        roots[id(a)] = a
    return sum(a.nbytes for a in roots.values())


def relative_extrema(alpha, beta, gamma):
    discriminant = (alpha+beta)**2-4*gamma*gamma
    if discriminant < -1e-10*max((alpha+beta)**2, 1.):
        raise ArithmeticError('Invalid relative-update Gram')
    root = np.sqrt(max(discriminant, 0.))
    return min(1., 1+.5*(beta-alpha-root)), max(1., 1+.5*(beta-alpha+root))


def rotate_probe(state, p, Kp, *, damped, audit=False):
    k = state.rank
    F = state.basis*np.sqrt(state.eigenvalues)
    cross = F.T@p
    curvature = float(p@Kp)
    delta = curvature-float(cross@cross)
    event = dict(accepted=False, probe_curvature=curvature, innovation_curvature=delta,
                 damped=damped, additional_kernel_products=0)
    if delta <= 1e-11*max(abs(curvature), float(cross@cross), 1e-300):
        event['reason'] = 'dependent_or_unreliable_probe'
        return state, event, None
    u = (Kp-F@cross)/np.sqrt(delta)
    aug = np.column_stack((F, u))
    Uaug, saug, Vaug = linalg.svd(aug, full_matrices=False, check_finite=False)
    oldtrace = float(state.eigenvalues.sum())
    targettrace = float(saug[:k]@saug[:k])
    if targettrace <= oldtrace*(1+1e-9):
        event['reason'] = 'no_material_trace_increase'
        return state, event, None
    discarded = Vaug[-1].copy()
    if discarded[-1] < 0:
        discarded *= -1
    d = float(np.clip(discarded[-1], 0., 1.))
    sine = float(np.linalg.norm(discarded[:-1]))
    if sine <= 32*np.finfo(float).eps:
        event['reason'] = 'zero_target_rotation'
        return state, event, None
    w = discarded[:-1]/sine
    target_angle = float(np.arctan2(sine, d))
    a = F@w
    Ha, Hu = state.apply(a), state.apply(u)
    alpha, eta, zeta = float(a@Ha), float(u@Hu), float(a@Hu)
    angle = target_angle
    halvings = 0
    while True:
        c, s = np.cos(angle), np.sin(angle)
        beta = c*c*alpha+s*s*eta-2*c*s*zeta
        gamma = c*alpha-s*zeta
        lo, hi = relative_extrema(alpha, beta, gamma)
        distortion = max(hi, 1/lo) if lo > 0 else float('inf')
        if not damped or distortion <= 2.:
            break
        angle *= .5
        halvings += 1
        if halvings > 60:
            event['reason'] = 'backtracking_failed'
            return state, event, None
    b = c*a-s*u
    Fnew = F+np.outer((c-1)*a-s*u, w)
    Unew, snew, Vnew = linalg.svd(Fnew, full_matrices=False, check_finite=False)
    new = LowRankPreconditioner(Unew, snew*snew, state.regularization)
    major_payload = unique_payload([state.basis, state.eigenvalues, state.indices, F, cross, u,
        aug, Uaug, saug, Vaug, discarded, w, a, Ha, Hu, b, Fnew, Unew, snew, Vnew,
        new.basis, new.eigenvalues, new.indices])
    event.update(accepted=True, target_angle=target_angle, accepted_angle=angle,
        angle_fraction=float(angle/target_angle), halvings=halvings,
        relative_eigenvalue_min=float(lo), relative_eigenvalue_max=float(hi),
        multiplicative_distortion=float(distortion), trace_before=oldtrace,
        target_trace=targettrace, trace_after=float(new.eigenvalues.sum()),
        retained_state_bytes=new.storage_bytes, peak_major_update_array_payload_bytes=major_payload)
    transform = None
    if audit:
        Q = np.vstack((np.eye(k)+(c-1)*np.outer(w, w), -s*w[None, :]))
        transform = Q@Vnew.T
        event['factor_identity_relative_error'] = float(np.linalg.norm(Fnew-aug@Q)/np.linalg.norm(Fnew))
        event['rotation_orthogonality_error'] = float(np.linalg.norm(Q.T@Q-np.eye(k), ord=2))
    return new, event, transform


def independent_checks():
    rng = np.random.default_rng(622018)
    G = rng.normal(size=(24, 24))
    K = G@G.T+.2*np.eye(24)
    indices = np.arange(5)
    base = NystromPreconditioner(indices, K[:, indices], .07)
    state = LowRankPreconditioner.from_nystrom(base)
    coefficients = linalg.solve(base.core, (base.columns.T@base.basis)/np.sqrt(base.eigenvalues)[None, :], assume_a='pos')
    Omega = np.zeros((24, 5))
    Omega[indices] = coefficients
    records = []
    for step in range(10):
        p = rng.normal(size=24)
        F = state.basis*np.sqrt(state.eigenvalues)
        cross = F.T@p
        delta = float(p@(K@p)-cross@cross)
        old_dense = .07*np.eye(24)+F@F.T
        new, event, transform = rotate_probe(state, p, K@p, damped=True, audit=True)
        assert event['accepted']
        Omega = np.column_stack((Omega, (p-Omega@cross)/np.sqrt(delta)))@transform
        Fnew = new.basis*np.sqrt(new.eigenvalues)
        new_dense = .07*np.eye(24)+Fnew@Fnew.T
        ev = linalg.eigvalsh(new_dense, old_dense)
        event['dense_relative_bound_error'] = float(max(abs(ev[0]-event['relative_eigenvalue_min']), abs(ev[-1]-event['relative_eigenvalue_max'])))
        event['implicit_factor_relative_error'] = float(np.linalg.norm(Fnew-K@Omega)/np.linalg.norm(Fnew))
        event['implicit_K_orthogonality_error'] = float(np.linalg.norm(Omega.T@K@Omega-np.eye(5), ord=2))
        event['minimum_kernel_remainder_eigenvalue'] = float(linalg.eigvalsh(K-Fnew@Fnew.T)[0])
        assert ev[0] >= .5-1e-11 and ev[-1] <= 2+1e-11
        assert event['dense_relative_bound_error'] < 1e-10
        assert event['implicit_K_orthogonality_error'] < 1e-10
        assert event['minimum_kernel_remainder_eigenvalue'] > -1e-10
        records.append(event)
        if step == 0:
            undamped, _, _ = rotate_probe(state, p, K@p, damped=False)
            reference, ref_event = compress_probe(state, p, K@p)
            assert ref_event['accepted']
            F1 = undamped.basis*np.sqrt(undamped.eigenvalues)
            F2 = reference.basis*np.sqrt(reference.eigenvalues)
            undamped_error = float(np.linalg.norm(F1@F1.T-F2@F2.T)/np.linalg.norm(F2@F2.T))
            assert undamped_error < 1e-11
        state = new
    return dict(updates=10, undamped_vs_existing_compress_probe_relative_error=undamped_error,
        maximum_factor_identity_error=max(r['factor_identity_relative_error'] for r in records),
        maximum_rotation_orthogonality_error=max(r['rotation_orthogonality_error'] for r in records),
        maximum_dense_relative_eigenvalue_error=max(r['dense_relative_bound_error'] for r in records),
        maximum_implicit_factor_error=max(r['implicit_factor_relative_error'] for r in records),
        maximum_implicit_K_orthogonality_error=max(r['implicit_K_orthogonality_error'] for r in records),
        minimum_kernel_remainder_eigenvalue=min(r['minimum_kernel_remainder_eigenvalue'] for r in records))


def solve(A, b, state, method, cpu_start):
    adaptive = method.startswith(('undamped', 'damped'))
    damped = method.startswith('damped')
    restart = method.endswith('restart10')
    x, r, last = np.zeros_like(b), b.copy(), None
    norm_b = np.linalg.norm(b)
    events, progress = [], []
    verifications = scheduled_restarts = numerical_restarts = 0
    max_state = state.storage_bytes
    max_update = 0
    true_rel = 1.
    last_verified = -1
    reason = 'max_iterations'
    started = perf_counter()
    for iteration in range(1, 2501):
        if process_time()-cpu_start > 95:
            iteration -= 1
            reason = 'global_cpu_guard'
            break
        p = state.apply(r)
        ap = A@p
        if last is not None:
            old, aold = last
            old_curvature = float(old@aold)
            for _ in range(2):
                beta = float(old@ap)/old_curvature
                p -= beta*old
                ap -= beta*aold
        curvature = float(p@ap)
        numerator = float(r@p)
        if min(curvature, numerator) <= 0:
            reason = 'nonpositive_progress'
            iteration -= 1
            break
        alpha = numerator/curvature
        x += alpha*p
        r -= alpha*ap
        suspected = np.linalg.norm(r)/norm_b <= 1e-8
        verified_r = None
        if iteration % 20 == 0 or suspected or (restart and iteration == 10) or iteration == 2500:
            verified_r = b-A@x
            true_rel = float(np.linalg.norm(verified_r)/norm_b)
            verifications += 1
            last_verified = iteration
            progress.append(dict(iteration=iteration, true_relative_residual=true_rel,
                recursive_residual_gap=float(np.linalg.norm(verified_r-r)/norm_b)))
            if true_rel <= 1e-8:
                reason = 'converged'
                break
        if adaptive and iteration <= 10:
            state, event, _ = rotate_probe(state, p, ap-state.regularization*p, damped=damped)
            event['iteration'] = iteration
            events.append(event)
            max_state = max(max_state, state.storage_bytes)
            max_update = max(max_update, event.get('peak_major_update_array_payload_bytes', 0))
        scale = np.sqrt(curvature)
        last = (p/scale, ap/scale)
        if restart and iteration == 10:
            r = verified_r.copy()
            last = None
            scheduled_restarts += 1
        elif suspected:
            r = verified_r.copy()
            last = None
            numerical_restarts += 1
    if last_verified != iteration:
        true_rel = float(np.linalg.norm(b-A@x)/norm_b)
        verifications += 1
    return dict(method=method, iterations=iteration, converged=bool(true_rel<=1e-8),
        true_relative_residual=true_rel, stop_reason=reason, events=events, progress=progress,
        accepted_updates=sum(e['accepted'] for e in events),
        additional_update_kernel_products=0, direction_matvecs=iteration,
        verification_matvecs=verifications, total_operator_applications=iteration+verifications,
        scheduled_restarts=scheduled_restarts, numerical_restarts=numerical_restarts,
        max_retained_preconditioner_bytes=max_state, retained_history_bytes=2*len(b)*8+8,
        max_major_update_array_payload_bytes=max_update,
        update_workspace_note='Major NumPy allocations include old/new factors and SVD outputs; LAPACK workspaces, dense oracle A, and ordinary solver vectors are additional. No persistent candidate bank or implicit probes retained.',
        final_trace=float(state.eigenvalues.sum()), oracle_dense_solve_seconds=perf_counter()-started)


def main():
    output = ROOT/'results/damped-probe-rotation-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (98, 100))
    started, cpu_start = perf_counter(), process_time()
    methods = ['static_rp', 'static_rp_restart10', 'undamped_freeze10',
        'undamped_freeze10_restart10', 'damped_cap2_freeze10', 'damped_cap2_freeze10_restart10', 'power_gaussian']
    sources = [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py',
               ROOT/'src/nystrom_project/preconditioners.py', ROOT/'src/nystrom_project/learning.py']
    result = dict(status='running', scope='Dense mechanism diagnostic; fixed-rank generalized Nyström factor, not actual-column exchange or production timing',
        protocol=dict(rank=32, regularizations=[.001, 1e-5], seed=0, initial_seed_sequence=[0, 6220, 1],
            gaussian_seed_sequence=[0, 6220, 3], methods=methods, update_steps=list(range(1, 11)),
            update_probe='just-used FCG1 direction p and cached Kp=Ap-lambda*p',
            relative_distortion_cap=2, backtracking='halve angle until max(upper,1/lower)<=2',
            target='best rank-k factor of [Fold,u], same material trace-increase gate as compress_probe',
            recurrence='FCG1 with two-pass A orthogonalization; freeze after10; restart controls explicitly named',
            extra_control='undamped freeze10 with matched restart to separate damping from restart effects',
            rtol=1e-8, max_iterations=2500, true_residual_verification_interval=20,
            blas_threads=1, cpu_limit_seconds=100, dense_operator_allowed_for_offline_only=True),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}, cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_checks'] = independent_checks()
        for name in ('synthetic-n640-smooth', 'synthetic-n640-random', 'concrete-observed', 'airfoil-observed'):
            path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data = np.load(path)
            b = data['y_train']
            kernel = GaussianKernel(data['X_train'], float(data['bandwidth']))
            K = kernel.dense()
            for lam in (.001, 1e-5):
                if process_time()-cpu_start > 88:
                    result['status'] = 'budget_exhausted'
                    save()
                    return
                A = K+lam*np.eye(len(K))
                base = initialize_rpcholesky(kernel, 32, lam, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                initial_U, initial_values = base.basis.copy(), base.eigenvalues.copy()
                initial_indices = base.indices.tolist()
                del base
                omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(len(b), 32))
                Q, _ = linalg.qr(K@omega, mode='economic', check_finite=False)
                Y = K@Q
                power = LowRankPreconditioner.from_sketch(Y, Q.T@Y, lam)
                del omega, Q, Y
                row = dict(dataset=name, n=len(b), regularization=lam,
                    prepared_input=str(path.relative_to(ROOT)), prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                    preparation_regularization=.001, stored_reference_solution_ignored=True,
                    actual_operator_regularization=lam, initial_indices=initial_indices,
                    shared_offline_operator_bytes=A.nbytes, power_setup_full_kernel_matmat_passes=2, runs=[])
                result['cases'].append(row)
                for method in methods:
                    state = (LowRankPreconditioner(power.basis, power.eigenvalues, lam) if method=='power_gaussian'
                             else LowRankPreconditioner(initial_U, initial_values, lam))
                    value = solve(A, b, state, method, cpu_start)
                    row['runs'].append(value)
                    save()
                    print(json.dumps(dict(dataset=name, regularization=lam, method=method,
                        iterations=value['iterations'], converged=value['converged'],
                        residual=value['true_relative_residual'], updates=value['accepted_updates'],
                        cpu_seconds=process_time()-cpu_start)), flush=True)
                    del state
                    if value['stop_reason']=='global_cpu_guard':
                        result['status'] = 'budget_exhausted'
                        save()
                        return
        result['status'] = 'complete'
        save()
        print(json.dumps(dict(status='complete', cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__ == '__main__':
    main()
