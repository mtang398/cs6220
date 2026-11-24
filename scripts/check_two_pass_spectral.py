#!/usr/bin/env python3
"""Predeclared two-pass spectral/LMP oracle with complete replay accounting."""
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
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner


def model(state, X):
    if X.ndim == 1:
        return state.regularization*X+state.basis@(state.eigenvalues*(state.basis.T@X))
    return state.regularization*X+state.basis@(state.eigenvalues[:, None]*(state.basis.T@X))


def inverse_matrix(state, X):
    return np.column_stack([state.apply(v) for v in X.T])


def prefix(A, b, state, m):
    x, r = np.zeros_like(b), b.copy()
    z = state.apply(r)
    rho = float(r@z)
    p = z.copy()
    alpha, beta, rhos, hashes = [], [], [rho], []
    for _ in range(m):
        hashes.append(sha256((z/np.sqrt(rho)).tobytes()).hexdigest())
        ap = A@p
        a = rho/float(p@ap)
        x += a*p
        r -= a*ap
        z = state.apply(r)
        newrho = float(r@z)
        be = newrho/rho
        p = z+be*p
        alpha.append(a)
        beta.append(be)
        rhos.append(newrho)
        rho = newrho
    return dict(x=x, r=r, p=p, rho=rho, z=z, alpha=np.array(alpha), beta=np.array(beta),
                rhos=np.array(rhos), hashes=hashes, A_products=m, inverse_applications=m+1)


def spectral_data(A, b, state, saved):
    m = len(saved['alpha'])
    diagonal = 1/saved['alpha']
    diagonal[1:] += saved['beta'][:-1]/saved['alpha'][:-1]
    off = -np.sqrt(saved['beta'][:-1])/saved['alpha'][:-1]
    values, Y = linalg.eigh_tridiagonal(diagonal, off, check_finite=False)
    selected = []
    duplicate_tolerance = np.sqrt(np.finfo(float).eps)
    for j in range(m-1, -1, -1):
        if all(abs(values[j]-values[i]) > duplicate_tolerance*max(abs(values[j]), abs(values[i])) for i in selected):
            selected.append(j)
        if len(selected) == 4:
            break
    coefficients = Y[:, selected]
    raw = np.zeros((len(b), len(selected)))
    x, r = np.zeros_like(b), b.copy()
    z = state.apply(r)
    p = z.copy()
    matched = []
    scalar_errors = []
    for i in range(m):
        q = z/np.sqrt(saved['rhos'][i])
        matched.append(sha256(q.tobytes()).hexdigest()==saved['hashes'][i])
        raw += q[:, None]*coefficients[i][None, :]
        ap = A@p
        computed_alpha = float(r@z)/float(p@ap)
        scalar_errors.append(abs(computed_alpha-saved['alpha'][i])/abs(saved['alpha'][i]))
        x += saved['alpha'][i]*p
        r -= saved['alpha'][i]*ap
        z = state.apply(r)
        p = z+saved['beta'][i]*p
    replay = dict(all_basis_hashes_identical=all(matched), matched_basis_vectors=sum(matched),
        final_x_bitwise_identical=bool(np.array_equal(x, saved['x'])),
        final_r_bitwise_identical=bool(np.array_equal(r, saved['r'])),
        final_p_bitwise_identical=bool(np.array_equal(p, saved['p'])),
        maximum_recomputed_alpha_relative_error=max(scalar_errors),
        sequential_A_products=m, inverse_applications=m+1,
        accumulation_scalar_multiply_add_pairs=int(len(b)*m*len(selected)))
    assert all(replay[k] for k in ('all_basis_hashes_identical', 'final_x_bitwise_identical',
                                  'final_r_bitwise_identical', 'final_p_bitwise_identical'))
    Mraw = model(state, raw)
    gram = (raw.T@Mraw+Mraw.T@raw)*.5
    ge, gv = linalg.eigh(gram, check_finite=False)
    threshold = 64*np.finfo(float).eps*max(raw.shape)*max(float(ge[-1]), np.finfo(float).tiny)
    keep = ge > threshold
    if not np.any(keep):
        raise ArithmeticError('No physically independent reconstructed mode')
    transform = gv[:, keep]/np.sqrt(ge[keep])[None, :]
    U = raw@transform
    MU = model(state, U)
    AU = A@U
    small = (U.T@AU+AU.T@U)*.5
    theta, rotation = linalg.eigh(small, check_finite=False)
    assert theta[0] > 0
    U, MU, AU = U@rotation, MU@rotation, AU@rotation
    residual = AU-MU*theta[None, :]
    Hresidual = inverse_matrix(state, residual)
    relative_residual = np.linalg.norm(residual, axis=0)/np.linalg.norm(AU, axis=0)
    relative_H_residual = np.sqrt(np.maximum(np.sum(residual*Hresidual, axis=0), 0))/theta
    boundary = saved['r']/np.sqrt(saved['rhos'][-1])
    boundary -= MU@(U.T@boundary)
    boundary_norm2 = float(boundary@boundary)
    weights = (boundary@residual)/boundary_norm2 if boundary_norm2 > 0 else np.zeros(len(theta))
    residual_error = residual-boundary[:, None]*weights[None, :]
    image_error = float(np.linalg.norm(residual_error)/np.linalg.norm(AU))
    relation_error = float(np.linalg.norm(residual_error)/max(np.linalg.norm(residual), np.finfo(float).tiny))
    relation_passes = image_error <= 1e-10 and relation_error <= 1e-8
    h = state.apply(boundary)
    delta = float(boundary@h)
    HAU = inverse_matrix(state, AU)
    physical_gram = U.T@model(state, U)
    raw_relation = A@raw-model(state, raw)*values[selected][None, :]
    raw_boundary = saved['r']/np.sqrt(saved['rhos'][-1])
    raw_weights = -np.sqrt(saved['beta'][-1])/saved['alpha'][-1]*coefficients[-1]
    raw_relation_error = float(np.linalg.norm(raw_relation-np.outer(raw_boundary, raw_weights))/np.linalg.norm(A@raw))
    # Raw relation diagnostics reuse products algebraically in a production
    # implementation; here the explicit evaluations above are charged too.
    diagnostic_A_rhs = 2*raw.shape[1]
    info = dict(replay=replay, tridiagonal_eigenvalues=values.tolist(), selected_tridiagonal_indices=selected,
        selected_tridiagonal_values=values[selected].tolist(), duplicate_relative_tolerance=duplicate_tolerance,
        raw_physical_M_gram_eigenvalues=ge.tolist(), physical_rank_threshold=threshold,
        reconstructed_columns=raw.shape[1], retained_independent_modes=U.shape[1],
        normalized_physical_M_gram_error=float(np.linalg.norm(physical_gram-np.eye(len(theta)), ord=2)),
        recomputed_Ritz_values=theta.tolist(), actual_Ritz_relative_residuals=relative_residual.tolist(),
        actual_Ritz_relative_Hnorm_residuals=relative_H_residual.tolist(),
        raw_Lanczos_A_image_relative_error=raw_relation_error,
        boundary_relation_A_image_relative_error=image_error, boundary_relation_residual_relative_error=relation_error,
        compact_boundary_relation_passes=bool(relation_passes),
        boundary_thresholds=dict(relative_A_image=1e-10, relative_residual_relation=1e-8),
        U_transpose_boundary_relative_norm=float(np.linalg.norm(U.T@boundary)/max(np.linalg.norm(boundary), 1e-300)),
        required_actual_image_A_rhs=U.shape[1], diagnostic_extra_A_rhs=diagnostic_A_rhs,
        required_reconstruction_inverse_applications=m+1,
        boundary_inverse_applications=1, Ritz_residual_and_SPD_inverse_applications=2*U.shape[1],
        model_M_rhs=int(2*raw.shape[1]+2*U.shape[1]),
        replay_scalar_state_bytes=int(sum(v.nbytes for v in (saved['alpha'], saved['beta'], saved['rhos']))),
        replay_Ritz_coefficients_bytes=coefficients.nbytes,
        raw_candidate_vector_bytes=raw.nbytes, replay_recurrence_vector_workspace_bytes=6*len(b)*8,
        reconstruction_major_array_payload_upper_bound_bytes=sum(v.nbytes for v in
            (raw, Mraw, gram, ge, gv, transform, U, MU, AU, small, theta, rotation, residual,
             Hresidual, boundary, weights, residual_error, h, HAU, physical_gram, raw_relation, raw_boundary)))
    return dict(U=U, AU=AU, MU=MU, theta=theta, boundary=boundary, h=h, weights=weights,
                delta=delta, HAU=HAU, info=info)


class LMP:
    def __init__(self, base, data, kind):
        self.base, self.kind = base, kind
        self.U = data['U'].copy()
        self.inverse_theta = 1/data['theta']
        self.AU = self.h = self.a = self.delta = None
        if kind == 'balanced_full':
            self.AU = data['AU'].copy()
        elif kind == 'balanced_compact':
            self.h = data['h'].copy()
            self.a = self.inverse_theta*data['weights']
            self.delta = np.asarray(data['delta'])
    def apply(self, r):
        ur = self.U.T@r
        if self.kind == 'balanced_full':
            z = self.base.apply(r-self.AU@(self.inverse_theta*ur))
            return z+self.U@(self.inverse_theta*(ur-self.AU.T@z))
        z = self.base.apply(r)
        if self.kind == 'spectral':
            return z+self.U@((self.inverse_theta-1)*ur)
        wr, hr = float(self.a@ur), float(self.h@r)
        return z+self.U@((self.inverse_theta-1)*ur-self.a*hr+self.delta*self.a*wr)-self.h*wr
    @property
    def storage_bytes(self):
        return self.base.storage_bytes+sum(a.nbytes for a in
            (self.U, self.inverse_theta, self.AU, self.h, self.a, self.delta) if isinstance(a, np.ndarray))


def relative_spd_minimum(base, Z, C):
    G = (Z.T@model(base, Z)+model(base, Z).T@Z)*.5
    values, vectors = linalg.eigh(G, check_finite=False)
    root = (vectors*np.sqrt(np.maximum(values, 0))[None, :])@vectors.T
    low = np.eye(len(C))+root@C@root
    return float(min(1., linalg.eigvalsh((low+low.T)*.5)[0]))


def validate_states(base, data):
    U, AU, HAU, theta = (data[k] for k in ('U', 'AU', 'HAU', 'theta'))
    inv = np.diag(1/theta)
    spectral_min = relative_spd_minimum(base, U, np.diag(1/theta-1))
    full_C = np.block([[inv@(AU.T@HAU+np.diag(theta))@inv, -inv], [-inv, np.zeros_like(inv)]])
    full_min = relative_spd_minimum(base, np.column_stack((U, HAU)), full_C)
    assert spectral_min > 0 and full_min > 0
    result = dict(spectral_relative_SPD_minimum=spectral_min, balanced_full_relative_SPD_minimum=full_min,
                  validation_model_M_rhs=6*U.shape[1])
    if data['info']['compact_boundary_relation_passes']:
        compact = LMP(base, data, 'balanced_compact')
        full = LMP(base, data, 'balanced_full')
        probes = np.random.default_rng(622021).normal(size=(len(U), 3))
        errors = [np.linalg.norm(compact.apply(v)-full.apply(v))/np.linalg.norm(full.apply(v)) for v in probes.T]
        a, delta = compact.a, float(compact.delta)
        C = np.block([[np.diag(1/theta-1)+delta*np.outer(a, a), -a[:, None]], [-a[None, :], np.zeros((1, 1))]])
        compact_min = relative_spd_minimum(base, np.column_stack((U, compact.h)), C)
        result.update(compact_vs_full_probe_relative_error=max(errors), compact_relative_SPD_minimum=compact_min,
            validation_inverse_applications=9, validation_model_M_rhs=8*U.shape[1]+2)
        if max(errors) > 1e-9 or compact_min <= 0:
            data['info']['compact_boundary_relation_passes'] = False
            result['compact_rejected_by_action_or_SPD_check'] = True
    else:
        result['validation_inverse_applications'] = 0
    return result


def independent_dense_check():
    rng = np.random.default_rng(622022)
    X = rng.normal(size=(20, 4))
    kernel = GaussianKernel(X, 2.)
    K = kernel.dense()
    base = initialize_rpcholesky(kernel, 5, .05, rng)
    A = K+.05*np.eye(20)
    M = model(base, np.eye(20))
    H = linalg.solve(M, np.eye(20), assume_a='pos')
    W = rng.normal(size=(20, 3))
    gram = W.T@M@W
    L = linalg.cholesky(gram, lower=True)
    U = linalg.solve_triangular(L, W.T, lower=True).T
    theta, rotation = linalg.eigh(U.T@A@U)
    U = U@rotation
    data = dict(U=U, AU=A@U, theta=theta)
    spectral, balanced = LMP(base, data, 'spectral'), LMP(base, data, 'balanced_full')
    S = H+U@np.diag(1/theta-1)@U.T
    P = np.eye(20)-U@np.diag(1/theta)@U.T@A
    B = P@H@P.T+U@np.diag(1/theta)@U.T
    linalg.cholesky(S)
    linalg.cholesky(B)
    probes = rng.normal(size=(20, 3))
    se = max(np.linalg.norm(spectral.apply(v)-S@v)/np.linalg.norm(S@v) for v in probes.T)
    be = max(np.linalg.norm(balanced.apply(v)-B@v)/np.linalg.norm(B@v) for v in probes.T)
    assert max(se, be) < 1e-10
    return dict(spectral_dense_inverse_action_relative_error=se, balanced_dense_inverse_action_relative_error=be,
                spectral_cholesky_SPD=True, balanced_cholesky_SPD=True)


def pcg(A, b, apply, *, x0=None, r0=None, p0=None, rho0=None, prefix_iterations=0, cpu_start=None):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b.copy() if r0 is None else r0.copy()
    mapps = 0
    if p0 is None:
        z = apply(r)
        mapps += 1
        p, rho = z.copy(), float(r@z)
    else:
        p, rho = p0.copy(), float(rho0)
    norm_b = np.linalg.norm(b)
    checks = replacements = 0
    last_checked = -1
    true_rel = float(np.linalg.norm(r)/norm_b)
    reason = 'max_iterations'
    for step in range(1, 2501-prefix_iterations):
        if cpu_start is not None and process_time()-cpu_start > 104:
            step -= 1
            reason = 'cpu_guard'
            break
        ap = A@p
        curvature = float(p@ap)
        if min(curvature, rho) <= 0 or not np.isfinite(curvature+rho):
            step -= 1
            reason = 'nonpositive_inner_product'
            break
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        suspected = np.linalg.norm(r)/norm_b <= 1e-8
        if step % 20 == 0 or suspected or step == 2500-prefix_iterations:
            true_r = b-A@x
            checks += 1
            last_checked = step
            true_rel = float(np.linalg.norm(true_r)/norm_b)
            if true_rel <= 1e-8:
                reason = 'converged'
                break
            if suspected:
                r = true_r
                z = apply(r)
                mapps += 1
                p, rho = z.copy(), float(r@z)
                replacements += 1
                continue
        z = apply(r)
        mapps += 1
        next_rho = float(r@z)
        p = z+(next_rho/rho)*p
        rho = next_rho
    if last_checked != step:
        true_rel = float(np.linalg.norm(b-A@x)/norm_b)
        checks += 1
    return dict(iterations=prefix_iterations+step, continuation_iterations=step,
        converged=bool(true_rel<=1e-8), true_relative_residual=true_rel, stop_reason=reason,
        direction_A_products=step, verification_A_products=checks, preconditioner_applications=mapps,
        numerical_restarts=replacements)


def add_costs(value, *, prefix_steps, replay_steps, image_rhs, diagnostics_A_rhs, correction,
              setup_inverse, diagnostic_inverse, initial_columns, n, factor_bytes):
    value.update(prefix_A_products=prefix_steps, replay_A_products=replay_steps,
        reconstructed_image_A_rhs=image_rhs, diagnostic_extra_A_rhs=diagnostics_A_rhs,
        prefix_true_residual_A_products=int(prefix_steps>0), coarse_correction_verification_A_products=int(correction),
        required_sequential_A_equivalent_products=prefix_steps+replay_steps+image_rhs+int(prefix_steps>0)+int(correction)
            +value['direction_A_products']+value['verification_A_products'],
        total_A_equivalent_products_including_diagnostics=prefix_steps+replay_steps+image_rhs+diagnostics_A_rhs
            +int(prefix_steps>0)+int(correction)+value['direction_A_products']+value['verification_A_products'],
        setup_inverse_applications=setup_inverse, diagnostic_inverse_applications=diagnostic_inverse,
        all_inverse_applications=setup_inverse+diagnostic_inverse+value['preconditioner_applications'],
        initial_kernel_column_queries=initial_columns, retained_preconditioner_bytes=factor_bytes)
    value['required_kernel_entry_pass_equivalents'] = value['required_sequential_A_equivalent_products']+initial_columns/n


def main():
    output = ROOT/'results/two-pass-spectral-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (108, 110))
    started, cpu_start = perf_counter(), process_time()
    config_path = ROOT/'configs/revision-two-pass-spectral-diagnostic.json'
    config = json.loads(config_path.read_text())
    sources = [Path(__file__).resolve(), ROOT/'src/nystrom_project/numerics.py', ROOT/'src/nystrom_project/preconditioners.py']
    result = dict(status='running', scope='Predeclared two-pass diagnostic, dense operators for numerical study; no production CPU claim',
        protocol=config, protocol_sha256=sha256(config_path.read_bytes()).hexdigest(),
        implementation_choices=dict(physical_rank_cutoff='64 eps max(n,d) max_eigenvalue(M-Gram)',
            boundary_relative_A_image_threshold=1e-10, boundary_relative_residual_threshold=1e-8,
            matched_rank_rule='largest compact factor rank with actual retained numerical payload <= LMP bytes',
            strong_control_seed_sequences=dict(RP=[0, 6220, 1], Gaussian=[0, 6220, 3]),
            actual_A_images='one dense block product, charged as d operator RHS; replay remains20 sequential products',
            diagnostic_products='extra explicit raw-relation A products and all verification inverse applications recorded separately and included in all-work totals'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}, cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['independent_checks'] = independent_dense_check()
        for name in config['workloads']:
            path = ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
            data = np.load(path)
            b = data['y_train']
            kernel = GaussianKernel(data['X_train'], float(data['bandwidth']))
            K = kernel.dense()
            for lam in config['regularizations']:
                A = K+lam*np.eye(len(b))
                unpreconditioned = pcg(A, b, lambda r:r.copy(), cpu_start=cpu_start)
                unpreconditioned['inverse_applications'] = 0
                unpreconditioned['total_A_products'] = unpreconditioned['direction_A_products']+unpreconditioned['verification_A_products']
                for k in config['initial_ranks']:
                    if process_time()-cpu_start > 94:
                        result['status'] = 'budget_exhausted'
                        save()
                        return
                    base = initialize_rpcholesky(kernel, k, lam, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                    saved = prefix(A, b, base, 20)
                    true_r = b-A@saved['x']
                    learned = spectral_data(A, b, base, saved)
                    spd = validate_states(base, learned)
                    U, theta = learned['U'], learned['theta']
                    corrected_x = saved['x']+U@((U.T@true_r)/theta)
                    corrected_r = b-A@corrected_x
                    balanced_kind = 'balanced_compact' if learned['info']['compact_boundary_relation_passes'] else 'balanced_full'
                    spectral = LMP(base, learned, 'spectral')
                    balanced = LMP(base, learned, balanced_kind)
                    row = dict(dataset=name, regularization=lam, rank=k, n=len(b),
                        prepared_input=str(path.relative_to(ROOT)), prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                        preparation_regularization=.001, stored_reference_ignored=True, actual_operator_regularization=lam,
                        initial_indices=base.indices.tolist(), reconstruction=learned['info'], SPD_checks=spd,
                        chosen_balanced_representation=balanced_kind,
                        checkpoint_true_relative_residual=float(np.linalg.norm(true_r)/np.linalg.norm(b)),
                        coarse_corrected_true_relative_residual=float(np.linalg.norm(corrected_r)/np.linalg.norm(b)),
                        coarse_constraint_norm=float(np.linalg.norm(U.T@corrected_r)),
                        unpreconditioned_CG=unpreconditioned, runs={}, strong_controls=[],
                        storage=dict(original_actual_column_state_bytes=base.storage_bytes,
                            spectral_state_bytes=spectral.storage_bytes, balanced_state_bytes=balanced.storage_bytes,
                            full_balanced_reference_state_bytes=LMP(base, learned, 'balanced_full').storage_bytes,
                            shared_oracle_reconstruction_array_bytes=sum(v.nbytes for v in learned.values() if isinstance(v, np.ndarray)),
                            dense_kernel_and_operator_bytes=K.nbytes+A.nbytes,
                            note='The comparison harness retains common reconstruction arrays and both LMP objects for validation; these are explicitly extra oracle allocations. Deployable-state budgets include all base columns/core/basis and the specified LMP arrays; no full PCG basis is retained. Ordinary recurrence vectors and LAPACK workspaces are additional.'))
                    result['cases'].append(row)
                    configurations = [
                        ('old_uninterrupted', base.apply, saved['x'], saved['r'], saved['p'], saved['rho'], False, False, base.storage_bytes),
                        ('old_restart20', base.apply, saved['x'], true_r, None, None, False, False, base.storage_bytes),
                        ('correction_only', base.apply, corrected_x, corrected_r, None, None, True, True, base.storage_bytes),
                        ('spectral_LMP', spectral.apply, corrected_x, corrected_r, None, None, True, True, spectral.storage_bytes),
                        ('balanced_Ritz_LMP', balanced.apply, corrected_x, corrected_r, None, None, True, True, balanced.storage_bytes)]
                    for method, apply, x0, r0, p0, rho0, replay, correction, storage in configurations:
                        value = pcg(A, b, apply, x0=x0, r0=r0, p0=p0, rho0=rho0, prefix_iterations=20, cpu_start=cpu_start)
                        d = U.shape[1]
                        add_costs(value, prefix_steps=20, replay_steps=20 if replay else 0,
                            image_rhs=d if replay else 0, diagnostics_A_rhs=learned['info']['diagnostic_extra_A_rhs'] if replay else 0,
                            correction=correction, setup_inverse=21+(22 if replay else 0),
                            diagnostic_inverse=(2*d+spd['validation_inverse_applications']) if replay else 0,
                            initial_columns=k, n=len(b), factor_bytes=storage)
                        row['runs'][method] = value
                    targets = dict(correction_only=base.storage_bytes, spectral_LMP=spectral.storage_bytes,
                                   balanced_Ritz_LMP=balanced.storage_bytes)
                    jobs = {}
                    for label, budget in targets.items():
                        for kind in ('compact_RP', 'power_Gaussian'):
                            per_rank = 8*len(b)+(16 if kind=='compact_RP' else 8)
                            rank = min(len(b), budget//per_rank)
                            jobs.setdefault((kind, rank), []).append(dict(method=label, budget_bytes=budget))
                    for (kind, rank), matched in jobs.items():
                        if process_time()-cpu_start > 103:
                            row['status'] = 'partial_strong_controls'
                            result['status'] = 'budget_exhausted'
                            save()
                            return
                        if kind == 'compact_RP':
                            reference = initialize_rpcholesky(kernel, rank, lam, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                            control = LowRankPreconditioner.from_nystrom(reference)
                            del reference
                            columns, passes = rank, 0
                        else:
                            omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(len(b), rank))
                            Q, _ = linalg.qr(K@omega, mode='economic', check_finite=False)
                            Y = K@Q
                            control = LowRankPreconditioner.from_sketch(Y, Q.T@Y, lam)
                            del omega, Q, Y
                            columns, passes = 0, 2
                        assert all(control.storage_bytes<=v['budget_bytes'] for v in matched)
                        value = pcg(A, b, control.apply, cpu_start=cpu_start)
                        value.update(kind=kind, rank=rank, effective_rank=control.rank, matched_to=matched,
                            retained_factor_bytes=control.storage_bytes, setup_column_queries=columns,
                            setup_kernel_matmat_passes=passes, setup_kernel_matmat_rhs=passes*rank,
                            required_kernel_entry_pass_equivalents=value['direction_A_products']+value['verification_A_products']+passes+columns/len(b),
                            note='Gaussian setup uses cached dense K; two block passes have rank RHS each. No end-to-end timing advantage is inferred.')
                        row['strong_controls'].append(value)
                    row['status'] = 'complete'
                    save()
                    print(json.dumps(dict(dataset=name, regularization=lam, rank=k, modes=U.shape[1], balanced=balanced_kind,
                        iterations={key:v['iterations'] for key,v in row['runs'].items()},
                        replay_inclusive_A={key:v['required_sequential_A_equivalent_products'] for key,v in row['runs'].items()},
                        cpu_seconds=process_time()-cpu_start)), flush=True)
        result['status'] = 'complete'
        save()
        print(json.dumps(dict(status='complete', cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__ == '__main__':
    main()
