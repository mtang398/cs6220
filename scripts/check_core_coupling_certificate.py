#!/usr/bin/env python3
"""Dense offline coupling-sensitive H1 gate on the frozen exact trajectory."""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, eigvalsh, svdvals
from threadpoolctl import threadpool_limits
from check_packed_float_factor import PackedFloat, unpack, ROOT
from check_packed_float_allocation import RawPointers


def main():
    output = ROOT/'results/core-coupling-certificate.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (27, 30))
    start, cpu = perf_counter(), process_time()
    frozen_source = ROOT/'scripts/check_core_order_certificate.py'
    frozen_result = ROOT/'results/core-order-certificate.json'
    old = json.loads(frozen_result.read_text())
    assert sha256(frozen_source.read_bytes()).hexdigest() == old['source_sha256']
    helper = RawPointers(ROOT/'results/packed_float_factor.so')
    sources = [Path(__file__), frozen_source, frozen_result,
               ROOT/'scripts/check_packed_float_factor.py',
               ROOT/'scripts/check_packed_float_allocation.py',
               ROOT/'scripts/check_schur_inplace_frontier_v2.py',
               ROOT/'scripts/packed_float_factor.c', ROOT/'results/packed_float_factor.so']
    multiplier = float(10*np.sqrt(2/np.pi))
    record = dict(status='running', sources={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(core=600, datasets=['smooth', 'mixed'], regularizations=[.001, 1e-5],
            seed=0, probe_seed=622108, probe_count=6, probe_multiplier=multiplier,
            probe_reuse='The same six Gaussian vectors are used for E and Z; union bound does not require independent events.',
            E='sym(L^-1 C L^-T)-I, matching the frozen whitening diagnostic',
            Z='B.T L^-T E, where B=K[S,R] and packed FP32 L is unpacked only in this dense oracle',
            old_bound='delta*sqrt(ntail/(1-delta)*h0.T H0 h0)',
            new_bound='sigma/(1-delta)*sqrt(h0.T H0 h0)',
            diagnostic_exact_norm_bound='||Z||2/(1-||E||2)*sqrt(h0.T H0 h0); free exact norms, not implementable screen',
            rtol=1e-8, allocated_gap_fraction=.25,
            allocation='Allowed increment=0.25*rtol*||b||/N using the exact future trajectory length N, free offline.',
            acceptance='alpha1*bound <= allowed increment, using the current actual H1 curvature alpha1.',
            probabilistic_statement='For fixed E,Z independent of the six Gaussian probes, delta and sigma simultaneously upper-bound operator norms with failure at most2e-6 in exact arithmetic. This excludes rounding error.',
            scope='Dense offline oracle only; no native timing, memory, adaptive recurrence, or H3-certification claim.',
            execution_cpu_cap=27), runs=[])

    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,
                      elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    save()
    with threadpool_limits(limits=1):
        for name in ['smooth', 'mixed']:
            path = ROOT/'data/prepared'/f'synthetic-n2560-{name}-s0-lambda0.001.npz'
            with np.load(path) as data:
                X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
            for lam in [.001, 1e-5]:
                case_start, case_cpu = perf_counter(), process_time()
                factor = PackedFloat(X, b, bw, 600, lam, helper)
                J = factor.indices
                R = np.setdiff1d(np.arange(len(b)), J)
                K = factor.kernel.block(X, X)
                C = K[np.ix_(J, J)]+lam*np.eye(600)
                B = K[np.ix_(J, R)]
                tail_kernel = K[np.ix_(R, R)]
                cf = cho_factor(C, lower=True)
                S = tail_kernel+lam*np.eye(len(R))-B.T@cho_solve(cf, B)
                L = unpack(factor.factor, 600)
                T = solve_triangular(L, C, lower=True)
                W = solve_triangular(L, T.T, lower=True).T
                E = (W+W.T)*.5-np.eye(600)
                spectrum = eigvalsh(E)
                actual_delta = float(max(abs(spectrum[0]), abs(spectrum[-1])))
                Z = B.T@solve_triangular(L, E, lower=True, trans='T')
                actual_sigma = float(svdvals(Z)[0])
                probes = np.random.default_rng(622108).normal(size=(600, 6))
                e_norms = np.linalg.norm(E@probes, axis=0)
                z_norms = np.linalg.norm(Z@probes, axis=0)
                delta = float(multiplier*e_norms.max())
                sigma = float(multiplier*z_norms.max())
                rhs = b[R]-B.T@cho_solve(cf, b[J])
                y = np.zeros(len(R))
                r = rhs.copy()
                p = r.copy()
                rr = float(r@r)
                norm_b = float(np.linalg.norm(b))
                trace = []
                for iteration in range(2000):
                    full = np.empty(len(b))
                    full[R] = y
                    full[J] = cho_solve(cf, b[J]-B@y)
                    original = float(np.linalg.norm(b-K@full-lam*full)/norm_b)
                    if original <= 1e-8:
                        break
                    sp = S@p
                    alpha = rr/float(p@sp)
                    trace.append((p.copy(), rr))
                    y += alpha*p
                    r -= alpha*sp
                    next_rr = float(r@r)
                    p = r+(next_rr/rr)*p
                    rr = next_rr
                assert original <= 1e-8
                allowed = float(.25e-8*norm_b/len(trace))
                history = []
                for iteration, (p, rr) in enumerate(trace, 1):
                    t = B@p
                    z0 = factor.solve(t)
                    h0 = t-C@z0
                    w0 = factor.solve(h0)
                    y1 = z0+w0
                    curvature = float(p@(tail_kernel@p+lam*p))-float(t@y1)
                    assert curvature > 0
                    alpha1 = rr/curvature
                    energy = max(float(h0@w0), 0.)
                    old_bound = float(delta*np.sqrt(len(R)/(1-delta)*energy)) if delta < 1 else None
                    new_bound = float(sigma/(1-delta)*np.sqrt(energy)) if delta < 1 else None
                    exact_bound = float(actual_sigma/(1-actual_delta)*np.sqrt(energy)) if actual_delta < 1 else None
                    reference = cho_solve(cf, t)
                    actual_error = float(np.linalg.norm(B.T@(reference-y1)))
                    history.append(dict(iteration=iteration, alpha1=float(alpha1),
                        old_bound=old_bound, coupling_bound=new_bound,
                        exact_norm_diagnostic_bound=exact_bound, actual_action_error=actual_error,
                        old_increment=None if old_bound is None else float(alpha1*old_bound),
                        coupling_increment=None if new_bound is None else float(alpha1*new_bound),
                        exact_norm_increment=None if exact_bound is None else float(alpha1*exact_bound),
                        actual_increment=float(alpha1*actual_error),
                        old_accept=bool(old_bound is not None and alpha1*old_bound <= allowed),
                        coupling_accept=bool(new_bound is not None and alpha1*new_bound <= allowed),
                        exact_norm_accept=bool(exact_bound is not None and alpha1*exact_bound <= allowed),
                        actual_oracle_accept=bool(alpha1*actual_error <= allowed),
                        measured_action_error_below_coupling_bound=bool(new_bound is not None and actual_error <= new_bound)))
                old_case = next(row for row in old['runs'] if row['dataset'] == name and row['regularization'] == lam)
                counts = {key:sum(row[key] for row in history) for key in ['old_accept', 'coupling_accept', 'exact_norm_accept', 'actual_oracle_accept']}
                case = dict(dataset=name, regularization=lam,
                    data_sha256=sha256(path.read_bytes()).hexdigest(), iterations=len(trace),
                    true_relative_residual=original, actual_whitened_delta=actual_delta,
                    probe_delta=delta, exact_Z_spectral_norm=actual_sigma, probe_sigma=sigma,
                    E_probe_norms=e_norms.tolist(), Z_probe_norms=z_norms.tolist(),
                    probe_delta_covers_actual=delta >= actual_delta,
                    probe_sigma_covers_actual=sigma >= actual_sigma,
                    coupling_to_old_bound_ratio=float(sigma/(delta*np.sqrt(len(R)*(1-delta)))) if delta < 1 else None,
                    acceptance_counts=counts, allowed_increment=allowed,
                    accepted_iterations={key:[row['iteration'] for row in history if row[key]] for key in counts},
                    frozen_comparison=dict(iterations_match=len(trace) == old_case['iterations'],
                        delta_absolute_difference=abs(delta-old_case['probe_delta']),
                        old_accept_count_match=counts['old_accept'] == old_case['accept_one_count'],
                        actual_oracle_count_match=counts['actual_oracle_accept'] == old_case['actual_error_oracle_accepts']),
                    elapsed_wall_seconds=perf_counter()-case_start,
                    elapsed_cpu_seconds=process_time()-case_cpu, history=history)
                assert case['frozen_comparison']['iterations_match']
                assert case['frozen_comparison']['old_accept_count_match']
                assert case['frozen_comparison']['actual_oracle_count_match']
                record['runs'].append(case)
                save()
                print(json.dumps({key:case[key] for key in ['dataset', 'regularization', 'iterations', 'acceptance_counts', 'exact_Z_spectral_norm', 'probe_sigma']}), flush=True)
    record['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
