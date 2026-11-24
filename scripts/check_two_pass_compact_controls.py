#!/usr/bin/env python3
"""Correct retained-state accounting without rerunning the spectral trajectories.

The fixed selected-column Nyström action only needs its compact factor and
selected indices after setup. Original replay observations remain frozen.
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
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from nystrom_project.preconditioners import LowRankPreconditioner
from check_two_pass_spectral import pcg


def main():
    source_path = ROOT/'results/two-pass-spectral-diagnostic.json'
    source = json.loads(source_path.read_text())
    output = ROOT/'results/two-pass-spectral-compact-controls-diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (43, 45))
    started, cpu_start = perf_counter(), process_time()
    sources = [Path(__file__).resolve(), ROOT/'scripts/check_two_pass_spectral.py',
        ROOT/'src/nystrom_project/numerics.py', ROOT/'src/nystrom_project/preconditioners.py']
    result = dict(status='running', scope='Corrected compact retained-state controls; original LMP/replay trajectories are inherited unchanged',
        source_result=dict(path=str(source_path.relative_to(ROOT)), sha256=sha256(source_path.read_bytes()).hexdigest()),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(optimization='Release raw columns/core after initialization; retain Nyström basis, eigenvalues and original selected indices',
            matched_rank='largest compact factor rank whose actual numerical array bytes fit the optimized LMP state',
            unchanged='kernel, split, seed, prefix, recurrence, selected/reconstructed modes, LMP actions and all replay-inclusive work counts',
            controls_only_rerun=True, additional_cpu_limit_seconds=45, blas_threads=1,
            method_claim='No later actual-column exchange; stored selected indices preserve initialization identity. Setup peak and oracle arrays remain extra.'), cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started, elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        for original in source['cases']:
            if process_time()-cpu_start > 39:
                result['status'] = 'budget_exhausted'
                save()
                return
            name, lam, k = original['dataset'], original['regularization'], original['rank']
            path = ROOT/original['prepared_input']
            data = np.load(path)
            b = data['y_train']
            kernel = GaussianKernel(data['X_train'], float(data['bandwidth']))
            K = kernel.dense()
            A = K+lam*np.eye(len(b))
            raw = initialize_rpcholesky(kernel, k, lam, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
            assert raw.indices.tolist()==original['initial_indices']
            compact = LowRankPreconditioner.from_nystrom(raw)
            probes = np.random.default_rng(622023).normal(size=(len(b), 3))
            old_values = [raw.apply(v) for v in probes.T]
            new_values = [compact.apply(v) for v in probes.T]
            identical = all(np.array_equal(a, b) for a, b in zip(old_values, new_values))
            action_error = max(np.linalg.norm(a-b)/np.linalg.norm(a) for a,b in zip(old_values,new_values))
            reconstructed_columns = compact.basis@(compact.eigenvalues[:, None]*compact.basis[compact.indices].T)
            column_error = float(np.linalg.norm(reconstructed_columns-raw.columns)/np.linalg.norm(raw.columns))
            assert identical and column_error < 1e-10
            old_bytes, compact_bytes = raw.storage_bytes, compact.storage_bytes
            setup_returned_states_bytes = old_bytes+compact_bytes
            del raw, reconstructed_columns, old_values, new_values, probes
            targets = dict(correction_only=compact_bytes,
                spectral_LMP=compact_bytes+original['storage']['spectral_state_bytes']-old_bytes,
                balanced_Ritz_LMP=compact_bytes+original['storage']['balanced_state_bytes']-old_bytes)
            row = dict(dataset=name, regularization=lam, rank=k, n=len(b),
                prepared_input=original['prepared_input'], prepared_sha256=sha256(path.read_bytes()).hexdigest(),
                initial_indices=compact.indices.tolist(), compact_action_bitwise_identical_on_probes=identical,
                compact_action_relative_error=action_error, selected_column_reconstruction_relative_error=column_error,
                original_redundant_base_bytes=old_bytes, optimized_compact_base_bytes=compact_bytes,
                optimized_retained_state_budgets=targets, chosen_balanced_representation=original['chosen_balanced_representation'],
                setup_returned_raw_plus_compact_state_bytes=setup_returned_states_bytes,
                setup_peak_note='Raw columns/core and factor construction/SVD are setup workspace, released before the intended prefix. Returned-state-plus-copy figure excludes internal LAPACK workspaces. This oracle retains dense K/A and a common compact reference while comparing controls.',
                inherited_reconstruction=original['reconstruction'], inherited_SPD_checks=original['SPD_checks'],
                inherited_unpreconditioned_CG=original['unpreconditioned_CG'], LMP_runs={}, strong_controls=[])
            result['cases'].append(row)
            for method, run in original['runs'].items():
                value = dict(run)
                value['original_redundant_retained_bytes'] = value['retained_preconditioner_bytes']
                value['retained_preconditioner_bytes'] = targets.get(method, compact_bytes)
                value['numerical_observation_source'] = 'frozen source result; compact inverse action uses identical basis/eigenvalues and arithmetic'
                row['LMP_runs'][method] = value
            jobs = {}
            for label, budget in targets.items():
                for kind in ('compact_RP', 'power_Gaussian'):
                    per_rank = 8*len(b)+(16 if kind=='compact_RP' else 8)
                    rank = min(len(b), budget//per_rank)
                    jobs.setdefault((kind, rank), []).append(dict(method=label, budget_bytes=budget))
            for (kind, rank), matched in jobs.items():
                if process_time()-cpu_start > 41:
                    row['status'] = 'partial'
                    result['status'] = 'budget_exhausted'
                    save()
                    return
                if kind=='compact_RP':
                    if rank==k:
                        control = LowRankPreconditioner(compact.basis, compact.eigenvalues, lam, compact.indices)
                    else:
                        raw_control = initialize_rpcholesky(kernel, rank, lam, np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
                        control = LowRankPreconditioner.from_nystrom(raw_control)
                        del raw_control
                    columns, passes = rank, 0
                else:
                    omega = np.random.default_rng(np.random.SeedSequence([0, 6220, 3])).normal(size=(len(b), rank))
                    Q, _ = linalg.qr(K@omega, mode='economic', check_finite=False)
                    Y = K@Q
                    control = LowRankPreconditioner.from_sketch(Y, Q.T@Y, lam)
                    del omega, Q, Y
                    columns, passes = 0, 2
                assert all(control.storage_bytes<=m['budget_bytes'] for m in matched)
                value = pcg(A, b, control.apply, cpu_start=cpu_start)
                value.update(kind=kind, rank=rank, effective_rank=control.rank, matched_to=matched,
                    retained_factor_bytes=control.storage_bytes, setup_column_queries=columns,
                    setup_kernel_matmat_passes=passes, setup_kernel_matmat_rhs=passes*rank,
                    required_kernel_entry_pass_equivalents=value['direction_A_products']+value['verification_A_products']+passes+columns/len(b))
                row['strong_controls'].append(value)
                del control
            row['status'] = 'complete'
            save()
            summary = {}
            for method in ('spectral_LMP', 'balanced_Ritz_LMP'):
                power = next(v for v in row['strong_controls'] if v['kind']=='power_Gaussian' and any(m['method']==method for m in v['matched_to']))
                rp = next(v for v in row['strong_controls'] if v['kind']=='compact_RP' and any(m['method']==method for m in v['matched_to']))
                summary[method] = dict(LMP_work=row['LMP_runs'][method]['required_kernel_entry_pass_equivalents'],
                    compact_RP_rank=rp['rank'], compact_RP_work=rp['required_kernel_entry_pass_equivalents'],
                    power_rank=power['rank'], power_work=power['required_kernel_entry_pass_equivalents'])
            print(json.dumps(dict(dataset=name, regularization=lam, rank=k, comparisons=summary,
                cpu_seconds=process_time()-cpu_start)), flush=True)
        result['status'] = 'complete'
        save()
        print(json.dumps(dict(status='complete', cpu_seconds=result['elapsed_cpu_seconds'])))


if __name__=='__main__':
    main()
