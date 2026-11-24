#!/usr/bin/env python3
"""Static core-size scope screen using the audited implicit Schur implementation.

One observation per case; all setup is charged. No adaptive algorithm, fitted
selection threshold, or confirmation seed is introduced. Failed/censored rows
remain in the output, and aggregate comparisons state their common-case set.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/check_schur_inplace_frontier_v2.py'
BASE_HASH=sha256(BASE.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('audited_schur_frontier',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
assert sha256(BASE.read_bytes()).hexdigest()==BASE_HASH


def plain_cg(X,b,bandwidth,lam,*,rtol=1e-8,maxiter=2500,cpu_cap=8.,deadline=float('inf')):
    started,cpu_started=perf_counter(),process_time()
    budget=base.Budget([X,b]);kernel=base.SharedKernel(X,bandwidth,budget)
    x=np.zeros_like(b);r=b.copy();p=r.copy();rr=float(r@r)
    norm_b=float(np.linalg.norm(b));checks=replacements=products=0
    last_check=-1;relative=None;progress=[];status='maxiter';iteration=0
    budget.live=[x,r,p];budget.observe('plain_setup')
    setup=perf_counter()-started
    def verify(iteration):
        nonlocal checks,last_check,relative
        budget.live=[x,r,p]
        residual=kernel.full(x)
        budget.observe('plain_verification_update',residual,temporary_allowance=x.nbytes)
        residual*=-1.;residual+=b;residual-=lam*x
        relative=float(np.linalg.norm(residual)/norm_b)
        checks+=1;last_check=iteration
        progress.append(dict(iteration=iteration,true_relative_residual=relative))
        return residual
    for iteration in range(1,maxiter+1):
        if process_time()-cpu_started>cpu_cap or process_time()>deadline:
            status='cpu_screen_cap';iteration-=1;break
        budget.live=[x,r,p]
        ap=kernel.full(p);products+=1
        budget.observe('plain_iteration_update',ap,temporary_allowance=2*x.nbytes)
        ap+=lam*p
        curvature=float(p@ap)
        if curvature<=0 or not np.isfinite(curvature):
            status='breakdown';del ap;break
        alpha=rr/curvature;x+=alpha*p;r-=alpha*ap
        del ap
        suspected=np.linalg.norm(r)<=rtol*norm_b
        if iteration%20==0 or suspected or iteration==maxiter:
            true=verify(iteration)
            if relative<=rtol:status='converged';del true;break
            if suspected:
                r=true.copy();p=r.copy();rr=float(r@r);replacements+=1
                del true
                continue
            del true
        next_rr=float(r@r);p*=next_rr/rr;p+=r;rr=next_rr
    if last_check!=iteration:
        true=verify(iteration);del true
    return x,dict(status=status,converged=relative<=rtol,iterations=iteration,
        true_relative_residual=relative,setup_seconds=setup,
        total_seconds=perf_counter()-started,total_cpu_seconds=process_time()-cpu_started,
        operator_products=products,original_verifications=checks,residual_replacements=replacements,
        memory=budget.report(),kernel_stats=kernel.stats,progress=progress,
        retained_factor_and_indices_bytes=0,common_immutable_input_bytes=X.nbytes+b.nbytes,
        kernel_X_is_common_input=kernel.X is X,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def tiny_plain_check():
    rng=np.random.default_rng(622099)
    X=np.ascontiguousarray(rng.normal(size=(27,5)));b=rng.normal(size=27)
    lam=.08;bandwidth=1.2
    A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bandwidth**2))+lam*np.eye(len(X))
    actual,run=plain_cg(X,b,bandwidth,lam,rtol=1e-11,maxiter=200)
    exact=linalg.solve(A,b,assume_a='pos')
    error=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
    residual=float(np.linalg.norm(b-A@actual)/np.linalg.norm(b))
    assert run['converged'] and error<1e-9 and residual<1e-10
    return dict(solution_relative_error=error,independent_true_relative_residual=residual,
        recorded_true_relative_residual=run['true_relative_residual'],iterations=run['iterations'])


def summarize(record):
    methods=['plain_cg','schur256','schur320','schur336']
    groups={}
    for row in record['runs']:
        groups.setdefault((row['dataset'],row['regularization']),{})[row['method']]=row
    def good(r):
        return bool(r.get('converged') and r.get('resource_validated_by_separate_trace'))
    cases=[]
    for (dataset,lam),rows in groups.items():
        eligible={m:r for m,r in rows.items() if good(r)}
        winner=min(eligible,key=lambda m:eligible[m]['total_seconds']) if eligible else None
        cases.append(dict(dataset=dataset,regularization=lam,
            fastest_observed_feasible_method=winner,
            fastest_seconds=eligible[winner]['total_seconds'] if winner else None,
            statuses={m:r['status'] for m,r in rows.items()},
            setup_fraction={m:r['setup_seconds']/r['total_seconds'] for m,r in rows.items() if 'total_seconds' in r}))
    def aggregate(required):
        common=[(key,rows) for key,rows in groups.items() if all(m in rows and good(rows[m]) for m in required)]
        if not common:return dict(required_methods=required,common_case_count=0)
        totals={m:sum(rows[m]['total_seconds'] for _,rows in common) for m in required}
        oracle=sum(min(rows[m]['total_seconds'] for m in required) for _,rows in common)
        best=min(totals,key=totals.get)
        return dict(required_methods=required,common_case_count=len(common),
            common_cases=[dict(dataset=k[0],regularization=k[1]) for k,_ in common],
            total_seconds_by_method=totals,best_single_method=best,
            best_single_total_seconds=totals[best],per_case_oracle_total_seconds=oracle,
            oracle_saving_fraction=1-oracle/totals[best])
    return dict(cases=cases,
        all_four_common=aggregate(methods),cores_common=aggregate(methods[1:]),
        baseline256_320_plain_common=aggregate(methods[:3]),
        note='Oracle uses observed single-run elapsed time and omits selection cost. A censored run is not assigned an invented completion time; infeasible methods are excluded from feasible per-case winners.')


def main():
    output=ROOT/'results/static-schur-scope-diagnostic.json'
    startup_attempt=None
    if output.exists():
        prior=json.loads(output.read_text())
        if prior.get('runs'):raise FileExistsError(output)
        startup_attempt=dict(source_sha256=prior['source_sha256'],
            elapsed_cpu_seconds=prior.get('elapsed_cpu_seconds'),native_runs=0,
            reason='Optional lambda0.1 prepared file missing before first native run; corrected input check permits missing duplicate files. datasets.prepare constructs X,b,bandwidth independently of regularization.')
    resource.setrlimit(resource.RLIMIT_CPU,(100,105))
    started,cpu_started=perf_counter(),process_time()
    # Prioritize the missing low-shift/control workloads; no outcome-based order.
    all_names=['smooth','random','mixed','Concrete','Airfoil']
    missing_first=['random','mixed','Airfoil','smooth','Concrete']
    cases=[(name,.1) for name in all_names]+[(name,lam) for lam in [.001,1e-5] for name in missing_first]
    methods=['plain_cg','schur256','schur320','schur336']
    record=dict(status='running',protocol=dict(seed=0,repeats=1,
        cases=[dict(dataset=name,regularization=lam) for name,lam in cases],methods=methods,
        rtol=1e-8,maxiter=2500,true_check_interval=20,per_run_cpu_cap=8.,total_cpu_cap=100.,
        source_reuse='Imported audited static Schur solver unchanged; own plain CG uses the same SharedKernel and Budget.',
        selection='All Schur controls recompute FPS from index0 and charge setup.',
        memory_cap_bytes=base.LIMIT,library_python_reserve_bytes=base.RESERVE,
        immutable_common_inputs='Single X and b; all kernel feature packs and owned arrays charged.',
        trace='One separate setup/action/verification audit per n/core; plain audit uses two iterations. Trace time excluded from native run times.',
        lock='/tmp/cs6220-project-benchmark.lock',interpretation='Bounded static headroom screen, no adaptive or confirmation claim.'),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__).read_bytes()).hexdigest(),str(BASE.relative_to(ROOT)):BASE_HASH},
        runs=[],allocation_audits=[],unrun_cases=[],input_invariance_checks={},
        startup_attempt=startup_attempt,
        dataset_source_sha256=sha256((ROOT/'src/nystrom_project/datasets.py').read_bytes()).hexdigest())
    cache={};audits={}
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summary=summarize(record))
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_plain_check']=tiny_plain_check();save()
        for name,lam in cases:
            if name not in cache:
                stem=f'synthetic-n640-{name}' if name in ('smooth','random','mixed') else f'{name.lower()}-observed'
                path=ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz'
                with np.load(path) as data:
                    X=data['X_train'];b=data['y_train'];bandwidth=float(data['bandwidth'])
                variants={}
                for spelling in ['0.1','1e-05']:
                    alternate=ROOT/'data/prepared'/f'{stem}-s0-lambda{spelling}.npz'
                    if not alternate.exists():
                        variants[spelling]=dict(status='duplicate_prepared_file_absent',
                            source_basis='datasets.prepare regularization affects reference system only; preprocessing inputs and bandwidth independent of shift')
                        continue
                    with np.load(alternate) as data:
                        variants[spelling]=dict(X_train=bool(np.array_equal(X,data['X_train'])),
                            y_train=bool(np.array_equal(b,data['y_train'])),
                            bandwidth=bandwidth==float(data['bandwidth']))
                    assert all(variants[spelling].values()), 'Regularization-specific inputs differ'
                record['input_invariance_checks'][name]=variants
                X.setflags(write=False);b.setflags(write=False)
                cache[name]=(X,b,bandwidth,str(path.relative_to(ROOT)),sha256(path.read_bytes()).hexdigest())
            X,b,bandwidth,data_path,data_hash=cache[name]
            for method in methods:
                metadata=dict(dataset=name,n=len(b),regularization=lam,method=method,
                    m=0 if method=='plain_cg' else int(method[5:]),data_path=data_path,data_sha256=data_hash)
                if process_time()-cpu_started>=95:
                    record['unrun_cases'].append({**metadata,'status':'global_cpu_budget_not_started'})
                    record['status']='total_cpu_budget';save();continue
                run_start,run_cpu=perf_counter(),process_time()
                try:
                    if method=='plain_cg':
                        full,run=plain_cg(X,b,bandwidth,lam,cpu_cap=8.,deadline=cpu_started+96.)
                    else:
                        full,run=base.solve(X,b,bandwidth,metadata['m'],lam,cpu_cap=8.,deadline=cpu_started+96.)
                    del full
                    run.update(metadata)
                except (MemoryError,np.linalg.LinAlgError) as error:
                    run={**metadata,'status':'memory_failure' if isinstance(error,MemoryError) else 'numerical_failure',
                        'error':str(error),'failed_elapsed_seconds':perf_counter()-run_start,
                        'failed_cpu_seconds':process_time()-run_cpu}
                record['runs'].append(run);save()
                print(json.dumps({k:run[k] for k in ['dataset','n','regularization','method','status','iterations','total_seconds','true_relative_residual'] if k in run}),flush=True)
                if 'total_seconds' not in run:continue
                audit_key=(len(b),metadata['m'])
                if audit_key not in audits and process_time()-cpu_started<94:
                    audit_start,audit_cpu=perf_counter(),process_time()
                    try:
                        if method=='plain_cg':
                            tracemalloc.start()
                            _,traced=plain_cg(X,b,bandwidth,lam,maxiter=2)
                            current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                            audit=dict(tracemalloc_peak_bytes=peak,tracemalloc_current_bytes=current,
                                trace_within_1MiB=peak<=base.LIMIT,named=traced['memory'])
                            del _,traced
                        else:
                            audit=base.allocation_audit(X,b,bandwidth,metadata['m'],lam)
                    except (MemoryError,np.linalg.LinAlgError) as error:
                        if tracemalloc.is_tracing():tracemalloc.stop()
                        audit=dict(trace_within_1MiB=False,error=str(error))
                    audit.update(n=len(b),m=metadata['m'],dataset=name,regularization=lam,
                        audit_wall_seconds=perf_counter()-audit_start,audit_cpu_seconds=process_time()-audit_cpu,
                        timing_excluded=True)
                    audits[audit_key]=audit;record['allocation_audits'].append(audit)
                audit=audits.get(audit_key)
                run['resource_validated_by_separate_trace']=bool(audit and audit['trace_within_1MiB'])
                run['allocation_audit_key']=list(audit_key)
                run['resource_status']='within_declared_cap' if run['resource_validated_by_separate_trace'] else 'unvalidated_or_trace_exceeds_cap'
                save()
    if record['status']=='running':record['status']='complete'
    save()


if __name__=='__main__':main()
