"""Free actual accumulated-error indicator in the frozen native refresh policy."""
import argparse
import hashlib
import inspect
import json
import resource
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
from check_schur_inplace_frontier_v2 import ROOT, SharedKernel, Budget


def actual_indicator_solver(lam, oracle_records):
    module=scope.refresh
    original=module.solve
    text=inspect.getsource(original)
    before='op.error_bound*np.linalg.norm(s)'
    after='op.actual_error_norm(s)'
    assert text.count(before)==1 and after not in text
    modified=text.replace(before,after,1)
    assert modified.replace(after,before,1)==text
    original_class=original.__globals__['Fused']
    def factory(X,b,bw,old_literal,helper):
        assert old_literal==1e-5
        op=original_class(X,b,bw,lam,helper)
        assert op.lam==lam and op.m==592
        # Separate oracle instance: these queries never enter solver kernel stats.
        kernel=SharedKernel(X,bw,Budget([X,b]))
        points=X[op.S]
        C=kernel.block(points,points)
        C.flat[::op.m+1]+=lam
        L=np.zeros((op.m,op.m))
        for j in range(op.m):
            offset=op.factor.offsets[j]
            source=op.factor.wide if op.factor.high[j] else op.factor.low
            L[j:,j]=source[offset:offset+op.m-j]
        E=C-L@L.T
        record=dict(core=op.m,regularization=op.lam,
            excluded_oracle_kernel_entries=kernel.stats['entry_evaluations'],
            retained_dense_error_bytes=E.nbytes,
            decoded_factor_sha256=hashlib.sha256(L.tobytes()).hexdigest(),
            selected_indices_sha256=hashlib.sha256(op.S.tobytes()).hexdigest(),
            error_frobenius=float(np.linalg.norm(E)),
            prior_factor_bound=float(op.error_bound),indicator_calls=[])
        oracle_records.append(record)
        bound=float(op.error_bound)
        def actual_error_norm(vector):
            value=float(np.linalg.norm(E@vector))
            upper=bound*float(np.linalg.norm(vector))
            record['indicator_calls'].append(dict(actual_norm=value,previous_bound_norm=upper))
            return value
        op.actual_error_norm=actual_error_norm
        return op
    private=dict(original.__globals__)
    private['Fused']=factory
    exec(compile(modified,'<frozen-refresh-with-actual-indicator>','exec'),private)
    assert original.__globals__['Fused'] is original_class and module.solve is original
    return private['solve'],dict(
        original_solve_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        modified_solve_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        replacement_count=1,replaced=before,replacement=after,
        reverse_replacement_recovers_original=True,original_module_unchanged=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=list(scope.DATASETS),required=True)
    parser.add_argument('--lam',type=float,choices=scope.SHIFTS,required=True)
    args=parser.parse_args()
    folder=ROOT/'results/actual-refresh-indicator';folder.mkdir(exist_ok=True)
    filename=f'{args.dataset.lower()}-lambda{args.lam:g}.json'
    out=folder/filename
    if out.exists():raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));started,cpu=perf_counter(),process_time()
    control_path=ROOT/'results/refresh-scope'/filename
    control=json.loads(control_path.read_text())
    assert control['status']=='complete' and control['regularization']==args.lam
    source_checks={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in control['sources'].items()}
    assert all(source_checks.values())
    path=ROOT/'data/prepared'/scope.DATASETS[args.dataset]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==control['data_sha256']
    with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
    input_hashes=dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest())
    assert input_hashes==control['input_array_hashes']
    X.setflags(write=False);b.setflags(write=False)
    oracle=[];solve,identity=actual_indicator_solver(args.lam,oracle)
    helper=scope.Helper(ROOT/'results/mixed_column_factor_first8.so')
    files=scope.source_paths()+[Path(__file__).resolve(),control_path]
    rec=dict(status='running',dataset=args.dataset,regularization=args.lam,n=len(b),
        data_sha256=control['data_sha256'],input_array_hashes=input_hashes,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        prior_source_hash_checks=source_checks,function_change=identity,
        protocol=dict(policy='Frozen FIRST8core592 native accumulated-refresh-diagonal; replace bound*norm(s) by actual dense norm((C−LL^T)s) only. Same .1*norm(r) trigger, continuation, factor and original1e-8 acceptance.',
            oracle='Dense E from actual streamed factor decoding and exact SharedKernel arithmetic on a separate instance. Reference kernel queries and dense indicator work excluded from solver-entry comparison. Computed floating-point E, not a rigorous exact-real error certificate.',
            comparison='Actual native kernel entry_evaluations include factor setup, actions, refreshes, reconstruction, all attempted original checks and final verification. Compare frozen scope controls with verified source/data identity.',
            scope='One policy, no threshold sweep. Dense E violates1MiB and is free oracle assistance for entry headroom only. Native measured wall includes oracle overhead and is NOT a speedup result. No future sketch, storage, or novelty claim.',
            limits='Onecase/command;80/85CPU;external120wall/flock;7CPU per solve;BLAS1'),
        oracle=oracle,run=None,controls=[],comparisons=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    with threadpool_limits(limits=1):
        answer,row=solve(X,b,bw,helper,mode='adaptive',cpu_cap=7)
        del answer
    row.update(method='actual_error_indicator',timing_includes_free_oracle_work_not_speed_evidence=True)
    rec['run']=row
    assert len(oracle)==1
    if row.get('converged'):assert row['original_relative_residual']<=1e-8
    for method in scope.METHODS:
        old=[r for r in control['runs'] if r['method']==method]
        assert len(old)==2 and all(r['converged'] for r in old)
        entries=[r['kernel_stats']['entry_evaluations'] for r in old]
        assert entries[0]==entries[1]
        baseline=dict(method=method,entry_evaluations=entries[0],iterations=[r['iterations'] for r in old],
            refresh_steps=[r.get('refresh_steps') for r in old],source_result=str(control_path.relative_to(ROOT)))
        rec['controls'].append(baseline)
        rec['comparisons'].append(dict(method=method,entry_ratio=row['kernel_stats']['entry_evaluations']/entries[0]))
    rec['source_hashes_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rec['sources'].items())
    rec['input_hashes_unchanged']=(input_hashes==dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest()))
    assert rec['source_hashes_unchanged'] and rec['input_hashes_unchanged']
    rec['status']='complete';save()
    print(json.dumps(dict(dataset=args.dataset,regularization=args.lam,status=row['status'],iterations=row['iterations'],
        refresh_steps=row['refresh_steps'],entry_evaluations=row['kernel_stats']['entry_evaluations'],comparisons=rec['comparisons'])),flush=True)


if __name__=='__main__':main()
