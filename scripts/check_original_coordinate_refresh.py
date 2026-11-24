"""Free original-coordinate trigger gate; frozen threshold and solver recurrence."""
import hashlib,inspect,json,resource
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_refresh_residual_coordinates as audit
scope=audit.scope;ROOT=audit.ROOT;sha=audit.sha

def changed_solver(records):
    observer,identity=audit.observer(1e-5,records)
    original_factory=observer.__globals__['Fused']
    text=inspect.getsource(scope.refresh.solve)
    before='op.error_bound*np.linalg.norm(s)>.1*np.linalg.norm(r)';after='op.coordinate_refresh(s,r)'
    assert text.count(before)==1
    modified=text.replace(before,after,1);assert modified.replace(after,before,1)==text
    def factory(*args):
        op=original_factory(*args)
        def trigger(s,r):
            op.audit_error_norm(s,r)
            return records[-1]['coordinate_calls'][-1]['mapped_indicator_trigger']
        op.coordinate_refresh=trigger;return op
    namespace=dict(observer.__globals__);namespace['Fused']=factory
    exec(compile(modified,'<original-coordinate-trigger>','exec'),namespace)
    return namespace['solve'],dict(before=before,after=after,replacement_count=1,reverse_recovers_original=True,observer_identity=identity)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
    out=ROOT/'results/original-coordinate-refresh.json';assert not out.exists()
    files=scope.source_paths()+[Path(__file__).resolve(),Path(audit.__file__).resolve(),Path(audit.actual.__file__).resolve(),ROOT/'results/refresh-residual-coordinates.json']
    for name in ['smooth','mixed','concrete','airfoil']:
        files.extend([ROOT/f'results/actual-refresh-indicator/{name}-lambda1e-05.json',ROOT/f'results/refresh-scope/{name}-lambda1e-05.json'])
    result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol='One changed trigger: ||Es|| > .1 ||mapped original residual||. Same factor, core, recurrence, and stopping. All dense oracle construction/mapping excluded from native kernel counts; no timing or memory feasibility claim.',cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['tiny_mapping_validation']=audit.tiny();save();helper=scope.Helper(ROOT/'results/mixed_column_factor_first8.so')
        for name in ['smooth','mixed','Concrete','Airfoil']:
            prior=json.loads((ROOT/f'results/actual-refresh-indicator/{name.lower()}-lambda1e-05.json').read_text())
            assert all(sha(ROOT/p)==h for p,h in prior['sources'].items())
            path=ROOT/'data/prepared'/scope.DATASETS[name];assert sha(path)==prior['data_sha256']
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            assert dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest())==prior['input_array_hashes']
            X.setflags(write=False);b.setflags(write=False);records=[];solve,identity=changed_solver(records)
            answer,row=solve(X,b,bw,helper,mode='adaptive',cpu_cap=7);del answer
            controls=prior['controls']+[dict(method='actual_transformed_trigger',entry_evaluations=prior['run']['kernel_stats']['entry_evaluations'],iterations=[prior['run']['iterations']],refresh_steps=[prior['run']['refresh_steps']])]
            comparisons=[dict(method=c['method'],entry_ratio=row['kernel_stats']['entry_evaluations']/c['entry_evaluations']) for c in controls]
            result['cases'].append(dict(dataset=name,run=row,oracle=records,identity=identity,controls=controls,comparisons=comparisons));save()
            print(name,row['status'],row['iterations'],row['refresh_steps'],comparisons,flush=True)
            if row['converged']:assert row['original_relative_residual']<=1e-8
    result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged']
    result['status']='complete';save()
if __name__=='__main__':main()
