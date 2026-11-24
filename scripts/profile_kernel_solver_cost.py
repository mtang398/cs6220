"""Instrument unchanged native solvers; profile times are not benchmark comparisons."""
import cProfile,pstats,json,hashlib,resource
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
ROOT=scope.ROOT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def profile_rows(stats,index):
    rows=[]
    for (filename,line,function),(primitive,total,selftime,cumulative,callers) in stats.stats.items():
        rows.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,total_calls=total,self_seconds=selftime,cumulative_seconds=cumulative))
    return sorted(rows,key=lambda r:r[index],reverse=True)[:30]

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
    out=ROOT/'results/kernel-solver-cost-profile.json';assert not out.exists()
    controls=[ROOT/'results/refresh-scope/smooth-lambda1e-05.json',ROOT/'results/refresh-scope/concrete-lambda0.001.json']
    files=scope.source_paths()+[Path(__file__).resolve()]+controls
    result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol='cProfile of unchanged native adaptive and packed24; identical shift binding and helpers. Instrumented CPU cap12 only. Profile wall/self/cumulative times diagnose cost and are not new performance comparisons.',runs=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    with threadpool_limits(limits=1):
        for dataset,lam,controlpath in [('smooth',1e-5,controls[0]),('Concrete',.001,controls[1])]:
            control=json.loads(controlpath.read_text());assert all(sha(ROOT/p)==h for p,h in control['sources'].items())
            datafile=ROOT/'data/prepared'/scope.DATASETS[dataset];assert sha(datafile)==control['data_sha256']
            with np.load(datafile) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            hashes=dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest());assert hashes==control['input_array_hashes']
            X.setflags(write=False);b.setflags(write=False)
            for method in ['adaptive','packed24']:
                module=scope.refresh if method=='adaptive' else scope.packed
                helper=scope.Helper(ROOT/'results/mixed_column_factor_first8.so') if method=='adaptive' else scope.Helper24(ROOT/'results/packed_int24_factor.so')
                solve,identity=scope.bind_shift(module,lam);profiler=cProfile.Profile();profiler.enable()
                try:
                    if method=='adaptive':answer,row=solve(X,b,bw,helper,mode='adaptive',cpu_cap=12)
                    else:answer,row=solve(X,b,bw,helper,cpu_cap=12)
                    del answer
                finally:profiler.disable()
                stats=pstats.Stats(profiler);prior=[r for r in control['runs'] if r['method']==method]
                checks={k:all(row[k]==p[k] for p in prior) for k in ['iterations','kernel_stats','original_relative_residual','converged']}
                checks['refresh_steps']=all(row.get('refresh_steps')==p.get('refresh_steps') for p in prior)
                record=dict(dataset=dataset,regularization=lam,method=method,data_sha256=control['data_sha256'],input_array_hashes=hashes,identity=identity,run=row,prior_checks=checks,profile_total_seconds=stats.total_tt,profile_total_calls=stats.total_calls,top30_self=profile_rows(stats,'self_seconds'),top30_cumulative=profile_rows(stats,'cumulative_seconds'))
                result['runs'].append(record);save();print(dataset,method,row['status'],checks,flush=True)
                assert all(checks.values()) and row['converged'] and row['original_relative_residual']<=1e-8
            assert hashes==dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest())
    result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
