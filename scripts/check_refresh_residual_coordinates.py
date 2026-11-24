"""Observer-only audit of residual coordinates in the actual-error refresh replay."""
import hashlib, inspect, json, resource
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
import check_actual_refresh_indicator as actual
from check_schur_inplace_frontier_v2 import ROOT, SharedKernel, Budget
scope=actual.scope

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def tiny():
    rng=np.random.default_rng(991);n,m=11,5
    F=rng.normal(size=(n,n));A=F@F.T+np.eye(n);C=A[:m,:m];B=A[:m,m:]
    L=np.linalg.cholesky(C)+np.tril(rng.normal(size=(m,m))*.001)
    H=cho_solve((L,True),np.eye(m));E=C-L@L.T
    T=np.eye(n);T[:m,m:]=-H@B
    G=T.T@A@T;U=np.vstack((np.eye(m),-B.T@H));Q=np.hstack((np.eye(m),-H@B))
    Gcheap=G-U@E@Q;b=rng.normal(size=n);y=rng.normal(size=n)
    r=T.T@b-Gcheap@y;s=Q@y
    mapped=np.r_[r[:m]-E@s,r[m:]+B.T@H@r[:m]]
    literal=b-A@(T@y)
    err=float(np.linalg.norm(mapped-literal)/np.linalg.norm(literal));assert err<1e-12
    return dict(relative_error=err,description='Dense SPD algebra check with arbitrary y; accumulated s=Qy before first refresh.')

def observer(lam,records):
    oldsolve,oldidentity=actual.actual_indicator_solver(lam,records)
    oldfactory=oldsolve.__globals__['Fused']
    text=inspect.getsource(scope.refresh.solve).replace('op.error_bound*np.linalg.norm(s)','op.actual_error_norm(s)',1)
    before='op.actual_error_norm(s)';after='op.audit_error_norm(s,r)'
    assert text.count(before)==1; modified=text.replace(before,after,1)
    assert modified.replace(after,before,1)==text
    def factory(X,b,bw,literal,helper):
        op=oldfactory(X,b,bw,literal,helper)
        E=inspect.getclosurevars(op.actual_error_norm).nonlocals['E']
        L=np.zeros((op.m,op.m))
        for j in range(op.m):
            offset=op.factor.offsets[j];source=op.factor.wide if op.factor.high[j] else op.factor.low
            L[j:,j]=source[offset:offset+op.m-j]
        kernel=SharedKernel(X,bw,Budget([X,b]));B=kernel.block(X[op.S],X[op.R])
        rec=records[-1];rec['coordinate_calls']=[];rec['additional_oracle_entries']=kernel.stats['entry_evaluations']
        previous=op.actual_error_norm
        def audit(s,r):
            value=previous(s)
            mapped=np.r_[r[:op.m]-E@s,r[op.m:]+B.T@cho_solve((L,True),r[:op.m],check_finite=False)]
            nt=float(np.linalg.norm(r));no=float(np.linalg.norm(mapped))
            rec['coordinate_calls'].append(dict(call=len(rec['coordinate_calls'])+1,error_norm=value,transformed_norm=nt,mapped_original_norm=no,norm_ratio=no/nt if nt else None,current_indicator_trigger=value>.1*nt,mapped_indicator_trigger=value>.1*no))
            return value
        op.audit_error_norm=audit;return op
    namespace=dict(oldsolve.__globals__);namespace['Fused']=factory
    exec(compile(modified,'<observer-only-coordinate-audit>','exec'),namespace)
    return namespace['solve'],dict(replacement_count=1,before=before,after=after,reverse_replacement_recovers_prior=True,prior_identity=oldidentity)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
    out=ROOT/'results/refresh-residual-coordinates.json';assert not out.exists()
    files=scope.source_paths()+[Path(__file__).resolve(),Path(actual.__file__).resolve()]
    for name in ['smooth','mixed','concrete','airfoil']:files.append(ROOT/f'results/actual-refresh-indicator/{name}-lambda1e-05.json')
    result=dict(status='running',sources={str(p.relative_to(ROOT)):sha(p) for p in files},protocol='Observer only: actual error return unchanged; dense original-coordinate mapping and all reference work excluded from native counters. No changed policy, native timing, or 1MiB claim.',cases=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        result['tiny_mapping_validation']=tiny();save()
        helper=scope.Helper(ROOT/'results/mixed_column_factor_first8.so')
        for name in ['smooth','mixed','Concrete','Airfoil']:
            prior=json.loads((ROOT/f'results/actual-refresh-indicator/{name.lower()}-lambda1e-05.json').read_text())
            assert all(sha(ROOT/p)==h for p,h in prior['sources'].items())
            path=ROOT/'data/prepared'/scope.DATASETS[name];assert sha(path)==prior['data_sha256']
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            assert dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest())==prior['input_array_hashes']
            X.setflags(write=False);b.setflags(write=False);records=[];solve,identity=observer(1e-5,records)
            answer,row=solve(X,b,bw,helper,mode='adaptive',cpu_cap=7);del answer
            checks={key:row[key]==prior['run'][key] for key in ['iterations','refresh_steps','kernel_stats','original_relative_residual','converged']}
            checks['indicator_values']=records[0]['indicator_calls']==prior['oracle'][0]['indicator_calls']
            calls=records[0]['coordinate_calls']; ratios=[c['norm_ratio'] for c in calls if c['norm_ratio'] is not None]
            summary=dict(changed_indicator_calls=[c['call'] for c in calls if c['current_indicator_trigger']!=c['mapped_indicator_trigger']],min_ratio=min(ratios),max_ratio=max(ratios))
            result['cases'].append(dict(dataset=name,run=row,oracle=records,identity=identity,replay_checks=checks,summary=summary));save()
            print(name,checks,summary,flush=True)
            assert all(checks.values()) and row['converged']
    result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged']
    result['status']='complete';save()
if __name__=='__main__':main()
