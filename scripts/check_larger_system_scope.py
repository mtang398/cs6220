"""Extend, rather than replace, the original development-size controls.

One setup-inclusive CPU observation at n_train=2048 and the existing 1MiB
working allowance. This is a static bottleneck screen, not adaptive evidence.
"""
import json
import resource
import sys
import tracemalloc
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
from check_packed_schur_feasibility import PackedSchur,base,ROOT
from check_static_schur_scope import plain_cg

sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.datasets import prepare


class UpperPacked(PackedSchur):
    def __init__(self,kernel,m,lam,*,indices=None):
        selected=kernel.farthest(m) if indices is None else indices
        super().__init__(kernel,m,lam,indices=selected,lower=False)


def main():
    output=ROOT/'results/larger-system-static-scope.json'
    if output.exists():raise FileExistsError(output)
    for target in ['smooth','random','mixed']:
        path=ROOT/f'data/prepared/synthetic-n2560-{target}-s0-lambda0.001.npz'
        if path.exists():raise FileExistsError('Inspect existing input before reuse: '+str(path))
    resource.setrlimit(resource.RLIMIT_CPU,(158,164))
    base.InplaceSchur=UpperPacked
    start,cpu=perf_counter(),process_time()
    sources=[Path(__file__),ROOT/'src/nystrom_project/datasets.py',
             ROOT/'scripts/check_packed_schur_feasibility.py',
             ROOT/'scripts/check_schur_inplace_frontier_v2.py',
             ROOT/'scripts/check_static_schur_scope.py']
    record=dict(status='running',protocol=dict(seed=0,total_synthetic_n=2560,n_train=2048,
        datasets=['smooth','random','mixed'],regularizations=[.001,1e-5],
        methods_by_shift={'0.001':[0,320,384,448],'1e-05':[384,448]},
        working_allowance=1048576,reserve_bytes=65536,common_inputs_separate=True,
        repeats=1,per_solve_cpu_cap=6.,campaign_cpu_cap=150.,
        preparation='Existing generator/preprocessing; prepare lambda.001; training values unchanged for other shift',
        protected='Original sizes, data and measurements retained; confirmation seeds untouched',
        scope='Static setup-inclusive CPU scale extension, no new adaptive method or speedup claim'),
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        preparation=[],runs=[],audits=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for target in record['protocol']['datasets']:
            path,config=prepare(ROOT,'synthetic',0,.001,target=target,synthetic_n=2560)
            record['preparation'].append(dict(path=str(path.relative_to(ROOT)),
                sha256=sha256(path.read_bytes()).hexdigest(),config=config));save()
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            assert len(b)==2048
            for lam,counts in [(.001,[0,320,384,448]),(1e-5,[384,448])]:
                for m in counts:
                    if process_time()-cpu>145:
                        record['status']='campaign_cpu_cap';save();return
                    row=dict(dataset=target,regularization=lam,core_count=m,
                             data_sha256=sha256(path.read_bytes()).hexdigest())
                    assert not tracemalloc.is_tracing()
                    try:
                        if m:
                            x,result=base.solve(X,b,bw,m,lam,cpu_cap=6.,deadline=cpu+150)
                        else:
                            x,result=plain_cg(X,b,bw,lam,cpu_cap=6.,deadline=cpu+150)
                        row.update(result);del x
                    except (MemoryError,np.linalg.LinAlgError) as error:
                        row.update(status='resource_or_numerical_failure',error=str(error))
                    record['runs'].append(row);save()
            if target=='smooth':
                for m in [320,384,448]:
                    try:
                        audit=base.allocation_audit(X,b,bw,m,.001)
                        audit.update(dataset=target,core_count=m)
                    except (MemoryError,np.linalg.LinAlgError) as error:
                        audit=dict(dataset=target,core_count=m,error=str(error))
                    finally:
                        tracemalloc.stop()
                    record['audits'].append(audit);save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
