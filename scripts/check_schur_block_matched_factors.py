"""Repeat the block oracle with balanced groups filling the factor allowance.

The first screen's size thresholds left substantial storage unused. This
comparison removes that confound. Dense setup/workspaces remain uncharged:
this is explicitly a factor-storage/action-count oracle, not a feasible solver.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_schur_block_headroom as base


def balanced(X, indices, groups):
    if groups == 1:
        return [indices]
    points = X[indices]
    axis = int(np.argmax(np.ptp(points, axis=0)))
    ordered = indices[np.argsort(points[:,axis], kind='stable')]
    left = groups//2
    cut = len(indices)*left//groups
    return balanced(X, ordered[:cut], left)+balanced(X, ordered[cut:], groups-left)


def main():
    output = base.ROOT/'results/schur-block-matched-factors.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (110,115))
    start, cpu = perf_counter(), process_time()
    base.partition = balanced
    record = dict(status='running', protocol=dict(
        seed=0, datasets=['smooth','mixed','Concrete'], shifts=[.001,1e-5],
        core_counts=[160,64,128], packed_factor_allowance_bytes=103040,
        scope='Dense static factor-storage-matched oracle; setup and workspace uncharged',
        grouping='Minimum balanced geometric group count fitting packed factor allowance',
        original_rtol=1e-8, reduced_rtol=1e-10),
        sources={str(p.relative_to(base.ROOT)):sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__),Path(base.__file__)]}, runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for name in record['protocol']['datasets']:
            stem = 'concrete-observed' if name=='Concrete' else f'synthetic-n640-{name}'
            path = base.ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:
                X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
            Smax=base.selected(K,160)
            for lam in record['protocol']['shifts']:
                A=K+lam*np.eye(len(K))
                for m in record['protocol']['core_counts']:
                    groups=0
                    if m!=160:
                        remainder=len(A)-m
                        for groups in range(1,remainder+1):
                            small,large_count=divmod(remainder,groups)
                            entries=(groups-large_count)*small*(small+1)//2+large_count*(small+1)*(small+2)//2
                            if 8*(entries+m*(m+1)//2)<=103040: break
                    row=dict(dataset=name,regularization=lam,core_count=m,group_count=groups,
                             data_sha256=sha256(path.read_bytes()).hexdigest())
                    row.update(base.run(A,X,b,Smax[:m],groups))
                    assert row['packed_factor_bytes']<=103040
                    record['runs'].append(row);save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__': main()
