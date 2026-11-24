"""Setup-inclusive fixed-memory pilot; preserve every failure and capped run."""
import json
import resource
import tracemalloc
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_mixed_precision_refinement as candidate
from check_packed_schur_feasibility import PackedSchur,base,ROOT


class UpperPacked(PackedSchur):
    def __init__(self,kernel,m,lam,*,indices=None):
        selected=kernel.farthest(m) if indices is None else indices
        super().__init__(kernel,m,lam,indices=selected,lower=False)


def main():
    output=ROOT/'results/mixed-precision-refinement-pilot.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(158,164))
    base.InplaceSchur=UpperPacked
    start,cpu=perf_counter(),process_time()
    methods=['exact384','exact416','fixed_0.1','fixed_0.01','fixed_0.001','adaptive']
    source_paths=[Path(__file__),Path(candidate.__file__),ROOT/'scripts/check_packed_schur_feasibility.py',
                  ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    record=dict(status='running',protocol=dict(datasets=['smooth2048','mixed2048'],
        seed=0,regularizations=[.001,1e-5],methods=methods,repeats=1,
        candidate_core=384,working_allowance=1048576,reserve_bytes=65536,
        common_inputs='Same immutable FP64 X,b separately reported',
        per_solve_cpu_cap=6.,total_cpu_cap=150.,rtol=1e-8,
        order='Cyclic method order across four cases',
        scope='Native development pilot; original small/real cases remain required',
        controls='Fixed tolerances omit adaptive diagnostic low-product; all final accuracy is original FP64'),
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths},
        checks={},audits=[],runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        record['checks']['tiny_preflight']=dict(
            executed_by='slide_fidelity before pilot; unchanged source',
            source_sha256=sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
            low_Schur_action_relative_error=3.46e-16,
            maximum_reported_original_relative_residual=4.40e-11,
            low_kernel_bitwise_symmetric=True,low_kernel_bitwise_block_consistent=True)
        path=ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
        with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
        audit=candidate.allocation_audit(X,b,bw,384,.001,'adaptive')
        record['audits'].append(audit);save()
        if audit['status']!='complete' or not audit['trace_within_1MiB']:
            record['status']='allocation_gate_failed';save();return
        ci=0
        for name in ['smooth','mixed']:
            path=ROOT/f'data/prepared/synthetic-n2560-{name}-s0-lambda0.001.npz'
            with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
            for lam in [.001,1e-5]:
                ordered=methods[ci:]+methods[:ci];ci+=1
                for method in ordered:
                    if process_time()-cpu>143:
                        record['status']='campaign_cpu_cap';save();return
                    row=dict(dataset=name+'2048',regularization=lam,method=method,
                             data_sha256=sha256(path.read_bytes()).hexdigest())
                    assert not tracemalloc.is_tracing()
                    try:
                        if method.startswith('exact'):
                            m=int(method[5:])
                            x,stats=base.solve(X,b,bw,m,lam,cpu_cap=6.,deadline=cpu+150)
                        else:
                            x,stats=candidate.solve(X,b,bw,384,lam,method,cpu_cap=6.)
                        row.update(stats);del x
                    except (MemoryError,np.linalg.LinAlgError) as error:
                        row.update(status='resource_or_numerical_failure',error=str(error))
                    record['runs'].append(row);save()
        record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
