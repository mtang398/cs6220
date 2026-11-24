"""Frozen native refresh policies across RHS/regularization; one case per command."""
import argparse
import hashlib
import inspect
import json
import resource
import sys
import tracemalloc
import types
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_accumulated_refresh_diagonal as refresh
import check_native_uniform_fused as uniform
import check_native_int24_fused as packed
import check_native_first8_schur as scalar
from check_core_precision_allocation import Helper
from check_packed_float_allocation import RawPointers
from check_packed_int24_factor import Helper24
from check_schur_inplace_frontier_v2 import ROOT, LIMIT

DATASETS = {
    'smooth':'synthetic-n2560-smooth-s0-lambda0.001.npz',
    'random':'synthetic-n2560-random-s0-lambda0.001.npz',
    'mixed':'synthetic-n2560-mixed-s0-lambda0.001.npz',
    'Concrete':'concrete-observed-s0-lambda1e-05.npz',
    'Airfoil':'airfoil-observed-s0-lambda1e-05.npz',
}
METHODS = ['adaptive','periodic8','uniform600','packed24','original_exact416']
SHIFTS = [.1,.001,1e-5]


def bind_shift(module, lam):
    """Reuse exact solve code in private globals; original module is untouched."""
    original = module.solve
    original_class = original.__globals__['Fused']
    def factory(X,b,bw,old_literal,helper):
        assert old_literal == 1e-5
        result = original_class(X,b,bw,lam,helper)
        assert result.lam == lam
        return result
    scope = dict(original.__globals__)
    scope['Fused'] = factory
    clone = types.FunctionType(original.__code__,scope,original.__name__,
                               original.__defaults__,original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    assert clone.__code__ is original.__code__
    assert original.__globals__['Fused'] is original_class
    return clone,dict(module=Path(module.__file__).name,
        original_solve_source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        identical_code_object=True,requested_constructor_shift=lam,
        original_constructor_literal=1e-5,original_module_globals_unchanged=True)


def source_paths():
    paths={Path(__file__).resolve()}
    scripts=(ROOT/'scripts').resolve()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.parent==scripts and p.suffix=='.py':paths.add(p)
    for filename in ['mixed_column_factor_first8.c','packed_int24_factor.c',
                     'packed_float_factor.c','first8_factor_product.c']:
        p=ROOT/'scripts'/filename
        if p.exists():paths.add(p)
    for filename in ['mixed_column_factor_first8.so','packed_float_factor.so',
                     'packed_int24_factor.so','first8_factor_product.so']:
        paths.add(ROOT/'results'/filename)
    return sorted(paths)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=list(DATASETS),required=True)
    parser.add_argument('--lam',type=float,choices=SHIFTS,required=True)
    args=parser.parse_args()
    folder=ROOT/'results/refresh-scope';folder.mkdir(exist_ok=True)
    out=folder/f'{args.dataset.lower()}-lambda{args.lam:g}.json'
    if out.exists():raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85))
    started,cpu=perf_counter(),process_time()
    path=ROOT/'data/prepared'/DATASETS[args.dataset]
    with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
    assert len(b)>675 and X.dtype==np.float64 and X.flags.c_contiguous
    X.setflags(write=False);b.setflags(write=False)
    input_hashes=dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest())
    refresh_solve,refresh_identity=bind_shift(refresh,args.lam)
    uniform_solve,uniform_identity=bind_shift(uniform,args.lam)
    packed_solve,packed_identity=bind_shift(packed,args.lam)
    mixed_helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
    raw_helper=RawPointers(ROOT/'results/packed_float_factor.so')
    packed_helper=Helper24(ROOT/'results/packed_int24_factor.so')
    files=source_paths()
    rec=dict(status='running',dataset=args.dataset,regularization=args.lam,n=len(b),
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_array_hashes=input_hashes,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        function_parameterization=[refresh_identity,uniform_identity,packed_identity],
        protocol=dict(seed=0,repeats=2,methods=METHODS,
            fixed_policy='FIRST8core592; diagonal cheap congruence; signed accumulated error; adaptive bound*norm(s)>.1*norm(r) or periodic8. Continued direction and original1e-8 acceptance unchanged.',
            controls='Frozen uniformFP32core600, packed24core675 exact congruence and original_exact416.',
            parameterization='Original solve code objects in private copied globals; only Fused constructor shift is rebound. Original modules unchanged. exact416 accepts explicit lambda.',
            order='Rotate five methods by dataset/shift case index; reverse the entire order on second repeat.',
            audits='Separate tracemalloc runs for adaptive andperiodic8 in everycase; not timed evidence. Everytiming retains native named-array+reserve guards. Staticcontrols retain existing allocation implementation/shapes; no new whole-trace claim for controls.',
            prior_static_allocation_evidence=['results/native-int24-fused.json','results/native-uniform-fused.json','results/native-first8-schur.json'],
            immutable_inputs='Same readonly X,b objects across all methods; hashes checked after all runs. Inputstorage excluded from owned1MiB as before.',
            scope='Larger-case regularization/RHS extension only: n2048smooth/random/mixed, Concrete andAirfoil. Originaln512 scope not covered by fixed cores and remains necessary if result is positive. No full-scope/adaptive novelty claim.',
            limits='Onecase percommand,80/85CPU,external120wall/flock,BLAS1,7CPU per solve; originalrelative1e-8.'),
        audits=[],runs=[],failures=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    def invoke(method):
        call_start,call_cpu=perf_counter(),process_time()
        try:
            if method in ['adaptive','periodic8']:
                answer,row=refresh_solve(X,b,bw,mixed_helper,mode=method,cpu_cap=7)
            elif method=='uniform600':answer,row=uniform_solve(X,b,bw,raw_helper,cpu_cap=7)
            elif method=='packed24':answer,row=packed_solve(X,b,bw,packed_helper,cpu_cap=7)
            else:answer,row=scalar.exact_control.run(X,b,bw,args.lam,'original_exact416',raw_helper,cpu_cap=7,deadline=cpu+78)
            del answer
        except (MemoryError,np.linalg.LinAlgError) as exc:
            row=dict(status='memory_failure' if isinstance(exc,MemoryError) else 'numerical_failure',
                     converged=False,error=repr(exc),total_seconds=perf_counter()-call_start,
                     total_cpu_seconds=process_time()-call_cpu)
        row['method']=method
        row['regularization']=args.lam
        relative=row.get('original_relative_residual',row.get('true_relative_residual'))
        row['verified_original_relative_residual']=relative
        if row.get('converged'):assert relative is not None and relative<=1e-8
        memory=row.get('memory',{})
        row['named_allocation_within_1MiB']=memory.get('peak_with_reserve_bytes',LIMIT+1)<=LIMIT
        if not row['named_allocation_within_1MiB']:
            rec['failures'].append(dict(method=method,reason='Named allocation absent or exceeds1MiB',status=row['status']))
        return row
    save()
    with threadpool_limits(limits=1):
        for method in ['adaptive','periodic8']:
            tracemalloc.start()
            try:
                row=invoke(method);_,peak=tracemalloc.get_traced_memory()
            finally:tracemalloc.stop()
            audit=dict(method=method,traced_peak_bytes=peak,within_1MiB=peak<=LIMIT,
                       metrics=row,excluded_from_timing_comparison=True)
            rec['audits'].append(audit);save()
            if not audit['within_1MiB'] or row['status']=='memory_failure':
                rec['status']='allocation_gate_failed';save();return
        index=list(DATASETS).index(args.dataset)*len(SHIFTS)+SHIFTS.index(args.lam)
        order=METHODS[index%len(METHODS):]+METHODS[:index%len(METHODS)]
        for repeat in range(2):
            for method in (order if repeat==0 else list(reversed(order))):
                if process_time()-cpu>70:
                    rec['status']='command_cpu_budget';save();return
                assert not tracemalloc.is_tracing()
                row=invoke(method);row['repeat']=repeat;rec['runs'].append(row);save()
                print(json.dumps({key:row.get(key) for key in ['method','repeat','status','iterations','total_seconds','refresh_steps','verified_original_relative_residual']}),flush=True)
    rec['input_hashes_unchanged']=(input_hashes==dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest()))
    assert rec['input_hashes_unchanged']
    rec['source_hashes_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in rec['sources'].items())
    assert rec['source_hashes_unchanged']
    rec['status']='complete';save()


if __name__=='__main__':main()
