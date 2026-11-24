"""Matched common-kernel accounting optimization with separate allocation audits."""
import argparse,contextlib,hashlib,inspect,json,resource,textwrap,tracemalloc
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
from check_schur_inplace_frontier_v2 import SharedKernel,ROOT,LIMIT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ORIGINAL=SharedKernel.product
TEXT=textwrap.dedent(inspect.getsource(ORIGINAL))
BEFORE='        self.budget.observe(label,*extra,columns,result,vector,points,tile,product)'
AFTER='        if start==0:self.budget.observe(label,*extra,columns,result,vector,points,tile,product)'
assert TEXT.count(BEFORE)==1
MODIFIED=TEXT.replace(BEFORE,AFTER,1)
assert MODIFIED.replace(AFTER,BEFORE,1)==TEXT
PRIVATE=dict(ORIGINAL.__globals__);exec(compile(MODIFIED,'<common-product-accounting-hoist>','exec'),PRIVATE)
HOISTED=PRIVATE['product']
@contextlib.contextmanager
def variant(name):
    assert SharedKernel.product is ORIGINAL
    SharedKernel.product=HOISTED if name=='hoisted' else ORIGINAL
    try:yield
    finally:SharedKernel.product=ORIGINAL

def comparable(row):
    memory=row['memory']
    return dict(iterations=row['iterations'],converged=row['converged'],kernel_stats=row['kernel_stats'],refresh_steps=row.get('refresh_steps'),relative=row.get('original_relative_residual',row.get('true_relative_residual')),memory=memory)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['smooth-hard','concrete-medium'],required=True);args=parser.parse_args()
    dataset,lam=('smooth',1e-5) if args.case=='smooth-hard' else ('Concrete',.001)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start=perf_counter();cpu=process_time()
    folder=ROOT/'results/kernel-accounting-hoist';folder.mkdir(exist_ok=True);out=folder/(args.case+'.json');assert not out.exists()
    priorpath=ROOT/f'results/refresh-scope/{dataset.lower()}-lambda{lam:g}.json';prior=json.loads(priorpath.read_text());assert all(sha(ROOT/p)==h for p,h in prior['sources'].items())
    datapath=ROOT/'data/prepared'/scope.DATASETS[dataset];assert sha(datapath)==prior['data_sha256']
    with np.load(datapath) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
    hashes=dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest());assert hashes==prior['input_array_hashes'];X.setflags(write=False);b.setflags(write=False)
    refresh,_=scope.bind_shift(scope.refresh,lam);uniform,_=scope.bind_shift(scope.uniform,lam);packed,_=scope.bind_shift(scope.packed,lam)
    mixedhelper=scope.Helper(ROOT/'results/mixed_column_factor_first8.so');raw=scope.RawPointers(ROOT/'results/packed_float_factor.so');h24=scope.Helper24(ROOT/'results/packed_int24_factor.so')
    files=scope.source_paths()+[Path(__file__).resolve(),priorpath]
    result=dict(status='running',dataset=dataset,regularization=lam,data_sha256=sha(datapath),input_array_hashes=hashes,sources={str(p.relative_to(ROOT)):sha(p) for p in files},replacement=dict(before=BEFORE,after=AFTER,count=1,reverse_recovers_original=True,original_sha256=hashlib.sha256(TEXT.encode()).hexdigest(),modified_sha256=hashlib.sha256(MODIFIED.encode()).hexdigest()),protocol='Common SharedKernel.product patch applies to all five methods, restored in finally. Only first tile observes accounting. All retained arrays, scopes, rows plan constant inside product; subsequent tile rows never exceed first; previous tile deleted. Two repeats, reversed method and variant orders. Five separate hoisted whole-trace audits. Native CPU7, trace CPU8. No scientific method or success claim.',audits=[],runs=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def invoke(method,name,cap):
        with variant(name):
            if method in ['adaptive','periodic8']:answer,row=refresh(X,b,bw,mixedhelper,mode=method,cpu_cap=cap)
            elif method=='uniform600':answer,row=uniform(X,b,bw,raw,cpu_cap=cap)
            elif method=='packed24':answer,row=packed(X,b,bw,h24,cpu_cap=cap)
            else:answer,row=scope.scalar.exact_control.run(X,b,bw,lam,'original_exact416',raw,cpu_cap=cap,deadline=cpu+78)
            del answer
        row.update(method=method,variant=name)
        relative=row.get('original_relative_residual',row.get('true_relative_residual'))
        assert row['converged'] and relative<=1e-8
        assert row['memory']['peak_with_reserve_bytes']<=LIMIT
        return row
    save()
    with threadpool_limits(limits=1):
        for method in scope.METHODS:
            tracemalloc.start()
            try:row=invoke(method,'hoisted',8);_,peak=tracemalloc.get_traced_memory()
            finally:tracemalloc.stop()
            result['audits'].append(dict(method=method,traced_peak_bytes=peak,within_1MiB=peak<=LIMIT,run=row));save()
            assert peak<=LIMIT
        order=list(scope.METHODS);index=0 if dataset=='smooth' else 1;order=order[index:]+order[:index]
        for repeat in range(2):
            for method in (order if repeat==0 else list(reversed(order))):
                pair=[]
                for name in (['old','hoisted'] if repeat==0 else ['hoisted','old']):
                    row=invoke(method,name,7);row['repeat']=repeat;result['runs'].append(row);pair.append(row);save()
                checks=comparable(pair[0])==comparable(pair[1])
                print(method,repeat,'exact_equivalence',checks,flush=True)
                assert checks
    result['sources_unchanged']=all(sha(ROOT/p)==h for p,h in result['sources'].items());result['inputs_unchanged']=hashes==dict(X=hashlib.sha256(X.tobytes()).hexdigest(),b=hashlib.sha256(b.tobytes()).hexdigest());assert result['sources_unchanged'] and result['inputs_unchanged'] and SharedKernel.product is ORIGINAL
    result['status']='complete';save()
if __name__=='__main__':main()
