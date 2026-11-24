#!/usr/bin/env python3
"""Price frozen inexact-CG histograms; no solver rerun or end-to-end claim."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter,process_time

import numpy as np
from scipy.special import gammainc
from threadpoolctl import threadpool_limits
from check_taylor_kernel_cost import Taylor,ROOT
from check_kernel_precision_cost import Product,TILE_BYTES
from check_taylor_selection_cost import select

DIAGNOSTIC=ROOT/'results/taylor-inexact-cg-diagnostic.json'


def checked_records():
    checks={}
    names=['taylor-inexact-cg-diagnostic','taylor-krylov-cost-oracle','taylor-selection-cost','taylor-kernel-cost']
    records={}
    for name in names:
        path=ROOT/'results'/f'{name}.json';record=json.loads(path.read_text())
        assert record['status']=='complete'
        pairs=dict(record.get('sources',{}))
        if 'source_sha256' in record:
            pairs[f'scripts/check_{name.replace("-","_")}.py']=record['source_sha256']
        matches={p:sha256((ROOT/p).read_bytes()).hexdigest()==expected for p,expected in pairs.items()}
        assert all(matches.values()),(name,matches)
        checks[name]=matches;records[name]=record
    return records,checks


def bound(s,v,q):
    root=np.sqrt(gammainc(q+1,s));absolute=np.abs(v)
    return float(np.linalg.norm(root*(root@absolute-root*absolute)))


def weight_rows(frozen,measurements,gaussians):
    prices={(r['dataset'],r['degree']):r['median_seconds'] for r in measurements}
    weighted=[]
    for row in frozen['runs']:
        ds=row['dataset'];g=gaussians[ds]
        if row['method']=='exact_A':
            action=row['iterations']*g['fastest_action_seconds'];selection=0.
        else:
            action=sum(int(count)*prices[(ds,int(q))]['taylor'] for q,count in row['degree_histogram'].items())
            selection=(sum(int(count)*prices[(ds,int(q))]['select'] for q,count in row['degree_histogram'].items())
                       if row['method']=='adaptive' else 0.)
        verification=row['original_reference_checks']*g['median_seconds']['cdist64']
        weighted.append(dict(dataset=ds,regularization=row['regularization'],method=row['method'],
            iterations=row['iterations'],degree_histogram=row['degree_histogram'],
            original_reference_checks=row['original_reference_checks'],
            true_relative_residual=row['true_relative_residual'],
            weighted_action_seconds=action,weighted_selector_estimate_seconds=selection,
            weighted_action_plus_selection_seconds=action+selection,
            weighted_literal_verification_seconds=verification,
            weighted_priced_total_seconds=action+selection+verification))
    comparisons=[]
    for ds,lam in sorted({(r['dataset'],r['regularization']) for r in weighted}):
        rows={r['method']:r for r in weighted if r['dataset']==ds and r['regularization']==lam}
        adaptive=rows['adaptive']['weighted_priced_total_seconds']
        replay=rows['static_replay']['weighted_priced_total_seconds']
        uniform=rows['static_uniform_beta']['weighted_priced_total_seconds']
        exact=rows['exact_A']['weighted_priced_total_seconds']
        comparisons.append(dict(dataset=ds,regularization=lam,
            adaptive_estimated_seconds=adaptive,static_replay_estimated_seconds=replay,
            static_uniform_estimated_seconds=uniform,exact_estimated_seconds=exact,
            adaptive_estimated_speedup_vs_replay_static=replay/adaptive,
            adaptive_estimated_speedup_vs_uniform_static=uniform/adaptive,
            adaptive_estimated_speedup_vs_exact=exact/adaptive,
            adaptive_action_only_estimated_speedup_vs_replay_static=rows['static_replay']['weighted_action_seconds']/rows['adaptive']['weighted_action_seconds'],
            adaptive_selector_estimate_fraction=rows['adaptive']['weighted_selector_estimate_seconds']/adaptive))
    return weighted,comparisons


def main():
    output=ROOT/'results/taylor-inexact-cg-pricing.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(20,25))
    started,cpu_started=perf_counter(),process_time()
    records,checks=checked_records();frozen=records['taylor-inexact-cg-diagnostic']
    library=ctypes.CDLL(str(ROOT/'results/taylor_kernel_tile.so'));function=library.taylor_tile
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    function.argtypes=[array,array,array,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]
    function.restype=None
    source_paths=[Path(__file__),DIAGNOSTIC,ROOT/'scripts/check_taylor_selection_cost.py',
        ROOT/'scripts/check_taylor_kernel_cost.py',ROOT/'scripts/check_kernel_precision_cost.py',
        ROOT/'scripts/taylor_kernel_tile.c',ROOT/'results/taylor_kernel_tile.so']
    result=dict(status='running',source_checks=checks,
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths},
        protocol=dict(repeats=5,tile_byte_allowance=TILE_BYTES,
            action='Existing native compiled corrected Taylor and cdist64/gram64 generators; combined Gram/output Taylor tiles share128KiB.',
            order='For each observed degree, five cyclic repetitions of select/Taylor/cdist64/gram64; offset varies with degree and dataset.',
            vector='Prepared alpha_reference at lambda.001 for standalone actions and selector-cost estimates.',
            selection='Existing lean binary search0..40; threshold is midpoint of current standalone vector bounds at q and q−1, asserted to returnq.',
            gaussian_price='Per dataset, median pooled observations for each Gaussian mode; exact actions use faster mode, literal verification uses cdist64.',
            weighting='Frozen actual diagnostic degree histograms. Selector estimate charged only to adaptive; static controls have none.',
            excluded='No solver rerun. Setup, degree initialization, preconditioning, dot products/vector work, allocator effects of a whole solver, and unmeasured actual-search-direction effects are excluded.',
            fairness='Fixed controls stopped at recursive0.5tol; adaptive stopped at recursive+bound<=tol. This gives adaptive an iteration-count advantage; every frozen run passed finaltol.',
            scope='Standalone action-plus-selector cost estimates, not actual selector runtimes along CG, total-memory compliance, or end-to-end speedups.',
            cpu_cap_seconds=20,lock='/tmp/cs6220-project-benchmark.lock'),
        measurements=[],gaussian_prices={},weighted_runs=[],comparisons=[])
    def save():
        result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    paths=[('smooth512','synthetic-n640-smooth'),('mixed512','synthetic-n640-mixed'),('Concrete824','concrete-observed')]
    with threadpool_limits(limits=1):
        for ci,(dataset,stem) in enumerate(paths):
            path=ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:X,v,bw=data['X_train'],data['alpha_reference'],float(data['bandwidth'])
            s=np.einsum('ij,ij->i',X,X)/(bw*bw)
            degrees=sorted({int(q) for r in frozen['runs'] if r['dataset']==dataset for q in r['degree_histogram']})
            exact={name:Product(X,v,bw,name) for name in ['cdist64','gram64']}
            for op in exact.values():op.apply()
            gaussian_samples={name:[] for name in exact}
            for qi,q in enumerate(degrees):
                threshold=(bound(s,v,q)+bound(s,v,q-1))/2
                observed,calls=select(s,v,threshold);assert observed==q
                op=Taylor(X,v,bw,q,True,function);op.apply()
                samples={name:[] for name in ['select','taylor','cdist64','gram64']}
                for rep in range(5):
                    operations=[('select',lambda:select(s,v,threshold)),('taylor',op.apply),
                        ('cdist64',exact['cdist64'].apply),('gram64',exact['gram64'].apply)]
                    offset=(ci+qi+rep)%len(operations)
                    for name,fn in operations[offset:]+operations[:offset]:
                        tick=perf_counter();answer=fn();elapsed=perf_counter()-tick
                        samples[name].append(elapsed)
                        if name=='select':assert answer[0]==q
                        del answer
                for name in exact:gaussian_samples[name].extend(samples[name])
                result['measurements'].append(dict(dataset=dataset,n=len(X),degree=q,
                    data_sha256=sha256(path.read_bytes()).hexdigest(),selector_returned_degree=observed,
                    selector_bound_evaluations=calls,forced_standalone_threshold=float(threshold),
                    Taylor_context_bytes=op.extra_bytes,Taylor_rows=op.rows,
                    Taylor_two_tile_bytes=16*op.rows*len(X),
                    Gaussian_rows={name:obj.rows for name,obj in exact.items()},
                    seconds=samples,median_seconds={name:float(np.median(a)) for name,a in samples.items()}))
                del op;save()
            med={name:float(np.median(samples)) for name,samples in gaussian_samples.items()}
            fastest=min(med,key=med.get)
            result['gaussian_prices'][dataset]=dict(median_seconds=med,
                fastest_mode=fastest,fastest_action_seconds=med[fastest],
                sample_counts={name:len(a) for name,a in gaussian_samples.items()})
            save()
            print(json.dumps(dict(dataset=dataset,degrees=degrees,measurements=len(degrees))),flush=True)
    result['weighted_runs'],result['comparisons']=weight_rows(frozen,result['measurements'],result['gaussian_prices'])
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['sources'].items())
    result['status']='complete';save()


if __name__=='__main__':main()
