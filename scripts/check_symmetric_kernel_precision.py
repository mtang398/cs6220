"""Fixed-entry low-precision kernel generation: consistency, SPD and cost.

Dense matrices and Cholesky workspaces belong only to the stability oracle.
Full products use at most128KiB for kernel/conversion tiles. New feature
copies, norms, output/index arrays and separate allocation traces are reported.
No generated matrix is averaged with its transpose or otherwise repaired.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter,process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.linalg import lapack
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
TILE_BYTES=131072
MODES=['cdist64','gram32_acc64','cdist64_exp32_acc64','direct32_acc64']
CASES=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),
       ('Concrete824','concrete-observed-s0-lambda0.001.npz'),
       ('Airfoil1202','airfoil-observed-s0-lambda0.001.npz'),
       ('smooth2048','synthetic-n2560-smooth-s0-lambda0.001.npz')]


def backing(a):
    while isinstance(a.base,np.ndarray):a=a.base
    return a


class Kernel:
    def __init__(self,X,v,bandwidth,mode,audit=False):
        self.mode,self.v,self.bw,self.audit=mode,v,float(bandwidth),audit
        self.common={id(backing(a)) for a in [X,v]}
        self.X=X.astype(np.float32) if mode in ['gram32_acc64','direct32_acc64'] else X
        self.norms=np.einsum('ij,ij->i',self.X,self.X) if mode=='gram32_acc64' else None
        self.scale64=-.5/(self.bw*self.bw);self.scale32=np.float32(self.scale64)
        self.n,self.dimension=X.shape
        self.entry_bytes=8 if mode=='cdist64' else 12
        self.rows=max(1,TILE_BYTES//(self.entry_bytes*self.n))
        self.extra_bytes=(0 if self.X is X else self.X.nbytes)+(0 if self.norms is None else self.norms.nbytes)
        self.live=[];self.named_peak=0;self.phase_peaks={};self.max_tile_live=0
        self.stats=dict(product_calls=0,entry_query_calls=0,kernel_blocks=0,
                        entry_evaluations=0,kernel_generation_seconds=0.)
        self.observe('context')

    def observe(self,label,*arrays,tile_bytes=0):
        self.max_tile_live=max(self.max_tile_live,int(tile_bytes))
        if not self.audit:return
        owners={}
        for a in [self.X,self.norms,*self.live,*arrays]:
            if a is None:continue
            root=backing(a)
            if id(root) not in self.common:owners[id(root)]=root.nbytes
        size=int(sum(owners.values()))
        self.named_peak=max(self.named_peak,size)
        self.phase_peaks[label]=max(self.phase_peaks.get(label,0),size)

    def raw(self,rows,columns,rownorm=None,colnorm=None,diagonal_start=None):
        before=perf_counter();self.stats['kernel_blocks']+=1
        self.stats['entry_evaluations']+=len(rows)*len(columns)
        if self.mode=='direct32_acc64':
            # All arrays and scalar operations in this distance loop are FP32.
            # The feature sequence is invariant to row/column tile shape.
            tile=np.zeros((len(rows),len(columns)),dtype=np.float32)
            difference=np.empty_like(tile)
            self.observe('direct_distance',rows,columns,tile,difference,
                         tile_bytes=tile.nbytes+difference.nbytes)
            for feature in range(self.dimension):
                np.subtract(rows[:,feature,None],columns[None,:,feature],out=difference)
                np.square(difference,out=difference)
                np.add(tile,difference,out=tile)
            del difference
            tile*=self.scale32
            np.exp(tile,out=tile)
        elif self.mode=='gram32_acc64':
            tile=rows@columns.T
            tile*=np.float32(-2.)
            tile+=rownorm[:,None];tile+=colnorm[None,:]
            np.maximum(tile,np.float32(0.),out=tile)
            if diagonal_start is not None:
                ids=np.arange(len(rows));shifted=ids+diagonal_start
                tile[ids,shifted]=0.
                self.observe('Gram_diagonal_indices',rows,columns,rownorm,colnorm,tile,ids,shifted,
                             tile_bytes=tile.nbytes)
                del ids,shifted
            tile*=self.scale32;np.exp(tile,out=tile)
        else:
            distance=cdist(rows,columns,'sqeuclidean')
            distance*=self.scale64
            if self.mode=='cdist64_exp32_acc64':
                tile=distance.astype(np.float32)
                self.observe('distance_conversion',rows,columns,distance,tile,
                             tile_bytes=distance.nbytes+tile.nbytes)
                del distance
            else:tile=distance
            np.exp(tile,out=tile)
        self.observe('generated_tile',rows,columns,rownorm,colnorm,tile,tile_bytes=tile.nbytes)
        self.stats['kernel_generation_seconds']+=perf_counter()-before
        return tile

    def query(self,row_ids,column_ids):
        self.stats['entry_query_calls']+=1
        rows=self.X[row_ids];columns=self.X[column_ids]
        rn=self.norms[row_ids] if self.norms is not None else None
        cn=self.norms[column_ids] if self.norms is not None else None
        tile=self.raw(rows,columns,rn,cn)
        if self.mode=='gram32_acc64':
            # Match the old full-product diagonal rule for arbitrary queries.
            where={int(i):j for j,i in enumerate(column_ids)}
            for i,index in enumerate(row_ids):
                if int(index) in where:tile[i,where[int(index)]]=1.
        promoted=tile if tile.dtype==np.float64 else tile.astype(np.float64)
        self.observe('query_promotion',rows,columns,rn,cn,tile,promoted,
                     tile_bytes=tile.nbytes+(0 if promoted is tile else promoted.nbytes))
        return promoted

    def apply(self):
        self.stats['product_calls']+=1
        result=np.empty(self.n,dtype=np.float64);self.live=[result]
        for start in range(0,self.n,self.rows):
            stop=min(start+self.rows,self.n)
            rn=self.norms[start:stop] if self.norms is not None else None
            tile=self.raw(self.X[start:stop],self.X,rn,self.norms,
                          diagonal_start=start if self.mode=='gram32_acc64' else None)
            promoted=tile if tile.dtype==np.float64 else tile.astype(np.float64)
            self.observe('product_promotion',tile,promoted,
                         tile_bytes=tile.nbytes+(0 if promoted is tile else promoted.nbytes))
            del tile
            values=promoted@self.v
            self.observe('FP64_dot',promoted,values,tile_bytes=promoted.nbytes)
            result[start:stop]=values
            del promoted,values
        self.live=[]
        assert self.max_tile_live<=TILE_BYTES
        return result

    def dense(self,row_count=None):
        """Explicit dense oracle; excluded from product memory allowance."""
        row_count=self.rows if row_count is None else row_count
        result=np.empty((self.n,self.n),dtype=np.float64)
        for start in range(0,self.n,row_count):
            stop=min(self.n,start+row_count)
            rn=self.norms[start:stop] if self.norms is not None else None
            tile=self.raw(self.X[start:stop],self.X,rn,self.norms,
                          diagonal_start=start if self.mode=='gram32_acc64' else None)
            result[start:stop]=tile
            del tile
        return result


def difference(reference,actual):
    delta=actual-reference
    return dict(bitwise_equal=bool(np.array_equal(reference,actual)),
                differing_entries=int(np.count_nonzero(reference!=actual)),
                max_absolute_difference=float(np.max(np.abs(delta))),
                relative_frobenius_difference=float(np.linalg.norm(delta)/max(np.linalg.norm(reference),np.finfo(float).tiny)))


def consistency(X,bw,label):
    rng=np.random.default_rng(622127);v=rng.normal(size=len(X));records=[]
    selected=rng.permutation(len(X))[:min(31,len(X))]
    remaining=rng.permutation(len(X))[:min(43,len(X))]
    for mode in MODES:
        op=Kernel(X,v,bw,mode)
        canonical=op.dense()
        symmetry=difference(canonical,canonical.T)
        row_checks=[]
        for count in [1,7,31]:
            alternate=op.dense(row_count=count)
            row_checks.append(dict(rows_per_tile=count,**difference(canonical,alternate)))
            del alternate
        whole=op.query(selected,remaining)
        transposed=op.query(remaining,selected).T
        split=np.empty_like(whole)
        for i in range(0,len(selected),3):
            for j in range(0,len(remaining),5):
                split[i:i+3,j:j+5]=op.query(selected[i:i+3],remaining[j:j+5])
        scalar=np.empty_like(whole)
        for i in range(min(4,len(selected))):
            for j in range(min(4,len(remaining))):
                scalar[i,j]=op.query(selected[i:i+1],remaining[j:j+1])[0,0]
        query_checks=dict(full_vs_canonical=difference(canonical[np.ix_(selected,remaining)],whole),
            transpose=difference(whole,transposed),split_3_by_5=difference(whole,split),
            scalar_subset=difference(whole[:4,:4],scalar[:4,:4]))
        okay=symmetry['bitwise_equal'] and all(z['bitwise_equal'] for z in row_checks) and all(z['bitwise_equal'] for z in query_checks.values())
        records.append(dict(dataset=label,mode=mode,n=len(X),symmetry=symmetry,
            row_shape_checks=row_checks,query_checks=query_checks,
            all_tested_entries_consistent_and_symmetric=okay,
            diagonal_max_deviation=float(np.max(np.abs(np.diag(canonical)-1.))),stats=op.stats))
    return records


def stability(X,bw,label,mode):
    v=np.zeros(len(X));op=Kernel(X,v,bw,mode)
    started,cpu=perf_counter(),process_time();K=op.dense()
    symmetry=difference(K,K.T);rows=[]
    for lam in [.001,1e-5]:
        tests=[]
        # A triangular Cholesky on an asymmetric matrix is not an SPD test of
        # that operator. Report both completions explicitly; never average.
        for lower in ([True] if symmetry['bitwise_equal'] else [True,False]):
            shifted=np.array(K,dtype=np.float64,order='F',copy=True)
            shifted.flat[::len(K)+1]+=lam
            before=shifted.__array_interface__['data'][0]
            t=perf_counter();c=process_time()
            factor,info=lapack.dpotrf(shifted,lower=int(lower),clean=0,overwrite_a=1)
            tests.append(dict(triangle='lower' if lower else 'upper',info=int(info),
                succeeded=info==0,seconds=perf_counter()-t,cpu_seconds=process_time()-c,
                factor_alias=factor.__array_interface__['data'][0]==before,
                minimum_factor_diagonal=float(np.min(np.diag(factor))) if info==0 else None))
            del shifted,factor
        item=dict(regularization=lam,cholesky_tests=tests,
            symmetric_operator_passes_Cholesky=bool(symmetry['bitwise_equal'] and tests[0]['succeeded']))
        if symmetry['bitwise_equal'] and not tests[0]['succeeded']:
            # This is a diagnostic eigenvalue, not a shifted/repaired operator.
            smallest=float(linalg.eigh(K,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0])
            item['smallest_kernel_eigenvalue']=smallest
            item['smallest_shifted_eigenvalue']=smallest+lam
        rows.append(item)
    return dict(dataset=label,mode=mode,n=len(X),symmetry=symmetry,regularizations=rows,
        dense_kernel_bytes=K.nbytes,oracle_scope='Dense kernel and dense Cholesky workspaces excluded from candidate memory; no symmetrization, diagonal repair or extra shift.',
        total_seconds=perf_counter()-started,total_cpu_seconds=process_time()-cpu,stats=op.stats)


def timed_case(X,b,v,bw,label,data_hash,case_index):
    reference=Kernel(X,v,bw,'cdist64').apply();ops={};records={}
    for mode in MODES:
        wall,cpu=perf_counter(),process_time();op=Kernel(X,v,bw,mode)
        setup,setup_cpu=perf_counter()-wall,process_time()-cpu
        observed=op.apply();ops[mode]=op
        records[mode]=dict(dataset=label,mode=mode,n=len(X),dimension=X.shape[1],data_sha256=data_hash,
            setup_seconds=setup,setup_cpu_seconds=setup_cpu,additional_retained_feature_and_norm_bytes=op.extra_bytes,
            rows_per_tile=op.rows,per_entry_tile_conversion_allowance=op.entry_bytes,
            maximum_tile_conversion_bytes=op.max_tile_live,
            output_vector_bytes=8*len(X),common_immutable_feature_and_input_bytes=X.nbytes+v.nbytes,
            error_over_rhs_norm=float(np.linalg.norm(observed-reference)/np.linalg.norm(b)),
            relative_product_error=float(np.linalg.norm(observed-reference)/np.linalg.norm(reference)),
            error_over_input_norm=float(np.linalg.norm(observed-reference)/np.linalg.norm(v)),seconds=[],cpu_seconds=[])
        del observed
    for repeat in range(5):
        offset=(repeat+case_index)%len(MODES)
        for mode in MODES[offset:]+MODES[:offset]:
            wall,cpu=perf_counter(),process_time();observed=ops[mode].apply()
            records[mode]['seconds'].append(perf_counter()-wall)
            records[mode]['cpu_seconds'].append(process_time()-cpu)
            del observed
    for mode in MODES:
        records[mode].update(median_seconds=float(np.median(records[mode]['seconds'])),
                            median_cpu_seconds=float(np.median(records[mode]['cpu_seconds'])),
                            measured_context_stats=ops[mode].stats)
    del ops,op
    for mode in MODES:
        tracemalloc.start()
        try:
            op=Kernel(X,v,bw,mode,audit=True);observed=op.apply()
            current,peak=tracemalloc.get_traced_memory()
            records[mode].update(named_context_and_action_peak_bytes=op.named_peak,
                phase_named_peaks=op.phase_peaks,traced_context_and_action_peak_bytes=peak,
                audited_tile_conversion_peak_bytes=op.max_tile_live,
                audit_scope='Separate untimed context creation plus one full product. Common original FP64 X/v allocated before trace. FP32 features/norms, one scratch distance tile, conversions, diagonal indices, output and dot-result arrays are included.')
            del op,observed
        finally:tracemalloc.stop()
    return list(records.values())


def main():
    output=ROOT/'results/symmetric-kernel-precision.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,112))
    started,cpu=perf_counter(),process_time()
    dependencies=[Path(__file__).resolve(),ROOT/'scripts/check_kernel_precision_cost.py',ROOT/'scripts/check_kernel_mixed_accumulation.py']
    record=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in dependencies},
        protocol=dict(modes=MODES,tile_and_conversion_byte_allowance=TILE_BYTES,repeats=5,
            order='Cyclic across four modes by repetition and case; one untimed full product first',
            direct32_arithmetic='Round features once to FP32; fixed increasing feature loop subtract/square/add in FP32; multiply FP32 coefficient and exp in FP32; promote tile then FP64 dot with original vector',
            baseline_arithmetic='Reproduces existing cdist64, Gram32-acc64 and cdist64-exp32-acc64 formulas; same per-mode8/12-byte tile budgeting.',
            accumulation='FP64 for all four methods',timing_vector='prepared alpha_reference atlambda.001',
            regularizations=[.001,1e-5],blas_threads=1,cpu_cap_seconds=112,
            lock='/tmp/cs6220-project-benchmark.lock',
            interpretation='Operator consistency, dense stability oracle and standalone product cost only; no inner/full solver, total-memory or efficiency claim.'),
        consistency_checks=[],stability_checks=[],timings=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(622126);small=np.ascontiguousarray(rng.normal(size=(37,8)))
        small[-1]=small[0];small[-2]=small[1]+1e-6
        record['consistency_checks'].extend(consistency(small,2.1,'small37'));save()
        for index,(name,filename) in enumerate(CASES):
            if process_time()-cpu>105:record['status']='cpu_cap';save();return
            path=ROOT/'data/prepared'/filename
            with np.load(path) as data:
                X,b,v,bw=data['X_train'],data['y_train'],data['alpha_reference'],float(data['bandwidth'])
            if index==0:record['consistency_checks'].extend(consistency(X,bw,name));save()
            for mode in MODES:
                item=stability(X,bw,name,mode);item['data_sha256']=sha256(path.read_bytes()).hexdigest()
                record['stability_checks'].append(item);save()
                print(json.dumps(dict(dataset=name,mode=mode,symmetric=item['symmetry']['bitwise_equal'],
                    shifted_Cholesky=[z['symmetric_operator_passes_Cholesky'] for z in item['regularizations']])) ,flush=True)
            record['timings'].extend(timed_case(X,b,v,bw,name,sha256(path.read_bytes()).hexdigest(),index));save()
    record['status']='complete';save()
    print(json.dumps(dict(output=str(output),status=record['status'],elapsed_cpu_seconds=record['elapsed_cpu_seconds'])))


if __name__=='__main__':main()
