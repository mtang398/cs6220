"""CPU kernel-product cost/error screen, not a complete solver comparison.

Compare cdist FP64 and identical norm/Gram generators in FP64/FP32. A fourth
mode isolates feature rounding from arithmetic rounding. Kernel tiles have the
same maximum byte count; preprocessing is measured separately and retained
copies plus one-action allocation traces are reported.
"""
import json
import resource
import tracemalloc
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
TILE_BYTES=131072
MODES=['cdist64','gram64','gram32','gram64_rounded_features']


class Product:
    def __init__(self,X,v,bw,mode):
        self.mode=mode;self.bw=bw
        if mode=='gram32':
            self.X=X.astype(np.float32);self.v=v.astype(np.float32)
        elif mode=='gram64_rounded_features':
            self.X=X.astype(np.float32).astype(np.float64);self.v=v
        else:
            self.X=X;self.v=v
        self.norms=None if mode=='cdist64' else np.einsum('ij,ij->i',self.X,self.X)
        self.rows=min(len(X),max(1,TILE_BYTES//(len(X)*self.X.dtype.itemsize)))
        self.extra_bytes=(0 if self.X is X else self.X.nbytes)+(0 if self.v is v else self.v.nbytes)
        if self.norms is not None:self.extra_bytes+=self.norms.nbytes

    def apply(self):
        n=len(self.X);out=np.empty(n,dtype=np.float64)
        for start in range(0,n,self.rows):
            stop=min(start+self.rows,n)
            if self.mode=='cdist64':
                tile=cdist(self.X[start:stop],self.X,'sqeuclidean')
            else:
                tile=self.X[start:stop]@self.X.T
                tile*=-2
                tile+=self.norms[start:stop,None]
                tile+=self.norms[None,:]
                np.maximum(tile,0,out=tile)
                tile[np.arange(stop-start),np.arange(start,stop)]=0
            tile*= -.5/(self.bw*self.bw)
            np.exp(tile,out=tile)
            out[start:stop]=tile@self.v
            del tile
        return out


def main():
    output=ROOT/'results/kernel-precision-cost.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    cases=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),
           ('Concrete824','concrete-observed-s0-lambda0.001.npz'),
           ('Airfoil1202','airfoil-observed-s0-lambda0.001.npz'),
           ('smooth2048','synthetic-n2560-smooth-s0-lambda0.001.npz')]
    for name,stem in cases:
        if not (ROOT/'data/prepared'/stem).exists():raise FileNotFoundError(stem)
    record=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(tile_byte_allowance=TILE_BYTES,modes=MODES,repeats=5,
            order='Cyclic order across modes per repetition and case; one untimed action before timings',
            vectors=['seeded_normal','prepared_reference_solution_lambda0.001'],
            output_dtype='float64 for every mode; FP32 tile and input vector multiply stays FP32',
            scope='Standalone CPU product cost/error, not inner-solver convergence or total solver memory',
            excluded_from_product_timing='Context creation/feature conversion separately measured'),runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for ci,(name,stem) in enumerate(cases):
            path=ROOT/'data/prepared'/stem
            with np.load(path) as data:
                X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
                solution=data['alpha_reference']
            vectors=[('seeded_normal',np.random.default_rng(622124).normal(size=len(b))),
                     ('prepared_reference_solution_lambda0.001',solution)]
            for vi,(vname,v) in enumerate(vectors):
                reference=Product(X,v,bw,'cdist64').apply()
                operators={};rows={}
                for mode in MODES:
                    t=perf_counter();op=Product(X,v,bw,mode);setup=perf_counter()-t
                    observed=op.apply()
                    rows[mode]=dict(dataset=name,vector=vname,mode=mode,n=len(b),dimension=X.shape[1],
                        data_sha256=sha256(path.read_bytes()).hexdigest(),setup_seconds=setup,
                        additional_retained_context_bytes=op.extra_bytes,rows_per_tile=op.rows,
                        maximum_tile_bytes=op.rows*len(b)*op.X.dtype.itemsize,
                        relative_product_error=float(np.linalg.norm(observed-reference)/np.linalg.norm(reference)),
                        error_over_rhs_norm=float(np.linalg.norm(observed-reference)/np.linalg.norm(b)),
                        error_over_input_norm=float(np.linalg.norm(observed-reference)/np.linalg.norm(v)),
                        seconds=[])
                    operators[mode]=op
                for repeat in range(5):
                    offset=(repeat+ci+vi)%len(MODES)
                    for mode in MODES[offset:]+MODES[:offset]:
                        t=perf_counter();observed=operators[mode].apply()
                        rows[mode]['seconds'].append(perf_counter()-t)
                del operators,op,observed
                for mode in MODES:
                    tracemalloc.start()
                    try:
                        op=Product(X,v,bw,mode);observed=op.apply()
                        current,peak=tracemalloc.get_traced_memory()
                    finally:tracemalloc.stop()
                    rows[mode].update(median_seconds=float(np.median(rows[mode]['seconds'])),
                        traced_context_and_action_peak_bytes=peak,
                        trace_scope='New context plus one action; common FP64 X and v allocated before trace')
                    del op,observed
                    record['runs'].append(rows[mode])
                save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
