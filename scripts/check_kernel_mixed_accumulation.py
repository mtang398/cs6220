"""Isolate kernel generation from accumulation precision at matched tile bytes."""
import json
import resource
import tracemalloc
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_kernel_precision_cost import Product,ROOT,TILE_BYTES


class MixedProduct:
    def __init__(self,X,v,bw,mode):
        self.mode=mode;self.v=v;self.bw=bw
        if mode=='gram32_acc64':
            self.X=X.astype(np.float32)
            self.norms=np.einsum('ij,ij->i',self.X,self.X)
            self.extra_bytes=self.X.nbytes+self.norms.nbytes
        else:
            self.X=X;self.norms=None;self.extra_bytes=0
        # FP32 tile and FP64 conversion coexist: charge twelve bytes per entry.
        self.rows=max(1,TILE_BYTES//(12*len(X)))
    def apply(self):
        n=len(self.X);out=np.empty(n,dtype=float)
        for start in range(0,n,self.rows):
            stop=min(start+self.rows,n)
            if self.mode=='gram32_acc64':
                tile=self.X[start:stop]@self.X.T
                tile*=-2;tile+=self.norms[start:stop,None];tile+=self.norms[None,:]
                np.maximum(tile,0,out=tile)
                tile[np.arange(stop-start),np.arange(start,stop)]=0
                tile*= -.5/(self.bw*self.bw)
            else:
                distance=cdist(self.X[start:stop],self.X,'sqeuclidean')
                distance*= -.5/(self.bw*self.bw)
                tile=distance.astype(np.float32)
                del distance
            np.exp(tile,out=tile)
            promoted=tile.astype(np.float64)
            del tile
            out[start:stop]=promoted@self.v
            del promoted
        return out


def main():
    output=ROOT/'results/kernel-mixed-accumulation.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    modes=['cdist64','gram64','gram32','gram32_acc64','cdist64_exp32_acc64']
    paths=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),
           ('Concrete824','concrete-observed-s0-lambda0.001.npz'),
           ('Airfoil1202','airfoil-observed-s0-lambda0.001.npz'),
           ('smooth2048','synthetic-n2560-smooth-s0-lambda0.001.npz')]
    record=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(__file__).with_name('check_kernel_precision_cost.py')]},
        protocol=dict(tile_and_conversion_byte_allowance=TILE_BYTES,modes=modes,repeats=5,
            order='Cyclic modes across repetitions/cases',vector='Prepared reference solution atlambda.001',
            scope='Standalone matrix-free product, no solver or total-memory claim',
            accumulation='Mixed modes retain FP64 vector and accumulate promoted kernel tiles in FP64'),runs=[])
    def build(X,v,bw,mode):
        return MixedProduct(X,v,bw,mode) if mode.endswith('acc64') else Product(X,v,bw,mode)
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for ci,(name,stem) in enumerate(paths):
            path=ROOT/'data/prepared'/stem
            with np.load(path) as data:X,b,bw,v=data['X_train'],data['y_train'],float(data['bandwidth']),data['alpha_reference']
            reference=Product(X,v,bw,'cdist64').apply()
            ops={};rows={}
            for mode in modes:
                t=perf_counter();op=build(X,v,bw,mode);setup=perf_counter()-t
                y=op.apply();ops[mode]=op
                rows[mode]=dict(dataset=name,mode=mode,n=len(b),data_sha256=sha256(path.read_bytes()).hexdigest(),
                    setup_seconds=setup,additional_retained_context_bytes=op.extra_bytes,rows_per_tile=op.rows,
                    error_over_rhs_norm=float(np.linalg.norm(y-reference)/np.linalg.norm(b)),
                    error_over_input_norm=float(np.linalg.norm(y-reference)/np.linalg.norm(v)),seconds=[])
            for rep in range(5):
                offset=(rep+ci)%len(modes)
                for mode in modes[offset:]+modes[:offset]:
                    t=perf_counter();y=ops[mode].apply();rows[mode]['seconds'].append(perf_counter()-t)
            del ops,op,y
            for mode in modes:
                tracemalloc.start()
                try:
                    op=build(X,v,bw,mode);y=op.apply()
                    current,peak=tracemalloc.get_traced_memory()
                finally:tracemalloc.stop()
                rows[mode].update(median_seconds=float(np.median(rows[mode]['seconds'])),
                                  traced_context_and_action_peak_bytes=peak)
                record['runs'].append(rows[mode]);del op,y
            save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
