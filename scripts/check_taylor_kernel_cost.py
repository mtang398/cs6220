"""Compiled Taylor action cost screen; not a solver or novelty claim."""
import ctypes, json, resource, subprocess, tracemalloc
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from scipy.special import gammainc
from threadpoolctl import threadpool_limits
from check_kernel_precision_cost import Product, ROOT, TILE_BYTES

class Taylor:
    def __init__(self,X,v,bw,p,correct,fun):
        self.X=np.ascontiguousarray(X/bw); self.v=v; self.p=p; self.correct=correct; self.fun=fun
        self.s=np.einsum('ij,ij->i',self.X,self.X)
        self.d=np.exp(-self.s/2); self.e=gammainc(p+1,self.s)
        self.rows=max(1,TILE_BYTES//(16*len(X)))
        self.extra_bytes=self.X.nbytes+self.s.nbytes+self.d.nbytes+self.e.nbytes
    def apply(self):
        n=len(self.X); out=np.empty(n)
        for a in range(0,n,self.rows):
            end=min(n,a+self.rows); g=self.X[a:end]@self.X.T; h=np.empty_like(g)
            self.fun(g,h,self.d,n,a,end-a,self.p)
            out[a:end]=h@self.v
            del g,h
        if self.correct: out+=self.e*self.v
        return out

def main():
    output=ROOT/'results/taylor-kernel-cost.json'
    if output.exists(): raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    source=Path(__file__).with_name('taylor_kernel_tile.c'); libpath=ROOT/'results/taylor_kernel_tile.so'
    cmd=['gcc','-O3','-march=native','-shared','-fPIC',str(source),'-o',str(libpath)]
    subprocess.run(cmd,check=True)
    lib=ctypes.CDLL(str(libpath)); fun=lib.taylor_tile
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fun.argtypes=[arr,arr,arr,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int];fun.restype=None
    paths=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),('Concrete824','concrete-observed-s0-lambda0.001.npz'),('Airfoil1202','airfoil-observed-s0-lambda0.001.npz'),('smooth2048','synthetic-n2560-smooth-s0-lambda0.001.npz')]
    record=dict(status='running',sources={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),source,Path(__file__).with_name('check_kernel_precision_cost.py')]},protocol=dict(tile_bytes=TILE_BYTES,repeats=5,compiler=cmd,center='zero in existing standardized input coordinates',regularizations=[.001,1e-5],selection='least p in 0..255 with trace or diagonal-corrected row bound <= lambda/2',extra_degrees=[0,2,4,8],scope='Standalone actions; extra context separately reported; no full-solver memory claim'),selections=[],runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        # Unique changed-path check: C polynomial against independent explicit powers.
        rng=np.random.default_rng(6220); xx=rng.normal(size=(17,3))*.2; vv=rng.normal(size=17)
        op=Taylor(xx,vv,1.,4,True,fun); gram=xx@xx.T
        poly=sum(gram**k/float(__import__('math').factorial(k)) for k in range(5))
        expected=(op.d[:,None]*poly*op.d[None,:])@vv+op.e*vv
        err=float(np.linalg.norm(op.apply()-expected));assert err<1e-12
        record['changed_path_check_absolute_error']=err
        for ci,(name,stem) in enumerate(paths):
            path=ROOT/'data/prepared'/stem
            with np.load(path) as z: X,v,b,bw=z['X_train'],z['alpha_reference'],z['y_train'],float(z['bandwidth'])
            s=np.einsum('ij,ij->i',X,X)/(bw*bw); selected=set()
            for lam in [.001,1e-5]:
                for correction in [False,True]:
                    for p in range(256):
                        e=gammainc(p+1,s);root=np.sqrt(e)
                        bound=float(np.max(root*(root.sum()-root))) if correction else float(e.sum())
                        if bound<=lam/2: break
                    else: raise RuntimeError('degree cap reached')
                    selected.add((p,correction));record['selections'].append(dict(dataset=name,regularization=lam,diagonal_correction=correction,degree=p,bound=bound))
            selected.update((p,True) for p in [0,2,4,8])
            ops={'cdist64':Product(X,v,bw,'cdist64'),'gram64':Product(X,v,bw,'gram64')}
            ops.update({f'taylor{p}_diag{int(c)}':Taylor(X,v,bw,p,c,fun) for p,c in sorted(selected)})
            ref=ops['cdist64'].apply(); rows={}
            for mode,op in ops.items():
                y=op.apply()
                rows[mode]=dict(dataset=name,mode=mode,n=len(X),data_sha256=sha256(path.read_bytes()).hexdigest(),context_bytes=op.extra_bytes,rows_per_tile=op.rows,error_over_rhs_norm=float(np.linalg.norm(y-ref)/np.linalg.norm(b)),seconds=[])
            modes=list(ops)
            for rep in range(5):
                offset=(ci+rep)%len(modes)
                for mode in modes[offset:]+modes[:offset]:
                    t=perf_counter();ops[mode].apply();rows[mode]['seconds'].append(perf_counter()-t)
            for mode,op in ops.items():
                tracemalloc.start(); op.apply();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                rows[mode].update(median_seconds=float(np.median(rows[mode]['seconds'])),action_traced_peak_bytes=peak)
                record['runs'].append(rows[mode])
            save()
    record['status']='complete';save();print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__': main()
