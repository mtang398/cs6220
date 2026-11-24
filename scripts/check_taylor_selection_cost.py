"""Memory-lean current-direction degree selection overhead, standalone only."""
import ctypes,json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.special import gammainc
from threadpoolctl import threadpool_limits
from check_taylor_kernel_cost import Taylor,ROOT
from check_kernel_precision_cost import Product

def select(s,v,threshold):
 absolute=np.abs(v);calls=0
 def bound(q):
  nonlocal calls
  calls+=1;root=gammainc(q+1,s);np.sqrt(root,out=root)
  weighted=root*absolute;total=float(weighted.sum());weighted*=-1;weighted+=total;weighted*=root
  return float(np.linalg.norm(weighted))
 lo,hi=0,40
 while lo<hi:
  mid=(lo+hi)//2
  if bound(mid)<=threshold:hi=mid
  else:lo=mid+1
 return lo,calls

def main():
 out=ROOT/'results/taylor-selection-cost.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60));t,c=perf_counter(),process_time()
 lib=ctypes.CDLL(str(ROOT/'results/taylor_kernel_tile.so'));fun=lib.taylor_tile
 arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fun.argtypes=[arr,arr,arr,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int];fun.restype=None
 paths=[('smooth512','synthetic-n640-smooth-s0-lambda0.001.npz'),('Concrete824','concrete-observed-s0-lambda0.001.npz'),('Airfoil1202','airfoil-observed-s0-lambda0.001.npz'),('smooth2048','synthetic-n2560-smooth-s0-lambda0.001.npz')]
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(degrees=[12,18,24],selection='Binary search 0..40 with stable gammainc tails, one norm vector retained; no n-by-degree cache',vector='prepared alpha_reference at shift .001',threshold='midpoint between actual directional bounds at degree q and q-1; cost measurement not a solver rule',repeats=11,scope='Standalone selection-plus-action cost only; no end-to-end claim'),runs=[])
 with threadpool_limits(limits=1):
  for ds,stem in paths:
   with np.load(ROOT/'data/prepared'/stem) as z:X,v,bw=z['X_train'],z['alpha_reference'],float(z['bandwidth'])
   s=np.einsum('ij,ij->i',X,X)/(bw*bw)
   for degree in [12,18,24]:
    def bound(q):
     root=np.sqrt(gammainc(q+1,s));return np.linalg.norm(root*(root@np.abs(v)-root*np.abs(v)))
    threshold=float((bound(degree)+bound(degree-1))/2)
    q,calls=select(s,v,threshold);assert q==degree
    op=Taylor(X,v,bw,degree,True,fun);exact=[Product(X,v,bw,m) for m in ['cdist64','gram64']]
    times={k:[] for k in ['select','taylor','cdist64','gram64']}
    for rep in range(11):
     operations=[('select',lambda:select(s,v,threshold)),('taylor',op.apply),('cdist64',exact[0].apply),('gram64',exact[1].apply)]
     offset=rep%4
     for name,fn in operations[offset:]+operations[:offset]:
      now=perf_counter();fn();times[name].append(perf_counter()-now)
    tracemalloc.start();select(s,v,threshold);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    med={k:float(np.median(a)) for k,a in times.items()};base=min(med['cdist64'],med['gram64'])
    rec['runs'].append(dict(dataset=ds,degree=degree,n=len(X),bound_evaluations=calls,retained_norm_bytes=s.nbytes,selection_traced_peak_bytes=peak,seconds=times,median_seconds=med,action_only_speedup=base/med['taylor'],selection_plus_action_speedup=base/(med['select']+med['taylor'])))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
