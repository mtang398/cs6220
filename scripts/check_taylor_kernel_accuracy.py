"""Numerical audit of Taylor evaluation and error certificates, not proof of FP bounds."""
import ctypes,json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gammainc
from threadpoolctl import threadpool_limits
from check_taylor_kernel_cost import Taylor,ROOT

def main():
 out=ROOT/'results/taylor-kernel-accuracy.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60)); t,c=perf_counter(),process_time()
 lib=ctypes.CDLL(str(ROOT/'results/taylor_kernel_tile.so'));fun=lib.taylor_tile
 arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 fun.argtypes=[arr,arr,arr,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int];fun.restype=None
 paths=['synthetic-n640-smooth-s0-lambda0.001.npz','concrete-observed-s0-lambda0.001.npz','airfoil-observed-s0-lambda0.001.npz','synthetic-n2560-smooth-s0-lambda0.001.npz']
 cost=json.loads((ROOT/'results/taylor-kernel-cost.json').read_text());runs=[]
 with threadpool_limits(limits=1):
  for path,name in zip(paths,['smooth512','Concrete824','Airfoil1202','smooth2048']):
   with np.load(ROOT/'data/prepared'/path) as z:X,v,bw=z['X_train'],z['alpha_reference'],float(z['bandwidth'])
   n=len(X);K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
   degrees=sorted(set([2,4]+[x['degree'] for x in cost['selections'] if x['dataset']==name]))
   for p in degrees:
    op=Taylor(X,v,bw,p,True,fun);g=op.X@op.X.T;h=np.empty_like(g);fun(g,h,op.d,n,0,n,p)
    e=op.e;root=np.sqrt(e);beta=float(np.max(root*(root.sum()-root)))
    E=K-h; diagerr=float(np.max(np.abs(np.diag(E)-e)))
    diagindex=np.diag_indices(n);h[diagindex]+=e
    # Compare matrix-free original-order query tiles with full dense construction.
    block_action=op.apply();dense_action=h@v
    off=K-h;actual=float(np.linalg.norm(off@v));bound=float(np.linalg.norm(root*(root@np.abs(v)-root*np.abs(v))))
    rowbound=float(np.max(np.sum(np.abs(off),axis=1)))
    row=dict(dataset=name,degree=p,analytic_beta=beta,measured_error_matrix_row_bound=rowbound,diagonal_tail_absolute_error=diagerr,symmetry_max=float(np.max(np.abs(h-h.T))),block_action_relative_difference=float(np.linalg.norm(block_action-dense_action)/np.linalg.norm(v)),current_action_error=actual,current_action_bound=bound,certificate_ratio=bound/max(actual,1e-300))
    assert actual <= bound+1e-11*np.linalg.norm(v)
    assert rowbound <= beta+1e-10
    # Independent lower triangle Cholesky only for most difficult tested shift.
    h[diagindex]+=1e-5
    np.linalg.cholesky(h);row['cholesky_at_shift_1e_5']=True
    runs.append(row)
 record=dict(status='complete',scope='Dense offline audit only; dense allocations excluded from candidate timing/memory claims',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),runs=runs,elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c)
 out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__': main()
