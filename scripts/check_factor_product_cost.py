"""Packed factor-product accuracy, streamed defect estimate, and primitive cost."""
import ctypes,json,resource,subprocess
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
from check_core_precision_allocation import Helper,MixedFactor,pointer
from check_native_first8_schur import StreamedFirst8
from check_schur_inplace_frontier_v2 import ROOT,Budget,SharedKernel

class Product:
 def __init__(self,path):
  self.library=ctypes.CDLL(str(path));self.fn=self.library.first8_factor_product;self.fn.argtypes=[ctypes.c_size_t]+[ctypes.c_void_p]*7;self.fn.restype=ctypes.c_int
 def apply(self,factor,q):
  temporary=np.empty(factor.m);out=np.empty(factor.m)
  with factor.budget.hold(q,temporary,out):
   factor.budget.observe('factor_product',q,temporary,out)
   info=self.fn(factor.m,pointer(factor.high),pointer(factor.offsets),pointer(factor.low),pointer(factor.wide),pointer(q),pointer(temporary),pointer(out))
   if info:raise ValueError(info)
  return out

def main():
 out=ROOT/'results/factor-product-cost.json';source=ROOT/'scripts/first8_factor_product.c';library=ROOT/'results/first8_factor_product.so'
 if out.exists() or library.exists():raise FileExistsError('Preserve artifacts')
 resource.setrlimit(resource.RLIMIT_CPU,(35,40));t,cpu=perf_counter(),process_time();command=['gcc','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(library)]
 subprocess.run(command,check=True,capture_output=True);product=Product(library);helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
 files=[Path(__file__).resolve(),source,library,ROOT/'scripts/check_native_first8_schur.py',ROOT/'scripts/check_core_precision_allocation.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'results/mixed_column_factor_first8.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},compile_command=command,protocol=dict(core=592,shift=1e-5,repeats=5,batch_calls=20,cost='Native packed product including two allocated m-vectors and budget tracking; compare streamed Gaussian core product. Dense construction only for independent reference/error measurement.',scope='Primitive costs, not full solver. sqrt2 column-defect norm initially treated as rounding-unadjusted estimate.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,fn in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   factor=StreamedFirst8(X,b,bw,592,1e-5,helper);rng=np.random.default_rng(622093);q=rng.normal(size=592)
   # Offline independent dense factor and matrix for accuracy, not timed or included in native allocations.
   C=factor.kernel.block(X[factor.S],X[factor.S]);C.flat[::593]+=1e-5;dense=MixedFactor(C,range(8),helper);L=dense.dense_L();assert factor.low.tobytes()==dense.low.tobytes() and factor.wide.tobytes()==dense.wide.tobytes()
   got=product.apply(factor,q);reference=L@(L.T@q);error=float(np.linalg.norm(got-reference)/np.linalg.norm(reference));assert error<1e-12
   defect=float(np.linalg.norm(C-L@L.T,'fro'));estimate=float(np.sqrt(2)*factor.initial_epsilon_norm)
   del C,L,dense,got,reference
   times={'factor':[],'kernel':[]}
   for repeat in range(5):
    for method in (['factor','kernel'] if repeat%2==0 else ['kernel','factor']):
     start=perf_counter()
     for _ in range(20):
      if method=='factor':v=product.apply(factor,q)
      else:v=factor.kernel.product(factor.S,factor.S,q,label='reference_exact_core');v+=1e-5*q
      del v
     times[method].append((perf_counter()-start)/20)
   row=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),product_relative_error=error,exact_core_defect_frobenius=defect,column_defect_estimate=estimate,estimate_over_actual=estimate/defect,seconds_per_product=times,medians={k:float(np.median(v)) for k,v in times.items()});rec['cases'].append(row);save();print(json.dumps(row),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
