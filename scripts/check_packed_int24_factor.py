"""Native packed signed24-bit triangular factor, streamed kernel construction."""
import ctypes
from time import perf_counter
import numpy as np
from check_native_first8_schur import old_native,SharedKernel,ROOT
from check_core_precision_allocation import pointer

class Helper24:
 def __init__(self,path):
  self.library=ctypes.CDLL(str(path))
  self.column=self.library.int24_column;self.column.argtypes=[ctypes.c_size_t]*2+[ctypes.c_void_p]*7;self.column.restype=ctypes.c_int
  self.solve=self.library.int24_solve;self.solve.argtypes=[ctypes.c_size_t]+[ctypes.c_void_p]*5;self.solve.restype=ctypes.c_int
  self.product=self.library.int24_product;self.product.argtypes=[ctypes.c_size_t]+[ctypes.c_void_p]*7;self.product.restype=ctypes.c_int

class Streamed24:
 def __init__(self,X,b,bw,m,lam,helper):
  assert m>=8
  self.m,self.n,self.lam,self.helper=m,len(b),lam,helper
  self.budget=old_native.ScopedBudget([X,b]);self.kernel=SharedKernel(X,bw,self.budget)
  start=perf_counter();indices=self.kernel.farthest(m);self.FPS_seconds=perf_counter()-start
  self.low=np.empty(3*(m-8)*(m-9)//2,dtype=np.uint8);self.wide=np.empty(8*m-28)
  self.diag=np.empty(m);self.scales=np.empty(m);epsilon=np.empty(m);pivots=np.empty(m)
  arrays=[self.low,self.wide,self.diag,self.scales];self.factor_bytes=sum(x.nbytes for x in arrays)
  self.budget.fixed=[indices,*arrays,epsilon,pivots];points=X[indices];self.budget.observe('int24_setup',points)
  start=perf_counter()
  with self.budget.hold(points):
   for j in range(m):
    work=self.kernel.block(points[j:],points[j:j+1]).reshape(m-j);work[0]+=lam
    self.budget.observe('int24_column',work)
    info=helper.column(m,j,pointer(self.low),pointer(self.wide),pointer(self.diag),pointer(self.scales),pointer(work),pointer(epsilon[j:j+1]),pointer(pivots[j:j+1]))
    del work
    if info:raise np.linalg.LinAlgError(f'int24 factor column{j} code{info}')
  self.factor_seconds=perf_counter()-start;self.minimum_raw_pivot=float(pivots.min());self.initial_epsilon_norm=float(np.linalg.norm(epsilon))
  self.budget.fixed=[indices,*arrays];del epsilon,pivots,points
  mask=np.ones(self.n,dtype=bool);mask[indices]=False;tail=np.flatnonzero(mask);self.permutation=np.r_[indices,tail]
  self.budget.observe('int24_permutation',mask,tail,self.permutation);self.S,self.R=self.permutation[:m],self.permutation[m:]
  self.budget.fixed=[self.permutation,*arrays];del indices,mask,tail
  self.solve_calls=0;self.solve_seconds=0.
 def solve(self,rhs):
  start=perf_counter();answer=np.array(rhs,dtype=float,copy=True);self.budget.observe('int24_solve',rhs,answer)
  info=self.helper.solve(self.m,pointer(self.low),pointer(self.wide),pointer(self.diag),pointer(self.scales),pointer(answer))
  if info:raise np.linalg.LinAlgError(f'int24 solve code{info}')
  self.solve_calls+=1;self.solve_seconds+=perf_counter()-start;return answer
 def product(self,q):
  temporary=np.empty(self.m);answer=np.empty(self.m)
  with self.budget.hold(q,temporary,answer):
   self.budget.observe('int24_product',q,temporary,answer)
   info=self.helper.product(self.m,pointer(self.low),pointer(self.wide),pointer(self.diag),pointer(self.scales),pointer(q),pointer(temporary),pointer(answer))
   if info:raise np.linalg.LinAlgError(f'int24 product code{info}')
  return answer
 def dense_reference(self):
  # Validation only; never called by native solver.
  L=np.zeros((self.m,self.m))
  for j in range(self.m):
   if j<8:
    offset=j*self.m-j*(j-1)//2;L[j:,j]=self.wide[offset:offset+self.m-j]
   else:
    offset=(j-8)*(2*self.m-j-9)//2
    raw=self.low[3*offset:3*(offset+self.m-j-1)].reshape(-1,3).astype(np.int32)
    v=raw[:,0]+256*raw[:,1]+65536*raw[:,2];v=np.where(v>=8388608,v-16777216,v)
    L[j,j]=self.diag[j];L[j+1:,j]=v*self.scales[j]
  return L
