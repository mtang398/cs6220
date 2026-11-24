"""Native streamed exact congruence with FIRST8 compressed core."""
import json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_first8_schur as scalar
from check_core_precision_allocation import Helper
from check_packed_float_allocation import RawPointers
from check_factor_product_cost import Product
from check_schur_inplace_frontier_v2 import ROOT,LIMIT

class Fused:
 def __init__(self,X,b,bw,lam,helper):
  self.factor=scalar.StreamedFirst8(X,b,bw,592,lam,helper);self.budget=self.factor.budget;self.kernel=self.factor.kernel
  self.S,self.R=self.factor.S,self.factor.R;self.permutation=self.factor.permutation;self.m=592;self.n=len(b);self.lam=lam
  self.actions=0;self.verifications=0
  self.product=Product(ROOT/'results/first8_factor_product.so');self.omitted=0;self.rejected=0;self.trials=0
  u=2.**-53;gamma=2*self.m*u/(1-2*self.m*u);eta=1/((1-u)**2*(1-gamma))
  self.error_bound=(np.sqrt(2)*eta*self.factor.initial_epsilon_norm+2*gamma*self.m*(1+lam))/(1-gamma*np.sqrt(self.m))
  self.remaining=.25e-8*float(np.linalg.norm(b));self.used=0.
 def rhs(self,b):
  r=b[self.permutation]
  with self.budget.hold(r):
   h=self.factor.solve(r[:self.m])
   with self.budget.hold(h):self.kernel.product(self.R,self.S,h,label='fused_rhs',target=r[self.m:],subtract=True)
  return r
 def apply(self,p,rho=None):
  m=self.m;cheap=False;bound=0.
  values=self.kernel.product(None,self.R,p[m:],label='fused_first_pass')
  out=values[self.permutation];self.budget.observe('fused_reorder',values,out);del values
  with self.budget.hold(out):
   h=self.factor.solve(out[:m]);q=p[:m]-h;self.budget.observe('fused_q',h,q);del h
   with self.budget.hold(q):
    if rho is not None:
     a=self.product.apply(self.factor,q);self.trials+=1
     den=float(q@a)+2*float(q@out[:m])+float(p[m:]@out[m:])+self.lam*float(p[m:]@p[m:])
     bound=self.error_bound*float(np.linalg.norm(q))
     cheap=den>0 and abs(rho/den)*bound<=self.remaining
     if not cheap:del a
    if not cheap:
     a=self.kernel.product(self.S,self.S,q,label='fused_core')
     self.budget.observe('fused_core_shift',a,temporary_allowance=q.nbytes);a+=self.lam*q
    a+=out[:m]
    with self.budget.hold(a):ha=self.factor.solve(a)
    self.budget.observe('fused_second_direction',a,ha);q-=ha;del ha
    out[:m]=a;del a;q*=-1
    self.kernel.product(self.R,self.S,q,label='fused_second_pass',target=out[m:],subtract=True)
   del q
   self.budget.observe('fused_tail_shift',temporary_allowance=out[m:].nbytes);out[m:]+=self.lam*p[m:]
  self.actions+=1
  if cheap:
   actual=float(p@out)
   if actual<=0 or abs(rho/actual)*bound>self.remaining:
    del out;self.rejected+=1;return self.apply(p,None)
   charge=abs(rho/actual)*bound;self.remaining-=charge;self.used+=charge;self.omitted+=1
  return out
 def precondition(self,r):
  z=np.empty_like(r)
  with self.budget.hold(z):
   top=self.factor.solve(r[:self.m]);z[:self.m]=top;del top
   np.divide(r[self.m:],self.lam,out=z[self.m:])
  return z
 def verify(self,y,b):
  x=np.empty_like(y);x[self.permutation]=y
  with self.budget.hold(x):
   t=self.kernel.product(self.S,self.R,y[self.m:],label='fused_reconstruction')
   with self.budget.hold(t):h=self.factor.solve(t)
   del t;self.budget.observe('fused_reconstruction_update',h,temporary_allowance=h.nbytes);x[self.S]-=h;del h
   residual=self.kernel.full(x,extra=(x,))
   self.budget.observe('fused_original_residual',residual,temporary_allowance=x.nbytes);residual+=self.lam*x;residual-=b
   relative=float(np.linalg.norm(residual)/np.linalg.norm(b));del residual
  self.verifications+=1;return x,relative

def solve(X,b,bw,helper,cpu_cap=6,recurrent=False,adaptive=True):
 assert not recurrent
 start,cpu=perf_counter(),process_time();op=None;relative=None;status='maxiter';error=None;steps=0;x=None;setup=None
 try:
  op=Fused(X,b,bw,1e-5,helper);setup=perf_counter()-start;y=np.zeros(len(b));op.budget.live=[y]
  r=op.rhs(b);op.budget.live=[y,r];z=op.precondition(r);p=z.copy();op.budget.observe('initial_direction_copy',z,p);rho=float(r@z)
  if recurrent:op.budget.live=[y,r,p,z]
  else:del z;op.budget.live=[y,r,p]
  threshold=.5e-8*np.linalg.norm(b)
  for steps in range(2001):
   if np.linalg.norm(r)<=threshold or steps==2000:
    x,relative=op.verify(y,b)
    if relative<=1e-8:status='converged';break
    del x;x=None
   if process_time()-cpu>cpu_cap:status='cpu_cap';break
   if steps==2000:break
   if recurrent:ap,ha=op.apply(p,True)
   else:ap=op.apply(p,rho if adaptive else None);ha=None
   den=float(p@ap)
   if den<=0 or rho<=0:
    status='nonpositive_curvature';del ap,ha;break
   alpha=rho/den;op.budget.observe('fused_iterate_update',ap,ha,temporary_allowance=y.nbytes);y+=alpha*p;r-=alpha*ap;del ap
   if recurrent:
    op.budget.observe('recurrent_top_update',ha,temporary_allowance=ha.nbytes);z[:op.m]-=alpha*ha;del ha
    np.divide(r[op.m:],op.lam,out=z[op.m:])
   else:z=op.precondition(r)
   new=float(r@z);op.budget.observe('fused_direction_update',z,temporary_allowance=p.nbytes);p*=new/rho;p+=z;rho=new
   if not recurrent:del z
  if status!='converged':
   if x is not None:del x
   x,relative=op.verify(y,b)
   if relative<=1e-8:status='converged'
 except (MemoryError,np.linalg.LinAlgError) as e:status='memory_failure' if isinstance(e,MemoryError) else 'numerical_failure';error=str(e)
 row=dict(method='adaptive_core' if adaptive else 'exact_core',status=status,converged=status=='converged',iterations=steps,original_relative_residual=relative,error=error,total_seconds=perf_counter()-start,total_cpu_seconds=process_time()-cpu)
 if op is not None:row.update(memory=op.budget.report(),kernel_stats=op.kernel.stats,actions=op.actions,verifications=op.verifications,core_solve_calls=op.factor.solve_calls,core_solve_seconds=op.factor.solve_seconds,setup_seconds=setup,omitted=op.omitted,rejected=op.rejected,trials=op.trials,truncation_bound_used=op.used,core_error_bound=op.error_bound)
 return x,row
