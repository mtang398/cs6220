"""Native exact congruence with uniform FP32 core600; no core refinement."""
import json,resource,tracemalloc
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_native_first8_schur as scalar
from check_core_precision_allocation import Helper
from check_packed_float_allocation import RawPointers
from check_schur_inplace_frontier_v2 import ROOT,LIMIT

class Fused:
 def __init__(self,X,b,bw,lam,helper):
  owner=scalar.old_native.Schur(X,b,bw,600,lam,'fixed2',helper)
  self.factor=owner.float;self.budget=owner.budget;self.kernel=owner.kernel
  self.S,self.R=owner.S,owner.R;self.permutation=owner.permutation;self.m=600;self.n=len(b);self.lam=lam
  self.actions=0;self.verifications=0
 def rhs(self,b):
  r=b[self.permutation]
  with self.budget.hold(r):
   h=self.factor.solve(r[:self.m])
   with self.budget.hold(h):self.kernel.product(self.R,self.S,h,label='fused_rhs',target=r[self.m:],subtract=True)
  return r
 def apply(self,p,return_ha=False):
  m=self.m
  values=self.kernel.product(None,self.R,p[m:],label='fused_first_pass')
  out=values[self.permutation];self.budget.observe('fused_reorder',values,out);del values
  with self.budget.hold(out):
   h=self.factor.solve(out[:m]);q=p[:m]-h;self.budget.observe('fused_q',h,q);del h
   with self.budget.hold(q):
    a=self.kernel.product(self.S,self.S,q,label='fused_core')
    self.budget.observe('fused_core_shift',a,temporary_allowance=q.nbytes);a+=self.lam*q;a+=out[:m]
    with self.budget.hold(a):ha=self.factor.solve(a)
    self.budget.observe('fused_second_direction',a,ha);q-=ha
    if not return_ha:del ha;ha=None
    out[:m]=a;del a;q*=-1
    with self.budget.hold(ha):self.kernel.product(self.R,self.S,q,label='fused_second_pass',target=out[m:],subtract=True)
   del q
   self.budget.observe('fused_tail_shift',ha,temporary_allowance=out[m:].nbytes);out[m:]+=self.lam*p[m:]
  self.actions+=1
  if return_ha:return out,ha
  del ha;return out
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

def solve(X,b,bw,helper,cpu_cap=6,recurrent=False):
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
   else:ap=op.apply(p);ha=None
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
 row=dict(method='uniform600_recurrent' if recurrent else 'uniform600_fused',status=status,converged=status=='converged',iterations=steps,original_relative_residual=relative,error=error,total_seconds=perf_counter()-start,total_cpu_seconds=process_time()-cpu)
 if op is not None:row.update(memory=op.budget.report(),kernel_stats=op.kernel.stats,actions=op.actions,verifications=op.verifications,core_solve_calls=op.factor.solve_calls,core_solve_seconds=op.factor.solve_seconds,setup_seconds=setup)
 return x,row
