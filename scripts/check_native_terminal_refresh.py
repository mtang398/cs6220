"""Streamed compressed-core actions with accumulated residual replacement."""
import numpy as np
from time import perf_counter,process_time
from check_native_adaptive_fused import Fused as Base,ROOT,LIMIT,Helper,RawPointers

class Fused(Base):
 def apply_cheap(self,p):
  m=self.m
  values=self.kernel.product(None,self.R,p[m:],label='refresh_first_pass')
  out=values[self.permutation];self.budget.observe('refresh_reorder',values,out);del values
  with self.budget.hold(out):
   h=self.factor.solve(out[:m]);q=p[:m]-h;self.budget.observe('refresh_q',h,q)
   with self.budget.hold(q,h):
    # H=Ctilde^-1: the approximate congruence has zero cross blocks.
    a=self.product.apply(self.factor,p[:m]);out[:m]=a;del a
    self.kernel.product(self.R,self.S,h,label='refresh_second_pass',target=out[m:],subtract=True)
    self.budget.observe('refresh_tail_shift',temporary_allowance=out[m:].nbytes);out[m:]+=self.lam*p[m:]
   del h
  self.actions+=1;self.omitted+=1
  return out,q
 def refresh(self,r,s):
  e=self.kernel.product(self.S,self.S,s,label='gap_core')
  with self.budget.hold(e):
   self.budget.observe('gap_shift',temporary_allowance=s.nbytes);e+=self.lam*s
   cheap=self.product.apply(self.factor,s);e-=cheap;del cheap
   h=self.factor.solve(e);r[:self.m]-=e;h*=-1
   with self.budget.hold(h):self.kernel.product(self.R,self.S,h,label='gap_cross',target=r[self.m:],subtract=True)
   del h
  s.fill(0)

def solve(X,b,bw,helper,mode='adaptive',cpu_cap=7):
 assert mode in ['adaptive','periodic8','terminal']
 start,cpu=perf_counter(),process_time();op=None;relative=None;status='maxiter';error=None;steps=0;x=None;setup=None;refreshes=[]
 try:
  op=Fused(X,b,bw,1e-5,helper);setup=perf_counter()-start
  y=np.zeros(len(b));s=np.zeros(op.m);op.budget.live=[y,s]
  r=op.rhs(b);op.budget.live=[y,s,r];z=op.precondition(r);p=z.copy();op.budget.observe('initial_direction_copy',z,p);rho=float(r@z);del z;op.budget.live=[y,s,r,p]
  threshold=.5e-8*np.linalg.norm(b)
  for steps in range(2001):
   if np.linalg.norm(r)<=threshold or steps==2000:
    x,relative=op.verify(y,b)
    if relative<=1e-8:status='converged';break
    del x;x=None
   if process_time()-cpu>cpu_cap:status='cpu_cap';break
   if steps==2000:break
   ap,q=op.apply_cheap(p);den=float(p@ap)
   if den<=0 or rho<=0:status='nonpositive_curvature';del ap,q;break
   alpha=rho/den;op.budget.observe('refresh_iterate_update',ap,q,temporary_allowance=y.nbytes);y+=alpha*p;r-=alpha*ap;del ap
   op.budget.observe('gap_accumulate',q,temporary_allowance=q.nbytes);s+=alpha*q;del q
   do=(steps+1)%8==0 if mode=='periodic8' else (np.linalg.norm(r)<=threshold if mode=='terminal' else op.error_bound*np.linalg.norm(s)>.1*np.linalg.norm(r))
   if do:op.refresh(r,s);refreshes.append(steps+1)
   z=op.precondition(r);new=float(r@z);op.budget.observe('refresh_direction_update',z,temporary_allowance=p.nbytes);p*=new/rho;p+=z;rho=new;del z
  if status!='converged':
   if x is not None:del x
   x,relative=op.verify(y,b)
   if relative<=1e-8:status='converged'
 except (MemoryError,np.linalg.LinAlgError) as e:status='memory_failure' if isinstance(e,MemoryError) else 'numerical_failure';error=str(e)
 row=dict(method=mode,status=status,converged=status=='converged',iterations=steps,original_relative_residual=relative,error=error,total_seconds=perf_counter()-start,total_cpu_seconds=process_time()-cpu,refresh_steps=refreshes)
 if op is not None:row.update(memory=op.budget.report(),kernel_stats=op.kernel.stats,actions=op.actions,verifications=op.verifications,core_solve_calls=op.factor.solve_calls,setup_seconds=setup,omitted=op.omitted)
 return x,row
