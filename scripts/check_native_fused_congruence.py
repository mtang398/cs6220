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
from check_schur_inplace_frontier_v2 import ROOT,LIMIT

class Fused:
 def __init__(self,X,b,bw,lam,helper):
  self.factor=scalar.StreamedFirst8(X,b,bw,592,lam,helper);self.budget=self.factor.budget;self.kernel=self.factor.kernel
  self.S,self.R=self.factor.S,self.factor.R;self.permutation=self.factor.permutation;self.m=592;self.n=len(b);self.lam=lam
  self.actions=0;self.verifications=0
 def rhs(self,b):
  r=b[self.permutation]
  with self.budget.hold(r):
   h=self.factor.solve(r[:self.m])
   with self.budget.hold(h):self.kernel.product(self.R,self.S,h,label='fused_rhs',target=r[self.m:],subtract=True)
  return r
 def apply(self,p):
  m=self.m
  values=self.kernel.product(None,self.R,p[m:],label='fused_first_pass')
  out=values[self.permutation];self.budget.observe('fused_reorder',values,out);del values
  with self.budget.hold(out):
   h=self.factor.solve(out[:m]);q=p[:m]-h;self.budget.observe('fused_q',h,q);del h
   with self.budget.hold(q):
    a=self.kernel.product(self.S,self.S,q,label='fused_core')
    self.budget.observe('fused_core_shift',a,temporary_allowance=q.nbytes);a+=self.lam*q;a+=out[:m]
    with self.budget.hold(a):ha=self.factor.solve(a)
    self.budget.observe('fused_second_direction',a,ha);q-=ha;del ha
    out[:m]=a;del a;q*=-1
    self.kernel.product(self.R,self.S,q,label='fused_second_pass',target=out[m:],subtract=True)
   del q
   self.budget.observe('fused_tail_shift',temporary_allowance=out[m:].nbytes);out[m:]+=self.lam*p[m:]
  self.actions+=1;return out
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

def solve(X,b,bw,helper,cpu_cap=6):
 start,cpu=perf_counter(),process_time();op=None;relative=None;status='maxiter';error=None;steps=0;x=None
 try:
  op=Fused(X,b,bw,1e-5,helper);setup=perf_counter()-start;y=np.zeros(len(b));op.budget.live=[y]
  r=op.rhs(b);op.budget.live=[y,r];z=op.precondition(r);p=z.copy();rho=float(r@z);del z;op.budget.live=[y,r,p];threshold=.5e-8*np.linalg.norm(b)
  for steps in range(2001):
   if np.linalg.norm(r)<=threshold or steps==2000:
    x,relative=op.verify(y,b)
    if relative<=1e-8:status='converged';break
    del x;x=None
   if process_time()-cpu>cpu_cap:status='cpu_cap';break
   if steps==2000:break
   ap=op.apply(p);den=float(p@ap)
   if den<=0 or rho<=0:status='nonpositive_curvature';break
   alpha=rho/den;op.budget.observe('fused_iterate_update',ap,temporary_allowance=y.nbytes);y+=alpha*p;r-=alpha*ap;del ap
   z=op.precondition(r);new=float(r@z);op.budget.observe('fused_direction_update',z,temporary_allowance=p.nbytes);p*=new/rho;p+=z;del z;rho=new
  if relative is None:x,relative=op.verify(y,b)
 except (MemoryError,np.linalg.LinAlgError) as e:status='memory_failure' if isinstance(e,MemoryError) else 'numerical_failure';error=str(e)
 row=dict(method='fused_H0',status=status,converged=status=='converged',iterations=steps,original_relative_residual=relative,error=error,total_seconds=perf_counter()-start,total_cpu_seconds=process_time()-cpu)
 if op is not None:row.update(memory=op.budget.report(),kernel_stats=op.kernel.stats,actions=op.actions,verifications=op.verifications,core_solve_calls=op.factor.solve_calls,core_solve_seconds=op.factor.solve_seconds,setup_seconds=setup)
 return x,row

def main():
 out=ROOT/'results/native-fused-congruence.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
 helper=Helper(ROOT/'results/mixed_column_factor_first8.so');raw=RawPointers(ROOT/'results/packed_float_factor.so')
 sources=[Path(__file__).resolve(),Path(scalar.__file__),ROOT/'scripts/check_schur_inplace_frontier_v2.py',ROOT/'scripts/check_compressed_core_schur.py',ROOT/'results/mixed_column_factor_first8.so',ROOT/'results/packed_float_factor.so']
 rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},protocol=dict(core=592,shift=1e-5,seed=0,repeats=1,methods=['fused_H0','original_exact416','FIRST8_H1_or_prefix'],stop='Suspected transformed residual<=.5e-8||b|| triggers charged reconstruction and literal original residual check at1e-8. No uncharged intermediate verification.',memory='Owned array accounting with64KiB reserve, independent full solve tracemalloc; immutable X,b excluded.',scope='Static native feasibility pilot. No adaptive novelty or research success claim.'),audits=[],runs=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
 with threadpool_limits(limits=1):
  for name,fn in [cases[0],cases[3]]:
   with np.load(ROOT/'data/prepared'/fn) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   tracemalloc.start()
   try:
    answer,row=solve(X,b,bw,helper,8);_,peak=tracemalloc.get_traced_memory();del answer
   finally:tracemalloc.stop()
   rec['audits'].append(dict(dataset=name,traced_peak_bytes=peak,metrics=row));save()
   if peak>LIMIT or row['status']=='memory_failure':rec['status']='allocation_gate_failed';save();return
  for index,(name,fn) in enumerate(cases):
   path=ROOT/'data/prepared'/fn
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   methods=['fused_H0','original_exact416','FIRST8_H1' if index<2 else 'FIRST8_H2prefix4_then_H1'];methods=methods[index%3:]+methods[:index%3]
   for method in methods:
    if process_time()-cpu>71:rec['status']='cpu_budget';save();return
    if method=='fused_H0':answer,row=solve(X,b,bw,helper)
    elif method=='original_exact416':answer,row=scalar.exact_control.run(X,b,bw,1e-5,method,raw,deadline=cpu+78)
    else:answer,row=scalar.solve(X,b,bw,1e-5,method,helper,raw,deadline=cpu+78)
    del answer;row.update(dataset=name,method=method,data_sha256=sha256(path.read_bytes()).hexdigest());rec['runs'].append(row);save();print(json.dumps({k:row.get(k) for k in ['dataset','method','status','total_seconds','iterations','original_relative_residual']}),flush=True)
 rec['status']='complete';save()
if __name__=='__main__':main()
