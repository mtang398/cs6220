"""Free-reference/calibration headroom for solve-conditioned factor rounding."""
import ctypes,json,resource,subprocess
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular,svdvals
from threadpoolctl import threadpool_limits
from check_int24_precision_selection import run
from check_int16_core_capacity import quantize
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def main():
 out=ROOT/'results/calibrated-rounding-inverse-diagnosis.json';lib=ROOT/'results/calibrated_factor_rounding_bounded.so';src=ROOT/'scripts/calibrated_factor_rounding_bounded.c'
 if out.exists():raise FileExistsError('Preserve evidence')
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));t,cpu=perf_counter(),process_time();command=['gcc','-O3','-ffp-contract=off','-shared','-fPIC',str(src),'-lm','-o',str(lib)];assert lib.exists()
 dll=ctypes.CDLL(str(lib));fn=dll.calibrated_rounding;fn.argtypes=[ctypes.c_size_t]*2+[ctypes.c_void_p]*7;fn.restype=ctypes.c_size_t
 files=[Path(__file__).resolve(),src,lib,ROOT/'scripts/check_int24_precision_selection.py',ROOT/'scripts/check_int16_core_capacity.py']
 rec=dict(diagnostic_of='results/calibrated-factor-rounding-bounded.json',status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},protocol=dict(core=818,comparison_core=600,shift=1e-5,rounding='Two coordinate-flip sweeps within floor/ceil of each exact offdiagonal entry on fixed signedint16 scale, candidates explicitly restricted to symmetric[-32767,32767]. First8 columns and diagonal unchanged. Minimize ||(Q-L)V||F²+||(Q-L)^T U||F². U from exact core inverse applied to [rS,B rR]; V=L^TU; columns normalized independently. Same4step nearest-rounding prefix.',scope='All dense reference construction, calibration, scoring and re-encoding free. Payload720442bytes unchanged; no fullmemory/native claim. Compare initialRHS/currentresidual calibration, matched nearestrestart and upfrontFP32core600. First-order action calibration is not GPTQ implementation or novelty claim.'),cases=[])
 def save():
  rec.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 with threadpool_limits(limits=1):
  for name,filename in [('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]:
   path=ROOT/'data/prepared'/filename
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(818);Ridx=np.setdiff1d(np.arange(len(b)),S);A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5;C=A[np.ix_(S,S)];B=A[np.ix_(S,Ridx)];L=np.ascontiguousarray(cholesky(C,lower=True));Q=quantize(L,'int16');prefix,pr=run(A,b,S,Q,stop_steps=4);assert pr['status']=='prefix';residual=b-A@prefix
   scales=np.zeros(818)
   for j in range(8,817):scales[j]=np.max(np.abs(L[j+1:,j]))/32767
   case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),prefix=pr,runs=[])
   for mode in ['nearest_restart','initial_restart','current_restart']:
    QQ=Q.copy();SS=S;xx=prefix;cal={}
    if mode=='uniform600_upfront':SS=S[:600];QQ=cholesky(A[np.ix_(SS,SS)],lower=True).astype(np.float32).astype(float);xx=None
    elif mode!='nearest_restart':
     rhs=b if mode=='initial_restart' else residual;U=cho_solve((L,True),np.column_stack([rhs[S],B@rhs[Ridx]]),check_finite=False);V=L.T@U
     U=np.ascontiguousarray(U/np.maximum(np.linalg.norm(U,axis=0),1e-300));V=np.ascontiguousarray(V/np.maximum(np.linalg.norm(V,axis=0),1e-300));E=QQ-L;rr=np.ascontiguousarray(E@V);tt=np.ascontiguousarray(E.T@U);before=float(np.sum(rr*rr)+np.sum(tt*tt))
     pointers=[a.ctypes.data for a in [L,QQ,scales,U,V,rr,tt]]
     # Seven pointer buffers follow n,d.
     fn.argtypes=[ctypes.c_size_t]*2+[ctypes.c_void_p]*7
     flips=fn(818,2,*pointers);after=float(np.sum(rr*rr)+np.sum(tt*tt));actual=float(np.linalg.norm((QQ-L)@V)**2+np.linalg.norm((QQ-L).T@U)**2)
     assert after<=before*(1+1e-10) and abs(actual-after)<=max(1e-25,1e-8*before)
     maxcode=0.;mincode=0.
     for j in range(8,817):
      if scales[j]>0:
       codes=np.rint(QQ[j+1:,j]/scales[j]);maxcode=max(maxcode,float(np.max(codes)));mincode=min(mincode,float(np.min(codes)))
     assert maxcode<=32767 and mincode>=-32767
     cal=dict(max_code=maxcode,min_code=mincode,flips=flips,calibration_before=before,calibration_after=actual,calibration_ratio=actual/before)
    Z=solve_triangular(QQ,L,lower=True,check_finite=False);singular=svdvals(Z,check_finite=False)
    inverse_errors={}
    for target,rrhs in [('initial',b),('current',residual)]:
     rhs=np.column_stack([rrhs[S],B@rrhs[Ridx]]);reference=cho_solve((L,True),rhs,check_finite=False);got=cho_solve((QQ,True),rhs,check_finite=False)
     inverse_errors[target]=[float(np.linalg.norm(got[:,j]-reference[:,j])/max(np.linalg.norm(reference[:,j]),1e-300)) for j in range(2)]
    row=dict(method=mode,factor_relative_error=float(np.linalg.norm(QQ-L)/np.linalg.norm(L)),inverse_factor_error=float(np.linalg.norm(Z-np.eye(818),'fro')),core_generalized_condition=float((singular[0]/singular[-1])**2),inverse_action_relative_errors=inverse_errors,**cal)
    if mode!='nearest_restart':
     prior=json.loads((ROOT/'results/calibrated-factor-rounding-bounded.json').read_text());old=next(x for c in prior['cases'] if c['dataset']==name for x in c['runs'] if x['method']==mode)
     assert old['flips']==flips and abs(old['calibration_after']-actual)<=1e-12*max(actual,1e-300)
    case['runs'].append(row);print(json.dumps(dict(dataset=name,**row)),flush=True)
   rec['cases'].append(case);save()
 rec['status']='complete';save()
if __name__=='__main__':main()
