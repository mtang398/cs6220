"""Exact paired energy exchange and matched old-core coordinate correction control."""
import json,resource
from pathlib import Path
from time import perf_counter,process_time
from hashlib import sha256
import numpy as np
from scipy.linalg import cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_large_core_exchange_oracle import ROOT,build,solve

def main():
 out=ROOT/'results/large-core-exchange-energy.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60));t,c=perf_counter(),process_time();prior=json.loads((ROOT/'results/large-core-exchange-oracle.json').read_text())
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(core=416,checkpoint=10,selection='j=argmax r_j²/Sjj; i=argmin (C^-1 A_Sj)_i²/(C^-1)ii',controls='same j, exact old-core Schur-coordinate correction then identical restart',free='same dense/rebuild/selection oracle as parent; no native claim'),runs=[])
 with threadpool_limits(limits=1):
  for ds in ['smooth','mixed']:
   path=ROOT/'data/prepared'/f'synthetic-n2560-{ds}-s0-lambda0.001.npz'
   with np.load(path) as z:X,b,bw=z['X_train'],z['y_train'],float(z['bandwidth'])
   K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));J=[0];dist=2-2*K[:,0]
   for _ in range(415):
    j=int(np.argmax(dist));J.append(j);dist=np.minimum(dist,2-2*K[:,j])
   J=np.asarray(J)
   for lam in [.001,1e-5]:
    A=K+lam*np.eye(len(K));op=build(A,b,J);baseline,state=solve(A,b,J,op,checkpoint=10);R,F,B,V,D,rhs=op
    old=next(x for x in prior['runs'] if x['dataset']==ds and x['regularization']==lam);assert baseline['iterations']==old['baseline']['iterations']
    residual=b-A@state;r=residual[R];diag=np.diag(D);j=int(np.argmax(r*r/diag));v=V[:,j];h=np.diag(cho_solve(F,np.eye(len(J))));i=int(np.argmin(v*v/h));pivot=diag[j]+v[i]**2/h[i]
    newJ=J.copy();newJ[i]=R[j];newop=build(A,b,newJ);swapped,_=solve(A,b,newJ,newop,state)
    warm=state.copy();delta=r[j]/diag[j];warm[R[j]]+=delta;warm[J]-=v*delta;control,_=solve(A,b,J,op,warm)
    # Reconstruct swapped initial state and verify the immediate energy identity independently.
    RR,FF,BB,_,_,_=newop;projected=state.copy();projected[newJ]=cho_solve(FF,b[newJ]-BB@state[RR]);d=projected-state;observed=float(residual@d-.5*d@(A@d));predicted=float(r[j]**2/(2*pivot))
    assert abs(observed-predicted)<=1e-8*max(abs(predicted),1e-20)
    rec['runs'].append(dict(dataset=ds,regularization=lam,baseline=baseline,swapped=swapped,old_core_coordinate_control=control,swapped_total_iterations=10+swapped['iterations'],coordinate_control_total_iterations=10+control['iterations'],added=int(R[j]),removed=int(J[i]),swapped_energy_gain=predicted,measured_swapped_energy_gain=observed,old_core_energy_gain=float(r[j]**2/(2*diag[j])),conditional_correlation_squared=float(v[i]**2/(h[i]*diag[j]+v[i]**2))))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-c);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
