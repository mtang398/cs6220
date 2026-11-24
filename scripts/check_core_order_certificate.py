"""Frozen exact-Schur trajectory audit of current-direction core-order certificate."""
import json,resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import cho_factor,cho_solve,solve_triangular,eigvalsh
from threadpoolctl import threadpool_limits
from check_packed_float_factor import PackedFloat,unpack,ROOT
from check_packed_float_allocation import RawPointers

def main():
 out=ROOT/'results/core-order-certificate.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(55,60));start,cpu=perf_counter(),process_time();helper=RawPointers(ROOT/'results/packed_float_factor.so')
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),protocol=dict(core=600,probe_count=6,probe_multiplier=10*np.sqrt(2/np.pi),rtol=1e-8,allocated_gap_fraction=.25,scope='Dense offline whitening/spectral audits and exact future trajectory length free. Not adaptive recurrence or fullmemory/timing claim',selection='First-pass actual H1 curvature; error bound delta*sqrt(ntail/(1−delta)*h0^TH0h0); use H1 if alpha1*bound<=gap/N, otherwise H1 notaccepted; H3 isnotcertifiedbythisscreen'),runs=[])
 with threadpool_limits(limits=1):
  for ds in ['smooth','mixed']:
   path=ROOT/'data/prepared'/f'synthetic-n2560-{ds}-s0-lambda0.001.npz'
   with np.load(path) as q:X,b,bw=q['X_train'],q['y_train'],float(q['bandwidth'])
   for lam in [.001,1e-5]:
    f=PackedFloat(X,b,bw,600,lam,helper);J=f.indices;R=np.setdiff1d(np.arange(len(b)),J);K=f.kernel.block(X,X);C=K[np.ix_(J,J)]+lam*np.eye(600);B=K[np.ix_(J,R)];cf=cho_factor(C,lower=True);S=K[np.ix_(R,R)]+lam*np.eye(len(R))-B.T@cho_solve(cf,B);L=unpack(f.factor,600)
    T=solve_triangular(L,C,lower=True);W=solve_triangular(L,T.T,lower=True).T;E=(W+W.T)*.5-np.eye(600)
    spectrum=eigvalsh(E);actual_delta=float(max(abs(spectrum[0]),abs(spectrum[-1])))
    probes=np.random.default_rng(622108).normal(size=(600,6));norms=np.linalg.norm(E@probes,axis=0);delta=float(10*np.sqrt(2/np.pi)*norms.max())
    rhs=b[R]-B.T@cho_solve(cf,b[J]);y=np.zeros(len(R));r=rhs.copy();p=r.copy();rr=float(r@r);nb=np.linalg.norm(b);trace=[]
    for it in range(2000):
     full=np.empty(len(b));full[R]=y;full[J]=cho_solve(cf,b[J]-B@y)
     original=float(np.linalg.norm(b-K@full-lam*full)/nb)
     if original<=1e-8:break
     sp=S@p;alpha=rr/float(p@sp);trace.append((p.copy(),rr));y+=alpha*p;r-=alpha*sp;new=float(r@r);p=r+(new/rr)*p;rr=new
    assert original<=1e-8
    allowed=.25e-8*nb/len(trace);hist=[]
    for p,rr in trace:
     t=B@p;z0=f.solve(t);h0=t-C@z0;w0=f.solve(h0);y1=z0+w0
     base_curvature=float(p@(K[np.ix_(R,R)]@p+lam*p));denom=base_curvature-float(t@y1);alpha1=rr/denom
     energy=max(float(h0@w0),0.);bound=float(delta*np.sqrt(len(R)/(1-delta)*energy)) if delta<1 else None
     zref=cho_solve(cf,t);actual_error=float(np.linalg.norm(B.T@(zref-y1)));accept=bound is not None and alpha1*bound<=allowed
     hist.append(dict(accept_one=bool(accept),alpha1=float(alpha1),bound=bound,actual_action_error=actual_error,bound_increment=None if bound is None else float(alpha1*bound),actual_increment=float(alpha1*actual_error)))
    rec['runs'].append(dict(dataset=ds,regularization=lam,iterations=len(trace),true_relative_residual=original,actual_whitened_delta=actual_delta,probe_delta=delta,probe_covers_actual=delta>=actual_delta,accept_one_count=sum(x['accept_one'] for x in hist),actual_error_oracle_accepts=int(sum(x['actual_increment']<=allowed for x in hist)),allowed_increment=float(allowed),history=hist))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))
if __name__=='__main__':main()
