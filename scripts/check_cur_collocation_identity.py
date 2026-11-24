"""Tiny algebra check of CUR row selection; not a solver performance experiment."""
import json
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import numpy as np
from scipy.linalg import solve,qr
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'results/cur-collocation-identity.json';assert not out.exists();start=perf_counter();rng=np.random.default_rng(622151);rows=[]
 with threadpool_limits(limits=1):
  for exactrank in [False,True]:
   F=rng.normal(size=(19,6));K=F@F.T
   if not exactrank:K+=.2*np.eye(19)
   lam=.03;A=K+lam*np.eye(19);J=np.arange(6);C=K[:,J];AC=A@C;r=rng.normal(size=19);target=solve(A,r,assume_a='pos');G=C.T@AC;ygal=solve(G,(AC-lam*C).T@r,assume_a='pos');hgal=(r-C@ygal)/lam
   energy=lambda h:float((h-target)@A@(h-target));reference=None;errors=[]
   for repeat in range(12):
    # QRCP of a randomly row-scaled C gives diverse invertible cross cores.
    weights=np.exp(rng.normal(size=19));_,_,piv=qr((weights[:,None]*C).T,mode='economic',pivoting=True);I=piv[:6];W=K[np.ix_(I,J)];R=K[I,:];core=lam*W+R@C
    hwood=(r-C@solve(core,R@r))/lam
    hcoll=(r-C@solve(AC[I,:],(K@r)[I]))/lam
    M=lam*np.eye(19)+C@solve(W,R);hdirect=solve(M,r)
    identity=float(np.linalg.norm(hwood-hcoll)/max(np.linalg.norm(hcoll),1e-300));inverse=float(np.linalg.norm(M@hwood-r)/np.linalg.norm(r));direct=float(np.linalg.norm(hwood-hdirect)/max(np.linalg.norm(hdirect),1e-300));orth=float(np.linalg.norm(C.T@(A@hgal-r))/max(np.linalg.norm(C.T@r),1e-300))
    assert max(identity,inverse,direct,orth)<1e-8
    gap=energy(hwood)-energy(hgal);assert gap>=-1e-8*max(energy(hwood),energy(hgal),1)
    if reference is None:reference=hwood
    variation=float(np.linalg.norm(hwood-reference)/max(np.linalg.norm(reference),1e-300))
    if exactrank:assert variation<1e-8
    errors.append(dict(collocation_identity_relative_error=identity,woodbury_inverse_relative_residual=inverse,direct_inverse_relative_error=direct,galerkin_stationarity_relative_error=orth,energy_gap_over_galerkin=gap,row_change_relative_action=variation,rows=I.tolist()))
   rows.append(dict(exact_rank_six=exactrank,checks=errors))
 record=dict(status='complete',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Tiny algebra only. Fixed C; CUR inverse actions lie in lambda^-1*r+range(C). Galerkin minimizes one-action A-energy error there. No outer-GMRES convergence domination or kernel benchmark claim.',cases=rows,elapsed_seconds=perf_counter()-start)
 out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print('24 collocation/energy checks passed; exact-rank row invariance passed.')
if __name__=='__main__':main()
