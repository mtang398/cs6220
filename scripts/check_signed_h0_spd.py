"""Schur SPD audit for the preceding signed-correction pilot."""
import json,resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import numpy as np
from scipy.linalg import eigh,solve_triangular,cho_solve,cholesky
from threadpoolctl import threadpool_limits
from check_core_precision_allocation import MixedFactor,Helper
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel,Budget

def main():
 out=ROOT/'results/signed-h0-spd.json'
 if out.exists():raise FileExistsError(out)
 resource.setrlimit(resource.RLIMIT_CPU,(30,35));start=perf_counter()
 old=json.loads((ROOT/'results/signed-h0-correction.json').read_text())
 for p,h in old['sources'].items():assert sha256((ROOT/p).read_bytes()).hexdigest()==h
 rec=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),pilot_sha256=sha256((ROOT/'results/signed-h0-correction.json').read_bytes()).hexdigest(),cases=[])
 helper=Helper(ROOT/'results/mixed_column_factor_first8.so')
 with threadpool_limits(limits=1):
  cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
  for name,filename in cases:
   path=ROOT/'data/prepared'/filename
   with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
   row=next(x for x in old['runs'] if x['dataset']==name)
   assert sha256(path.read_bytes()).hexdigest()==row['data_sha256']
   kernel=SharedKernel(X,bw,Budget([X,b]));J=kernel.farthest(584);R=np.setdiff1d(np.arange(len(b)),J);K=kernel.block(X,X)
   C=K[np.ix_(J,J)]+1e-5*np.eye(584);B=K[np.ix_(J,R)];D=K[np.ix_(R,R)]+1e-5*np.eye(len(R))
   factor=MixedFactor(C,np.arange(8),helper);L=factor.dense_L()
   E=solve_triangular(L,C,lower=True);E=solve_triangular(L,E.T,lower=True).T-np.eye(584);E=(E+E.T)*.5
   theta,V=eigh(E);coeff=-theta/(1+theta);selected=np.argsort(np.abs(coeff))[-4:];signs=np.sign(coeff[selected])
   assert np.allclose(theta[selected],row['oracle']['selected_theta'],rtol=1e-12,atol=1e-15)
   W=solve_triangular(L.T,V[:,selected],lower=False)*np.sqrt(abs(coeff[selected]))
   H0B=cho_solve((L,True),B);HcB=H0B+W@(signs[:,None]*(W.T@B))
   S=D-B.T@HcB;S=(S+S.T)*.5;passed=True
   try:cholesky(S,lower=True)
   except np.linalg.LinAlgError:passed=False
   rec['cases'].append(dict(dataset=name,cholesky_passed=passed,minimum_eigenvalue=float(eigh(S,eigvals_only=True,subset_by_index=[0,0])[0])))
 rec.update(status='complete',elapsed_wall_seconds=perf_counter()-start)
 out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');print(json.dumps(rec['cases']))
if __name__=='__main__':main()
