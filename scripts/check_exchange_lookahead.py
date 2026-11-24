"""Offline search for beneficial single swaps; dense A is only a diagnostic oracle."""
import json,resource,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from nystrom_project.numerics import GaussianKernel,NystromPreconditioner,initialize_rpcholesky

root=Path(__file__).resolve().parents[1]
out=root/'results/exchange-lookahead-diagnostic.json'
if out.exists():raise FileExistsError(out)
resource.setrlimit(resource.RLIMIT_CPU,(115,120))
t0=time.perf_counter()
record=dict(purpose='Offline finite-pool swap opportunity diagnostic; dense kernel, not a timed algorithm',status='running',cases=[])
def save():
    record['elapsed_wall_seconds']=time.perf_counter()-t0
    out.write_text(json.dumps(record,indent=2)+'\n')
def advance(A,M,b,x,r,p,rho,steps):
    x,r,p=x.copy(),r.copy(),p.copy();decreases=[]
    for it in range(steps):
        ap=A@p; alpha=rho/(p@ap)
        x+=alpha*p;r-=alpha*ap;decreases.append(float(alpha*rho))
        z=M.apply(r);newrho=r@z;p=z+(newrho/rho)*p;rho=newrho
    return x,r,p,rho,decreases
with threadpool_limits(limits=1):
    d=np.load(root/'data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz')
    b=d['y_train'];Kobj=GaussianKernel(d['X_train'],float(d['bandwidth']));K=Kobj.dense();A=K+.001*np.eye(len(b))
    for k in [16,32]:
        M=initialize_rpcholesky(Kobj,k,.001,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
        for checkpoint in [5,20]:
            x=np.zeros_like(b);r=b.copy();p=M.apply(r);rho=r@p
            x,r,p,rho,_=advance(A,M,b,x,r,p,rho,checkpoint)
            _,_,_,_,olddecr=advance(A,M,b,x,r,p,rho,10)
            oldgain=np.cumsum(olddecr)
            available=np.setdiff1d(np.arange(len(b)),M.indices)
            rng=np.random.default_rng(901)
            pool=np.unique(np.r_[rng.choice(available,24,replace=False),available[np.argsort(np.abs(r[available]))[-8:]]])
            row=dict(rank=k,checkpoint=checkpoint,pool_size=len(pool),old_continuation_gain5=float(oldgain[4]),old_continuation_gain10=float(oldgain[9]),trials=[])
            for j in pool:
                inds=np.append(M.indices,j);C=K[:,inds]
                for i in range(k):
                    take=np.arange(k+1)!=i
                    N=NystromPreconditioner(inds[take],C[:,take],.001)
                    z=N.apply(r)
                    _,_,_,_,ds=advance(A,N,b,x,r,z,r@z,10)
                    gain=np.cumsum(ds)
                    row['trials'].append(dict(add=int(j),remove=int(M.indices[i]),gain5=float(gain[4]),gain10=float(gain[9]),q=float(r@z)))
            row['better5']=int(sum(v['gain5']>oldgain[4]*(1+1e-8) for v in row['trials']))
            row['better10']=int(sum(v['gain10']>oldgain[9]*(1+1e-8) for v in row['trials']))
            row['best5']=max(row['trials'],key=lambda v:v['gain5'])
            row['best10']=max(row['trials'],key=lambda v:v['gain10'])
            record['cases'].append(row);save()
            print(json.dumps({key:v for key,v in row.items() if key!='trials'}),flush=True)
record['status']='complete';save()
