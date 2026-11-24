"""Diagnose the already-tested HSS matrix; no candidate performance claim."""
import json, resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import eigh, solve_triangular
from threadpoolctl import threadpool_limits
from check_scaled_hss_headroom import partition, planner, construct, ROOT, SharedKernel, Budget


def main():
    out=ROOT/'results/hss-spectrum.json'
    if out.exists(): raise FileExistsError(out)
    start=perf_counter();resource.setrlimit(resource.RLIMIT_CPU,(80,85))
    old=json.loads((ROOT/'results/scaled-hss-headroom.json').read_text())
    files=[Path(__file__).resolve(),ROOT/'scripts/check_scaled_hss_headroom.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    rec={'status':'running','sources':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},
         'protocol':{'question':'Are failures explained by a few isolated generalized eigenvalues or a broad spectrum?',
                     'scope':'Exact dense generalized eigenanalysis of the previous four development cases. No runtime, memory feasibility, or adaptive-method claim.',
                     'spectral_correction':'Replace r extreme eigenvalues in an exact generalized eigenbasis. Minimum remaining condition number is min_j lambda[j+n-r-1]/lambda[j]. This does not optimize CG iterations.',
                     'budgets':'120 wall seconds, 80 CPU seconds, BLAS1; charged to existing campaign'},'cases':[]}
    def save():
        rec['elapsed_wall_seconds']=perf_counter()-start
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
    with threadpool_limits(limits=1):
        for name,filename in cases:
            path=ROOT/'data/prepared'/filename
            with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
            previous=next(c for c in old['cases'] if c['dataset']==name)
            assert sha256(path.read_bytes()).hexdigest()==previous['data_sha256']
            perm,levels=partition(X);plan,_=planner(len(b),levels)
            A=SharedKernel(X,bw,Budget([X,b])).block(X,X);A.flat[::len(b)+1]+=1e-5
            A=A[np.ix_(perm,perm)];b=b[perm];M,L,_=construct(A,levels,plan['rank'])
            matrix_error=float(np.linalg.norm(M-A)/np.linalg.norm(A))
            assert abs(matrix_error-previous['relative_matrix_error'])<1e-12
            C=solve_triangular(L,A,lower=True);C=solve_triangular(L,C.T,lower=True).T
            vals,V=eigh((C+C.T)*.5);assert vals[0]>0
            rhs=solve_triangular(L,b,lower=True);weights=(V.T@rhs)**2/vals
            windows={}
            for rank in [0,4,16,32]:
                n=len(vals);ratios=vals[n-rank-1:]/vals[:rank+1];j=int(np.argmin(ratios))
                windows[str(rank)]={'best_condition_number':float(ratios[j]),'remove_low':j,'remove_high':rank-j,
                    'removed_initial_energy_fraction':float((sum(weights[:j])+sum(weights[n-rank+j:]))/sum(weights)),
                    'dense_vector_storage_lower_bound_bytes':8*n*rank}
            needed={}
            for target in [10,100,1000]:
                widths=np.searchsorted(vals,vals*target,side='right')-np.arange(len(vals))
                needed[str(target)]=int(len(vals)-max(widths))
            row={'dataset':name,'n':len(b),'rank':plan['rank'],'relative_matrix_error':matrix_error,
                 'eigenvalue_quantiles':dict(zip(['min','p1','p10','median','p90','p99','max'],map(float,np.quantile(vals,[0,.01,.1,.5,.9,.99,1])))),
                 'ideal_spectral_corrections':windows,'minimum_eigenmode_replacements_for_condition_number':needed,
                 'initial_energy_fraction_below_0_1':float(sum(weights[vals<.1])/sum(weights)),
                 'initial_energy_fraction_above_10':float(sum(weights[vals>10])/sum(weights))}
            rec['cases'].append(row);save();print(json.dumps(row),flush=True)
    rec['status']='complete';save()

if __name__=='__main__':main()
