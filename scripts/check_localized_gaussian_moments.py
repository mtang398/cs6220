"""Actual search-vector global/local Gaussian moment accuracy and work proxies."""
import json,resource,math,itertools
from pathlib import Path
from hashlib import sha256
from time import perf_counter,process_time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import check_refresh_regularization_scope as scope
import check_kernel_accounting_hoist as hoist
from check_schur_inplace_frontier_v2 import ROOT,SharedKernel

def digest(p):return sha256(p.read_bytes()).hexdigest()
def polynomial(X,Y,c,p):
 x=X-c;y=Y-c;g=x@y.T;term=np.ones_like(g);poly=term.copy()
 for k in range(1,p+1):term*=g/k;poly+=term
 return np.exp(-np.sum(x*x,axis=1)/2)[:,None]*poly*np.exp(-np.sum(y*y,axis=1)/2)[None,:]
def tiny():
 rng=np.random.default_rng(28);X=rng.normal(size=(7,3))*.2;Y=rng.normal(size=(8,3))*.2;c=Y.mean(0);v=rng.normal(size=8);p=3
 alphas=[a for a in itertools.product(range(p+1),repeat=3) if sum(a)<=p]
 def features(Z):
  Z=Z-c;return np.exp(-np.sum(Z*Z,axis=1)/2)[:,None]*np.array([np.prod(Z**np.array(a),axis=1)/math.sqrt(math.prod(math.factorial(k) for k in a)) for a in alphas]).T
 explicit=features(X)@(features(Y).T@v);pair=polynomial(X,Y,c,p)@v;error=float(np.linalg.norm(explicit-pair));assert error<1e-12;return dict(absolute_error=error,feature_count=len(alphas),binomial_count=math.comb(3+p,p))
def partition(Y):
 centers=[int(np.argmin(np.sum((Y-Y.mean(0))**2,axis=1)))];distance=np.sum((Y-Y[centers[0]])**2,axis=1)
 for _ in range(3):
  j=int(np.argmax(distance));centers.append(j);distance=np.minimum(distance,np.sum((Y-Y[j])**2,axis=1))
 labels=np.argmin(cdist(Y,Y[centers],'sqeuclidean'),axis=1)
 return [(np.flatnonzero(labels==j),Y[centers[j]]) for j in range(4) if np.any(labels==j)]
def main():
 out=ROOT/'results/localized-gaussian-moments.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(80,85));t=perf_counter();cpu=process_time()
 files=scope.source_paths()+[Path(__file__).resolve(),ROOT/'scripts/check_taylor_inexact_cg_diagnostic.py',ROOT/'scripts/check_taylor_krylov_cost_oracle.py']
 priorpaths=[ROOT/f'results/refresh-scope/{name}-lambda1e-05.json' for name in ['smooth','mixed','concrete','airfoil']];files+=priorpaths
 result=dict(status='running',sources={str(p.relative_to(ROOT)):digest(p) for p in files},tiny=tiny(),protocol=dict(representation='Global source mean or four deterministic FPS source centers; Gaussian Taylor degrees2/3/4. Uncorrected kernel expansion, no diagonal correction.',captures='Read-only copies of exact packed24core675 congruence product inputs under common hoisted kernel. First/middle/last selected AFTER run, zero-based indices. Dense captures/reference computations are free offline oracle, no native timing/memory evidence.',bound='norm(abs(K-Kp) @ abs(v)) is an offline oracle triangle bound, not a cheap computable certificate.',work='Feature visits and pair counts are heterogeneous work proxies, not equivalent FLOPs/runtime. Partition, source coordinate scaling, feature generation costs and scratch excluded. Moments can stream one cluster. No fixed-memory proof.',scope='Four development hard cases only, not small-n/held-out validation or a solver policy.'),cases=[])
 def save():
  result.update(elapsed_wall_seconds=perf_counter()-t,elapsed_cpu_seconds=process_time()-cpu);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 with threadpool_limits(limits=1):
  for name,priorpath in zip(['smooth','mixed','Concrete','Airfoil'],priorpaths):
   prior=json.loads(priorpath.read_text());assert all(digest(ROOT/p)==h for p,h in prior['sources'].items())
   path=ROOT/'data/prepared'/scope.DATASETS[name];assert digest(path)==prior['data_sha256']
   with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
   X.setflags(write=False);b.setflags(write=False);captures={label:[] for label in ['fused_first_pass','fused_core','fused_second_pass']}
   with hoist.variant('hoisted'):
    base=SharedKernel.product
    def capture(self,rows,cols,v,**kwargs):
     label=kwargs['label']
     if label in captures:captures[label].append((None if rows is None else rows.copy(),None if cols is None else cols.copy(),v.copy()))
     return base(self,rows,cols,v,**kwargs)
    SharedKernel.product=capture
    try:answer,row=scope.packed.solve(X,b,bw,scope.Helper24(ROOT/'results/packed_int24_factor.so'),cpu_cap=7);del answer
    finally:SharedKernel.product=base
   expected=[r for r in prior['runs'] if r['method']=='packed24'];checks={k:all(row[k]==r[k] for r in expected) for k in ['iterations','kernel_stats','original_relative_residual','converged']};assert all(checks.values())
   case=dict(dataset=name,n=len(b),dimension=X.shape[1],data_sha256=digest(path),baseline=row,baseline_checks=checks,products=[]);result['cases'].append(case)
   for label,history in captures.items():
    selected=sorted(set([0,len(history)//2,len(history)-1]))
    samples=[(i,history[i]) for i in selected];captures[label]=[]
    for index,(ri,ci,v) in samples:
     targets=(X if ri is None else X[ri])/bw;sources=(X if ci is None else X[ci])/bw;nt,ns=len(targets),len(sources);K=np.exp(-cdist(targets,sources,'sqeuclidean')/2);exact=K@v;normv=float(np.linalg.norm(v));normexact=float(np.linalg.norm(exact))
     for representation,groups in [('global',[(np.arange(ns),sources.mean(0))]),('localized',partition(sources))]:
      for degree in [2,3,4]:
       approximation=np.zeros(nt);triangle=np.zeros(nt)
       for ids,center in groups:
        P=polynomial(targets,sources[ids],center,degree);approximation+=P@v[ids];triangle+=np.abs(K[:,ids]-P)@np.abs(v[ids])
       error=float(np.linalg.norm(approximation-exact));bound=float(np.linalg.norm(triangle));features=math.comb(X.shape[1]+degree,degree);visits=sum((len(ids)+nt)*features for ids,_ in groups);moments=8*features*len(groups)
       case['products'].append(dict(label=label,capture_index_zero_based=index,capture_count=len(history),representation=representation,degree=degree,source_count=ns,target_count=nt,clusters=len(groups),feature_count=features,feature_visits=visits,exact_pair_count=ns*nt,feature_visit_pair_ratio=visits/(ns*nt),moment_bytes_all_clusters=moments,moment_bytes_one_cluster=8*features,incomplete_factor_fivevector_moment_bytes=720109+40*len(b)+moments,exact_product_norm=normexact,vector_norm=normv,absolute_error=error,relative_product_error=error/max(normexact,1e-300),error_per_vector_norm=error/max(normv,1e-300),oracle_triangle_bound=bound,signed_to_triangle_ratio=error/max(bound,1e-300)))
   print(name,row['iterations'],len(case['products']),flush=True);save()
 result['sources_unchanged']=all(digest(ROOT/p)==h for p,h in result['sources'].items());assert result['sources_unchanged'];result['status']='complete';save()
if __name__=='__main__':main()
