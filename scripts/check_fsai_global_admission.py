"""Dense diagnostic with explicit column accounting for global FSAI admission.

Selection sees only requested Schur columns and the available diagonal. Dense
S is an offline oracle behind that interface, not free selection information.
"""
from hashlib import sha256
import heapq
import json
from pathlib import Path
import resource
from time import process_time,perf_counter

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from check_schur_fsai import (GaussianKernel,initialize_rpcholesky,schur_blocks,
                             build_fsai,SchurFSAI,pcg)
from check_fsai_pattern_exchange import checkpoint,canonical

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/fsai-global-admission-diagnostic.json'
STATIC=ROOT/'results/schur-fsai-diagnostic.json'
PRIOR=ROOT/'results/fsai-global-admission-diagnostic-attempt1.json'
START_CPU,START_WALL=process_time(),perf_counter()
PREFIX,BUDGET,ROWS,CANDIDATES=8,8,8,4
record=dict(status='running',purpose='Offline nonlocal admission diagnostic, not a production timing claim',
    protocol=dict(checkpoint=PREFIX,row_nnz=BUDGET,rows_per_fair_method=ROWS,
        admission_candidates=CANDIDATES,canonical_fixed_nnz=True,
        global_admission='largest conditional variance reduction using current-neighbor conditional denominator',
        local_quality='sigma_i/S_ii=1/(G_ii**2*S_ii), remaining conditional variance fraction',
        current_priority='abs(Gq)**2',uniform_seed=[0,6220,93],
        fair_schur_column_queries=ROWS*BUDGET,
        static_oracle='all eligible rows, Concrete k32 lambda.001 only, EXTRA QUERY upper-opportunity diagnostic',
        current_rhs_selection_products=0,blas_threads=1,cpu_limit_seconds=89),
    source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).resolve(),ROOT/'scripts/check_fsai_pattern_exchange.py',
         ROOT/'scripts/check_schur_fsai.py',ROOT/'scripts/check_projected_balancing.py']},
    static_results_sha256=sha256(STATIC.read_bytes()).hexdigest(),cases=[])
if PRIOR.exists():
    record['cases']=json.loads(PRIOR.read_text())['cases']
    record['continued_from']=dict(path=str(PRIOR.relative_to(ROOT)),sha256=sha256(PRIOR.read_bytes()).hexdigest(),
        reason='First attempt stopped on numpy row-index JSON serialization in extra-query oracle',
        change='Convert oracle row index to Python int; no scientific change; completed branches preserved')


def save():
    record.update(cpu_seconds=process_time()-START_CPU,wall_seconds=perf_counter()-START_WALL,
                  offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def budget():
    if process_time()-START_CPU>85:raise TimeoutError('85 CPU-second graceful budget reached')


class Columns:
    def __init__(self,S,original_n):
        self.S=S;self.n=original_n;self.count=0;self.unique=set();self.max_workspace=0
    def query(self,J):
        self.count+=len(J);self.unique.update(map(int,J))
        result=self.S[:,J].copy()
        self.max_workspace=max(self.max_workspace,result.nbytes)
        return result
    def stats(self):
        return dict(schur_column_queries=self.count,unique_schur_columns=len(self.unique),
            newly_evaluated_kernel_entries_equivalent=self.count*len(self.S),
            full_original_kernel_column_equivalents=self.count*len(self.S)/self.n,
            maximum_column_workspace_bytes=self.max_workspace,
            dense_schur_matvecs_for_selection=0)


def best_row(i,G,diagonal,points,queries,generator):
    first,last=G.indptr[i:i+2];oldJ,oldg=G.indices[first:last],G.data[first:last]
    columns=queries.query(oldJ)
    oldblock=columns[oldJ,:]
    outside=np.setdiff1d(np.arange(i),oldJ,assume_unique=True)
    if generator=='global':
        LN=linalg.cholesky(oldblock[:-1,:-1],lower=True,check_finite=False)
        white=linalg.solve_triangular(LN,columns[:,:-1].T,lower=True,check_finite=False)
        conditional=diagonal-np.sum(white*white,axis=0)
        gradient=columns@oldg
        assert np.min(conditional[outside])>0
        ratios=gradient[outside]**2/conditional[outside]
        # ratio=(conditional covariance)^2/(sigma_i*conditional variance_j).
        assert np.max(ratios)<1+1e-6
        new_indices=outside[np.argsort(-ratios,kind='stable')[:CANDIDATES]]
    else:
        distances=np.sum((points[outside]-points[i])**2,axis=1)
        new_indices=outside[np.argsort(distances,kind='stable')[:CANDIDATES]]
    best=None
    for j in new_indices:
        for drop_position in range(len(oldJ)-1):
            mask=np.arange(len(oldJ))!=drop_position
            unsorted=np.r_[oldJ[mask],j];order=np.argsort(unsorted,kind='stable')
            block=np.empty((len(oldJ),len(oldJ)))
            block[:-1,:-1]=oldblock[np.ix_(mask,mask)]
            block[:-1,-1]=columns[j,mask];block[-1,:-1]=columns[j,mask]
            block[-1,-1]=diagonal[j]
            block=block[np.ix_(order,order)];J=unsorted[order].astype(np.int32)
            g=canonical(block,np.arange(len(J)))
            gain=float(2*np.log(g[-1]/oldg[-1]))
            if best is None or gain>best['gain']:
                best=dict(row=i,J=J,g=g,gain=gain,added=int(j),removed=int(oldJ[drop_position]),
                          normalization_error=abs(float(g@block@g)-1))
    return best


def update(G,proposals):
    new=G.copy();changes=[]
    for p in proposals:
        if p['gain']<=1e-12:continue
        first,last=new.indptr[p['row']:p['row']+2]
        new.indices[first:last]=p['J'];new.data[first:last]=p['g']
        changes.append({key:p[key] for key in ['row','gain','added','removed','normalization_error']})
    assert new.nnz==G.nnz and np.array_equal(new.indptr,G.indptr)
    assert np.all(new.diagonal()>0)
    return new,dict(changes=changes,accepted_rows=len(changes),nnz=int(new.nnz),
        logdet_H_change=2*float(np.sum(np.log(new.diagonal()/G.diagonal()))),
        spd_certificate='unchanged lower-triangular pattern orientation and strictly positive diagonal')


def run():
    static=json.loads(STATIC.read_text())
    for lam in [.001,1e-5]:
        for name in ['synthetic-n640-smooth','concrete-observed','airfoil-observed']:
            data=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
            X,b=data['X_train'],data['y_train'];kernel=GaussianKernel(X,float(data['bandwidth']))
            K=kernel.dense();A=K+lam*np.eye(len(b))
            for rank in [16,32]:
                budget();prior=next(c for c in static['runs'] if c['dataset']==name and
                                    c['regularization']==lam and c['rank']==rank)
                case=next((c for c in record['cases'] if c['dataset']==name and
                           c['regularization']==lam and c['rank']==rank),None)
                if case is not None and case['status']=='complete':continue
                if case is None:
                    case=dict(dataset=name,regularization=lam,rank=rank,status='running',runs=[],
                        fixed_rp=prior['fixed_rp'],static_uninterrupted=next(v for v in prior['variants']
                            if v['row_budget_including_diagonal']==BUDGET))
                    record['cases'].append(case)
                base=initialize_rpcholesky(kernel,rank,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
                permutation,L,B,S=schur_blocks(A,base.indices)
                points=X[permutation[rank:]];G,info=build_fsai(S,points,BUDGET)
                model=SchurFSAI(permutation,L,B,G);x,r,prefix=checkpoint(A,b,model)
                diagonal=1+lam-np.sum(B*B,axis=1)
                np.testing.assert_allclose(diagonal,np.diag(S),rtol=1e-9,atol=1e-12)
                rr=r[permutation];q=rr[rank:]-B@linalg.solve_triangular(L,rr[:rank],lower=True,check_finite=False)
                exposure=(G@q)**2;quality=1/(G.diagonal()**2*diagonal)
                eligible=np.arange(BUDGET,len(S))
                exposed=eligible[np.argsort(-exposure[eligible],kind='stable')[:ROWS]]
                local=eligible[np.argsort(-quality[eligible],kind='stable')[:ROWS]]
                uniform=np.random.default_rng(np.random.SeedSequence([0,6220,93,rank])).choice(eligible,ROWS,replace=False)
                case.update(prefix=prefix,initial_factor=info,retained_state_bytes=int(model.storage_bytes),
                    priorities=dict(exposure_rows=exposed.tolist(),local_quality_rows=local.tolist(),uniform_rows=uniform.tolist()),
                    persistent_added_full_kernel_columns=0)
                variants=[('matched_restart',None,None),('nearest_exposure',exposed,'nearest'),
                          ('global_exposure',exposed,'global'),('global_uniform',uniform,'global'),
                          ('global_local_quality',local,'global')]
                if name=='concrete-observed' and rank==32 and lam==.001:
                    variants.append(('extra_query_global_row_search',eligible,'global'))
                for label,selected,generator in variants:
                    if any(run['method']==label for run in case['runs']):continue
                    budget();start=perf_counter();queries=Columns(S,len(b))
                    if selected is None:newG=G;selection=None
                    else:
                        proposals=[]
                        if label=='extra_query_global_row_search':
                            heap=[]
                            for i in selected:
                                if i%32==0:budget()
                                proposal=best_row(int(i),G,diagonal,points,queries,generator)
                                item=(proposal['gain'],int(i),proposal)
                                if len(heap)<ROWS:heapq.heappush(heap,item)
                                elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
                            proposals=[item[2] for item in sorted(heap,key=lambda t:(-t[0],t[1]))]
                        else:
                            proposals=[best_row(int(i),G,diagonal,points,queries,generator) for i in selected]
                        newG,selection=update(G,proposals)
                        if label!='extra_query_global_row_search':assert queries.count==ROWS*BUDGET
                    selection_seconds=perf_counter()-start
                    newmodel=SchurFSAI(permutation,L,B,newG)
                    solved,_=pcg(A,b,newmodel,x0=x,maxiter=2500)
                    solved.update(method=label,total_iterations=PREFIX+solved['iterations'],selection=selection,
                        queries=queries.stats(),selection_dense_seconds=selection_seconds,
                        retained_state_bytes=int(newmodel.storage_bytes),
                        helper_final_check_products_not_in_legacy_counter=1,
                        nearest_query_note=('Offline interface reads full existing-support columns; a production '
                            'nearest-only control could use fewer local entry queries' if label=='nearest_exposure' else None))
                    case['runs'].append(solved);save()
                    print(json.dumps(dict(dataset=name,lam=lam,k=rank,method=label,
                        total_iterations=solved['total_iterations'],converged=solved['converged'],
                        schur_column_queries=queries.count,
                        determinant_gain=None if selection is None else selection['logdet_H_change'])),flush=True)
                case['status']='complete';save()


if __name__=='__main__':
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_CPU,(88,89))
    try:
        with threadpool_limits(limits=1):run()
        record['status']='complete'
    except TimeoutError as error:record.update(status='cpu_budget_exhausted',error=str(error))
    except Exception as error:
        record.update(status='failed',error=repr(error));save();raise
    save()
