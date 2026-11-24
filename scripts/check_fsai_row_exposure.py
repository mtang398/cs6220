"""Offline FSAI row allocation by current exposure times determinant potential.

Canonical row values and the best pattern within each row remain entirely
matrix-based. Only the choice of eight rows uses the current solve residual.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter,process_time

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from check_schur_fsai import (GaussianKernel,initialize_rpcholesky,schur_blocks,
                             build_fsai,SchurFSAI,pcg)
from check_fsai_pattern_exchange import canonical,checkpoint

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/fsai-row-exposure-diagnostic.json'
PREVIOUS=ROOT/'results/fsai-pattern-exchange-diagnostic.json'
START_CPU,START_WALL=process_time(),perf_counter()
ROW_BUDGET,UPDATE_ROWS,NEIGHBORS,PREFIX=8,8,4,8
record=dict(status='running',purpose='Offline row-allocation mechanism test, no production timing claim',
    protocol=dict(row_budget=ROW_BUDGET,updated_rows=UPDATE_ROWS,new_neighbors=NEIGHBORS,
        checkpoint=PREFIX,within_row_selection='maximum logdet gain, independent of RHS',
        static_row_score='available positive logdet gain',
        exposed_row_score='(Gq)_i**2 times available positive logdet gain',
        candidate_pool='same four nearest earlier outside neighbors and every drop-one old neighbor',
        candidate_cache='regenerated once per case and shared by both rules',
        extra_schur_matvecs_for_selection=0,blas_threads=1,cpu_budget_seconds=54),
    source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).resolve(),ROOT/'scripts/check_fsai_pattern_exchange.py',
         ROOT/'scripts/check_schur_fsai.py',ROOT/'scripts/check_projected_balancing.py']},
    previous_results=dict(path=str(PREVIOUS.relative_to(ROOT)),sha256=sha256(PREVIOUS.read_bytes()).hexdigest()),
    cases=[])


def save():
    record.update(cpu_seconds=process_time()-START_CPU,wall_seconds=perf_counter()-START_WALL,
                  offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def budget():
    if process_time()-START_CPU>54:raise TimeoutError('54 CPU-second budget reached')


def best_patterns(S,points,G):
    distances=cdist(points,points,'sqeuclidean')
    rows=[];count=0
    for i in range(ROW_BUDGET,len(S)):
        if i%32==0:budget()
        first,last=G.indptr[i:i+2];oldJ,oldg=G.indices[first:last],G.data[first:last]
        outside=np.setdiff1d(np.arange(i),oldJ,assume_unique=True)
        nearest=outside[np.argsort(distances[i,outside],kind='stable')[:NEIGHBORS]]
        best=None
        for new_index in nearest:
            for drop in oldJ[:-1]:
                J=np.sort(np.r_[oldJ[oldJ!=drop],new_index]).astype(np.int32)
                g=canonical(S,J);gain=float(2*np.log(g[-1]/oldg[-1]));count+=1
                if best is None or gain>best['gain']:
                    best=dict(row=i,J=J,g=g,gain=gain,added=int(new_index),removed=int(drop))
        if best and best['gain']>1e-12:rows.append(best)
    return rows,dict(canonical_candidates_solved=count,positive_potential_rows=len(rows))


def updated(G,rows,exposure,rule):
    priority=(lambda item:item['gain']) if rule=='static_determinant' else (
        lambda item:exposure[item['row']]*item['gain'])
    chosen=sorted(rows,key=lambda item:(-priority(item),item['row']))[:UPDATE_ROWS]
    result=G.copy();changes=[]
    for row in chosen:
        i=row['row'];first,last=result.indptr[i:i+2]
        result.indices[first:last]=row['J'];result.data[first:last]=row['g']
        changes.append(dict(row=i,added=row['added'],removed=row['removed'],
                            determinant_gain=row['gain'],exposure=float(exposure[i]),
                            allocation_score=float(priority(row))))
    assert result.nnz==G.nnz and np.array_equal(result.indptr,G.indptr)
    assert np.all(result.diagonal()>0)
    return result,dict(changes=changes,accepted_rows=len(chosen),nnz=int(result.nnz),
        positive_factor_diagonal=True,logdet_H_change=2*float(np.sum(np.log(result.diagonal()/G.diagonal()))))


def run():
    previous=json.loads(PREVIOUS.read_text())
    for name in ['synthetic-n640-smooth','concrete-observed']:
        data=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
        X,b=data['X_train'],data['y_train'];kernel=GaussianKernel(X,float(data['bandwidth']));K=kernel.dense()
        for lam in [.001,1e-5]:
            A=K+lam*np.eye(len(b))
            for rank in [16,32]:
                budget();prior=next(c for c in previous['cases'] if c['dataset']==name and
                    c['regularization']==lam and c['rank']==rank)
                case=dict(dataset=name,regularization=lam,rank=rank,status='running',runs=[],
                    fixed_rp=prior['fixed_rp'],static_uninterrupted=prior['static_uninterrupted'],
                    matched_restart=next(r for r in prior['runs'] if r['method']=='matched_restart'))
                record['cases'].append(case)
                base=initialize_rpcholesky(kernel,rank,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
                permutation,L,B,S=schur_blocks(A,base.indices)
                G,info=build_fsai(S,X[permutation[rank:]],ROW_BUDGET)
                model=SchurFSAI(permutation,L,B,G);x,r,prefix=checkpoint(A,b,model)
                case.update(prefix=prefix,initial_factor=info,retained_state_bytes=int(model.storage_bytes))
                rr=r[permutation]
                q=rr[rank:]-B@linalg.solve_triangular(L,rr[:rank],lower=True,check_finite=False)
                exposure=(G@q)**2
                start=perf_counter();rows,stats=best_patterns(S,X[permutation[rank:]],G)
                case.update(candidate_pool=stats,shared_candidate_pool_dense_seconds=perf_counter()-start,
                            q_and_exposure_storage_bytes=int(q.nbytes+exposure.nbytes))
                for rule in ['static_determinant','residual_exposure']:
                    budget();newG,selection=updated(G,rows,exposure,rule)
                    newmodel=SchurFSAI(permutation,L,B,newG)
                    solved,_=pcg(A,b,newmodel,x0=x,maxiter=2500)
                    solved.update(method=rule,total_iterations=PREFIX+solved['iterations'],selection=selection,
                        retained_state_bytes=int(newmodel.storage_bytes),helper_final_check_products_not_in_legacy_counter=1)
                    case['runs'].append(solved);save()
                    print(json.dumps(dict(dataset=name,lam=lam,k=rank,method=rule,
                        total_iterations=solved['total_iterations'],converged=solved['converged'],
                        true_residual=solved['true_relative_residual'],rows=[c['row'] for c in selection['changes']])),flush=True)
                expected=next(r for r in prior['runs'] if r['method']=='determinant')
                case['static_control_iteration_difference_from_previous']=case['runs'][0]['total_iterations']-expected['total_iterations']
                case['status']='complete';save()


if __name__=='__main__':
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_CPU,(58,59))
    try:
        with threadpool_limits(limits=1):run()
        record['status']='complete'
    except TimeoutError as error:record.update(status='cpu_budget_exhausted',error=str(error))
    except Exception as error:
        record.update(status='failed',error=repr(error));save();raise
    save()
