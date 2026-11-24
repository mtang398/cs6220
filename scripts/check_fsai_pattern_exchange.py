"""Bounded offline fixed-nnz FSAI pattern exchanges on the actual Schur matrix.

Reuses the static AFN implementation. Current-residual selection uses exact
inverse-action error-energy differences; this is not a production timing test.
"""
from hashlib import sha256
import heapq
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from check_schur_fsai import (GaussianKernel, initialize_rpcholesky, schur_blocks,
                              build_fsai, SchurFSAI, pcg)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/fsai-pattern-exchange-diagnostic.json'
STATIC = ROOT/'results/schur-fsai-diagnostic.json'
PREFIX, ROW_BUDGET, UPDATE_ROWS, NEW_NEIGHBORS = 8, 8, 8, 4
START_CPU, START_WALL = process_time(), perf_counter()
record = dict(status='running', purpose='Offline dense mechanism test, no production timing claim',
    protocol=dict(checkpoint=PREFIX, row_budget=ROW_BUDGET, selected_rows=UPDATE_ROWS,
        nearest_outside_candidates=NEW_NEIGHBORS, current_schur_rhs_only=True,
        row_selection='Eight largest available determinant gains, independent of RHS',
        candidate_selection='same four nearest earlier outside neighbors and every drop-one old neighbor',
        same_candidate_pool=True, same_eight_update_opportunities=True,
        energy_rule_can_reject=True, sequential_cross_terms=True,
        initial_score_extra_dense_schur_products=1,blas_threads=1,cpu_budget_seconds=54),
    source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).resolve(),ROOT/'scripts/check_schur_fsai.py',ROOT/'scripts/check_projected_balancing.py']},
    imported_static_results=dict(path=str(STATIC.relative_to(ROOT)),sha256=sha256(STATIC.read_bytes()).hexdigest()),
    cases=[])


def save():
    record.update(cpu_seconds=process_time()-START_CPU,wall_seconds=perf_counter()-START_WALL,
                  offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def budget():
    if process_time()-START_CPU>54:
        raise TimeoutError('54 CPU-second budget reached')


class CaptureApplications:
    def __init__(self,state):
        self.state=state
        self.inner_products=[]
    def apply(self,r):
        z=self.state.apply(r)
        self.inner_products.append(float(r@z))
        return z


def checkpoint(A,b,state):
    """Replay captured helper steps using its exact r^T z scalars, no new solver."""
    wrapped=CaptureApplications(state)
    stats,saved=pcg(A,b,wrapped,maxiter=PREFIX,capture=True)
    assert len(saved['directions'])==PREFIX
    x=np.zeros_like(b)
    for rho,p,ap in zip(wrapped.inner_products,saved['directions'],saved['images']):
        x+=(rho/float(p@ap))*p
    r=b-A@x
    observed=float(np.linalg.norm(r)/np.linalg.norm(b))
    assert abs(observed-stats['true_relative_residual'])<1e-12*max(observed,1e-15)
    return x,r,dict(**stats,reconstruction_check_relative_residual=observed,
        helper_final_check_products_not_in_legacy_counter=1,
        reconstruction_true_residual_extra_products=1)


def canonical(S,J):
    rhs=np.zeros(len(J));rhs[-1]=1
    v=linalg.cho_solve(linalg.cho_factor(S[np.ix_(J,J)],lower=True,check_finite=False),
                      rhs,check_finite=False)
    if v[-1]<=0:
        raise ArithmeticError('Nonpositive canonical FSAI diagonal')
    return v/np.sqrt(v[-1])


def pool(S,points,G):
    """Build a RHS-independent candidate pool; retain only eight best rows."""
    distances=cdist(points,points,'sqeuclidean')
    heap=[];candidate_count=0;local_entries=0
    for i in range(ROW_BUDGET,len(S)):
        if i%32==0:budget()
        start,stop=G.indptr[i:i+2]
        oldJ,oldg=G.indices[start:stop],G.data[start:stop]
        outside=np.setdiff1d(np.arange(i),oldJ,assume_unique=True)
        nearest=outside[np.argsort(distances[i,outside],kind='stable')[:NEW_NEIGHBORS]]
        proposals=[]
        for new_index in nearest:
            for drop in oldJ[:-1]:
                J=np.sort(np.r_[oldJ[oldJ!=drop],new_index]).astype(np.int32)
                g=canonical(S,J)
                gain=2*np.log(g[-1]/oldg[-1])
                proposals.append(dict(J=J,g=g,determinant_gain=float(gain),
                                      added=int(new_index),removed=int(drop)))
                candidate_count+=1;local_entries+=len(J)**2
        if proposals:
            best=max(c['determinant_gain'] for c in proposals)
            item=(best,i,proposals)
            if len(heap)<UPDATE_ROWS:heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
    selected=sorted(heap,key=lambda item:(-item[0],item[1]))
    return selected,dict(all_canonical_candidate_rows_solved=candidate_count,
        local_schur_entries_requested_with_repeats=local_entries,
        selected_row_ids=[i for _,i,_ in selected],retained_candidates=sum(len(p) for _,_,p in selected))


def candidate_delta(G,row,proposal,q):
    first,last=G.indptr[row:row+2]
    oldJ,oldg=G.indices[first:last],G.data[first:last]
    newJ,newg=proposal['J'],proposal['g']
    J=np.union1d(oldJ,newJ)
    delta=np.zeros(len(J))
    delta[np.searchsorted(J,newJ)]+=newg*float(newg@q[newJ])
    delta[np.searchsorted(J,oldJ)]-=oldg*float(oldg@q[oldJ])
    return J,delta


def select(S,G,q,d,rows,rule):
    model=G.copy();total_delta=np.zeros(len(S));support=np.empty(0,dtype=int)
    changes=[];predicted=0.0;scale=max(float(q@(G.T@(G@q))),1e-300)
    for _,row,proposals in rows:
        best=None
        for proposal in proposals:
            J,delta=candidate_delta(G,row,proposal,q)
            local=float(-2*d[J]@delta+delta@S[np.ix_(J,J)]@delta)
            if support.size:
                local+=2*float(total_delta[support]@S[np.ix_(support,J)]@delta)
            score=local if rule=='inverse_action_energy' else -proposal['determinant_gain']
            if best is None or score<best[0]:best=(score,proposal,J,delta,local)
        score,proposal,J,delta,local=best
        accept=(score < -1e-10*scale if rule=='inverse_action_energy' else score < -1e-12)
        if not accept:continue
        first,last=model.indptr[row:row+2]
        model.indices[first:last]=proposal['J'];model.data[first:last]=proposal['g']
        total_delta[J]+=delta;support=np.union1d(support,J);predicted+=local
        changes.append(dict(row=row,added=proposal['added'],removed=proposal['removed'],
            determinant_gain=proposal['determinant_gain'],conditional_energy_change=local))
    assert model.nnz==G.nnz and np.all(np.diff(model.indptr)==np.diff(G.indptr))
    assert np.all(model.diagonal()>0)
    oldz=G.T@(G@q);newz=model.T@(model@q)
    actual_delta=newz-oldz
    delta_discrepancy=float(np.linalg.norm(actual_delta-total_delta)/max(np.linalg.norm(actual_delta),1e-300))
    # One extra full S product verifies the combined result; selection used local entries only.
    direct_change=float(-2*d@actual_delta+actual_delta@(S@actual_delta))
    error=abs(predicted-direct_change)/max(abs(predicted),abs(direct_change),scale*1e-12)
    assert error<1e-6,(rule,error,predicted,direct_change)
    if rule=='inverse_action_energy':assert direct_change<=1e-8*scale
    determinant_change=2*float(np.sum(np.log(model.diagonal()/G.diagonal())))
    normalization=max(abs(float(model.data[model.indptr[i]:model.indptr[i+1]]@
        S[np.ix_(model.indices[model.indptr[i]:model.indptr[i+1]],model.indices[model.indptr[i]:model.indptr[i+1]])]@
        model.data[model.indptr[i]:model.indptr[i+1]])-1) for i in range(len(S)))
    return model,dict(accepted_rows=len(changes),changes=changes,
        predicted_combined_energy_change=predicted,verified_combined_energy_change=direct_change,
        energy_change_normalized_by_old_qHq=direct_change/scale,
        combined_score_relative_verification_error=error,delta_action_relative_discrepancy=delta_discrepancy,
        logdet_H_change=determinant_change,unchanged_total_nnz=int(model.nnz),
        positive_factor_diagonal=bool(np.all(model.diagonal()>0)),
        maximum_row_S_energy_normalization_error=float(normalization),
        combined_verification_extra_dense_schur_products=1)


def run():
    static=json.loads(STATIC.read_text())
    for name in ['synthetic-n640-smooth','concrete-observed']:
        data=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
        X,b=data['X_train'],data['y_train'];kernel=GaussianKernel(X,float(data['bandwidth']));K=kernel.dense()
        for lam in [.001,1e-5]:
            A=K+lam*np.eye(len(b))
            for rank in [16,32]:
                budget()
                case=dict(dataset=name,regularization=lam,rank=rank,row_budget=ROW_BUDGET,status='running',runs=[])
                record['cases'].append(case)
                baseline=next(r for r in static['runs'] if r['dataset']==name and
                              r['regularization']==lam and r['rank']==rank)
                base=initialize_rpcholesky(kernel,rank,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
                permutation,L,B,S=schur_blocks(A,base.indices)
                G,info=build_fsai(S,X[permutation[rank:]],ROW_BUDGET)
                model=SchurFSAI(permutation,L,B,G)
                case.update(static_uninterrupted=next(v for v in baseline['variants'] if
                            v['row_budget_including_diagonal']==ROW_BUDGET),fixed_rp=baseline['fixed_rp'],
                            initial_sparse_factor=info,retained_state_bytes=int(model.storage_bytes))
                x,r,prefix=checkpoint(A,b,model);case['prefix']=prefix
                permuted=r[permutation]
                q=permuted[rank:]-B@linalg.solve_triangular(L,permuted[:rank],lower=True,check_finite=False)
                z=G.T@(G@q);d=q-S@z
                case['initial_score_extra_dense_schur_products']=1
                start=perf_counter();rows,details=pool(S,X[permutation[rank:]],G)
                case.update(candidate_pool=details,static_candidate_pool_dense_seconds=perf_counter()-start,
                            q_and_d_storage_bytes=int(q.nbytes+d.nbytes))
                for label,newG,selection in [('matched_restart',G,None)]+[
                    (rule,*select(S,G,q,d,rows,rule)) for rule in ['determinant','inverse_action_energy']]:
                    budget();newmodel=SchurFSAI(permutation,L,B,newG)
                    solved,_=pcg(A,b,newmodel,x0=x,maxiter=2500)
                    solved.update(method=label,total_iterations=PREFIX+solved['iterations'],
                                  retained_state_bytes=int(newmodel.storage_bytes),selection=selection,
                                  helper_final_check_products_not_in_legacy_counter=1)
                    case['runs'].append(solved);save()
                    print(json.dumps(dict(dataset=name,lam=lam,k=rank,**solved)),flush=True)
                case['status']='complete';save()


if __name__=='__main__':
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_CPU,(58,59))
    try:
        with threadpool_limits(limits=1):run()
        record['status']='complete'
    except TimeoutError as error:
        record.update(status='cpu_budget_exhausted',error=str(error))
    except Exception as error:
        record.update(status='failed',error=repr(error));save();raise
    save()
