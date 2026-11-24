"""Fixed-global-nnz allocation across all canonical FSAI rows, offline only."""
from hashlib import sha256
import heapq
import json
from pathlib import Path
import resource
from time import perf_counter,process_time

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy import linalg
from threadpoolctl import threadpool_limits

from check_schur_fsai import (GaussianKernel,initialize_rpcholesky,schur_blocks,
                             build_fsai,SchurFSAI,pcg)
from check_fsai_pattern_exchange import checkpoint,canonical

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/fsai-global-allocation-diagnostic.json'
STATIC=ROOT/'results/schur-fsai-diagnostic.json'
START_CPU,START_WALL=process_time(),perf_counter()
LEVELS=(2,4,8,16);PREFIX=8
record=dict(status='running',purpose='Offline global sparse-capacity allocation; no production timing claim',
    protocol=dict(initial_uniform_row_budget=8,checkpoint=PREFIX,nested_sizes=list(LEVELS),
        total_nnz='Exactly sum_i min(8,i+1), identical to uniform8',
        allocator='Deterministic precedence greedy by next-level logdet gain per extra entry; then true single-entry increments',
        static_weights='one',residual_weights='fixed old abs(Gq)**2 at common true-residual checkpoint8',
        coefficient_rule='canonical FSAI from actual nested Schur principal blocks',
        single_entry_completion=True,exact_csr_bytes=True,blas_threads=1,cpu_limit_seconds=89,
        comparison_limit='No equal-memory compact/power Nystrom comparison or production runtime claim'),
    source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).resolve(),ROOT/'scripts/check_schur_fsai.py',
         ROOT/'scripts/check_fsai_pattern_exchange.py',ROOT/'scripts/check_projected_balancing.py']},
    imported_static_results=dict(path=str(STATIC.relative_to(ROOT)),sha256=sha256(STATIC.read_bytes()).hexdigest()),cases=[])


def save():
    record.update(cpu_seconds=process_time()-START_CPU,wall_seconds=perf_counter()-START_WALL,
                  offline_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def budget():
    if process_time()-START_CPU>84:raise TimeoutError('84 CPU-second graceful limit reached')


class Profiles:
    def __init__(self,S,points):
        distances=cdist(points,points,'sqeuclidean')
        self.rows=[];self.fit_calls=0;self.model_requests=0;self.query_entries=0
        for i in range(len(S)):
            if i%64==0:budget()
            neighbors=np.argsort(distances[i,:i],kind='stable')[:15]
            fullJ=np.sort(np.r_[neighbors,i]).astype(np.int32)
            block=S[np.ix_(fullJ,fullJ)].copy()
            self.query_entries+=len(fullJ)**2
            self.rows.append(dict(index=i,neighbors=neighbors,fullJ=fullJ,block=block,
                maximum=len(fullJ),levels=sorted(set(min(level,i+1) for level in LEVELS)),cache={}))
        for i,row in enumerate(self.rows):
            for size in row['levels']:self.get(i,size)

    def get(self,i,size):
        self.model_requests+=1;row=self.rows[i]
        if size not in row['cache']:
            J=np.sort(np.r_[row['neighbors'][:size-1],i]).astype(np.int32)
            positions=np.searchsorted(row['fullJ'],J)
            block=row['block'][np.ix_(positions,positions)]
            g=canonical(block,np.arange(size));self.fit_calls+=1
            row['cache'][size]=dict(J=J,g=g,logdiag=float(np.log(g[-1])),
                normalization_error=abs(float(g@block@g)-1))
        return row['cache'][size]

    def candidate_bytes(self):
        return int(sum(row['neighbors'].nbytes+row['fullJ'].nbytes+row['block'].nbytes+
            sum(v['J'].nbytes+v['g'].nbytes for v in row['cache'].values()) for row in self.rows))


def allocate(profiles,total,weights):
    initial_fits,initial_requests=profiles.fit_calls,profiles.model_requests
    sizes=np.array([row['levels'][0] for row in profiles.rows],dtype=int)
    remain=int(total-sizes.sum());heap=[];major=0;completion=0
    def offer(i):
        row=profiles.rows[i];larger=[v for v in row['levels'] if v>sizes[i]]
        if not larger:return
        nxt=larger[0];old=profiles.get(i,int(sizes[i]));new=profiles.get(i,nxt)
        gain=2*(new['logdiag']-old['logdiag'])
        if gain < -1e-8:raise ArithmeticError('Nested canonical determinant decreased')
        cost=nxt-int(sizes[i]);priority=float(weights[i]*max(gain,0)/cost)
        heapq.heappush(heap,(-priority,i,nxt,cost))
    for i in range(len(sizes)):offer(i)
    while heap and remain:
        _,i,nxt,cost=heapq.heappop(heap)
        if cost>remain:continue
        sizes[i]=nxt;remain-=cost;major+=1;offer(i)
    # Complete the exact global cap with canonical one-neighbor increments.
    while remain:
        budget();best=None
        for i,row in enumerate(profiles.rows):
            if sizes[i]>=row['maximum']:continue
            old=profiles.get(i,int(sizes[i]));new=profiles.get(i,int(sizes[i]+1))
            gain=2*(new['logdiag']-old['logdiag'])
            priority=float(weights[i]*max(gain,0))
            candidate=(priority,-i,i)
            if best is None or candidate>best:best=candidate
        if best is None:raise ArithmeticError('Cannot fill exact nonzero cap')
        i=best[2];sizes[i]+=1;remain-=1;completion+=1
    assert int(sizes.sum())==total
    data=[];indices=[];indptr=[0];errors=[];logdiag=[]
    for i,size in enumerate(sizes):
        value=profiles.get(i,int(size));data.extend(value['g']);indices.extend(value['J'])
        indptr.append(len(data));errors.append(value['normalization_error']);logdiag.append(value['logdiag'])
    G=sparse.csr_matrix((np.asarray(data),np.asarray(indices,dtype=np.int32),
        np.asarray(indptr,dtype=np.int32)),shape=(len(sizes),len(sizes)))
    assert G.nnz==total and np.all(G.diagonal()>0)
    assert np.all(G.indices<=np.repeat(np.arange(len(sizes)),np.diff(G.indptr)))
    unique,counts=np.unique(sizes,return_counts=True)
    return G,dict(row_sizes=sizes.tolist(),row_size_histogram={str(int(k)):int(v) for k,v in zip(unique,counts)},
        nnz=int(G.nnz),unused_capacity=0,coarse_level_upgrades=major,single_entry_final_upgrades=completion,
        canonical_model_requests=profiles.model_requests-initial_requests,
        newly_computed_canonical_models=profiles.fit_calls-initial_fits,
        maximum_row_energy_normalization_error=float(max(errors)),
        positive_factor_diagonal=True,logdet_H=2*float(sum(logdiag)))


def run():
    static=json.loads(STATIC.read_text())
    for lam in [.001,1e-5]:
        for name in ['synthetic-n640-smooth','synthetic-n640-random','concrete-observed','airfoil-observed']:
            data=np.load(ROOT/f'data/prepared/{name}-s0-lambda0.001.npz')
            X,b=data['X_train'],data['y_train'];kernel=GaussianKernel(X,float(data['bandwidth']));K=kernel.dense()
            A=K+lam*np.eye(len(b));rank=32;budget()
            case=dict(dataset=name,regularization=lam,rank=rank,status='running',runs=[])
            record['cases'].append(case)
            base=initialize_rpcholesky(kernel,rank,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
            permutation,L,B,S=schur_blocks(A,base.indices);points=X[permutation[rank:]]
            G,initial=build_fsai(S,points,8);model=SchurFSAI(permutation,L,B,G)
            prior=next((c for c in static['runs'] if c['dataset']==name and c['regularization']==lam
                        and c['rank']==rank),None)
            if prior is None:
                uninterrupted,_=pcg(A,b,model,maxiter=2500)
                uninterrupted['source']='executed in this diagnostic'
            else:
                uninterrupted=dict(next(v for v in prior['variants'] if v['row_budget_including_diagonal']==8))
                uninterrupted['source']=str(STATIC.relative_to(ROOT))
                assert np.array_equal(base.indices,prior['indices'])
            case.update(uniform_uninterrupted=uninterrupted,initial_factor=initial,
                        uniform_retained_state_bytes=int(model.storage_bytes))
            x,r,prefix=checkpoint(A,b,model);case['prefix']=prefix
            rr=r[permutation];q=rr[rank:]-B@linalg.solve_triangular(L,rr[:rank],lower=True,check_finite=False)
            old_coordinates=G@q;weights=old_coordinates**2
            case['residual_score_provenance']=dict(rhs='true b-A*x at common uniform8 step8',
                q_sha256=sha256(q.tobytes()).hexdigest(),weights_sha256=sha256(weights.tobytes()).hexdigest(),
                schur_operator_products=0,extra_vector_bytes=int(q.nbytes+weights.nbytes))
            start=perf_counter();profiles=Profiles(S,points)
            max_profile_error=0.
            for i in range(len(S)):
                value=profiles.get(i,min(8,i+1));first,last=G.indptr[i:i+2]
                assert np.array_equal(value['J'],G.indices[first:last])
                max_profile_error=max(max_profile_error,float(np.linalg.norm(value['g']-G.data[first:last])/
                    np.linalg.norm(G.data[first:last])))
            assert max_profile_error<1e-9
            case['candidate_setup']=dict(shared_profile_seconds=perf_counter()-start,
                maximum_pattern_kernel_entry_queries=profiles.query_entries,
                initial_uniform8_kernel_entry_queries=int(sum(min(8,i+1)**2 for i in range(len(S)))),
                candidate_endpoint_canonical_refits=profiles.fit_calls,
                uniform8_profile_relative_error=max_profile_error,
                query_note='Candidate max16 principal blocks are charged in full; no free off-pattern access')
            staticG,static_info=allocate(profiles,G.nnz,np.ones(len(S)))
            weightedG,weighted_info=allocate(profiles,G.nnz,weights)
            case['candidate_setup'].update(total_distinct_canonical_refits=profiles.fit_calls,
                candidate_array_bytes_after_both_allocators=profiles.candidate_bytes())
            for label,newG,allocation in [('uniform_matched_restart',G,None),
                ('global_static_allocation',staticG,static_info),('global_residual_allocation',weightedG,weighted_info)]:
                budget();newmodel=SchurFSAI(permutation,L,B,newG)
                assert newmodel.storage_bytes==model.storage_bytes
                solved,_=pcg(A,b,newmodel,x0=x,maxiter=2500)
                solved.update(method=label,total_iterations=PREFIX+solved['iterations'],allocation=allocation,
                    retained_state_bytes=int(newmodel.storage_bytes),helper_final_check_products_not_in_legacy_counter=1)
                if allocation:
                    allocated=np.asarray(allocation['row_sizes'])
                    solved['rows_changed_from_uniform8']=int(np.count_nonzero(allocated!=np.diff(G.indptr)))
                case['runs'].append(solved);save()
                print(json.dumps(dict(dataset=name,lam=lam,method=label,total_iterations=solved['total_iterations'],
                    converged=solved['converged'],true_relative_residual=solved['true_relative_residual'],
                    nnz=int(newG.nnz),histogram=None if allocation is None else allocation['row_size_histogram'])),flush=True)
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
