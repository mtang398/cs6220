#!/usr/bin/env python3
"""Dense actual inexact-PCG recurrence; no native-memory or solver-time claim.

Dense Gaussian/Taylor matrices, the FPS64 Nyström setup, and all 41 cached tail
vectors are diagnostic conveniences. Action-cost weights are copied only from
the existing replay measurements, with missing degrees left unpriced.
"""
from collections import Counter
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter,process_time

import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh,solve
from scipy.special import gammainc
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
REPLAY=ROOT/'results/taylor-krylov-cost-oracle.json'
DEGREE_CAP=40
TOL=1e-8
MAXITER=2000


def load_polynomial():
    library=ctypes.CDLL(str(ROOT/'results/taylor_kernel_tile.so'))
    function=library.taylor_tile
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    function.argtypes=[array,array,array,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]
    function.restype=None
    return library,function


class DenseTaylor:
    def __init__(self,X,bw,function):
        self.function=function;self.n=len(X)
        self.Z=np.ascontiguousarray(X/bw)
        self.G=self.Z@self.Z.T
        # Same diagonal-tail definition as the replay source.
        self.s=np.einsum('ij,ij->i',X,X)/(bw*bw)
        self.scales=np.exp(-np.einsum('ij,ij->i',self.Z,self.Z)/2)
        self.tails=gammainc(np.arange(DEGREE_CAP+1)[:,None]+1,self.s[None,:])
        self.roots=np.sqrt(self.tails)
        self.beta=np.max(self.roots*(self.roots.sum(axis=1)[:,None]-self.roots),axis=1)
        self.cache={}
    def matrix(self,q):
        q=int(q)
        if q not in self.cache:
            h=np.empty_like(self.G)
            self.function(self.G,h,self.scales,self.n,0,self.n,q)
            h.flat[::self.n+1]+=self.tails[q]
            self.cache[q]=h
        return self.cache[q]
    def bounds(self,p):
        absolute=np.abs(p)
        inner=self.roots@absolute
        w=self.roots*(inner[:,None]-self.roots*absolute[None,:])
        return np.linalg.norm(w,axis=1)


def fps_nystrom(K,rank=64):
    """Verbatim algebra/order of the already executed replay preconditioner."""
    n=len(K);selected=[0];distance=2-2*K[:,0]
    for _ in range(rank-1):
        j=int(np.argmax(distance));selected.append(j)
        distance=np.minimum(distance,2-2*K[:,j])
    C=K[:,selected];W=K[np.ix_(selected,selected)]
    W.flat[::rank+1]+=1e-12
    N=C@cho_solve(cho_factor(W,lower=True),C.T)
    values,Q=eigh(N,subset_by_index=[n-rank,n-1]);values=np.maximum(values,0)
    return values,Q,np.asarray(selected,dtype=np.int64)


def pcg(A,b,lam,H,taylor,method,*,degree=None,tol=TOL,maxiter=MAXITER,deadline=float('inf')):
    started,cpu_started=perf_counter(),process_time()
    norm_b=float(np.linalg.norm(b));tolerance=tol*norm_b
    per_step=tolerance/(4*maxiter)
    x=np.zeros_like(b);r=b.copy();z=H(r);p=z.copy();rho=float(r@z)
    total_bound=0.;degrees=[];bound_steps=[];alpha_ratios=[];recursive_history=[]
    status='maxiter';iterations=0;true_checks=0;last_true=None;true=None
    for iteration in range(1,maxiter+1):
        if process_time()>deadline:status='diagnostic_cpu_cap';break
        if not np.isfinite(rho) or rho<=0:status='nonpositive_preconditioned_residual';break
        p2=float(p@p)
        if p2<=0:status='zero_direction';break
        alpha_upper=rho/(lam*p2)
        if method=='exact_A':
            q=None;bound=0.;ap=A@p
        else:
            all_bounds=taylor.bounds(p)
            if method=='adaptive':
                eligible=np.flatnonzero(alpha_upper*all_bounds<=per_step)
                if not len(eligible):status='degree_cap';break
                q=int(eligible[0])
            else:q=int(degree)
            bound=float(all_bounds[q]);ap=taylor.matrix(q)@p+lam*p
        curvature=float(p@ap)
        if not np.isfinite(curvature) or curvature<=0:status='nonpositive_curvature';break
        alpha=rho/curvature
        alpha_ratios.append(float(alpha/alpha_upper))
        degrees.append(q)
        contribution=abs(alpha)*bound;total_bound+=contribution;bound_steps.append(contribution)
        x+=alpha*p;r-=alpha*ap;iterations=iteration
        recursive=float(np.linalg.norm(r))
        recursive_history.append(recursive/norm_b)
        suspected=(recursive+total_bound<=tolerance) if method=='adaptive' else (recursive<=.5*tolerance)
        if suspected:
            true=b-A@x;true_checks+=1;last_true=iteration
            true_relative=float(np.linalg.norm(true)/norm_b)
            status='converged' if true_relative<=tol else 'recursive_stop_true_residual_failed'
            break
        z=H(r);newrho=float(r@z)
        p=z+(newrho/rho)*p;rho=newrho
    if last_true!=iterations:
        true=b-A@x;true_checks+=1
    true_relative=float(np.linalg.norm(true)/norm_b)
    gap=float(np.linalg.norm(true-r))
    recursive=float(np.linalg.norm(r))
    return dict(method=method,status=status,converged=true_relative<=tol,
        iterations=iterations,degree=degree,chosen_degrees=degrees,
        degree_histogram=dict(sorted(Counter(q for q in degrees if q is not None).items())),
        bound_sum=total_bound,bound_sum_over_rhs_norm=total_bound/norm_b,
        per_step_gap_budget=per_step,maximum_total_allocated_gap=tolerance/4,
        step_bound_contributions=bound_steps,maximum_alpha_over_upper=max(alpha_ratios,default=0.),
        step_alpha_over_upper=alpha_ratios,recursive_relative_residual=recursive/norm_b,
        recursive_plus_bound_relative=(recursive+total_bound)/norm_b,
        true_relative_residual=true_relative,final_residual_gap_norm=gap,
        final_residual_gap_over_rhs_norm=gap/norm_b,
        analytic_bound_covers_observed_final_gap=gap<=total_bound,
        original_reference_checks=true_checks,recursive_history=recursive_history,
        diagnostic_wall_seconds=perf_counter()-started,diagnostic_cpu_seconds=process_time()-cpu_started,
        semantics='Analytic truncation accumulation excludes floating-point action/recurrence errors; final original-A verification is decisive.')


def weighted_cost(run,replay):
    costs=replay['median_action_seconds']
    gaussian=min(costs['cdist64'],costs['gram64'])
    if run['method']=='exact_A':
        known=run['iterations']*gaussian;missing={};coverage=run['iterations']
    else:
        known=0.;missing={};coverage=0
        for q,count in run['degree_histogram'].items():
            name=f'taylor{q}'
            if name in costs:known+=count*costs[name];coverage+=count
            else:missing[str(q)]=count
    return dict(pricing_source=str(REPLAY.relative_to(ROOT)),
        measured_degree_action_count=coverage,total_actions=run['iterations'],
        missing_degree_action_counts=missing,known_action_seconds=known,
        complete_weighted_action_seconds=known if not missing else None,
        reference_verification_weighted_seconds=run['original_reference_checks']*gaussian,
        complete_weighted_action_plus_verification_seconds=(known+run['original_reference_checks']*gaussian) if not missing else None,
        exclusions='Dense cache construction, bound search, preconditioner/setup/application and vector work are unpriced. No extrapolation or new timings.')


def tiny_check(function):
    rng=np.random.default_rng(622102);X=rng.normal(size=(17,3));bw=2.;lam=.2
    K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw));A=K+lam*np.eye(len(X))
    taylor=DenseTaylor(X,bw,function);p=rng.normal(size=len(X))
    errors=[];bound_slack=[]
    for q in [0,2,5]:
        poly=sum(taylor.G**j/float(__import__('math').factorial(j)) for j in range(q+1))
        expected=taylor.scales[:,None]*poly*taylor.scales[None,:]
        expected.flat[::len(X)+1]+=taylor.tails[q]
        actual=taylor.matrix(q)
        errors.append(float(np.linalg.norm(actual-expected)))
        error=float(np.linalg.norm((K-actual)@p));bound=float(taylor.bounds(p)[q])
        assert error<=bound+1e-12
        bound_slack.append(bound-error)
        assert np.linalg.eigvalsh(actual+lam*np.eye(len(X)))[0]>0
    b=rng.normal(size=len(X))
    result=pcg(A,b,lam,lambda v:v.copy(),taylor,'adaptive',maxiter=100)
    assert max(errors)<1e-11 and result['converged'] and result['maximum_alpha_over_upper']<=1+1e-12
    return dict(polynomial_absolute_errors=errors,action_bound_slack=bound_slack,
        adaptive_iterations=result['iterations'],true_relative_residual=result['true_relative_residual'],
        maximum_alpha_over_upper=result['maximum_alpha_over_upper'])


def main():
    output=ROOT/'results/taylor-inexact-cg-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(100,105))
    started,cpu_started=perf_counter(),process_time();deadline=cpu_started+20.
    replay_record=json.loads(REPLAY.read_text())
    replay={(row['dataset'],row['regularization']):row for row in replay_record['runs']}
    library,function=load_polynomial()
    record=dict(status='running',protocol=dict(rank=64,regularizations=[.001,1e-5],rtol=TOL,
        maxiter=MAXITER,degree_cap=DEGREE_CAP,soft_diagnostic_cpu_cap_seconds=20.,hard_cpu_cap_seconds=100.,
        adaptive='Least q with rho/(lambda||p||²)*B_q(p)<=tol||b||/(4maxiter); ordinary PCG recurrence, actual |alpha|B accumulated.',
        corrected_bound='B_q(p)=||u_q*(u_q^T|p|−u_q*|p|)||_2; beta_q=max_i u_i*(sum_j u_j−u_i).',
        controls=['exact_A','static_replay','static_uniform_beta','adaptive'],
        fixed_stop='Recursive norm<=.5tol||b||, then original-A verify. No restart/replacement after failed verification.',
        adaptive_stop='Recursive norm+actual accumulated bound<=tol||b||, then original-A verify.',
        reference_checks='Stopping candidate and final on any cap/failure only; no exact residual feedback during adaptive iterations.',
        free_diagnostic_resources='Dense Gaussian, dense Taylor matrices and all-degree roots cache; FPS64 Nyström setup copied from replay. Not a1MiB or native solver measurement.',
        weights='Only matched case/shift replay medians for available degrees. Missing prices remain null.',
        lock='/tmp/cs6220-project-benchmark.lock'),
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),REPLAY,ROOT/'scripts/check_taylor_krylov_cost_oracle.py',ROOT/'scripts/taylor_kernel_tile.c',ROOT/'results/taylor_kernel_tile.so']},
        runs=[],setups=[],unrun=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    cases=[('smooth512','synthetic-n640-smooth'),('mixed512','synthetic-n640-mixed'),('Concrete824','concrete-observed')]
    with threadpool_limits(limits=1):
        record['tiny_check']=tiny_check(function);save()
        for dataset,stem in cases:
            if process_time()>deadline:
                record['unrun'].append(dict(dataset=dataset,reason='soft_diagnostic_cpu_cap'));continue
            tick=perf_counter()
            path=ROOT/'data/prepared'/f'{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
            values,Q,selected=fps_nystrom(K)
            taylor=DenseTaylor(X,bw,function)
            record['setups'].append(dict(dataset=dataset,n=len(b),diagnostic_seconds=perf_counter()-tick,
                indices=selected.tolist(),indices_sha256=sha256(selected.tobytes()).hexdigest()))
            for lam in [.001,1e-5]:
                A=K+lam*np.eye(len(K));coefficients=values/(values+lam)
                def H(v):return (v-Q@(coefficients*(Q.T@v)))/lam
                matched=replay[(dataset,lam)]
                eligible=np.flatnonzero(taylor.beta<=TOL*lam/4)
                uniform=int(eligible[0]) if len(eligible) else None
                for method,degree in [('exact_A',None),('static_replay',matched['static_degree']),('static_uniform_beta',uniform),('adaptive',None)]:
                    if process_time()>deadline:
                        record['unrun'].append(dict(dataset=dataset,regularization=lam,method=method,reason='soft_diagnostic_cpu_cap'));save();continue
                    if method=='static_uniform_beta' and degree is None:
                        record['runs'].append(dict(dataset=dataset,regularization=lam,method=method,status='degree_cap'));save();continue
                    run=pcg(A,b,lam,H,taylor,method,degree=degree,deadline=deadline)
                    run.update(dataset=dataset,n=len(b),regularization=lam,
                        data_sha256=sha256(path.read_bytes()).hexdigest(),
                        static_replay_degree=matched['static_degree'],uniform_beta_degree=uniform,
                        uniform_beta_threshold=TOL*lam/4,
                        uniform_beta_value=float(taylor.beta[uniform]) if uniform is not None else None)
                    run['weighted_cost']=weighted_cost(run,matched)
                    record['runs'].append(run);save()
                    print(json.dumps({k:run[k] for k in ['dataset','regularization','method','status','iterations','degree','true_relative_residual','bound_sum_over_rhs_norm']}),flush=True)
            del taylor,K,A,Q,values
    record['status']='complete' if not record['unrun'] else 'soft_diagnostic_cpu_cap';save()


if __name__=='__main__':main()
