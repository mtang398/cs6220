"""Packed-upper shared Schur-CG implementation and bounded validation only.

No native full-race timing is enabled by this script. The original dense-core
pilot and dependencies remain unchanged. Packed prefixes use dpptrs directly;
all stopping and original-kernel residual logic comes from the frozen wrapper.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import resource
from time import perf_counter,process_time

import numpy as np
from scipy import linalg
from scipy.linalg import blas,lapack
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import check_shared_schur_race as race
import check_packed_schur_feasibility as packed

ROOT=Path(__file__).resolve().parents[1]
DENSE_CERTIFIED=race.CertifiedShared
ORIGINAL_SET_BUDGET=race.set_budget


def set_budget(cap):
    ORIGINAL_SET_BUDGET(cap)
    packed.base.LIMIT=int(cap)
    packed.base.RESERVE=race.RESERVE


class PackedShared(DENSE_CERTIFIED):
    def __init__(self,kernel,maximum,regularization,indices=None):
        if indices is None:indices=kernel.farthest(maximum)
        saved_live=kernel.budget.live.copy()
        kernel.budget.live=[*saved_live,indices]
        self.base=packed.PackedSchur(kernel,maximum,regularization,
                                    indices=indices,lower=False)
        kernel.budget.live=saved_live
        self.kernel,self.budget=kernel,kernel.budget
        self.n,self.M,self.lam=kernel.n,maximum,regularization
        self.L,self.S,self.R=self.base.L,self.base.S,self.base.R
        self.permutation=self.base.permutation
        self.aliases=[]
        assert self.L.flags.c_contiguous and self.L.dtype==np.float64

    def core_product(self,vector):
        """A_SS=U.T U; packed-upper multiplies U then U.T in place."""
        before=vector.__array_interface__['data'][0]
        first=blas.dtpmv(self.M,self.L,vector,lower=0,trans=0,overwrite_x=1)
        second=blas.dtpmv(self.M,self.L,first,lower=0,trans=1,overwrite_x=1)
        alias=bool(first.__array_interface__['data'][0]==before and
                   second.__array_interface__['data'][0]==before and
                   np.shares_memory(first,vector) and np.shares_memory(second,vector))
        self.aliases.append(alias);assert alias
        self.budget.observe('packed_full_core_product',vector,first,second,
                            temporary_allowance=4096)
        return second

    def leading_solve(self,vector,m):
        if m==0:
            vector.fill(0.)
            return vector
        prefix=self.L[:m*(m+1)//2]
        rhs=vector[:m].reshape(m,1)
        assert prefix.flags.c_contiguous and np.shares_memory(prefix,self.L)
        assert rhs.flags.f_contiguous and np.shares_memory(rhs,vector)
        before=vector.__array_interface__['data'][0]
        solution,info=lapack.dpptrs(m,prefix,rhs,lower=0,overwrite_b=1)
        if info:raise np.linalg.LinAlgError(f'packed leading dpptrs info={info}')
        alias=bool(np.shares_memory(solution,vector) and
                   solution.__array_interface__['data'][0]==before and
                   prefix.__array_interface__['data'][0]==self.L.__array_interface__['data'][0])
        self.aliases.append(alias);assert alias
        vector[m:]=0.
        self.budget.observe('packed_prefix_solve',vector,prefix,rhs,solution,
                            temporary_allowance=4096)
        return vector


def install():
    # Process-local dependency injection; no measured source is rewritten.
    race.CertifiedShared=PackedShared
    race.set_budget=set_budget


def run_packed(X,b,bandwidth,regularization,mode='race',*,maximum,
               budget_bytes=262144,**kwargs):
    if mode not in ['race','shared_plain','shared_schur']:
        raise ValueError('Only shared-representation packed modes belong to this wrapper')
    install()
    return race.run(X,b,bandwidth,regularization,mode,maximum=maximum,
                    cap=budget_bytes,**kwargs)


def operator_validation(cap):
    set_budget(cap);rng=np.random.default_rng(62201126);records=[]
    for lam in [.001,1e-5]:
        X=np.ascontiguousarray(rng.normal(size=(47,5)));b=rng.normal(size=47)
        bw=1.6;M=17;ms=[0,9,M]
        A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))+lam*np.eye(len(b))
        budget=race.frontier.Budget([X,b]);kernel=race.frontier.SharedKernel(X,bw,budget)
        op=PackedShared(kernel,M,lam)
        ordered=A[np.ix_(op.permutation,op.permutation)]
        W=ordered[:M,:M];leading=[]
        for m in [1,9,M]:
            original=rng.normal(size=m);padded=np.zeros(M);padded[:m]=original
            actual=op.leading_solve(padded,m)
            expected=linalg.solve(W[:m,:m],original,assume_a='pos')
            error=float(np.linalg.norm(actual[:m]-expected)/np.linalg.norm(expected))
            assert np.count_nonzero(actual[m:])==0
            leading.append(error)
        v=rng.normal(size=M);expected=W@v;actual=op.core_product(v.copy())
        core_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        rhs=op.rhs(ms,b);directions=[rng.normal(size=len(b)-m) for m in ms]
        before=kernel.stats['entry_evaluations'];actions=op.apply(ms,directions)
        entries=kernel.stats['entry_evaluations']-before
        action_errors=[];rhs_errors=[]
        for j,m in enumerate(ms):
            if m:
                D=ordered[m:,m:]-ordered[m:,:m]@linalg.solve(ordered[:m,:m],ordered[:m,m:],assume_a='pos')
                correct_rhs=b[op.permutation[m:]]-ordered[m:,:m]@linalg.solve(
                    ordered[:m,:m],b[op.permutation[:m]],assume_a='pos')
            else:D=ordered;correct_rhs=b[op.permutation]
            expected=D@directions[j]
            action_errors.append(float(np.linalg.norm(actions[j]-expected)/np.linalg.norm(expected)))
            rhs_errors.append(float(np.linalg.norm(rhs[j]-correct_rhs)/np.linalg.norm(correct_rhs)))
        xs=[rng.normal(size=len(b)-m) for m in ms]
        full,residual=op.verify(ms,xs,b)
        original_errors=[];constraint_errors=[];reconstruction_errors=[]
        for j,m in enumerate(ms):
            expected_full=np.empty(len(b));expected_full[op.permutation[m:]]=xs[j]
            if m:
                expected_full[op.permutation[:m]]=linalg.solve(
                    ordered[:m,:m],b[op.permutation[:m]]-ordered[:m,m:]@xs[j],assume_a='pos')
            expected_residual=b-A@full[:,j]
            original_errors.append(float(np.linalg.norm(residual[:,j]-expected_residual[op.permutation])/np.linalg.norm(b)))
            constraint_errors.append(float(np.linalg.norm(expected_residual[op.permutation[:m]])/np.linalg.norm(b)))
            reconstruction_errors.append(float(np.linalg.norm(full[:,j]-expected_full)/max(1.,np.linalg.norm(expected_full))))
        values=[*leading,core_error,*action_errors,*rhs_errors,*original_errors,*constraint_errors,*reconstruction_errors]
        assert max(values)<1e-10 and all(op.aliases) and op.base.factor_alias
        assert entries==len(b)**2-M**2
        records.append(dict(regularization=lam,prefix_solve_relative_errors=leading,
            core_product_relative_error=core_error,Schur_action_relative_errors=action_errors,
            reduced_rhs_relative_errors=rhs_errors,literal_residual_relative_errors=original_errors,
            selected_constraint_relative_residuals=constraint_errors,
            reconstruction_relative_errors=reconstruction_errors,all_buffer_aliases=all(op.aliases),
            factor_alias=op.base.factor_alias,queried_entries_per_joint_action=entries,
            formula_entries=len(b)**2-M**2))
    return records


def sources():
    files=[Path(__file__).resolve(),Path(race.__file__).resolve(),Path(race.shared.__file__).resolve(),
           Path(race.frontier.__file__).resolve(),Path(packed.__file__).resolve()]
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget-bytes',type=int,default=262144)
    args=parser.parse_args();cap=args.budget_bytes
    resource.setrlimit(resource.RLIMIT_CPU,(10,12))
    output=ROOT/'results/packed-shared-schur-race-validation.json'
    if output.exists():raise FileExistsError(output)
    install();race.CAP=cap;set_budget(cap)
    wall,cpu=perf_counter(),process_time()
    record=dict(status='running',source_sha256=sources(),protocol=dict(
        purpose='Packed implementation correctness and memory feasibility only; no native full-race timing',
        budget_bytes=cap,reserve_bytes=65536,allocation_core_sizes=[128,160],
        allocation_shapes=['synthetic-n640-smooth','concrete-observed'],
        candidates='[0,M]',packing='upper; m-prefix is first m*(m+1)//2 entries',
        prefix_solve='dpptrs directly on contiguous prefix and one-column RHS view',
        core_product='dtpmv trans0 followed by trans1, full maximum packed factor',
        numerical_regularizations=[.001,1e-5],allocation_regularization=1e-5,
        blas_threads=1,cpu_cap_seconds=10,lock='/tmp/cs6220-project-benchmark.lock'),
        allocation_checks=[])
    def save():
        record.update(elapsed_cpu_seconds=process_time()-cpu,elapsed_wall_seconds=perf_counter()-wall,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_operator_checks']=operator_validation(cap)
        record['independent_full_race_check']=race.validation();save()
        for stem in record['protocol']['allocation_shapes']:
            with np.load(ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz') as data:
                X,b=data['X_train'],data['y_train'];bw=float(data['bandwidth'])
            for maximum in record['protocol']['allocation_core_sizes']:
                try:
                    row=race.trace_path(X,b,bw,1e-5,'race',maximum=maximum)
                    row.update(status='feasible' if row['trace_within_budget'] else 'trace_exceeds_cap',
                        dataset=stem,n=len(b),maximum=maximum,packed_core_bytes=4*maximum*(maximum+1),
                        dense_core_bytes=8*maximum*maximum)
                except (MemoryError,np.linalg.LinAlgError) as error:
                    row=dict(status='failure',dataset=stem,n=len(b),maximum=maximum,error=str(error))
                record['allocation_checks'].append(row);save()
                print(json.dumps({k:row[k] for k in ['dataset','maximum','status','tracemalloc_peak_bytes'] if k in row}),flush=True)
    record['status']='validated' if all(x['status']=='feasible' for x in record['allocation_checks']) else 'completed_with_infeasible_cases'
    save();print(json.dumps(dict(output=str(output),status=record['status'],elapsed_cpu_seconds=record['elapsed_cpu_seconds'])))


if __name__=='__main__':main()
