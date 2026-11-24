"""One predeclared fixed-memory two-trajectory Schur-CG pilot.

Both independent branches start at time zero. First certified convergence
wins; no residual-based forecasting or pruning. Literal original kernel
entries certify every returned residual. This is a pilot, not a novelty claim.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import resource
from time import perf_counter, process_time
import tracemalloc

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import check_shared_schur_action as shared
import check_schur_inplace_frontier_v2 as frontier

ROOT=Path(__file__).resolve().parents[1]
CAP=262144
RESERVE=65536
MAXIMUM=96


def set_budget(cap):
    # These are process-local settings; dependency source files stay unchanged.
    shared.LIMIT=frontier.LIMIT=int(cap)
    shared.RESERVE=frontier.RESERVE=RESERVE


class CertifiedShared(shared.SharedAction):
    def product(self,rows,columns,vectors,target,label,*,add=False):
        if columns is None:
            return super().product(rows,columns,vectors,target,label,add=add)
        # At256KiB the two branch states cannot coexist with a full X[R]
        # feature pack. Column blocking preserves the exact queried entries.
        if not add:target.fill(0.)
        width=vectors.shape[1];X=self.kernel.X
        for first in range(0,len(columns),128):
            last=min(len(columns),first+128)
            packed=X[columns[first:last]]
            available=shared.LIMIT-RESERVE-self.budget.size(packed,vectors,target)-4096
            per_row=8*(len(packed)+X.shape[1]+width)
            row_count=min(128,int(available//per_row))
            if row_count<1:raise MemoryError(f'No column-blocked row tile fits at {label}')
            self.budget.row_sizes.setdefault(label,set()).add(row_count)
            for begin in range(0,len(rows),row_count):
                end=min(len(rows),begin+row_count)
                points=X[rows[begin:end]]
                tile=self.kernel.block(points,packed)
                values=tile@vectors[first:last]
                self.budget.observe(label,packed,vectors,target,points,tile,values)
                target[begin:end]+=values
                del points,tile,values
            del packed

    def verify(self,ms,xs,b):
        """Return original-coordinate solutions and original residuals.

        Residual rows are in the maximum-core permutation. The final matmat
        queries K at every pair of original points, never substitutes L L.T.
        """
        n,M,q=self.n,self.M,len(ms)
        live=self.budget.live.copy()
        full=np.zeros((n,q),order='F')
        self.budget.live=[*live,full]
        for j,(m,x) in enumerate(zip(ms,xs)):
            full[self.permutation[m:],j]=x
        if any(ms):
            top=np.zeros((M,q),order='F')
            bottom=np.empty((n-M,q),order='F')
            self.budget.live=[*live,full,top,bottom]
            for j,(m,x) in enumerate(zip(ms,xs)):
                work=np.zeros(M);work[m:]=x[:M-m]
                if m:top[:,j]=self.core_product(work)
                del work
                bottom[:,j]=x[M-m:]
            self.product(self.S,self.R,bottom,top,'verification_reconstruction_cross',add=True)
            del bottom
            self.budget.live=[*live,full,top]
            for j,m in enumerate(ms):
                if m:
                    top[:m,j]*=-1.;top[:m,j]+=b[self.S[:m]]
                    self.leading_solve(top[:,j],m)
                    full[self.S[:m],j]=top[:m,j]
            del top
        residual=np.empty((n,q),order='F')
        self.budget.live=[*live,full,residual]
        self.product(self.permutation,None,full,residual,'literal_original_kernel_verification')
        for j in range(q):
            residual[:,j]*=-1.
            residual[:,j]+=b[self.permutation]
            residual[:,j]-=self.lam*full[self.permutation,j]
        self.budget.observe('original_residual_vector_update',full,residual,
                            temporary_allowance=16*n)
        self.kernel.stats['full_matvecs']+=q
        self.budget.live=live
        return full,residual


class Natural:
    def __init__(self,kernel,m,lam):
        self.kernel,self.budget,self.m,self.lam=kernel,kernel.budget,m,lam
        self.base=frontier.InplaceSchur(kernel,m,lam) if m else None
        self.permutation=self.base.permutation if m else None
    def rhs(self,ms,b):
        return [self.base.reduced_rhs(b) if self.base else b.copy()]
    def apply(self,ms,ps):
        if self.base:return [self.base.apply(ps[0])]
        ap=self.kernel.full(ps[0]);ap+=self.lam*ps[0]
        self.budget.observe('plain_operator_shift',ap,temporary_allowance=ps[0].nbytes)
        return [ap]
    def verify(self,ms,xs,b):
        full=self.base.reconstruct(b,xs[0]) if self.base else xs[0]
        residual=self.kernel.full(full,extra=(full,))
        residual*=-1.;residual+=b;residual-=self.lam*full
        self.budget.observe('natural_true_residual_update',full,residual,
                            temporary_allowance=full.nbytes)
        if self.base:
            ordered=residual[self.permutation]
            self.budget.observe('natural_order_true_residual',full,residual,ordered)
            residual=ordered
        return full[:,None],residual[:,None]


def aliases_now(op):
    values=[]
    if isinstance(op,CertifiedShared):
        values.extend(op.aliases);op.aliases.clear();base=op.base
    else:base=op.base
    if base:
        values.extend(base.solve_alias_checks);base.solve_alias_checks.clear()
        values.append(base.factor_alias)
    return all(values)


def run(X,b,bw,lam,mode,*,maximum=MAXIMUM,cap=CAP,maxiter=2500,
        rtol=1e-8,per_run_cpu=8.,deadline=float('inf'),callback=None):
    set_budget(cap)
    wall,cpu=perf_counter(),process_time()
    budget=frontier.Budget([X,b]);kernel=frontier.SharedKernel(X,bw,budget)
    modes={'race':[0,maximum],'shared_plain':[0],'shared_schur':[maximum],
           'natural_plain':[0],'natural_schur':[maximum]}
    ms=modes[mode]
    op=(Natural(kernel,ms[0],lam) if mode.startswith('natural') else
        CertifiedShared(kernel,maximum,lam))
    rs=op.rhs(ms,b);xs=[np.zeros_like(r) for r in rs];ps=[r.copy() for r in rs]
    norm_b=float(np.linalg.norm(b));q=len(ms)
    true_relative=[None]*q;last_checked=[-1]*q;replacements=[0]*q
    true_checks=[0]*q;kernel_passes=0;operator_batches=0;scalar_products=0
    aliases=aliases_now(op);winner=None;last_full=None;status='maxiter'
    last_full_indices=[];first_verified_converged=[]
    budget.live=[*xs,*rs,*ps]
    setup_seconds=perf_counter()-wall;setup_cpu_seconds=process_time()-cpu
    setup_stats=dict(kernel.stats)
    # Reserve200ms of the8s CPU screen for mandatory terminal certification.
    action_deadline=min(cpu+max(.01,per_run_cpu-.2),deadline-.2)
    def certify(indices,iteration):
        nonlocal aliases,kernel_passes,last_full,last_full_indices
        chosen_ms=[ms[j] for j in indices];chosen_xs=[xs[j] for j in indices]
        budget.live=[*xs,*rs,*ps]
        full,residual=op.verify(chosen_ms,chosen_xs,b)
        kernel_passes+=1
        for col,j in enumerate(indices):
            true_relative[j]=float(np.linalg.norm(residual[:,col])/norm_b)
            last_checked[j]=iteration;true_checks[j]+=1
        aliases=aliases and aliases_now(op)
        if callback:callback(iteration,indices,full,residual,op)
        return full,residual
    for iteration in range(1,maxiter+1):
        if process_time()>action_deadline:
            iteration-=1;status='cpu_screen_cap';break
        budget.live=[*xs,*rs,*ps]
        aps=op.apply(ms,ps);operator_batches+=1;scalar_products+=q
        budget.live=[*xs,*rs,*ps,*aps]
        bad=False
        for j in range(q):
            old_rr=float(rs[j]@rs[j]);curvature=float(ps[j]@aps[j])
            if curvature<=0 or not np.isfinite(curvature):bad=True;break
            alpha=old_rr/curvature
            budget.observe('CG_vector_update',temporary_allowance=2*ps[j].nbytes)
            xs[j]+=alpha*ps[j];rs[j]-=alpha*aps[j]
            ps[j]*=float(rs[j]@rs[j])/old_rr;ps[j]+=rs[j]
        del aps
        budget.live=[*xs,*rs,*ps]
        aliases=aliases and aliases_now(op)
        if bad:status='breakdown';break
        suspected=[np.linalg.norm(r)<=rtol*norm_b for r in rs]
        eligible=(list(range(q)) if iteration%20==0 or iteration==maxiter else
                  [j for j in range(q) if suspected[j]])
        if eligible:
            full,residual=certify(eligible,iteration)
            converged=[j for j in eligible if true_relative[j]<=rtol]
            if converged:
                winner=converged[0];status='converged'
                first_verified_converged=converged
                last_full=full[:,eligible.index(winner)]
                last_full_indices=eligible
                del residual,full
                break
            for col,j in enumerate(eligible):
                if suspected[j]:
                    rs[j][:]=residual[ms[j]:,col];ps[j][:]=rs[j]
                    replacements[j]+=1
            del full,residual
    if status!='converged':
        full,residual=certify(list(range(q)),iteration)
        converged=[j for j in range(q) if true_relative[j]<=rtol]
        if converged:
            winner=converged[0];status='converged';first_verified_converged=converged
            last_full=full[:,winner];last_full_indices=list(range(q))
        del residual,full
    # Bounded metadata: do not retain an ever-growing alias/checkpoint list.
    aliases=aliases and aliases_now(op)
    base=op.base
    index_hash=(sha256(base.S.tobytes()).hexdigest() if base else None)
    selected=base.S.tolist() if base else []
    row=dict(mode=mode,status=status,converged=status=='converged',iterations=iteration,
        winner_size=(ms[winner] if winner is not None else None),candidate_sizes=ms,
        first_verified_converged_sizes=[ms[j] for j in first_verified_converged],
        true_relative_residuals=true_relative,last_checked_iterations=last_checked,
        original_verifications=true_checks,literal_original_kernel_passes=kernel_passes,
        residual_replacements=replacements,operator_batches=operator_batches,
        scalar_operator_products=scalar_products,setup_seconds=setup_seconds,
        setup_cpu_seconds=setup_cpu_seconds,total_seconds=perf_counter()-wall,
        total_cpu_seconds=process_time()-cpu,kernel_stats=kernel.stats,
        setup_kernel_stats=setup_stats,memory=budget.report(),all_core_buffer_aliases=aliases,
        selected_indices=selected,selected_indices_sha256=index_hash,
        common_immutable_input_bytes=X.nbytes+b.nbytes,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        final_full_buffer_columns=(len(last_full_indices) if last_full is not None else 0))
    return last_full,row


def dense_single(A,b,S,rtol=1e-8,maxiter=200):
    n=len(b);mask=np.ones(n,dtype=bool);mask[S]=False
    R=np.flatnonzero(mask);m=len(S)
    if m:
        D=A[np.ix_(R,R)]-A[np.ix_(R,S)]@linalg.solve(A[np.ix_(S,S)],A[np.ix_(S,R)],assume_a='pos')
        rhs=b[R]-A[np.ix_(R,S)]@linalg.solve(A[np.ix_(S,S)],b[S],assume_a='pos')
    else:D=A;rhs=b.copy()
    x=np.zeros_like(rhs);r=rhs.copy();p=r.copy();norm_b=np.linalg.norm(b)
    for iteration in range(1,maxiter+1):
        ap=D@p;rr=float(r@r);alpha=rr/float(p@ap)
        x+=alpha*p;r-=alpha*ap;p*=float(r@r)/rr;p+=r
        if iteration%20==0 or np.linalg.norm(r)<=rtol*norm_b:
            full=np.empty(n);full[R]=x
            if m:full[S]=linalg.solve(A[np.ix_(S,S)],b[S]-A[np.ix_(S,R)]@x,assume_a='pos')
            rel=float(np.linalg.norm(b-A@full)/norm_b)
            if rel<=rtol:return iteration,full,rel
            if np.linalg.norm(r)<=rtol*norm_b:r=(b-A@full)[R];p=r.copy()
    raise AssertionError('Independent dense reference failed')


def validation():
    set_budget(CAP);rng=np.random.default_rng(62201125)
    X=np.ascontiguousarray(rng.normal(size=(37,5)));b=rng.normal(size=37)
    bw=1.4;lam=.015;maximum=9
    A=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))+lam*np.eye(len(b))
    calls=[]
    def check(iteration,indices,full,residual,op):
        for col,j in enumerate(indices):
            exact=b-A@full[:,col]
            err=np.linalg.norm(residual[:,col]-exact[op.permutation])/np.linalg.norm(b)
            assert err<1e-12
            calls.append(dict(iteration=iteration,candidate_size=[0,maximum][j],
                              original_relative_residual=float(np.linalg.norm(exact)/np.linalg.norm(b)),
                              literal_residual_relative_error=float(err)))
    full,row=run(X,b,bw,lam,'race',maximum=maximum,callback=check,maxiter=200,per_run_cpu=8.)
    S=np.asarray(row['selected_indices'],dtype=int)
    independent=[dense_single(A,b,np.array([],dtype=int)),dense_single(A,b,S)]
    winning_index=int(independent[1][0]<independent[0][0])
    exact=linalg.solve(A,b,assume_a='pos')
    solution_error=float(np.linalg.norm(full-exact)/np.linalg.norm(exact))
    assert row['converged'] and row['iterations']==min(t[0] for t in independent)
    assert row['winner_size']==[0,maximum][winning_index] and solution_error<2e-8
    assert not any(c['original_relative_residual']<=1e-8 and c['iteration']<row['iterations'] for c in calls)
    return dict(n=len(b),maximum=maximum,checks=calls,winner_size=row['winner_size'],
        iterations=row['iterations'],independent_branch_iterations=[t[0] for t in independent],
        solution_relative_error=solution_error,true_relative_residual=row['true_relative_residuals'][winning_index],
        all_core_buffer_aliases=row['all_core_buffer_aliases'])


def trace_path(X,b,bw,lam,mode,maximum=MAXIMUM):
    """Separate untimed setup+one step+literal original verification."""
    set_budget(CAP);tracemalloc.start()
    try:
        budget=frontier.Budget([X,b]);kernel=frontier.SharedKernel(X,bw,budget)
        ms={'race':[0,maximum],'shared_plain':[0],'shared_schur':[maximum],
            'natural_plain':[0],'natural_schur':[maximum]}[mode]
        op=Natural(kernel,ms[0],lam) if mode.startswith('natural') else CertifiedShared(kernel,maximum,lam)
        rs=op.rhs(ms,b);xs=[np.zeros_like(r) for r in rs];ps=[r.copy() for r in rs]
        budget.live=[*xs,*rs,*ps];aps=op.apply(ms,ps)
        budget.live=[*xs,*rs,*ps,*aps]
        for x,r,p,ap in zip(xs,rs,ps,aps):
            budget.observe('CG_vector_update',temporary_allowance=2*p.nbytes)
            shared.cg_update(x,r,p,ap)
        del aps,ap
        budget.live=[*xs,*rs,*ps]
        full,residual=op.verify(ms,xs,b)
        current,peak=tracemalloc.get_traced_memory()
        base=op.base
        if base:peak=max(peak,base.factor_trace['earlier_peak'])
        return dict(mode=mode,memory=budget.report(),tracemalloc_peak_bytes=peak,
            trace_within_budget=peak<=CAP,all_core_buffer_aliases=aliases_now(op),
            verification_relative_residuals=[float(np.linalg.norm(residual[:,j])/np.linalg.norm(b)) for j in range(len(ms))])
    finally:tracemalloc.stop()


def provenance():
    files=[Path(__file__).resolve(),Path(shared.__file__).resolve(),Path(frontier.__file__).resolve()]
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(88,90))
    output=ROOT/('results/shared-schur-race-validation.json' if args.validate_only else
                 'results/shared-schur-race-pilot.json')
    if output.exists():raise FileExistsError(output)
    started,cpu=perf_counter(),process_time()
    record=dict(status='running',source_sha256=provenance(),protocol=dict(
        budget_bytes=CAP,reserve_bytes=RESERVE,maximum=MAXIMUM,candidates=[0,MAXIMUM],
        datasets=['smooth','mixed','Concrete'],regularizations=[.001,1e-5],seed=0,repeats=1,
        rtol=1e-8,maxiter=2500,per_run_cpu_cap=8.,terminal_verification_reserve_seconds=.2,
        global_cpu_cap=90,blas_threads=1,true_checks='Every20steps for all branches and at own suspected convergence; literal original kernel entries. First certified branch wins, candidate order breaks simultaneous ties.',
        failure_policy='Only a false suspected stop replaces that branch residual/direction. No pruning or other restarts.',
        controls=['natural_plain','natural_schur','shared_plain','shared_schur'],
        shared_cross_feature_column_chunk=128,
        immutable_inputs='Shared X/b additional to algorithm-owned arrays; temporary feature packs counted.',
        lock='/tmp/cs6220-project-benchmark.lock',interpretation='One native pilot, no confirmation seeds or novelty claim'),
        runs=[],allocation_audits=[])
    def save():
        record.update(elapsed_cpu_seconds=process_time()-cpu,elapsed_wall_seconds=perf_counter()-started)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        if args.validate_only:
            record['tiny_full_reference']=validation()
            for name,file in [('smooth','synthetic-n640-smooth'),('Concrete','concrete-observed')]:
                data=np.load(ROOT/f'data/prepared/{file}-s0-lambda0.001.npz')
                X,b=data['X_train'],data['y_train'];bw=float(data['bandwidth'])
                for mode in ['race','natural_plain','natural_schur','shared_plain','shared_schur']:
                    audit=trace_path(X,b,bw,1e-5,mode);audit['dataset']=name
                    record['allocation_audits'].append(audit)
                    assert audit['trace_within_budget'] and audit['all_core_buffer_aliases']
            record['status']='validated';save();print(json.dumps(dict(output=str(output),status=record['status'])));return
        validation_path=ROOT/'results/shared-schur-race-validation.json'
        checked=json.loads(validation_path.read_text())
        if checked['status']!='validated' or checked['source_sha256']!=provenance():
            raise ValueError('The frozen source must pass allocation/full-solve validation first')
        record['validation_sha256']=sha256(validation_path.read_bytes()).hexdigest();save()
        for name,file in [('smooth','synthetic-n640-smooth'),('mixed','synthetic-n640-mixed'),('Concrete','concrete-observed')]:
            data_path=ROOT/f'data/prepared/{file}-s0-lambda0.001.npz';data=np.load(data_path)
            X,b=data['X_train'],data['y_train'];bw=float(data['bandwidth'])
            for lam in [.001,1e-5]:
                # One fixed ordering, no time-dependent or result-dependent selection.
                for mode in ['natural_plain','natural_schur','shared_plain','shared_schur','race']:
                    if process_time()-cpu>83:
                        record['status']='global_cpu_cap';save();return
                    try:
                        full,row=run(X,b,bw,lam,mode,deadline=cpu+85.)
                        row.update(dataset=name,n=len(b),regularization=lam,data_sha256=sha256(data_path.read_bytes()).hexdigest())
                        del full
                    except (MemoryError,np.linalg.LinAlgError) as error:
                        row=dict(dataset=name,regularization=lam,mode=mode,status='failure',error=str(error))
                    record['runs'].append(row);save()
                    print(json.dumps({k:row[k] for k in ['dataset','regularization','mode','status','iterations','winner_size','total_seconds'] if k in row}),flush=True)
        # Allocation audits are separate from native timings, one per shape and
        # method; mixed shares smooth shape. They always stop tracemalloc.
        record['allocation_audits']=checked['allocation_audits']
        record['allocation_audit_source']='Separate pre-timing validation artifact; same frozen code and shapes, lambda1e-5.'
    record['status']='complete';save()
    print(json.dumps(dict(output=str(output),status=record['status'],elapsed_cpu_seconds=record['elapsed_cpu_seconds'])))


if __name__=='__main__':main()
