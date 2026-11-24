"""Same-factor native PCG: direct streamed projections vs fused reuse.

The stable positive row-basis inverse is unchanged. Fused products collect
U.T@(A p) from row-local U already recoverable from the current full K block.
No n-by-m basis is retained. All candidate setup/application/operator arrays
are counted against a predeclared1MiB allowance; dense H is validation only.
"""
from __future__ import annotations

from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import traceback

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel
from check_streamed_nystrom_inverse import StreamedOrthogonalInverse,unique_bytes

MEMORY_LIMIT=1024*1024


class MemoryLedger:
    def __init__(self,model,b):
        self.fixed=[model.indices,model.T,model.theta,model.kernel.X,model.selected_points,b]
        self.live=[]
        self.peak=0
        self.stages={}
    def observe(self,stage,*arrays):
        size=unique_bytes(*self.fixed,*self.live,*arrays)
        self.stages[stage]=max(size,self.stages.get(stage,0))
        self.peak=max(self.peak,size)
        if size>MEMORY_LIMIT:raise MemoryError(f'{stage}: {size}>1MiB')


class NativeStreamedInverse:
    def __init__(self,kernel,indices,lam):
        self.core=StreamedOrthogonalInverse(kernel,indices,lam,row_size=16)
        self.indices=self.core.indices;self.T=self.core.T;self.theta=self.core.theta
        self.kernel=kernel;self.lam=lam;self.n=kernel.n
        self.selected_points=np.ascontiguousarray(kernel.X[self.indices])
        self.memory=None
        self.counters=dict(H_applications=0,C_passes=0,C_entry_queries=0,
            C_query_seconds=0.,basis_seconds=0.,projection_seconds=0.,
            full_A_passes=0,full_A_rhs=0,full_A_seconds=0.,fused_projection_passes=0)

    @property
    def storage_bytes(self):
        return unique_bytes(self.indices,self.T,self.theta,self.selected_points)

    def basis_rows(self):
        self.counters['C_passes']+=1
        for start in range(0,self.n,16):
            stop=min(start+16,self.n)
            begin=perf_counter()
            C=self.kernel._block(self.kernel.X[start:stop],self.selected_points)
            self.counters['C_query_seconds']+=perf_counter()-begin
            count=(stop-start)*len(self.indices)
            self.counters['C_entry_queries']+=count
            self.kernel.stats['entry_blocks']=self.kernel.stats.get('entry_blocks',0)+1
            self.kernel.stats['entry_evaluations']=self.kernel.stats.get('entry_evaluations',0)+count
            begin=perf_counter();U=C@self.T
            self.counters['basis_seconds']+=perf_counter()-begin
            yield slice(start,stop),C,U
            # Release both row arrays before acquiring the next block.
            del C,U

    def project(self,r):
        q=np.zeros(len(self.theta))
        for window,C,U in self.basis_rows():
            begin=perf_counter();q+=U.T@r[window]
            self.counters['projection_seconds']+=perf_counter()-begin
            if window.start==0 and self.memory:self.memory.observe('projection',r,q,C,U)
        return q

    def apply(self,r,c=None):
        self.counters['H_applications']+=1
        q=self.project(r) if c is None else c
        t=np.empty_like(r);qt=np.zeros_like(q)
        for window,C,U in self.basis_rows():
            t[window]=r[window]-U@q
            qt+=U.T@t[window]
            if window.start==0 and self.memory:self.memory.observe('H_middle',r,q,t,qt,C,U)
        spectral=q/(self.lam+self.theta)
        out=np.empty_like(r)
        for window,C,U in self.basis_rows():
            out[window]=(t[window]-U@qt)/self.lam+U@spectral
            if window.start==0 and self.memory:self.memory.observe('H_final',r,q,t,qt,spectral,out,C,U)
        return out

    def A_product(self,p,*,collect=False,b=None,recursive_r=None):
        """One K pass; optional projections use selected columns of its tile.

        During a true-residual check, collect both the true and recursive
        residual projections so the cache always matches the retained r.
        """
        start=perf_counter()
        result=np.empty(self.n)
        q=np.zeros(len(self.theta)) if collect else None
        q_recursive=np.zeros(len(self.theta)) if collect and recursive_r is not None else None
        for first in range(0,self.n,128):
            last=min(first+128,self.n)
            tile=self.kernel._block(self.kernel.X[first:last],self.kernel.X)
            result[first:last]=tile@p
            result[first:last]+=self.lam*p[first:last]
            if collect:
                for offset in range(0,last-first,16):
                    end=min(offset+16,last-first)
                    # Use the same16-row C-order multiply as H's basis stream.
                    packed=np.array(tile[offset:end,self.indices],order='C',copy=True)
                    begin=perf_counter();U=packed@self.T
                    self.counters['basis_seconds']+=perf_counter()-begin
                    window=slice(first+offset,first+end)
                    values=result[window] if b is None else b[window]-result[window]
                    begin=perf_counter();q+=U.T@values
                    if q_recursive is not None:q_recursive+=U.T@recursive_r[window]
                    self.counters['projection_seconds']+=perf_counter()-begin
                    if first==0 and offset==0 and self.memory:
                        self.memory.observe('A_fused',p,result,q,q_recursive,tile,packed,U,values,b,recursive_r)
                    del packed,U,values
            elif first==0 and self.memory:
                self.memory.observe('A_direct',p,result,tile,b,recursive_r)
            # Without this deletion two128-by-n tiles overlap on reassignment.
            del tile
        elapsed=perf_counter()-start
        self.kernel.stats['matvecs']+=1
        self.kernel.stats['matvec_seconds']+=elapsed
        self.counters['full_A_passes']+=1
        self.counters['full_A_rhs']+=1
        self.counters['full_A_seconds']+=elapsed
        self.counters['fused_projection_passes']+=int(collect)
        return result,q,q_recursive


def solve(model,b,mode,*,rtol=1e-8,maxiter=2500,capture=False,explicit_H=None):
    x=np.zeros_like(b);r=b.copy();norm_b=np.linalg.norm(b)
    fused=mode=='fused'
    c=model.project(r) if fused else None
    def H(value,coefficient):
        return explicit_H@value if explicit_H is not None else model.apply(value,coefficient if fused else None)
    z=H(r,c);p=z.copy();rho=float(r@z)
    trace=[];cache_checks=[];replacements=[]
    converged=False;breakdown=None;true_relative=None
    last_true_iteration=-1
    for iteration in range(1,maxiter+1):
        if model.memory:model.memory.live=[x,r,z,p,c]
        ap,cap,_=model.A_product(p,collect=fused)
        curvature=float(p@ap)
        if curvature<=0 or rho<=0 or not np.isfinite(curvature+rho):
            breakdown='nonpositive_or_nonfinite_PCG_inner_product';break
        alpha=rho/curvature
        x+=alpha*p;r-=alpha*ap
        if fused:c-=alpha*cap
        del ap,cap
        if capture:trace.append(x.copy())
        suspected=np.linalg.norm(r)<=rtol*norm_b
        if iteration%20==0 or suspected or iteration==maxiter:
            if model.memory:model.memory.live=[x,r,z,p,c]
            ax,ctrue,crec=model.A_product(x,collect=fused,b=b,recursive_r=r if fused else None)
            true_r=b-ax
            del ax
            true_relative=float(np.linalg.norm(true_r)/norm_b)
            last_true_iteration=iteration
            if fused:
                cache_checks.append(dict(iteration=iteration,
                    projection_gap_relative_to_rhs=float(np.linalg.norm(c-crec)/norm_b),
                    projection_gap_relative_to_current_projection=float(np.linalg.norm(c-crec)/max(np.linalg.norm(crec),1e-300))))
                c=crec
            if true_relative<=rtol:
                converged=True;break
            if suspected:
                r=true_r
                if fused:c=ctrue
                if model.memory:model.memory.live=[x,r,z,p,c]
                z=H(r,c);rho=float(r@z);p=z.copy()
                replacements.append(iteration)
                del ctrue,crec,true_r
                continue
            del ctrue,crec,true_r
        if model.memory:model.memory.live=[x,r,z,p,c]
        z=H(r,c)
        new_rho=float(r@z)
        p=z+(new_rho/rho)*p;rho=new_rho
    if last_true_iteration!=iteration:
        ax,_,_=model.A_product(x,collect=False)
        true_relative=float(np.linalg.norm(b-ax)/norm_b)
        converged=true_relative<=rtol
    return dict(iterations=iteration,converged=bool(converged),true_relative_residual=true_relative,
        breakdown=breakdown,residual_replacement_restarts=replacements,
        cache_checks=cache_checks,counters=dict(model.counters),kernel_stats=dict(model.kernel.stats)),x,trace


def tiny_checks():
    rng=np.random.default_rng(622096)
    X=rng.normal(size=(48,5));b=rng.normal(size=48)
    indices=np.random.default_rng(93).permutation(48)[:12]
    results=[]
    for lam in [.001,1e-5]:
        kernel=GaussianKernel(X,.9)
        model=NativeStreamedInverse(kernel,indices,lam)
        C=kernel.entries(np.arange(48),indices)
        U=C@model.T;P=np.eye(48)-U@U.T
        H=P@P/lam+(U/(lam+model.theta))@U.T
        H_applied=np.column_stack([model.apply(column) for column in np.eye(48).T])
        H_error=float(np.linalg.norm(H_applied-H)/np.linalg.norm(H))
        assert H_error<1e-12
        A=kernel.dense()+lam*np.eye(48)
        exact=linalg.solve(A,b,assume_a='pos')
        runs={};traces={};solutions={}
        for mode in ['explicit','direct','fused']:
            run,x,trace=solve(model,b,mode,rtol=1e-10,maxiter=300,capture=True,
                explicit_H=H if mode=='explicit' else None)
            assert run['converged'],(mode,run)
            runs[mode]=dict(iterations=run['iterations'],true_relative_residual=run['true_relative_residual'],
                solution_relative_error=float(np.linalg.norm(x-exact)/np.linalg.norm(exact)))
            traces[mode]=trace;solutions[mode]=x
        differences={};early_differences={};solution_differences={}
        for mode in ['direct','fused']:
            count=min(len(traces['explicit']),len(traces[mode]))
            gaps=[np.linalg.norm(traces[mode][j]-traces['explicit'][j])/max(np.linalg.norm(traces['explicit'][j]),1e-300)
                for j in range(count)]
            differences[mode]=float(max(gaps))
            early_differences[mode]=float(max(gaps[:8]))
            solution_differences[mode]=float(np.linalg.norm(solutions[mode]-solutions['explicit'])/np.linalg.norm(solutions['explicit']))
            # Dense-vs-streamed scalar CG itself loses all-trajectory agreement
            # after roundoff changes loss of conjugacy. Preserve those gaps;
            # validate the action, early recurrence and converged solution.
            assert early_differences[mode]<1e-10,(mode,early_differences[mode])
            assert solution_differences[mode]<1e-8
            assert runs[mode]['solution_relative_error']<1e-8
        # Fused projection uses the same row-local basis, not C/T factored contractions.
        p=rng.normal(size=48)
        ap,q,_=model.A_product(p,collect=True)
        projection_error=float(np.linalg.norm(q-U.T@ap)/np.linalg.norm(U.T@ap))
        assert projection_error<1e-12
        results.append(dict(regularization=lam,runs=runs,
            dense_H_vs_streamed_action_relative_error=H_error,
            iterate_relative_errors_against_explicit_H=differences,
            first8_iterate_relative_errors=early_differences,
            final_solution_relative_gaps=solution_differences,
            strict_all_iterate_agreement_passed=all(value<1e-7 for value in differences.values()),
            finite_precision_note='Strict all-iterate agreement failed, including direct-vs-dense. Action, first8 iterates and final solutions agree; later CG rounding trajectories and iteration counts remain reported.',
            fused_projection_vs_independent_dense_U_error=projection_error,
            explicit_H_min_eigenvalue=float(linalg.eigvalsh((H+H.T)*.5)[0])))
    return results


def worker(lam,mode,connection,cpu_budget):
    try:
        soft=max(1,int(cpu_budget-.5))
        resource.setrlimit(resource.RLIMIT_CPU,(soft,soft+1))
        with threadpool_limits(limits=1):
            data=np.load(ROOT/'data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz')
            X,b=data['X_train'],data['y_train']
            indices=np.random.default_rng(np.random.SeedSequence([0,6220,93])).permutation(len(b))[:64]
            started,cpu=perf_counter(),process_time()
            kernel=GaussianKernel(X,float(data['bandwidth']))
            model=NativeStreamedInverse(kernel,indices,lam)
            setup_seconds=perf_counter()-started
            setup_stats=dict(kernel.stats)
            setup_allowance=(model.core.diagnostics['setup_array_allowance_with_lapack_bytes']
                +kernel.X.nbytes+b.nbytes+model.selected_points.nbytes)
            if setup_allowance>MEMORY_LIMIT:raise MemoryError('setup allowance exceeds1MiB')
            ledger=MemoryLedger(model,b);model.memory=ledger
            run,x,_=solve(model,b,mode)
            elapsed,cpu_elapsed=perf_counter()-started,process_time()-cpu
            # Add room for block-expression temporaries and small BLAS work.
            application_allowance=ledger.peak+8*(4*len(b)+8*len(indices))
            if application_allowance>MEMORY_LIMIT:raise MemoryError('application allowance exceeds1MiB')
            result=dict(regularization=lam,mode=mode,**run,total_seconds=elapsed,total_cpu_seconds=cpu_elapsed,
                setup_seconds=setup_seconds,solve_seconds=elapsed-setup_seconds,
                worker_cpu_seconds=process_time(),setup_kernel_stats=setup_stats,
                retained_factor_and_selected_points_bytes=model.storage_bytes,
                setup_array_allowance_bytes=setup_allowance,
                measured_named_live_array_peak_bytes=ledger.peak,
                application_array_allowance_bytes=application_allowance,
                algorithm_owned_peak_array_allowance_bytes=max(setup_allowance,application_allowance),
                memory_limit_bytes=MEMORY_LIMIT,memory_stages=ledger.stages,
                kernel_X_copy_bytes=kernel.X.nbytes,common_original_X_bytes=X.nbytes,common_b_bytes=b.nbytes,
                common_b_also_conservatively_counted_in_algorithm_allowance=True,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                factor_sha256=sha256(model.indices.tobytes()+model.T.tobytes()+model.theta.tobytes()).hexdigest(),
                solution_sha256=sha256(x.tobytes()).hexdigest(),
                setup_diagnostics=model.core.diagnostics,
                full_kernel_entry_equivalents=(kernel.stats['matvecs']*len(b)**2+kernel.stats.get('entry_evaluations',0))/len(b)**2)
            connection.send(dict(ok=True,result=result))
    except BaseException:
        connection.send(dict(ok=False,regularization=lam,mode=mode,error=traceback.format_exc(),worker_cpu_seconds=process_time()))
    finally:connection.close()


def main():
    output=ROOT/'results/streamed-projection-reuse-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    started,cpu_start=perf_counter(),process_time();child_cpu=0.
    record=dict(status='running',scope='Identical S/T/theta streamed Nyström control: current-solve projection reuse, no broader efficiency or research-success claim',
        protocol=dict(dataset='synthetic-n640-smooth',n=512,m=64,seed=0,regularizations=[.001,1e-5],
            modes=['direct','fused'],repeats=1,rtol=1e-8,maxiter=2500,
            H='P^2/lambda + U diag(1/(lambda+theta)) U.T; P=I-UU.T, U=C*T',
            basis_row_size=16,operator_row_size=128,algorithm_owned_peak_array_budget=MEMORY_LIMIT,
            direct='3 C passes per H application',
            fused='initial C projection;2 C passes per H; collect U.T(Ap) in existing full Kp tiles and update c-=alpha*U.T(Ap)',
            refresh='every20/suspected true check collects both U.T r_recursive and U.T r_true; cache always matches whichever residual is retained',
            residual_policy='same scalar PCG in both branches, restart only after a false recursive stopping test',
            iteration_order=[dict(regularization=.001,modes=['direct','fused']),dict(regularization=1e-5,modes=['fused','direct'])],
            factor_seed_sequence=[0,6220,93],blas_threads=1,cpu_cap_seconds=80,
            timing='fresh isolated fork per solve; kernel, factor setup, packed selected features and all solver work included; frozen data loading excluded',
            memory='named NumPy arrays and conservative setup/application workspace allowances; shared immutable inputs and interpreter/library RSS separately disclosed',
            dense_validation='tiny48x48 explicit H and A only; not available inside native workers',
            validation_limitation='All-trajectory equality is not assumed: the preserved tiny test exhibits finite-precision CG trajectory changes even for direct-vs-dense H; operator, first8 iterates, final solution and true tolerance are checked.',
            lock='/tmp/cs6220-project-benchmark.lock'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),
            ROOT/'scripts/check_streamed_nystrom_inverse.py',ROOT/'src/nystrom_project/numerics.py']},runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,total_accounted_cpu_seconds=process_time()-cpu_start+child_cpu)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):record['tiny_validations']=tiny_checks()
    save();ctx=mp.get_context('fork')
    for lam,modes in [(.001,['direct','fused']),(1e-5,['fused','direct'])]:
        for mode in modes:
            if process_time()-cpu_start+child_cpu>70:record['status']='cpu_budget';save();return
            receive,send=ctx.Pipe(duplex=False)
            remaining=80-(process_time()-cpu_start+child_cpu)
            process=ctx.Process(target=worker,args=(lam,mode,send,remaining));process.start();send.close()
            message=None
            while process.is_alive():
                if receive.poll(.2):
                    try:message=receive.recv()
                    except EOFError:pass
                    break
            process.join()
            if message is None and receive.poll():
                try:message=receive.recv()
                except EOFError:pass
            receive.close()
            if message is None:
                record['runs'].append(dict(regularization=lam,mode=mode,status='worker_stopped',exitcode=process.exitcode))
                record['status']='cpu_budget_or_worker_failure';save();return
            child_cpu+=message.get('result',message).get('worker_cpu_seconds',0.)
            if not message['ok']:
                record['runs'].append(message);record['status']='worker_failure';save();return
            run=message['result'];record['runs'].append(run);save()
            print(json.dumps({key:run[key] for key in ['regularization','mode','iterations','converged','true_relative_residual',
                'total_seconds','total_cpu_seconds','algorithm_owned_peak_array_allowance_bytes','full_kernel_entry_equivalents']}),flush=True)
    for lam in [.001,1e-5]:
        pair=[run for run in record['runs'] if run['regularization']==lam]
        assert len({run['factor_sha256'] for run in pair})==1
    record['status']='complete';save()


if __name__=='__main__':main()
