"""Numerical feasibility of a core-only, streamed Nyström inverse.

Performance objects never retain n-by-m arrays. Dense C, U, M and applied H
are independent validation oracles only. This changes numerical rank under
a storage allowance; it is not a fixed-rank column-exchange method.
"""
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel


def unique_bytes(*arrays):
    roots={}
    for value in arrays:
        if value is None:continue
        while isinstance(value.base,np.ndarray):value=value.base
        roots[id(value)]=value.nbytes
    return sum(roots.values())


def rows(kernel,indices,row_size):
    for start in range(0,kernel.n,row_size):
        stop=min(kernel.n,start+row_size)
        yield slice(start,stop),kernel.entries(np.arange(start,stop),indices)


class StreamedOrthogonalInverse:
    """Store T and theta, with U=C T reconstructed only one row block at a time.

    H=(I-UU.T)^2/lambda + U diag(1/(lambda+theta)) U.T is SPD for
    arbitrary finite U in exact arithmetic. Orthogonal U makes this exactly
    the inverse of lambda I + U diag(theta) U.T. Unlike a subtractive
    Woodbury core, the construction tolerates small Gram errors without
    creating an intrinsically indefinite preconditioner.
    """
    def __init__(self,kernel,indices,regularization,row_size=16):
        self.kernel=kernel;self.indices=np.array(indices,dtype=np.int64,copy=True)
        self.regularization=float(regularization);self.row_size=row_size
        m=len(indices);n=kernel.n
        before=dict(kernel.stats)
        W=kernel.entries(self.indices,self.indices)
        w,V=linalg.eigh((W+W.T)*.5,check_finite=False)
        threshold=32*np.finfo(float).eps*m*w[-1]
        keep=w>threshold
        E=V[:,keep]/np.sqrt(w[keep])
        rank=E.shape[1]
        peak=unique_bytes(self.indices,W,w,V,E)
        diagnostics=dict(requested_landmarks=m,core_rank=rank,
            W_min_eigenvalue=float(w[0]),W_max_eigenvalue=float(w[-1]),
            W_truncation_threshold=float(threshold))
        del W,w,V
        R=np.empty((0,rank))
        for window,C in rows(kernel,self.indices,row_size):
            F=C@E
            stacked=np.vstack((R,F))
            raw=linalg.qr(stacked,mode='r',check_finite=False)[0]
            Rnext=raw[:min(raw.shape[0],rank),:].copy()
            peak=max(peak,unique_bytes(self.indices,E,R,C,F,stacked,raw,Rnext))
            R=Rnext
        del C,F,stacked,raw,Rnext
        left,s,right=linalg.svd(R,full_matrices=False,check_finite=False)
        keep=s>32*np.finfo(float).eps*max(n,rank)*s[0]
        T=(E@right.T[:,keep])/s[keep]
        theta=s[keep]**2
        peak=max(peak,unique_bytes(self.indices,E,R,left,s,right,T,theta))
        del E,R,left,s,right
        rank=len(theta)
        G=np.zeros((rank,rank))
        for window,C in rows(kernel,self.indices,row_size):
            U=C@T
            G+=U.T@U
            peak=max(peak,unique_bytes(self.indices,T,theta,G,C,U))
        del C,U
        diagnostics['initial_basis_gram_defect_2']=float(linalg.norm(G-np.eye(rank),2))
        g,Z=linalg.eigh((G+G.T)*.5,check_finite=False)
        if g[0]<=0:raise ArithmeticError('implicit basis Gram is not SPD')
        half=(Z*np.sqrt(g))@Z.T
        inverse_half=(Z/np.sqrt(g))@Z.T
        core=(half*theta)@half
        values,rotation=linalg.eigh((core+core.T)*.5,check_finite=False)
        order=np.argsort(values)[::-1]
        values,rotation=values[order],rotation[:,order]
        if values[-1]<-128*np.finfo(float).eps*values[0]:
            raise ArithmeticError('corrected small kernel core is indefinite')
        values=np.maximum(values,0.)
        newT=(T@inverse_half)@rotation
        peak=max(peak,unique_bytes(self.indices,T,theta,G,g,Z,half,inverse_half,core,values,rotation,newT))
        self.T=newT;self.theta=values
        del T,theta,G,g,Z,half,inverse_half,core,values,rotation,newT
        G=np.zeros((rank,rank))
        for window,C in rows(kernel,self.indices,row_size):
            U=C@self.T
            G+=U.T@U
            peak=max(peak,unique_bytes(self.indices,self.T,self.theta,G,C,U))
        diagnostics.update(final_basis_gram_defect_2=float(linalg.norm(G-np.eye(rank),2)),
            rank=rank,setup_named_array_peak_bytes=int(peak),
            setup_array_allowance_with_lapack_bytes=int(peak+8*(4*m*m+8*m)),
            setup_kernel_entries=int(kernel.stats.get('entry_evaluations',0)-before.get('entry_evaluations',0)),
            setup_entry_blocks=int(kernel.stats.get('entry_blocks',0)-before.get('entry_blocks',0)),
            setup_passes='one W query + QR pass + basis Gram pass + corrected Gram audit pass; all charged')
        self.diagnostics=diagnostics

    @property
    def storage_bytes(self):
        return unique_bytes(self.indices,self.T,self.theta)

    def apply(self,values,mode='row_basis'):
        """Vector performance path; multiple RHS only for dense validation.

        row_basis forms U_block=C_block T. factored uses T.T(C.T r) and
        C(T q), reducing arithmetic but potentially increasing cancellation.
        Both require exactly three queried row streams.
        """
        values=np.asarray(values)
        vector=values.ndim==1
        rhs=values[:,None] if vector else values
        rank=len(self.theta);width=rhs.shape[1];m=len(self.indices)
        if mode=='row_basis':
            q=np.zeros((rank,width))
            for window,C in rows(self.kernel,self.indices,self.row_size):
                U=C@self.T;q+=U.T@rhs[window]
            t=np.empty_like(rhs);qt=np.zeros_like(q)
            for window,C in rows(self.kernel,self.indices,self.row_size):
                U=C@self.T;t[window]=rhs[window]-U@q;qt+=U.T@t[window]
            coefficient=q/(self.regularization+self.theta[:,None])
            out=np.empty_like(rhs)
            for window,C in rows(self.kernel,self.indices,self.row_size):
                U=C@self.T
                out[window]=(t[window]-U@qt)/self.regularization+U@coefficient
        elif mode=='factored':
            raw=np.zeros((m,width))
            for window,C in rows(self.kernel,self.indices,self.row_size):raw+=C.T@rhs[window]
            q=self.T.T@raw;first=self.T@q
            t=np.empty_like(rhs);raw_t=np.zeros_like(raw)
            for window,C in rows(self.kernel,self.indices,self.row_size):
                t[window]=rhs[window]-C@first;raw_t+=C.T@t[window]
            qt=self.T.T@raw_t
            second=self.T@qt
            spectral=self.T@(q/(self.regularization+self.theta[:,None]))
            out=np.empty_like(rhs)
            for window,C in rows(self.kernel,self.indices,self.row_size):
                out[window]=(t[window]-C@second)/self.regularization+C@spectral
        else:raise ValueError(mode)
        return out[:,0] if vector else out


class StreamedWoodburyInverse:
    def __init__(self,kernel,indices,regularization,row_size=16):
        self.kernel=kernel;self.indices=np.array(indices,dtype=np.int64,copy=True)
        self.regularization=regularization;self.row_size=row_size
        before=dict(kernel.stats)
        W=kernel.entries(self.indices,self.indices)
        G=np.zeros_like(W)
        for window,C in rows(kernel,self.indices,row_size):G+=C.T@C
        core=(regularization*W+G);core=(core+core.T)*.5
        spectrum=linalg.eigvalsh(core,check_finite=False)
        self.diagnostics=dict(core_min_eigenvalue=float(spectrum[0]),core_max_eigenvalue=float(spectrum[-1]),
            core_condition_from_eigenvalues=float(spectrum[-1]/spectrum[0]) if spectrum[0]>0 else None,
            setup_kernel_entries=int(kernel.stats.get('entry_evaluations',0)-before.get('entry_evaluations',0)))
        self.L=linalg.cholesky(core,lower=True,check_finite=False)
    @property
    def storage_bytes(self):return self.indices.nbytes+self.L.nbytes
    def apply(self,values):
        vector=values.ndim==1;rhs=values[:,None] if vector else values
        q=np.zeros((len(self.indices),rhs.shape[1]))
        for window,C in rows(self.kernel,self.indices,self.row_size):q+=C.T@rhs[window]
        coefficient=linalg.cho_solve((self.L,True),q,check_finite=False)
        out=np.empty_like(rhs)
        for window,C in rows(self.kernel,self.indices,self.row_size):out[window]=(rhs[window]-C@coefficient)/self.regularization
        return out[:,0] if vector else out


def vector_memory(model,mode):
    n,m,r,b=model.kernel.n,len(model.indices),len(model.theta),model.row_size
    # Count input, t, output, row C, basis if formed, and conservative small
    # coefficient/temporary buffers. Shared X and indexed feature copies are
    # separate, so these are not claims about exact total process RSS.
    workspace=8*(3*n+b*m+(b*r+5*r+4*b if mode=='row_basis' else 7*m+4*r+4*b))
    feature=(m+b)*model.kernel.X.shape[1]*8
    return dict(retained_state_bytes=model.storage_bytes,
        vector_application_array_allowance_bytes=workspace,
        row_indexed_feature_copy_bytes=feature,
        state_plus_application_allowance_bytes=model.storage_bytes+workspace+feature,
        shared_kernel_X_bytes=model.kernel.X.nbytes,
        fits_n_times16_retained=model.storage_bytes<=8*n*16,
        fits_n_times32_retained=model.storage_bytes<=8*n*32,
        fits_n_times16_application=model.storage_bytes+workspace+feature<=8*n*16,
        fits_n_times32_application=model.storage_bytes+workspace+feature<=8*n*32)


def audit_application(model,apply,reference_M,reference_factor,reference_inverse,probes):
    before=dict(model.kernel.stats)
    vector_result=apply(probes[:,0])
    vector_query_entries=int(model.kernel.stats['entry_evaluations']-before.get('entry_evaluations',0))
    observed=apply(probes)
    expected=linalg.cho_solve(reference_factor,probes,check_finite=False)
    relative=np.linalg.norm(observed-expected,axis=0)/np.linalg.norm(expected,axis=0)
    # Dense applied H is an oracle only, with512 RHS and explicitly excluded
    # from vector-state/application memory accounting.
    H=apply(np.eye(len(reference_M)))
    symmetry=float(np.linalg.norm(H-H.T)/np.linalg.norm(H))
    symmetric=(H+H.T)*.5
    spectrum=linalg.eigvalsh(symmetric,check_finite=False)
    L=np.tril(reference_factor[0])
    congruence=L.T@symmetric@L
    inverse_error=float(np.max(np.abs(linalg.eigvalsh((congruence+congruence.T)*.5,check_finite=False)-1.)))
    return dict(vector_query_entries=vector_query_entries,
        vector_vs_multiple_rhs_action_relative_error=float(np.linalg.norm(vector_result-observed[:,0])/np.linalg.norm(vector_result)),
        probe_relative_action_errors=relative.tolist(),max_probe_relative_action_error=float(relative.max()),
        dense_applied_H_relative_asymmetry=symmetry,
        dense_symmetrized_applied_H_min_eigenvalue=float(spectrum[0]),
        dense_symmetrized_applied_H_max_eigenvalue=float(spectrum[-1]),
        dense_inverse_relative_frobenius_error=float(np.linalg.norm(H-reference_inverse)/np.linalg.norm(reference_inverse)),
        M_weighted_inverse_error_2=inverse_error,
        floating_symmetric_positive_check=bool(symmetry<1e-10 and spectrum[0]>0))


def main():
    output=ROOT/'results/streamed-nystrom-inverse-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(29,30))
    started,cpu_start=perf_counter(),process_time()
    data_path=ROOT/'data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz'
    data=np.load(data_path);X=data['X_train'];bandwidth=float(data['bandwidth'])
    assert len(X)==512
    selected=np.random.default_rng(np.random.SeedSequence([0,6220,93])).permutation(512)
    record=dict(status='running',scope='Numerical feasibility/control only; larger effective rank at a storage allowance, not fixed rank, novelty, or an end-to-end solve claim',
        protocol=dict(n=512,landmarks=[64,128],regularizations=[.001,1e-5],bandwidth_multipliers=[1.,4.],
            stress_case='4x original bandwidth is explicitly a conditioning stress instance, not a new claimed project workload',
            selected_indices='nested uniform permutation prefixes; no n-by-m RP initialization',seed_sequence=[0,6220,93],
            row_size=16,blas_threads=1,cpu_cap_seconds=30,
            setup='row-streamed QR of C Wdaggerhalf, small SVD, one Gram-based basis correction with transformed spectral core; corrected Gram measured in a charged extra pass',
            positive_action='P^2/lambda + U diag(1/(lambda+theta)) U.T, P=I-U U.T, U=C T',
            applications=['explicit row basis','factored C/T contractions','raw streamed Woodbury'],
            no_full_basis_in_performance_state=True,dense_oracles='C,F,U,M and applied H only for independent numerical validation;512-RHS audit is not vector workspace or performance evidence',
            expected_positive_application_entries='3*n*m per vector; row-basis O(n*m*r) arithmetic, factored O(n*m+m*r) arithmetic',
            expected_raw_application_entries='2*n*m per vector; O(n*m+m^2) arithmetic'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'src/nystrom_project/numerics.py']},
        data_sha256=sha256(data_path.read_bytes()).hexdigest(),cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        for multiplier in [1.,4.]:
            for m in [64,128]:
                for lam in [.001,1e-5]:
                    if process_time()-cpu_start>25:record['status']='cpu_budget';save();return
                    kernel=GaussianKernel(X,bandwidth*multiplier)
                    indices=selected[:m].copy()
                    case=dict(bandwidth_multiplier=multiplier,landmarks=m,regularization=lam,
                        indices=indices.tolist(),runs=[])
                    record['cases'].append(case)
                    try:
                        model=StreamedOrthogonalInverse(kernel,indices,lam)
                    except (ArithmeticError,np.linalg.LinAlgError) as error:
                        case['setup_failure']=str(error);save();continue
                    case.update(setup=model.diagnostics,retained_array_shapes=dict(indices=list(model.indices.shape),
                        T=list(model.T.shape),theta=list(model.theta.shape)),n_times_k_byte_budgets={'16':512*16*8,'32':512*32*8})
                    # Independent dense reference, built from the exact same
                    # selected-column Nyström core truncation rule.
                    C=kernel.entries(np.arange(512),indices)
                    W=C[indices]
                    w,V=linalg.eigh((W+W.T)*.5,check_finite=False)
                    keep=w>32*np.finfo(float).eps*m*w[-1]
                    F=(C@V[:,keep])/np.sqrt(w[keep])
                    denseU,s,_=linalg.svd(F,full_matrices=False,check_finite=False)
                    M=F@F.T+lam*np.eye(512)
                    reference_factor=linalg.cho_factor(M,lower=True,check_finite=False)
                    reference_inverse=linalg.cho_solve(reference_factor,np.eye(512),check_finite=False)
                    probes=np.column_stack((data['y_train'],np.random.default_rng(622094).normal(size=(512,2)),denseU[:,0],denseU[:,-1]))
                    case['dense_reference']=dict(nystrom_rank=F.shape[1],
                        M_cholesky_spd=True,min_M_eigenvalue=float(linalg.eigvalsh(M,subset_by_index=[0,0],check_finite=False)[0]),
                        max_F_singular_value=float(s[0]),min_F_singular_value=float(s[-1]),
                        dense_M_inverse_residual=float(np.linalg.norm(M@reference_inverse-np.eye(512))/np.sqrt(512)))
                    for mode in ['row_basis','factored']:
                        audit=audit_application(model,lambda rhs:model.apply(rhs,mode),M,reference_factor,reference_inverse,probes)
                        case['runs'].append(dict(method=mode,**audit,**vector_memory(model,mode)))
                    try:
                        raw=StreamedWoodburyInverse(kernel,indices,lam)
                        audit=audit_application(raw,raw.apply,M,reference_factor,reference_inverse,probes)
                        case['runs'].append(dict(method='raw_woodbury',**audit,retained_state_bytes=raw.storage_bytes,setup=raw.diagnostics))
                    except np.linalg.LinAlgError as error:
                        case['runs'].append(dict(method='raw_woodbury',setup_failure=str(error)))
                    save()
                    print(json.dumps(dict(multiplier=multiplier,m=m,lam=lam,rank=len(model.theta),
                        gram=model.diagnostics['final_basis_gram_defect_2'],
                        runs=[dict(method=v['method'],weighted_error=v.get('M_weighted_inverse_error_2'),
                            min_eigenvalue=v.get('dense_symmetrized_applied_H_min_eigenvalue'),failure=v.get('setup_failure')) for v in case['runs']])),flush=True)
    record['status']='complete';save()


if __name__=='__main__':main()
