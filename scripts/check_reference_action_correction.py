"""Dense oracle for preserving reference actions after factor compression.

Compare persistent block-BFGS repair against spending the same reference actions
on an initial Galerkin correction. Dense eigenbasis, decoded factors, reference
actions and setup workspace are free. Retained representation bytes and all
outer A/verification products are reported; this is not native timing evidence.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator,cg
from threadpoolctl import threadpool_limits
from check_selective_precision_oracle import action

ROOT=Path(__file__).resolve().parents[1]
BUDGET=131072


def corrected(base,Y,S):
    C=Y.T@S
    C=(C+C.T)*.5
    factor=cho_factor(C,lower=True)
    def apply(v):
        coefficients=cho_solve(factor,S.T@v)
        z=base(v-Y@coefficients)
        return z-S@cho_solve(factor,Y.T@z)+S@coefficients
    return apply


def warm(A,b,x,S,residual):
    AS=A@S
    G=S.T@AS
    G=(G+G.T)*.5
    c=cho_solve(cho_factor(G,lower=True),S.T@residual)
    return x+S@c, S.shape[1]


def solve(A,b,preconditioner,x0):
    counts=dict(operator_products=0,verification_products=0,iterations=0)
    normb=np.linalg.norm(b)
    def verify(x):
        counts['verification_products']+=1
        return float(np.linalg.norm(b-A@x)/normb)
    initial=verify(x0)
    if initial<=1e-8:
        return dict(**counts,converged=True,true_relative_residual=initial,solver_info=0)
    saved=None
    class Finished(Exception):pass
    def product(v):
        counts['operator_products']+=1
        return A@v
    def callback(x):
        nonlocal saved
        counts['iterations']+=1
        if verify(x)<=1e-8:
            saved=x.copy()
            raise Finished
    op=LinearOperator(A.shape,matvec=product,dtype=float)
    M=LinearOperator(A.shape,matvec=preconditioner,dtype=float)
    try:
        x,info=cg(op,b,x0=x0,M=M,rtol=0.,atol=0.,maxiter=2500,callback=callback)
    except Finished:
        x,info=saved,0
    residual=verify(x)
    return dict(**counts,converged=bool(info==0 and residual<=1e-8),
                true_relative_residual=residual,solver_info=int(info))


def checks():
    rng=np.random.default_rng(622122)
    V=rng.normal(size=(13,13));Hq=V@V.T+np.eye(13)
    V=rng.normal(size=(13,13));Href=V@V.T+np.eye(13)
    Y=rng.normal(size=(13,2));S=Href@Y
    Hc=corrected(lambda v:Hq@v,Y,S)(np.eye(13))
    P=np.eye(13)-S@np.linalg.solve(Y.T@S,Y.T)
    interpolation=np.linalg.norm(Hc@Y-S)/np.linalg.norm(S)
    identity=np.linalg.norm(Hc-Href-P@(Hq-Href)@P.T)/np.linalg.norm(Href)
    assert interpolation<1e-12 and identity<1e-12
    assert np.linalg.eigvalsh((Hc+Hc.T)*.5)[0]>0
    return dict(interpolation_error=float(interpolation),error_identity_error=float(identity))


def main():
    out=ROOT/'results/reference-action-correction-oracle.json'
    if out.exists():raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    old=ROOT/'results/quantized-spectral-headroom.json'
    controls=json.loads(old.read_text())
    record=dict(status='running',protocol=dict(seed=0,datasets=['smooth','mixed','Concrete'],
        shifts=[.001,1e-5],retained_byte_allowance=BUDGET,
        rank_by_size={'512':224,'824':132},
        feedback='One plain-CG step; Y=[normalized b,normalized true r1]',
        reference='Exact eigenbasis spectral inverse with theta_min+lambda floor',
        correction='Block BFGS interpolates Href Y; static correction frozen before continuation',
        primary_control='Identical reference images used only for Galerkin warmstart',
        stopping='First original residual<=1e-8 checked every iterate and at initial point',
        excluded='Dense setup/eigenbasis/decoded factors, reference-action formation, all construction workspaces',
        claim='Convergence oracle, not memory-feasible or complete-cost method'),
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__),Path(__file__).with_name('check_selective_precision_oracle.py'),old]},
        independent_checks=checks(),runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        out.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for name in record['protocol']['datasets']:
            stem='concrete-observed' if name=='Concrete' else f'synthetic-n640-{name}'
            path=ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
            with np.load(path) as d:X,b,bw=d['X_train'],d['y_train'],float(d['bandwidth'])
            n=len(b);rank=record['protocol']['rank_by_size'][str(n)]
            retained=n*rank+16*rank+24*n+48
            assert retained<=BUDGET
            K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
            val,U=eigh(K);theta=np.maximum(val[::-1][:rank],0.);Q=U[:,::-1][:,:rank]
            scale=np.max(np.abs(Q),axis=0)/127
            codes=np.rint(Q/scale).clip(-127,127).astype(np.int8)
            Qq=codes.astype(float)*scale
            for lam in record['protocol']['shifts']:
                A=K+lam*np.eye(n);zero=np.zeros(n)
                Ab=A@b;x1=b*(b@b)/(b@Ab);r1=b-A@x1
                Y=np.column_stack([b/np.linalg.norm(b),r1/np.linalg.norm(r1)])
                # b is common immutable input; retain two normalization scalars
                # in addition to r1, S, and the two-by-two correction core.
                base=lambda v:action(Qq,theta,lam,v)
                reference=lambda v:action(Q,theta,lam,v)
                S=reference(Y)
                Hb=corrected(base,Y[:,:1],S[:,:1]);Htwo=corrected(base,Y,S)
                xb,costb=warm(A,b,zero,S[:,:1],b)
                xtwo,costtwo=warm(A,b,x1,S,r1)
                cases=[('quantized_reserved_rank',base,zero,0,0),
                       ('fp64_same_rank_oracle',reference,zero,0,0),
                       ('b_correction',Hb,zero,0,0),
                       ('b_warmstart_quantized',base,xb,0,costb),
                       ('two_action_correction',Htwo,x1,1,0),
                       ('two_action_warmstart_quantized',base,xtwo,1,costtwo),
                       ('two_action_warmstart_corrected',Htwo,xtwo,1,costtwo)]
                rows=[]
                for mode,H,x0,prefix,setup_products in cases:
                    stats=solve(A,b,H,x0)
                    stats['iterations']+=prefix
                    stats['operator_products']+=prefix+setup_products
                    stats['verification_products']+=prefix
                    rows.append(dict(method=mode,prefix_iterations=prefix,
                                     setup_A_products=setup_products,**stats))
                record['runs'].append(dict(dataset=name,regularization=lam,rank=rank,
                    retained_representation_bytes=retained,data_sha256=sha256(path.read_bytes()).hexdigest(),
                    reference_interpolation_error=float(np.linalg.norm(Htwo(Y)-S)/np.linalg.norm(S)),
                    prefix_original_relative_residual=float(np.linalg.norm(r1)/np.linalg.norm(b)),branches=rows,
                    static_controls=[x for x in controls['runs'] if x['dataset']==name and
                                     x['regularization']==lam and x['role']=='matched_factor_bytes']))
                save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
