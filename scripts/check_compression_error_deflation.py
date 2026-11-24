"""One-step compression-error acquisition followed by balanced deflation.

Exact reference actions and decoded dense factors are unavailable oracle inputs.
Compare the same-direction warm start with persistent correction, then strong
static/error-probe controls. No native CPU or total-memory feasibility claim.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from check_selective_precision_oracle import action
from check_reference_action_correction import corrected,solve

ROOT=Path(__file__).resolve().parents[1]


def coarse(A,b,x,r,d,base):
    d=d/np.linalg.norm(d)
    y=A@d
    energy=float(d@y)
    assert energy>0
    new=x+d*(d@r)/energy
    H=corrected(base,y[:,None],d[:,None])
    check=float(abs(d@(b-A@new))/(np.linalg.norm(d)*np.linalg.norm(b)))
    # This diagnostic residual is additional verification, charged below.
    return new,H,dict(energy=energy,coarse_orthogonality=check,
                      secant_relative_error=float(np.linalg.norm(H(y)-d)))


def main():
    output=ROOT/'results/compression-error-deflation-oracle.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    oldpath=ROOT/'results/quantized-spectral-headroom.json'
    old=json.loads(oldpath.read_text())
    record=dict(status='running',protocol=dict(seed=0,datasets=['smooth','mixed','Concrete'],
        shifts=[.001,1e-5],checkpoint=1,updates=1,rank_by_size={'512':232,'824':140},
        retained_byte_allowance=131072,direction='(Href-Hq)r after one quantized-PCG step',
        primary_control='Identical Galerkin correction, then Hq versus balanced Hd',
        stopping='First original residual<=1e-8, checked at starting point and every iterate',
        reference_actions='Count separately; construction/reconstruction cost excluded, not free in production',
        static_top_control='Two exact K eigenvectors; A-images obtained by free exact eigenvalue identity',
        exclusions='Dense eigendecomposition, dense A, decoded factor, reference actions, setup/application workspace',
        claim='Oracle convergence discriminator only, not native efficiency or total-memory feasibility'),
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),Path(__file__).with_name('check_reference_action_correction.py'),
                  Path(__file__).with_name('check_selective_precision_oracle.py'),oldpath]},runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for name in record['protocol']['datasets']:
            stem='concrete-observed' if name=='Concrete' else f'synthetic-n640-{name}'
            path=ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            n=len(b);rank=record['protocol']['rank_by_size'][str(n)]
            retained=n*rank+16*rank+16*n+8
            assert retained<=131072
            K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
            val,U=eigh(K);theta=np.maximum(val[::-1][:rank],0.);Q=U[:,::-1][:,:rank]
            scale=np.max(np.abs(Q),axis=0)/127
            codes=np.rint(Q/scale).clip(-127,127).astype(np.int8);Qq=codes.astype(float)*scale
            rng=np.random.default_rng(622123);g=rng.normal(size=n);g/=np.linalg.norm(g)
            for lam in record['protocol']['shifts']:
                A=K+lam*np.eye(n);zero=np.zeros(n)
                base=lambda v:action(Qq,theta,lam,v)
                ref=lambda v:action(Q,theta,lam,v)
                z=base(b);Az=A@z;x1=z*(b@z)/(z@Az);r1=b-A@x1
                zref=ref(r1);d=zref-base(r1)
                xc,Hc,checkc=coarse(A,b,x1,r1,d,base)
                xr,Hr,checkr=coarse(A,b,x1,r1,zref,base)
                xg,Hg,checkg=coarse(A,b,zero,b,ref(g)-base(g),base)
                xb,Hb,checkb=coarse(A,b,zero,b,ref(b)-base(b),base)
                xref,Hunused,checkref=coarse(A,b,zero,b,ref(b),base)
                top=Q[:,:2];topvalues=theta[:2]+lam
                def Htop(v):
                    w=v-top@(top.T@v)
                    z=base(w)
                    return z-top@(top.T@z)+top@((top.T@v)/topvalues)
                xtop=top@((top.T@b)/topvalues)
                # mode,H,start,prefix,coarse A-products,reference applications,
                # separately charged coarse-orthogonality verification products.
                cases=[('quantized_uninterrupted',base,zero,0,0,0,0),
                       ('quantized_restart1',base,x1,1,0,0,0),
                       ('current_error_warmstart',base,xc,1,1,1,1),
                       ('current_error_balanced',Hc,xc,1,1,1,1),
                       ('reference_step_warmstart',base,xr,1,1,1,1),
                       ('reference_step_balanced',Hr,xr,1,1,1,1),
                       ('random_error_balanced_upfront',Hg,xg,0,1,1,1),
                       ('b_error_balanced_upfront',Hb,xb,0,1,1,1),
                       ('top2_balanced_upfront',Htop,xtop,0,0,0,0),
                       ('reference_b_warmstart',base,xref,0,1,1,1),
                       ('fp64_same_rank_oracle',ref,zero,0,0,None,0)]
                rows=[]
                for mode,H,initial,prefix,setup,refs,extra_checks in cases:
                    stats=solve(A,b,H,initial)
                    stats['continuation_iterations']=stats['iterations']
                    stats['iterations']+=prefix
                    stats['operator_products']+=prefix+setup
                    stats['verification_products']+=prefix+extra_checks
                    rows.append(dict(method=mode,prefix_iterations=prefix,setup_A_products=setup,
                        additional_reference_actions=refs,**stats))
                record['runs'].append(dict(dataset=name,regularization=lam,rank=rank,
                    retained_representation_bytes=retained,top2_retained_representation_bytes=retained+8,
                    data_sha256=sha256(path.read_bytes()).hexdigest(),
                    direction_relative_size=float(np.linalg.norm(d)/np.linalg.norm(zref)),
                    direction_checks=dict(current=checkc,reference=checkr,random=checkg,b=checkb,reference_b=checkref),
                    branches=rows,static_controls=[v for v in old['runs'] if v['dataset']==name
                        and v['regularization']==lam and v['role']=='matched_factor_bytes']))
                save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
