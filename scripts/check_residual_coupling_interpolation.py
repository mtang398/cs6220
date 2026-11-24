"""One frozen rank-two coupling interpolation; optimistic dense gate."""
import json
import resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cholesky, cho_solve, svd, solve_triangular
from threadpoolctl import threadpool_limits
from check_int16_core_capacity import quantize
from check_schur_inplace_frontier_v2 import ROOT, SharedKernel, Budget


def interpolate(base, truth, target):
    m = base.shape[0]
    a = target[:m]
    v = target[m:]-truth.T@a
    d = (truth-base).T@a
    c = (truth-base)@v
    aa, vv = float(a@a), float(v@v)
    if aa == 0. and vv == 0.:
        U = np.zeros((m, 0)); V = np.zeros((len(v), 0))
    elif aa == 0.:
        U = c[:, None]; V = (v/vv)[:, None]
    elif vv == 0.:
        U = (a/aa)[:, None]; V = d[:, None]
    else:
        U = np.column_stack((a/aa, c-a*float(d@v)/aa))
        V = np.column_stack((d, v/vv))
    delta = U@V.T
    return delta, dict(stored_rank=U.shape[1],
        compatibility_defect=float(abs(a@c-d@v)),
        left_constraint_relative_error=float(np.linalg.norm(delta.T@a-d)/max(np.linalg.norm(d),1e-300)),
        right_constraint_relative_error=float(np.linalg.norm(delta@v-c)/max(np.linalg.norm(c),1e-300)))


def inverse_action(Q, Z, r, lam=1e-5):
    m = len(Q)
    v = (r[m:]-Z.T@r[:m])/lam
    return np.r_[cho_solve((Q,True),r[:m],check_finite=False)-Z@v, v]


def run(A, b, S, Q, Z, x0=None, stop_steps=None):
    n, m = len(b), len(S)
    R = np.setdiff1d(np.arange(n), S)
    C, B, D = A[np.ix_(S,S)], A[np.ix_(S,R)], A[np.ix_(R,R)]
    H = lambda v:cho_solve((Q,True),v,check_finite=False)
    x0 = np.zeros(n) if x0 is None else x0
    residual = b-A@x0
    y = np.zeros(n); r = np.r_[residual[S],residual[R]-Z.T@residual[S]]
    z = np.r_[H(r[:m]),r[m:]/1e-5]; p = z.copy(); rho = float(r@z)
    nb = np.linalg.norm(b); start = process_time(); status = 'maxiter'; actions = 0
    for k in range(2001):
        x = x0.copy(); x[S] += y[:m]-Z@y[m:]; x[R] += y[m:]
        relative = float(np.linalg.norm(b-A@x)/nb)
        if relative <= 1e-8: status='converged'; break
        if stop_steps is not None and k == stop_steps: status='prefix'; break
        if process_time()-start > 4.: status='cpu_cap'; break
        if k == 2000: break
        v=p[m:]; q=p[:m]-Z@v; a=C@q+B@v
        ap=np.r_[a,D@v+B.T@q-Z.T@a]; actions += 1
        den=float(p@ap)
        if den <= 0 or rho <= 0: status='nonpositive_curvature'; break
        alpha=rho/den; y+=alpha*p; r-=alpha*ap
        z=np.r_[H(r[:m]),r[m:]/1e-5]; new=float(r@z); p=z+(new/rho)*p; rho=new
    components=dict(FPS=n*m, factor=m*(m+1)//2, actions=actions*n*n,
                    boundary_cross=2*m*(n-m), verification=n*n)
    return x, dict(status=status, converged=status=='converged', iterations=k,
        original_relative_residual=relative, kernel_action_batches=actions,
        entry_components=components, analytical_entries=sum(components.values()),
        diagnostic_cpu_seconds=process_time()-start)


def tiny():
    rng=np.random.default_rng(622094); F=rng.normal(size=(29,29)); A=F@F.T+np.eye(29)
    m=12; C=A[:m,:m]; B=A[:m,m:]; L=cholesky(C,lower=True)
    Q=L+.03*np.tril(rng.normal(size=L.shape),-1)
    base=cho_solve((Q,True),B); truth=cho_solve((L,True),B)
    general=rng.normal(size=29); a=rng.normal(size=m)
    targets=[general,np.r_[np.zeros(m),rng.normal(size=29-m)],np.r_[a,truth.T@a],np.zeros(29)]
    checks=[]
    for r in targets:
        delta, info=interpolate(base,truth,r); Z=base+delta
        ref=inverse_action(Q,truth,r,.3); actual=inverse_action(Q,Z,r,.3)
        err=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1.))
        T=np.eye(29); T[:m,m:]=-Z
        H=cho_solve((Q,True),np.eye(m)); P=np.zeros((29,29));P[:m,:m]=H;P[m:,m:]=np.eye(29-m)/.3
        M=T@P@T.T; cholesky((M+M.T)*.5,lower=True)
        assert err<1e-11
        checks.append(dict(action_interpolation_relative_error=err,**info))
    return checks


def main():
    out=ROOT/'results/residual-coupling-interpolation.json'
    if out.exists(): raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85)); start,cpu=perf_counter(),process_time()
    controls_path=ROOT/'results/int16-core-capacity.json'; controls=json.loads(controls_path.read_text())
    files=[Path(__file__).resolve(),ROOT/'scripts/check_int16_core_capacity.py',
           ROOT/'scripts/check_schur_inplace_frontier_v2.py',controls_path]
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},
        protocol=dict(payload_budget=721200,rank=2,prefix=4,shift=1e-5,
            core='Maximum m<n with m²+63m−152+16n<=721200, postquantized int16; includes duplicate first8 diagonals and rank2 cache.',
            update='a=rS,d=(Zstar−Z)^T a,v=rR−Zstar^T a,c=(Zstar−Z)v. DeltaZ=a d^T/(a^T a)+(c−a(d^T v)/(a^T a))v^T/(v^T v), with rank1/zero degeneracies. One frozen update, original residual target, exact congruence restart.',
            controls=['unchanged_restart','initial_target_restart','weighted_rank2_upfront','reference_warm_restart','saved_FP32core600'],
            cost='Reference actions, construction/SVD, cache arithmetic and dense decode free; n² entries per fused action. Common prefixsetup charged once; all prefix/restart boundaries and final verifications charged. Warmstart pays one additional original A action. Per-iterate true stopping free; no native/fullmemory/novelty claim.',
            limits='80/85CPU; command120wall serialized flock;4CPU per solve;BLAS1'),tiny_check=None,cases=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    with threadpool_limits(limits=1):
        rec['tiny_check']=tiny();save()
        cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),
               ('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
        for name,filename in cases:
            if process_time()-cpu>66: rec['status']='cpu_budget';save();return
            path=ROOT/'data/prepared'/filename
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            n=len(b);m=max(j for j in range(8,n) if j*j+63*j-152+16*n<=721200)
            kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(m);R=np.setdiff1d(np.arange(n),S)
            A=kernel.block(X,X);A.flat[::n+1]+=1e-5;C=A[np.ix_(S,S)];B=A[np.ix_(S,R)]
            L=cholesky(C,lower=True);Q=quantize(L,'int16');base=cho_solve((Q,True),B);truth=cho_solve((L,True),B)
            prefix,pr=run(A,b,S,Q,base,stop_steps=4);assert pr['status']=='prefix'
            residual=b-A@prefix;perm=np.r_[S,R]
            current,ci=interpolate(base,truth,residual[perm]);initial,ii=interpolate(base,truth,b[perm])
            for target,delta,info in [(residual[perm],current,ci),(b[perm],initial,ii)]:
                ref=inverse_action(Q,truth,target)
                info['full_action_relative_error']=float(np.linalg.norm(inverse_action(Q,base+delta,target)-ref)/np.linalg.norm(ref))
            U,s,V=svd(L.T@(truth-base),full_matrices=False,check_finite=False)
            weighted=solve_triangular(L.T,U[:,:2]*s[:2],lower=False,check_finite=False)@V[:2]
            direction=np.empty(n);direction[perm]=inverse_action(Q,truth,residual[perm])
            Ad=A@direction;den=float(direction@Ad);assert den>0
            warm=prefix+float(residual@direction)/den*direction
            previous=next(c for c in controls['cases'] if c['dataset']==name)
            assert previous['data_sha256']==sha256(path.read_bytes()).hexdigest()
            case=dict(dataset=name,n=n,core=m,payload_bytes=m*m+63*m-152+16*n,
                data_sha256=previous['data_sha256'],prefix=pr,interpolation=dict(current=ci,initial=ii),
                strong_FP32core600=next(r for r in previous['runs'] if r['method']=='float32'),runs=[])
            rec['cases'].append(case);save()
            branches=[('unchanged_restart',base,prefix),('current_target_restart',base+current,prefix),
                      ('initial_target_restart',base+initial,prefix),('weighted_rank2_upfront',base+weighted,None),
                      ('reference_warm_restart',base,warm)]
            for mode,Z,x0 in branches:
                _,row=run(A,b,S,Q,Z,x0=x0)
                if x0 is not None:
                    row['analytical_entries']+=pr['analytical_entries']-(n*m+m*(m+1)//2)
                    row['iterations']+=4;row['kernel_action_batches']+=4
                if mode=='reference_warm_restart':
                    row['analytical_entries']+=n*n;row['warm_original_operator_products']=1
                row['method']=mode;case['runs'].append(row);save()
                print(json.dumps(dict(dataset=name,core=m,**{k:row[k] for k in ['method','status','iterations','analytical_entries','original_relative_residual']})),flush=True)
        rec['status']='complete';save()


if __name__=='__main__':main()
