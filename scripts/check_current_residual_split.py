"""Fresh spatial splitting after preconditioning; free-history capacity gate."""
import json
import resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cholesky, cho_solve, eigh
from threadpoolctl import threadpool_limits
import check_fused_congruence_core as fused
from check_schur_inplace_frontier_v2 import ROOT, SharedKernel, Budget


def run(A, b, S, L, mask, mode, cap=8., lam=1e-5):
    H = lambda v: cho_solve((L, True), v, check_finite=False)
    action, precondition, original, rhs = fused.setup(A, b, S, H, lam)
    n = len(b)
    y = np.zeros(n)
    r = rhs.copy()
    W = np.empty((n, 0)); AW = np.empty((n, 0))
    last = last_image = None
    p = None; rho = None
    normb = np.linalg.norm(b)
    start = process_time()
    batches = rhs_work = precondition_rhs = rankdrops = 0
    history = []
    status = 'maxiter'
    for step in range(2001):
        x = original(y)
        relative = float(np.linalg.norm(b-A@x)/normb)
        if relative <= 1e-8:
            status = 'converged'; break
        if process_time()-start > cap:
            status = 'cpu_cap'; break
        if step == 2000: break
        if mode == 'scalar_pcg':
            z = precondition(r); precondition_rhs += 1
            newrho = float(r@z)
            p = z.copy() if p is None else z+(newrho/rho)*p
            rho = newrho
            ap = action(p); batches += 1; rhs_work += 1
            den = float(p@ap)
            if den <= 0 or rho <= 0:
                status = 'nonpositive_curvature'; break
            alpha = rho/den
            y += alpha*p; r -= alpha*ap
            history.append(dict(rank=1, input_width=1))
            continue
        if mode == 'initial_split_full_history' and last is not None:
            V = precondition(last_image)
            precondition_rhs += V.shape[1]
        else:
            z = precondition(r); precondition_rhs += 1
            V = z[:, None] if mode == 'scalar_full_history' else np.column_stack((mask*z, (~mask)*z))
        norms = np.linalg.norm(V, axis=0)
        good = norms > 0
        V = V[:, good]/norms[good]
        if not V.shape[1]:
            status = 'zero_candidate'; break
        AV = action(V); batches += 1; rhs_work += V.shape[1]
        input_width = V.shape[1]
        input_energy = float(np.linalg.norm(V.T@AV, 2))
        # Two-pass full G-orthogonalization; stored G images need no new action.
        for _ in range(2):
            coefficients = AW.T@V
            V -= W@coefficients; AV -= AW@coefficients
        gram = V.T@AV; gram = (gram+gram.T)*.5
        values, vectors = eigh(gram)
        threshold = max(1e-12*max(float(values[-1]), 0.),
                        64*np.finfo(float).eps*input_energy)
        keep = values > threshold
        if not np.any(keep):
            status = 'rank_breakdown'; break
        rankdrops += input_width-int(np.sum(keep))
        transform = vectors[:, keep]/np.sqrt(values[keep])
        last, last_image = V@transform, AV@transform
        alpha = last.T@r
        y += last@alpha; r -= last_image@alpha
        W = np.column_stack((W, last)); AW = np.column_stack((AW, last_image))
        history.append(dict(rank=last.shape[1], input_width=input_width,
                            input_energy=input_energy, rank_threshold=threshold))
    orth = float(np.linalg.norm(W.T@AW-np.eye(W.shape[1]))) if W.shape[1] else None
    m = len(S)
    components = dict(FPS=n*m, factor=m*(m+1)//2, operator_batches=batches*n*n,
                      boundary_cross=2*m*(n-m), final_verification=n*n)
    return dict(method=mode, status=status, converged=status=='converged',
                iterations=len(history), original_relative_residual=relative,
                kernel_batches=batches, operator_scalar_rhs=rhs_work,
                precondition_scalar_rhs=precondition_rhs, rank_drops=rankdrops,
                retained_history_bytes=W.nbytes+AW.nbytes,
                history_G_orthogonality=orth, history=history,
                analytical_kernel_entries=sum(components.values()), entry_components=components,
                diagnostic_cpu_seconds=process_time()-start)


def main():
    out = ROOT/'results/current-residual-split.json'
    if out.exists(): raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    start, cpu = perf_counter(), process_time()
    sources = [Path(__file__).resolve(), Path(fused.__file__).resolve(),
               ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    modes = ['scalar_pcg', 'scalar_full_history', 'initial_split_full_history', 'fresh_split_full_history']
    rec = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(core=600, shift=1e-5, width=2, modes=modes,
            factor='Exact dense Cholesky postquantized uniformly to FP32, decoded to FP64 for oracle application; identical factor in all branches.',
            split='Fixed longest-feature-axis stable median partition of original rows; reorder mask by [S,R]. Fresh candidates [D z,(I-D)z], z=diag(H,lambda^-1 I) r in fixed congruence coordinates. Split AFTER preconditioning.',
            recurrence='Two-pass full G-orthogonalization and local Galerkin minimization. Retain Gram eigenvalues above max(1e-12 residual Gram norm,64*eps*preprojection energy). Initial-split control advances last block by M G using its already stored G image. Scalar-full-history isolates orthogonalization benefits.',
            reference='Moufawad, Flexibly Enlarged Conjugate Gradient Methods, arXiv:2305.19013v2 Section2.2 Algorithm2 https://arxiv.org/html/2305.19013v2 . Inspired by current-residual splitting; not a verbatim preconditioned MSDO-CG reproduction or novelty claim.',
            cost='One n^2 kernel traversal per action block, irrespective of width, is optimistic. Scalar RHS work separately counted. All history construction/storage, dense factor/kernel setup and per-step original stopping checks free except counted common setup/boundaries/final verification. No native timing or1MiB feasibility claim.',
            limits='80/85 CPU; command120wall serialized flock;8CPU per solve; original relative1e-8;2000steps. No width or threshold sweep.'),
        tiny_check=None, cases=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start, elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp'); tmp.write_text(json.dumps(rec, indent=2, allow_nan=False)+'\n'); tmp.replace(out)
    save()
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(622093); F=rng.normal(size=(31,31)); A=F@F.T+np.eye(31); b=rng.normal(size=31); S=np.arange(9)
        L=cholesky(A[:9,:9],lower=True).astype(np.float32).astype(float); mask=np.arange(31)<15
        tiny=[run(A,b,S,L,mask,mode,lam=1.) for mode in modes]
        assert all(r['converged'] for r in tiny)
        assert max(r['history_G_orthogonality'] or 0. for r in tiny)<1e-7
        H=lambda v:cho_solve((L,True),v,check_finite=False)
        action,precondition,_,rhs=fused.setup(A,b,S,H,1.)
        z=precondition(rhs);V0=np.column_stack((mask*z,(~mask)*z));AV0=action(V0)
        alpha=np.linalg.solve(V0.T@AV0,V0.T@rhs);z1=precondition(rhs-AV0@alpha)
        fresh=np.column_stack((mask*z1,(~mask)*z1));initial=np.column_stack((V0,precondition(AV0)))
        basis=np.linalg.qr(initial,mode='reduced')[0]
        distinction=float(np.linalg.norm(fresh-basis@(basis.T@fresh))/np.linalg.norm(fresh))
        assert distinction>1e-8
        rec['tiny_check']=dict(lam=1.,runs=tiny,fresh_outside_initial_two_block_space_fraction=distinction); save()
        cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),
               ('Concrete','concrete-observed-s0-lambda1e-05.npz'),('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
        for name,filename in cases:
            if process_time()-cpu>65:
                rec['status']='cpu_budget'; save(); return
            path=ROOT/'data/prepared'/filename
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            kernel=SharedKernel(X,bw,Budget([X,b])); S=kernel.farthest(600); R=np.setdiff1d(np.arange(len(b)),S)
            A=kernel.block(X,X); A.flat[::len(b)+1]+=1e-5
            L=cholesky(A[np.ix_(S,S)],lower=True).astype(np.float32).astype(float)
            axis=int(np.argmax(np.ptp(X,axis=0))); order=np.argsort(X[:,axis],kind='stable')
            mask=np.zeros(len(b),dtype=bool); mask[order[:len(b)//2]]=True; mask=mask[np.r_[S,R]]
            case=dict(dataset=name,n=len(b),data_sha256=sha256(path.read_bytes()).hexdigest(),partition_axis=axis,runs=[])
            rec['cases'].append(case)
            for mode in modes:
                row=run(A,b,S,L,mask,mode);case['runs'].append(row);save()
                print(json.dumps(dict(dataset=name,**{k:row[k] for k in ['method','status','iterations','kernel_batches','operator_scalar_rhs','original_relative_residual']})),flush=True)
        rec['status']='complete'; save()


if __name__ == '__main__': main()
