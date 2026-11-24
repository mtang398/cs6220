"""C-energy minimum-norm coupling interpolation; follow-up to Euclidean gate."""
import json
import resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from scipy.linalg import cholesky, cho_solve
from threadpoolctl import threadpool_limits
import check_residual_coupling_interpolation as previous
from check_schur_inplace_frontier_v2 import ROOT, SharedKernel, Budget


def interpolate(base, truth, target, L):
    m=base.shape[0];a=target[:m];v=target[m:]-truth.T@a
    d=(truth-base).T@a;c=(truth-base)@v
    vv=float(v@v);u=cho_solve((L,True),a,check_finite=False);den=float(a@u)
    if np.any(a):
        assert den>0
        u/=den
        U=np.column_stack((u,c-u*float(d@v))) if vv else u[:,None]
        V=np.column_stack((d,v/vv)) if vv else d[:,None]
    elif vv:
        U=c[:,None];V=(v/vv)[:,None]
    else:
        U=np.zeros((m,0));V=np.zeros((len(v),0))
    delta=U@V.T
    return delta,dict(stored_rank=U.shape[1],compatibility_defect=float(abs(a@c-d@v)),
        left_constraint_relative_error=float(np.linalg.norm(delta.T@a-d)/max(np.linalg.norm(d),1e-300)),
        right_constraint_relative_error=float(np.linalg.norm(delta@v-c)/max(np.linalg.norm(c),1e-300)),
        correction_C_energy_frobenius=float(np.linalg.norm(L.T@delta)))


def tiny():
    rng=np.random.default_rng(622094);F=rng.normal(size=(29,29));A=F@F.T+np.eye(29)
    m=12;L=cholesky(A[:m,:m],lower=True);Q=L+.03*np.tril(rng.normal(size=L.shape),-1)
    B=A[:m,m:];base=cho_solve((Q,True),B);truth=cho_solve((L,True),B)
    a=rng.normal(size=m);targets=[rng.normal(size=29),np.r_[np.zeros(m),rng.normal(size=29-m)],np.r_[a,truth.T@a],np.zeros(29)]
    rows=[]
    for target in targets:
        delta,info=interpolate(base,truth,target,L)
        ref=previous.inverse_action(Q,truth,target,.3)
        error=float(np.linalg.norm(previous.inverse_action(Q,base+delta,target,.3)-ref)/max(np.linalg.norm(ref),1.))
        euclidean,_=previous.interpolate(base,truth,target)
        energy=float(np.linalg.norm(L.T@delta));other=float(np.linalg.norm(L.T@euclidean))
        assert error<1e-11 and energy<=other*(1+1e-10)+1e-14
        T=np.eye(29);T[:m,m:]=-(base+delta);P=np.zeros((29,29))
        P[:m,:m]=cho_solve((Q,True),np.eye(m));P[m:,m:]=np.eye(29-m)/.3
        M=T@P@T.T;cholesky((M+M.T)*.5,lower=True)
        rows.append(dict(action_relative_error=error,euclidean_interpolant_C_energy=other,**info))
    return rows


def main():
    out=ROOT/'results/residual-coupling-interpolation-weighted.json'
    if out.exists():raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_CPU,(80,85));start,cpu=perf_counter(),process_time()
    oldpath=ROOT/'results/residual-coupling-interpolation.json';old=json.loads(oldpath.read_text())
    assert old['status']=='complete'
    verified={p:sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old['sources'].items()}
    assert all(verified.values())
    files=[Path(__file__).resolve(),Path(previous.__file__).resolve(),oldpath,
           ROOT/'scripts/check_int16_core_capacity.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    rec=dict(status='running',sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},
        old_executed_source_hash_checks=verified,
        protocol=dict(metric='Minimum ||L^T DeltaZ||F with both secant constraints. u=C^-1 a/(a^T C^-1 a); DeltaZ=u d^T+(c-u(d^T v))v^T/(v^T v).',
            tested_metrics='Euclidean interpolation already executed before the metric revision arrived; this separate run tests C-energy geometry. Two metrics tested, not a single predeclared metric.',
            core_budget='Identical maximum m and factor+rank2 payload to prior721200byte gate.',
            provenance='Deterministic replay of old unchanged prefix/factor code with verified old executed source/data hashes. Prefix iterations/status/original residual must match exactly. Newly recorded factor/prefix hashes are replay hashes, not historically recorded hash matches.',
            controls='Reuse unchangedrestart, weightedSVDupfront, referencewarmrestart and strongFP32core600 from prior result after replay checks. No control reruns.',
            costs='Same optimistic kernel counts and free reference/construction assumptions as prior. One shared prefixsetup; both boundaries and verifications charged. No native/fullmemory/current-feedback success claim.',
            limits='80/85CPU; command120wall serialized flock;4CPU per solve;BLAS1;8weighted rows'),tiny_check=None,cases=[])
    def save():
        rec.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    with threadpool_limits(limits=1):
        rec['tiny_check']=tiny();save()
        names=['synthetic-n2560-smooth-s0-lambda0.001.npz','synthetic-n2560-mixed-s0-lambda0.001.npz','concrete-observed-s0-lambda1e-05.npz','airfoil-observed-s0-lambda1e-05.npz']
        for oldcase,filename in zip(old['cases'],names):
            path=ROOT/'data/prepared'/filename;assert sha256(path.read_bytes()).hexdigest()==oldcase['data_sha256']
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            n=len(b);m=max(j for j in range(8,n) if j*j+63*j-152+16*n<=721200);assert m==oldcase['core']
            kernel=SharedKernel(X,bw,Budget([X,b]));S=kernel.farthest(m);R=np.setdiff1d(np.arange(n),S)
            A=kernel.block(X,X);A.flat[::n+1]+=1e-5;C=A[np.ix_(S,S)];B=A[np.ix_(S,R)]
            L=cholesky(C,lower=True);Q=previous.quantize(L,'int16');base=cho_solve((Q,True),B);truth=cho_solve((L,True),B)
            prefix,pr=previous.run(A,b,S,Q,base,stop_steps=4)
            matches={key:pr[key]==oldcase['prefix'][key] for key in ['status','iterations','original_relative_residual','kernel_action_batches','analytical_entries']};assert all(matches.values())
            case=dict(dataset=oldcase['dataset'],core=m,payload_bytes=oldcase['payload_bytes'],data_sha256=oldcase['data_sha256'],
                prefix_replay_matches=matches,replayed_prefix_sha256=sha256(prefix.tobytes()).hexdigest(),
                replayed_factor_sha256=sha256(Q.tobytes()).hexdigest(),replayed_selected_sha256=sha256(S.tobytes()).hexdigest(),
                prefix=pr,reused_controls=[r for r in oldcase['runs'] if r['method'] in ['unchanged_restart','weighted_rank2_upfront','reference_warm_restart']],
                strong_FP32core600=oldcase['strong_FP32core600'],runs=[])
            rec['cases'].append(case);save();residual=b-A@prefix;perm=np.r_[S,R]
            for mode,target in [('current_target_restart',residual[perm]),('initial_target_restart',b[perm])]:
                delta,info=interpolate(base,truth,target,L);reference=previous.inverse_action(Q,truth,target)
                info['full_action_relative_error']=float(np.linalg.norm(previous.inverse_action(Q,base+delta,target)-reference)/np.linalg.norm(reference))
                _,row=previous.run(A,b,S,Q,base+delta,x0=prefix)
                row['analytical_entries']+=pr['analytical_entries']-(n*m+m*(m+1)//2)
                row['iterations']+=4;row['kernel_action_batches']+=4
                row.update(method=mode,interpolation=info);case['runs'].append(row);save()
                print(json.dumps(dict(dataset=oldcase['dataset'],**{k:row[k] for k in ['method','status','iterations','analytical_entries','original_relative_residual']})),flush=True)
        rec['status']='complete';save()


if __name__=='__main__':main()
