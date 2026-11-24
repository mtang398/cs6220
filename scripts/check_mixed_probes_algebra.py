#!/usr/bin/env python3
"""Read-only audit of the archived unforced mixed-probe prototype."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import process_time, perf_counter
import zipfile
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
output=root/'results/mixed-probes-algebra-diagnostic.json'
if output.exists():raise FileExistsError(output)
resource.setrlimit(resource.RLIMIT_CPU,(55,56))
started,cpu_started=perf_counter(),process_time()
archive=root/'results/mixed-probes-source.zip'
with zipfile.ZipFile(archive) as z:source=z.read('check_mixed_probes.py')
recorded=json.loads((root/'results/mixed-probes-diagnostic.json').read_text())
assert sha256(source).hexdigest()==recorded['source_sha256']
nodes=[node for node in ast.parse(source).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ('Model','solve')]
namespace=dict(np=np,linalg=linalg)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(archive),'exec'),namespace)
Model=namespace['Model']
result=dict(status='running',scope='Archived unforced source; dense offline algebra audit, no performance claims',
    archived_source_sha256=sha256(source).hexdigest(),archive_sha256=sha256(archive.read_bytes()).hexdigest(),
    script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),runs=[])

def kh(model,v):
    if v.ndim==1:return model.U@(model.s*(model.U.T@v))
    return model.U@(model.s[:,None]*(model.U.T@v))

def augmented(model,p,kp,it):
    v=p@kp;p=p/np.sqrt(v);kp=kp/np.sqrt(v)
    cross=model.Y.T@p
    Y=np.column_stack((model.Y,kp))
    W=np.block([[model.W,cross[:,None]],[cross[None,:],np.ones((1,1))]])
    return Y,W,p

def audit_update(old,new,p,kp,mode,it,Omega,K):
    Y,W,probe=augmented(old,p,kp,it)
    inverse=linalg.cho_solve(linalg.cho_factor(W,lower=True),np.eye(len(W)))
    G=Y.T@Y
    if mode=='trace':
        V=inverse@G
        fast=np.trace(V)-np.diag(V@inverse)/np.diag(inverse)
    else:
        H=W+G/old.lam
        hi=linalg.cho_solve(linalg.cho_factor(H,lower=True),np.eye(len(W)))
        fast=np.linalg.slogdet(H)[1]-np.linalg.slogdet(W)[1]+np.log(np.diag(hi))-np.log(np.diag(inverse))
    direct=[]
    for remove in range(len(W)):
        keep=np.arange(len(W))!=remove
        direct.append(Model(Y[:,keep],W[np.ix_(keep,keep)],old.lam,list(range(len(W)-1))).score(mode))
    direct=np.asarray(direct)
    expanded=Model(Y,W,old.lam,old.labels+[it])
    delta=kp-kh(old,p);denominator=float(p@delta)
    probes=np.random.default_rng(84).standard_normal((len(p),3))
    change=kh(expanded,probes)-kh(old,probes)
    sr1=delta[:,None]*(delta@probes)[None,:]/denominator
    removed=next(label for label in old.labels if label not in new.labels)
    remove_index=old.labels.index(removed)
    keep=np.arange(len(W))!=remove_index
    next_Omega=np.column_stack((Omega,probe))[:,keep]
    actual_Y=K@next_Omega
    actual_W=next_Omega.T@actual_Y
    secant_errors=np.linalg.norm(kh(new,next_Omega)-actual_Y,axis=0)/np.linalg.norm(actual_Y,axis=0)
    old_probe=Omega[:,remove_index]
    return next_Omega,dict(iteration=it,removed_label=int(removed),removed_kind='learned' if removed>0 else 'coordinate',
        score_before=old.score(mode),score_after=new.score(mode),
        maximum_all_removal_score_absolute_error=float(np.max(np.abs(fast-direct))),
        fast_and_direct_same_best=bool(np.argmax(fast)==np.argmax(direct)),
        chosen_direct_score_gap=float(direct.max()-direct[remove_index]),
        positive_sr1_denominator=denominator,
        growth_sr1_action_relative_error=float(np.linalg.norm(change-sr1)/np.linalg.norm(sr1)),
        growth_new_secant_relative_error=float(np.linalg.norm(kh(expanded,p)-kp)/np.linalg.norm(kp)),
        retained_probe_secant_max_relative_error=float(secant_errors.max()),
        Y_equals_KOmega_relative_error=float(np.linalg.norm(new.Y-actual_Y)/np.linalg.norm(actual_Y)),
        W_equals_OmegaTKOmega_relative_error=float(np.linalg.norm(new.W-actual_W)/np.linalg.norm(actual_W)),
        removed_probe_secant_error_after=float(np.linalg.norm(kh(new,old_probe)-K@old_probe)/np.linalg.norm(K@old_probe)))

with threadpool_limits(limits=1):
    data=np.load(root/'data/prepared/synthetic-n640-smooth-s0-lambda0.001.npz')
    b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']))
    K=kernel.dense();A=K+.001*np.eye(len(b))
    for k in (16,32):
        base=initialize_rpcholesky(kernel,k,.001,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
        for mode in ('trace','logdet'):
            M=Model(base.columns,base.core,.001,[-int(j)-1 for j in base.indices])
            Omega=np.eye(len(b))[:,base.indices]
            x=np.zeros_like(b);r=b.copy();prev=None;events=[];previous_accepted=False;next_conjugacy=[]
            initial_score=M.score(mode)
            for it in range(1,2001):
                z=M.apply(r);az=A@z;p=z.copy();ap=az.copy()
                if prev is not None:
                    p0,ap0=prev
                    if previous_accepted:
                        next_conjugacy.append(float(abs(p0@az)/np.sqrt((p0@ap0)*(z@az))))
                    for _ in range(2):
                        beta=(p0@ap)/(p0@ap0);p-=beta*p0;ap-=beta*ap0
                curv=p@ap;num=r@p
                assert curv>0 and num>0
                step=num/curv;x+=step*p;r-=step*ap
                if np.linalg.norm(r)<=1e-8*np.linalg.norm(b):
                    r=b-A@x
                    if np.linalg.norm(r)<=1e-8*np.linalg.norm(b):break
                    prev=None;continue
                kp=ap-M.lam*p
                new,accepted=M.update(p,kp,mode,it)
                if accepted:
                    Omega,event=audit_update(M,new,p,kp,mode,it,Omega,K)
                    events.append(event)
                M=new;prev=(p,ap);previous_accepted=accepted
            original=next(row for row in recorded['runs'] if row['dataset']=='synthetic-n640-smooth' and row['rank']==k and row['mode']==mode and row['interval']==1 and row['recurrence']=='fcg')
            assert it==original['iterations'] and len(events)==original['updates']
            result['runs'].append(dict(rank=k,mode=mode,iterations=it,updates=len(events),
                learned_evictions=sum(e['removed_kind']=='learned' for e in events),initial_score=initial_score,final_score=M.score(mode),
                true_relative_residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),
                max_next_step_unprojected_A_conjugacy_after_acceptance=max(next_conjugacy),
                matches_archived_run=True,events=events))
    # Unevicted positive SR1 is an equivalence/correctness control, not evidence
    # of an improved solver: compare its exact-line-search iterates with PCG.
    X=np.random.default_rng(91).standard_normal((64,5));kernel=GaussianKernel(X,1.6)
    K=kernel.dense();A=K+.05*np.eye(64);b=np.random.default_rng(92).normal(size=64)
    base=initialize_rpcholesky(kernel,4,.05,np.random.default_rng(93))
    initial=Model(base.columns,base.core,.05,[-int(j)-1 for j in base.indices])
    pcg_x=np.zeros(64);r=b.copy();z=initial.apply(r);p=z.copy();rho=r@z;pcg_path=[]
    for it in range(12):
        ap=A@p;alpha=rho/(p@ap);pcg_x+=alpha*p;r-=alpha*ap
        pcg_path.append(pcg_x.copy());z=initial.apply(r);next_rho=r@z;p=z+(next_rho/rho)*p;rho=next_rho
    sr1=initial;x=np.zeros(64);r=b.copy();differences=[]
    for it in range(12):
        p=sr1.apply(r);ap=A@p;alpha=(r@p)/(p@ap);x+=alpha*p;r-=alpha*ap
        differences.append(float(np.linalg.norm(x-pcg_path[it])/np.linalg.norm(pcg_path[it])))
        Y,W,_=augmented(sr1,p,ap-.05*p,it+1);sr1=Model(Y,W,.05,sr1.labels+[it+1])
    result['unevicted_sr1_vs_fixed_pcg']=dict(steps=12,relative_iterate_differences=differences,
        maximum_relative_iterate_difference=max(differences))
result.update(status='complete',elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({**{f"k{v['rank']}_{v['mode']}":{k:w for k,w in v.items() if k!='events'} for v in result['runs']},
                  'sr1_control':result['unevicted_sr1_vs_fixed_pcg'],'cpu_seconds':result['elapsed_cpu_seconds']},indent=2))
