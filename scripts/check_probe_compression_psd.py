#!/usr/bin/env python3
"""Independent implicit-probe and PSD audit of the root's current prototype.

Load only its State class. Reconstruct Omega solely for this dense diagnostic;
record the exact orthogonal transformations returned inside its update.
"""
import ast
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter,process_time
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from nystrom_project.numerics import GaussianKernel,initialize_rpcholesky
source_path=root/'scripts/check_probe_compression.py'
source=source_path.read_bytes()
output=root/'results/probe-compression-psd-diagnostic.json'
if output.exists():raise FileExistsError(output)
resource.setrlimit(resource.RLIMIT_CPU,(55,56))
started,cpu_started=perf_counter(),process_time()

class CapturedLinalg:
    def __getattr__(self,name):return getattr(linalg,name)
    def null_space(self,*args,**kwargs):
        value=linalg.null_space(*args,**kwargs);self.Q=value.copy();return value
    def svd(self,*args,**kwargs):
        value=linalg.svd(*args,**kwargs);self.Vh=value[2].copy();return value
    def qr(self,*args,**kwargs):
        self.C=args[0].copy();return linalg.qr(*args,**kwargs)

proxy=CapturedLinalg()
namespace=dict(np=np,linalg=proxy)
nodes=[node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='State']
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),namespace)
State=namespace['State']
result=dict(status='running',scope='Dense offline invariants after first20 steps; no performance claims',
    source_path=str(source_path.relative_to(root)),source_sha256=sha256(source).hexdigest(),
    script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    protocol=dict(steps=20,protected=[0,1,4],ranks=[16,32],regularization=[.001,1e-5],blas_threads=1),runs=[])

def inspect(K,state,Omega,protected,iteration):
    F=state.U*state.s
    KOmega=K@Omega
    gram=Omega.T@KOmega
    residual=K-F@F.T
    residual=(residual+residual.T)*.5
    mineig=float(linalg.eigvalsh(residual,subset_by_index=[0,0],check_finite=False)[0])
    errors=[];coord_errors=[]
    for (p,kp),coords in zip(protected,state.coords):
        errors.append(float(np.linalg.norm(F@(F.T@p)-kp)/np.linalg.norm(kp)))
        coord_errors.append(float(np.linalg.norm(F.T@p-coords)/np.linalg.norm(coords)))
    return dict(iteration=iteration,updates=state.updates,skips=state.skips,
        F_equals_KOmega_relative_error=float(np.linalg.norm(F-KOmega)/np.linalg.norm(F)),
        OmegaTKOmega_identity_spectral_error=float(np.linalg.norm(gram-np.eye(len(gram)),ord=2)),
        OmegaTKOmega_identity_max_entry=float(np.max(np.abs(gram-np.eye(len(gram))))),
        minimum_eigenvalue_K_minus_FFt=mineig,
        negative_eigenvalue_magnitude_relative_lambda=max(0,-mineig)/state.lam,
        trace_approximation=float(state.s@state.s),
        protected_secant_max_relative_error=max(errors,default=0),
        protected_coordinate_max_relative_error=max(coord_errors,default=0))

with threadpool_limits(limits=1):
    for name in ('synthetic-n640-smooth','concrete-observed'):
        data=np.load(root/f'data/prepared/{name}-s0-lambda0.001.npz')
        b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']));K=kernel.dense()
        for lam in (.001,1e-5):
            A=K+lam*np.eye(len(b))
            for k in (16,32):
                base=initialize_rpcholesky(kernel,k,lam,np.random.default_rng(np.random.SeedSequence([0,6220,1])))
                # F=U Sigma=K[:,S] W^-1 K[:,S].T U Sigma^-1.
                coefficients=linalg.solve(base.core,(base.columns.T@base.basis)/np.sqrt(base.eigenvalues)[None,:],assume_a='pos')
                initial_Omega=np.zeros((len(b),k));initial_Omega[base.indices,:]=coefficients
                for protect in (0,1,4):
                    state=State(base,protect);Omega=initial_Omega.copy();protected=[]
                    row=dict(dataset=name,regularization=lam,rank=k,protected=protect,
                             initial=inspect(K,state,Omega,protected,0),checks=[],updates=[])
                    x=np.zeros_like(b);r=b.copy();last=None
                    for it in range(1,21):
                        p=state.apply(r);ap=A@p
                        if last:
                            previous,aprevious=last
                            for _ in range(2):
                                beta=(previous@ap)/(previous@aprevious);p-=beta*previous;ap-=beta*aprevious
                        alpha=(r@p)/(p@ap);x+=alpha*p;r-=alpha*ap
                        Kp=ap-lam*p
                        F=state.U*state.s;cross=F.T@p;delta=float(p@Kp-cross@cross)
                        previous_updates=state.updates
                        state.update(p,Kp)
                        accepted=state.updates>previous_updates
                        update=dict(iteration=it,accepted=accepted,residual_pivot=delta,
                            supplied_Kp_relative_error=float(np.linalg.norm(Kp-K@p)/np.linalg.norm(K@p)))
                        if accepted:
                            appended=(p-Omega@cross)/np.sqrt(delta)
                            Omega=np.column_stack((Omega,appended))@proxy.Q@proxy.Vh.T
                            protected=(protected+[(p.copy(),Kp.copy())])[-protect:] if protect else []
                            if protect:
                                singular=linalg.svdvals(proxy.C)
                                update['protected_coordinate_rank']=int(np.count_nonzero(singular>1e-10*np.linalg.norm(proxy.C)))
                                update['protected_coordinate_columns']=proxy.C.shape[1]
                        row['updates'].append(update)
                        if it in (1,5,10,20):row['checks'].append(inspect(K,state,Omega,protected,it))
                        last=(p,ap)
                    row['true_relative_residual_after20']=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
                    result['runs'].append(row)
                    print(json.dumps(dict(dataset=name,lam=lam,k=k,protected=protect,**row['checks'][-1])),flush=True)
result.update(status='complete',elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
assert source_path.read_bytes()==source,'Prototype changed during audit; rerun against frozen source.'
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=result['status'],cpu_seconds=result['elapsed_cpu_seconds'])))
