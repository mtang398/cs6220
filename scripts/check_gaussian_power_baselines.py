#!/usr/bin/env python3
"""Dense two-pass Gaussian Nyström and actual-column QRCP diagnostic.

The fixed history-one FCG loop is loaded from the compression prototype to
match its arithmetic and stopping rule. No timing advantage is inferred.
"""
import ast
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import process_time,perf_counter
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from nystrom_project.numerics import GaussianKernel,NystromPreconditioner
output=root/'results/gaussian-power-sketch-baselines-diagnostic.json'
if output.exists():raise FileExistsError(output)
resource.setrlimit(resource.RLIMIT_CPU,(55,56))
started,cpu_started=perf_counter(),process_time()
source_path=root/'scripts/check_probe_compression.py';source=source_path.read_bytes()
nodes=[node for node in ast.parse(source).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ('State','solve')]
namespace=dict(np=np,linalg=linalg)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),namespace)
State,solve=namespace['State'],namespace['solve']

class GaussianState:
    def __init__(self,U,s,lam):
        self.U=U;self.s=s;self.lam=lam;self.updates=0;self.skips=0
    def apply(self,r):
        c=self.U.T@r;t=r-self.U@c;e=self.U.T@t;t-=self.U@e;c+=e
        return t/self.lam+self.U@(c/(self.s*self.s+self.lam))

comparison_paths={name:root/'results'/name for name in
    ('probe-compression-restart-diagnostic.json','sketch-column-selection-diagnostic.json')}
comparisons={name:json.loads(path.read_text()) for name,path in comparison_paths.items()}
one_pass_path=root/'results/gaussian-sketch-baselines-diagnostic.json'
one_pass=json.loads(one_pass_path.read_text())
result=dict(status='running',scope='One-seed dense offline solver comparison; no CPU or end-to-end performance claims',
    protocol=dict(ranks=[16,32],regularization=[.001,1e-5],seed_sequence=[0,6220,3],
        gaussian_entries='independent N(0,1)',paired_sketch_across_lambda_and_methods=True,
        subspace_iterations=1,setup_kernel_matmat_calls=2,
        construction='Y1=K Omega;Q=qr(Y1);Y2=K Q;W=Q.T Y2',
        retained_Omega=False,recurrence='exact fixed FCG-history1 loop from compression prototype',
        rtol=1e-8,maxiter=2500,blas_threads=1,cpu_limit_seconds=55),
    source_sha256={str(source_path.relative_to(root)):sha256(source).hexdigest(),
        str(Path(__file__).resolve().relative_to(root)):sha256(Path(__file__).read_bytes()).hexdigest(),
        'src/nystrom_project/numerics.py':sha256((root/'src/nystrom_project/numerics.py').read_bytes()).hexdigest()},
    comparison_sources={name:dict(sha256=sha256(path.read_bytes()).hexdigest(),prototype_source_sha256=comparisons[name]['source_sha256'])
                        for name,path in comparison_paths.items()},
    one_pass_source=dict(path=str(one_pass_path.relative_to(root)),sha256=sha256(one_pass_path.read_bytes()).hexdigest()),runs=[])

def save():
    result.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

def reference(name,k,lam,file,limit):
    matches=[row for row in comparisons[file]['runs'] if row['dataset']==name and row['rank']==k
        and row['regularization']==lam and row['limit']==limit and row['protected']==0 and row['interval']==1]
    if not matches:return None
    assert len(matches)==1
    value=matches[0]
    return dict(iterations=value['iterations'],true_relative_residual=value['relative_residual'],
                converged=bool(value['relative_residual']<=1e-8),source=file)

with threadpool_limits(limits=1):
    for name in ('synthetic-n640-smooth','synthetic-n640-random','concrete-observed','airfoil-observed'):
        prepared=root/f'data/prepared/{name}-s0-lambda0.001.npz'
        data=np.load(prepared);b=data['y_train'];kernel=GaussianKernel(data['X_train'],float(data['bandwidth']));K=kernel.dense()
        config=json.loads(str(data['config']))
        for k in (16,32):
            if process_time()-cpu_started>48:
                result['status']='budget_exhausted';save();raise SystemExit(0)
            Omega=np.random.default_rng(np.random.SeedSequence([0,6220,3])).standard_normal((len(b),k))
            Y1=K@Omega;Q,_=linalg.qr(Y1,mode='economic',check_finite=False)
            Y=K@Q;W=Q.T@Y;W=(W+W.T)*.5
            values,V=linalg.eigh(W,check_finite=False)
            threshold=32*np.finfo(float).eps*k*max(values[-1],1e-300)
            keep=values>threshold
            F=(Y@V[:,keep])/np.sqrt(values[keep])
            U,s,_=linalg.svd(F,full_matrices=False,check_finite=False)
            _,_,pivots=linalg.qr(F.T,mode='economic',pivoting=True,check_finite=False)
            indices=pivots[:k]
            residual=K-F@F.T;residual=(residual+residual.T)*.5
            minimum_residual_eigenvalue=float(linalg.eigvalsh(residual,subset_by_index=[0,0],check_finite=False)[0])
            assert minimum_residual_eigenvalue>-1e-9
            sketch=dict(effective_rank=int(len(s)),core_eigenvalue_min=float(values[0]),
                core_eigenvalue_max=float(values[-1]),core_threshold=float(threshold),
                trace_approximation=float(s@s),minimum_eigenvalue_K_minus_FFt=minimum_residual_eigenvalue,
                sketch_sha256=sha256(Omega.tobytes()).hexdigest(),selected_actual_indices=indices.tolist())
            del Omega
            for lam in (.001,1e-5):
                A=K+lam*np.eye(len(b))
                gauss=solve(A,b,GaussianState(U,s,lam),limit=0,interval=1)
                selected=NystromPreconditioner(indices,K[:,indices],lam)
                columns=solve(A,b,State(selected,0),limit=0,interval=1)
                for value in (gauss,columns):
                    value['converged']=bool(value['relative_residual']<=1e-8)
                one_pass_row=next(row for row in one_pass['runs'] if row['dataset']==name
                                  and row['rank']==k and row['regularization']==lam)
                assert sketch['sketch_sha256']==one_pass_row['sketch']['sketch_sha256']
                row=dict(dataset=name,regularization=lam,rank=k,n=len(b),
                    prepared_feature_source=str(prepared.relative_to(root)),prepared_sha256=sha256(prepared.read_bytes()).hexdigest(),
                    dataset_sha256=config['dataset_sha256'],train_indices_sha256=config['train_indices_sha256'],
                    source_preparation_regularization=config['regularization'],
                    source_reference_ignored=True,actual_operator_regularization=lam,
                    sketch=sketch,power_gaussian_nystrom=gauss,power_gaussian_qrcp_actual_columns=columns,
                    one_pass_gaussian_nystrom=one_pass_row['gaussian_nystrom'],
                    one_pass_gaussian_qrcp_actual_columns=one_pass_row['gaussian_qrcp_actual_columns'],
                    fixed_rp=reference(name,k,lam,'sketch-column-selection-diagnostic.json',0),
                    compressed_freeze10=reference(name,k,lam,'probe-compression-restart-diagnostic.json',10))
                result['runs'].append(row);save()
                print(json.dumps(dict(dataset=name,k=k,lam=lam,power_gauss=gauss['iterations'],gauss_converged=gauss['converged'],
                    power_gaussian_qrcp=columns['iterations'],qrcp_converged=columns['converged'],
                    fixed_rp=row['fixed_rp'],compressed_freeze10=row['compressed_freeze10'])),flush=True)
result['status']='complete';save()
assert source_path.read_bytes()==source,'Prototype changed during audit.'
print(json.dumps(dict(status=result['status'],cpu_seconds=result['elapsed_cpu_seconds'])))
