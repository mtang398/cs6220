"""Exact-eigenbasis oracle for precision versus rank under retained factor bytes.

Construction, decoded FP64 factors and dense operators exceed the allowance.
Only stored representation bytes and verified convergence are compared. This
is not a feasible low-memory implementation, adaptive result or CPU benchmark.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg,LinearOperator
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
FACTOR_BYTES=131072


def quantize(Q,kind):
    if kind=='int8':
        scale=np.max(np.abs(Q),axis=0)/127
        codes=np.rint(Q/scale).clip(-127,127).astype(np.int8)
        return codes.astype(float)*scale, codes.nbytes+scale.nbytes
    stored=Q.astype(kind)
    return stored.astype(float), stored.nbytes


def solve(A,b,Q,theta,lam):
    floor=theta[-1]+lam
    def precondition(v):
        a=Q.T@v
        w=v-Q@a
        return w/floor+Q@(a/(theta+lam)-(Q.T@w)/floor)
    M=LinearOperator(A.shape,matvec=precondition,dtype=np.float64)
    count=0
    saved=None
    residual=None
    class Finished(Exception):pass
    def callback(x):
        nonlocal count,saved,residual
        count+=1
        residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
        if residual<=1e-8:
            saved=x.copy()
            raise Finished
    try:
        x,info=cg(A,b,M=M,rtol=0.,atol=0.,maxiter=2500,callback=callback)
    except Finished:
        x,info=saved,0
    final=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
    return dict(iterations=count,solver_info=int(info),converged=bool(info==0 and final<=1e-8),
                true_relative_residual=final,original_verification_products=count+1,
                operator_products=count,complement_scale=float(floor))


def main():
    output=ROOT/'results/quantized-spectral-headroom.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))
    start,cpu=perf_counter(),process_time()
    record=dict(status='running',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(seed=0,datasets=['smooth','mixed','Concrete'],shifts=[.001,1e-5],
                      factor_byte_allowance=FACTOR_BYTES,
                      representation='Quantized eigenbasis plus FP64 eigenvalues; int8 adds FP64 column scales',
                      inverse='(I-QQ^T)^2/(theta_min+lambda) + Q diag(1/(theta+lambda)) Q^T',
                      controls='FP64 at each compressed rank isolates quantization degradation',
                      stopping='First literal original relative residual<=1e-8',
                      excluded='Dense exact eigendecomposition, decoded factor, setup/workspace, decode/application CPU cost',
                      claim='Static precision/rank oracle only; no adaptive benefit or memory-feasible method'),runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record,indent=2)+'\n')
    with threadpool_limits(limits=1):
        for name in record['protocol']['datasets']:
            stem='concrete-observed' if name=='Concrete' else f'synthetic-n640-{name}'
            path=ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
            K=np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
            values,U=eigh(K);values=np.maximum(values[::-1],0.);U=U[:,::-1]
            schemes=[]
            for kind,size in [('float64',8),('float32',4),('float16',2),('int8',1)]:
                rank=min(len(K),FACTOR_BYTES//(len(K)*size+8+(8 if kind=='int8' else 0)))
                schemes.append((kind,rank,'matched_factor_bytes'))
                if kind!='float64':schemes.append(('float64',rank,'same_rank_precision_oracle'))
            for lam in record['protocol']['shifts']:
                A=K+lam*np.eye(len(K))
                for kind,rank,role in schemes:
                    Q,storage=quantize(U[:,:rank],kind)
                    row=dict(dataset=name,regularization=lam,precision=kind,rank=rank,role=role,
                             retained_bytes=storage+8*rank,data_sha256=sha256(path.read_bytes()).hexdigest(),
                             basis_relative_error=float(np.linalg.norm(Q-U[:,:rank])/np.linalg.norm(U[:,:rank])))
                    if role=='matched_factor_bytes':assert row['retained_bytes']<=FACTOR_BYTES
                    row.update(solve(A,b,Q,values[:rank],lam))
                    record['runs'].append(row);save()
    record['status']='complete';save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
