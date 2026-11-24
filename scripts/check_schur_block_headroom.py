"""Dense oracle: can local Schur solves justify spending core storage on blocks?

This deliberately retains dense matrices beyond the working-memory allowance.
Only iterations, original residuals and hypothetical packed factor bytes are
compared. It is not a CPU timing, memory-feasible implementation or adaptive win.
"""
import json
import resource
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def partition(X, indices, size):
    if len(indices) <= size:
        return [indices]
    points = X[indices]
    axis = int(np.argmax(np.ptp(points, axis=0)))
    ordered = indices[np.argsort(points[:, axis], kind='stable')]
    middle = len(ordered)//2
    return partition(X, ordered[:middle], size)+partition(X, ordered[middle:], size)


def selected(K, count):
    result = []
    similarity = np.zeros(len(K))
    j = 0
    for _ in range(count):
        result.append(j)
        np.maximum(similarity, K[:, j], out=similarity)
        similarity[result] = np.inf
        j = int(np.argmin(similarity))
    return np.array(result)


def run(A, X, b, S, block_size):
    R = np.setdiff1d(np.arange(len(A)), S)
    W = cho_factor(A[np.ix_(S, S)], lower=True)
    C = A[np.ix_(R, S)]
    D = A[np.ix_(R, R)]-C@cho_solve(W, C.T)
    D = (D+D.T)*.5
    g = b[R]-C@cho_solve(W, b[S])
    groups = partition(X[R], np.arange(len(R)), block_size) if block_size else []
    factors = [(q, cho_factor(D[np.ix_(q, q)], lower=True)) for q in groups]
    def precondition(v):
        z = np.empty_like(v)
        for q, factor in factors:
            z[q] = cho_solve(factor, v[q])
        return z
    M = LinearOperator(D.shape, matvec=precondition) if factors else None
    count = 0
    def callback(x):
        nonlocal count
        count += 1
    # A stricter reduced tolerance allows independent full-system verification.
    y, info = cg(D, g, M=M, rtol=1e-10, atol=0., maxiter=2500, callback=callback)
    x = np.empty_like(b)
    x[R] = y
    x[S] = cho_solve(W, b[S]-C.T@y)
    rel = float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
    factor_entries = len(S)*(len(S)+1)//2 + sum(len(q)*(len(q)+1)//2 for q in groups)
    return dict(iterations=count, solver_info=int(info), true_relative_residual=rel,
                converged=bool(info == 0 and rel <= 1e-8),
                packed_factor_bytes=8*factor_entries,
                block_sizes=[len(q) for q in groups],
                optimistic_action_kernel_entries=count*(len(A)**2-len(S)**2),
                dense_operator_bytes=A.nbytes+D.nbytes+C.nbytes)


def main():
    output = ROOT/'results/schur-block-headroom.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (110, 115))
    start, cpu = perf_counter(), process_time()
    record = dict(status='running', source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol=dict(seed=0, datasets=['smooth','mixed','Concrete'],
                                shifts=[.001,1e-5], configurations=[[160,0],[64,24],[128,8]],
                                scope='Dense static headroom oracle; no adaptive, memory or runtime claim',
                                reduced_rtol=1e-10, original_rtol=1e-8,
                                selection='FPS starting at index 0',
                                grouping='recursive median split along widest feature'), runs=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,
                      elapsed_cpu_seconds=process_time()-cpu)
        output.write_text(json.dumps(record, indent=2)+'\n')
    with threadpool_limits(limits=1):
        for name in record['protocol']['datasets']:
            stem = 'concrete-observed' if name == 'Concrete' else f'synthetic-n640-{name}'
            path = ROOT/f'data/prepared/{stem}-s0-lambda0.001.npz'
            with np.load(path) as data:
                X, b, bw = data['X_train'], data['y_train'], float(data['bandwidth'])
            K = np.exp(-cdist(X,X,'sqeuclidean')/(2*bw*bw))
            Smax = selected(K, 160)
            for lam in record['protocol']['shifts']:
                A = K+lam*np.eye(len(K))
                for m, block in record['protocol']['configurations']:
                    row = dict(dataset=name, regularization=lam, core_count=m, block_size=block,
                               data_sha256=sha256(path.read_bytes()).hexdigest())
                    row.update(run(A, X, b, Smax[:m], block))
                    record['runs'].append(row)
                    save()
    record['status']='complete'
    save()
    print(json.dumps({k:record[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__ == '__main__':
    main()
