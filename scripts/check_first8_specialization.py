"""Bounded FIRST8 primitive compatibility/cost gate; no solver benchmark."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter,process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from check_core_precision_allocation import Helper,MixedFactor,pointer
from check_native_first8_schur import StreamedFirst8
from check_schur_inplace_frontier_v2 import SharedKernel,Budget,ROOT


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def bits_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def compare_factors(a,b,rhs):
    fields=['high','offsets','low','wide','epsilon','pivots']
    checks={name:bits_equal(getattr(a,name),getattr(b,name)) for name in fields}
    checks['solve']=bits_equal(a.solve(rhs),b.solve(rhs))
    assert all(checks.values()),checks
    return checks


def main():
    output=ROOT/'results/first8-specialization.json'
    source=ROOT/'scripts/mixed_column_factor_first8.c'
    library=ROOT/'results/mixed_column_factor_first8.so'
    if output.exists() or library.exists():raise FileExistsError('Preserve existing microgate outputs')
    resource.setrlimit(resource.RLIMIT_CPU,(9,10))
    start,cpu=perf_counter(),process_time()
    dependencies=[Path(__file__),source,ROOT/'scripts/mixed_column_factor.c',
        ROOT/'results/mixed_column_factor.so',ROOT/'scripts/check_core_precision_allocation.py',
        ROOT/'scripts/check_native_first8_schur.py',ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in dependencies}
    command=['gcc','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(library),'-lm']
    build=perf_counter();subprocess.run(command,check=True,capture_output=True)
    record=dict(status='running',protocol=dict(m=592,regularizations=[.001,1e-5],repetitions=5,
        method_order='Cyclic generic/specialized across five repetitions and the two shifts',
        scope='Primitive factor/solve timing using an already formed dense core. Kernel construction/FPS and dense oracle storage are excluded from these medians; no full-solver or fixed-memory claim.',
        arithmetic='Identical FP64 operation order, FIRST8 FP64 stores and remaining FP32 stores; FP contraction disabled; disjoint buffers required.',
        ABI='Original mixed_column/mixed_solve signatures. FIRST8 layout validated on first column and every solve. No metadata loads in numerical inner loops.',
        cpu_limit_seconds=10,external_wall_limit_seconds=20,blas_threads=1,lock='/tmp/cs6220-project-benchmark.lock'),
        sources=hashes,compile_command=command,compile_seconds=perf_counter()-build,
        compiler_version=subprocess.run(['gcc','--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0],
        specialized_library_sha256=digest(library),tiny_checks=[],cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu,
            child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    generic=Helper(ROOT/'results/mixed_column_factor.so');special=Helper(library)
    validate=special.library.first8_validate
    validate.argtypes=[ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p];validate.restype=ctypes.c_int
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(622137)
        for m in [7,17]:
            matrix=rng.normal(size=(m,m));C=matrix@matrix.T+np.eye(m);rhs=rng.normal(size=m)
            a=MixedFactor(C,list(range(min(8,m))),generic)
            b=MixedFactor(C,list(range(min(8,m))),special)
            checks=compare_factors(a,b,rhs)
            L=b.dense_L();reference=linalg.solve_triangular(L,rhs,lower=True)
            reference=linalg.solve_triangular(L.T,reference,lower=False)
            error=float(np.linalg.norm(b.solve(rhs)-reference)/np.linalg.norm(reference))
            assert error<1e-12
            assert validate(m,pointer(b.high),pointer(b.offsets))==0
            bad_high=b.high.copy();bad_high[0]=0
            bad_offsets=b.offsets.copy();bad_offsets[1]+=1
            assert validate(m,pointer(bad_high),pointer(b.offsets))==-1
            assert validate(m,pointer(b.high),pointer(bad_offsets))==-2
            probe=rhs.copy()
            assert special.solve(m,pointer(bad_high),pointer(b.offsets),pointer(b.low),pointer(b.wide),pointer(probe))==-1
            assert bits_equal(probe,rhs)
            record['tiny_checks'].append(dict(m=m,bitwise_checks=checks,
                independent_triangular_relative_error=error,invalid_flags_and_offsets_rejected=True))
        save()
        path=ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz'
        with np.load(path) as data:X,bdata,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
        m=592;kernel=SharedKernel(X,bw,Budget([X,bdata]));selected=kernel.farthest(m)
        points=X[selected];Kcore=kernel.block(points,points)
        rhs=rng.normal(size=m)
        for case_index,lam in enumerate([.001,1e-5]):
            C=Kcore.copy();C.flat[::m+1]+=lam
            a=MixedFactor(C,list(range(8)),generic);b=MixedFactor(C,list(range(8)),special)
            checks=compare_factors(a,b,rhs)
            streamed=StreamedFirst8(X,bdata,bw,m,lam,generic)
            stream_checks=dict(indices=bits_equal(streamed.S,selected),low=bits_equal(streamed.low,a.low),
                wide=bits_equal(streamed.wide,a.wide),solve=bits_equal(streamed.solve(rhs),a.solve(rhs)),
                minimum_pivot=bits_equal(np.array(streamed.minimum_raw_pivot),np.array(a.pivots.min())),
                epsilon_norm=bits_equal(np.array(streamed.initial_epsilon_norm),np.array(np.linalg.norm(a.epsilon))))
            assert all(stream_checks.values()),stream_checks
            rows={name:dict(factor_seconds=[],factor_cpu_seconds=[],solve_seconds=[],solve_cpu_seconds=[],combined_seconds=[]) for name in ['generic','specialized']}
            for repetition in range(5):
                names=['generic','specialized'] if (repetition+case_index)%2==0 else ['specialized','generic']
                for name in names:
                    helper=generic if name=='generic' else special
                    wall,c=perf_counter(),process_time();f=MixedFactor(C,list(range(8)),helper)
                    fw,fc=perf_counter()-wall,process_time()-c
                    wall,c=perf_counter(),process_time();answer=f.solve(rhs)
                    sw,sc=perf_counter()-wall,process_time()-c
                    assert bits_equal(f.low,a.low) and bits_equal(f.wide,a.wide)
                    assert bits_equal(f.epsilon,a.epsilon) and bits_equal(f.pivots,a.pivots)
                    assert bits_equal(answer,a.solve(rhs))
                    row=rows[name]
                    for key,value in [('factor_seconds',fw),('factor_cpu_seconds',fc),('solve_seconds',sw),('solve_cpu_seconds',sc),('combined_seconds',fw+sw)]:row[key].append(value)
                    del f,answer
            for row in rows.values():
                for key in list(row):row['median_'+key]=float(np.median(row[key]))
            case=dict(n=len(X),m=m,regularization=lam,data_sha256=digest(path),
                selected_indices_sha256=sha256(selected.tobytes()).hexdigest(),dense_core_sha256=sha256(C.tobytes()).hexdigest(),
                bitwise_checks=checks,original_streamed_build_checks=stream_checks,
                low_factor_sha256=sha256(a.low.tobytes()).hexdigest(),wide_factor_sha256=sha256(a.wide.tobytes()).hexdigest(),
                original_streamed_factor_seconds=streamed.factor_seconds,methods=rows,
                factor_speedup=rows['generic']['median_factor_seconds']/rows['specialized']['median_factor_seconds'],
                solve_speedup=rows['generic']['median_solve_seconds']/rows['specialized']['median_solve_seconds'])
            record['cases'].append(case);save()
            print(json.dumps(dict(regularization=lam,factor_speedup=case['factor_speedup'],solve_speedup=case['solve_speedup'])),flush=True)
        record['sources_unchanged']=all(digest(p)==hashes[str(p.relative_to(ROOT))] for p in dependencies)
        assert record['sources_unchanged']
    record['status']='complete';save()
    print(json.dumps(dict(status=record['status'],cpu=record['elapsed_cpu_seconds'],wall=record['elapsed_wall_seconds'])))


if __name__=='__main__':main()
