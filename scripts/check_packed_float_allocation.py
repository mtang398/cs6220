#!/usr/bin/env python3
"""Isolate ctypes wrapper allocations; original factor/results stay unchanged."""
import ctypes
import gc
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter,process_time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_limits
from check_packed_float_factor import PackedFloat,ROOT,LIMIT


class RawPointers:
    """Caller-owned contiguous arrays stay alive throughout each synchronous call."""
    def __init__(self,path):
        self.library=ctypes.CDLL(str(path))
        self.column=self.library.packed_float_column
        self.column.argtypes=[ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.column.restype=ctypes.c_int
        self.solve=self.library.packed_float_solve
        self.solve.argtypes=[ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
        self.solve.restype=ctypes.c_int
    def packed_float_column(self,n,j,packed,work,pivot):
        return self.column(n,j,packed.__array_interface__['data'][0],
            work.__array_interface__['data'][0],pivot.__array_interface__['data'][0])
    def packed_float_solve(self,n,packed,rhs):
        return self.solve(n,packed.__array_interface__['data'][0],rhs.__array_interface__['data'][0])


def original_interface(path):
    library=ctypes.CDLL(str(path))
    f32=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    library.packed_float_column.argtypes=[ctypes.c_size_t,ctypes.c_size_t,f32,f64,f64]
    library.packed_float_column.restype=ctypes.c_int
    library.packed_float_solve.argtypes=[ctypes.c_size_t,f32,f64]
    library.packed_float_solve.restype=ctypes.c_int
    return library


def audit(X,b,bw,helper,name):
    gc.collect();before_gc_counts=gc.get_count()
    started,cpu_started=perf_counter(),process_time()
    tracemalloc.start()
    try:
        factor=PackedFloat(X,b,bw,600,1e-5,helper)
        current,peak=tracemalloc.get_traced_memory()
        # Capture the algorithm peak before snapshot/statistics allocations.
        snapshot=tracemalloc.take_snapshot()
        top=[dict(traceback=str(stat.traceback),bytes=stat.size,count=stat.count)
            for stat in snapshot.statistics('lineno')[:12]]
        del snapshot
        before_collection=tracemalloc.get_traced_memory()[0]
        collected=gc.collect()
        after_collection=tracemalloc.get_traced_memory()[0]
    finally:tracemalloc.stop()
    # Hashing allocates bytes buffers; deliberately outside the allocation trace.
    factor_hash=sha256(factor.factor.tobytes()).hexdigest()
    indices_hash=sha256(factor.indices.tobytes()).hexdigest()
    rhs=np.random.default_rng(622105).normal(size=600)
    answer=factor.solve(rhs)
    solve_hash=sha256(answer.tobytes()).hexdigest()
    result=dict(interface=name,m=600,n=len(X),regularization=1e-5,
        setup_algorithm_traced_current_bytes=current,setup_algorithm_traced_peak_bytes=peak,
        trace_within_1MiB=peak<=LIMIT,named_setup=factor.setup_memory,
        traced_bytes_before_explicit_gc=before_collection,traced_bytes_after_explicit_gc=after_collection,
        bytes_released_by_explicit_gc=before_collection-after_collection,
        gc_collected_objects=collected,gc_counts_before=before_gc_counts,gc_thresholds=gc.get_threshold(),
        top_live_allocations_after_setup=top,factor_sha256=factor_hash,indices_sha256=indices_hash,
        single_solve_sha256=solve_hash,minimum_pivot=float(factor.pivots.min()),
        audit_wall_seconds=perf_counter()-started,audit_cpu_seconds=process_time()-cpu_started,
        scope='Separate traced FPS+factor construction. Snapshot/GC/hash/one solve are not included in the saved setup peak. Audit times are not native benchmark times.')
    del factor,answer,rhs;gc.collect()
    return result


def main():
    output=ROOT/'results/packed-float-allocation.json'
    if output.exists():raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU,(10,15))
    started,cpu_started=perf_counter(),process_time()
    original_result=ROOT/'results/packed-float-factor.json'
    prior=json.loads(original_result.read_text())
    checks={p:sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prior['sources'].items()}
    assert all(checks.values())
    lib=ROOT/'results/packed_float_factor.so'
    helpers=[('original_ndpointer',original_interface(lib)),('raw_pointer_adapter',RawPointers(lib))]
    with np.load(ROOT/'data/prepared/synthetic-n2560-smooth-s0-lambda0.001.npz') as data:
        X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
    X.setflags(write=False);b.setflags(write=False)
    with threadpool_limits(limits=1):
        runs=[audit(X,b,bw,helper,name) for name,helper in helpers]
    equivalence={key:runs[0][key]==runs[1][key] for key in ['factor_sha256','indices_sha256','single_solve_sha256']}
    assert all(equivalence.values())
    result=dict(status='complete',original_sources_unchanged=checks,
        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),original_result,ROOT/'scripts/check_packed_float_factor.py',ROOT/'scripts/packed_float_factor.c',lib]},
        changes='Only Python ctypes argument marshalling differs. Same original PackedFloat class, C arithmetic, FPS, factor storage and solve.',
        equivalence=equivalence,runs=runs,
        elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Raw adapter trusts the caller contract: contiguous native float32 factor, contiguous float64 work/RHS, valid lengths, and live arrays during synchronous C calls. This validates setup allocation only, not a full solver.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','elapsed_wall_seconds','elapsed_cpu_seconds']}))


if __name__=='__main__':main()
