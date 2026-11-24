"""End-to-end wrappers for method-revision experiments and matched controls."""
from time import perf_counter
import numpy as np

from .solvers import solve
from .numerics import initialize_rpcholesky,exchange_columns
from .exchange import energy_exchange
from .flexible import flexible_solve
from .curvature import curvature_exchange
from .preconditioners import (SpectralPreconditioner,DiagonalPreconditioner,initialize_fps,
                             LowRankPreconditioner,initialize_gaussian)
from .learning import compress_probe,select_columns_from_factor
from .schur import initialize_schur_fsai


def revised_solve(kernel,y,regularization,*,method="flex_fixed",rank=16,seed=0,
                  history_size=8,checkpoint=20,candidate_count=4,rtol=1e-8,
                  maxiter=2000,update_limit=None,min_improvement=1e-6,warmup_steps=10,sparsity=8):
    if not method.startswith("flex_"):
        return solve(kernel,y,regularization,method=method,rank=rank,seed=seed,
                     checkpoint=checkpoint,candidate_count=candidate_count,
                     rtol=rtol,maxiter=maxiter)
    if method not in ("flex_fixed","flex_fsai","flex_compact","flex_gaussian","flex_power","flex_gaussian_qr","flex_compress","flex_compress_columns","flex_restart","flex_static_qr","flex_greedy","flex_fps","flex_spectral","flex_diagonal","flex_q","flex_energy","flex_uniform_energy","flex_direction_energy",
                      "flex_curvature","flex_uniform_curvature","flex_residual_curvature"):
        raise ValueError(method)
    kernel.reset_stats()
    started=perf_counter()
    def factory():
        rng=np.random.default_rng(np.random.SeedSequence([seed,6220,1]))
        if method=="flex_fsai":return initialize_schur_fsai(kernel,rank,regularization,rng,sparsity=sparsity)
        if method in ("flex_gaussian","flex_power","flex_gaussian_qr"):
            model=initialize_gaussian(kernel,rank,regularization,np.random.default_rng(np.random.SeedSequence([seed,6220,3])),power=method=='flex_power')
            return select_columns_from_factor(kernel,model) if method=="flex_gaussian_qr" else model
        if method=="flex_fps":return initialize_fps(kernel,rank,regularization,rng)
        base=initialize_rpcholesky(kernel,rank,regularization,
                                  rng,pivoting="greedy" if method=="flex_greedy" else "random")
        if method=="flex_spectral":return SpectralPreconditioner(base)
        if method=="flex_diagonal":return DiagonalPreconditioner(base)
        if method=="flex_static_qr":return select_columns_from_factor(kernel,base)
        if method in ("flex_compact","flex_compress","flex_compress_columns","flex_restart"):
            return LowRankPreconditioner.from_nystrom(base)
        return base
    update_rng=np.random.default_rng(np.random.SeedSequence([seed,6220,2]))
    updates=0
    learning=method in ("flex_compress","flex_compress_columns","flex_restart")
    def learn(p,ap,residual,old,iteration):
        nonlocal updates
        if iteration>warmup_steps:return None
        if method=="flex_restart":
            if iteration<warmup_steps:return None
            new,event=old,dict(kind='matched_restart',accepted=False,exchanges=0)
        else:
            updates+=1
            new,event=compress_probe(old,p,ap-regularization*p)
        if iteration==warmup_steps:
            if method=="flex_compress_columns":
                new=select_columns_from_factor(kernel,new)
                event.update(column_reselection=True,selected_indices=new.indices.tolist())
            event['restart']=True
        return new,event
    def callback(x,residual,old,iteration,history):
        nonlocal updates
        if update_limit is not None and updates>=update_limit:
            return old,dict(accepted=False,exchanges=0,kind="update_limit")
        updates+=1
        if method=="flex_q":
            return exchange_columns(kernel,old,residual,update_rng,candidate_count=candidate_count)
        if method in ("flex_curvature","flex_uniform_curvature","flex_residual_curvature"):
            sampling={"flex_curvature":"curvature","flex_uniform_curvature":"uniform",
                      "flex_residual_curvature":"residual"}[method]
            return curvature_exchange(kernel,old,residual,update_rng,history=history,
                                      candidate_count=candidate_count,sampling=sampling)
        sampling={"flex_energy":"residual","flex_uniform_energy":"uniform",
                  "flex_direction_energy":"direction"}[method]
        return energy_exchange(kernel,old,residual,update_rng,candidate_count=candidate_count,
                               sampling=sampling,history=history,min_improvement=min_improvement)
    result=flexible_solve(kernel,y,regularization,preconditioner_factory=factory,
                          history_size=history_size,checkpoint=checkpoint,rtol=rtol,maxiter=maxiter,
                          update_callback=None if learning or method in ("flex_fixed","flex_fsai","flex_compact","flex_gaussian","flex_power","flex_gaussian_qr","flex_static_qr","flex_greedy","flex_fps","flex_spectral","flex_diagonal") else callback,
                          direction_update=learn if learning else None)
    result.metrics.update(total_seconds=perf_counter()-started,
                          method=method,update_attempts=updates,**kernel.stats)
    if learning:
        result.metrics.update(representation='selected_columns_after_learned_sketch' if method=='flex_compress_columns' else 'rank_k_factor',
                              warmup_steps=warmup_steps)
    return result
