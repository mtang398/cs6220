"""Nyström rank truncation from current-solve products.

This changes the representation from actual selected columns to a rank-k
generalized Nyström factor. The optional final QR selection restores an actual
column preconditioner. Rank and column-query counts are reported separately.
"""
import numpy as np
from scipy import linalg
from .preconditioners import LowRankPreconditioner
from .numerics import NystromPreconditioner


def compress_probe(state,p,Kp):
    """Append a K-orthogonalized probe, then take the best rank-k factor."""
    s=np.sqrt(state.eigenvalues)
    cross=s*(state.basis.T@p)
    curvature=float(p@Kp)
    delta=curvature-float(cross@cross)
    event=dict(kind='rank_truncation',accepted=False,exchanges=0,
               probe_curvature=curvature,schur_complement=delta)
    if delta<=1e-11*max(abs(curvature),float(cross@cross),1e-300):
        event['reason']='dependent_or_unreliable_probe'
        return state,event
    innovation=(Kp-state.basis@(s*cross))/np.sqrt(delta)
    factor=np.column_stack((state.basis*s,innovation))
    U,snew,_=linalg.svd(factor,full_matrices=False,check_finite=False)
    U,snew=U[:,:state.rank],snew[:state.rank]
    oldtrace=float(state.eigenvalues.sum());newtrace=float(snew@snew)
    event.update(trace_before=oldtrace,trace_after=newtrace)
    if newtrace<=oldtrace*(1+1e-9):
        event['reason']='no_material_trace_increase'
        return state,event
    new=LowRankPreconditioner(U,snew*snew,state.regularization)
    event.update(accepted=True,exchanges=1)
    return new,event


def select_columns_from_factor(kernel,state):
    """QR pivoting on the learned factor, followed by actual kernel queries."""
    _,_,pivots=linalg.qr((state.basis*np.sqrt(state.eigenvalues)).T,
                        mode='economic',pivoting=True,check_finite=False)
    indices=pivots[:state.rank]
    columns=np.column_stack([kernel.column(int(j)) for j in indices])
    return NystromPreconditioner(indices,columns,state.regularization)
