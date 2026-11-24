"""Column replacement using the actual decrease in the quadratic objective."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .numerics import NystromPreconditioner, NumericalStabilityError


def removal_directions(state, candidate, column, residual):
    """Return the inverse-applied residual for every deletion, including the no-op.

    The common augmented inverse is applied stably. A principal-submatrix
    identity supplies rank-one corrections. Ill-conditioned cases use direct
    inverses through the existing stable Nyström representation.
    """
    lam=state.regularization
    indices=np.append(state.indices, candidate)
    columns=np.column_stack((state.columns,column))
    cross=state.columns.T @ column
    k=len(state.indices)
    gram=np.empty((k+1,k+1))
    gram[:k,:k]=state.gram
    gram[:k,k]=gram[k,:k]=cross
    gram[k,k]=column @ column
    core=columns[indices,:]
    def direct():
        directions=[]
        for i in range(k+1):
            take=np.arange(k+1)!=i
            if i==k:
                directions.append(state.apply(residual))
            else:
                model=NystromPreconditioner(indices[take],columns[:,take],lam,
                                            gram=gram[np.ix_(take,take)])
                directions.append(model.apply(residual))
        return np.column_stack(directions)
    fallback=False
    try:
        eig=linalg.eigvalsh(core,check_finite=False)
        if eig[0]<=32*np.finfo(float).eps*(k+1)*eig[-1]:
            raise NumericalStabilityError("rank-deficient augmented core")
        h=core+gram/lam
        h=(h+h.T)*.5
        eig=linalg.eigvalsh(h,check_finite=False)
        if eig[0] <= 1e-12*eig[-1]:
            raise NumericalStabilityError("ill-conditioned augmented inverse system")
        factor=linalg.cho_factor(h,lower=True,check_finite=False)
        inverse=linalg.cho_solve(factor,np.eye(k+1),check_finite=False)
        b=columns.T @ residual
        coefficients=inverse @ b
        augmented=NystromPreconditioner(indices,columns,lam,gram=gram)
        base=augmented.apply(residual)
        directions=base[:,None]+(columns @ inverse)*(coefficients/(lam**2*np.diag(inverse)))[None,:]
        old=state.apply(residual)
        if np.linalg.norm(directions[:,-1]-old)>1e-7*max(np.linalg.norm(old),1e-300):
            raise NumericalStabilityError("no-op direction verification failed")
        directions[:,-1]=old
    except np.linalg.LinAlgError:
        directions=direct()
        fallback=True
    return directions, indices, columns, gram, fallback


def projected_decreases(kernel, state, residual, directions, history=None):
    az=kernel.matmat(directions)+state.regularization*directions
    if history is not None:
        directions,az=history.project(directions,az)
    numerator=residual @ directions
    curvature=np.sum(directions*az,axis=0)
    values=np.full(len(curvature),-np.inf)
    good=(curvature>0)&np.isfinite(curvature)&np.isfinite(numerator)
    values[good]=numerator[good]**2/curvature[good]
    return values


def energy_exchange(kernel,state,residual,rng,*,candidate_count=4,
                    sampling="residual",history=None,min_improvement=1e-6):
    """Sequential candidates; retain k columns and preserve FCG direction history.

    Each candidate uses one batched kernel application to rank every deletion.
    A selected replacement is independently evaluated with the factor actually
    used by the solver. Its additional matvec is charged to the update.
    """
    current=state
    event=dict(accepted=False,exchanges=0,candidates=0,numerical_fallbacks=0,
               criterion="projected_first_step_energy_decrease",details=[])
    direction=adirection=None
    if sampling=="direction":
        direction=current.apply(residual)
        adirection=kernel.matvec(direction)+current.regularization*direction
        if history is not None:direction,adirection=history.project(direction,adirection)
    for _ in range(candidate_count):
        mask=np.ones(kernel.n,dtype=bool);mask[current.indices]=False
        choices=np.flatnonzero(mask)
        if len(choices)==0:break
        if sampling=="residual":
            weights=np.abs(residual[choices]);scale=weights.max()
            if scale==0:break
            weights=(weights/scale)**2;weights/=weights.sum()
        elif sampling=="direction":
            represented=current.basis @ (current.eigenvalues*(current.basis.T @ direction))
            missing=adirection-current.regularization*direction-represented
            diagonal=1-np.sum(current.basis**2*current.eigenvalues[None,:],axis=1)
            weights=np.zeros(len(choices))
            keep=diagonal[choices]>1e-13
            weights[keep]=missing[choices[keep]]**2/diagonal[choices[keep]]
            if not np.any(weights>0):break
            weights/=weights.sum()
        elif sampling=="uniform":weights=None
        else:raise ValueError(sampling)
        j=int(rng.choice(choices,p=weights))
        z,indices,columns,gram,fallback=removal_directions(current,j,kernel.column(j),residual)
        event["candidates"]+=1
        event["numerical_fallbacks"]+=int(fallback)
        values=projected_decreases(kernel,current,residual,z,history)
        old_value=float(values[-1]);best=int(np.argmax(values))
        if not np.isfinite(old_value):
            event.setdefault("invalid_directions",0)
            event["invalid_directions"]+=1
            continue
        detail=dict(candidate=j,accepted=False,energy_decrease_before=old_value,
                    best_predicted_decrease=float(values[best]),direct_fallback=fallback)
        if best<len(current.indices) and values[best]>old_value*(1+min_improvement):
            take=np.arange(len(indices))!=best
            replacement=NystromPreconditioner(indices[take],columns[:,take],current.regularization,
                                               gram=gram[np.ix_(take,take)])
            stable=replacement.apply(residual)
            az=kernel.matvec(stable)+current.regularization*stable
            if history is not None:stable,az=history.project(stable,az)
            denom=float(stable @ az)
            verified=float(residual @ stable)**2/denom if denom>0 else -np.inf
            detail["verified_decrease"]=verified if np.isfinite(verified) else None
            if np.isfinite(verified) and verified>old_value*(1+min_improvement):
                detail.update(accepted=True,removed=int(indices[best]),energy_decrease_after=verified)
                current=replacement;event["exchanges"]+=1
                if sampling=="direction":direction,adirection=stable,az
        event["details"].append(detail)
    event["accepted"]=bool(event["exchanges"])
    return current,event
