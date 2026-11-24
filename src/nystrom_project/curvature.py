"""Fixed-budget column exchange fitted to curvature observed during the solve."""
import numpy as np
from scipy import linalg
from .numerics import NystromPreconditioner,NumericalStabilityError


def logdet_projected(model,P):
    B=model.basis.T @ P
    N=model.regularization*(P.T @ P)+B.T @ (model.eigenvalues[:,None]*B)
    factor=linalg.cholesky((N+N.T)*.5,lower=True,check_finite=False)
    return float(2*np.log(np.diag(factor)).sum())


def curvature_exchange(kernel,state,residual,rng,*,history,candidate_count=4,sampling="curvature"):
    """Maximize log det(P.T M P) on the same retained direction subspace.

    P contains observed A-conjugate search directions. The residual matrix
    E=K-Khat supplies candidate weights sum((E P)[j,:]**2)/E[j,j].
    Stored A*P removes any need for additional kernel applications in selection.
    """
    snapshot=history.snapshot()
    if not snapshot.pairs:return state,dict(accepted=False,exchanges=0,candidates=0)
    P=np.column_stack([p for p,_,_ in snapshot.pairs])
    AP=np.column_stack([ap for _,ap,_ in snapshot.pairs])
    # The objective is basis-invariant for rankings. Euclidean QR prevents
    # near dependencies from contaminating small projected matrix factorizations.
    Q,R=linalg.qr(P,mode="economic",check_finite=False)
    singular=linalg.svdvals(R,check_finite=False)
    if singular[-1] < 1e-10*singular[0]:
        return state,dict(accepted=False,exchanges=0,candidates=0,reason="dependent_history")
    AQ=linalg.solve_triangular(R.T,AP.T,lower=True,check_finite=False).T
    T=(Q.T @ AQ+AQ.T @ Q)*.5
    L=linalg.cholesky(T,lower=True,check_finite=False)
    P=linalg.solve_triangular(L,Q.T,lower=True,check_finite=False).T
    AP=linalg.solve_triangular(L,AQ.T,lower=True,check_finite=False).T
    del snapshot
    current=state;old_score=logdet_projected(current,P)
    initial_score=old_score
    event=dict(accepted=False,exchanges=0,candidates=0,numerical_fallbacks=0,
               criterion="logdet_projected_preconditioner",directions=P.shape[1],details=[])
    for _ in range(candidate_count):
        available=np.ones(kernel.n,dtype=bool);available[current.indices]=False
        choices=np.flatnonzero(available)
        if len(choices)==0:break
        if sampling=="curvature":
            KhatP=current.basis @ (current.eigenvalues[:,None]*(current.basis.T @ P))
            EP=AP-current.regularization*P-KhatP
            diagonal=np.maximum(1-np.sum(current.basis**2*current.eigenvalues[None,:],axis=1),0)
            weights=np.zeros(len(choices));good=diagonal[choices]>1e-13
            weights[good]=np.sum(EP[choices[good]]**2,axis=1)/diagonal[choices[good]]
        elif sampling=="residual":weights=residual[choices]**2
        elif sampling=="uniform":weights=np.ones(len(choices))
        else:raise ValueError(sampling)
        if not np.any(weights>0):break
        weights/=weights.sum();j=int(rng.choice(choices,p=weights))
        columns=np.column_stack((current.columns,kernel.column(j)))
        indices=np.append(current.indices,j);k=len(current.indices)
        W=columns[indices,:];cross=columns.T @ P
        G=columns.T @ columns
        scores=np.full(k+1,-np.inf);scores[-1]=old_score
        fallback=False
        try:
            eig=linalg.eigvalsh(W,check_finite=False)
            if eig[0]<1e-11*eig[-1]:raise NumericalStabilityError("dependent augmented core")
            chol=linalg.cho_factor(W,lower=True,check_finite=False)
            inverse=linalg.cho_solve(chol,np.eye(k+1),check_finite=False)
            projected=inverse @ cross
            base=current.regularization*(P.T @ P)+cross.T @ projected
            for i in range(k):
                N=base-np.outer(projected[i],projected[i])/inverse[i,i]
                factor=linalg.cholesky((N+N.T)*.5,lower=True,check_finite=False)
                scores[i]=2*np.log(np.diag(factor)).sum()
        except np.linalg.LinAlgError:
            fallback=True;event['numerical_fallbacks']+=1
            for i in range(k):
                take=np.arange(k+1)!=i
                replacement=NystromPreconditioner(indices[take],columns[:,take],current.regularization,
                                                   gram=G[np.ix_(take,take)])
                scores[i]=logdet_projected(replacement,P)
        best=int(np.argmax(scores));event['candidates']+=1
        detail=dict(candidate=j,accepted=False,logdet_before=old_score,
                    best_predicted_logdet=float(scores[best]),fallback=fallback)
        if best<k and scores[best]>old_score+1e-7:
            take=np.arange(k+1)!=best
            replacement=NystromPreconditioner(indices[take],columns[:,take],current.regularization,
                                               gram=G[np.ix_(take,take)])
            verified=logdet_projected(replacement,P)
            detail['verified_logdet']=verified
            if verified>old_score+1e-7:
                detail.update(accepted=True,removed=int(indices[best]))
                current=replacement;old_score=verified;event['exchanges']+=1
        event['details'].append(detail)
    event.update(accepted=bool(event['exchanges']),logdet_before=initial_score,logdet_after=old_score)
    return current,event
