"""Established fixed preconditioner controls using the same selected columns."""
import numpy as np
from scipy import linalg
from .numerics import NystromPreconditioner


class LowRankPreconditioner:
    """Factor-only inverse of U diag(values) U.T + lambda I."""
    def __init__(self,basis,values,regularization,indices=None):
        self.basis=np.array(basis,dtype=float,order='C',copy=True)
        self.eigenvalues=np.array(values,dtype=float,copy=True)
        self.regularization=float(regularization)
        self.indices=np.array([] if indices is None else indices,dtype=np.int64)
        self.rank=len(self.eigenvalues)
        for a in (self.basis,self.eigenvalues,self.indices):a.setflags(write=False)

    @classmethod
    def from_nystrom(cls,base):
        return cls(base.basis,base.eigenvalues,base.regularization,base.indices)

    @classmethod
    def from_sketch(cls,images,core,regularization):
        w,V=linalg.eigh((core+core.T)*.5,check_finite=False)
        threshold=32*np.finfo(float).eps*max(len(w),1)*max(w[-1],np.finfo(float).tiny)
        if w[0]<-max(threshold,64*np.finfo(float).eps):
            raise np.linalg.LinAlgError('sketch core is not numerically positive semidefinite')
        keep=w>threshold
        F=(images@V[:,keep])/np.sqrt(w[keep])
        U,s,_=linalg.svd(F,full_matrices=False,check_finite=False)
        return cls(U,s*s,regularization)

    def apply(self,v):
        c=self.basis.T@v;perp=v-self.basis@c
        correction=self.basis.T@perp;c+=correction;perp-=self.basis@correction
        return perp/self.regularization+self.basis@(c/(self.eigenvalues+self.regularization))

    @property
    def storage_bytes(self):
        return self.basis.nbytes+self.eigenvalues.nbytes+self.indices.nbytes


def initialize_gaussian(kernel,rank,regularization,rng,*,power=False):
    """Gaussian-sketch Nyström; charge the block kernel product to setup."""
    omega=rng.normal(size=(kernel.n,rank))
    images=kernel.matmat(omega)
    if power:
        omega,_=linalg.qr(images,mode='economic',check_finite=False)
        images=kernel.matmat(omega)
    return LowRankPreconditioner.from_sketch(images,omega.T@images,regularization)


def initialize_fps(kernel, rank, regularization, rng):
    """Farthest-point selection for the Gaussian kernel, with one random start."""
    indices = []
    columns = []
    similarity = np.zeros(kernel.n)
    j = int(rng.integers(kernel.n))
    for _ in range(rank):
        indices.append(j)
        column = kernel.column(j)
        columns.append(column)
        similarity = np.maximum(similarity,column)
        similarity[indices] = np.inf
        j = int(np.argmin(similarity))
    return NystromPreconditioner(np.array(indices),np.column_stack(columns),regularization)


class SpectralPreconditioner:
    """Nyström spectral scaling, up to an irrelevant positive scalar.

    M = U (Lambda + lambda I) U.T + (lambda_min + lambda)(I-U U.T).
    Frangella, Tropp, Udell (2023), randomized Nyström preconditioning.
    Only the spectrum's smallest retained value changes the complement scale.
    """
    def __init__(self, base):
        self.base = base
        self.indices, self.rank = base.indices, base.rank
        self.regularization = base.regularization
        self.floor = self.regularization + (base.eigenvalues[-1] if base.rank else 0.)

    def apply(self, v):
        projected, complement = self.base._components(v)
        return complement/self.floor + self.base.basis @ (projected/(self.base.eigenvalues+self.regularization))

    @property
    def storage_bytes(self):
        return self.base.storage_bytes


class DiagonalPreconditioner:
    """Diagonal-corrected Nyström inverse, applied after diagonal whitening."""
    def __init__(self, base):
        self.base = base
        self.indices, self.rank = base.indices, base.rank
        self.regularization = base.regularization
        diagonal = np.maximum(1-np.sum(base.basis**2*base.eigenvalues[None,:],axis=1),0)+base.regularization
        self.scale = 1/np.sqrt(diagonal)
        factor = self.scale[:,None]*base.basis*np.sqrt(base.eigenvalues)[None,:]
        self.basis, singular, _ = linalg.svd(factor,full_matrices=False,check_finite=False)
        self.eigenvalues = singular**2

    def apply(self, v):
        w = self.scale*v
        projected = self.basis.T @ w
        complement = w-self.basis @ projected
        correction = self.basis.T @ complement
        projected += correction
        complement -= self.basis @ correction
        return self.scale*(complement+self.basis @ (projected/(1+self.eigenvalues)))

    @property
    def storage_bytes(self):
        return self.base.storage_bytes+self.scale.nbytes+self.basis.nbytes+self.eigenvalues.nbytes
