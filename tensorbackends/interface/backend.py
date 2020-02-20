"""
This module defines the interface of a backend.
"""

from . import options
from .. import extensions


class Backend:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def nproc(self):
        raise NotImplementedError()

    @property
    def rank(self):
        raise NotImplementedError()

    @property
    def random(self):
        raise NotImplementedError()

    @property
    def tensor(self):
        raise NotImplementedError()

    def astensor(self, obj, dtype=None):
        raise NotImplementedError()

    def empty(self, shape, dtype=float):
        raise NotImplementedError()

    def zeros(self, shape, dtype=float):
        raise NotImplementedError()

    def ones(self, shape, dtype=float):
        raise NotImplementedError()

    def shape(self, a):
        raise NotImplementedError()

    def ndim(self, a):
        raise NotImplementedError()

    def copy(self, a):
        raise NotImplementedError()

    def moveaxis(self, a, source, destination):
        return extensions.moveaxis(self, a, source, destination)

    def einsum(self, subscripts, *operands):
        raise NotImplementedError()

    def einsvd(self, subscripts, a, option=options.ReducedSVD()):
        if isinstance(option, options.ReducedSVD):
            return self.einsvd_reduced(subscripts, a, option.rank)
        elif isinstance(option, options.RandomizedSVD):
            return self.einsvd_rand(subscripts, a, option.rank, option.niter, option.oversamp)
        else:
            raise ValueError('{} is not a valid option for einsvd'.format(type(option).__qualname__))

    def einsvd_reduced(self, subscripts, a, rank=None):
        raise NotImplementedError()

    def einsvd_rand(self, subscripts, a, rank, niter=1, oversamp=5):
        raise NotImplementedError()

    def einsumsvd(self, subscripts, *operands, option=options.ReducedSVD()):
        if isinstance(option, options.ReducedSVD):
            return self.einsumsvd_reduced(subscripts, *operands, rank=option.rank)
        elif isinstance(option, options.RandomizedSVD):
            return self.einsumsvd_rand(subscripts, *operands, rank=option.rank, niter=option.niter, oversamp=option.oversamp)
        elif isinstance(option, options.ImplicitRandomizedSVD):
            return self.einsumsvd_implicit_rand(subscripts, *operands, rank=option.rank, niter=option.niter)
        else:
            raise ValueError('{} is not a valid option for einsumsvd'.format(type(option).__qualname__))

    def einsumsvd_reduced(self, subscripts, *operands, rank=None):
        raise NotImplementedError()

    def einsumsvd_rand(self, subscripts, *operands, rank, niter=1, oversamp=5):
        raise NotImplementedError()

    def einsumsvd_implicit_rand(self, subscripts, *operands, rank, niter=1):
        return extensions.einsumsvd_implicit_rand(self, subscripts, *operands, rank=rank, niter=niter)

    def isclose(self, a, b, *, rtol=1e-9, atol=0.0):
        raise NotImplementedError()

    def allclose(self, a, b, *, rtol=1e-9, atol=0.0):
        raise NotImplementedError()

    def inv(self, a):
        raise NotImplementedError()

    def svd(self, a):
        raise NotImplementedError()

    def rsvd(self, a, rank, niter=1, oversamp=5):
        return extensions.rsvd(self, a, rank, niter, oversamp)
