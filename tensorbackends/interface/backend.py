"""
This module defines the interface of a backend.
"""

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

    def einsum(self, subscripts, *operands):
        raise NotImplementedError()

    def einsvd(self, subscripts, a):
        raise NotImplementedError()

    def einsumsvd(self, subscripts, *operands):
        raise NotImplementedError()

    def einsumsvd_rand(self, subscripts, *operands, rank, niter=1):
        return extensions.einsumsvd_rand(self, subscripts, *operands, rank=rank, niter=niter)

    def isclose(self, a, b, *, rtol=1e-9, atol=0.0):
        raise NotImplementedError()

    def allclose(self, a, b, *, rtol=1e-9, atol=0.0):
        raise NotImplementedError()

    def inv(self, a):
        raise NotImplementedError()

    def svd(self, a):
        raise NotImplementedError()
