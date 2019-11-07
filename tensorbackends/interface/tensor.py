"""
This module defines the interface of a tensor.
"""

class Tensor:
    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def ndim(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def astype(self, dtype):
        raise NotImplementedError()

    def write(self, inds, vals):
        raise NotImplementedError()
