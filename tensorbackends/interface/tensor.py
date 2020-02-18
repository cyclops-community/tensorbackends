"""
This module defines the interface of a tensor.
"""

class Tensor:
    @property
    def backend(self):
        raise NotImplementedError()

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

    def unwrap(self):
        raise NotImplementedError()

    def numpy(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def astype(self, dtype):
        raise NotImplementedError()

    def write(self, inds, vals):
        raise NotImplementedError()

    @property
    def T(self):
        return self.transpose(1, 0)

    @property
    def H(self):
        return self.conj().T
