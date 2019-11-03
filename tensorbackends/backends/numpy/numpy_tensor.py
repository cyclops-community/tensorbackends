"""
This module implements the numpy tensor.
"""

import numpy as np

from ...interface import Tensor


class NumPyTensor(Tensor):
    def __init__(self, tsr):
        self.tsr = tsr

    @property
    def shape(self):
        return self.tsr.shape

    @property
    def ndim(self):
        return self.tsr.ndim

    @property
    def size(self):
        return self.tsr.size

    @property
    def dtype(self):
        return self.tsr.dtype

    def __repr__(self):
        return repr(self.tsr)

    def __str__(self):
        return str(self.tsr)

    def copy(self):
        return NumPyTensor(np.copy(self.tsr))

    def astype(self, dtype):
        return NumPyTensor(self.tsr.astype(dtype))
