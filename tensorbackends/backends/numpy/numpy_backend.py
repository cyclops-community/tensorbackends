"""
This module implements the numpy backend.
"""

import numpy as np

from ...interface import Backend
from .numpy_tensor import NumPyTensor


class NumPyBackend(Backend):
    @property
    def name(self):
        return 'numpy'

    @property
    def tensor(self):
        return NumPyTensor

    def astensor(self, obj, dtype=None):
        if isinstance(obj, self.tensor):
            return obj.astype(dtype)
        elif isinstance(obj, np.ndarray):
            return self.tensor(obj.astype(dtype))
        else:
            return self.tensor(np.array(obj, dtype=dtype))

    def empty(self, shape, dtype=float):
        return self.tensor(np.empty(shape, dtype=dtype))

    def zeros(self, shape, dtype=float):
        return self.tensor(np.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=float):
        return self.tensor(np.ones(shape, dtype=dtype))

    def copy(self, a):
        return a.copy()
