"""
This module implements the numpy backend.
"""

import functools

import numpy as np
import numpy.linalg as la

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

    def __getattr__(self, attr):
        wrap = lambda val: NumPyTensor(val) if isinstance(val, np.ndarray) else val
        unwrap = lambda val: val.tsr if isinstance(val, NumPyTensor) else val
        try:
            result = getattr(np, attr) if hasattr(np, attr) else getattr(la, attr)
            if callable(result):
                @functools.wraps(result)
                def wrapped_result(*args, **kwargs):
                    unwrapped_args = tuple(unwrap(v) for v in args)
                    unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
                    retval = result(*unwrapped_args, **unwrapped_kwargs)
                    if isinstance(retval, tuple):
                        wrapped_retval = tuple(wrap(v) for v in retval)
                    elif isinstance(retval, list):
                        wrapped_retval = [wrap(v) for v in retval]
                    elif isinstance(retval, dict):
                        wrapped_retval = {k: wrap(v) for k, v in retval.items()}
                    else:
                        wrapped_retval = wrap(retval)
                    return wrapped_retval
                return wrapped_result
            else:
                return result
        except Exception as e:
            raise ValueError('Failed to designate to numpy') from e
