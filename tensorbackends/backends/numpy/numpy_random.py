"""
This module implements the random module for numpy backend.
"""

import numpy as np

from ...interface import Random
from .numpy_tensor import NumPyTensor


class NumPyRandom(Random):
    def seed(self, seed):
        np.random.seed(seed)

    def rand(self, *dims):
        return NumPyTensor(np.random.rand(*dims))

    def __getattr__(self, attr):
        wrap = lambda val: NumPyTensor(val) if isinstance(val, np.ndarray) else val
        unwrap = lambda val: val.unwrap() if isinstance(val, NumPyTensor) else val
        try:
            result = getattr(np.random, attr)
            if callable(result):
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
                wrapped_result.__module__ = type(self).__module__
                wrapped_result.__name__ = attr
                wrapped_result.__qualname__ = '{}.{}'.format(type(self).__qualname__, attr)
                return wrapped_result
            else:
                return result
        except Exception as e:
            raise ValueError('failed to get {} from numpy.random'.format(attr)) from e
