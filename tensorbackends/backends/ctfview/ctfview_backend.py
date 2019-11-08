"""
This module implements the ctfview backend.
"""

import numpy as np
import ctf

from ...interface import Backend
from .ctfview_tensor import CTFViewTensor


class CTFViewBackend(Backend):
    @property
    def name(self):
        return 'ctfview'

    @property
    def tensor(self):
        return CTFViewTensor

    def astensor(self, obj, dtype=None):
        if isinstance(obj, self.tensor):
            return obj.astype(dtype)
        elif isinstance(obj, ctf.tensor):
            return self.tensor(obj.astype(dtype))
        else:
            return self.tensor(ctf.astensor(obj, dtype=dtype))

    def empty(self, shape, dtype=float):
        return self.tensor(ctf.empty(shape, dtype=dtype))

    def zeros(self, shape, dtype=float):
        return self.tensor(ctf.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=float):
        return self.tensor(ctf.ones(shape, dtype=dtype))

    def copy(self, a):
        return a.copy()

    def reshape(self, a, newshape):
        return a.reshape(*newshape)

    def transpose(self, a, axes=None):
        if axes is None:
            axes = reversed(range(a.ndim))
        return a.transpose(*axes)

    def __getattr__(self, attr):
        def extract(vtsr):
            vtsr.match_indices()
            return vtsr.tsr
        wrap = lambda val: CTFViewTensor(val) if isinstance(val, ctf.tensor) else val
        unwrap = lambda val: extract(val) if isinstance(val, CTFViewTensor) else val
        try:
            result = getattr(ctf, attr)
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
                wrapped_result.__qualname__ = '{}.{}'.format(type(self).__qualname__,attr)
                return wrapped_result
            else:
                return result
        except Exception as e:
            raise ValueError('Failed to get {} from ctf'.format(attr)) from e
