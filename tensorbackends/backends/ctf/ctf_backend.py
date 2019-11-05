"""
This module implements the ctf backend.
"""

import functools

import ctf

from ...interface import Backend
from .ctf_tensor import CTFTensor


class CTFBackend(Backend):
    @property
    def name(self):
        return 'ctf'

    @property
    def tensor(self):
        return CTFTensor

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

    def __getattr__(self, attr):
        wrap = lambda val: CTFTensor(val) if isinstance(val, ctf.tensor) else val
        unwrap = lambda val: val.tsr if isinstance(val, CTFTensor) else val
        try:
            result = getattr(ctf, attr)
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
            raise ValueError('Failed to designate to ctf') from e
