"""
This module implements the ctf tensor.
"""

import ctf

from ...interface import Tensor
from ..utils import mimic_operators


@mimic_operators('tsr')
class CTFTensor(Tensor):
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
        return CTFTensor(self.tsr.copy())

    def astype(self, dtype):
        return CTFTensor(self.tsr.astype(dtype))

    def __getattr__(self, attr):
        wrap = lambda val: CTFTensor(val) if isinstance(val, ctf.tensor) else val
        unwrap = lambda val: val.tsr if isinstance(val, CTFTensor) else val
        try:
            result = getattr(self.tsr, attr)
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
                wrapped_result.__qualname__ = f'{type(self).__qualname__}.{attr}'
                return wrapped_result
            else:
                return result
        except Exception as e:
            raise ValueError('Failed to designate to ctf') from e

    def __getitem__(self, key):
        value = self.tsr[key]
        return CTFTensor(value) if isinstance(value, ctf.tensor) else value

    def __setitem__(self, key, value):
        self.tsr[key] = value.tsr if isinstance(value, ctf.tensor) else value
