"""
This module implements the ctfview backend.
"""

import numpy as np
import ctf

from ...interface import Backend
from ...utils import einstr
from .ctfview_tensor import CTFViewTensor


class CTFViewBackend(Backend):
    @property
    def name(self):
        return 'ctfview'

    @property
    def tensor(self):
        return CTFViewTensor

    def astensor(self, obj, dtype=None):
        if isinstance(obj, self.tensor) and dtype is None:
            return obj
        elif isinstance(obj, self.tensor) and dtype is not None:
            return obj.astype(dtype)
        elif isinstance(obj, ctf.tensor) and dtype is None:
            return self.tensor(obj)
        elif isinstance(obj, ctf.tensor) and dtype is not None:
            return self.tensor(obj.astype(dtype))
        else:
            return self.tensor(ctf.astensor(obj, dtype=dtype))

    def empty(self, shape, dtype=float):
        return self.tensor(ctf.empty(shape, dtype=dtype))

    def zeros(self, shape, dtype=float):
        return self.tensor(ctf.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=float):
        return self.tensor(ctf.ones(shape, dtype=dtype))

    def shape(self, a):
        return a.shape

    def ndim(self, a):
        return a.ndim

    def copy(self, a):
        return a.copy()

    def reshape(self, a, newshape):
        return a.reshape(*newshape)

    def transpose(self, a, axes=None):
        if axes is None:
            axes = reversed(range(a.ndim))
        return a.transpose(*axes)

    def einsum(self, subscripts, *operands):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        for operand in operands:
            operand.match_indices()
        ndims = [operand.ndim for operand in operands]
        operands_indices = [operand.indices for operand in operands]
        expr = einstr.parse_einsum(subscripts, ndims)
        result = ctf.einsum(expr.indices_string, *(operand.tsr for operand in operands))
        if isinstance(result, ctf.tensor):
            newshape = expr.outputs[0].newshape(result.shape)
            return self.tensor(result).reshape(*newshape)
        else:
            return result

    def einsvd(self, subscripts, a):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        expr = einstr.parse_einsvd(subscripts, a.ndim)
        a.match_indices()
        u, s, vh = a.tsr.i(expr.inputs[0].indices_string).svd(
            expr.outputs[0].indices_string,
            expr.outputs[1].indices_string,
        )
        u_newshape = expr.outputs[0].newshape(u.shape)
        vh_newshape = expr.outputs[1].newshape(vh.shape)
        return self.tensor(u).reshape(*u_newshape), self.tensor(s), self.tensor(vh).reshape(*vh_newshape)

    def isclose(self, a, b, *, rtol=1e-9, atol=0.0):
        if isinstance(a, self.tensor): a.match_indices()
        if isinstance(b, self.tensor): b.match_indices()
        return abs(a - b) <= atol + rtol * abs(b)

    def allclose(self, a, b, *, rtol=1e-9, atol=0.0):
        return self.all(self.isclose(a, b, rtol=rtol, atol=atol))

    def __getattr__(self, attr):
        wrap = lambda val: CTFViewTensor(val) if isinstance(val, ctf.tensor) else val
        unwrap = lambda val: val.unwrap() if isinstance(val, CTFViewTensor) else val
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
            raise ValueError('failed to get {} from ctf'.format(attr)) from e
