"""
This module implements the numpy tensor.
"""

import numpy as np
import numpy.linalg as la

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

    def write(self, inds, vals):
        self.tsr.put(inds, vals)

    def __getattr__(self, attr):
        wrap = lambda val: NumPyTensor(val) if isinstance(val, np.ndarray) else val
        unwrap = lambda val: val.tsr if isinstance(val, NumPyTensor) else val
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
                wrapped_result.__qualname__ = '{}.{}'.format(type(self).__qualname__,attr)
                return wrapped_result
            else:
                return result
        except Exception as e:
            raise ValueError('Failed to get {} from numpy'.format(attr)) from e

    def __getitem__(self, key):
        value = self.tsr[key]
        return NumPyTensor(value) if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value):
        self.tsr[key] = value.tsr if isinstance(value, np.ndarray) else value


def add_unary_operators(*operator_names):
    def add_unary_operator(operator_name):
        def method(self):
            return NumPyTensor(getattr(self.tsr, operator_name)())
        method.__module__ = NumPyTensor.__module__
        method.__qualname__ = '{}.{}'.format(NumPyTensor.__qualname__,operator_name)
        method.__name__ = operator_name
        setattr(NumPyTensor, operator_name, method)
    for op_name in operator_names:
        add_unary_operator(op_name)


def add_binary_operators(*operator_names):
    def add_binary_operator(operator_name):
        def method(self, other):
            return NumPyTensor(getattr(self.tsr, operator_name)(
                other.tsr if isinstance(other, NumPyTensor) else other
            ))
        method.__module__ = NumPyTensor.__module__
        method.__qualname__ = '{}.{}'.format(NumPyTensor.__qualname__,operator_name)
        method.__name__ = operator_name
        setattr(NumPyTensor, operator_name, method)
    for op_name in operator_names:
        add_binary_operator(op_name)


add_unary_operators(
    '__pos__',
    '__neg__',
    '__abs__',
)

add_binary_operators(
    '__add__',
    '__sub__',
    '__mul__',
    '__matmul__',
    '__truediv__',
    '__floordiv__',
    '__pow__',

    '__radd__',
    '__rsub__',
    '__rmul__',
    '__rmatmul__',
    '__rtruediv__',
    '__rfloordiv__',
    '__rpow__',

    '__iadd__',
    '__isub__',
    '__imul__',
    '__imatmul__',
    '__itruediv__',
    '__ifloordiv__',
    '__ipow__',
)
