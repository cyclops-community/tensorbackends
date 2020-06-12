"""
This module implements the cupy tensor.
"""

import cupy as np

from ...interface import Tensor


class CuPyTensor(Tensor):
    def __init__(self, tsr):
        self.tsr = tsr

    @property
    def backend(self):
        from . import CuPyBackend
        return CuPyBackend()

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

    def unwrap(self):
        return self.tsr

    def numpy(self):
        return self.tsr.get()

    def __repr__(self):
        return repr(self.tsr)

    def __str__(self):
        return str(self.tsr)

    def __getitem__(self, key):
        value = self.tsr[key]
        return CuPyTensor(value) if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value):
        self.tsr[key] = value.unwrap() if isinstance(value, CuPyTensor) else value

    def copy(self):
        return CuPyTensor(np.copy(self.tsr))

    def astype(self, dtype):
        return CuPyTensor(self.tsr.astype(dtype))

    def write(self, inds, vals):
        self.tsr.put(inds, vals)

    def __getattr__(self, attr):
        wrap = lambda val: CuPyTensor(val) if isinstance(val, np.ndarray) else val
        unwrap = lambda val: val.unwrap() if isinstance(val, CuPyTensor) else val
        try:
            result = getattr(self.tsr, attr)
        except AttributeError as e:
            raise AttributeError("failed to get '{}' from cupy.ndarray".format(attr)) from e
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


def add_unary_operators(*operator_names):
    def add_unary_operator(operator_name):
        def method(self):
            return CuPyTensor(getattr(self.tsr, operator_name)())
        method.__module__ = CuPyTensor.__module__
        method.__qualname__ = '{}.{}'.format(CuPyTensor.__qualname__, operator_name)
        method.__name__ = operator_name
        setattr(CuPyTensor, operator_name, method)
    for op_name in operator_names:
        add_unary_operator(op_name)


def add_binary_operators(*operator_names):
    def add_binary_operator(operator_name):
        def method(self, other):
            return CuPyTensor(getattr(self.tsr, operator_name)(
                other.tsr if isinstance(other, CuPyTensor) else other
            ))
        method.__module__ = CuPyTensor.__module__
        method.__qualname__ = '{}.{}'.format(CuPyTensor.__qualname__, operator_name)
        method.__name__ = operator_name
        setattr(CuPyTensor, operator_name, method)
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

    '__lt__',
    '__le__',
    '__eq__',
    '__ne__',
    '__gt__',
    '__ge__',
)
