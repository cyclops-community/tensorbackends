"""
This module implements the numpy backend.
"""

import itertools, functools, operator

import numpy as np
import numpy.linalg as la

from ...interface import Backend
from ...utils import einstr
from .numpy_tensor import NumPyTensor


class NumPyBackend(Backend):
    @property
    def name(self):
        return 'numpy'

    @property
    def tensor(self):
        return NumPyTensor

    def astensor(self, obj, dtype=None):
        if isinstance(obj, self.tensor) and dtype is None:
            return obj
        elif isinstance(obj, self.tensor) and dtype is not None:
            return obj.astype(dtype)
        elif isinstance(obj, np.ndarray) and dtype is None:
            return self.tensor(obj)
        elif isinstance(obj, np.ndarray) and dtype is not None:
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

    def einsum(self, subscripts, *operands):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsum(subscripts, ndims)
        result = np.einsum(expr.indices_string, *(operand.tsr for operand in operands))
        if isinstance(result, np.ndarray):
            newshape = expr.outputs[0].newshape(result.shape)
            result = result.reshape(*newshape)
            return self.tensor(result)
        else:
            return result

    def einsvd(self, subscripts, a):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        expr = einstr.parse_einsvd(subscripts, a.ndim)
        newindex = (expr.output_indices - expr.input_indices).pop()
        prod = lambda iterable: functools.reduce(operator.mul, iterable, 1)
        axis_of_index = {index: axis for axis, index in enumerate(expr.inputs[0])}
        u_axes_from_a = [axis_of_index[index] for index in expr.outputs[0] if index != newindex]
        vh_axes_from_a = [axis_of_index[index] for index in expr.outputs[1] if index != newindex]
        # form matrix of a
        a_matrix_axes = [*u_axes_from_a, *vh_axes_from_a]
        a_matrix_shape = (prod(a.shape[axis] for axis in u_axes_from_a), -1)
        a_matrix = a.transpose(*a_matrix_axes).reshape(*a_matrix_shape)
        u, s, vh = la.svd(a_matrix, full_matrices=False)
        # form u
        u = u.reshape(*(a.shape[axis] for axis in u_axes_from_a), len(s))
        u = np.moveaxis(u, -1, expr.outputs[0].find(newindex))
        u = u.reshape(*expr.outputs[0].newshape(u.shape))
        # form vh
        vh = vh.reshape(len(s), *(a.shape[axis] for axis in vh_axes_from_a))
        vh = np.moveaxis(vh, 0, expr.outputs[1].find(newindex))
        vh = vh.reshape(*expr.outputs[1].newshape(vh.shape))
        return self.tensor(u), self.tensor(s), self.tensor(vh)

    def isclose(self, a, b, *, rtol=1e-9, atol=0.0):
        a = b.tsr if isinstance(b, NumPyTensor) else b
        b = b.tsr if isinstance(b, NumPyTensor) else b
        y = np.isclose(a, b, rtol=rtol, atol=atol)
        return NumPyTensor(y) if isinstance(y, np.ndarray) else y

    def allclose(self, a, b, *, rtol=1e-9, atol=0.0):
        a = b.tsr if isinstance(b, NumPyTensor) else b
        b = b.tsr if isinstance(b, NumPyTensor) else b
        return np.allclose(a, b, rtol=rtol, atol=atol)

    def __getattr__(self, attr):
        wrap = lambda val: NumPyTensor(val) if isinstance(val, np.ndarray) else val
        unwrap = lambda val: val.unwrap() if isinstance(val, NumPyTensor) else val
        try:
            result = getattr(np, attr) if hasattr(np, attr) else getattr(la, attr)
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
            raise ValueError('failed to get {} from numpy'.format(attr)) from e
