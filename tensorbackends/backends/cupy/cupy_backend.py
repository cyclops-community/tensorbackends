"""
This module implements the cupy backend.
"""

import functools, operator

import cupy as cp
import cupy.linalg as la

from ...interface import Backend
from ...utils import einstr
from ...utils.svd_absorb_s import svd_absorb_s
from .cupy_random import CuPyRandom
from .cupy_tensor import CuPyTensor


class CuPyBackend(Backend):
    @property
    def name(self):
        return 'cupy'

    @property
    def nproc(self):
        return 1

    @property
    def rank(self):
        return 0

    @property
    def random(self):
        return CuPyRandom()

    @property
    def tensor(self):
        return CuPyTensor

    def astensor(self, obj, dtype=None):
        if isinstance(obj, self.tensor) and dtype is None:
            return obj
        elif isinstance(obj, self.tensor) and dtype is not None:
            return obj.astype(dtype)
        elif isinstance(obj, cp.ndarray) and dtype is None:
            return self.tensor(obj)
        elif isinstance(obj, cp.ndarray) and dtype is not None:
            return self.tensor(obj.astype(dtype))
        else:
            return self.tensor(cp.array(obj, dtype=dtype))

    def empty(self, shape, dtype=float):
        return self.tensor(cp.empty(shape, dtype=dtype))

    def zeros(self, shape, dtype=float):
        return self.tensor(cp.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=float):
        return self.tensor(cp.ones(shape, dtype=dtype))

    def shape(self, a):
        return a.shape

    def ndim(self, a):
        return a.ndim

    def copy(self, a):
        return a.copy()

    def save(self, tsr, filename):
        with open(filename, 'w+b') as file:
            cp.save(file, tsr.unwrap(), allow_pickle=False)

    def load(self, filename):
        return self.tensor(cp.load(filename))

    def hstack(self, tensors):
        return self.tensor(cp.hstack(tuple(tsr.unwrap() for tsr in tensors)))

    def vstack(self, tensors):
        return self.tensor(cp.vstack(tuple(tsr.unwrap() for tsr in tensors)))

    def einsum(self, subscripts, *operands):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsum(subscripts, ndims)
        return self._einsum(expr, operands)

    def einsvd_reduced(self, subscripts, a, rank=None, absorb_s=False):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        expr = einstr.parse_einsvd(subscripts, a.ndim)
        def svd_func(matrix):
            u, s, vh = self.svd(matrix, absorb_s=absorb_s)
            if rank is not None and s.shape[0] > rank:
                u, s, vh = u[:,:rank], s[:rank], vh[:rank,:]
            return u, s, vh
        return self._einsvd(expr, a, svd_func)

    def einsvd_rand(self, subscripts, a, rank, niter=1, oversamp=5, absorb_s=False):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        expr = einstr.parse_einsvd(subscripts, a.ndim)
        def svd_func(matrix):
            return self.rsvd(matrix, rank, niter, oversamp, absorb_s=absorb_s)
        return self._einsvd(expr, a, svd_func)

    def einsumsvd_reduced(self, subscripts, *operands, rank=None, absorb_s=False):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsumsvd(subscripts, ndims)
        einsum_expr, einsvd_expr = einstr.split_einsumsvd(expr)
        a = self._einsum(einsum_expr, operands)
        def svd_func(matrix):
            u, s, vh = self.svd(matrix, absorb_s=absorb_s)
            if rank is not None and s.shape[0] > rank:
                u, s, vh = u[:,:rank], s[:rank], vh[:rank,:]
            return u, s, vh
        return self._einsvd(einsvd_expr, a, svd_func)

    def einsumsvd_rand(self, subscripts, *operands, rank, niter=1, oversamp=5, absorb_s=False):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsumsvd(subscripts, ndims)
        einsum_expr, einsvd_expr = einstr.split_einsumsvd(expr)
        a = self._einsum(einsum_expr, operands)
        def svd_func(matrix):
            return self.rsvd(matrix, rank, niter, oversamp, absorb_s=absorb_s)
        return self._einsvd(einsvd_expr, a, svd_func)

    def isclose(self, a, b, *, rtol=1e-9, atol=0.0):
        a = a.tsr if isinstance(a, CuPyTensor) else a
        b = b.tsr if isinstance(b, CuPyTensor) else b
        y = cp.isclose(a, b, rtol=rtol, atol=atol)
        return CuPyTensor(y) if isinstance(y, cp.ndarray) else y

    def allclose(self, a, b, *, rtol=1e-9, atol=0.0):
        a = a.tsr if isinstance(a, CuPyTensor) else a
        b = b.tsr if isinstance(b, CuPyTensor) else b
        return cp.allclose(a, b, rtol=rtol, atol=atol)

    def inv(self, a):
        return CuPyTensor(la.inv(a.unwrap()))

    def svd(self, a, absorb_s=False):
        u, s, vh = la.svd(a.unwrap(), full_matrices=False)
        u, s, vh = self.tensor(u), self.tensor(s), self.tensor(vh)
        u, s, vh = svd_absorb_s(u, s, vh, absorb_s)
        return u, s, vh

    def __getattr__(self, attr):
        wrap = lambda val: CuPyTensor(val) if isinstance(val, cp.ndarray) else val
        unwrap = lambda val: val.unwrap() if isinstance(val, CuPyTensor) else val
        try:
            result = getattr(cp, attr) if hasattr(cp, attr) else getattr(la, attr)
        except AttributeError as e:
            raise AttributeError("failed to get '{}' from cupy or cupy.linalg".format(attr)) from e
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

    def _einsum(self, expr, operands):
        result = cp.einsum(expr.indices_string, *(operand.tsr for operand in operands), optimize='greedy')
        if isinstance(result, cp.ndarray) and result.ndim != 0:
            newshape = expr.outputs[0].newshape(result.shape)
            result = result.reshape(*newshape)
            return self.tensor(result)
        elif isinstance(result, cp.ndarray):
            return result.item()
        else:
            return result

    def _einsvd(self, expr, a, svd_func):
        newindex = (expr.output_indices - expr.input_indices).pop()
        prod = lambda iterable: functools.reduce(operator.mul, iterable, 1)
        axis_of_index = {index: axis for axis, index in enumerate(expr.inputs[0])}
        u_axes_from_a = [axis_of_index[index] for index in expr.outputs[0] if index != newindex]
        vh_axes_from_a = [axis_of_index[index] for index in expr.outputs[1] if index != newindex]
        # form matrix of a
        a_matrix_axes = [*u_axes_from_a, *vh_axes_from_a]
        a_matrix_shape = (prod(a.shape[axis] for axis in u_axes_from_a), -1)
        a_matrix = a.transpose(*a_matrix_axes).reshape(*a_matrix_shape)
        u, s, vh = svd_func(a_matrix)
        # form u
        u = u.reshape(*(a.shape[axis] for axis in u_axes_from_a), s.shape[0])
        u = self.moveaxis(u, -1, expr.outputs[0].find(newindex))
        u = u.reshape(*expr.outputs[0].newshape(u.shape))
        # form vh
        vh = vh.reshape(s.shape[0], *(a.shape[axis] for axis in vh_axes_from_a))
        vh = self.moveaxis(vh, 0, expr.outputs[1].find(newindex))
        vh = vh.reshape(*expr.outputs[1].newshape(vh.shape))
        return u, s, vh
