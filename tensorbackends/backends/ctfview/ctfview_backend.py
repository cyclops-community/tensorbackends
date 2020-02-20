"""
This module implements the ctfview backend.
"""

import ctf
import numpy as np

from ...interface import Backend
from ...utils import einstr
from .ctfview_random import CTFViewRandom
from .ctfview_tensor import CTFViewTensor
from . import indices_utils


class CTFViewBackend(Backend):
    @property
    def name(self):
        return 'ctfview'

    @property
    def nproc(self):
        return ctf.comm().np()

    @property
    def rank(self):
        return ctf.comm().rank()

    @property
    def random(self):
        return CTFViewRandom()

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
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsum(subscripts, ndims)
        return self._einsum(expr, operands)

    def einsvd_reduced(self, subscripts, a, rank=None):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        expr = einstr.parse_einsvd(subscripts, a.ndim)
        return self._einsvd_reduced(expr, a, rank)

    def einsvd_rand(self, subscripts, a, rank, niter=1, oversamp=5):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        expr = einstr.parse_einsvd(subscripts, a.ndim)
        return self._einsvd_rand(expr, a, rank, niter, oversamp)

    def einsumsvd_reduced(self, subscripts, *operands, rank=None):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsumsvd(subscripts, ndims)
        einsum_expr, einsvd_expr = einstr.split_einsumsvd(expr)
        a = self._einsum(einsum_expr, operands)
        return self._einsvd_reduced(einsvd_expr, a, rank)

    def einsumsvd_rand(self, subscripts, *operands, rank, niter=1, oversamp=5):
        if not all(isinstance(operand, self.tensor) for operand in operands):
            raise TypeError('all operands should be {}'.format(self.tensor.__qualname__))
        ndims = [operand.ndim for operand in operands]
        expr = einstr.parse_einsumsvd(subscripts, ndims)
        einsum_expr, einsvd_expr = einstr.split_einsumsvd(expr)
        a = self._einsum(einsum_expr, operands)
        return self._einsvd_rand(einsvd_expr, a, rank, niter, oversamp)

    def isclose(self, a, b, *, rtol=1e-9, atol=0.0):
        if isinstance(a, self.tensor): a.match_indices()
        if isinstance(b, self.tensor): b.match_indices()
        return abs(a - b) <= atol + rtol * abs(b)

    def allclose(self, a, b, *, rtol=1e-9, atol=0.0):
        return self.all(self.isclose(a, b, rtol=rtol, atol=atol))

    def inv(self, a):
        u, s, v = self.einsvd('ij->ia,ja', a)
        return self.einsum('ia,a,ja->ji', u, 1/s, v)

    def svd(self, a):
        if not isinstance(a, self.tensor):
            raise TypeError('the input should be {}'.format(self.tensor.__qualname__))
        if a.ndim != 2:
            raise TypeError('the input tensor should be a matrix')
        u, s, vh = ctf.svd(a.unwrap())
        return self.tensor(u), self.tensor(ctf.real(s)), self.tensor(vh)

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
                wrapped_result.__qualname__ = '{}.{}'.format(type(self).__qualname__, attr)
                return wrapped_result
            else:
                return result
        except Exception as e:
            raise ValueError('failed to get {} from ctf'.format(attr)) from e

    def _einsum(self, expr, operands):
        inputs_indices = [operand.indices for operand in operands]
        inputs_shapes = [operand.tsr.shape for operand in operands]
        expanded_expr = indices_utils.expand_einsum(expr, inputs_indices, inputs_shapes)
        if expanded_expr is not None:
            result = ctf.einsum(expanded_expr.indices_string, *(operand.tsr for operand in operands))
            if isinstance(result, ctf.tensor):
                newshape = expanded_expr.outputs[0].newshape(result.shape)
                return self.tensor(result).reshape(*newshape)
            else:
                return result
        else:
            result = ctf.einsum(expr.indices_string, *(operand.unwrap() for operand in operands))
            if isinstance(result, ctf.tensor):
                newshape = expr.outputs[0].newshape(result.shape)
                return self.tensor(result).reshape(*newshape)
            else:
                return result

    def _einsvd_reduced(self, expr, a, rank):
        expanded_expr = indices_utils.expand_einsvd(expr, a.indices)
        if expanded_expr is None:
            a.match_indices()
        else:
            expr = expanded_expr
        u, s, vh = a.tsr.i(expr.inputs[0].indices_string).svd(
            expr.outputs[0].indices_string,
            expr.outputs[1].indices_string,
            rank=rank,
        )
        u_newshape = expr.outputs[0].newshape(u.shape)
        vh_newshape = expr.outputs[1].newshape(vh.shape)
        return self.tensor(u).reshape(*u_newshape), self.tensor(ctf.real(s)), self.tensor(vh).reshape(*vh_newshape)

    def _einsvd_rand(self, expr, a, rank, niter, oversamp):
        expanded_expr = indices_utils.expand_einsvd(expr, a.indices)
        if expanded_expr is None:
            a.match_indices()
        else:
            expr = expanded_expr
        newindex = (expr.output_indices - expr.input_indices).pop()
        axis_of_index = {index: axis for axis, index in enumerate(expr.inputs[0])}
        u_axes_from_a = [axis_of_index[index] for index in expr.outputs[0] if index != newindex]
        vh_axes_from_a = [axis_of_index[index] for index in expr.outputs[1] if index != newindex]
        # form matrix of a
        a_matrix_axes = [*u_axes_from_a, *vh_axes_from_a]
        a_matrix_shape = (np.prod([a.shape[axis] for axis in u_axes_from_a]), -1)
        a_matrix = a.transpose(*a_matrix_axes).reshape(*a_matrix_shape)
        u, s, vh = self.rsvd(a_matrix, rank, niter, oversamp)
        # form u
        u = u.reshape(*(a.shape[axis] for axis in u_axes_from_a), s.shape[0])
        u = self.moveaxis(u, -1, expr.outputs[0].find(newindex))
        u = u.reshape(*expr.outputs[0].newshape(u.shape))
        # form vh
        vh = vh.reshape(s.shape[0], *(a.shape[axis] for axis in vh_axes_from_a))
        vh = self.moveaxis(vh, 0, expr.outputs[1].find(newindex))
        vh = vh.reshape(*expr.outputs[1].newshape(vh.shape))
        return u, s, vh
