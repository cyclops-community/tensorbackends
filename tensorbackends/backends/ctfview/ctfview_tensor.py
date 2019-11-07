"""
This module implements the ctfview tensor.
"""

import ctf

from ...interface import Tensor
from . import indices_utils


class CTFViewTensor(Tensor):
    def __init__(self, tsr, indices=None):
        self.tsr = tsr
        self.indices = indices_utils.identity(tsr.ndim) if indices is None else indices

    @property
    def shape(self):
        return indices_utils.shape(self.indices, self.tsr)

    @property
    def ndim(self):
        return len(self.indices)

    @property
    def size(self):
        return self.tsr.size

    @property
    def dtype(self):
        return self.tsr.dtype

    def __repr__(self):
        self.match_indices()
        return repr(self.tsr)

    def __str__(self):
        self.match_indices()
        return str(self.tsr)

    def __getitem__(self, key):
        self.match_indices()
        return self.tsr[key]

    def __setitem__(self, key, value):
        self.match_indices()
        self.tsr[key] = value

    def copy(self):
        return CTFViewTensor(self.tsr.copy(), self.indices)

    def astype(self, dtype):
        return CTFViewTensor(self.tsr.astype(dtype), self.indices)

    def reshape(self, *newshape):
        if newshape.count(-1) > 1:
            raise ValueError('At most one -1 can appear in a new shape')
        newshape = tuple(s if s != -1 else self.size // -indices_utils.prod(newshape) for s in newshape)
        if self.size != indices_utils.prod(newshape):
            raise ValueError(f'Cannot reshape tensor of size {self.size} into shape {newshape}')
        axes = indices_utils.flatten(self.indices)
        oldshape = tuple(self.tsr.shape[axis] for axis in axes)
        need_true_reshape = False
        groups = []
        start, end = 0, None
        for s in newshape:
            end = start
            while indices_utils.prod(oldshape[start:end]) < s:
                end += 1
            if indices_utils.prod(oldshape[start:end]) > s:
                need_true_reshape = True
                break
            groups.append(tuple(axes[start:end]))
            start = end
        if need_true_reshape:
            self.match_axes()
            return CTFViewTensor(self.tsr.reshape(*newshape))
        else:
            return CTFViewTensor(self.tsr, tuple(groups))

    def transpose(self, *axes):
        if len(axes) != self.ndim:
            raise ValueError(f'Axes number do not match ndim: {len(axes)} != {self.ndim}')
        return CTFViewTensor(self.tsr, indices_utils.permute(self.indices, axes))

    def write(self, inds, vals):
        self.match_indices()
        self.tsr.write(inds, vals)

    def match_indices(self):
        self.indices, self.tsr = indices_utils.apply(self.indices, self.tsr)

    def match_axes(self):
        self.indices, self.tsr = indices_utils.apply_transpose(self.indices, self.tsr)

    def __getattr__(self, attr):
        def extract(vtsr):
            vtsr.match_indices()
            return vtsr.tsr
        wrap = lambda val: CTFViewTensor(val) if isinstance(val, ctf.tensor) else val
        unwrap = lambda val: extract(val)if isinstance(val, CTFViewTensor) else val
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
            raise ValueError(f'Failed to get {attr} from ctf') from e


def add_unary_operators(*operator_names):
    def add_unary_operator(operator_name):
        def method(self):
            self.match_indices()
            return CTFViewTensor(getattr(self.tsr, operator_name)())
        method.__module__ = CTFViewTensor.__module__
        method.__qualname__ = f'{CTFViewTensor.__qualname__}.{operator_name}'
        method.__name__ = operator_name
        setattr(CTFViewTensor, operator_name, method)
    for op_name in operator_names:
        add_unary_operator(op_name)


def add_binary_operators(*operator_names):
    def add_binary_operator(operator_name):
        def method(self, other):
            self.match_indices()
            if isinstance(other, CTFViewTensor):
                other.match_indices()
                return CTFViewTensor(getattr(self.tsr, operator_name)(other.tsr))
            else:
                return CTFViewTensor(getattr(self.tsr, operator_name)(other))
        method.__module__ = CTFViewTensor.__module__
        method.__qualname__ = f'{CTFViewTensor.__qualname__}.{operator_name}'
        method.__name__ = operator_name
        setattr(CTFViewTensor, operator_name, method)
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
