"""
This module implements the ctfview tensor.
"""

import itertools
import operator

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
        num_minus_one = newshape.count(-1)
        if num_minus_one > 1:
            raise ValueError('At most one -1 can appear in a new shape')
        elif num_minus_one == 1:
            newshape[newshape.index(-1)] = self.size // (-indices_utils.prod(newshape))
        if self.size != indices_utils.prod(newshape):
            raise ValueError(f'Cannot reshape tensor of size {self.size} into shape {newshape}')
        axes = indices_utils.flatten(self.indices)
        oldshape = tuple(self.tsr.shape[axis] for axis in axes)
        old_prefix_sizes = [1, *itertools.accumulate(oldshape, func=operator.mul)]
        new_prefix_sizes = [1, *itertools.accumulate(newshape, func=operator.mul)]
        steps = [0]
        need_true_reshape = False
        old_axis, new_axis = 0, 0
        while True:
            if old_prefix_sizes[old_axis] == new_prefix_sizes[new_axis]:
                new_axis += 1
                old_axis += 1
                if old_axis == len(old_prefix_sizes) and new_axis == len(new_prefix_sizes):
                    break # done
                steps.append(steps[-1] + 1) # move to next new axis
            elif old_prefix_sizes[old_axis] < new_prefix_sizes[new_axis]:
                steps[-1] += 1 # fuse this old axis
                old_axis += 1
            else: # old_prefix_sizes[old_axis] > new_prefix_sizes[new_axis]
                need_true_reshape = True
                break
        if need_true_reshape:
            self.match_axes()
            return CTFViewTensor(self.tsr.reshape(*newshape))
        else:
            newindices = tuple(
                tuple(axes[start:end]) for start, end in zip(steps, steps[1:])
            )
            return CTFViewTensor(self.tsr, newindices)

    def transpose(self, *axes):
        if len(axes) != self.ndim:
            raise ValueError(f'Axes number do not match ndim: {len(axes)} != {self.ndim}')
        return CTFViewTensor(self.tsr, indices_utils.permute(self.indices, axes))

    def match_indices(self):
        self.indices, self.tsr = indices_utils.apply(self.indices, self.tsr)

    def match_axes(self):
        self.indices, self.tsr = indices_utils.apply_transpose(self.indices, self.tsr)


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
