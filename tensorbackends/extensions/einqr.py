import functools, operator
from ..utils import einstr


def parse_einqr(subscripts, ndim):
    expr = einstr.parse(subscripts).match([ndim])
    if len(expr.inputs) != 1:
        raise ValueError('expect one input for einqr: "{}"'.format(expr.source))
    if len(expr.outputs) != 2:
        raise ValueError('expect two outputs for einqr: "{}"'.format(expr.source))
    if len(set(expr.inputs[0])) != len(expr.inputs[0]):
        raise ValueError('input indices should not repeat for einqr: "{}"'.format(expr.source))
    input_indices, output_indices = expr.input_indices, expr.output_indices
    if not input_indices.issubset(output_indices):
        raise ValueError('expect input indices subset of output indices for einqr: "{}"'.format(expr.source))
    newindices = output_indices - input_indices
    if len(newindices) != 1:
        raise ValueError('expect one new index in outputs for einqr: "{}"'.format(expr.source))
    newindex = newindices.pop()
    if newindex not in expr.outputs[0] or newindex not in expr.outputs[1]:
        raise ValueError('expect new index in both outputs for einqr: "{}"'.format(expr.source))
    if len(expr.outputs[0]) == 1 or len(expr.outputs[1]) == 1:
        raise ValueError('expect outputs to be at least two dimensional for einqr: "{}"'.format(expr.source))
    if len(output_indices) != len(expr.outputs[0]) + len(expr.outputs[1]) - 1:
        raise ValueError('only the new index can repeat in the output for einqr: "{}"'.format(expr.source))
    return expr


def einqr(backend, subscripts, a):
    if not isinstance(a, backend.tensor):
        raise TypeError('the input should be {}'.format(backend.tensor.__qualname__))
    expr = parse_einqr(subscripts, a.ndim)
    newindex = (expr.output_indices - expr.input_indices).pop()
    prod = lambda iterable: functools.reduce(operator.mul, iterable, 1)
    axis_of_index = {index: axis for axis, index in enumerate(expr.inputs[0])}
    q_axes_from_a = [axis_of_index[index] for index in expr.outputs[0] if index != newindex]
    r_axes_from_a = [axis_of_index[index] for index in expr.outputs[1] if index != newindex]
    # form matrix of a
    a_matrix_axes = [*q_axes_from_a, *r_axes_from_a]
    a_matrix_shape = (prod(a.shape[axis] for axis in q_axes_from_a), -1)
    a_matrix = a.transpose(*a_matrix_axes).reshape(*a_matrix_shape)
    q, r = backend.qr(a_matrix)
    # form q
    q = q.reshape(*(a.shape[axis] for axis in q_axes_from_a), -1)
    q = backend.moveaxis(q, -1, expr.outputs[0].find(newindex))
    q = q.reshape(*expr.outputs[0].newshape(q.shape))
    # form r
    r = r.reshape(-1, *(a.shape[axis] for axis in r_axes_from_a))
    r = backend.moveaxis(r, 0, expr.outputs[1].find(newindex))
    r = r.reshape(*expr.outputs[1].newshape(r.shape))
    return q, r
