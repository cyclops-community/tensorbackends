import itertools, functools, operator


def identity(ndim):
    return tuple((i,) for i in range(ndim))

def shape(indices, tsr):
    return tuple(prod(tsr.shape[i] for i in group) for group in indices)

def permute(indices, permutation):
    return tuple(indices[i] for i in permutation)

def flatten(indices):
    return tuple(i for i in itertools.chain(*indices))

def accumulate_group_size(indices):
    return accumulate(len(group) for group in indices)

def apply_transpose(indices, tsr):
    axes = flatten(indices)
    if axes != tuple(range(tsr.ndim)):
        steps = accumulate_group_size(indices)
        indices = tuple(tuple(range(start, end)) for start, end in zip(steps, steps[1:]))
        tsr = tsr.transpose(*axes)
    return indices, tsr

def apply(indices, tsr):
    indices, tsr = apply_transpose(indices, tsr)
    steps = accumulate_group_size(indices)
    newshape = tuple(prod(tsr.shape[start:end]) for start, end in zip(steps, steps[1:]))
    if newshape != tsr.shape:
        tsr = tsr.reshape(*newshape)
    return identity(tsr.ndim), tsr

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

def accumulate(iterable):
    result = [0]
    for n in iterable:
        result.append(result[-1] + n)
    return result
