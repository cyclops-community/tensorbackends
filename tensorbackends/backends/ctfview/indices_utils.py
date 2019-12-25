import itertools, functools, operator

from ...utils import einstr


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

def expand_einsum(expr, inputs_indices, inputs_shapes):
    index_mappping = {}
    shape_of_index = {}
    nindices = 0

    def fresh(n):
        nonlocal nindices
        start, end = nindices, nindices + n
        nindices += n
        return tuple(range(start, end))

    def expand_input_term(term, indices, shape):
        expanded_indices = []
        for idx, axis in zip(term, indices):
            idx_shape = tuple(shape[i] for i in axis)
            if idx in index_mappping and shape_of_index[idx] != idx_shape:
                return None
            elif idx in index_mappping:
                expanded_indices.extend(index_mappping[idx])
            else:
                mapped_index = fresh(len(axis))
                expanded_indices.extend(mapped_index)
                index_mappping[idx] = mapped_index
                shape_of_index[idx] = idx_shape
        new_term_indices = [None] * len(expanded_indices)
        for i, axis in enumerate(flatten(indices)):
            new_term_indices[axis] = expanded_indices[i]
        return einstr.InputTerm(new_term_indices, term.source)

    def expand_output_term(term):
        new_term_indices = []
        new_term_fusing = []
        offset = 0
        def add_term_indices(idx_start, idx_end, fusing=False):
            nonlocal offset
            for j in range(idx_start, idx_end):
                mapped_index = index_mappping[term.indices[j]]
                new_term_indices.extend(mapped_index)
                if len(mapped_index) > 1 and not fusing:
                    new_term_fusing.append((offset, offset + len(mapped_index)))
                offset += len(mapped_index)
        i = 0
        for start, end in term.fusing:
            add_term_indices(i, start)
            new_start = offset
            add_term_indices(start, end, fusing=True)
            new_end = offset
            new_term_fusing.append((new_start, new_end))
            i = end
        add_term_indices(i, len(term))
        return einstr.OutputTerm(new_term_indices, new_term_fusing, term.source)

    newinputs = []
    for term, indices, shape in zip(expr.inputs, inputs_indices, inputs_shapes):
        new_input_term = expand_input_term(term, indices, shape)
        if new_input_term is None: return None
        newinputs.append(new_input_term)
    if nindices > len(einstr.chars): return None
    newoutputs = [expand_output_term(expr.outputs[0])]
    return einstr.Expression(newinputs, newoutputs, nindices, expr.source)

def expand_einsvd(expr, input_indices):
    index_mappping = {}
    nindices = 0

    def fresh(n):
        nonlocal nindices
        start, end = nindices, nindices + n
        nindices += n
        return tuple(range(start, end))

    def expand_input_term(term, indices):
        expanded_indices = []
        for idx, axis in zip(term, indices):
            assert idx not in index_mappping
            mapped_index = fresh(len(axis))
            expanded_indices.extend(mapped_index)
            index_mappping[idx] = mapped_index
        new_term_indices = [None] * len(expanded_indices)
        for i, axis in enumerate(flatten(indices)):
            new_term_indices[axis] = expanded_indices[i]
        return einstr.InputTerm(new_term_indices, term.source)

    def expand_output_term(term):
        new_term_indices = []
        new_term_fusing = []
        offset = 0
        def add_term_indices(idx_start, idx_end, fusing=False):
            nonlocal offset
            for j in range(idx_start, idx_end):
                if term.indices[j] in index_mappping:
                    mapped_index = index_mappping[term.indices[j]]
                    new_term_indices.extend(mapped_index)
                    if len(mapped_index) > 1 and not fusing:
                        new_term_fusing.append((offset, offset + len(mapped_index)))
                    offset += len(mapped_index)
                else:
                    mapped_index = fresh(1)
                    new_term_indices.extend(mapped_index)
                    index_mappping[term.indices[j]] = mapped_index
        i = 0
        for start, end in term.fusing:
            add_term_indices(i, start)
            new_start = offset
            add_term_indices(start, end, fusing=True)
            new_end = offset
            new_term_fusing.append((new_start, new_end))
            i = end
        add_term_indices(i, len(term))
        return einstr.OutputTerm(new_term_indices, new_term_fusing, term.source)

    newinputs = [expand_input_term(expr.inputs[0], input_indices)]
    if nindices > len(einstr.chars): return None
    newoutputs = [expand_output_term(term) for term in expr.outputs]
    return einstr.Expression(newinputs, newoutputs, nindices, expr.source)

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

def accumulate(iterable):
    result = [0]
    for n in iterable:
        result.append(result[-1] + n)
    return result
