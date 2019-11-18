"""
This module implements einstr utilities.
"""

import re, string, itertools, functools, operator
from collections import namedtuple


chars = string.ascii_letters + string.digits


def parse(subscripts):
    return Expression.parse(subscripts)


class Expression:
    def __init__(self, inputs, outputs, nindices, source):
        self.inputs = inputs
        self.outputs = outputs
        self.nindices = nindices
        self.source = source

    @staticmethod
    def parse(subscripts):
        subscripts = re.sub(r'\s', '', subscripts)
        inputs_outputs = subscripts.split('->')
        if len(inputs_outputs) != 2:
            raise ValueError("Invalid subscripts: '{}'".format(subscripts))
        mapping = {}
        input_subscripts = inputs_outputs[0].split(',')
        output_subscripts = inputs_outputs[1].split(',')
        inputs = [InputTerm.parse(s, mapping) for s in input_subscripts]
        outputs = [OutputTerm.parse(s, mapping) for s in output_subscripts]
        if len(mapping) > len(chars):
            raise ValueError('Too many indices: {} (maximum {})'.format(len(mapping), len(chars)))
        return Expression(inputs, outputs, len(mapping), subscripts)

    def match(self, ndims):
        if len(ndims) != len(self.inputs):
            raise ValueError("Number of operands does not match subscripts '{}': {}".format(self.source, len(ndims)))
        nindices = self.nindices
        def fresh():
            nonlocal nindices
            newindex = nindices
            nindices += 1
            return newindex
        newinputs = [t.match(ndim, fresh) for t, ndim in zip(self.inputs, ndims)]
        ellipsis = list(range(self.nindices, nindices))
        newoutputs = [t.expand(ellipsis) for t in self.outputs]
        return Expression(newinputs, newoutputs, nindices, self.source)

    def __str__(self):
        inputs = ','.join(str(t) for t in self.inputs)
        outputs = ','.join(str(t) for t in self.outputs)
        return '{}->{}'.format(inputs, outputs)

    def __repr__(self):
        return "Expression('{}')".format(str(self))


class InputTerm:
    def __init__(self, indices, source):
        self.indices = indices
        self.source = source

    @property
    def indices_string(self):
        return ''.join('...' if idx is Ellipsis else chars[idx] for idx in self.indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        yield from self.indices

    @staticmethod
    def parse(subscripts, mapping):
        indices = []
        found_ellipsis = False
        i = 0
        while i < len(subscripts):
            if subscripts[i:].startswith('...'):
                if found_ellipsis:
                    raise ValueError('Each term can contain at most one ellipsis')
                found_ellipsis = True
                indices.append(Ellipsis)
                i += 3
            elif subscripts[i] in '()':
                raise ValueError("Indices fusing is not allowed in input subscripts: '{}'".format(subscripts))
            else:
                indices.append(mapping.setdefault(subscripts[i], len(mapping)))
                i += 1
        return InputTerm(indices, subscripts)

    def match(self, ndim, fresh):
        newindices = []
        i = 0
        for j, idx in enumerate(self.indices):
            if idx is Ellipsis:
                count = (ndim - i) - (len(self) - j - 1)
                if count < 0:
                    raise ValueError("Indices '{}' do not match ndim: {}".format(self.source, ndim))
                newindices.extend(fresh() for _ in range(count))
                i += count
            else:
                newindices.append(idx)
                i += 1
        if i != ndim:
            raise ValueError("Indices '{}' do not match ndim: {}".format(self.source, ndim))
        return InputTerm(newindices, self.source)

    def __str__(self):
        return ''.join('...' if idx is Ellipsis else chars[idx] for idx in self.indices)

    def __repr__(self):
        return "InputTerm('{}')".format(str(self))


class OutputTerm:
    def __init__(self, indices, fusing, source):
        self.indices = indices
        self.fusing = fusing
        self.source = source

    @property
    def indices_string(self):
        return ''.join('...' if idx is Ellipsis else chars[idx] for idx in self.indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        yield from self.indices

    @staticmethod
    def parse(subscripts, mapping):
        indices = []
        fusing = []
        i = 0
        found_ellipsis = False
        start = None
        while i < len(subscripts):
            if subscripts[i:].startswith('...'):
                if found_ellipsis:
                    raise ValueError('Each term can contain at most one ellipsis')
                found_ellipsis = True
                indices.append(Ellipsis)
                i += 3
            elif subscripts[i] == '(':
                if start is not None:
                    raise ValueError("Nested parentheses are not allowed: '{}'".format(subscripts))
                start = len(indices)
                i += 1
            elif subscripts[i] == ')':
                if start is None:
                    raise ValueError("Unmatched parentheses: '{}'".format(subscripts))
                end = len(indices)
                if end > start + 1: fusing.append((start, len(indices)))
                start = None
                i += 1
            else:
                indices.append(mapping.setdefault(subscripts[i], len(mapping)))
                i += 1
        if start is not None:
            raise ValueError("Unmatched parentheses: '{}'".format(subscripts))
        return OutputTerm(indices, fusing, subscripts)

    def expand(self, ellipsis):
        newindices = []
        ellipsis_position = -1
        for i, idx in enumerate(self.indices):
            if idx is Ellipsis:
                newindices.extend(ellipsis)
                ellipsis_position = i
            else:
                newindices.append(idx)
        if ellipsis and ellipsis_position < 0:
            raise ValueError("Expect ellipsis in output subscripts: '{}'".format(self.source))
        pad = lambda j: (j + len(ellipsis) - 1) if j > ellipsis_position else j
        newfusing = [(pad(start), pad(end)) for start, end in self.fusing]
        return OutputTerm(newindices, newfusing, self.source)

    def newshape(self, shape):
        if Ellipsis in self.indices:
            raise ValueError('Ellipsis not expanded')
        if len(shape) != len(self):
            raise ValueError("Indices '{}' do not match shape: {}".format(str(self), shape))
        newshape = []
        i = 0
        for start, end in self.fusing:
            newshape.extend(shape[i:start])
            newshape.append(functools.reduce(operator.mul, shape[start:end], 1))
            i = end
        newshape.extend(shape[i:])
        return tuple(newshape)

    def __str__(self):
        result = []
        asstr = lambda idx: '...' if idx is Ellipsis else chars[idx]
        i = 0
        for start, end in self.fusing:
            result.extend(asstr(idx) for idx in self.indices[i:start])
            result.append('(')
            result.extend(asstr(idx) for idx in self.indices[start:end])
            result.append(')')
            i = end
        result.extend(asstr(idx) for idx in self.indices[i:])
        return ''.join(result)

    def __repr__(self):
        return "OutputTerm('{}')".format(str(self))
