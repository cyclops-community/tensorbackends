import numpy as np

def moveaxis(backend, a, source, destination):
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]
    if len(source) != len(destination):
        raise ValueError('lengths of `source` and `destination` should be same')
    ndim = a.ndim
    normalize = lambda n: a.ndim + n if n < 0 else n
    source = [normalize(src) for src in source]
    destination = [normalize(dest) for dest in destination]
    source_set, destination_set = set(source), set(destination)
    other_source = [i for i in range(ndim) if i not in source_set]
    other_destination = [i for i in range(ndim) if i not in destination_set]
    axes = np.empty(ndim, dtype=int)
    axes[destination] = source
    axes[other_destination] = other_source
    return a.transpose(*axes)
