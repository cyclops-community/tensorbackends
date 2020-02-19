"""
This module defines the option objects used in the backend interface.
"""


class Option:
    def __str__(self):
        return '{}({})'.format(
            type(self).__name__,
            ','.join('{}={}'.format(k, v) for k, v in vars(self).items())
        )

    def __repr__(self):
        return str(self)


class ReducedSVD(Option):
    def __init__(self, rank=None):
        self.rank = rank

class RandomizedSVD(Option):
    def __init__(self, rank, niter=1, oversamp=5):
        self.rank = rank
        self.niter = niter
        self.oversamp = oversamp

class ImplicitRandomizedSVD(Option):
    def __init__(self, rank, niter=1):
        self.rank = rank
        self.niter = niter
