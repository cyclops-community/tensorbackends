"""
This module implements the random module for ctfview backend.
"""

import ctf

from ...interface import Random
from .ctfview_tensor import CTFViewTensor


class CTFViewRandom(Random):
    def seed(self, seed):
        ctf.random.seed(seed)

    def random(self, size):
        if size is None:
            return ctf.random.random(1)[0]
        else:
            return CTFViewTensor(ctf.random.random(size))

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            tsr = ctf.empty(1)
            tsr.fill_random(low, high)
            return tsr[0]
        else:
            tsr = ctf.empty(size)
            tsr.fill_random(low, high)
            return CTFViewTensor(tsr)
