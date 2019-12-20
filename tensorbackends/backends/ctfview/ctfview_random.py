"""
This module implements the random module for ctfview backend.
"""

import ctf

from ...interface import Random
from .ctfview_tensor import CTFViewTensor


class CTFViewRandom(Random):
    def seed(self, seed):
        ctf.random.seed(seed)

    def rand(self, *dims):
        return CTFViewTensor(ctf.random.random(dims))
