"""
This module implements the random module for ctf backend.
"""

import ctf

from ...interface import Random
from .ctf_tensor import CTFTensor


class CTFRandom(Random):
    def seed(self, seed):
        ctf.random.seed(seed)

    def rand(self, *dims):
        return CTFTensor(ctf.random.random(dims))
