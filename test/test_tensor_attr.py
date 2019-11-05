import unittest
import operator
from functools import reduce

from tensorbackends.utils import test_with_backend


@test_with_backend(['numpy'], optional=['ctf'])
class TensorAttrTest(unittest.TestCase):
    def test_shape(self, tb):
        for shape in [(2,), (2,3), (2,1,3)]:
            a = tb.empty(shape)
            self.assertEqual(a.shape, shape)

    def test_ndim(self, tb):
        for ndim in [1, 2, 3]:
            a = tb.empty((2,)*ndim)
            self.assertEqual(a.ndim, ndim)

    def test_size(self, tb):
        for shape in [(2,), (2,3), (2,1,3)]:
            a = tb.empty(shape)
            self.assertEqual(a.size, reduce(operator.mul, shape))
