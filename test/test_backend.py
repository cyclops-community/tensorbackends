import unittest

import tensorbackends as tbs
from tensorbackends.utils import test_with_backend


@test_with_backend()
class BackendTest(unittest.TestCase):
    def test_astensor(self, tb):
        a = tb.astensor([[1,2,3],[4,5,6]])
        self.assertEqual(a.shape, (2,3))

    def test_empty(self, tb):
        for dtype in [int, float, complex]:
            a = tb.empty((2,3), dtype=dtype)
            self.assertIsInstance(a, tb.tensor)
            self.assertEqual(a.shape, (2,3))

    def test_zeros(self, tb):
        for dtype in [int, float, complex]:
            a = tb.zeros((2,3), dtype=dtype)
            self.assertIsInstance(a, tb.tensor)
            self.assertEqual(a.shape, (2,3))

    def test_ones(self, tb):
        for dtype in [int, float, complex]:
            a = tb.ones((2,3), dtype=dtype)
            self.assertIsInstance(a, tb.tensor)
            self.assertEqual(a.shape, (2,3))
