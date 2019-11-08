import unittest

import tensorbackends as tbs
from tensorbackends.utils import test_with_backend


@test_with_backend()
class BackendTest(unittest.TestCase):
    def test_astensor(self, tb):
        a = tb.astensor([[1,2,3],[4,5,6]])
        b = tb.astensor([[1,2,3],[4,5,6]])
        self.assertTrue(tb.allclose(a, b))

    def test_empty(self, tb):
        for dtype in [int, float, complex]:
            with self.subTest(dtype=dtype):
                a = tb.empty((2,3), dtype=dtype)
                self.assertIsInstance(a, tb.tensor)
                self.assertEqual(a.shape, (2,3))

    def test_zeros(self, tb):
        for dtype in [int, float, complex]:
            with self.subTest(dtype=dtype):
                a = tb.zeros((2,3), dtype=dtype)
                self.assertIsInstance(a, tb.tensor)
                self.assertEqual(a.shape, (2,3))
                self.assertTrue(tb.allclose(a, 0))

    def test_ones(self, tb):
        for dtype in [int, float, complex]:
            with self.subTest(dtype=dtype):
                a = tb.ones((2,3), dtype=dtype)
                self.assertIsInstance(a, tb.tensor)
                self.assertEqual(a.shape, (2,3))
                self.assertTrue(tb.allclose(a, 1))
