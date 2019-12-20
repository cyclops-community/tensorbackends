import unittest

import tensorbackends as tbs
from tensorbackends.utils import test_with_backend


@test_with_backend()
class BackendTest(unittest.TestCase):
    def test_random(self, tb):
        self.assertIsInstance(tb.random, tbs.interface.Random)

    def test_tensor(self, tb):
        self.assertTrue(issubclass(tb.tensor, tbs.interface.Tensor))

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
                self.assertEqual(a.dtype, dtype)

    def test_zeros(self, tb):
        for dtype in [int, float, complex]:
            with self.subTest(dtype=dtype):
                a = tb.zeros((2,3), dtype=dtype)
                self.assertIsInstance(a, tb.tensor)
                self.assertEqual(a.shape, (2,3))
                self.assertTrue(tb.allclose(a, 0))
                self.assertEqual(a.dtype, dtype)

    def test_ones(self, tb):
        for dtype in [int, float, complex]:
            with self.subTest(dtype=dtype):
                a = tb.ones((2,3), dtype=dtype)
                self.assertIsInstance(a, tb.tensor)
                self.assertEqual(a.shape, (2,3))
                self.assertTrue(tb.allclose(a, 1))
                self.assertEqual(a.dtype, dtype)

    def test_einsum(self, tb):
        a = tb.astensor([1,2,3,4,5,6]).reshape(1,3,2)
        b = tb.astensor([1,2,3,4,5,6]).reshape(2,3,1,1)
        c = tb.astensor([9,12,15,19,26,33,29,40,51])

        w1 = tb.einsum('ijk,klmn->ijlmn', a, b)
        w2 = tb.einsum('...k,k...->...', a, b)
        self.assertTrue(tb.allclose(w1, c.reshape(1,3,3,1,1)))
        self.assertTrue(tb.allclose(w2, c.reshape(1,3,3,1,1)))

        x1 = tb.einsum('ijk,klmi->ijlm', a, b)
        x2 = tb.einsum('i...k,k...i->i...', a, b)
        self.assertTrue(tb.allclose(x1, c.reshape(1,3,3,1)))
        self.assertTrue(tb.allclose(x2, c.reshape(1,3,3,1)))

        y1 = tb.einsum('ijk,klmi->(ijlm)', a, b)
        y2 = tb.einsum('i...k,k...i->(i...)', a, b)
        y3 = tb.einsum('...k,k...->(...)', a, b)
        self.assertTrue(tb.allclose(y1, c.reshape(-1)))
        self.assertTrue(tb.allclose(y2, c.reshape(-1)))
        self.assertTrue(tb.allclose(y3, c.reshape(-1)))

        z1 = tb.einsum('ijk,klmi->(i)(jlm)', a, b)
        z2 = tb.einsum('i...k,k...i->i(...)', a, b)
        z3 = tb.einsum('i...k,k...i->(i)(...)', a, b)
        z4 = tb.einsum('i...k,k...i->()(i...)', a, b)
        self.assertTrue(tb.allclose(z1, c.reshape(1,9)))
        self.assertTrue(tb.allclose(z2, c.reshape(1,9)))
        self.assertTrue(tb.allclose(z3, c.reshape(1,9)))
        self.assertTrue(tb.allclose(z4, c.reshape(1,9)))


    def test_einsvd(self, tb):
        a = tb.astensor([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]], dtype=float).reshape(2,2,2,2)
        u, s, v = tb.einsvd('ijkl->(ij)s,s(kl)', a)
        self.assertEqual(u.shape, (4,4))
        self.assertEqual(s.shape, (4,))
        self.assertEqual(v.shape, (4,4))
        u_true = tb.astensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        s_true = tb.astensor([4,3,2,1])
        v_true = tb.astensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        self.assertTrue(tb.allclose(u, u_true))
        self.assertTrue(tb.allclose(s, s_true))
        self.assertTrue(tb.allclose(v, v_true))


    def test_einsumsvd(self, tb):
        a = tb.astensor([[0,2,0,0],[1,0,0,0],[0,0,3,0],[0,0,0,4]], dtype=float)
        p = tb.astensor([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        u, s, v = tb.einsumsvd('ij,jk->is,sk', p, a)
        self.assertEqual(u.shape, (4,4))
        self.assertEqual(s.shape, (4,))
        self.assertEqual(v.shape, (4,4))
        u_true = tb.astensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        s_true = tb.astensor([4,3,2,1])
        v_true = tb.astensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        self.assertTrue(tb.allclose(u, u_true))
        self.assertTrue(tb.allclose(s, s_true))
        self.assertTrue(tb.allclose(v, v_true))
