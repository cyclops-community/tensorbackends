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

    def test_hstack(self, tb):
        a = tb.zeros((2,3))
        b = tb.zeros((2,3))
        c = tb.hstack((a,b))
        self.assertIsInstance(c, tb.tensor)
        self.assertEqual(c.shape, (2,6))

    def test_vstack(self, tb):
        a = tb.zeros((2,3))
        b = tb.zeros((2,3))
        c = tb.vstack((a,b))
        self.assertIsInstance(c, tb.tensor)
        self.assertEqual(c.shape, (4,3))

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


    def test_einsvd_options(self, tb):
        from tensorbackends.interface import ReducedSVD, RandomizedSVD
        a = tb.astensor([[1e-3,0,0,0],[0,2e-3j,0,0],[0,0,3,0],[0,0,0,4j]], dtype=complex).reshape(2,2,2,2)
        s_true = tb.astensor([4,3])
        low_rank = tb.astensor([[0,0,0,0],[0,0,0,0],[0,0,3,0],[0,0,0,4j]], dtype=complex)
        for option in [ReducedSVD(rank=2), RandomizedSVD(rank=2, niter=2, oversamp=1)]:
            with self.subTest(option=option):
                u, s, v = tb.einsvd('ijkl->(ij)s,s(kl)', a, option=option)
                usv = tb.einsum('is,s,sk->ik', u, s, v)
                self.assertEqual(u.shape, (4,2))
                self.assertEqual(s.shape, (2,))
                self.assertEqual(v.shape, (2,4))
                self.assertTrue(tb.allclose(s, s_true))
                self.assertTrue(tb.allclose(usv, low_rank, atol=1e-9))


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


    def test_einsumsvd_options(self, tb):
        from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD
        a = tb.astensor([[0,2e-3j,0,0],[1e-3,0,0,0],[0,0,3,0],[0,0,0,4j]], dtype=complex)
        p = tb.astensor([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]], dtype=complex)
        s_true = tb.astensor([4,3])
        low_rank = tb.astensor([[0,0,0,0],[0,0,0,0],[0,0,3,0],[0,0,0,4j]], dtype=complex)
        options = [
            ReducedSVD(rank=2),
            RandomizedSVD(rank=2, niter=2, oversamp=1),
            ImplicitRandomizedSVD(rank=2, niter=2, orth_method='qr'),
            ImplicitRandomizedSVD(rank=2, niter=2, orth_method='local_gram'),
        ]
        for option in options:
            with self.subTest(option=option):
                u, s, v = tb.einsumsvd('ij,jk->is,sk', p, a, option=option)
                usv = tb.einsum('is,s,sk->ik', u, s, v)
                self.assertEqual(u.shape, (4,2))
                self.assertEqual(s.shape, (2,))
                self.assertEqual(v.shape, (2,4))
                self.assertTrue(tb.allclose(s, s_true))
                self.assertTrue(tb.allclose(usv, low_rank, atol=1e-9))


    def test_einsumsvd_absorb_s(self, tb):
        from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD
        a = tb.astensor([[0,2e-3j,0,0],[1e-3,0,0,0],[0,0,3,0],[0,0,0,4j]], dtype=complex)
        p = tb.astensor([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]], dtype=complex)
        s_true = tb.astensor([4,3])
        low_rank = tb.astensor([[0,0,0,0],[0,0,0,0],[0,0,3,0],[0,0,0,4j]], dtype=complex)
        options = [
            ReducedSVD(rank=2),
            RandomizedSVD(rank=2, niter=2, oversamp=1),
            ImplicitRandomizedSVD(rank=2, niter=2, orth_method='qr'),
            ImplicitRandomizedSVD(rank=2, niter=2, orth_method='local_gram'),
        ]
        for option in options:
            for absorb_s in ['even', 'u', 'v']:
                with self.subTest(option=option):
                    u, _, v = tb.einsumsvd('ij,jk->is,sk', p, a, option=option, absorb_s=absorb_s)
                    usv = tb.einsum('is,sk->ik', u, v)
                    self.assertEqual(u.shape, (4,2))
                    self.assertEqual(v.shape, (2,4))
                    self.assertTrue(tb.allclose(usv, low_rank, atol=1e-9))


    def test_einsumsvd_rand(self, tb):
        tb.random.seed(42)
        A1 = tb.random.random((2,3)) + tb.random.random((2,3)) * 1j
        A2 = tb.random.random((5,2,3)) + tb.random.random((5,2,3)) * 1j
        A3 = tb.random.random((3,2,3)) + tb.random.random((3,2,3)) * 1j
        A = tb.einsum("ij,mnk,kpq->impjnq", A1, A2, A3)
        u, s, v = tb.einsumsvd_rand('ij,mnk,kpq->(imp)y,y(jnq)', A1, A2, A3, rank=8, niter=8)
        mu, ms, mv = tb.svd(A.reshape(20, 18))
        usv = tb.einsum('is,s,sj->ij', u[:,:4], s[:4], v[:4,:])
        musv = tb.einsum('is,s,sj->ij', mu[:,:4], ms[:4], mv[:4:])
        self.assertTrue(tb.allclose(s[:4], ms[:4]))
        self.assertTrue(tb.allclose(usv, musv))


    def test_einsumsvd_implicit_rand(self, tb):
        tb.random.seed(42)
        A1 = tb.random.random((2,3)) + tb.random.random((2,3)) * 1j
        A2 = tb.random.random((5,2,3)) + tb.random.random((5,2,3)) * 1j
        A3 = tb.random.random((3,2,3)) + tb.random.random((3,2,3)) * 1j
        A = tb.einsum("ij,mnk,kpq->impjnq", A1, A2, A3)
        u, s, v = tb.einsumsvd_implicit_rand('ij,mnk,kpq->(imp)y,y(jnq)', A1, A2, A3, rank=8, niter=8)
        mu, ms, mv = tb.svd(A.reshape(20, 18))
        usv = tb.einsum('is,s,sj->ij', u[:,:4], s[:4], v[:4,:])
        musv = tb.einsum('is,s,sj->ij', mu[:,:4], ms[:4], mv[:4:])
        self.assertTrue(tb.allclose(s[:4], ms[:4]))
        self.assertTrue(tb.allclose(usv, musv))


    def test_inv(self, tb):
        a = tb.astensor([[1,2],[3,4]], dtype=float)
        b = tb.inv(a)
        self.assertTrue(tb.allclose(a @ b, tb.eye(2), atol=1e-8))


    def test_rsvd(self, tb):
        a = tb.astensor([[1,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,20]], dtype=float)
        u, s, vh = tb.rsvd(a, rank=2, niter=4, oversamp=1)
        self.assertEqual(u.shape, (4,2))
        self.assertEqual(s.shape, (2,))
        self.assertEqual(vh.shape, (2,4))
        s_true = tb.astensor([20, 10])
        self.assertTrue(tb.allclose(s, s_true))
