from ..utils.svd_absorb_s import svd_absorb_s

def rsvd(backend, a, rank, niter, oversamp, absorb_s):
    dtype = a.dtype
    m, n = a.shape
    r = min(rank + oversamp, m, n)
    # find subspace
    q = backend.random.uniform(low=-1.0, high=1.0, size=(n, r)).astype(dtype)
    a_H = a.H
    for i in range(niter):
        q = a_H @ (a @ q)
        q, _ = backend.qr(q)
    q = a @ q
    q, _ = backend.qr(q)
    # svd in subspace
    a_sub = q.H @ a
    u_sub, s, vh = backend.svd(a_sub)
    u = q @ u_sub
    if rank < r:
        u, s, vh = u[:,:rank], s[:rank], vh[:rank,:]
    u, s, vh = svd_absorb_s(u, s, vh, absorb_s)
    return u, s, vh
