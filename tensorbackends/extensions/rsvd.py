
def rsvd(backend, a, rank, niter, oversamp):
    dtype = a.dtype
    m, n = a.shape
    r = min(rank + oversamp, m, n)
    # find subspace
    q = backend.random.uniform(low=-1.0, high=1.0, size=(n, r)).astype(dtype)
    a_H = a.H
    a_H_a = a_H @ a
    for i in range(niter):
        q = a_H_a @ q
        q, _ = backend.qr(q)
    q = a @ q
    q, _ = backend.qr(q)
    # svd in subspace
    a_sub = q.H @ a
    u_sub, s, vh = backend.svd(a_sub)
    u = q @ u_sub
    if rank < r:
        u, s, vh = u[:,:rank], s[:rank], vh[:rank,:]
    return u, s, vh
