
def svd_absorb_s(u, s, vh, absorb_s):
    if not absorb_s:
        return u, s, vh
    elif absorb_s == 'even':
        s **= 0.5
        u = u.backend.einsum('ij,j->ij', u, s)
        vh = vh.backend.einsum('j,jk->jk', s, vh)
        return u, s, vh
    elif absorb_s == 'u':
        u = u.backend.einsum('ij,j->ij', u, s)
        return u, s, vh
    elif absorb_s == 'v':
        vh = vh.backend.einsum('j,jk->jk', s, vh)
        return u, s, vh
    else:
        raise ValueError('invalid `absorb_s` option: {}'.format(absorb_s))


def svd_absorb_s_ctf(u, s, vh, absorb_s, u_str, vh_str):
    if not absorb_s:
        return u, s, vh
    elif absorb_s == 'even':
        s **= 0.5
        s_str, = set(u_str) & set(vh_str)
        u = u.backend.einsum(f'{u_str},{s_str}->{u_str}', u, s)
        vh = vh.backend.einsum(f'{s_str},{vh_str}->{vh_str}', s, vh)
        return u, s, vh
    elif absorb_s == 'u':
        s_str, = set(u_str) & set(vh_str)
        u = u.backend.einsum(f'{u_str},{s_str}->{u_str}', u, s)
        return u, s, vh
    elif absorb_s == 'v':
        s_str, = set(u_str) & set(vh_str)
        vh = vh.backend.einsum(f'{s_str},{vh_str}->{vh_str}', s, vh)
        return u, s, vh
    else:
        raise ValueError('invalid `absorb_s` option: {}'.format(absorb_s))
