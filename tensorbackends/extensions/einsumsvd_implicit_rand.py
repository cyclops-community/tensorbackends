from ..utils import einstr
from string import ascii_letters as chars
import numpy as np
import scipy.linalg as la


def einsumsvd_implicit_rand(backend, subscripts, *operands, rank, niter, orth_method, absorb_s):
    if orth_method == 'qr':
        orthogonalize = orthogonalize_qr
    elif orth_method == 'local_gram':
        orthogonalize = orthogonalize_local_gram
    else:
        raise ValueError('unknown orthogonalization method: {}'.format(orth_method))

    ndims = [operand.ndim for operand in operands]
    expr = einstr.parse_einsumsvd(subscripts, ndims)
    expr_A, einsvd_expr = einstr.split_einsumsvd(expr)

    shape_U = get_shape(expr_A, operands, einsvd_expr.outputs[0])
    shape_VT = get_shape(expr_A, operands, einsvd_expr.outputs[1])
    m = np.prod([d for d in shape_U if d != 0])
    n = np.prod([d for d in shape_VT if d != 0])
    r = min(rank, m, n) # + oversamp if oversampling, but then need to extract

    ops_A = operands
    ops_A_conj = [tsr.conj() for tsr in ops_A]
    shape_X = []
    need_transpose_X = False
    permutation_X = []
    idx_X = [None]*len(shape_U)
    for i in range(len(shape_U)):
        if shape_U[i] == 0:
            shape_U[i] = r
            if i < len(shape_U)-1:
                need_transpose_X = True
            permutation_X.append(len(shape_U)-1)
        else:
            shape_X.append(shape_U[i])
            if need_transpose_X:
                permutation_X.append(i-1)
            else:
                permutation_X.append(i)
        idx_X[permutation_X[i]] = einsvd_expr.outputs[0].indices[i]
    term_X = einstr.InputTerm(idx_X, '')
    shape_X.append(r)

    shape_YT = []
    need_transpose_VT = False
    permutation_VT = []
    idx_YT = [None]*len(shape_VT)
    for i in range(len(shape_VT)):
        if shape_VT[i] == 0:
            shape_VT[i] = r
            if i < len(shape_VT)-1:
                need_transpose_VT = True
            permutation_VT.append(len(shape_VT)-1)
        else:
            shape_YT.append(shape_VT[i])
            if need_transpose_VT:
                permutation_VT.append(i-1)
            else:
                permutation_VT.append(i)
        idx_YT[permutation_VT[i]] = einsvd_expr.outputs[1].indices[i]
    term_YT = einstr.InputTerm(idx_YT, '')
    shape_YT.append(r)

    op_X = backend.random.uniform(low=-1.0, high=1.0, size=shape_X)
    # FIXME: start by QR of op_X if rank is not too large
    for iter in range(niter):
        op_YT = apply_A(backend,expr_A,ops_A_conj,term_X,op_X,term_YT)
        op_X = apply_A(backend,expr_A,ops_A,term_YT,op_YT,term_X)
        op_X = orthogonalize(backend, op_X)
    op_YT = apply_A(backend,expr_A,ops_A_conj,term_X,op_X,term_YT)
    op_YT = orthogonalize(backend, op_YT)

    op_X = apply_A(backend,expr_A,ops_A,term_YT,op_YT,term_X)
    mat_U, S, mat_XVT = backend.svd(op_X.reshape(np.prod(op_X.shape)//r, r), absorb_s=absorb_s)
    op_YT = backend.tensordot(op_YT.conj(), mat_XVT, axes=((-1),(-1)))
    op_X = mat_U.reshape(*op_X.shape)
    U = op_X

    if need_transpose_X:
        U = backend.einsum(term_X.indices_string+'->'+einsvd_expr.outputs[0].indices_string,U)
    VT = op_YT
    if need_transpose_VT:
        VT = backend.einsum(term_YT.indices_string+'->'+einsvd_expr.outputs[1].indices_string,VT)
    U_newshape = einsvd_expr.outputs[0].newshape(U.shape)
    if U_newshape != U.shape:
        U = U.reshape(*U_newshape)
    VT_newshape = einsvd_expr.outputs[1].newshape(VT.shape)
    if VT_newshape != VT.shape:
        VT = VT.reshape(*VT_newshape)

    return U, S, VT


def apply_A(backend, expr_A, ops_A, expr_X, op_X, expr_Y):
    exp = einstr.Expression([*expr_A.inputs, expr_X], [expr_Y])
    return backend.einsum(str(exp), *ops_A, op_X)


def get_shape(expr, op_inputs, output):
    out_str = output.indices_string
    out_shape = [0]*len(out_str)
    for i in range(len(expr.inputs)):
        idx = expr.inputs[i].indices_string
        shape = op_inputs[i].shape
        for j in range(op_inputs[i].ndim):
            for k in range(len(out_str)):
                if out_str[k] == idx[j]:
                    out_shape[k] = shape[j]
    return out_shape


def orthogonalize_qr(backend, a):
    a_matrix, _ = backend.qr(a.reshape(-1, a.shape[-1]))
    return a_matrix.reshape(*a.shape)


def orthogonalize_local_gram(backend, a):
    remaining_indices = ''.join(chars[i] for i in range(a.ndim-1))
    gram_subscripts = '{}{},{}{}->{}{}'.format(
        remaining_indices, chars[a.ndim],
        remaining_indices, chars[a.ndim+1],
        chars[a.ndim], chars[a.ndim+1],
    )
    gram_a = backend.einsum(gram_subscripts, a.conj(), a)

    # local
    gram_a = gram_a.numpy()
    w, v = la.eigh(gram_a, overwrite_a=True)
    s = np.clip(w, 0, None) ** 0.5
    s_pinv = np.divide(1, s, out=np.zeros_like(s), where=s!=0)
    r_inv = np.einsum('j,ij->ij', s_pinv, v)

    r_inv = backend.astensor(r_inv)
    orthogonalize_subscripts = '{}{},{}{}->{}{}'.format(
        remaining_indices, chars[a.ndim],
        chars[a.ndim], chars[a.ndim+1],
        remaining_indices, chars[a.ndim+1],
    )
    q = backend.einsum(orthogonalize_subscripts, a, r_inv)

    return q
