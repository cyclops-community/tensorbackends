from . import einstr
import numpy as np

def apply_A(backend,expr_A,ops_A,expr_X,op_X,expr_Y):
    exp = einstr.Expression(expr_A.inputs + [expr_X], [expr_Y]) 
    return backend.einsum(str(exp),*(list(ops_A)+[op_X]))

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

def einsumsvd_rand(backend, subscripts, *operands, rank, niter=1):
    ndims = [operand.ndim for operand in operands]
    expr = einstr.parse_einsumsvd(subscripts, ndims)
    expr_A, einsvd_expr = einstr.split_einsumsvd(expr)
    dim_aux = rank # + oversamp # if oversampling, but then need to extract
    ops_A = operands[:len(expr_A.inputs)]
    shape_X = []
    need_transpose_X = False
    permutation_X = []
    shape_U = get_shape(expr_A, ops_A, einsvd_expr.outputs[0])
    idx_X = ['_']*len(shape_U)
    for i in range(len(shape_U)):
        if shape_U[i] == 0:
            shape_U[i] = dim_aux
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
    shape_X.append(dim_aux)
    shape_YT = []
    need_transpose_VT = False
    permutation_VT = []
    shape_VT = get_shape(expr_A, ops_A, einsvd_expr.outputs[1])
    idx_YT = ['_']*len(shape_VT)
    for i in range(len(shape_VT)):
        if shape_VT[i] == 0:
            shape_VT[i] = dim_aux
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
    shape_YT.append(dim_aux)
    op_X = backend.random.random(shape_X)
    # FIXME: start by QR of op_X if rank is not too large
    for iter in range(niter):
        op_YT = apply_A(backend,expr_A,ops_A,term_X,op_X,term_YT)
        op_X = apply_A(backend,expr_A,ops_A,term_YT,op_YT,term_X)
        mat_X, _ = backend.qr(backend.reshape(op_X,[np.prod(op_X.shape)//rank,rank]))
        op_X = backend.reshape(mat_X,op_X.shape)
    op_YT = apply_A(backend,expr_A,ops_A,term_X,op_X,term_YT)
    mat_VT, _ = backend.qr(backend.reshape(op_YT,[np.prod(op_YT.shape)//rank,rank]))
    op_YT = backend.reshape(mat_VT,op_YT.shape)
    op_X = apply_A(backend,expr_A,ops_A,term_YT,op_YT,term_X)
    mat_U, S, mat_XVT = backend.svd(backend.reshape(op_X,[np.prod(op_X.shape)//rank,rank]))
    op_YT = backend.tensordot(op_YT, mat_VT, axes=((-1),(-1)))
    op_X = backend.reshape(mat_U,op_X.shape)
    U = op_X
    if need_transpose_X:
      U = backend.einsum(term_X.indices_string+"->"+einsvd_expr.outputs[0].indices_string,U)
    VT = op_YT
    if need_transpose_VT:
      VT = backend.einsum(term_YT.indices_string+"->"+einsvd_expr.outputs[1].indices_string,VT)
    return U, S, VT 
