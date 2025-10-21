__docformat__ = "restructuredtext en"

__all__ = []


import numpy as np

from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
     IdentityOperator

_coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
                 ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
                 ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
                 ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
                 ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
                 ('D','D'):'D'}


def coerce(x,y):
    if x not in 'fdFD':
        x = 'd'
    if y not in 'fdFD':
        y = 'd'
    return _coerce_rules[x,y]


def id(x):
    return x


def make_system(A, M, x0, b):
    """Make a linear system Ax=b

    Parameters
    ----------
    A : LinearOperator
        sparse or dense matrix (or any valid input to aslinearoperator)
    M : {LinearOperator, None}
        preconditioner
        sparse or dense matrix (or any valid input to aslinearoperator)
    x0 : {array_like, str, None}
        initial guess to iterative method.
        ``x0 = 'Mb'`` means using the nonzero initial guess ``M @ b``.
        Default is `None`, which means using the zero initial guess.
    b : array_like
        right hand side

    Returns
    -------
    (A, M, x, b)
        A : LinearOperator
            matrix of the linear system
        M : LinearOperator
            preconditioner
        x : rank 1 ndarray
            initial guess
        b : rank 1 ndarray
            right hand side

    """
    A_ = A
    A = aslinearoperator(A)

    if (N := A.shape[-2]) != A.shape[-1]:
        raise ValueError(f'expected square matrix or stack of square matrices, but got shape={(A.shape,)}')

    b = np.asanyarray(b)

    column_vector = b.ndim == 2 and b.shape[-2:] == (N, 1) # maintain column vector backwards-compatibility in 2-D case
    row_vector = b.shape[-1] == N # otherwise treat as a row-vector

    if not (column_vector or row_vector):
        raise ValueError(f'shapes of A {A.shape} and b {b.shape} are '
                         'incompatible')

    if b.dtype.char not in 'fdFD':
        b = b.astype('d')  # upcast non-FP types to double

    if hasattr(A,'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)

    b = np.asarray(b, dtype=xtype)  # make b the same type as x
    if column_vector:
        b = np.ravel(b)

    # process preconditioner
    if M is None:
        if hasattr(A_,'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_,'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                               dtype=A.dtype)
    else:
        M = aslinearoperator(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    # set initial guess
    if x0 is None:
        x = np.zeros((*M.shape[:-2], N), dtype=xtype)
    elif isinstance(x0, str):
        if x0 == 'Mb':  # use nonzero initial guess ``M @ b``
            bCopy = b.copy()
            x = M.matvec(bCopy)
    else:
        x = np.array(x0, dtype=xtype)
        
        column_vector = x.ndim == 2 and x.shape[-2:] == (N, 1) # maintain column vector backwards-compatibility in 2-D case
        row_vector = x.shape[-1] == N # otherwise treat as a row-vector
        
        if not (row_vector or column_vector):
            raise ValueError(f'shapes of A {A.shape} and '
                             f'x0 {x.shape} are incompatible')
        if column_vector:
            x = np.ravel(x)

    return A, M, x, b
