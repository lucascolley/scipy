import numpy as np
from numpy.linalg import LinAlgError
from .blas import get_blas_funcs
from .lapack import get_lapack_funcs

from scipy._lib._array_api import array_namespace

__all__ = ['LinAlgError', 'LinAlgWarning', 'norm', 'cross', 'diagonal',
           'matmul', 'matrix_norm', 'matrix_power', 'matrix_rank',
           'matrix_transpose', 'outer', 'slogdet', 'tensordot', 'vecdot',
           'trace', 'vector_norm']


class LinAlgWarning(RuntimeWarning):
    """
    The warning emitted when a linear algebra related operation is close
    to fail conditions of the algorithm or loss of accuracy is expected.
    """
    pass


def norm(a, ord=None, axis=None, keepdims=False, check_finite=True):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter. For tensors with rank different from
    1 or 2, only `ord=None` is supported.

    Parameters
    ----------
    a : array_like
        Input array. If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``a.ravel`` will be returned.
    ord : {int, inf, -inf, 'fro', 'nuc', None}, optional
        Order of the norm (see table under ``Notes``). inf means NumPy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms. If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed. If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one. With this option the result will
        broadcast correctly against the original `a`.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(a), axis=1))      max(abs(a))
    -inf   min(sum(abs(a), axis=1))      min(abs(a))
    0      --                            sum(a != 0)
    1      max(sum(abs(a), axis=0))      as below
    -1     min(sum(abs(a), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(a)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import norm
    >>> a = np.arange(9) - 4.0
    >>> a
    array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4., -3., -2.],
           [-1.,  0.,  1.],
           [ 2.,  3.,  4.]])

    >>> norm(a)
    7.745966692414834
    >>> norm(b)
    7.745966692414834
    >>> norm(b, 'fro')
    7.745966692414834
    >>> norm(a, np.inf)
    4
    >>> norm(b, np.inf)
    9
    >>> norm(a, -np.inf)
    0
    >>> norm(b, -np.inf)
    2

    >>> norm(a, 1)
    20
    >>> norm(b, 1)
    7
    >>> norm(a, -1)
    -4.6566128774142013e-010
    >>> norm(b, -1)
    6
    >>> norm(a, 2)
    7.745966692414834
    >>> norm(b, 2)
    7.3484692283495345

    >>> norm(a, -2)
    0
    >>> norm(b, -2)
    1.8570331885190563e-016
    >>> norm(a, 3)
    5.8480354764257312
    >>> norm(a, -3)
    0

    """
    # Differs from numpy only in non-finite handling and the use of blas.
    if check_finite:
        a = np.asarray_chkfinite(a)
    else:
        a = np.asarray(a)

    if a.size and a.dtype.char in 'fdFD' and axis is None and not keepdims:

        if ord in (None, 2) and (a.ndim == 1):
            # use blas for fast and stable euclidean norm
            nrm2 = get_blas_funcs('nrm2', dtype=a.dtype, ilp64='preferred')
            return nrm2(a)

        if a.ndim == 2:
            # Use lapack for a couple fast matrix norms.
            # For some reason the *lange frobenius norm is slow.
            lange_args = None
            # Make sure this works if the user uses the axis keywords
            # to apply the norm to the transpose.
            if ord == 1:
                if np.isfortran(a):
                    lange_args = '1', a
                elif np.isfortran(a.T):
                    lange_args = 'i', a.T
            elif ord == np.inf:
                if np.isfortran(a):
                    lange_args = 'i', a
                elif np.isfortran(a.T):
                    lange_args = '1', a.T
            if lange_args:
                lange = get_lapack_funcs('lange', dtype=a.dtype, ilp64='preferred')
                return lange(*lange_args)

    # fall back to numpy in every other case
    return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)

    """
    if arr is original:
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None


def cross(x1, x2, *, axis=-1):
    xp = array_namespace(x1, x2)
    if hasattr(xp, 'linalg'):
        return xp.linalg.cross(x1, x2, axis=axis)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return xp.asarray(np.cross(x1, x2, axis=axis))


def diagonal(a, *, offset=0):
    xp = array_namespace(a)
    if hasattr(xp, 'linalg'):
        return xp.linalg.diagonal(a, offset=offset)
    a = np.asarray(a)
    return xp.asarray(np.diagonal(a, offset=offset))


def matmul(x1, x2):
    xp = array_namespace(x1, x2)
    return xp.matmul(x1, x2)


def matrix_norm(x, *, keepdims=False, ord='fro'):
    xp = array_namespace(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.matrix_norm(x, keepdims=keepdims, ord=ord)
    x = np.asarray(x)
    return xp.asarray(norm(x, keepdims=keepdims, ord=ord))


def matrix_power(x, n):
    xp = array_namespace(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.matrix_power(x, n)
    x = np.asarray(x)
    return xp.asarray(np.linalg.matrix_power(x, n))


def matrix_rank(x, *, rtol=None):
    xp = array_namespace(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.matrix_rank(x, rtol=rtol)
    tol = rtol * xp.max(x)
    x = np.asarray(x)
    return xp.asarray(np.linalg.matrix_rank(x, tol=tol))


def matrix_transpose(x):
    xp = array_namespace(x)
    return xp.matrix_transpose(x)


def outer(x1, x2):
    xp = array_namespace(x1, x2)
    if hasattr(xp, 'linalg'):
        return xp.linalg.outer(x1, x2)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return xp.asarray(np.outer(x1, x2))


def slogdet(x):
    xp = array_namespace(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.slogdet(x)
    x = np.asarray(x)
    return xp.asarray(np.linalg.slogdet(x))


def tensordot(x1, x2, *, axes=2):
    xp = array_namespace(x1, x2)
    return xp.tensordot(x1, x2, axes=axes)


def vecdot(x1, x2, *, axis=None):
    xp = array_namespace(x1, x2)
    return xp.vecdot(x1, x2, axis=axis)


def trace(x, *, offset=0, dtype=None):
    xp = array_namespace(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.trace(x, offset=offset, dtype=dtype)
    x = np.asarray(x)
    return xp.asarray(np.trace(x, offset=offset, dtype=dtype))


def vector_norm(x, *, axis=None, keepdims=False, ord=2):
    xp = array_namespace(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    x = np.asarray(x)
    return xp.asarray(norm(x, axis=axis, keepdims=keepdims, ord=ord))
