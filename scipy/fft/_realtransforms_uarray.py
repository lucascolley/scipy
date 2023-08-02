from ._basic_uarray import _dispatch
from scipy._lib.uarray import Dispatchable
import numpy as np

__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']


@_dispatch
def dctn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, *, orthogonalize=None):
    """dctn multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idctn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, orthogonalize=None):
    """idctn multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dstn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    """dstn multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idstn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, orthogonalize=None):
    """idstn multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dct(x, type=2, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, orthogonalize=None):
    """dct multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idct(x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    """idct multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dst(x, type=2, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, orthogonalize=None):
    """dst multimethod."""
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idst(x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    """idst multimethod."""
    return (Dispatchable(x, np.ndarray),)