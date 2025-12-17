"""
Check the speed of the conjugate gradient solver.
"""
import inspect

import numpy as np
from numpy.testing import assert_equal

from .common import Benchmark, safe_import

with safe_import():
    from scipy import linalg, sparse
    from scipy.sparse.linalg import cg, minres, gmres, tfqmr, spsolve, LinearOperator
with safe_import():
    from scipy.sparse.linalg import lgmres
with safe_import():
    from scipy.sparse.linalg import gcrotmk


def _create_sparse_poisson1d(n):
    # Make Gilbert Strang's favorite matrix
    # http://www-math.mit.edu/~gs/PIX/cupcakematrix.jpg
    P1d = sparse.diags_array(
        [[-1]*(n-1), [2]*n, [-1]*(n-1)],
        offsets=[-1, 0, 1],
        dtype=np.float64
    )
    assert_equal(P1d.shape, (n, n))
    return P1d


def _create_sparse_poisson2d(n):
    P1d = _create_sparse_poisson1d(n)
    P2d = sparse.kronsum(P1d, P1d)
    assert_equal(P2d.shape, (n*n, n*n))
    return P2d.tocsr()


def _create_sparse_poisson2d_coo(n):
    P1d = _create_sparse_poisson1d(n)
    P2d = sparse.kronsum(P1d, P1d)
    assert_equal(P2d.shape, (n*n, n*n))
    return P2d.tocoo()


class Bench(Benchmark):
    params = [
        [4, 8, 16, 32, 64, 128, 256, 512],
        ['dense', 'spsolve', 'cg', 'minres', 'gmres', 'lgmres', 'gcrotmk',
         'tfqmr']
    ]
    mapping = {'spsolve': spsolve, 'cg': cg, 'minres': minres, 'gmres': gmres,
               'lgmres': lgmres, 'gcrotmk': gcrotmk, 'tfqmr': tfqmr}
    param_names = ['(n,n)', 'solver']

    def setup(self, n, solver):
        if solver == 'dense' and n >= 25:
            raise NotImplementedError()

        self.b = np.ones(n*n)
        self.P_sparse = _create_sparse_poisson2d(n)

        if solver == 'dense':
            self.P_dense = self.P_sparse.toarray()

    def time_solve(self, n, solver):
        if solver == 'dense':
            linalg.solve(self.P_dense, self.b)
        else:
            self.mapping[solver](self.P_sparse, self.b)


class BatchedCG(Benchmark):
    params = [
        [2, 4, 6, 8],
        [1, 10, 100, 500, 1000, 2500, 5000, 10000]
    ]
    param_names = ['(n,n)', 'batch_size']

    def setup(self, n, batch_size):
        rng = np.random.default_rng(42)
        
        self.batched = "xp" in inspect.signature(LinearOperator.__init__).parameters
        if self.batched:
            P_sparse = _create_sparse_poisson2d_coo(n)
            if batch_size > 1:
                self.P_sparse = sparse.vstack(
                    [P_sparse] * batch_size, format="coo"
                ).reshape(batch_size, n*n, n*n)
                self.b = rng.standard_normal((batch_size, n*n))
            else:
                self.P_sparse = P_sparse
                self.b = rng.standard_normal(n*n)
        else:
            self.P_sparse = _create_sparse_poisson2d(n)
            self.b = [rng.standard_normal(n*n) for _ in range(batch_size)]

    def time_solve(self, n, batch_size):
        if self.batched:
            cg(self.P_sparse, self.b)
        else:
            for i in range(batch_size):
                cg(self.P_sparse, self.b[i])


def _create_dense_random(n, batch_shape=None):
    rng = np.random.default_rng(42)
    M = rng.standard_normal((n*n, n*n))
    reg = 1e-3
    if batch_shape:
        M = np.broadcast_to(M[np.newaxis, ...], (*batch_shape, n*n, n*n))
    
    def matvec(x):
        return np.squeeze(M.mT @ (M @ x[..., np.newaxis]), axis=-1) + reg * x
    
    return LinearOperator(shape=M.shape, matvec=matvec, dtype=np.float64)
    

class BatchedCGDense(Benchmark):
    params = [
        [2, 4, 8, 16, 24],
        [1, 10, 100, 500, 1000]
    ]
    param_names = ['(n,n)', 'batch_size']

    def setup(self, n, batch_size):
        rng = np.random.default_rng(42)
        
        self.batched = "xp" in inspect.signature(LinearOperator.__init__).parameters
        if self.batched:
            if batch_size > 1:
                self.A = _create_dense_random(n, batch_shape=(batch_size,))
                self.b = rng.standard_normal((batch_size, n*n))
            else:
                self.A = _create_dense_random(n)
                self.b = rng.standard_normal(n*n)
        else:
            self.A = _create_dense_random(n)
            self.b = [rng.standard_normal(n*n) for _ in range(batch_size)]

    def time_solve(self, n, batch_size):
        if self.batched:
            cg(self.A, self.b)
        else:
            for i in range(batch_size):
                cg(self.A, self.b[i])


class Lgmres(Benchmark):
    params = [
        [10, 50, 100, 1000, 10000],
        [10, 30, 60, 90, 180],
    ]
    param_names = ['n', 'm']

    def setup(self, n, m):
        rng = np.random.default_rng(1234)
        self.A = sparse.eye(n, n) + sparse.rand(n, n, density=0.01, random_state=rng)
        self.b = np.ones(n)

    def time_inner(self, n, m):
        lgmres(self.A, self.b, inner_m=m, maxiter=1)
