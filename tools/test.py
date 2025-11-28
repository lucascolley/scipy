import sys
sys.path.insert(0, "/Users/lucascolley/ghq/github.com/rgommers/pixi-dev-scipystack/scipy/scipy/build-install/usr/lib/python3.13/site-packages")

import numpy as np
from scipy._lib.array_api_compat import numpy as xp
from scipy.sparse.linalg import cg, LinearOperator

def solve(N, batch, report_index=0, batched=False):
    rng = np.random.default_rng(0)
    M = rng.standard_normal((N, N))
    M = xp.asarray(M)
    reg = 1e-3
        
    if batched:
        M = xp.broadcast_to(M[xp.newaxis, ...], (batch, *M.shape))
    
    def matvec(x):
        return xp.squeeze(M.mT @ (M @ x[..., xp.newaxis]), axis=-1) + reg * x
    
    shape = (batch, N, N) if batched else (N, N)
    A = LinearOperator(shape, matvec=matvec, dtype=xp.float64, xp=xp)
    # A = LinearOperator(shape, matvec=matvec, dtype=xp.float64)
    
    b = rng.standard_normal(N)
    b = xp.asarray(b)
    
    if batched:
        b = xp.reshape(xp.arange(batch, dtype=xp.float64), (batch, 1)) * b
        x, info = cg(A, b, atol=1e-8, maxiter=5000)
        assert info == 0
        print(f"{x[report_index, ...]}")
    else:
        for i in xp.arange(batch, dtype=xp.float64):
            x, info = cg(A, i*b, atol=1e-8, maxiter=5000)
            assert info == 0
            if i == report_index:
                print(x)

solve(40, 10000)
