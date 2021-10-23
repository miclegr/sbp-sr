import numpy as np
import numba as nb
import scipy.optimize


def opti_path(X, y, eps=1e-3, n_lambdas=100, lambdas=None,
              max_iter=1000, gtol=1e-30, ftol=1e-30, maxfun=10000,
              return_n_iter=False):

    n, p = X.shape
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y

    if lambdas is None:
        max_lam = np.max(Xty)
        lambdas = max_lam * np.geomspace(1, eps, n_lambdas,
                                         dtype=X.dtype)

    betas = np.zeros((p, len(lambdas)), dtype=X.dtype)
    n_iters = np.zeros(len(lambdas), dtype=np.int64)

    for i, lam in enumerate(lambdas):

        v0 = np.random.normal(size=p) if i == 0 else out.x
        opts = {'gtol': gtol, 'maxiter': max_iter, 'maxcor': 10,
                'ftol': ftol, 'maxfun': maxfun}
        out = scipy.optimize.minimize(f_n_grad, v0, method='L-BFGS-B',
                                      jac=True, options=opts,
                                      args=(lam, X, y, XtX, Xt, Xty))
        beta = out.x * u_star(out.x, lam, XtX, Xty)
        betas[:, i] = beta
        n_iters[i] = out.nit

    out = lambdas, betas, np.zeros_like(lambdas), np.zeros_like(lambdas)
    if return_n_iter:
        out += (n_iters,)

    return out


@nb.njit(cache=True)
def u_star(v, lam, XtX, Xty):
    d = np.diag(v)
    T = d @ XtX @ d + lam*np.eye(XtX.shape[0])
    u = np.linalg.solve(T, v * Xty)
    return u


@nb.njit(cache=True)
def f_n_grad(v, lam, X, y, XtX, Xt, Xty):
    u = u_star(v, lam, XtX, Xty)
    beta = v * u
    Xbeta = X@beta
    loss = ((Xbeta - y)**2).sum()
    pen = 0.5*((u**2).sum() + (v**2).sum())
    f = 1/(2*lam) * loss + pen
    grad = 1/lam*(Xt @ (Xbeta - y)) * u + v
    return f, grad
