from .path import opti_path
import numpy as np

from sklearn.base import RegressorMixin
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model._coordinate_descent import LinearModelCV


class Lasso(Lasso_sklearn):
    r"""
    Lasso scikit-learn estimator based on SBP-LR solver

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    max_iter : int, optional
        The maximum number of iterations of L-BFGS-B optimizer

    max_fun : int
        The maximum number of objective function call
        made by L-BFGS-B optimizer

    ftol : float, optional
        Tolerance parameter passed to internal call to L-BFGS-B optimizer

    gtol : float, optional
        Tolerance parameter passed to internal call to L-BFGS-B optimizer


    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    """

    def __init__(self, alpha=1., fit_intercept=True,
                 warm_start=False, max_iter=1000,
                 gtol=1e-5, ftol=2e-9, max_fun=10000
                 ):
        super(Lasso, self).__init__(
            alpha=alpha, tol=None, max_iter=max_iter,
            fit_intercept=fit_intercept, warm_start=warm_start)

        self.gtol = gtol
        self.ftol = ftol
        self.max_fun = max_fun

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path"""

        if alphas is not None:
            lambdas = [1/(alpha * 2) for alpha in alphas]
        else:
            lambdas = None

        lambdas, coeffs, dual_gaps, etc = opti_path(
                X, y, lambdas=lambdas, max_iter=self.max_iter,
                gtol=self.gtol, ftol=self.ftol, maxfun=self.max_fun,
                return_n_iter=False)

        return lambdas, coeffs, dual_gaps, etc


class LassoCV(RegressorMixin, LinearModelCV):
    r"""
    LassoCV scikit-learn estimator based on SBP-SR solver

    The best model is selected by cross-validation.

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas: int, optional
        Number of lambdas along the regularization path.

    alphas : numpy array, optional
        List of lambdas where to compute the models.
        If ``None`` lambdas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    precompute : bool, optional
        Kept for sklearn.linear_model.LassoCV compatibility

    max_iter : int, optional
        The maximum number of iterations of L-BFGS-B optimizer

    max_fun : int
        The maximum number of objective function call
        made by L-BFGS-B optimizer

    ftol : float, optional
        Tolerance parameter passed to internal call to L-BFGS-B optimizer

    gtol : float, optional
        Tolerance parameter passed to internal call to L-BFGS-B optimizer

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape ()
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, cv=None, n_jobs=None, precompute=False,
                 max_iter=1000, gtol=1e-5, ftol=2e-2, max_fun=10000):
        super(LassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=None, cv=cv, fit_intercept=fit_intercept,
            verbose=False, n_jobs=n_jobs)

        self.verbose = 0
        self.gtol = gtol
        self.ftol = ftol
        self.max_fun = max_fun

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path"""

        if alphas is not None:
            lambdas = [(alpha * X.shape[0]) for alpha in alphas]
        else:
            lambdas = None

        lambdas, coefs, dual_gaps, _ = opti_path(
                X, y, lambdas=lambdas,
                n_lambdas=self.n_alphas, max_iter=self.max_iter,
                gtol=self.gtol, ftol=self.ftol, maxfun=self.max_fun,
                return_n_iter=False)

        return np.ones_like(alphas[::-1]), coefs, dual_gaps

    def _get_estimator(self):
        return Lasso()

    def _is_multitask(self):
        return False
