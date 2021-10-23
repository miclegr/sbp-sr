******
sbp-sr
******

Implementation of `Smooth Bilevel Programming for Sparse Regularization <https://arxiv.org/abs/2106.01429>`_ with Scikit-Learn API.

Installation
*************

::

    pip install -U git+https:///github.com/miclegr/sbp-sr
    
Requirements are Numpy, Scikit-Learn and Numba
    
Examples
*********

.. code-block:: python
    
    import sbpsr
    import sklearn.datasets
    
    n, p = 1000, 20
    X, y = sklearn.datasets.make_regression(n,p)

    model = sbpsr.Lasso(alpha=1e-3).fit(X,y)
    cv = sbpsr.LassoCV().fit(X,y)
    
Paper
*****

| Smooth Bilevel Programming for Sparse Regularization
| Clarice Poon, Gabriel Peyré
| `arXiv:2106.01429 [stat.ML] <https://arxiv.org/abs/2106.01429>`_
