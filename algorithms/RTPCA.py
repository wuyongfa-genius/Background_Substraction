import tensorly as tl
import numpy as np
import torch
from tensorly import backend as T
from tensorly.base import fold, unfold
from tensorly.tenalg.proximal import soft_thresholding, svd_thresholding

# Author: Jean Kossaifi

# License: BSD 3 clause

def robust_pca(X, mask=None, tol=10e-7, reg_E=1, reg_J=1,
               mu_init=10e-5, mu_max=10e9, learning_rate=1.1,
               n_iter_max=100, verbose=1):
    """Robust Tensor PCA via ALM with support for missing values

        Decomposes a tensor `X` into the sum of a low-rank component `D`
        and a sparse component `E`.

    Parameters
    ----------
    X : ndarray
        tensor data of shape (n_samples, N1, ..., NS)
    mask : ndarray
        array of booleans with the same shape as `X`
        should be zero where the values are missing and 1 everywhere else
    tol : float
        convergence value
    reg_E : float, optional, default is 1
        regularisation on the sparse part `E`
    reg_J : float, optional, default is 1
        regularisation on the low rank part `D`
    mu_init : float, optional, default is 10e-5
        initial value for mu
    mu_max : float, optional, default is 10e9
        maximal value for mu
    learning_rate : float, optional, default is 1.1
        percentage increase of mu at each iteration
    n_iter_max : int, optional, default is 100
        maximum number of iteration
    verbose : int, default is 1
        level of verbosity

    Returns
    -------
    (D, E)
        Robust decomposition of `X`

    D : `X`-like array
        low-rank part
    E : `X`-like array
        sparse error part

    Notes
    -----
    The problem we solve is, for an input tensor :math:`\\tilde X`:

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
           & \\min_{\\{J_i\\}, \\tilde D, \\tilde E}
           & & \\sum_{i=1}^N  \\text{reg}_J \\|J_i\\|_* + \\text{reg}_E \\|E\\|_1 \\\\
           & \\text{subject to}
           & & \\tilde X  = \\tilde A + \\tilde E \\\\
           & & & A_{[i]} =  J_i,  \\text{ for each } i \\in \\{1, 2, \\cdots, N\\}\\\\
        \\end{aligned}
        \\end{equation*}

    """
    if mask is None:
        mask = 1

    # Initialise the decompositions
    D = T.zeros_like(X, **T.context(X))  # low rank part
    E = T.zeros_like(X, **T.context(X))  # sparse part
    L_x = T.zeros_like(X, **T.context(X))  # Lagrangian variables for the (X - D - E - L_x/mu) term
    J = [T.zeros_like(X, **T.context(X)) for _ in range(T.ndim(X))] # Low-rank modes of X
    L = [T.zeros_like(X, **T.context(X)) for _ in range(T.ndim(X))] # Lagrangian or J

    # Norm of the reconstructions at each iteration
    rec_X = []
    rec_D = []

    mu = mu_init

    for iteration in range(n_iter_max):

        for i in range(T.ndim(X)):
            J[i] = fold(svd_thresholding(unfold(D, i) + unfold(L[i], i)/mu, reg_J/mu), i, X.shape)

        D = L_x/mu + X - E
        for i in range(T.ndim(X)):
            D += J[i] - L[i]/mu
        D /= (T.ndim(X) + 1)

        E = soft_thresholding(X - D + L_x/mu, mask*reg_E/mu)

        # Update the lagrangian multipliers
        for i in range(T.ndim(X)):
            L[i] += mu * (D - J[i])

        L_x += mu*(X - D - E)

        mu = min(mu*learning_rate, mu_max)

        # Evolution of the reconstruction errors
        rec_X.append(T.norm(X - D - E, 2))
        rec_D.append(max([T.norm(low_rank - D, 2) for low_rank in J]))

        # Convergence check
        if iteration > 1:
            if max(rec_X[-1], rec_D[-1]) <= tol:
                if verbose:
                    print('\nConverged in {} iterations'.format(iteration))
                break
            else:
                print("[INFO] iter:", iteration, " error:", (max(rec_X[-1], rec_D[-1]).item()))

    return D, E

class RTPCA:
    """Robust Tensor PCA via ALM with support for missing values
        Decomposes a tensor `X` into the sum of a low-rank component `D`
        and a sparse component `E`.
    Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        mask : ndarray
            array of booleans with the same shape as `X`
            should be zero where the values are missing and 1 everywhere else
        tol : float
            convergence value
        reg_E : float, optional, default is 1
            regularisation on the sparse part `E`
        reg_J : float, optional, default is 1
            regularisation on the low rank part `D`
        mu_init : float, optional, default is 10e-5
            initial value for mu
        mu_max : float, optional, default is 10e9
            maximal value for mu
        learning_rate : float, optional, default is 1.1
            percentage increase of mu at each iteration
        n_iter_max : int, optional, default is 100
            maximum number of iteration
        verbose : int, default is 1
            level of verbosity
    Returns
        -------
        (D, E)
            Robust decomposition of `X`

        D : `X`-like array
            low-rank part
        E : `X`-like array
            sparse error part
"""
    def __init__(self,
                    mask=None,
                    tol=10e-7,
                    reg_E=1,
                    reg_J=1,
                    mu_init=10e-5,
                    mu_max=10e9,
                    learning_rate=1.1,
                    n_iter_max=100,
                    verbose=1,
                    backend='numpy'):
        self.mask = mask
        self.tol = tol
        self.reg_E = reg_E
        self.reg_J = reg_J
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.lr = learning_rate
        self.n_iter_max = n_iter_max
        self.verbose = verbose
        self.backend = backend
        tl.set_backend(backend)

    def __call__(self, X):
        if self.backend=='numpy':
            assert isinstance(X, np.ndarray)
            X = np.transpose(X)
        elif self.backend=='pytorch':
            assert isinstance(X, torch.Tensor)
            X = torch.transpose(X, 0, 1)
        L, S = robust_pca(X, self.mask, self.tol, self.reg_E, self.reg_J,
                            self.mu_init, self.mu_max, self.lr,
                            self.n_iter_max, self.verbose)
        if self.backend=='numpy':
            L = np.transpose(L)
            S = np.transpose(S)
        elif self.backend=='pytorch':
            L = torch.transpose(L, 0, 1)
            S = torch.transpose(S, 0, 1)
        return L, S