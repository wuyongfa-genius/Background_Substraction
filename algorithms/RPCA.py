try:
    # try import from tensorly package
    import tensorly as tl
    from tensorly.decomposition import robust_pca

    class RPCA:
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
            tl.set_backend(backend)

        def __call__(self, X):
            return robust_pca(X, self.mask, self.tol, self.reg_E, self.reg_J,
                              self.mu_init, self.mu_max, self.lr,
                              self.n_iter_max, self.verbose)

except ImportError:
    # import the custom implementation
    import numpy as np

    class RPCA:
        """Robust PCA using principal component pursuit (PCP) algorithm.
        Referenced from https://github.com/dganguli/robust-pca/blob/master/r_pca.py
        Args:
            mu(float): coefficient of the augmented lagrange multiplier
            lmbda(float): coefficient of the sparse part
            max_iter(int): maximal iterations to perform
            log_interval(int): how often do you want to print error info
            tol(float): tolerance(error bound) of the result
        """
        def __init__(self,
                     D,
                     mu=None,
                     lmbda=None,
                     max_iter=100,
                     log_interval=10,
                     tol=None):
            self.D = D
            self.S = np.zeros(self.D.shape)
            self.Y = np.zeros(self.D.shape)
            self.max_iter = max_iter
            self.log_interval = log_interval
            self.tol = tol

            if mu:
                self.mu = mu
            else:
                self.mu = np.prod(
                    self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

            self.mu_inv = 1 / self.mu

            if lmbda:
                self.lmbda = lmbda
            else:
                self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

        @staticmethod
        def frobenius_norm(M):
            return np.linalg.norm(M, ord='fro')

        @staticmethod
        def shrink(M, tau):
            return np.sign(M) * np.maximum(
                (np.abs(M) - tau), np.zeros(M.shape))

        def svd_threshold(self, M, tau):
            U, S, V = np.linalg.svd(M, full_matrices=False)
            return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

        def __call__(self):
            iter = 0
            err = np.Inf
            Sk = self.S
            Yk = self.Y
            Lk = np.zeros(self.D.shape)

            if self.tol:
                _tol = self.tol
            else:
                _tol = 1E-7 * self.frobenius_norm(self.D)

            #this loop implements the principal component pursuit (PCP) algorithm
            #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
            while (err > _tol) and iter < self.max_iter:
                Lk = self.svd_threshold(
                    self.D - Sk + self.mu_inv * Yk,
                    self.mu_inv)  #this line implements step 3
                Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk),
                                 self.mu_inv *
                                 self.lmbda)  #this line implements step 4
                Yk = Yk + self.mu * (self.D - Lk - Sk
                                     )  #this line implements step 5
                err = self.frobenius_norm(self.D - Lk - Sk)
                iter += 1
                if (iter % self.log_interval
                    ) == 0 or iter == 1 or iter > self.max_iter or err <= _tol:
                    print('iteration: {0}, error: {1}'.format(iter, err))

            self.L = Lk
            self.S = Sk
            return Lk, Sk
