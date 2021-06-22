# The custom implementation of Robust PCA
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F


__all__ = ['RPCA_gpu', 'RPCA']


class RPCA:
    """Robust PCA using principal component pursuit (PCP) algorithm.
    Referenced from https://github.com/dganguli/robust-pca/blob/master/r_pca.py
    Args:
        mu(float): coefficient of the augmented lagrange multiplier
        lmbda(float): coefficient of the sparse part
        max_iter(int): maximal iterations to perform
        tol(float): tolerance(error bound) of the result
    """
    def __init__(self, D, mu=None, lmbda=None, max_iter=100, tol=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.max_iter = max_iter
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
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

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
            _tol = (1e-7) * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < self.max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk,
                                    self.mu_inv)  #this line implements step 3
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv *
                             self.lmbda)  #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk
                                 )  #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            print("[INFO] iter: ", iter, "error: ", err)
            if err < _tol:
                break
        print(f"[INFO] Converged after {iter} iterations.")

        self.L = Lk
        self.S = Sk
        return Lk, Sk


class RPCA_gpu:
    """ Referenced from https://gist.github.com/jcreinhold/ebf27f997f4c93c2f637c3c900d6388f
        low-rank and sparse matrix decomposition via RPCA [1] with CUDA capabilities 
        implementations of RPCA on the GPU (leveraging pytorch) 
        for low-rank and sparse matrix decomposition as well as 
        a nuclear-norm minimization routine via singular value 
        thresholding for matrix completion
        The RPCA implementation is heavily based on:
            https://github.com/dganguli/robust-pca
            
        The svt implementation was based on:
            https://github.com/tonyduan/matrix-completion
        References:
        [1] Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). 
            Robust principal component analysis?. Journal of the ACM (JACM), 
            58(3), 11.
        [2] Cai, J. F., Candès, E. J., & Shen, Z. (2010). 
            A singular value thresholding algorithm for matrix completion. 
            SIAM Journal on Optimization, 20(4), 1956-1982.
            
        Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
    """
    def __init__(self, D, mu=None, lmbda=None, max_iter=100, tol=None):
        self.D = D
        self.S = torch.zeros_like(self.D, device=D.device)
        self.Y = torch.zeros_like(self.D, device=D.device)
        self.mu = mu or (
            torch.prod(torch.tensor(self.D.shape, device=D.device)) /
            (4 * self.norm_p(self.D, 2))).item()
        self.mu_inv = 1 / self.mu
        self.lmbda = lmbda or (1 / torch.sqrt(
            torch.max(
                torch.tensor(self.D.shape,
                             device=D.device,
                             dtype=torch.float32)))).item()
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def norm_p(M, p):
        return torch.sum(torch.pow(M, p))

    @staticmethod
    def shrink(M, tau):
        return torch.sign(M) * F.relu(
            torch.abs(M) - tau)  # hack to save memory

    def svd_threshold(self, M, tau):
        U, s, V = torch.svd(M, some=True)
        return torch.mm(U, torch.mm(torch.diag(self.shrink(s, tau)), V.t()))

    def __call__(self):
        i, err = 0, np.inf
        Sk, Yk, Lk = self.S, self.Y, torch.zeros_like(self.D,
                                                      device=self.D.device)
        _tol = self.tol or 1e-7 * self.norm_p(torch.abs(self.D), 2)
        while err > _tol and i < self.max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk,
                                    self.mu_inv)
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk),
                             self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(torch.abs(self.D - Lk - Sk), 2) / self.norm_p(
                self.D, 2)
            i += 1
            print(f'[INFO] iter: {i}; error: {err}')
            if err < _tol or i > self.max_iter:
                print(f"[INFO] Converged after {i} iterations. ")
                break
        self.L, self.S = Lk, Sk
        return Lk, Sk

