"""REFERENCE: Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix
Decomposition in Noisy Case", ICML 2011
Tianyi Zhou, 2011, All rights reserved."""
import numpy as np


class GoDec:
    """
    GoDec - Go Decomposition (Tianyi Zhou and Dacheng Tao, 2011)
    The algorithm estimate the low-rank part L and the sparse part S of a matrix X = L + S + G with noise G.
    Args:
        X : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
            and a low-rank matrix L.
        rank : int >= 1, optional
            The rank of low-rank matrix. The default is 1.
        card : int >= 0, optional
            The cardinality of the sparse matrix. The default is None (number of array elements in X).
        iterated_power : int >= 1, optional
            Number of iterations for the power method, increasing it lead to better accuracy and more time cost. The default is 1.
        max_iter : int >= 0, optional
            Maximum number of iterations to be run. The default is 100.
        error_bound : float >= 0, optional
            error_bounderance for stopping criteria. The default is 0.001.
        return_error: bool, whether to return error.
    """
    def __init__(self,
                 X,
                 rank=1,
                 card=None,
                 iterated_power=2,
                 max_iter=100,
                 error_bound=0.001,
                 return_error=False):
        self.X = X
        self.rank = rank
        self.card = card = np.prod(X.shape) if card is None else card
        self.iterated_power = iterated_power
        self.max_iter = max_iter
        self.error_bound = error_bound
        self.return_error = return_error

    def __call__(self, ):
        return self._godec(self.X, self.rank, self.card, self.iterated_power,
                           self.max_iter, self.error_bound, self.return_error)

    @staticmethod
    def _godec(X,
               rank=1,
               card=None,
               iterated_power=2,
               max_iter=100,
               error_bound=0.001,
               return_error=False):
        """
        Returns:
            L : array-like, low-rank matrix.
            S : array-like, sparse matrix.
            LS : array-like, reconstruction matrix.
            RMSE : root-mean-square error.
        """

        if return_error:
            RMSE = []

        X = X.T if (X.shape[0] < X.shape[1]) else X
        _, n = X.shape

        # Initialization of L and S
        L = X
        S = np.zeros(X.shape)
        LS = np.zeros(X.shape)

        for i in range(max_iter):
            # Update of L
            Y2 = np.random.randn(n, rank)
            for _ in range(iterated_power):
                Y1 = L.dot(Y2)
                Y2 = L.T.dot(Y1)
            Q, _ = np.linalg.qr(Y2)
            L_new = (L.dot(Q)).dot(Q.T)

            # Update of S
            T = L - L_new + S
            L = L_new
            T_vec = T.reshape(-1)
            S_vec = S.reshape(-1)
            idx = abs(T_vec).argsort()[::-1]
            S_vec[idx[:card]] = T_vec[idx[:card]]
            S = S_vec.reshape(S.shape)

            # Reconstruction
            LS = L + S

            # Stopping criteria
            error = np.sqrt(np.mean((X - LS)**2))
            if return_error:
                RMSE.append(error)

            print("iter: ", i, "error: ", error)
            if (error <= error_bound):
                break

        if return_error:
            L, S, LS, RMSE

        return L, S


# if __name__=="__main__":
#     X = np.random.randint(0, 100, (12000, 100))
#     rank = 50
#     L,S = godec(X, rank=rank)
#     print(L.shape, S.shape)
#     print(np.linalg.matrix_rank(L))
