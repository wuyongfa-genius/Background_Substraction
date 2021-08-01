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
                 rank=2,
                 card=None,
                 iterated_power=2,
                 max_iter=10,
                 error_bound=1e-6,
                 return_error=False,
                 **kwargs):
        self.X = X
        self.rank = rank
        self.card = int(np.prod(X.shape)/20) if card is None else card
        self.iterated_power = iterated_power
        self.max_iter = max_iter
        self.error_bound = error_bound
        self.return_error = return_error
        self.percent = kwargs.get('percent', None)
        self.start_percent = kwargs.get('start_percent', None)
        self.rcode = kwargs.get('rcode', None)
        self.task_id = kwargs.get('task_id', None)

    def __call__(self):
        return self._godec(self.X, self.rank, self.card, self.iterated_power,
                           self.max_iter, self.error_bound, self.return_error)

    def _godec(self,
                X,
               rank=2,
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
        update_progress = self.percent is not None and self.start_percent is not None
        if update_progress:
            try:
                import TaskUtil
            except:
                raise ModuleNotFoundError
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
            S_vec[idx[:card]] = T_vec[idx[:card]] # K largest entries of |X-Lt|
            S = S_vec.reshape(S.shape)

            # Reconstruction
            LS = L + S

            # Stopping criteria
            error = np.sqrt(np.mean((X - LS)**2))
            if return_error:
                RMSE.append(error)

            print("[INFO] iter: ", i, "error: ", error)
            if update_progress:
                current_percent = int(self.start_percent+(i+1)*self.percent/self.max_iter)
                TaskUtil.SetPercent(self.rcode, self.task_id, current_percent, '')
            if (error <= error_bound):
                print(f"[INFO] Converged after {i} iterations.")
                break

        if return_error:
            L, S, LS, RMSE

        return L, S


def godec_original(X, r, k, e=1e-6, q=0, max_iter=100):
    """GoDec implemented exactly following the oiginal paper.
    Args:
        X(np.ndarray): the dense matrix to be decomposited.
        r(int): the rank of the desired Low-rank component.
        k(int): the cardinality of the desired Sparse component.
        e(float): the error bound between the final reconstruction L+S 
            and the actual input X.
        q(int): when q=0, directly perform BRP(bilateral random project)
            approximation if L. When q>0, perform the power scheme.
    Return:
        the final decomposition L(Low-rank) and S(Sparse).
    """
    m, n = X.shape
    t = 0 # init iteration
    Lt = X # init Low-rank approximation
    St = np.zeros_like(X) # init Sparse residual component
    desired_err = np.linalg.norm(X, ord='fro')*e # termination condition
    ## start GoDec
    cur_err = np.inf
    while cur_err>desired_err and t<max_iter:
        ## update Lt
        L_wave = X-St
        for _ in range(q):
            L_wave = (L_wave.dot(L_wave.T)).dot(L_wave)
        A1 = np.random.randn(n, r)
        Y1 = L_wave.dot(A1)
        A2 = Y1
        Y2 = L_wave.T.dot(Y1)
        Q2, R2 = np.linalg.qr(Y2)
        Y1 = L_wave.dot(Y2)
        Q1, R1 = np.linalg.qr(Y1)
        T = A2.T.dot(Y1)
        # if np.linalg.matrix_rank(T)<r:
        #     print("[INFO] Rank of the Low-rank approximation has been under r.")
        #     break
        T_inv = np.linalg.inv(T)
        _mid = (R1.dot(T_inv)).dot(R2.T)
        mid_inv = np.linalg.inv(_mid)
        mid = np.ones_like(mid_inv)
        for _ in range(2*q+1):
            mid = mid.dot(mid_inv)
        Lt = (Q1.dot(mid)).dot(Q2.T)
        ## update St
        _St = X-Lt
        _St_vec = _St.reshape(-1)
        St_vec = St.reshape(-1)
        idx = abs(_St_vec).argsort()[::-1]
        St_vec[idx[:k]] = _St_vec[idx[:k]] # K largest entries of |X-Lt|
        St = St_vec.reshape(St.shape)
        ## print error information
        Reconstructed_X = Lt+St
        cur_err = np.linalg.norm(x=X-Reconstructed_X, ord='fro')
        print(f"[INFO] iter: [{t}/{max_iter}], error: {cur_err}, target_error: {desired_err}.")
        t += 1
    print("[INFO] GoDec finished!")
    return Lt, St


# if __name__=="__main__":
#     X = np.random.randn(65536, 200)
#     L,S = godec_original(X, r=20, k=5000)

