"""
Part E: Neighbourhood CF recast as a regression problem.

The idea
--------
Classical item-based CF *asserts* the weights: it uses the similarity s(i,j) as the
interpolation weight, normalised by sum |s|.  That choice is a heuristic.  Nothing in
the data says a correlation of 0.6 should carry weight 0.6.

The regression view instead *learns* the weights.  Fix a target item i and its
neighbourhood N(i).  Treat every user u who rated i as one training example:

    target   y_u  = r_ui - b_ui
    features z_uj = r_uj - b_uj   for j in N(i)   (0 if u never rated j)

and fit

    min_w  sum_{u in U_i} ( y_u - sum_{j in N(i)} w_ij z_uj )^2  +  lambda ||w||^2

whose stationarity condition is the regularised normal equation

    (Z^T Z + lambda I) w = Z^T y            ->      w = (Z^T Z + lambda I)^{-1} Z^T y

solved exactly, once per item, with a k x k dense solve.  Prediction is then

    rhat_ui = b_ui + sum_{j in N(i)} w_ij (r_uj - b_uj)

Collecting the per-item solutions into a sparse matrix W (column i holds w_i) turns the
whole prediction step into a single sparse product:

    P = B + D W ,   D = deviation matrix (r - b on observed cells, 0 elsewhere)

Statistical interpretation -- and why this differs from similarity
------------------------------------------------------------------
s(i,j) is a *marginal* association: how item j relates to item i ignoring every other
item.  w_ij is a *partial* regression coefficient: how j relates to i **holding the
other neighbours fixed**.  Three consequences that the experiments confirm:

1. Redundancy is discounted.  If two neighbours are near-duplicates (Star Wars /
   The Empire Strikes Back), each has a high marginal similarity, but the regression
   splits one coefficient between them -- their *joint* information is counted once.
2. Coefficients can go negative even for positively-similar items.  That is a genuine
   suppressor effect, not noise: once the strong neighbours are in the model, a
   remaining item may carry information mostly about the part of the signal already
   explained, and the least-squares fit subtracts it.
3. Weights are not constrained to sum to one, so the model can express "this
   neighbourhood only weakly determines the target" by shrinking everything toward 0 --
   something the sum-|s|-normalised heuristic literally cannot do, because it always
   normalises to a full-strength prediction no matter how weak the evidence.

Regularisation lambda controls the bias/variance trade-off: U_i can be as small as a
handful of users while k is 20-50, so the unregularised problem is often
under-determined and ridge is what makes the estimator usable at all.

The zero-imputation of unobserved neighbour ratings ("this user is at baseline on j")
is the standard choice; it is what allows a fixed k-dimensional design matrix instead of
a different regression per missing-data pattern, and it is unbiased under the modelling
assumption that an unobserved rating is expected to equal its baseline.

Complexity
----------
    per item : O(|U_i| k^2 + k^3)   ->   sum_i |U_i| k^2 = nnz * k^2
    ML-100K with k=40 : 70,771 * 1,600 ~ 1.1e8 flops, plus 1,682 solves of 40x40.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from .base import Recommender
from .baselines import BiasBaseline
from .similarity import build_similarity


class RegressionCF(Recommender):
    def __init__(self, k: int = 40, lam: float = 25.0, similarity: str = "adjusted_cosine",
                 min_support: int = 3, beta: float | None = 25.0,
                 weight_mode: str = "learned", rank_mode: str = "score",
                 lam_bias: float = 10.0, like_threshold: float = 4.0):
        super().__init__()
        self.k = int(k)
        self.lam = float(lam)
        self.similarity = similarity
        self.min_support = int(min_support)
        self.beta = beta
        self.weight_mode = weight_mode      # 'learned' | 'similarity'
        self.rank_mode = rank_mode
        self.lam_bias = lam_bias
        self.like_threshold = float(like_threshold)
        self.name = f"RegCF[{weight_mode},k={k},lam={lam:g}]"

    # ---------------------------------------------------------------- fit
    def _fit(self, R: sp.csr_matrix) -> None:
        n_users, n_items = R.shape

        # 1. Baseline b_ui = mu + b_u + b_i, learned on train only.
        self.bias = BiasBaseline(lam_u=self.lam_bias, lam_i=self.lam_bias).fit(R)
        self.B = self.bias.baseline_matrix()

        # 2. Deviation matrix D (dense; 943 x 1682 float64 = 12.7 MB).
        self.D = np.zeros((n_users, n_items))
        coo = R.tocoo()
        self.D[coo.row, coo.col] = coo.data - self.B[coo.row, coo.col]
        self.observed = np.zeros((n_users, n_items), dtype=bool)
        self.observed[coo.row, coo.col] = True
        # Binary "liked" matrix, used only by the implicit-affinity ranking head so that
        # the regression model can be compared to SLIM / graph on equal footing.
        keep = coo.data >= self.like_threshold
        self.X_pos = sp.csr_matrix(
            (np.ones(int(keep.sum())), (coo.row[keep], coo.col[keep])), shape=(n_users, n_items))

        # 3. Neighbourhoods from a similarity (used only to *select* N(i), not to weight).
        user_means = np.asarray(R.sum(axis=1)).ravel() / np.maximum(np.diff(R.indptr), 1)
        self.S, self.n_co = build_similarity(
            R.T.tocsr(), self.similarity, min_support=self.min_support,
            beta=self.beta, user_means_for_adjusted=user_means,
        )

        S_sel = self.S.copy()
        np.fill_diagonal(S_sel, -np.inf)
        k = min(self.k, n_items - 1)
        nbrs = np.argpartition(-S_sel, k - 1, axis=1)[:, :k]          # (n_items, k)
        # order neighbours by descending similarity for readable explanations
        ord_ = np.argsort(-np.take_along_axis(S_sel, nbrs, axis=1), axis=1)
        self.neighbors = np.take_along_axis(nbrs, ord_, axis=1)
        self.neighbor_sims = np.take_along_axis(self.S, self.neighbors, axis=1)

        # 4. Per-item ridge solve.
        Rc = R.tocsc()
        self.coef = np.zeros((n_items, k))
        self.n_train_users = np.zeros(n_items, dtype=np.int64)
        self.rank_deficient = 0

        for i in range(n_items):
            lo, hi = Rc.indptr[i], Rc.indptr[i + 1]
            U_i = Rc.indices[lo:hi]
            self.n_train_users[i] = U_i.size
            if U_i.size == 0:
                continue
            N_i = self.neighbors[i]
            Z = self.D[np.ix_(U_i, N_i)]                 # (|U_i|, k)
            y = self.D[U_i, i]                           # (|U_i|,)

            A = Z.T @ Z
            A.flat[:: k + 1] += self.lam                 # ridge on the diagonal
            b = Z.T @ y
            try:
                w = sla.solve(A, b, assume_a="pos")
            except (sla.LinAlgError, ValueError):
                w = np.linalg.lstsq(A, b, rcond=None)[0]
                self.rank_deficient += 1
            self.coef[i] = w

        if self.weight_mode == "similarity":
            # Ablation arm: the heuristic weights, same neighbourhoods, same baseline.
            den = np.abs(self.neighbor_sims).sum(axis=1, keepdims=True)
            self.coef = np.divide(self.neighbor_sims, np.maximum(den, 1e-12),
                                  out=np.zeros_like(self.neighbor_sims), where=den > 1e-12)

        # 5. Assemble the sparse weight matrix W with W[j, i] = w_ij.
        rows = self.neighbors.ravel()
        cols = np.repeat(np.arange(n_items), k)
        self.W = sp.csr_matrix((self.coef.ravel(), (rows, cols)), shape=(n_items, n_items))
        self.W.setdiag(0.0)
        self.W.eliminate_zeros()

    # ---------------------------------------------------------------- inference
    def _personalised(self) -> np.ndarray:
        return self.D @ self.W          # (n_users, n_items)

    def predict_all(self) -> np.ndarray:
        return self.B + np.asarray(self._personalised())

    def score_all(self) -> np.ndarray:
        if self.rank_mode == "rating":
            return self.predict_all()
        if self.rank_mode == "affinity":
            return np.asarray((self.X_pos @ self.W).todense())
        return np.asarray(self._personalised())

    def model_bytes(self) -> int:
        return self._nbytes(self.W, self.coef, self.neighbors, self.bias.b_u, self.bias.b_i)

    # ---------------------------------------------------------------- analysis
    def similarity_vs_coefficient(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flattened (similarity, learned coefficient, target item) over all neighbourhoods."""
        mask = self.n_train_users[:, None] > 0
        sims = self.neighbor_sims[np.broadcast_to(mask, self.neighbor_sims.shape)]
        coefs = self.coef[np.broadcast_to(mask, self.coef.shape)]
        items = np.repeat(np.arange(self.coef.shape[0]), self.coef.shape[1])
        items = items.reshape(self.coef.shape)[np.broadcast_to(mask, self.coef.shape)]
        return sims, coefs, items

    def explain(self, u: int, i: int, top_n: int = 5) -> list[dict]:
        """Learned coefficients weighted by this user's own signal on each neighbour.

        In the `affinity` ranking head the score is `x_u^+ W`, so the per-neighbour
        signal is the binary 'liked' indicator; in the rating head it is the baseline
        deviation.  The explanation follows whichever head is active.
        """
        N_i = self.neighbors[i]
        w = self.coef[i]
        if self.rank_mode == "affinity":
            z = np.asarray(self.X_pos[u, N_i].todense()).ravel()
        else:
            z = self.D[u, N_i] * self.observed[u, N_i]
        contrib = w * z
        idx = np.argsort(-np.abs(contrib))[:top_n]
        return [
            {
                "neighbor_item": int(N_i[j]),
                "learned_coef": float(w[j]),
                "similarity": float(self.neighbor_sims[i, j]),
                "user_signal": float(z[j]),
                "observed": bool(self.observed[u, N_i[j]]),
                "contribution": float(contrib[j]),
                "ranking_head": self.rank_mode,
            }
            for j in idx
        ]
