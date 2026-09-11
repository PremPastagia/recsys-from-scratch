"""
Part C: User-Based Collaborative Filtering, from scratch.

Prediction rule
---------------
With mean-centring (the default, and the textbook form):

                     sum_{v in N_k(u,i)} s(u,v) * (r_vi - mu_v)
    rhat_ui = mu_u + -------------------------------------------
                          sum_{v in N_k(u,i)} |s(u,v)|

Without mean-centring (raw weighted average):

              sum_{v in N_k(u,i)} s(u,v) * r_vi
    rhat_ui = -----------------------------------
              sum_{v in N_k(u,i)} |s(u,v)|

where N_k(u,i) = the k users most similar to u **among those who actually rated i**.
Note this is the *exact* per-(u,i) neighbourhood, not the cheaper "global top-k
neighbours of u, then keep whoever happened to rate i" approximation: the latter
silently shrinks the effective k for unpopular items and is a common source of
optimistic k-curves.

Robustness
----------
* |s| in the denominator so a negatively-correlated neighbour pushes the prediction
  the right way without the normaliser cancelling.
* If the denominator is 0 (no usable neighbour rated i) we fall back to mu_u -- the
  best available evidence -- rather than emitting NaN.
* Similarities are zeroed below `min_support` co-ratings and shrunk by significance
  weighting (see similarity.py).
* The self-similarity is excluded from every neighbourhood.

Ranking score
-------------
Ranking by rhat alone rewards items whose single supporting neighbour was enthusiastic.
We therefore also expose the *unnormalised* evidence score

    score_ui = sum_{v in N_k(u,i)} s(u,v) * (r_vi - mu_v)

which keeps the amount of supporting evidence in the numerator.  Which of the two is
used for Top-N is selected on the validation split, never assumed.

Complexity
----------
    similarity : O(|U| * nnz) worst case via sparse GEMM, O(|U|^2) memory   (943^2 -> 7 MB)
    prediction : sum_i |U| * |U_i| = |U| * nnz  =  943 * 70,771 ~ 6.7e7 element ops
                 done as 1,682 vectorised argpartition calls.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender
from .similarity import build_similarity


class UBCF(Recommender):
    def __init__(self, k: int = 30, similarity: str = "pearson", mean_center: bool = True,
                 min_support: int = 3, beta: float | None = 25.0,
                 rank_mode: str = "affinity", like_threshold: float = 4.0):
        super().__init__()
        self.k = int(k)
        self.similarity = similarity
        self.mean_center = bool(mean_center)
        self.min_support = int(min_support)
        self.beta = beta
        self.rank_mode = rank_mode
        self.like_threshold = float(like_threshold)
        self.name = (f"UBCF[{similarity},{'mc' if mean_center else 'raw'},k={k}]")

    # ---------------------------------------------------------------- fit
    def _fit(self, R: sp.csr_matrix) -> None:
        self.S, self.n_co = build_similarity(
            R, self.similarity, min_support=self.min_support, beta=self.beta
        )
        counts = np.diff(R.indptr).astype(np.float64)
        sums = np.asarray(R.sum(axis=1)).ravel()
        self.mu = float(R.data.mean())
        self.user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), self.mu)
        self._num = self._den = self._aff = None

    # ---------------------------------------------------------------- inference
    def _accumulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (numerator, denominator, affinity) matrices of shape (n_users, n_items).

        All three share the same expensive per-(u,i) neighbourhood selection, so they are
        accumulated in a single pass:
            num = sum_v s(u,v) * contrib_v        (rating numerator)
            den = sum_v |s(u,v)|                  (normaliser)
            aff = sum_v s(u,v) * 1[r_vi >= tau]   (implicit top-N affinity)
        """
        if self._num is not None:
            return self._num, self._den, self._aff

        R = self._R.tocsc()
        n_users, n_items = R.shape
        num = np.zeros((n_users, n_items))
        den = np.zeros((n_users, n_items))
        aff = np.zeros((n_users, n_items))

        # Selection copy: -inf on the diagonal guarantees a user is never its own
        # neighbour even when every genuine similarity is negative.
        S_sel = self.S.copy()
        np.fill_diagonal(S_sel, -np.inf)

        k = self.k
        for i in range(n_items):
            lo, hi = R.indptr[i], R.indptr[i + 1]
            raters = R.indices[lo:hi]
            if raters.size == 0:
                continue
            vals = R.data[lo:hi]
            contrib = vals - self.user_mean[raters] if self.mean_center else vals
            liked = (vals >= self.like_threshold).astype(np.float64)

            Ssub = S_sel[:, raters]                      # (n_users, m)
            if raters.size > k:
                top = np.argpartition(-Ssub, k - 1, axis=1)[:, :k]
                w = np.take_along_axis(Ssub, top, axis=1)
                c = contrib[top]
                lk = liked[top]
            else:
                w = Ssub
                c = np.broadcast_to(contrib, Ssub.shape)
                lk = np.broadcast_to(liked, Ssub.shape)

            w = np.where(np.isneginf(w), 0.0, w)         # drop the self slot
            num[:, i] = np.einsum("ij,ij->i", w, c)
            den[:, i] = np.abs(w).sum(axis=1)
            aff[:, i] = np.einsum("ij,ij->i", np.maximum(w, 0.0), lk)

        self._num, self._den, self._aff = num, den, aff
        return num, den, aff

    def predict_all(self) -> np.ndarray:
        num, den, _ = self._accumulate()
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den > 1e-10, num / np.maximum(den, 1e-10), 0.0)
        base = self.user_mean[:, None] if self.mean_center else 0.0
        pred = base + frac
        if not self.mean_center:
            # Nothing to fall back on in raw mode except the user's own mean.
            pred = np.where(den > 1e-10, pred, self.user_mean[:, None])
        return pred

    def score_all(self) -> np.ndarray:
        if self.rank_mode == "rating":
            return self.predict_all()
        num, _, aff = self._accumulate()
        return aff if self.rank_mode == "affinity" else num

    def model_bytes(self) -> int:
        return self._nbytes(self.S, self.user_mean)

    # ---------------------------------------------------------------- explainability
    def explain(self, u: int, i: int, top_n: int = 5) -> list[dict]:
        """The strongest contributing neighbours behind the score that ranked item i.

        The contribution is computed for the *active ranking head*, not for the rating
        head: an explanation that describes a quantity the model did not use to order the
        list is not an explanation of the ranking.
        """
        R = self._R.tocsc()
        lo, hi = R.indptr[i], R.indptr[i + 1]
        raters, vals = R.indices[lo:hi], R.data[lo:hi]
        if raters.size == 0:
            return []
        s = self.S[u, raters].copy()
        s[raters == u] = 0.0
        order = np.argsort(-s)[: self.k]
        raters, vals, s = raters[order], vals[order], s[order]
        if self.rank_mode == "affinity":
            contrib = np.maximum(s, 0.0) * (vals >= self.like_threshold)
        else:
            contrib = s * ((vals - self.user_mean[raters]) if self.mean_center else vals)
        den = np.abs(s).sum()
        idx = np.argsort(-np.abs(contrib))[:top_n]
        return [
            {
                "neighbor_u": int(raters[j]),
                "similarity": float(s[j]),
                "neighbor_rating": float(vals[j]),
                "neighbor_mean": float(self.user_mean[raters[j]]),
                "contribution": float(contrib[j]),
                "share_of_numerator": float(contrib[j] / den) if den > 0 else 0.0,
                "co_ratings": int(self.n_co[u, raters[j]]),
                "ranking_head": self.rank_mode,
            }
            for j in idx
        ]
