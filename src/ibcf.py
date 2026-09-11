"""
Part D: Item-Based Collaborative Filtering, from scratch.

Prediction rule (item-mean centred, the default)

                     sum_{j in N_k(i,u)} s(i,j) * (r_uj - mu_j)
    rhat_ui = mu_i + -------------------------------------------
                          sum_{j in N_k(i,u)} |s(i,j)|

Variants supported
------------------
    center_by = "item"  : centre by item mean  (shown above)
    center_by = "user"  : centre by user mean  -> rhat = mu_u + sum s (r_uj - mu_u)/sum|s|
                          This is the natural partner of adjusted-cosine similarity.
    mean_center = False : raw weighted average, rhat = sum s r_uj / sum |s|

N_k(i,u) = the k items most similar to i **among the items u has actually rated** --
again the exact per-(u,i) neighbourhood.

Why UBCF and IBCF behave differently
------------------------------------
1. *Shape of the evidence.*  MovieLens 100K has 943 users and 1682 items, so a user
   profile averages ~75 ratings while an item profile averages ~42.  But item-item
   co-rating counts concentrate on a small popular core, which makes item similarities
   estimated from far more co-observations -- and therefore far more stable -- than user
   similarities, which are built from short, idiosyncratic profiles.
2. *Stability over time.*  Item-item relations are close to stationary; user tastes and
   user sets churn.  This is why industrial systems standardised on item-item.
3. *Failure modes.*  UBCF degrades when a user has a short profile (few reliable
   neighbours).  IBCF degrades on unpopular *items* (few reliable neighbours for the
   target item), and it is structurally more conservative: it recommends things close to
   what the user already consumed, which lowers novelty and catalogue coverage.
4. *Cost.*  UBCF's similarity matrix is |U|^2 = 889k entries; IBCF's is |I|^2 = 2.83M.
   Here IBCF costs more; in a real catalogue with |U| >> |I| the ordering reverses, and
   item-item wins because the model can be precomputed offline and is stable between
   refreshes.

Complexity
----------
    similarity : five sparse GEMMs over R^T (the exact co-rated Pearson identity), giving
                 an O(|I|^2) dense matrix -- 1682^2 * 8 B = 22.6 MB
    prediction : sum_u |I| * |I_u| = |I| * nnz = 1,682 * 70,771 ~ 1.2e8 element ops,
                 executed as 943 vectorised argpartition calls (one per user), never as a
                 Python loop over (u, i) pairs
    memory     : one |I| x |I| similarity matrix plus three |U| x |I| output buffers
Scalability: the |I|^2 dense similarity is the binding constraint (~80 GB at 10^5 items).
The production form replaces it with a truncated top-k neighbour list built by approximate
nearest-neighbour search over the item vectors, which is O(|I| * k) in memory and is what
makes item-item CF deployable at catalogue scale.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender
from .similarity import build_similarity


class IBCF(Recommender):
    def __init__(self, k: int = 30, similarity: str = "cosine", mean_center: bool = True,
                 center_by: str = "item", min_support: int = 3, beta: float | None = 25.0,
                 rank_mode: str = "affinity", like_threshold: float = 4.0):
        super().__init__()
        self.k = int(k)
        self.similarity = similarity
        self.mean_center = bool(mean_center)
        self.center_by = center_by
        self.min_support = int(min_support)
        self.beta = beta
        self.rank_mode = rank_mode
        self.like_threshold = float(like_threshold)
        tag = f"{'mc-' + center_by if mean_center else 'raw'}"
        self.name = f"IBCF[{similarity},{tag},k={k}]"

    # ---------------------------------------------------------------- fit
    def _fit(self, R: sp.csr_matrix) -> None:
        counts_u = np.diff(R.indptr).astype(np.float64)
        sums_u = np.asarray(R.sum(axis=1)).ravel()
        self.mu = float(R.data.mean())
        self.user_mean = np.where(counts_u > 0, sums_u / np.maximum(counts_u, 1), self.mu)

        Rc = R.tocsc()
        counts_i = np.diff(Rc.indptr).astype(np.float64)
        sums_i = np.asarray(R.sum(axis=0)).ravel()
        self.item_mean = np.where(counts_i > 0, sums_i / np.maximum(counts_i, 1), self.mu)

        # Rows of the matrix passed to build_similarity must be *items*.
        self.S, self.n_co = build_similarity(
            R.T.tocsr(), self.similarity, min_support=self.min_support, beta=self.beta,
            user_means_for_adjusted=self.user_mean,
        )
        self._num = self._den = self._aff = None

    # ---------------------------------------------------------------- inference
    def _centre(self) -> np.ndarray:
        if not self.mean_center:
            return None
        return self.item_mean if self.center_by == "item" else self.user_mean

    def _accumulate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(numerator, denominator, affinity) -- one pass, three heads (see UBCF)."""
        if self._num is not None:
            return self._num, self._den, self._aff

        R = self._R.tocsr()
        n_users, n_items = R.shape
        num = np.zeros((n_users, n_items))
        den = np.zeros((n_users, n_items))
        aff = np.zeros((n_users, n_items))

        S_sel = self.S.copy()
        np.fill_diagonal(S_sel, -np.inf)

        k = self.k
        for u in range(n_users):
            lo, hi = R.indptr[u], R.indptr[u + 1]
            rated = R.indices[lo:hi]
            if rated.size == 0:
                continue
            vals = R.data[lo:hi]
            if not self.mean_center:
                contrib = vals
            elif self.center_by == "item":
                contrib = vals - self.item_mean[rated]
            else:
                contrib = vals - self.user_mean[u]
            liked = (vals >= self.like_threshold).astype(np.float64)

            Ssub = S_sel[:, rated]                       # (n_items, m)
            if rated.size > k:
                top = np.argpartition(-Ssub, k - 1, axis=1)[:, :k]
                w = np.take_along_axis(Ssub, top, axis=1)
                c = contrib[top]
                lk = liked[top]
            else:
                w = Ssub
                c = np.broadcast_to(contrib, Ssub.shape)
                lk = np.broadcast_to(liked, Ssub.shape)

            w = np.where(np.isneginf(w), 0.0, w)
            num[u, :] = np.einsum("ij,ij->i", w, c)
            den[u, :] = np.abs(w).sum(axis=1)
            aff[u, :] = np.einsum("ij,ij->i", np.maximum(w, 0.0), lk)

        self._num, self._den, self._aff = num, den, aff
        return num, den, aff

    def predict_all(self) -> np.ndarray:
        num, den, _ = self._accumulate()
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den > 1e-10, num / np.maximum(den, 1e-10), 0.0)
        if not self.mean_center:
            return np.where(den > 1e-10, frac, self.user_mean[:, None])
        if self.center_by == "item":
            base = np.broadcast_to(self.item_mean[None, :], frac.shape)
        else:
            base = np.broadcast_to(self.user_mean[:, None], frac.shape)
        return base + frac

    def score_all(self) -> np.ndarray:
        if self.rank_mode == "rating":
            return self.predict_all()
        num, _, aff = self._accumulate()
        return aff if self.rank_mode == "affinity" else num

    def model_bytes(self) -> int:
        return self._nbytes(self.S, self.item_mean, self.user_mean)

    # ---------------------------------------------------------------- explainability
    def explain(self, u: int, i: int, top_n: int = 5) -> list[dict]:
        """The user's own previously-rated movies that pushed item i up or down.

        Contributions are computed for the *active ranking head* so the explanation
        describes the quantity that actually produced the ordering.
        """
        R = self._R.tocsr()
        lo, hi = R.indptr[u], R.indptr[u + 1]
        rated, vals = R.indices[lo:hi], R.data[lo:hi]
        if rated.size == 0:
            return []
        s = self.S[i, rated].copy()
        s[rated == i] = 0.0
        order = np.argsort(-s)[: self.k]
        rated, vals, s = rated[order], vals[order], s[order]
        if self.rank_mode == "affinity":
            contrib = np.maximum(s, 0.0) * (vals >= self.like_threshold)
        elif not self.mean_center:
            contrib = s * vals
        elif self.center_by == "item":
            contrib = s * (vals - self.item_mean[rated])
        else:
            contrib = s * (vals - self.user_mean[u])
        den = np.abs(s).sum()
        idx = np.argsort(-np.abs(contrib))[:top_n]
        return [
            {
                "neighbor_item": int(rated[j]),
                "similarity": float(s[j]),
                "user_rating": float(vals[j]),
                "item_mean": float(self.item_mean[rated[j]]),
                "contribution": float(contrib[j]),
                "share_of_numerator": float(contrib[j] / den) if den > 0 else 0.0,
                "co_ratings": int(self.n_co[i, rated[j]]),
                "ranking_head": self.rank_mode,
            }
            for j in idx
        ]
