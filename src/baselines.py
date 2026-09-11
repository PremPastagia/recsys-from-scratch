"""
Part B: non-personalised and bias-only baselines.

Why baselines matter
--------------------
A recommender's absolute RMSE is meaningless on its own.  On a 1-5 scale, simply
predicting each user's own mean already achieves an RMSE around 1.04; a sophisticated
model that reports 1.00 has bought very little.  Likewise for ranking: because item
popularity in MovieLens is extremely skewed, "recommend the most-rated movies to
everyone" is a genuinely hard-to-beat Top-10 baseline.  Baselines convert an absolute
number into a *decision*: they tell us how much of a model's performance comes from the
personalisation we paid for, and how much we would have got for free.

Models
------
    GlobalMean     rhat_ui = mu
    UserMean       rhat_ui = mu_u              (global mean if the user is empty)
    ItemMean       rhat_ui = mu_i              (global mean if the item is empty)
    BiasBaseline   rhat_ui = mu + b_u + b_i    (regularised, alternating closed form)
    Popularity     score_ui = |{v : r_vi observed}|
    PopularityPos  score_ui = |{v : r_vi >= tau}|      (popularity among *liked* ratings)
    RandomRec      score_ui ~ U(0,1)           (floor reference for ranking metrics)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


def _col_mean(R: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(R.getnnz(axis=0)).ravel().astype(np.float64)
    sums = np.asarray(R.sum(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return means, counts


def _row_mean(R: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(R.getnnz(axis=1)).ravel().astype(np.float64)
    sums = np.asarray(R.sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return means, counts


class GlobalMean(Recommender):
    name = "GlobalMean"

    def _fit(self, R):
        self.mu = float(R.data.mean())

    def predict_all(self):
        return np.full(self._R.shape, self.mu)

    def model_bytes(self):
        return 8


class UserMean(Recommender):
    name = "UserMean"

    def _fit(self, R):
        self.mu = float(R.data.mean())
        m, _ = _row_mean(R)
        self.user_mean = np.where(np.isnan(m), self.mu, m)

    def predict_all(self):
        return np.repeat(self.user_mean[:, None], self._R.shape[1], axis=1)

    def model_bytes(self):
        return self._nbytes(self.user_mean)


class ItemMean(Recommender):
    name = "ItemMean"

    def _fit(self, R):
        self.mu = float(R.data.mean())
        m, _ = _col_mean(R)
        self.item_mean = np.where(np.isnan(m), self.mu, m)

    def predict_all(self):
        return np.repeat(self.item_mean[None, :], self._R.shape[0], axis=0)

    def model_bytes(self):
        return self._nbytes(self.item_mean)


class BiasBaseline(Recommender):
    """rhat_ui = mu + b_u + b_i with L2-regularised, alternating closed-form updates.

        b_i <- sum_{u in U_i} (r_ui - mu - b_u) / (lambda_i + |U_i|)
        b_u <- sum_{i in I_u} (r_ui - mu - b_i) / (lambda_u + |I_u|)

    Each update is the exact minimiser of the regularised squared error holding the
    other bias fixed, so the alternation is block coordinate descent on a convex
    objective and converges monotonically.  The denominators shrink biases of rarely
    observed users/items toward zero, which is exactly the behaviour we want on a
    93.7%-sparse matrix.
    """

    name = "BiasBaseline"

    def __init__(self, lam_u: float = 10.0, lam_i: float = 10.0, n_iter: int = 20):
        super().__init__()
        self.lam_u, self.lam_i, self.n_iter = lam_u, lam_i, n_iter

    def _fit(self, R):
        R = R.tocsr()
        self.mu = float(R.data.mean())
        n_users, n_items = R.shape
        b_u = np.zeros(n_users)
        b_i = np.zeros(n_items)

        coo = R.tocoo()
        rows, cols, vals = coo.row, coo.col, coo.data
        cnt_u = np.bincount(rows, minlength=n_users).astype(float)
        cnt_i = np.bincount(cols, minlength=n_items).astype(float)

        self.trace = []
        for _ in range(self.n_iter):
            resid = vals - self.mu - b_u[rows]
            b_i = np.bincount(cols, weights=resid, minlength=n_items) / (self.lam_i + cnt_i)
            resid = vals - self.mu - b_i[cols]
            b_u = np.bincount(rows, weights=resid, minlength=n_users) / (self.lam_u + cnt_u)
            err = vals - (self.mu + b_u[rows] + b_i[cols])
            self.trace.append(float(np.sqrt(np.mean(err ** 2))))

        self.b_u, self.b_i = b_u, b_i

    def baseline_matrix(self) -> np.ndarray:
        return self.mu + self.b_u[:, None] + self.b_i[None, :]

    def predict_all(self):
        return self.baseline_matrix()

    def model_bytes(self):
        return self._nbytes(self.b_u, self.b_i)


class Popularity(Recommender):
    """Non-personalised Top-K by raw training interaction count."""

    name = "Popularity"

    def _fit(self, R):
        self.counts = np.asarray(R.getnnz(axis=0)).ravel().astype(np.float64)
        self.mu = float(R.data.mean())
        m, _ = _col_mean(R)
        self.item_mean = np.where(np.isnan(m), self.mu, m)

    def predict_all(self):
        # For RMSE purposes a pure popularity ranker has no rating model; the least
        # arbitrary rating it can offer is the item mean.
        return np.repeat(self.item_mean[None, :], self._R.shape[0], axis=0)

    def score_all(self):
        return np.repeat(self.counts[None, :], self._R.shape[0], axis=0)

    def model_bytes(self):
        return self._nbytes(self.counts)


class PopularityPositive(Recommender):
    """Popularity counted only over *liked* ratings (r >= tau).

    This corrects popularity's blind spot: a widely-watched but poorly-received film
    ranks high on raw counts.  Counting only positives ranks by "how many people
    actually liked it", which is a stronger and fairer non-personalised baseline.
    """

    name = "PopularityPositive"

    def __init__(self, tau: float = 4.0):
        super().__init__()
        self.tau = tau

    def _fit(self, R):
        X = R.tocoo()
        keep = X.data >= self.tau
        self.pos_counts = np.bincount(X.col[keep], minlength=R.shape[1]).astype(float)
        self.mu = float(R.data.mean())
        m, _ = _col_mean(R)
        self.item_mean = np.where(np.isnan(m), self.mu, m)

    def predict_all(self):
        return np.repeat(self.item_mean[None, :], self._R.shape[0], axis=0)

    def score_all(self):
        return np.repeat(self.pos_counts[None, :], self._R.shape[0], axis=0)

    def model_bytes(self):
        return self._nbytes(self.pos_counts)


class RandomRec(Recommender):
    """Uniform-random scoring: the floor that any real model must clear."""

    name = "Random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed

    def _fit(self, R):
        self.mu = float(R.data.mean())
        self._rng = np.random.default_rng(self.seed)
        self._S = self._rng.random(R.shape)

    def predict_all(self):
        return np.full(self._R.shape, self.mu)

    def score_all(self):
        return self._S

    def model_bytes(self):
        return self._nbytes(self._S)
