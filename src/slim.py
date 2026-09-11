"""
Part F: SLIM-style sparse linear item-item recommender, solved from scratch.

Model
-----
Let X be the binary implicit-feedback matrix (users x items).  SLIM learns an item-item
coefficient matrix W by asking every item column to reconstruct itself from the *other*
item columns:

    min_W  (1/2) || X - X W ||_F^2  +  lambda_1 ||W||_1  +  (lambda_2 / 2) ||W||_F^2
    s.t.   W >= 0 ,   diag(W) = 0

* ``diag(W) = 0`` is essential: without it the trivial solution W = I reconstructs X
  perfectly and learns nothing.
* ``W >= 0`` makes every coefficient an interpretable "item j is evidence *for* item i"
  weight and empirically improves top-N accuracy on implicit data.
* The L1 term drives most coefficients to exactly zero -> a sparse, fast, inspectable
  model.  The L2 term keeps the problem strictly convex and stops correlated duplicates
  from fighting over a single coefficient.

Recommendation score for user u:  s_u = x_u W  (a row vector over items).

Optimisation -- projected FISTA, implemented here
-------------------------------------------------
Split the objective into a smooth part f and a non-smooth part g:

    f(W) = (1/2)||X - XW||_F^2 + (lambda_2/2)||W||_F^2
    g(W) = lambda_1 ||W||_1  +  indicator(W >= 0)  +  indicator(diag(W) = 0)

With the Gram matrix G = X^T X the gradient of the smooth part is exactly

    grad f(W) = X^T (X W - X) + lambda_2 W = G W - G + lambda_2 W

so a single dense GEMM per iteration covers all 1,682 column problems at once.  The
Lipschitz constant of grad f is  L = lambda_max(G) + lambda_2, obtained by power
iteration, giving the step size eta = 1/L.

Because W >= 0, the L1 penalty is linear on the feasible set and the proximal operator
of g collapses to a *shifted* projection:

    prox_{eta g}(V) = max(0, V - eta * lambda_1)   followed by  zeroing the diagonal

FISTA then adds Nesterov momentum, improving the rate from O(1/t) to O(1/t^2):

    W_t = prox( Y_t - eta grad f(Y_t) )
    t_{t+1} = (1 + sqrt(1 + 4 t_t^2)) / 2
    Y_{t+1} = W_t + ((t_t - 1)/t_{t+1}) (W_t - W_{t-1})

Optimality certificate (KKT)
----------------------------
At a minimiser of this convex problem, with  g = grad f(W)  (off-diagonal entries):

    W_ij  > 0   =>   g_ij + lambda_1  = 0
    W_ij  = 0   =>   g_ij + lambda_1 >= 0

``kkt_residual`` measures the worst violation of those two conditions.  It is an
independent certificate: it is computed from G and W alone, so it will catch a solver
that merely *stopped* rather than converged.

Complexity
----------
    Gram      : O(nnz * avg_item_degree) sparse GEMM, O(|I|^2) memory (1682^2 -> 22.6 MB)
    per iter  : one |I| x |I| x |I| dense GEMM  ~  4.8 GFLOP
    memory    : 4 dense |I| x |I| float64 buffers  ~  90 MB
Scalability note: the dense-Gram formulation is what makes 1,682 items trivial and
100,000 items impossible.  At catalogue scale one restricts each column's feature set to
its top-M co-occurring items (the ``feature_mask`` option below) and solves the columns
independently, which is embarrassingly parallel and reduces memory to O(|I| * M).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .base import Recommender


# ======================================================================================
# Solver
# ======================================================================================
def power_iteration_lmax(G: np.ndarray, iters: int = 100, seed: int = 0) -> float:
    """Largest eigenvalue of the symmetric PSD Gram matrix, by power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(G.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = G @ v
        nw = np.linalg.norm(w)
        if nw < 1e-15:
            return 0.0
        v = w / nw
        lam = float(v @ (G @ v))
    return lam


def slim_fista(G: np.ndarray, l1: float, l2: float, *, max_iter: int = 3000,
               tol: float = 1e-11, kkt_tol: float = 1e-4, check_every: int = 25,
               feature_mask: np.ndarray | None = None,
               verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Solve the non-negative elastic-net SLIM problem by projected FISTA.

    Parameters
    ----------
    G : (n_items, n_items) Gram matrix X^T X.
    feature_mask : optional boolean matrix; False entries are pinned to zero
        (candidate-neighbour restriction used for the scalability study).
    """
    n = G.shape[0]
    L = power_iteration_lmax(G) + l2
    eta = 1.0 / max(L, 1e-12)

    W = np.zeros((n, n))
    Y = np.zeros((n, n))
    t = 1.0
    history = []
    t0 = time.perf_counter()

    def project(V: np.ndarray) -> np.ndarray:
        V = np.maximum(V - eta * l1, 0.0)
        np.fill_diagonal(V, 0.0)
        if feature_mask is not None:
            V *= feature_mask
        return V

    prev_obj = np.inf
    rel = np.inf
    kkt_rel = np.inf
    for it in range(max_iter):
        grad = G @ Y - G + l2 * Y
        W_new = project(Y - eta * grad)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = W_new + ((t - 1.0) / t_new) * (W_new - W)

        # Objective: 0.5||X - XW||^2 = 0.5(tr(G) - 2 tr(W^T G) + tr(W^T G W))
        GW = G @ W_new
        smooth = 0.5 * (np.trace(G) - 2.0 * np.sum(W_new * G) + np.sum(W_new * GW))
        obj = smooth + l1 * W_new.sum() + 0.5 * l2 * np.sum(W_new ** 2)
        history.append(float(obj))

        rel = np.inf if not np.isfinite(prev_obj) else \
            abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
        W, t, prev_obj = W_new, t_new, obj

        # Stop on the *optimality certificate*, not merely on a stalled objective:
        # a plateau in the objective is necessary but not sufficient for optimality.
        if (it + 1) % check_every == 0 or it == max_iter - 1:
            kkt_rel = kkt_residual(G, W, l1, l2, feature_mask)["relative_violation"]
            if verbose:
                print(f"  it={it:4d} obj={obj:.6e} rel={rel:.3e} kkt={kkt_rel:.3e}")
            if kkt_rel < kkt_tol:
                break
        if rel < tol and it > 50:
            # The objective has plateaued. Re-measure the certificate so the reported
            # residual always corresponds to the iterate actually returned, rather than
            # to whatever the last scheduled check happened to see.
            kkt_rel = kkt_residual(G, W, l1, l2, feature_mask)["relative_violation"]
            break

    info = {
        "iterations": it + 1,
        "lipschitz": float(L),
        "step": float(eta),
        "objective": float(prev_obj),
        "history": history,
        "solve_time_s": time.perf_counter() - t0,
        "kkt_relative_violation": float(kkt_rel),
        "converged": bool(kkt_rel < kkt_tol),
    }
    return W, info


def kkt_residual(G: np.ndarray, W: np.ndarray, l1: float, l2: float,
                 feature_mask: np.ndarray | None = None) -> dict:
    """Worst KKT violation of the non-negative elastic-net stationarity conditions.

    ``feature_mask`` matters for correctness, not just speed.  A masked-out cell carries
    an extra equality constraint ``W_ij = 0`` with its own multiplier, so no stationarity
    condition applies to it: judging such a cell by the *unmasked* conditions reports a
    violation for a coefficient that is optimal for the problem actually being solved,
    and a solver stopping on that residual can never converge.  Cells outside the mask
    are therefore excluded from both conditions.
    """
    grad = G @ W - G + l2 * W
    s = grad + l1
    off = ~np.eye(G.shape[0], dtype=bool)
    if feature_mask is not None:
        off = off & feature_mask
    active = (W > 0) & off
    inactive = (W <= 0) & off

    viol_active = np.abs(s[active]) if active.any() else np.array([0.0])
    viol_inactive = np.maximum(-s[inactive], 0.0) if inactive.any() else np.array([0.0])
    scale = max(np.abs(G).max(), 1.0)
    return {
        "max_violation_active": float(viol_active.max()),
        "max_violation_inactive": float(viol_inactive.max()),
        "max_violation": float(max(viol_active.max(), viol_inactive.max())),
        "relative_violation": float(max(viol_active.max(), viol_inactive.max()) / scale),
        "gram_scale": float(scale),
    }


# ======================================================================================
# Recommender wrapper
# ======================================================================================
class SLIM(Recommender):
    # The rating head is an increasing affine map of the ranking score, so 'rating' and
    # 'score' induce the identical ordering; there is nothing to tune.
    rank_modes = ("score",)

    def __init__(self, l1: float = 5.0, l2: float = 50.0, threshold: float = 4.0,
                 max_iter: int = 3000, tol: float = 1e-11, kkt_tol: float = 1e-4,
                 top_m_features: int | None = None, rank_mode: str = "score"):
        super().__init__()
        self.l1, self.l2 = float(l1), float(l2)
        self.threshold = float(threshold)
        self.max_iter, self.tol, self.kkt_tol = max_iter, tol, kkt_tol
        self.top_m_features = top_m_features
        self.rank_mode = rank_mode
        self.name = f"SLIM[l1={l1:g},l2={l2:g}]"

    # ---------------------------------------------------------------- fit
    def _fit(self, R: sp.csr_matrix) -> None:
        X = R.copy().tocoo()
        keep = X.data >= self.threshold
        self.X = sp.csr_matrix(
            (np.ones(int(keep.sum())), (X.row[keep], X.col[keep])), shape=R.shape
        )
        self.G = np.asarray((self.X.T @ self.X).todense(), dtype=np.float64)

        mask = None
        if self.top_m_features:
            mask = self._topm_mask(self.G, self.top_m_features)
        self.feature_mask = mask

        self.W, self.info = slim_fista(
            self.G, self.l1, self.l2, max_iter=self.max_iter, tol=self.tol,
            kkt_tol=self.kkt_tol, feature_mask=mask,
        )
        self.W_sparse = sp.csr_matrix(self.W)
        self.kkt = kkt_residual(self.G, self.W, self.l1, self.l2, mask)

        # Provisional rating head from train; `calibrate()` replaces it with a
        # held-out fit, which is the one actually used for reported RMSE.
        self._fit_rating_head(R)

    @staticmethod
    def _topm_mask(G: np.ndarray, m: int) -> np.ndarray:
        n = G.shape[0]
        Gs = G.copy()
        np.fill_diagonal(Gs, -np.inf)
        idx = np.argpartition(-Gs, min(m, n - 1) - 1, axis=0)[: min(m, n - 1)]
        mask = np.zeros((n, n), dtype=bool)
        cols = np.broadcast_to(np.arange(n), idx.shape)
        mask[idx, cols] = True
        return mask

    def _fit_rating_head(self, R: sp.csr_matrix) -> None:
        coo = R.tocoo()
        self.calibrate(coo.row, coo.col, coo.data)

    def calibrate(self, u, i, r):
        """Least-squares fit of  rating ~ a + b * score  on the supplied pairs."""
        S = self.score_all()
        x = S[np.asarray(u, int), np.asarray(i, int)]
        y = np.asarray(r, float)
        A = np.column_stack([np.ones_like(x), x])
        self.rating_head, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    # ---------------------------------------------------------------- inference
    def score_all(self) -> np.ndarray:
        return np.asarray(self.X @ self.W)

    def predict_all(self) -> np.ndarray:
        a, b = self.rating_head
        return a + b * self.score_all()

    def model_bytes(self) -> int:
        return self._nbytes(self.W_sparse)

    # ---------------------------------------------------------------- persistence
    def __getstate__(self):
        """Drop the two large recomputable dense buffers (G, feature_mask) when pickling."""
        state = self.__dict__.copy()
        state["G"] = None
        state["feature_mask"] = None
        state["_R"] = None
        return state

    def restore_gram(self) -> None:
        if self.G is None:
            self.G = np.asarray((self.X.T @ self.X).todense(), dtype=np.float64)

    # ---------------------------------------------------------------- diagnostics
    def sparsity(self) -> dict:
        n = self.W.shape[0]
        offdiag = n * n - n
        nnz = int((self.W > 0).sum())
        per_col = (self.W > 0).sum(axis=0)
        return {
            "nnz": nnz,
            "offdiag_cells": offdiag,
            "sparsity_pct": 100.0 * (1.0 - nnz / offdiag),
            "density_pct": 100.0 * nnz / offdiag,
            "mean_nnz_per_column": float(per_col.mean()),
            "median_nnz_per_column": float(np.median(per_col)),
            "max_nnz_per_column": int(per_col.max()),
            "empty_columns": int((per_col == 0).sum()),
        }

    def strongest_pairs(self, top_n: int = 20) -> list[tuple[int, int, float]]:
        """The largest learned item-item coefficients W[j, i]: 'j predicts i'."""
        Wc = sp.coo_matrix(self.W_sparse)
        order = np.argsort(-Wc.data)[:top_n]
        return [(int(Wc.row[o]), int(Wc.col[o]), float(Wc.data[o])) for o in order]

    def explain(self, u: int, i: int, top_n: int = 5) -> list[dict]:
        """Which of the user's liked items contributed to item i's score."""
        xu = self.X[u].toarray().ravel()
        liked = np.nonzero(xu)[0]
        if liked.size == 0:
            return []
        w = self.W[liked, i]
        idx = np.argsort(-w)[:top_n]
        return [
            {"source_item": int(liked[j]), "coefficient": float(w[j]),
             "contribution": float(w[j])}
            for j in idx if w[j] > 0
        ]
