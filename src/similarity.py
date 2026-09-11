"""
Similarity computation, expressed entirely as sparse matrix algebra.

All similarities below are computed between the **rows** of a sparse matrix M whose
structural non-zeros are the observed ratings.  Pass R for user-user similarity and
R.T for item-item similarity.

1. Cosine
---------
                  sum_i r_ai r_bi
    cos(a,b) = ---------------------- ,   unobserved entries treated as 0
               ||r_a||_2 * ||r_b||_2

    Vectorised:  S = D^-1 (M M^T) D^-1  with  D = diag(||r_a||_2).

2. Exact co-rated Pearson
-------------------------
Textbook Pearson for CF is computed over the *co-rated* set I_ab = I_a ∩ I_b only:

                     n * S_ab - S_a S_b
    p(a,b) = ------------------------------------------- ,  n = |I_ab|
             sqrt(n Q_a - S_a^2) * sqrt(n Q_b - S_b^2)

    where all sums range over I_ab.  A naive implementation is O(n^2 * m) with a Python
    loop over pairs.  Every one of the five required sums is in fact a matrix product,
    because multiplying by the binary indicator B restricts a sum to the co-rated set:

        n    = B B^T            (co-rating counts)
        S_a  = M B^T            (sum of a's ratings over I_ab)
        S_b  = B M^T  = S_a^T
        S_ab = M M^T            (sum of products over I_ab)
        Q_a  = M2 B^T           (sum of a's squared ratings over I_ab), M2 = M elementwise^2
        Q_b  = B M2^T = Q_a^T

    So exact co-rated Pearson costs five sparse GEMMs -- no pairwise Python loop at all.
    This is the "clearly show the mathematics" version, not an approximation.

3. Adjusted cosine (item-item)
------------------------------
    Centre each rating by the *rating user's* mean before taking cosine over items:

                    sum_u (r_ui - mu_u)(r_uj - mu_u)
    adjcos(i,j) = ------------------------------------
                  sqrt(sum_u (r_ui-mu_u)^2) sqrt(sum_u (r_uj-mu_u)^2)

    This removes per-user rating scale (the "everything is a 4" user) which plain
    item-item cosine cannot see.

4. Reliability corrections
--------------------------
    * minimum support: a similarity backed by fewer than `min_support` co-ratings is
      set to 0 -- with 2 co-ratings Pearson is +-1 by construction and is pure noise.
    * significance weighting / shrinkage:  s' = s * min(n, beta) / beta.
      This is the standard Herlocker correction; statistically it is a crude shrinkage
      of the sample correlation toward 0 proportional to the evidence behind it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _binary(M: sp.csr_matrix) -> sp.csr_matrix:
    B = M.copy()
    B.data = np.ones_like(B.data)
    return B


def co_counts(M: sp.csr_matrix) -> np.ndarray:
    """n_ab = |I_a ∩ I_b| for every pair of rows."""
    B = _binary(M)
    return np.asarray((B @ B.T).todense(), dtype=np.float64)


def cosine_similarity(M: sp.csr_matrix) -> np.ndarray:
    """Row-wise cosine similarity, unobserved entries treated as zero."""
    M = M.tocsr()
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    inv = np.where(norms > 0, 1.0 / np.maximum(norms, 1e-12), 0.0)
    G = np.asarray((M @ M.T).todense(), dtype=np.float64)
    S = G * inv[:, None] * inv[None, :]
    np.fill_diagonal(S, 0.0)
    return S


def pearson_corated(M: sp.csr_matrix) -> np.ndarray:
    """Exact Pearson correlation restricted to co-rated entries (5 sparse GEMMs)."""
    M = M.tocsr().astype(np.float64)
    B = _binary(M)
    M2 = M.copy()
    M2.data = M2.data ** 2

    n = np.asarray((B @ B.T).todense(), dtype=np.float64)
    S_ab = np.asarray((M @ M.T).todense(), dtype=np.float64)
    S_a = np.asarray((M @ B.T).todense(), dtype=np.float64)
    S_b = S_a.T
    Q_a = np.asarray((M2 @ B.T).todense(), dtype=np.float64)
    Q_b = Q_a.T

    num = n * S_ab - S_a * S_b
    var_a = n * Q_a - S_a ** 2
    var_b = n * Q_b - S_b ** 2
    den = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))

    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    S = np.clip(S, -1.0, 1.0)
    np.fill_diagonal(S, 0.0)
    return S


def apply_reliability(S: np.ndarray, n: np.ndarray, min_support: int = 0,
                      beta: float | None = None) -> np.ndarray:
    """Zero out under-supported pairs and shrink the rest toward 0 by evidence.

    `min_support` drops similarities backed by too few co-ratings (with 2 co-ratings
    Pearson is +-1 by construction and is pure noise); `beta` applies Herlocker
    significance weighting, s' = s * min(n, beta) / beta, which is a crude shrinkage of
    the sample correlation toward zero in proportion to the evidence behind it.
    """
    out = S.copy()
    if min_support and min_support > 0:
        out[n < min_support] = 0.0
    if beta and beta > 0:
        out = out * (np.minimum(n, beta) / beta)
    np.fill_diagonal(out, 0.0)
    return out


_CACHE: dict = {}


def clear_similarity_cache() -> None:
    _CACHE.clear()


def _cache_key(M: sp.csr_matrix, kind: str, min_support, beta) -> tuple:
    """Identify a similarity request by the matrix *content*, not by object identity."""
    M = M.tocsr()
    h = hash((M.shape, M.indptr.tobytes(), M.indices.tobytes(), M.data.tobytes()))
    return (h, kind, min_support, beta)


def build_similarity(M: sp.csr_matrix, kind: str, *, min_support: int = 0,
                     beta: float | None = None,
                     user_means_for_adjusted: np.ndarray | None = None,
                     use_cache: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to a named similarity and apply the reliability corrections.

    Returns (S, n) where n is the co-rating count matrix.  Results are memoised on the
    matrix content, because a k-sweep re-requests the identical similarity for every k
    and the GEMMs dominate the sweep otherwise.
    """
    key = _cache_key(M, kind, min_support, beta) if use_cache else None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    n = co_counts(M)
    if kind == "cosine":
        S = cosine_similarity(M)
    elif kind == "pearson":
        S = pearson_corated(M)
    elif kind == "adjusted_cosine":
        if user_means_for_adjusted is None:
            raise ValueError("adjusted_cosine needs the centring means")
        # M rows are items, columns are users -> centre by the *column* (user) mean.
        C = M.tocsr().copy().astype(np.float64)
        C.data = C.data - user_means_for_adjusted[C.indices]
        S = cosine_similarity(C)
    else:
        raise ValueError(f"unknown similarity kind: {kind!r}")
    out = (apply_reliability(S, n, min_support=min_support, beta=beta), n)
    if key is not None:
        _CACHE[key] = out
    return out
