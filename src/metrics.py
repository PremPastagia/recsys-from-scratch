"""
Evaluation metrics, implemented from scratch.

Rating metrics
--------------
    RMSE = sqrt( (1/|T|) * sum_{(u,i) in T} (r_ui - rhat_ui)^2 )
    MAE  = (1/|T|)      * sum_{(u,i) in T} |r_ui - rhat_ui|

Ranking metrics (binary relevance, per user, averaged over users)
-----------------------------------------------------------------
    Let L_u = (i_1..i_K) be the Top-K list and G_u the held-out relevant set.
    rel_p = 1[i_p in G_u]

    Precision@K = (1/K) sum_p rel_p
    Recall@K    = (1/|G_u|) sum_p rel_p
    HitRate@K   = 1[ sum_p rel_p > 0 ]
    DCG@K       = sum_p rel_p / log2(p+1)
    IDCG@K      = sum_{p=1}^{min(K,|G_u|)} 1 / log2(p+1)
    NDCG@K      = DCG@K / IDCG@K
    AP@K        = (1/min(K,|G_u|)) sum_p rel_p * Precision@p
    MAP@K       = mean_u AP@K

Beyond-accuracy metrics
-----------------------
    Novelty      = mean over recommended items of -log2( pop_i / n_users )   [bits]
    ILD          = 1 - mean over distinct pairs of cosine(genre_i, genre_j)
    PopRank      = mean normalized popularity rank of recommended items in [0,1]
                   (0 = most popular item in the catalog, 1 = least popular)
    CatalogCov   = |union of all recommended items| / |catalog|
    UserCov      = fraction of evaluated users that received a full-length list
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# Rating accuracy
# --------------------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


# --------------------------------------------------------------------------------------
# Ranking (single user)
# --------------------------------------------------------------------------------------
def _rel_vector(ranked: np.ndarray, relevant: set, k: int) -> np.ndarray:
    top = ranked[:k]
    return np.fromiter((1.0 if int(i) in relevant else 0.0 for i in top),
                       dtype=np.float64, count=len(top))


def precision_at_k(ranked, relevant, k=10) -> float:
    if k == 0:
        return 0.0
    return float(_rel_vector(ranked, relevant, k).sum() / k)


def recall_at_k(ranked, relevant, k=10) -> float:
    if not relevant:
        return float("nan")
    return float(_rel_vector(ranked, relevant, k).sum() / len(relevant))


def hit_rate_at_k(ranked, relevant, k=10) -> float:
    return float(_rel_vector(ranked, relevant, k).sum() > 0)


def dcg(rel: np.ndarray) -> float:
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.dot(rel, discounts))


def ndcg_at_k(ranked, relevant, k=10) -> float:
    if not relevant:
        return float("nan")
    rel = _rel_vector(ranked, relevant, k)
    ideal = np.ones(min(k, len(relevant)))
    idcg = dcg(ideal)
    return float(dcg(rel) / idcg) if idcg > 0 else 0.0


def average_precision_at_k(ranked, relevant, k=10) -> float:
    if not relevant:
        return float("nan")
    rel = _rel_vector(ranked, relevant, k)
    if rel.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(rel)
    ranks = np.arange(1, rel.size + 1)
    prec_at_p = cum_hits / ranks
    return float(np.sum(prec_at_p * rel) / min(k, len(relevant)))


# --------------------------------------------------------------------------------------
# Beyond-accuracy
# --------------------------------------------------------------------------------------
def novelty(ranked, item_pop: np.ndarray, n_users: int, k=10) -> float:
    """Mean self-information (bits) of the Top-K list.

    An item seen by every user carries 0 bits; a rarely-seen item carries many bits.
    A +1 Laplace correction keeps items with zero training support finite.
    """
    top = np.asarray(ranked[:k], dtype=int)
    if top.size == 0:
        return float("nan")
    p = (item_pop[top] + 1.0) / (n_users + 1.0)
    return float(np.mean(-np.log2(p)))


def intra_list_diversity(ranked, genre_matrix: np.ndarray, k=10) -> float:
    """1 - mean pairwise cosine similarity over the list's *content* (genre) vectors.

    Content features are used deliberately rather than CF similarities: measuring
    diversity with the same similarity a model optimises would make every model look
    diverse by construction. Genres are an external, model-independent yardstick.
    """
    top = np.asarray(ranked[:k], dtype=int)
    if top.size < 2:
        return float("nan")
    G = genre_matrix[top]
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Gn = G / norms
    S = Gn @ Gn.T
    iu = np.triu_indices(len(top), k=1)
    return float(1.0 - np.mean(S[iu]))


def popularity_rank_score(ranked, pop_rank: np.ndarray, k=10) -> float:
    """Mean normalized popularity rank in [0, 1]; higher = deeper into the tail."""
    top = np.asarray(ranked[:k], dtype=int)
    if top.size == 0:
        return float("nan")
    return float(np.mean(pop_rank[top]))


def group_fraction(ranked, group: np.ndarray, name: str, k=10) -> float:
    top = np.asarray(ranked[:k], dtype=int)
    if top.size == 0:
        return float("nan")
    return float(np.mean(group[top] == name))


# --------------------------------------------------------------------------------------
# Aggregation helper
# --------------------------------------------------------------------------------------
def aggregate(values) -> dict:
    """mean / median / std over a per-user metric, ignoring undefined (NaN) users."""
    a = np.asarray([v for v in values], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}
