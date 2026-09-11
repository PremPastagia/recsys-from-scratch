"""
Cross-cutting analyses: cold-start / sparsity strata, item-popularity strata,
popularity-bias accounting and cross-model disagreement.

All of these consume the *saved* Top-K lists and test predictions from Part H, so they
never refit a model and never re-touch the raw data -- the analysis is a pure function of
what the models actually produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .evaluation import EvalContext, rank_metrics


# ======================================================================================
# Stratified evaluation
# ======================================================================================
def user_group_breakdown(artifacts: dict, ctx: EvalContext, split, labels: list[str],
                         k: int) -> pd.DataFrame:
    """Every model evaluated separately inside each user-history tertile."""
    test_u = split.test.u.to_numpy()
    y = split.test.rating.to_numpy()
    rows = []
    for label in labels:
        topk = artifacts[f"{label}__topk"]
        pred = artifacts[f"{label}__testpred"].astype(np.float64)
        for group in ("sparse", "medium", "heavy"):
            users = np.nonzero(ctx.user_group == group)[0]
            agg, _ = rank_metrics(topk, ctx.relevant_test, ctx, k, users=users)
            sel = np.isin(test_u, users)
            rel_sizes = [len(ctx.relevant_test.get(int(u), ())) for u in users]
            rel_sizes = [n for n in rel_sizes if n > 0]
            rows.append({
                "model": label, "user_group": group, "n_users": int(users.size),
                "mean_history": float(ctx.user_history[users].mean()),
                "mean_heldout_positives": float(np.mean(rel_sizes)) if rel_sizes else 0.0,
                "median_heldout_positives": float(np.median(rel_sizes)) if rel_sizes else 0.0,
                "rmse": M.rmse(y[sel], pred[sel]), "mae": M.mae(y[sel], pred[sel]),
                "n_test_ratings": int(sel.sum()),
                "precision": agg["precision"]["mean"], "recall": agg["recall"]["mean"],
                "hit_rate": agg["hit_rate"]["mean"], "ndcg": agg["ndcg"]["mean"],
                "map": agg["map"]["mean"], "novelty": agg["novelty"]["mean"],
                "ild": agg["ild"]["mean"], "tail_frac": agg["tail_frac"]["mean"],
                "catalog_coverage": agg["catalog_coverage"],
                "n_users_with_relevant": agg["n_users_with_relevant"],
            })
    return pd.DataFrame(rows)


def item_group_breakdown(artifacts: dict, ctx: EvalContext, labels: list[str],
                         k: int) -> pd.DataFrame:
    """Ranking quality restricted to held-out positives from one item-popularity group.

    The Top-K list is *not* changed -- only the ground truth is filtered.  This answers
    "how well does the model retrieve head / medium / long-tail items that the user
    actually liked", which is the fair way to compare across popularity strata.
    """
    rows = []
    for label in labels:
        topk = artifacts[f"{label}__topk"]
        for group in ("head", "medium", "long_tail"):
            mask = ctx.item_group == group
            agg, _ = rank_metrics(topk, ctx.relevant_test, ctx, k,
                                  restrict_items=mask)
            rows.append({
                "model": label, "item_group": group, "n_items": int(mask.sum()),
                "precision": agg["precision"]["mean"], "recall": agg["recall"]["mean"],
                "hit_rate": agg["hit_rate"]["mean"], "ndcg": agg["ndcg"]["mean"],
                "map": agg["map"]["mean"],
                "n_users_with_relevant": agg["n_users_with_relevant"],
            })
    return pd.DataFrame(rows)


# ======================================================================================
# Popularity bias
# ======================================================================================
def popularity_bias_table(artifacts: dict, ctx: EvalContext, labels: list[str],
                          k: int) -> pd.DataFrame:
    """Per-model exposure accounting over all recommended slots."""
    rows = []
    for label in labels:
        topk = artifacts[f"{label}__topk"][:, :k]
        flat = topk.ravel()
        counts = ctx.item_pop[flat]
        groups = ctx.item_group[flat]
        rows.append({
            "model": label,
            "mean_train_popularity": float(counts.mean()),
            "median_train_popularity": float(np.median(counts)),
            "mean_pop_rank": float(ctx.pop_rank[flat].mean()),
            "head_pct": 100.0 * float((groups == "head").mean()),
            "medium_pct": 100.0 * float((groups == "medium").mean()),
            "long_tail_pct": 100.0 * float((groups == "long_tail").mean()),
            "catalog_coverage": len(np.unique(flat)) / ctx.n_items,
            "n_distinct_items": int(len(np.unique(flat))),
            "novelty_bits": float(np.mean(
                [M.novelty(topk[u], ctx.item_pop, ctx.n_users, k)
                 for u in range(topk.shape[0])])),
            "ild": float(np.nanmean(
                [M.intra_list_diversity(topk[u], ctx.genre, k)
                 for u in range(topk.shape[0])])),
            "gini_exposure": gini(np.bincount(flat, minlength=ctx.n_items)),
        })
    return pd.DataFrame(rows)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of the exposure distribution (0 = uniform, 1 = winner-take-all)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


# ======================================================================================
# Cross-model agreement
# ======================================================================================
def disagreement_items(recs: dict[str, list[int]], k: int) -> pd.DataFrame:
    """Items ranked by how *divisively* the models treat them.

    An item recommended by exactly one model in a strong position, while every other
    model leaves it out entirely, is the most informative disagreement; we score that as
    ``(1 - coverage_fraction) * best_rank_weight``.
    """
    models = list(recs)
    rows = []
    universe = sorted({i for lst in recs.values() for i in lst[:k]})
    for item in universe:
        ranks = {}
        for m in models:
            lst = recs[m][:k]
            ranks[m] = lst.index(item) + 1 if item in lst else None
        present = [m for m in models if ranks[m] is not None]
        cover = len(present) / len(models)
        best_rank = min(r for r in ranks.values() if r is not None)
        rows.append({
            "item": item, "n_models": len(present), "coverage": cover,
            "best_rank": best_rank,
            "divisiveness": (1.0 - cover) * (1.0 / np.log2(best_rank + 1)),
            **{f"rank_{m}": ranks[m] for m in models},
        })
    return pd.DataFrame(rows).sort_values("divisiveness", ascending=False)


# ======================================================================================
# Ranking of models on the six report axes
# ======================================================================================
AXES = {
    "predictive_accuracy": ("rmse", False),
    "ranking_quality": ("ndcg_mean", True),
    "coverage": ("catalog_coverage", True),
    "novelty": ("novelty_mean", True),
    "diversity": ("ild_mean", True),
    "efficiency": ("total_time_s", False),
}


def rank_models(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Rank every model on each reporting axis (1 = best)."""
    out = pd.DataFrame({label_col: df[label_col]})
    for axis, (col, higher_better) in AXES.items():
        vals = df[col].astype(float)
        out[f"rank_{axis}"] = vals.rank(ascending=not higher_better,
                                        method="min").astype(int)
        out[f"value_{axis}"] = vals.to_numpy()
    return out
