"""
Evaluation harness: one protocol, applied identically to every model.

Protocol
--------
*Rating task.*  Models are fitted on R_train only.  RMSE / MAE are measured on the
held-out test ratings, with predictions clipped to the observed rating scale [1, 5]
(a prediction of 6.3 is never right and clipping is what a deployed system would do).

*Top-N task.*  For each evaluated user u:
    candidates  = all items MINUS the items u has in the model's *visible* matrix
    relevant    = { i in test(u) : r_ui >= relevance_threshold }
    prediction  = Top-K of the model's ranking score over the candidates
Users with no relevant held-out item are excluded from the per-user ranking averages
(their Recall/NDCG are undefined, not zero) but are still counted for coverage.

Leakage
-------
Three separate guards, each checked by the gate suite:
  1. ``fit`` only ever receives R_train.
  2. The candidate set is built from the *visible* matrix, never from test.  Excluding
     test items would hand the model the answer key.
  3. Hyper-parameters (k, lambda, rank mode, ...) are selected on the *validation*
     split; the test split is touched exactly once, at reporting time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import metrics as M
from .base import Recommender, topk_from_scores
from .config import CONFIG, ExperimentConfig
from .data import Dataset, Split, popularity_groups, user_history_groups


@dataclass
class EvalContext:
    """Everything the harness needs, precomputed once and shared by all models."""

    ds: Dataset
    split: Split
    cfg: ExperimentConfig = field(default_factory=lambda: CONFIG)

    def __post_init__(self):
        self.n_users, self.n_items = self.ds.n_users, self.ds.n_items
        self.R_train = self.split.R_train

        pg = popularity_groups(self.R_train, self.cfg.popularity_mass_cuts)
        self.item_pop = pg["counts"].astype(np.float64)
        self.item_group = pg["group"]
        self.pop_groups = pg
        # Normalised popularity rank in [0,1]: 0 = most popular item.
        order = np.argsort(-self.item_pop, kind="stable")
        rank = np.empty(self.n_items)
        rank[order] = np.arange(self.n_items) / max(self.n_items - 1, 1)
        self.pop_rank = rank

        ug = user_history_groups(self.R_train)
        self.user_group = ug["group"]
        self.user_history = ug["history"]
        self.user_groups = ug

        self.genre = self.ds.genre_matrix
        self.relevant_test = self._relevant(self.split.test)
        self.relevant_val = self._relevant(self.split.val)

    def _relevant(self, df: pd.DataFrame) -> dict[int, set]:
        tau = self.cfg.relevance_threshold
        hits = df[df.rating >= tau]
        out: dict[int, set] = {}
        for u, i in zip(hits.u.to_numpy(), hits.i.to_numpy()):
            out.setdefault(int(u), set()).add(int(i))
        return out


# ======================================================================================
def rank_metrics(topk: np.ndarray, relevant: dict[int, set], ctx: EvalContext,
                 k: int, users: np.ndarray | None = None,
                 restrict_items: np.ndarray | None = None) -> dict:
    """Per-user ranking + beyond-accuracy metrics, aggregated.

    ``restrict_items`` (a boolean mask over items) restricts the *ground truth* to an item
    group, which is how Part I measures performance by item popularity: the Top-K list is
    unchanged, only the set of items we are willing to count as a hit changes.
    """
    if users is None:
        users = np.arange(topk.shape[0])

    per = {m: [] for m in ("precision", "recall", "hit_rate", "ndcg", "map",
                           "novelty", "ild", "pop_rank", "head_frac", "medium_frac",
                           "tail_frac")}
    covered_items, n_covered_users, n_scored_users = set(), 0, 0

    for u in users:
        ranked = topk[u]
        covered_items.update(int(x) for x in ranked[:k])
        n_scored_users += 1

        rel = relevant.get(int(u), set())
        if restrict_items is not None:
            rel = {i for i in rel if restrict_items[i]}

        if rel:
            per["precision"].append(M.precision_at_k(ranked, rel, k))
            per["recall"].append(M.recall_at_k(ranked, rel, k))
            per["hit_rate"].append(M.hit_rate_at_k(ranked, rel, k))
            per["ndcg"].append(M.ndcg_at_k(ranked, rel, k))
            per["map"].append(M.average_precision_at_k(ranked, rel, k))

        per["novelty"].append(M.novelty(ranked, ctx.item_pop, ctx.n_users, k))
        per["ild"].append(M.intra_list_diversity(ranked, ctx.genre, k))
        per["pop_rank"].append(M.popularity_rank_score(ranked, ctx.pop_rank, k))
        per["head_frac"].append(M.group_fraction(ranked, ctx.item_group, "head", k))
        per["medium_frac"].append(M.group_fraction(ranked, ctx.item_group, "medium", k))
        per["tail_frac"].append(M.group_fraction(ranked, ctx.item_group, "long_tail", k))

    out = {name: M.aggregate(vals) for name, vals in per.items()}
    out["catalog_coverage"] = len(covered_items) / ctx.n_items
    out["n_covered_items"] = len(covered_items)
    out["n_users_scored"] = n_scored_users
    out["n_users_with_relevant"] = len(per["ndcg"])
    return out, per


def user_coverage_from_scores(scores: np.ndarray, topk: np.ndarray, k: int,
                              users: np.ndarray) -> float:
    """Fraction of users for whom the model produced a genuinely *ranked* list.

    A model that assigns the same score to every candidate has no opinion; its "Top-K"
    is just an arbitrary tie-break.  We count a user as covered only when the model
    discriminates within its own Top-K (at least two distinct scores) -- this is what
    separates "the system can serve this user" from "the system fell back to a tie".
    """
    covered = 0
    for u in users:
        s = scores[u, topk[u][:k]]
        if np.unique(np.round(s, 12)).size > 1:
            covered += 1
    return covered / max(len(users), 1)


# ======================================================================================
def evaluate(model: Recommender, ctx: EvalContext, *, split: str = "test",
             k: int | None = None, exclude: sp.csr_matrix | None = None,
             predictions: np.ndarray | None = None,
             scores: np.ndarray | None = None) -> dict:
    """Full metric bundle for one fitted model on one split."""
    k = k or ctx.cfg.top_k
    df = ctx.split.test if split == "test" else ctx.split.val
    relevant = ctx.relevant_test if split == "test" else ctx.relevant_val
    exclude = ctx.R_train if exclude is None else exclude

    t0 = time.perf_counter()
    P = model.predict_all() if predictions is None else predictions
    S = model.score_all() if scores is None else scores
    predict_time = time.perf_counter() - t0

    lo, hi = ctx.ds.rating_scale
    Pc = np.clip(P, lo, hi)
    yy = df.rating.to_numpy()
    pp = Pc[df.u.to_numpy(), df.i.to_numpy()]

    t1 = time.perf_counter()
    topk = topk_from_scores(S, exclude, k)
    rank_time = time.perf_counter() - t1

    users = np.arange(ctx.n_users)
    agg, per = rank_metrics(topk, relevant, ctx, k, users)
    agg["user_coverage"] = user_coverage_from_scores(S, topk, k, users)

    return {
        "model": model.name,
        "split": split,
        "k": k,
        "rmse": M.rmse(yy, pp),
        "mae": M.mae(yy, pp),
        "n_test_ratings": int(len(df)),
        "train_time_s": float(model.fit_time_s),
        "predict_time_s": float(predict_time),
        "rank_time_s": float(rank_time),
        "model_bytes": int(model.model_bytes()),
        **agg,
        "_topk": topk,
        "_per_user": per,
        "_scores": S,
        "_pred": Pc,
    }


def flat_row(res: dict) -> dict:
    """Flatten an ``evaluate`` result into one row of a results table."""
    row = {kk: vv for kk, vv in res.items() if not kk.startswith("_")
           and not isinstance(vv, dict)}
    for name in ("precision", "recall", "hit_rate", "ndcg", "map", "novelty", "ild",
                 "pop_rank", "head_frac", "medium_frac", "tail_frac"):
        a = res[name]
        row[f"{name}_mean"] = a["mean"]
        row[f"{name}_median"] = a["median"]
        row[f"{name}_std"] = a["std"]
    return row


def evaluate_rank_modes(model: Recommender, ctx: EvalContext, split: str = "val",
                        k: int | None = None) -> list[dict]:
    """Evaluate every ranking head the model supports on one split.

    The expensive neighbourhood accumulation is cached inside the model, so the extra
    heads cost only the top-K selection and the metric loop.
    """
    modes = list(getattr(model, "rank_modes", ("rating", "score", "affinity")))
    if not hasattr(model, "rank_mode"):
        modes = ["n/a"]
    elif not hasattr(model, "like_threshold"):
        modes = [m for m in modes if m != "affinity"]

    out = []
    original = getattr(model, "rank_mode", None)
    for mode in modes:
        if original is not None:
            model.rank_mode = mode
        row = flat_row(evaluate(model, ctx, split=split, k=k))
        row["rank_mode"] = mode
        out.append(row)
    if original is not None:
        model.rank_mode = original
    return out
