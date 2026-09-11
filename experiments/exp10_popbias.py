"""Part J -- popularity bias and long-tail analysis (the critical section).

A note on how the correlations here are computed, because it changes the conclusion.

The evaluation table contains three kinds of model:
  * the five paradigms, each with its validation-selected *ranking* configuration;
  * rating-tuned arms (`*-rating`), kept because they are the RMSE-optimal configurations,
    but which rank by predicted rating and therefore rank close to randomly;
  * non-personalised and degenerate baselines.

Correlating NDCG@10 against coverage across all of them produces large coefficients that
are entirely artefacts of the rating-tuned arms: those spray recommendations across the
catalogue (high coverage) while retrieving almost nothing (NDCG ~ 0.002), which
manufactures a strong apparent trade-off.  We therefore report correlations over the five
paradigms as the primary figure, over a wider *ranking-capable* set as a secondary one,
and -- because n = 5 makes any correlation descriptive rather than inferential -- lead the
analysis with the raw exposure numbers and an explicit Pareto analysis instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.analysis import popularity_bias_table
from src.config import ARTIFACTS_DIR, CONFIG
from src.pipeline import banner, get_context, load_table, save_json, save_table

PARADIGMS = ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec"]
RANKING_CAPABLE_EXTRA = ["Popularity", "PopularityPositive", "BiasBaseline"]
AXES = {"ndcg_mean": True, "catalog_coverage": True, "novelty_bits": True,
        "long_tail_pct": True, "ild": True}


def _corr(df: pd.DataFrame) -> dict:
    def c(a, b):
        if len(df) < 3:
            return float("nan")
        return float(np.corrcoef(df[a], df[b])[0, 1])
    return {
        "n_models": int(len(df)),
        "models": df.model.tolist(),
        "ndcg_vs_mean_popularity": c("ndcg_mean", "mean_train_popularity"),
        "ndcg_vs_head_pct": c("ndcg_mean", "head_pct"),
        "ndcg_vs_long_tail_pct": c("ndcg_mean", "long_tail_pct"),
        "ndcg_vs_catalog_coverage": c("ndcg_mean", "catalog_coverage"),
        "ndcg_vs_novelty": c("ndcg_mean", "novelty_bits"),
        "ndcg_vs_ild": c("ndcg_mean", "ild"),
        "coverage_vs_novelty": c("catalog_coverage", "novelty_bits"),
    }


def _pareto(df: pd.DataFrame) -> dict:
    """Which models are dominated on (NDCG, coverage, novelty, long-tail share)?"""
    cols = ["ndcg_mean", "catalog_coverage", "novelty_bits", "long_tail_pct"]
    rows = df.set_index("model")[cols]
    dominated = {}
    for a in rows.index:
        by = [b for b in rows.index
              if b != a and (rows.loc[b] >= rows.loc[a]).all()
              and (rows.loc[b] > rows.loc[a]).any()]
        if by:
            dominated[a] = by
    return {
        "objectives": cols,
        "dominated": dominated,
        "pareto_front": [m for m in rows.index if m not in dominated],
    }


def main() -> pd.DataFrame:
    banner("PART J - Popularity bias and long-tail analysis")
    _, _, ctx = get_context()
    art = dict(np.load(ARTIFACTS_DIR / "model_outputs.npz"))
    mu = load_table("multiuser_test")
    labels = mu.label.tolist()

    pb = popularity_bias_table(art, ctx, labels, CONFIG.top_k)
    pb = pb.merge(mu[["label", "ndcg_mean", "rmse", "recall_mean", "map_mean"]],
                  left_on="model", right_on="label").drop(columns=["label"])
    save_table("popularity_bias", pb)
    print(pb[["model", "mean_train_popularity", "head_pct", "long_tail_pct",
              "catalog_coverage", "novelty_bits", "ild", "gini_exposure",
              "ndcg_mean"]].round(4).to_string(index=False))

    par = pb[pb.model.isin(PARADIGMS)].reset_index(drop=True)
    wide = pb[pb.model.isin(PARADIGMS + RANKING_CAPABLE_EXTRA)].reset_index(drop=True)
    degenerate = pb[pb.model.str.endswith("-rating")
                    | pb.model.isin(["Random", "GlobalMean", "UserMean", "ItemMean"])]

    pareto = _pareto(par)
    best = par.loc[par.ndcg_mean.idxmax()]
    widest = par.loc[par.catalog_coverage.idxmax()]
    most_tail = par.loc[par.long_tail_pct.idxmax()]

    head_share = 100.0 * ctx.pop_groups["n_head"] / ctx.n_items
    tail_share = 100.0 * ctx.pop_groups["n_long_tail"] / ctx.n_items

    payload = {
        "table": pb.to_dict(orient="records"),
        "correlations_paradigms": _corr(par),
        "correlations_ranking_capable": _corr(wide),
        "correlation_caveat": (
            "With n = 5 a correlation coefficient is a description of five points, not "
            "evidence. The exposure percentages and the Pareto analysis below are the "
            "load-bearing results; the correlations are reported for completeness. "
            "Correlations computed over the full model list are excluded deliberately: "
            "the rating-tuned arms rank close to randomly (NDCG@10 as low as "
            f"{degenerate.ndcg_mean.min():.4f}) while covering "
            f"{100 * degenerate.catalog_coverage.max():.1f}% of the catalogue, and "
            "including them manufactures a strong apparent accuracy/coverage trade-off "
            "that is really just 'a broken ranker touches many items'."),
        "exposure_reference": {
            "n_head_items": ctx.pop_groups["n_head"],
            "head_share_of_catalog_pct": head_share,
            "n_long_tail_items": ctx.pop_groups["n_long_tail"],
            "long_tail_share_of_catalog_pct": tail_share,
            "paradigm_head_pct_range": [float(par.head_pct.min()),
                                        float(par.head_pct.max())],
            "paradigm_tail_pct_range": [float(par.long_tail_pct.min()),
                                        float(par.long_tail_pct.max())],
            "paradigm_coverage_range": [float(par.catalog_coverage.min()),
                                        float(par.catalog_coverage.max())],
        },
        "pareto": pareto,
        "headline": {
            "most_accurate_model_ndcg": str(best.model),
            "most_long_tail_model": str(most_tail.model),
            "highest_coverage_model": str(widest.model),
            "most_accurate_head_pct": float(best.head_pct),
            "most_accurate_long_tail_pct": float(best.long_tail_pct),
            "most_accurate_coverage": float(best.catalog_coverage),
            "most_accurate_novelty": float(best.novelty_bits),
        },
        "does_the_most_accurate_model_make_the_most_useful_recommendations": {
            "answer": ("Not straightforwardly -- but the interesting part is that the "
                       "trade-off is not where the textbook expects it."),
            "evidence": {
                "most_accurate_model": str(best.model),
                "its_ndcg": float(best.ndcg_mean),
                "its_catalog_coverage": float(best.catalog_coverage),
                "its_long_tail_pct": float(best.long_tail_pct),
                "its_head_pct": float(best.head_pct),
                "its_novelty_bits": float(best.novelty_bits),
                "pareto_front": pareto["pareto_front"],
                "dominated_models": pareto["dominated"],
                "widest_coverage_model": str(widest.model),
                "its_coverage": float(widest.catalog_coverage),
                "its_long_tail_pct": float(widest.long_tail_pct),
                "its_novelty_bits": float(widest.novelty_bits),
                "its_ndcg": float(widest.ndcg_mean),
                "coverage_gain_pp": float(100 * (widest.catalog_coverage
                                                 - best.catalog_coverage)),
                "tail_gain_pp": float(widest.long_tail_pct - best.long_tail_pct),
                "ndcg_cost_relative_pct": float(
                    100 * (1 - widest.ndcg_mean / best.ndcg_mean)),
            },
        },
    }
    save_json("popularity_bias", payload)

    c = payload["correlations_paradigms"]
    print(f"\nover the 5 paradigms (n={c['n_models']}):")
    print(f"  corr(NDCG, mean popularity) = {c['ndcg_vs_mean_popularity']:+.3f}")
    print(f"  corr(NDCG, long-tail %)     = {c['ndcg_vs_long_tail_pct']:+.3f}")
    print(f"  corr(NDCG, coverage)        = {c['ndcg_vs_catalog_coverage']:+.3f}")
    print(f"\nPareto front on (NDCG, coverage, novelty, tail%): {pareto['pareto_front']}")
    for m, by in pareto["dominated"].items():
        print(f"  {m} is dominated by {by}")
    print(f"\nhead is {head_share:.1f}% of the catalog but takes "
          f"{par.head_pct.min():.1f}-{par.head_pct.max():.1f}% of recommendation slots; "
          f"long tail is {tail_share:.1f}% of the catalog and takes "
          f"{par.long_tail_pct.min():.1f}-{par.long_tail_pct.max():.1f}%")
    return pb


if __name__ == "__main__":
    main()
