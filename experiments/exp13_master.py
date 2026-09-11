"""Part N -- master comparison table and per-axis model rankings."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.analysis import AXES, rank_models
from src.pipeline import banner, load_json, load_table, save_json, save_table

MASTER_COLS = [
    ("label", "Model"), ("rmse", "RMSE"), ("mae", "MAE"),
    ("precision_mean", "P@10"), ("recall_mean", "R@10"),
    ("hit_rate_mean", "HR@10"), ("ndcg_mean", "NDCG@10"), ("map_mean", "MAP@10"),
    ("catalog_coverage", "Coverage"), ("user_coverage", "UserCov"),
    ("novelty_mean", "Novelty"), ("ild_mean", "Diversity"),
    ("tail_frac_mean", "LongTail"), ("head_frac_mean", "Head"),
    ("train_time_s", "TrainS"), ("predict_time_s", "PredictS"),
    ("model_mb", "ModelMB"), ("model_sparsity_pct", "SparsityPct"),
]


def main() -> pd.DataFrame:
    banner("PART N - Master comparison")
    mu = load_table("multiuser_test")
    pb = load_table("popularity_bias")

    slim = load_json("slim")
    sparsity = {"SLIM": slim["sparsity"]["sparsity_pct"]}
    # Neighbourhood models are structurally sparse too: k neighbours out of n.
    ub = load_json("ubcf")["selection"]["ranking_selected"]
    ib = load_json("ibcf")["selection"]["ranking_selected"]
    rg = load_json("regression")["ranking_selected"]
    eda = load_json("eda")
    n_users, n_items = eda["overview"]["n_users"], eda["overview"]["n_items"]
    sparsity["UBCF"] = 100.0 * (1 - ub["k"] / (n_users - 1))
    sparsity["IBCF"] = 100.0 * (1 - ib["k"] / (n_items - 1))
    sparsity["RegressionCF"] = 100.0 * (1 - rg["k"] / (n_items - 1))

    df = mu.copy()
    df["model_sparsity_pct"] = df.label.map(sparsity)
    df["total_time_s"] = df.train_time_s + df.predict_time_s
    df = df.merge(pb[["model", "gini_exposure", "mean_train_popularity"]],
                  left_on="label", right_on="model", how="left").drop(columns=["model_y"],
                                                                     errors="ignore")
    if "model" in df.columns and "label" in df.columns:
        df = df.drop(columns=["model"])

    master = df[[c for c, _ in MASTER_COLS if c in df.columns]].copy()
    master.columns = [n for c, n in MASTER_COLS if c in df.columns]
    save_table("master_comparison", master)
    print(master.round(4).to_string(index=False))

    # Per-axis rankings are reported inside master.json (and rendered in REPORT.md);
    # writing them out again as CSVs would duplicate the same numbers in a second place.
    ranks = rank_models(df)
    paradigms = ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec"]
    sub = df[df.label.isin(paradigms)].reset_index(drop=True)
    sub_ranks = rank_models(sub)

    axis_winners = {}
    for axis, (col, higher) in AXES.items():
        s = sub.set_index("label")[col].astype(float)
        order = s.sort_values(ascending=not higher)
        axis_winners[axis] = {
            "metric": col, "higher_is_better": higher,
            "order": [{"model": m, "value": float(v)} for m, v in order.items()],
            "winner": str(order.index[0]),
        }

    tradeoffs = {
        "no_single_best": (
            "The five paradigms win on different axes, and the axes are in tension: the "
            "measured ordering on ranking quality is nearly the reverse of the ordering "
            "on coverage and novelty. Declaring one 'best' requires first declaring which "
            "axis the product is optimising."),
        "accuracy_vs_coverage": {
            "ranking_winner": axis_winners["ranking_quality"]["winner"],
            "ranking_winner_coverage": float(
                sub.set_index("label").loc[axis_winners["ranking_quality"]["winner"],
                                           "catalog_coverage"]),
            "coverage_winner": axis_winners["coverage"]["winner"],
            "coverage_winner_ndcg": float(
                sub.set_index("label").loc[axis_winners["coverage"]["winner"],
                                           "ndcg_mean"]),
        },
        "accuracy_vs_rating_error": {
            "rmse_winner": axis_winners["predictive_accuracy"]["winner"],
            "rmse_winner_ndcg": float(
                sub.set_index("label").loc[axis_winners["predictive_accuracy"]["winner"],
                                           "ndcg_mean"]),
            "ndcg_winner_rmse": float(
                sub.set_index("label").loc[axis_winners["ranking_quality"]["winner"],
                                           "rmse"]),
            "note": ("Minimising squared error on observed ratings and ordering unseen "
                     "items well are different objectives; this table measures how far "
                     "apart they land on the same data."),
        },
    }

    save_json("master", {
        "master_table": master.to_dict(orient="records"),
        "rankings_all_models": ranks.to_dict(orient="records"),
        "rankings_paradigms_only": sub_ranks.to_dict(orient="records"),
        "axis_winners": axis_winners,
        "tradeoffs": tradeoffs,
        "model_sparsity_note": (
            "For SLIM, sparsity is the measured fraction of zero off-diagonal entries in "
            "the learned W. For the neighbourhood models it is the structural sparsity "
            "1 - k/(n-1) of the retained neighbour lists, which is a definition of "
            "convenience, not a learned quantity."),
    })

    print("\nper-axis winners among the five paradigms:")
    for axis, v in axis_winners.items():
        print(f"  {axis:22s} {v['winner']:14s} "
              f"({v['metric']}={v['order'][0]['value']:.4f})")
    return master


if __name__ == "__main__":
    main()
