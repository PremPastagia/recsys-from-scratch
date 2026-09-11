"""Part I -- cold-start / sparsity analysis across user-history and item-popularity strata."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.analysis import item_group_breakdown, user_group_breakdown
from src.config import ARTIFACTS_DIR, CONFIG
from src.pipeline import banner, get_context, load_table, save_json, save_table


def main():
    banner("PART I - Cold-start and sparsity analysis")
    _, split, ctx = get_context()
    art = dict(np.load(ARTIFACTS_DIR / "model_outputs.npz"))
    labels = load_table("multiuser_test").label.tolist()
    k = CONFIG.top_k

    ug = user_group_breakdown(art, ctx, split, labels, k)
    save_table("coldstart_user_groups", ug)
    ig = item_group_breakdown(art, ctx, labels, k)
    save_table("coldstart_item_groups", ig)

    piv = ug.pivot(index="model", columns="user_group", values="ndcg")
    piv_rmse = ug.pivot(index="model", columns="user_group", values="rmse")
    print("\nNDCG@10 by user-history tertile")
    print(piv[["sparse", "medium", "heavy"]].round(4).to_string())
    print("\nRMSE by user-history tertile")
    print(piv_rmse[["sparse", "medium", "heavy"]].round(4).to_string())

    paradigms = [l for l in labels if l in
                 ("UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec")]
    p_ndcg = piv.loc[paradigms]
    p_rmse = piv_rmse.loc[paradigms]

    gain = (p_ndcg["heavy"] - p_ndcg["sparse"])
    rel_gain = gain / p_ndcg["sparse"].replace(0, np.nan)

    answers = {
        "best_for_sparse_users_ndcg": {
            "model": str(p_ndcg["sparse"].idxmax()),
            "ndcg": float(p_ndcg["sparse"].max()),
            "runner_up": str(p_ndcg["sparse"].drop(p_ndcg["sparse"].idxmax()).idxmax()),
        },
        "best_for_sparse_users_rmse": {
            "model": str(p_rmse["sparse"].idxmin()),
            "rmse": float(p_rmse["sparse"].min()),
        },
        "benefits_most_from_history": {
            "model": str(gain.idxmax()),
            "absolute_ndcg_gain_heavy_minus_sparse": float(gain.max()),
            "relative_gain": float(rel_gain[gain.idxmax()]),
            "all_gains": {m: float(gain[m]) for m in paradigms},
        },
        "least_sensitive_to_history": {
            "model": str(gain.abs().idxmin()),
            "absolute_ndcg_gain": float(gain[gain.abs().idxmin()]),
        },
        "sensitivity_ranking_by_relative_ndcg_gain": {
            m: float(rel_gain[m]) for m in
            rel_gain.sort_values(ascending=False).index.tolist()
        },
        "user_group_definition": {
            "rule": "tertiles of TRAIN history length",
            "q1": ctx.user_groups["q1"], "q2": ctx.user_groups["q2"],
            "n_sparse": int((ctx.user_group == "sparse").sum()),
            "n_medium": int((ctx.user_group == "medium").sum()),
            "n_heavy": int((ctx.user_group == "heavy").sum()),
        },
        "item_group_ndcg": ig.pivot(index="model", columns="item_group",
                                    values="ndcg").to_dict(),
        "item_group_recall": ig.pivot(index="model", columns="item_group",
                                      values="recall").to_dict(),
        "partition_check": {
            "user_groups_cover_all_users":
                bool((np.isin(ctx.user_group, ["sparse", "medium", "heavy"])).all()),
            "item_groups_cover_all_items":
                bool((np.isin(ctx.item_group, ["head", "medium", "long_tail"])).all()),
        },
    }
    save_json("coldstart", answers)

    print("\nNDCG@10 by item-popularity group (ground truth restricted)")
    print(ig.pivot(index="model", columns="item_group",
                   values="ndcg")[["head", "medium", "long_tail"]].round(4).to_string())
    print(f"\nbest for sparse users : {answers['best_for_sparse_users_ndcg']}")
    print(f"benefits most from history: {answers['benefits_most_from_history']['model']} "
          f"(+{answers['benefits_most_from_history']['absolute_ndcg_gain_heavy_minus_sparse']:.4f} NDCG)")
    return ug, ig


if __name__ == "__main__":
    main()
