"""Part C -- User-Based CF: cosine vs Pearson, mean-centring on/off, neighbourhood sweep.

All sweep numbers are measured on the VALIDATION split.  The test split is deliberately
not consulted here: selecting k or the ranking head on test and then reporting test
numbers would be selection leakage, and the resulting k-curves would be optimistic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import CONFIG
from src.evaluation import evaluate_rank_modes
from src.pipeline import banner, build_model, get_context, save_json, save_table


def main() -> pd.DataFrame:
    banner("PART C - User-Based Collaborative Filtering")
    _, split, ctx = get_context()

    rows = []
    for similarity in ("cosine", "pearson"):
        for mean_center in (True, False):
            for k in CONFIG.k_grid:
                spec = {"family": "ubcf", "k": k, "similarity": similarity,
                        "mean_center": mean_center,
                        "min_support": CONFIG.min_support,
                        "beta": CONFIG.shrinkage_beta}
                model = build_model(spec).fit(split.R_train)
                for row in evaluate_rank_modes(model, ctx, split="val"):
                    row.update(spec)
                    row["fit_time_s"] = model.fit_time_s
                    rows.append(row)
                best = max((r for r in rows if r["k"] == k and r["similarity"] == similarity
                            and r["mean_center"] == mean_center),
                           key=lambda r: r["ndcg_mean"])
                print(f"  {similarity:8s} mc={str(mean_center):5s} k={k:4d} "
                      f"RMSE={best['rmse']:.4f} MAE={best['mae']:.4f} "
                      f"NDCG={best['ndcg_mean']:.4f} ({best['rank_mode']}) "
                      f"Recall={best['recall_mean']:.4f}")

    df = pd.DataFrame(rows)
    save_table("ubcf_sweep", df)

    best_rank = df.loc[df.ndcg_mean.idxmax()]
    best_rate = df.loc[df.rmse.idxmin()]
    sel = {
        "ranking_selected": {"family": "ubcf", "k": int(best_rank.k),
                             "similarity": best_rank.similarity,
                             "mean_center": bool(best_rank.mean_center),
                             "min_support": CONFIG.min_support,
                             "beta": CONFIG.shrinkage_beta,
                             "rank_mode": best_rank.rank_mode},
        "rating_selected": {"family": "ubcf", "k": int(best_rate.k),
                            "similarity": best_rate.similarity,
                            "mean_center": bool(best_rate.mean_center),
                            "min_support": CONFIG.min_support,
                            "beta": CONFIG.shrinkage_beta,
                            "rank_mode": best_rate.rank_mode},
        "val_ndcg_at_selection": float(best_rank.ndcg_mean),
        "val_rmse_at_selection": float(best_rate.rmse),
        "selection_split": "validation",
    }
    save_json("ubcf", {"selection": sel, "n_configs": int(len(df))})
    print(f"\nselected for ranking : {sel['ranking_selected']}  "
          f"(val NDCG@10={best_rank.ndcg_mean:.4f})")
    print(f"selected for rating  : {sel['rating_selected']}  (val RMSE={best_rate.rmse:.4f})")
    return df


if __name__ == "__main__":
    main()
