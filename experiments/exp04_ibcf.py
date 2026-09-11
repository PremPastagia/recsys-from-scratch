"""Part D -- Item-Based CF: cosine / Pearson / adjusted cosine, centring, k sweep.

Measured on the VALIDATION split, for the same anti-selection-leakage reason as Part C.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import CONFIG
from src.evaluation import evaluate_rank_modes
from src.pipeline import banner, build_model, get_context, save_json, save_table

# (similarity, mean_center, center_by) -- the three centring regimes that are actually
# distinct for item-item CF.
VARIANTS = [
    ("cosine", True, "item"),
    ("cosine", False, "item"),
    ("pearson", True, "item"),
    ("pearson", False, "item"),
    ("adjusted_cosine", True, "item"),
    ("adjusted_cosine", True, "user"),
]


def main() -> pd.DataFrame:
    banner("PART D - Item-Based Collaborative Filtering")
    _, split, ctx = get_context()

    rows = []
    for similarity, mean_center, center_by in VARIANTS:
        for k in CONFIG.k_grid:
            spec = {"family": "ibcf", "k": k, "similarity": similarity,
                    "mean_center": mean_center, "center_by": center_by,
                    "min_support": CONFIG.min_support, "beta": CONFIG.shrinkage_beta}
            model = build_model(spec).fit(split.R_train)
            for row in evaluate_rank_modes(model, ctx, split="val"):
                row.update(spec)
                row["fit_time_s"] = model.fit_time_s
                rows.append(row)
            best = max((r for r in rows if r["k"] == k and r["similarity"] == similarity
                        and r["mean_center"] == mean_center
                        and r["center_by"] == center_by),
                       key=lambda r: r["ndcg_mean"])
            print(f"  {similarity:16s} mc={str(mean_center):5s}/{center_by:4s} k={k:4d} "
                  f"RMSE={best['rmse']:.4f} MAE={best['mae']:.4f} "
                  f"NDCG={best['ndcg_mean']:.4f} ({best['rank_mode']}) "
                  f"Recall={best['recall_mean']:.4f}")

    df = pd.DataFrame(rows)
    save_table("ibcf_sweep", df)

    best_rank = df.loc[df.ndcg_mean.idxmax()]
    best_rate = df.loc[df.rmse.idxmin()]

    def spec_of(r):
        return {"family": "ibcf", "k": int(r.k), "similarity": r.similarity,
                "mean_center": bool(r.mean_center), "center_by": r.center_by,
                "min_support": CONFIG.min_support, "beta": CONFIG.shrinkage_beta,
                "rank_mode": r.rank_mode}

    sel = {"ranking_selected": spec_of(best_rank), "rating_selected": spec_of(best_rate),
           "val_ndcg_at_selection": float(best_rank.ndcg_mean),
           "val_rmse_at_selection": float(best_rate.rmse),
           "selection_split": "validation"}
    save_json("ibcf", {"selection": sel, "n_configs": int(len(df))})
    print(f"\nselected for ranking : {sel['ranking_selected']}  "
          f"(val NDCG@10={best_rank.ndcg_mean:.4f})")
    print(f"selected for rating  : {sel['rating_selected']}  (val RMSE={best_rate.rmse:.4f})")
    return df


if __name__ == "__main__":
    main()
