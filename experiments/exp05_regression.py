"""Part E -- Regression-based neighbourhood CF: learned vs similarity weights."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.config import CONFIG
from src.evaluation import evaluate_rank_modes
from src.pipeline import banner, build_model, get_context, save_json, save_table

K_GRID = (10, 20, 40, 80)
LAM_GRID = (0.0, 1.0, 5.0, 25.0, 100.0, 300.0, 1000.0, 3000.0)


def main() -> pd.DataFrame:
    banner("PART E - Regression-based neighbourhood model")
    ds, split, ctx = get_context()

    rows = []
    for k in K_GRID:
        for lam in LAM_GRID:
            spec = {"family": "regression", "k": k, "lam": lam,
                    "similarity": "adjusted_cosine", "weight_mode": "learned"}
            model = build_model(spec).fit(split.R_train)
            sims, coefs, _ = model.similarity_vs_coefficient()
            r_p = float(pearsonr(sims, coefs)[0])
            r_s = float(spearmanr(sims, coefs)[0])
            for row in evaluate_rank_modes(model, ctx, split="val"):
                row.update(spec)
                row["pearson_sim_coef"] = r_p
                row["spearman_sim_coef"] = r_s
                row["frac_negative_coef"] = float((coefs < 0).mean())
                row["mean_abs_coef"] = float(np.abs(coefs).mean())
                row["rank_deficient_solves"] = int(model.rank_deficient)
                row["fit_time_s"] = model.fit_time_s
                rows.append(row)
            best = max((r for r in rows if r["k"] == k and r["lam"] == lam),
                       key=lambda r: r["ndcg_mean"])
            print(f"  learned    k={k:3d} lam={lam:7.1f} RMSE={best['rmse']:.4f} "
                  f"NDCG={best['ndcg_mean']:.4f} r(sim,coef)={r_p:+.3f} "
                  f"rho={r_s:+.3f} neg={best['frac_negative_coef']:.3f}")

        # Heuristic-weight arm: identical neighbourhoods and baseline, similarity weights.
        spec = {"family": "regression", "k": k, "lam": 0.0,
                "similarity": "adjusted_cosine", "weight_mode": "similarity"}
        model = build_model(spec).fit(split.R_train)
        for row in evaluate_rank_modes(model, ctx, split="val"):
            row.update(spec)
            row["pearson_sim_coef"] = 1.0
            row["spearman_sim_coef"] = 1.0
            row["frac_negative_coef"] = 0.0
            row["mean_abs_coef"] = float(np.abs(model.coef).mean())
            row["rank_deficient_solves"] = 0
            row["fit_time_s"] = model.fit_time_s
            rows.append(row)
        best = max((r for r in rows if r["k"] == k and r["weight_mode"] == "similarity"),
                   key=lambda r: r["ndcg_mean"])
        print(f"  similarity k={k:3d} {'':11s} RMSE={best['rmse']:.4f} "
              f"NDCG={best['ndcg_mean']:.4f}")

    df = pd.DataFrame(rows)
    save_table("regression_sweep", df)

    # ---- deep-dive on the rating-optimal learned model -------------------------------
    learned = df[df.weight_mode == "learned"]
    best_rate = learned.loc[learned.rmse.idxmin()]
    best_rank = learned.loc[learned.ndcg_mean.idxmax()]
    deep_spec = {"family": "regression", "k": int(best_rate.k), "lam": float(best_rate.lam),
                 "similarity": "adjusted_cosine", "weight_mode": "learned"}
    model = build_model(deep_spec).fit(split.R_train)
    sims, coefs, items = model.similarity_vs_coefficient()

    # Sub-sample for the scatter figure (keeps the CSV small and the plot readable).
    rng = np.random.default_rng(CONFIG.seed)
    idx = rng.choice(len(sims), size=min(20000, len(sims)), replace=False)
    save_table("regression_sim_vs_coef",
               pd.DataFrame({"similarity": sims[idx], "coefficient": coefs[idx],
                             "target_item": items[idx]}))

    # Where does regression disagree most with the heuristic?
    sim_norm = model.neighbor_sims / np.maximum(
        np.abs(model.neighbor_sims).sum(axis=1, keepdims=True), 1e-12)
    gap = model.coef - sim_norm
    flat = np.argsort(-np.abs(gap).ravel())[:200]
    ii, jj = np.unravel_index(flat, gap.shape)
    disagree = pd.DataFrame({
        "target_item": ii,
        "target_title": ds.titles[ii],
        "neighbor_item": model.neighbors[ii, jj],
        "neighbor_title": ds.titles[model.neighbors[ii, jj]],
        "similarity": model.neighbor_sims[ii, jj],
        "normalised_similarity_weight": sim_norm[ii, jj],
        "learned_coefficient": model.coef[ii, jj],
        "gap": gap[ii, jj],
        "n_train_users_for_target": model.n_train_users[ii],
    }).head(50)
    save_table("regression_disagreements", disagree)

    sign_flip = float(((sims > 0.05) & (coefs < 0)).mean())
    summary = {
        "deep_dive_spec": deep_spec,
        "pearson_sim_coef": float(pearsonr(sims, coefs)[0]),
        "spearman_sim_coef": float(spearmanr(sims, coefs)[0]),
        "frac_negative_coefficients": float((coefs < 0).mean()),
        "frac_sign_flip_positive_sim_negative_coef": sign_flip,
        "coef_mean": float(coefs.mean()), "coef_std": float(coefs.std(ddof=1)),
        "coef_p01": float(np.percentile(coefs, 1)), "coef_p99": float(np.percentile(coefs, 99)),
        "coef_sum_mean_per_item": float(model.coef.sum(axis=1).mean()),
        "similarity_weight_sum_per_item": 1.0,
        "rank_deficient_solves": int(model.rank_deficient),
        "best_rating_config": {"k": int(best_rate.k), "lam": float(best_rate.lam),
                               "val_rmse": float(best_rate.rmse)},
        "best_ranking_config": {"k": int(best_rank.k), "lam": float(best_rank.lam),
                                "rank_mode": best_rank.rank_mode,
                                "val_ndcg": float(best_rank.ndcg_mean)},
        "ranking_selected": {"family": "regression", "k": int(best_rank.k),
                             "lam": float(best_rank.lam), "similarity": "adjusted_cosine",
                             "weight_mode": "learned", "rank_mode": best_rank.rank_mode},
        "rating_selected": {"family": "regression", "k": int(best_rate.k),
                            "lam": float(best_rate.lam), "similarity": "adjusted_cosine",
                            "weight_mode": "learned", "rank_mode": best_rate.rank_mode},
        "similarity_arm_best_val_rmse":
            float(df[df.weight_mode == "similarity"].rmse.min()),
        "learned_arm_best_val_rmse": float(learned.rmse.min()),
        "learned_beats_similarity_on_rmse":
            bool(learned.rmse.min() < df[df.weight_mode == "similarity"].rmse.min()),
        "selection_split": "validation",
    }
    save_json("regression", summary)

    print(f"\nlearned best val RMSE  = {summary['learned_arm_best_val_rmse']:.4f} "
          f"(k={best_rate.k:g}, lam={best_rate.lam:g})")
    print(f"similarity best val RMSE = {summary['similarity_arm_best_val_rmse']:.4f}")
    print(f"r(sim,coef)={summary['pearson_sim_coef']:+.3f}  "
          f"rho={summary['spearman_sim_coef']:+.3f}  "
          f"negative coefficients={summary['frac_negative_coefficients']:.3f}")
    return df


if __name__ == "__main__":
    main()
