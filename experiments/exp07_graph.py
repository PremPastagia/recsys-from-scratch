"""Part G -- Network-based recommendation: RWR vs bounded-length path proximity.

Also answers the required question "does deeper propagation make recommendations more
popular?" by measuring popularity of the Top-10 as a function of propagation depth
(path model) and of restart strength (RWR).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation import evaluate, flat_row
from src.pipeline import banner, fit_model, get_context, save_json, save_table

ALPHAS = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.95)
DEPTHS = (1, 3, 5, 7, 9)
BETAS = (0.2, 0.4, 0.6, 0.8)


def main() -> pd.DataFrame:
    banner("PART G - Network-based recommender (bipartite graph)")
    _, split, ctx = get_context()
    rows = []

    print("\n-- random walk with restart --")
    for a in ALPHAS:
        spec = {"family": "graph", "method": "rwr", "alpha": a,
                "threshold": CONFIG.implicit_threshold}
        model = fit_model(spec, split.R_train, split.val)
        res = evaluate(model, ctx, split="val")
        row = flat_row(res)
        row.update({"method": "rwr", "alpha": a, "max_path_len": -1, "beta": np.nan,
                    "normalize": "degree", "fit_time_s": model.fit_time_s,
                    "iterations": model.diagnostics["iterations"],
                    "mass_min": model.diagnostics["mass_min"],
                    "mass_max": model.diagnostics["mass_max"],
                    "final_delta": model.diagnostics["final_delta"]})
        rows.append(row)
        print(f"  alpha={a:4.2f} NDCG={res['ndcg']['mean']:.4f} "
              f"cov={res['catalog_coverage']:.3f} nov={res['novelty']['mean']:.2f} "
              f"head={res['head_frac']['mean']:.3f} poprank={res['pop_rank']['mean']:.4f} "
              f"iters={model.diagnostics['iterations']:3d} {model.fit_time_s:.1f}s")

    print("\n-- bounded-length path proximity --")
    for normalize in ("degree", "none"):
        for beta in BETAS:
            for L in DEPTHS:
                spec = {"family": "graph", "method": "path", "max_path_len": L,
                        "beta": beta, "normalize": normalize,
                        "threshold": CONFIG.implicit_threshold}
                model = fit_model(spec, split.R_train, split.val)
                res = evaluate(model, ctx, split="val")
                row = flat_row(res)
                row.update({"method": "path", "alpha": np.nan, "max_path_len": L,
                            "beta": beta, "normalize": normalize,
                            "fit_time_s": model.fit_time_s, "iterations": L,
                            "mass_min": np.nan, "mass_max": np.nan,
                            "final_delta": np.nan})
                rows.append(row)
            sub = [r for r in rows if r.get("method") == "path" and r["beta"] == beta
                   and r["normalize"] == normalize]
            print(f"  {normalize:6s} beta={beta:3.1f} " + "  ".join(
                f"L={r['max_path_len']}:NDCG={r['ndcg_mean']:.3f}/pop={r['pop_rank_mean']:.3f}"
                for r in sub))

    df = pd.DataFrame(rows)
    save_table("graph_sweep", df)

    # ---- depth / attenuation vs popularity -------------------------------------------
    depth_curve = (df[(df.method == "path") & (df.normalize == "degree")]
                   .groupby("max_path_len")[["ndcg_mean", "pop_rank_mean", "head_frac_mean",
                                             "tail_frac_mean", "novelty_mean",
                                             "catalog_coverage"]].mean().reset_index())

    rwr = df[df.method == "rwr"].sort_values("alpha")
    path_deg = df[(df.method == "path") & (df.normalize == "degree")]

    best_rwr = rwr.loc[rwr.ndcg_mean.idxmax()]
    best_path = path_deg.loc[path_deg.ndcg_mean.idxmax()]
    overall_best = best_rwr if best_rwr.ndcg_mean >= best_path.ndcg_mean else best_path

    if overall_best.method == "rwr":
        sel = {"family": "graph", "method": "rwr", "alpha": float(overall_best.alpha),
               "threshold": CONFIG.implicit_threshold}
    else:
        sel = {"family": "graph", "method": "path",
               "max_path_len": int(overall_best.max_path_len),
               "beta": float(overall_best.beta), "normalize": overall_best.normalize,
               "threshold": CONFIG.implicit_threshold}

    summary = {
        "ranking_selected": sel, "rating_selected": sel,
        "best_rwr": {"alpha": float(best_rwr.alpha), "val_ndcg": float(best_rwr.ndcg_mean)},
        "best_path": {"L": int(best_path.max_path_len), "beta": float(best_path.beta),
                      "normalize": best_path.normalize,
                      "val_ndcg": float(best_path.ndcg_mean)},
        "depth_vs_popularity": {
            "corr_depth_pop_rank": float(np.corrcoef(path_deg.max_path_len,
                                                     path_deg.pop_rank_mean)[0, 1]),
            "corr_depth_head_frac": float(np.corrcoef(path_deg.max_path_len,
                                                      path_deg.head_frac_mean)[0, 1]),
            "corr_depth_novelty": float(np.corrcoef(path_deg.max_path_len,
                                                    path_deg.novelty_mean)[0, 1]),
            "curve": depth_curve.to_dict(orient="records"),
            "interpretation": (
                "pop_rank is the mean normalised popularity rank of the Top-10 (0 = the "
                "single most-rated movie in the catalogue, 1 = the least). A NEGATIVE "
                "correlation between path length and pop_rank therefore means deeper "
                "propagation drifts toward MORE popular items, which is the predicted "
                "behaviour: as walks lengthen, the distribution over nodes converges to "
                "the degree-proportional stationary distribution, i.e. to raw popularity."),
        },
        "restart_vs_popularity": {
            "corr_alpha_pop_rank": float(np.corrcoef(rwr.alpha, rwr.pop_rank_mean)[0, 1]),
            "corr_alpha_novelty": float(np.corrcoef(rwr.alpha, rwr.novelty_mean)[0, 1]),
            "corr_alpha_coverage": float(np.corrcoef(rwr.alpha,
                                                     rwr.catalog_coverage)[0, 1]),
            "interpretation": (
                "A LARGER alpha restarts more often, keeps the walk local to the seed "
                "user, and should raise novelty / coverage; a small alpha lets the walk "
                "wander to the global stationary (degree) distribution."),
        },
        "probability_mass_check": {
            "min_over_configs": float(rwr.mass_min.min()),
            "max_over_configs": float(rwr.mass_max.max()),
        },
        "selection_split": "validation",
        "rating_head_note": (
            "The graph models are rankers; their score -> rating map is a two-parameter "
            "least-squares calibration on log1p(score), fitted on the VALIDATION split "
            "so that it is estimated under the same conditions it is applied in. Their "
            "validation RMSE is therefore in-sample for those two parameters; the TEST "
            "RMSE in Part H is out-of-sample, and no selection uses graph RMSE."),
    }
    save_json("graph", summary)
    print(f"\nselected graph model: {sel}")
    print(f"corr(depth, pop_rank)   = {summary['depth_vs_popularity']['corr_depth_pop_rank']:+.3f}")
    print(f"corr(depth, head_frac)  = {summary['depth_vs_popularity']['corr_depth_head_frac']:+.3f}")
    print(f"corr(alpha, pop_rank)   = {summary['restart_vs_popularity']['corr_alpha_pop_rank']:+.3f}")
    return df


if __name__ == "__main__":
    main()
