"""Part F -- SLIM: training, sparsity, regularisation study, threshold sensitivity."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation import evaluate, flat_row
from src.pipeline import banner, fit_model, get_context, save_json, save_table

# Three orthogonal cuts through the (l1, l2) plane, plus an explicit elastic-net mixing
# sweep at fixed total strength -- so L1 strength, L2 strength and the mixing ratio are
# each varied while the other two factors are held fixed.
L1_SWEEP = [(l1, 50.0, "l1_sweep") for l1 in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)]
L2_SWEEP = [(2.0, l2, "l2_sweep") for l2 in (0.0, 1.0, 10.0, 200.0, 1000.0)]
TOTAL = 50.0
MIX_SWEEP = [(round(rho * TOTAL, 4), round((1 - rho) * TOTAL, 4), "elasticnet_mix")
             for rho in (0.02, 0.05, 0.2, 0.5, 0.8, 0.98)]
THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)
TOP_M = (25, 50, 100, 200, 400, None)


def _row(model, res, tag, extra=None):
    sp = model.sparsity()
    row = flat_row(res)
    row.update({
        "tag": tag, "l1": model.l1, "l2": model.l2, "threshold": model.threshold,
        "top_m_features": model.top_m_features if model.top_m_features else -1,
        "nnz": sp["nnz"], "sparsity_pct": sp["sparsity_pct"],
        "density_pct": sp["density_pct"],
        "mean_nnz_per_column": sp["mean_nnz_per_column"],
        "empty_columns": sp["empty_columns"],
        "iterations": model.info["iterations"],
        "converged": model.info["converged"],
        "kkt_relative_violation": model.kkt["relative_violation"],
        "objective": model.info["objective"],
        "fit_time_s": model.fit_time_s,
        "implicit_nnz": int(model.X.nnz),
    })
    if extra:
        row.update(extra)
    return row


def main() -> pd.DataFrame:
    banner("PART F - SLIM-style recommender")
    ds, split, ctx = get_context()
    rows = []

    print("\n-- regularisation study (validation split) --")
    for l1, l2, tag in L1_SWEEP + L2_SWEEP + MIX_SWEEP:
        spec = {"family": "slim", "l1": l1, "l2": l2,
                "threshold": CONFIG.implicit_threshold, "max_iter": 1500}
        model = fit_model(spec, split.R_train, split.val)
        res = evaluate(model, ctx, split="val")
        rho = l1 / (l1 + l2) if (l1 + l2) > 0 else 0.0
        rows.append(_row(model, res, tag, {"l1_ratio": rho}))
        print(f"  {tag:15s} l1={l1:6.2f} l2={l2:7.2f} rho={rho:.2f} "
              f"sparsity={rows[-1]['sparsity_pct']:6.2f}% nnz={rows[-1]['nnz']:7d} "
              f"NDCG={res['ndcg']['mean']:.4f} cov={res['catalog_coverage']:.3f} "
              f"it={model.info['iterations']:4d} kkt={model.kkt['relative_violation']:.1e} "
              f"{model.fit_time_s:5.1f}s")

    reg_df = pd.DataFrame(rows)
    best = reg_df.loc[reg_df.ndcg_mean.idxmax()]
    best_l1, best_l2 = float(best.l1), float(best.l2)

    print("\n-- implicit-feedback threshold sensitivity --")
    thr_rows = []
    for tau in THRESHOLDS:
        spec = {"family": "slim", "l1": best_l1, "l2": best_l2,
                "threshold": tau, "max_iter": 1500}
        model = fit_model(spec, split.R_train, split.val)
        res = evaluate(model, ctx, split="val")
        thr_rows.append(_row(model, res, "threshold", {"l1_ratio": best_l1 / (best_l1 + best_l2)}))
        print(f"  tau={tau:g}  positives={model.X.nnz:6d} "
              f"({100 * model.X.nnz / split.R_train.nnz:5.1f}% of train) "
              f"sparsity={thr_rows[-1]['sparsity_pct']:6.2f}% "
              f"NDCG={res['ndcg']['mean']:.4f} cov={res['catalog_coverage']:.3f}")

    print("\n-- feature-restriction (sparsity vs runtime vs quality) --")
    fm_rows = []
    for m in TOP_M:
        spec = {"family": "slim", "l1": best_l1, "l2": best_l2,
                "threshold": CONFIG.implicit_threshold, "max_iter": 1500,
                "top_m_features": m}
        model = fit_model(spec, split.R_train, split.val)
        res = evaluate(model, ctx, split="val")
        fm_rows.append(_row(model, res, "feature_mask", {"l1_ratio": best_l1 / (best_l1 + best_l2)}))
        print(f"  top_m={str(m):>5s} sparsity={fm_rows[-1]['sparsity_pct']:6.2f}% "
              f"NDCG={res['ndcg']['mean']:.4f} fit={model.fit_time_s:5.1f}s")

    df = pd.concat([reg_df, pd.DataFrame(thr_rows), pd.DataFrame(fm_rows)], ignore_index=True)
    save_table("slim_sweep", df)

    # ---- final model at the selected regularisation ----------------------------------
    best_thr = pd.DataFrame(thr_rows).loc[pd.DataFrame(thr_rows).ndcg_mean.idxmax()]
    final_spec = {"family": "slim", "l1": best_l1, "l2": best_l2,
                  "threshold": float(best_thr.threshold), "max_iter": 1500}
    model = fit_model(final_spec, split.R_train, split.val)
    sp_stats = model.sparsity()

    pairs = model.strongest_pairs(30)
    save_table("slim_strongest_pairs", pd.DataFrame([
        {"from_item": a, "from_title": ds.titles[a], "to_item": b,
         "to_title": ds.titles[b], "coefficient": w,
         "from_train_pop": int(ctx.item_pop[a]), "to_train_pop": int(ctx.item_pop[b])}
        for a, b, w in pairs]))

    nz = model.W[model.W > 0]
    summary = {
        "final_spec": final_spec,
        "ranking_selected": {**final_spec, "rank_mode": "score"},
        "rating_selected": {**final_spec, "rank_mode": "score"},
        "sparsity": sp_stats,
        "solver": {kk: vv for kk, vv in model.info.items() if kk != "history"},
        "kkt": model.kkt,
        "coefficient_stats": {
            "n_nonzero": int(nz.size), "mean": float(nz.mean()),
            "median": float(np.median(nz)), "std": float(nz.std(ddof=1)),
            "max": float(nz.max()), "p99": float(np.percentile(nz, 99)),
            "min_nonzero": float(nz.min()),
        },
        "implicit_conversion": {
            "rule": "X[u,i] = 1 iff r_ui >= tau, computed on TRAIN ratings only",
            "selected_tau": float(best_thr.threshold),
            "justification": (
                "MovieLens is an explicit 1-5 dataset, but SLIM models item co-occurrence "
                "in *consumption* data, so the ratings must be reduced to a binary "
                "'this user endorsed this item' signal. tau is not assumed: it is swept. "
                "tau=1 keeps every rating and therefore treats a 1-star review as an "
                "endorsement, which injects negative evidence as if it were positive. "
                "tau=5 is too strict and starves most items of support. The sweep in "
                "slim_sweep.csv (tag='threshold') reports NDCG@10, sparsity and catalogue "
                "coverage at every tau on the validation split, and the value recorded "
                "here is the measured argmax."),
            "sweep": [{"tau": float(r["threshold"]), "positives": int(r["implicit_nnz"]),
                       "ndcg": float(r["ndcg_mean"]),
                       "coverage": float(r["catalog_coverage"])} for r in thr_rows],
        },
        "best_regularisation": {"l1": best_l1, "l2": best_l2,
                                "l1_ratio": best_l1 / (best_l1 + best_l2),
                                "val_ndcg": float(best.ndcg_mean)},
        "accuracy_sparsity_tradeoff": {
            "corr_sparsity_ndcg": float(np.corrcoef(reg_df.sparsity_pct,
                                                    reg_df.ndcg_mean)[0, 1]),
            "corr_sparsity_fit_time": float(np.corrcoef(reg_df.sparsity_pct,
                                                        reg_df.fit_time_s)[0, 1]),
            "corr_sparsity_coverage": float(np.corrcoef(reg_df.sparsity_pct,
                                                        reg_df.catalog_coverage)[0, 1]),
        },
        "selection_split": "validation",
        "rating_head_note": (
            "SLIM is an implicit-feedback ranker. Its score -> rating map is a "
            "two-parameter least-squares calibration fitted on the VALIDATION split, "
            "because fitting it on training pairs estimates the line in a regime the "
            "model never sees at inference time. Consequently the validation RMSE "
            "reported in the sweep is in-sample for those two parameters (with 2 "
            "parameters on ~9.6k points the optimism is negligible, but it is stated "
            "for completeness); the TEST RMSE in Part H is out-of-sample. No selection "
            "anywhere in this project uses SLIM's RMSE."),
    }
    save_json("slim", summary)
    print(f"\nfinal SLIM: l1={best_l1:g} l2={best_l2:g} tau={best_thr.threshold:g}")
    print(f"  sparsity={sp_stats['sparsity_pct']:.3f}%  nnz={sp_stats['nnz']}  "
          f"mean nnz/col={sp_stats['mean_nnz_per_column']:.1f}")
    print(f"  KKT relative violation={model.kkt['relative_violation']:.2e} "
          f"(converged={model.info['converged']})")
    return df


if __name__ == "__main__":
    main()
