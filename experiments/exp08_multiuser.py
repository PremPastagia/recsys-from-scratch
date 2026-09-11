"""Part H -- multi-user evaluation of the selected model from every paradigm, on TEST.

This is the single point in the project where the test split is consumed.  Every
hyper-parameter carried in here was selected on validation in stages 03-07.

Test-user set (reproducible by construction): every one of the 943 MovieLens users takes
part, because the per-user stratified split guarantees each of them held-out ratings.
Per-user ranking metrics are averaged over the users that have at least one held-out
*relevant* item (rating >= 4); users without one are excluded from those averages
(their recall is undefined) but still contribute to coverage and popularity statistics.
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CONFIG
from src.evaluation import evaluate, flat_row
from src.pipeline import (banner, fit_model, get_context, load_json, save_json,
                          save_table)

BASELINE_SPECS = [
    ("Random", {"family": "random", "seed": CONFIG.seed}),
    ("GlobalMean", {"family": "global_mean"}),
    ("UserMean", {"family": "user_mean"}),
    ("ItemMean", {"family": "item_mean"}),
    ("BiasBaseline", {"family": "bias", "lam_u": 10.0, "lam_i": 10.0}),
    ("Popularity", {"family": "popularity"}),
    ("PopularityPositive", {"family": "popularity_positive",
                            "tau": CONFIG.relevance_threshold}),
]

PARADIGMS = [("UBCF", "ubcf"), ("IBCF", "ibcf"), ("RegressionCF", "regression"),
             ("SLIM", "slim"), ("GraphRec", "graph")]


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 2) if sys.platform == "darwin" else r / 1024


def main() -> pd.DataFrame:
    banner("PART H - Multi-user evaluation on the TEST split")
    ds, split, ctx = get_context()

    specs: list[tuple[str, dict, str]] = [(n, s, "baseline") for n, s in BASELINE_SPECS]
    for label, key in PARADIGMS:
        sel = load_json(key)
        spec = sel["selection"]["ranking_selected"] if "selection" in sel \
            else sel["ranking_selected"]
        specs.append((label, spec, "paradigm"))
        rate = sel["selection"]["rating_selected"] if "selection" in sel \
            else sel["rating_selected"]
        if rate != spec:
            specs.append((f"{label}-rating", rate, "paradigm_rating_tuned"))

    rows, artifacts = [], {}
    test_u = split.test.u.to_numpy()
    test_i = split.test.i.to_numpy()

    for label, spec, kind in specs:
        rss0 = _rss_mb()
        model = fit_model(spec, split.R_train, split.val)
        rss1 = _rss_mb()
        res = evaluate(model, ctx, split="test")

        t0 = time.perf_counter()
        _ = model.score_all()
        score_time = time.perf_counter() - t0

        row = flat_row(res)
        row.update({"label": label, "kind": kind, "spec": str(spec),
                    "model_mb": model.model_bytes() / 1e6,
                    "peak_rss_delta_mb": max(rss1 - rss0, 0.0),
                    "score_time_s": score_time,
                    "predict_time_per_user_ms": 1000 * res["predict_time_s"] / ds.n_users})
        rows.append(row)

        artifacts[f"{label}__topk"] = res["_topk"].astype(np.int32)
        artifacts[f"{label}__testpred"] = res["_pred"][test_u, test_i].astype(np.float32)
        for m in ("ndcg", "recall", "precision", "hit_rate", "map", "novelty", "ild",
                  "pop_rank", "head_frac", "tail_frac"):
            artifacts[f"{label}__per_{m}"] = np.asarray(res["_per_user"][m], dtype=np.float32)

        print(f"{label:22s} RMSE={res['rmse']:.4f} MAE={res['mae']:.4f} "
              f"NDCG={res['ndcg']['mean']:.4f} P@10={res['precision']['mean']:.4f} "
              f"R@10={res['recall']['mean']:.4f} HR={res['hit_rate']['mean']:.4f} "
              f"MAP={res['map']['mean']:.4f} cov={res['catalog_coverage']:.3f} "
              f"nov={res['novelty']['mean']:.2f} ild={res['ild']['mean']:.3f} "
              f"fit={model.fit_time_s:6.2f}s")

    np.savez_compressed(ARTIFACTS_DIR / "model_outputs.npz", **artifacts)
    df = pd.DataFrame(rows)
    save_table("multiuser_test", df)

    save_json("multiuser", {
        "n_users_total": ds.n_users,
        "n_users_with_test_positives": len(ctx.relevant_test),
        "relevance_threshold": CONFIG.relevance_threshold,
        "k": CONFIG.top_k,
        "test_ratings": int(len(split.test)),
        "models": df.label.tolist(),
        "note": ("Aggregates are mean / median / std over per-user values; the *_std "
                 "columns are the across-user standard deviation, not a bootstrap "
                 "confidence interval."),
        "rows": df.to_dict(orient="records"),
    })
    return df


if __name__ == "__main__":
    main()
