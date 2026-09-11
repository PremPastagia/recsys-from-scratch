"""Part B -- non-personalised and bias-only baselines."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import CONFIG
from src.evaluation import evaluate, flat_row
from src.pipeline import banner, build_model, get_context, save_json, save_table

SPECS = [
    {"family": "random", "seed": CONFIG.seed},
    {"family": "global_mean"},
    {"family": "user_mean"},
    {"family": "item_mean"},
    {"family": "bias", "lam_u": 10.0, "lam_i": 10.0},
    {"family": "popularity"},
    {"family": "popularity_positive", "tau": CONFIG.relevance_threshold},
]


def main() -> pd.DataFrame:
    banner("PART B - Baselines")
    _, split, ctx = get_context()
    rows = []
    for spec in SPECS:
        model = build_model(spec).fit(split.R_train)
        res = evaluate(model, ctx, split="test")
        row = flat_row(res)
        row["spec"] = str(spec)
        rows.append(row)
        print(f"{model.name:22s} RMSE={res['rmse']:.4f} MAE={res['mae']:.4f} "
              f"NDCG@10={res['ndcg']['mean']:.4f} P@10={res['precision']['mean']:.4f} "
              f"R@10={res['recall']['mean']:.4f} HR@10={res['hit_rate']['mean']:.4f} "
              f"cov={res['catalog_coverage']:.3f}")

    df = pd.DataFrame(rows)
    save_table("baselines", df)

    g = df.set_index("model")
    save_json("baselines", {
        "rows": df.to_dict(orient="records"),
        "sanity_checks": {
            "user_mean_beats_global_rmse":
                bool(g.loc["UserMean", "rmse"] < g.loc["GlobalMean", "rmse"]),
            "item_mean_beats_global_rmse":
                bool(g.loc["ItemMean", "rmse"] < g.loc["GlobalMean", "rmse"]),
            "bias_beats_both_rmse":
                bool(g.loc["BiasBaseline", "rmse"] < min(g.loc["UserMean", "rmse"],
                                                         g.loc["ItemMean", "rmse"])),
            "popularity_beats_random_ndcg":
                bool(g.loc["Popularity", "ndcg_mean"] > g.loc["Random", "ndcg_mean"]),
        },
        "why_baselines": (
            "Absolute error numbers are uninterpretable without a reference. Predicting "
            "each user's own mean already reaches RMSE "
            f"{g.loc['UserMean', 'rmse']:.4f} and the regularised bias model reaches "
            f"{g.loc['BiasBaseline', 'rmse']:.4f}; any collaborative model must be judged "
            "against that, not against zero. On the ranking side the asymmetry is even "
            "sharper: non-personalised popularity reaches NDCG@10 = "
            f"{g.loc['Popularity', 'ndcg_mean']:.4f} versus "
            f"{g.loc['Random', 'ndcg_mean']:.4f} for random, so a personalised model that "
            "scores 0.15 has added exactly nothing while paying for a similarity matrix."
        ),
    })
    return df


if __name__ == "__main__":
    main()
