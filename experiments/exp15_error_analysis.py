"""Section 19 -- error analysis: where each model's error actually lives."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics as Met
from src.config import ARTIFACTS_DIR, CONFIG
from src.pipeline import banner, get_context, load_table, save_json, save_table

PARADIGMS = ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec"]


def main():
    banner("ERROR ANALYSIS")
    ds, split, ctx = get_context()
    art = dict(np.load(ARTIFACTS_DIR / "model_outputs.npz"))
    labels = load_table("multiuser_test").label.tolist()

    te = split.test
    y = te.rating.to_numpy()
    tu, ti = te.u.to_numpy(), te.i.to_numpy()
    item_grp = ctx.item_group[ti]
    user_grp = ctx.user_group[tu]

    rows = []
    for label in labels:
        p = art[f"{label}__testpred"].astype(np.float64)
        err = p - y
        for val in sorted(np.unique(y)):
            m = y == val
            rows.append({"model": label, "stratum": "true_rating",
                         "level": f"{val:g}", "n": int(m.sum()),
                         "rmse": Met.rmse(y[m], p[m]), "mae": Met.mae(y[m], p[m]),
                         "bias": float(err[m].mean())})
        for g in ("head", "medium", "long_tail"):
            m = item_grp == g
            rows.append({"model": label, "stratum": "item_group", "level": g,
                         "n": int(m.sum()), "rmse": Met.rmse(y[m], p[m]),
                         "mae": Met.mae(y[m], p[m]), "bias": float(err[m].mean())})
        for g in ("sparse", "medium", "heavy"):
            m = user_grp == g
            rows.append({"model": label, "stratum": "user_group", "level": g,
                         "n": int(m.sum()), "rmse": Met.rmse(y[m], p[m]),
                         "mae": Met.mae(y[m], p[m]), "bias": float(err[m].mean())})
        rows.append({"model": label, "stratum": "overall", "level": "all",
                     "n": int(len(y)), "rmse": Met.rmse(y, p), "mae": Met.mae(y, p),
                     "bias": float(err.mean())})
    strata = pd.DataFrame(rows)
    save_table("error_strata", strata)

    # ---- ranking-side error: where does the first hit land? --------------------------
    rank_rows = []
    for label in labels:
        topk = art[f"{label}__topk"]
        first_hits, n_hit, n_eval = [], 0, 0
        for u, rel in ctx.relevant_test.items():
            n_eval += 1
            lst = topk[u][:CONFIG.top_k]
            pos = [j + 1 for j, it in enumerate(lst) if int(it) in rel]
            if pos:
                n_hit += 1
                first_hits.append(pos[0])
        rank_rows.append({
            "model": label, "users_evaluated": n_eval, "users_with_a_hit": n_hit,
            "hit_rate": n_hit / max(n_eval, 1),
            "mean_first_hit_rank": float(np.mean(first_hits)) if first_hits else np.nan,
            "median_first_hit_rank": float(np.median(first_hits)) if first_hits else np.nan,
        })
    rankdf = pd.DataFrame(rank_rows)
    save_table("error_ranking", rankdf)

    # ---- hardest items for the best rating model -------------------------------------
    best_rating_model = strata[(strata.stratum == "overall")
                               & (strata.model.isin(PARADIGMS))].sort_values("rmse").iloc[0]
    p = art[f"{best_rating_model.model}__testpred"].astype(np.float64)
    per_item = pd.DataFrame({"i": ti, "err2": (p - y) ** 2}).groupby("i")
    hard = per_item.agg(n=("err2", "size"), rmse=("err2", lambda s: float(np.sqrt(s.mean()))))
    hard = hard[hard.n >= 10].sort_values("rmse", ascending=False).head(20).reset_index()
    hard["title"] = ds.titles[hard.i.to_numpy()]
    hard["train_popularity"] = ctx.item_pop[hard.i.to_numpy()].astype(int)
    hard["group"] = ctx.item_group[hard.i.to_numpy()]
    hard["model"] = best_rating_model.model

    # Characterise the hard items instead of asserting why they are hard: compare the
    # dispersion and polarisation of their FULL rating distributions against the
    # catalogue-wide values over items with comparable support.
    allr = ds.ratings
    by_item = allr.groupby("i").rating
    var_all = by_item.var(ddof=1)
    n_all = by_item.size()
    extreme_all = allr.assign(ext=allr.rating.isin([1.0, 2.0, 5.0])).groupby("i").ext.mean()
    comparable = n_all[n_all >= 50].index          # the hard items all have >= 50 ratings
    hard_ids = hard.i.to_numpy()
    hard["rating_variance"] = var_all.reindex(hard_ids).to_numpy()
    hard["extreme_share"] = extreme_all.reindex(hard_ids).to_numpy()
    hard["n_ratings_total"] = n_all.reindex(hard_ids).to_numpy()
    save_table("error_hardest_items", hard)

    polarisation = {
        "n_hard_items": int(len(hard)),
        "min_test_ratings": 10,
        "hard_mean_rating_variance": float(np.nanmean(hard.rating_variance)),
        "catalog_mean_rating_variance_comparable_support":
            float(var_all.reindex(comparable).mean()),
        "hard_mean_extreme_share": float(np.nanmean(hard.extreme_share)),
        "catalog_mean_extreme_share_comparable_support":
            float(extreme_all.reindex(comparable).mean()),
        "comparable_support_threshold": 50,
        "n_comparable_items": int(len(comparable)),
        "hard_items_more_dispersed": bool(
            np.nanmean(hard.rating_variance) > var_all.reindex(comparable).mean()),
    }

    ov = strata[strata.stratum == "overall"].set_index("model")
    tr = strata[strata.stratum == "true_rating"]
    payload = {
        "best_rating_model_among_paradigms": str(best_rating_model.model),
        "overall": ov[["rmse", "mae", "bias"]].to_dict(orient="index"),
        "regression_to_the_mean": {
            m: {r.level: {"bias": float(r.bias), "rmse": float(r.rmse), "n": int(r.n)}
                for _, r in tr[tr.model == m].iterrows()}
            for m in PARADIGMS if m in set(tr.model)
        },
        "by_item_group": strata[strata.stratum == "item_group"]
            .pivot(index="model", columns="level", values="rmse").to_dict(),
        "by_user_group": strata[strata.stratum == "user_group"]
            .pivot(index="model", columns="level", values="rmse").to_dict(),
        "ranking_errors": rankdf.to_dict(orient="records"),
        "hardest_items": hard.to_dict(orient="records"),
        "hard_item_polarisation": polarisation,
    }
    save_json("error_analysis", payload)

    print(strata[(strata.stratum == "true_rating") & (strata.model.isin(PARADIGMS))]
          .pivot(index="model", columns="level", values="bias").round(3).to_string())
    print("\nfirst-hit position:")
    print(rankdf.round(3).to_string(index=False))
    print(f"\nhardest items: mean rating variance "
          f"{polarisation['hard_mean_rating_variance']:.3f} vs catalogue "
          f"{polarisation['catalog_mean_rating_variance_comparable_support']:.3f} "
          f"(items with >= 50 ratings); extreme-rating share "
          f"{polarisation['hard_mean_extreme_share']:.3f} vs "
          f"{polarisation['catalog_mean_extreme_share_comparable_support']:.3f}")
    return strata


if __name__ == "__main__":
    main()
