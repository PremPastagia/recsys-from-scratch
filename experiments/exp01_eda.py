"""Part A -- exploratory analysis, rating matrix, popularity groups, split, target user."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CONFIG, ML100K_ZIP_MD5, roll_digits, target_user_id
from src.data import build_matrix, sparsity_stats
from src.pipeline import banner, get_context, get_target, save_json, save_table


def main() -> dict:
    banner("PART A - Data exploration and preparation")
    ds, split, ctx = get_context()
    tgt = get_target()

    R_full = build_matrix(ds.ratings, ds.n_users, ds.n_items)
    stats = sparsity_stats(R_full)

    per_user = ds.ratings.groupby("u").size()
    per_item = ds.ratings.groupby("i").size().reindex(range(ds.n_items), fill_value=0)
    lo, hi = ds.rating_scale

    overview = {
        "dataset": "MovieLens 100K",
        "zip_md5": ML100K_ZIP_MD5,
        "n_users": ds.n_users,
        "n_items": ds.n_items,
        "n_ratings": int(len(ds.ratings)),
        "rating_scale_min": lo,
        "rating_scale_max": hi,
        "rating_values": sorted(float(v) for v in np.unique(ds.ratings.rating)),
        "mean_rating": float(ds.ratings.rating.mean()),
        "std_rating": float(ds.ratings.rating.std(ddof=1)),
        "avg_ratings_per_user": float(per_user.mean()),
        "median_ratings_per_user": float(per_user.median()),
        "min_ratings_per_user": int(per_user.min()),
        "max_ratings_per_user": int(per_user.max()),
        "avg_ratings_per_item": float(per_item.mean()),
        "median_ratings_per_item": float(per_item.median()),
        "min_ratings_per_item": int(per_item.min()),
        "max_ratings_per_item": int(per_item.max()),
        "items_with_zero_ratings": int((per_item == 0).sum()),
        **stats,
    }

    pg = ctx.pop_groups
    groups = {
        "threshold_rule": (
            "Items are sorted by descending TRAIN interaction count and the cumulative "
            "interaction curve is cut at 1/3 and 2/3 of total interaction mass. Each "
            "group therefore absorbs the same share of observed attention, which makes "
            "the partition scale-free and exposure-weighted rather than dependent on an "
            "arbitrary rating-count constant."
        ),
        "cuts_of_interaction_mass": pg["cuts"],
        "n_head": pg["n_head"], "n_medium": pg["n_medium"], "n_long_tail": pg["n_long_tail"],
        "head_pct_of_catalog": 100.0 * pg["n_head"] / ds.n_items,
        "medium_pct_of_catalog": 100.0 * pg["n_medium"] / ds.n_items,
        "long_tail_pct_of_catalog": 100.0 * pg["n_long_tail"] / ds.n_items,
        "head_min_train_count": pg["head_min_count"],
        "medium_min_train_count": pg["medium_min_count"],
        "total_train_interactions": pg["total_interactions"],
        "alternative_top20pct_rule_head_size": int(round(0.2 * ds.n_items)),
    }

    ug = ctx.user_groups
    user_groups = {
        "rule": "Tertiles of TRAIN history length (<=q1 sparse, <=q2 medium, else heavy).",
        "q1_history": ug["q1"], "q2_history": ug["q2"],
        "n_sparse": int((ug["group"] == "sparse").sum()),
        "n_medium": int((ug["group"] == "medium").sum()),
        "n_heavy": int((ug["group"] == "heavy").sum()),
        "min_history": int(ug["history"].min()),
        "max_history": int(ug["history"].max()),
    }

    split_info = {
        **split.summary(),
        "protocol": (
            "Per-user stratified random split: within each user's ratings, 20% go to "
            "test and 10% to validation, the rest to train, with every user guaranteed "
            "at least one training rating."
        ),
        "leakage_argument": (
            "1) Models are fitted on R_train alone -- every similarity, bias, learned "
            "weight, Gram matrix and graph edge is derived from train. "
            "2) Top-N candidate sets exclude only TRAIN-rated items; excluding test items "
            "would reveal which items are held out. "
            "3) Hyper-parameters (k, lambda, ranking head, restart strength) are chosen on "
            "the validation split; the test split is consumed once, at reporting time. "
            "4) Popularity groups, novelty weights and user-history tertiles are all "
            "computed from train counts, so even the analysis strata are leakage-free."
        ),
        "test_positives": int((split.test.rating >= CONFIG.relevance_threshold).sum()),
        "val_positives": int((split.val.rating >= CONFIG.relevance_threshold).sum()),
        "users_with_test_positives": len(ctx.relevant_test),
    }

    digits = "".join(c for c in CONFIG.roll if c.isdigit())
    target = {
        "roll_number": CONFIG.roll,
        "digits_extracted": digits,
        "digits_as_int": roll_digits(),
        "n_users": ds.n_users,
        "modulo": roll_digits() % ds.n_users,
        "formula": "target_user_id = (int(digits(roll)) mod n_users) + 1",
        "target_user_id": tgt.user_id,
        "target_internal_index": tgt.u,
        "seed_formula": "SEED = int(digits(roll)) mod (2**31 - 1)",
        "seed": CONFIG.seed,
        "mapping_explanation": tgt.mapping_explanation,
        "n_ratings_total": int((ds.ratings.u == tgt.u).sum()),
        "n_train_before_hiding": int(len(tgt.kept) + len(tgt.hidden)),
        "n_kept_visible": int(len(tgt.kept)),
        "n_hidden": int(len(tgt.hidden)),
        "n_global_test": int((split.test.u == tgt.u).sum()),
        "hide_fraction": CONFIG.target_hide_frac,
        "target_mean_rating": float(ds.ratings.loc[ds.ratings.u == tgt.u, "rating"].mean()),
    }

    payload = {"overview": overview, "popularity_groups": groups,
               "user_history_groups": user_groups, "split": split_info,
               "target_user": target, "config": CONFIG.as_dict()}
    save_json("eda", payload)

    # Tables that drive the Part O figures.
    save_table("rating_distribution",
               ds.ratings.rating.value_counts().sort_index()
               .rename_axis("rating").reset_index(name="count"))
    save_table("ratings_per_user",
               pd.DataFrame({"u": per_user.index, "n_ratings": per_user.to_numpy()}))
    save_table("ratings_per_item",
               pd.DataFrame({"i": per_item.index, "n_ratings": per_item.to_numpy(),
                             "title": ds.titles[per_item.index.to_numpy()]}))
    order = pg["order"]
    save_table("long_tail",
               pd.DataFrame({"rank": np.arange(1, ds.n_items + 1),
                             "item": order,
                             "train_count": pg["counts"][order],
                             "cum_share": pg["cum_mass"],
                             "group": pg["group"][order]}))

    print(f"users={ds.n_users} items={ds.n_items} ratings={len(ds.ratings)} "
          f"scale=[{lo:g},{hi:g}]")
    print(f"sparsity={stats['sparsity_pct']:.3f}%  density={stats['density_pct']:.3f}%")
    print(f"avg ratings/user={overview['avg_ratings_per_user']:.1f}  "
          f"avg ratings/item={overview['avg_ratings_per_item']:.1f}")
    print(f"popularity groups: head={pg['n_head']} medium={pg['n_medium']} "
          f"tail={pg['n_long_tail']}")
    print(f"split: {split_info['n_train']}/{split_info['n_val']}/{split_info['n_test']}")
    print(f"target user: {tgt.mapping_explanation}")
    return payload


if __name__ == "__main__":
    main()
