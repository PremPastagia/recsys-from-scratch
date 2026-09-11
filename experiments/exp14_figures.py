"""Part O -- render every report figure from the saved results tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import visualization as V
from src.config import CONFIG
from src.data import build_matrix
from src.pipeline import banner, get_context, load_json, load_table, save_json


def main() -> list:
    banner("PART O - Figures")
    ds, split, _ = get_context()
    eda = load_json("eda")
    paths = []

    paths.append(V.fig01_rating_distribution(load_table("rating_distribution"),
                                             eda["overview"]))
    paths.append(V.fig02_user_activity(load_table("ratings_per_user")))
    paths.append(V.fig03_item_popularity(load_table("ratings_per_item")))
    paths.append(V.fig04_long_tail(load_table("long_tail"), eda["popularity_groups"]))
    paths.append(V.fig05_sparsity(build_matrix(ds.ratings, ds.n_users, ds.n_items),
                                  eda["overview"], CONFIG.seed))

    ub, ib = load_table("ubcf_sweep"), load_table("ibcf_sweep")
    paths.append(V.fig06_k_vs_error(ub, ib))
    paths.append(V.fig07_k_vs_ranking(ub, ib))

    mu = load_table("multiuser_test")
    paths.append(V.fig08_ranking_comparison(mu))
    paths.append(V.fig09_coverage_comparison(mu))
    paths.append(V.fig10_novelty_diversity(mu))

    pb = load_table("popularity_bias")
    paths.append(V.fig11_popularity_bias(pb))
    paths.append(V.fig12_long_tail_exposure(pb))

    paths.append(V.fig13_regression_coefficients(load_table("regression_sim_vs_coef"),
                                                 load_json("regression")))
    sl = load_table("slim_sweep")
    paths.append(V.fig14_slim_sparsity_performance(sl))
    paths.append(V.fig15_slim_regularisation(sl))
    paths.append(V.fig16_runtime(mu))
    paths.append(V.fig17_coldstart(load_table("coldstart_user_groups"),
                                   load_table("coldstart_item_groups"),
                                   ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec"]))

    manifest = [{"figure": p.name, "bytes": p.stat().st_size} for p in paths]
    save_json("figures", {"n_figures": len(paths), "manifest": manifest})
    for m in manifest:
        print(f"  {m['figure']:44s} {m['bytes'] / 1024:7.1f} KB")
    return paths


if __name__ == "__main__":
    main()
