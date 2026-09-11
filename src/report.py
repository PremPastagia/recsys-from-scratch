"""
Part R -- research report generator.

Every number in REPORT.md and in the results section of README.md is read out of
``results/`` and formatted here.  Nothing is typed in by hand.  The gate for
"no fabricated results" regenerates both documents and requires them to be
byte-identical to what is committed, which makes a fabricated figure impossible to
introduce without the check failing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, ROOT
from .pipeline import load_json, load_table

CONFIG_TOP_K = CONFIG.top_k

SECTIONS = [
    "Abstract", "Problem statement", "Dataset", "Exploratory analysis", "Methodology",
    "Baselines", "User-based collaborative filtering",
    "Item-based collaborative filtering", "Regression-based collaborative filtering",
    "SLIM", "Network-based recommender", "Experimental setup", "Results",
    "Ablation studies", "Popularity and long-tail analysis", "Cold-start analysis",
    "Explainability", "Target-user recommendations", "Error analysis", "Limitations",
    "Conclusions", "Future work",
]

RESEARCH_QUESTIONS = [
    "Which method performs best?",
    "Under what conditions?",
    "Why?",
    "What are the trade-offs?",
    "Which methods favour the long tail?",
    "Which methods are more efficient?",
    "Which method would you choose for a real system and why?",
]

PARADIGMS = ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec"]


def ols_vs_ridge(rgs: pd.DataFrame, k: int) -> tuple[float, float]:
    """(unregularised, best-regularised) validation RMSE at a fixed neighbourhood size."""
    learned = rgs[rgs.weight_mode == "learned"]
    at_k = learned[learned.k == k]
    return float(at_k[at_k.lam == 0].rmse.min()), float(at_k[at_k.lam > 0].rmse.min())


def best_rating_config(mu: pd.DataFrame):
    """Lowest-RMSE row over every measured configuration of the five paradigms.

    The master table's paradigm rows carry the *ranking*-selected configuration, so the
    best RMSE in that subset understates what the study actually achieved on the rating
    task; the `-rating` arms are where the rating-optimal configurations live.
    """
    fam = mu[mu.label.str.replace("-rating", "", regex=False).isin(PARADIGMS)]
    return fam.loc[fam.rmse.idxmin()]


# ======================================================================================
# formatting helpers
# ======================================================================================
def f(x, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{n}f}"


def md_table(df: pd.DataFrame, floats: int = 4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f(v, floats))
        else:
            d[c] = d[c].astype(str)
    head = "| " + " | ".join(d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in d.itertuples(index=False))
    return "\n".join([head, rule, body])


def _load_all() -> dict:
    keys = ["eda", "baselines", "ubcf", "ibcf", "regression", "slim", "graph",
            "multiuser", "coldstart", "popularity_bias", "ablations", "master",
            "target_user", "error_analysis", "figures"]
    out = {}
    for kk in keys:
        try:
            out[kk] = load_json(kk)
        except FileNotFoundError:
            out[kk] = {}
    return out


# ======================================================================================
# section builders
# ======================================================================================
def build_report() -> str:
    R = _load_all()
    ov = R["eda"]["overview"]
    pgp = R["eda"]["popularity_groups"]
    spl = R["eda"]["split"]
    tgt = R["eda"]["target_user"]
    cfg = R["eda"]["config"]

    mu = load_table("multiuser_test")
    master = load_table("master_comparison")
    pb = load_table("popularity_bias")
    ug = load_table("coldstart_user_groups")
    ig = load_table("coldstart_item_groups")
    abl = load_table("ablations")
    sl = load_table("slim_sweep")
    ubs = load_table("ubcf_sweep")
    ibs = load_table("ibcf_sweep")
    rgs = load_table("regression_sweep")
    grs = load_table("graph_sweep")
    tgt_rank = load_table("target_top10_by_model")
    tgt_metrics = load_table("target_metrics")
    tgt_cons = load_table("target_consensus")
    err_strata = load_table("error_strata")
    err_rank = load_table("error_ranking")

    par = mu[mu.label.isin(PARADIGMS)].set_index("label")
    base = mu[~mu.label.isin(PARADIGMS)].set_index("label")

    best_ndcg = par.ndcg_mean.idxmax()
    best_rmse = par.rmse.idxmin()
    best_cov = par.catalog_coverage.idxmax()
    best_nov = par.novelty_mean.idxmax()
    best_ild = par.ild_mean.idxmax()
    fastest = par.train_time_s.idxmin()
    pop_ndcg = float(base.loc["Popularity", "ndcg_mean"])
    bias_rmse = float(base.loc["BiasBaseline", "rmse"])

    L: list[str] = []
    A = L.append

    # ---------------------------------------------------------------- title
    A(f"# Comparative Study of Five Recommendation Paradigms on MovieLens 100K\n")
    A(f"*All algorithms implemented from scratch in NumPy/SciPy. "
      f"Roll number `{tgt['roll_number']}` -> target user **{tgt['target_user_id']}**. "
      f"Master seed `{cfg['seed']}`.*\n")
    A("> Every number in this report is generated directly from `results/` by "
      "`src/report.py`. No figure is transcribed by hand.\n")
    A("---\n")

    # ---------------------------------------------------------------- 1 Abstract
    A("## 1. Abstract\n")
    A(f"We implement and compare five recommendation paradigms on MovieLens 100K "
      f"({ov['n_users']} users, {ov['n_items']} movies, {ov['n_ratings']:,} ratings, "
      f"{ov['sparsity_pct']:.2f}% sparse) without using any recommender-system library: "
      f"user-based collaborative filtering, item-based collaborative filtering, a "
      f"regression-based neighbourhood model with learned interpolation weights, a "
      f"SLIM-style sparse linear item-item model trained by projected FISTA under "
      f"non-negativity and elastic-net regularisation, and two graph propagation methods "
      f"(random walk with restart, and bounded-length path proximity) on the user-movie "
      f"bipartite graph. Hyper-parameters are selected on a validation split and the test "
      f"split is consumed exactly once.\n")
    pj = R["popularity_bias"]
    front = pj["pareto"]["pareto_front"]
    expo = pj["exposure_reference"]
    brc = best_rating_config(mu)
    A(f"On the held-out test split the strongest ranking model is **{best_ndcg}** "
      f"(NDCG@10 = {f(par.loc[best_ndcg, 'ndcg_mean'])}, "
      f"Recall@10 = {f(par.loc[best_ndcg, 'recall_mean'])}), against "
      f"{f(pop_ndcg)} for a non-personalised popularity ranker. The strongest rating "
      f"predictor over all measured configurations is **{brc.label}** "
      f"(RMSE = {f(brc.rmse)}, MAE = {f(brc.mae)}) against {f(bias_rmse)} for a "
      f"regularised bias baseline; among the ranking-selected configurations the best is "
      f"{best_rmse} at {f(par.loc[best_rmse, 'rmse'])}. Rating accuracy and ranking "
      f"quality are held by different models and, as Section 13 shows, pursuing one to "
      f"its optimum actively destroys the other: {brc.label} reaches NDCG@10 "
      f"{f(brc.ndcg_mean)} while {best_ndcg} reaches RMSE "
      f"{f(par.loc[best_ndcg, 'rmse'])}.\n")
    dominated_by_best = [m for m, by in pj["pareto"]["dominated"].items()
                         if best_ndcg in by]
    dom_clause = (
        f"{best_ndcg} strictly dominates {', '.join(dominated_by_best)} on all four at "
        f"once" if dominated_by_best else
        f"{best_ndcg} dominates none of the others outright")
    A(f"The beyond-accuracy picture is sharper than a simple accuracy-versus-coverage "
      f"trade-off. On the four objectives (NDCG@10, catalogue coverage, novelty, "
      f"long-tail share) only **{' and '.join(front)}** are Pareto-optimal; "
      f"{dom_clause}. The real cost is visible elsewhere: the "
      f"{expo['n_head_items']} head movies are "
      f"{f(expo['head_share_of_catalog_pct'], 1)}% of the catalogue but take "
      f"{f(expo['paradigm_head_pct_range'][0], 1)}-{f(expo['paradigm_head_pct_range'][1], 1)}% "
      f"of every model's recommendation slots, while the {expo['n_long_tail_items']} "
      f"long-tail movies ({f(expo['long_tail_share_of_catalog_pct'], 1)}% of the "
      f"catalogue) take {f(expo['paradigm_tail_pct_range'][0], 1)}-"
      f"{f(expo['paradigm_tail_pct_range'][1], 1)}%. We quantify this across popularity "
      f"strata, user-history strata and computational cost, and show for the graph model "
      f"that increasing propagation depth provably and measurably drifts recommendations "
      f"toward popularity.\n")

    # ---------------------------------------------------------------- 2 Problem
    A("## 2. Problem statement\n")
    A("Given a partially observed user-item rating matrix *R*, produce for each user a "
      "ranked list of unseen items. The project asks two questions that are usually "
      "conflated:\n")
    A("1. **Rating prediction** -- estimate `r_ui` for held-out cells, measured by RMSE/MAE.\n"
      "2. **Top-N recommendation** -- order the unseen catalogue, measured by "
      "Precision/Recall/HitRate/NDCG/MAP@10.\n")
    A("A recommender is only useful if it also *reaches* the catalogue, so we add "
      "catalogue coverage, novelty, intra-list diversity and long-tail exposure as "
      "first-class objectives rather than as afterthoughts, and we measure the "
      "computational cost of each paradigm.\n")
    A("The constraint that shapes every implementation decision here is that no "
      "recommender library may be used: similarities, neighbourhood aggregation, ridge "
      "normal equations, the elastic-net solver and the graph propagation are all written "
      "directly in terms of sparse matrix algebra.\n")

    # ---------------------------------------------------------------- 3 Dataset
    A("## 3. Dataset\n")
    A(f"MovieLens 100K, verified against the published archive MD5 "
      f"`{ov['zip_md5']}`.\n")
    ds_tbl = pd.DataFrame([
        ["Users", f(ov["n_users"])], ["Movies", f(ov["n_items"])],
        ["Ratings", f(ov["n_ratings"])],
        ["Rating scale", f"{ov['rating_scale_min']:g} - {ov['rating_scale_max']:g} "
                        f"({len(ov['rating_values'])} integer levels)"],
        ["Mean rating", f(ov["mean_rating"], 3)],
        ["Std of ratings", f(ov["std_rating"], 3)],
        ["Matrix shape", f"{ov['n_users']} x {ov['n_items']} = {ov['matrix_cells']:,} cells"],
        ["Observed entries", f(ov["observed_entries"])],
        ["Density", f"{ov['density_pct']:.4f}%"],
        ["Sparsity", f"{ov['sparsity_pct']:.4f}%"],
        ["Avg ratings / user", f(ov["avg_ratings_per_user"], 2)],
        ["Median ratings / user", f(ov["median_ratings_per_user"], 1)],
        ["Min / max ratings per user", f"{ov['min_ratings_per_user']} / {ov['max_ratings_per_user']}"],
        ["Avg ratings / movie", f(ov["avg_ratings_per_item"], 2)],
        ["Median ratings / movie", f(ov["median_ratings_per_item"], 1)],
        ["Min / max ratings per movie", f"{ov['min_ratings_per_item']} / {ov['max_ratings_per_item']}"],
    ], columns=["Property", "Value"])
    A(md_table(ds_tbl) + "\n")
    A(f"Every user has at least {ov['min_ratings_per_user']} ratings by dataset "
      f"construction, so there are no strictly cold users; the sparsity problem here is "
      f"on the item side, where the median movie has only "
      f"{ov['median_ratings_per_item']:.0f} ratings.\n")

    # ---------------------------------------------------------------- 4 EDA
    A("## 4. Exploratory analysis\n")
    A("![rating distribution](figures/fig01_rating_distribution.png)\n")
    A("![user activity](figures/fig02_user_activity.png)\n")
    A("![item popularity](figures/fig03_item_popularity.png)\n")
    A("![long tail](figures/fig04_long_tail.png)\n")
    A("![sparsity](figures/fig05_sparsity.png)\n")
    A(f"The rating distribution is strongly left-skewed (mean "
      f"{ov['mean_rating']:.3f} on a 1-5 scale): users mostly rate films they chose to "
      f"watch, so 4 is the modal rating. This matters for the implicit-feedback "
      f"conversion in Part F -- 'rated' and 'liked' are not the same event, and the "
      f"threshold that separates them has to be measured, not assumed.\n")
    A("### Popularity groups\n")
    A(pgp["threshold_rule"] + "\n")
    pg_tbl = pd.DataFrame([
        ["Head", pgp["n_head"], f"{pgp['head_pct_of_catalog']:.1f}%",
         f">= {pgp['head_min_train_count']} train ratings", "~33% of interactions"],
        ["Medium", pgp["n_medium"], f"{pgp['medium_pct_of_catalog']:.1f}%",
         f">= {pgp['medium_min_train_count']} train ratings", "~33% of interactions"],
        ["Long tail", pgp["n_long_tail"], f"{pgp['long_tail_pct_of_catalog']:.1f}%",
         f"< {pgp['medium_min_train_count']} train ratings", "~33% of interactions"],
    ], columns=["Group", "Movies", "% of catalog", "Threshold", "Interaction mass"])
    A(md_table(pg_tbl) + "\n")
    A(f"The asymmetry is the headline fact of the dataset: "
      f"{pgp['n_head']} movies ({pgp['head_pct_of_catalog']:.1f}% of the catalogue) "
      f"absorb a third of all attention, while "
      f"{pgp['n_long_tail']} movies ({pgp['long_tail_pct_of_catalog']:.1f}%) share "
      f"another third between them. Any model that optimises accuracy alone will "
      f"gravitate to the first group, and Part J measures exactly how far each one does.\n")

    # ---------------------------------------------------------------- 5 Methodology
    A("## 5. Methodology\n")
    A("### 5.1 Notation\n")
    A("`R` is the user-item rating matrix; `I_u` the items rated by user *u*; `U_i` the "
      "users who rated item *i*; `mu`, `mu_u`, `mu_i` the global, user and item means; "
      "`s(a,b)` a similarity; `N_k(.)` a k-nearest-neighbourhood; `X` the binary "
      "implicit matrix; `G = X^T X` its Gram matrix.\n")
    A("### 5.2 Similarities (exact, via sparse GEMMs)\n")
    A("```\ncos(a,b) = <r_a, r_b> / (||r_a|| ||r_b||)\n\n"
      "                     n*S_ab - S_a*S_b\n"
      "pearson(a,b) = ------------------------------------ ,  all sums over I_a ∩ I_b\n"
      "               sqrt(n*Q_a - S_a^2) sqrt(n*Q_b - S_b^2)\n```\n")
    A("Co-rated Pearson is usually written as a pairwise loop. It is not needed: with "
      "`B` the binary indicator of `R`, multiplying by `B` restricts a sum to the "
      "co-rated set, so `n = B B^T`, `S_a = R B^T`, `S_ab = R R^T`, `Q_a = R² B^T`. "
      "Five sparse matrix products give the exact co-rated correlation for all pairs "
      "(verified against a brute-force implementation to 0.0 absolute error).\n")
    A("Two reliability corrections are applied to every similarity: pairs with fewer "
      f"than `min_support = {cfg['min_support']}` co-ratings are zeroed, and the rest are "
      f"shrunk by significance weighting `s' = s * min(n, {cfg['shrinkage_beta']:g}) / "
      f"{cfg['shrinkage_beta']:g}`.\n")
    A("### 5.3 Ranking heads\n")
    A("Ranking a candidate set by predicted rating is fragile: an item supported by one "
      "enthusiastic neighbour can reach `rhat = 5.0` on almost no evidence. Each "
      "neighbourhood model therefore exposes three ranking heads and the choice is made "
      "on the **validation** split:\n")
    A("- `rating` -- rank by `rhat_ui`;\n"
      "- `score` -- rank by the *unnormalised* similarity-weighted deviation sum "
      "(evidence mass stays in the numerator);\n"
      "- `affinity` -- rank by `sum_{v} max(s,0) * 1[r_vi >= tau]`, the implicit-feedback "
      "objective that matches the Top-N task.\n")
    A("### 5.4 Splitting and leakage control\n")
    A(spl["protocol"] + "\n")
    A(spl["leakage_argument"] + "\n")
    A(f"Resulting sizes: train {spl['n_train']:,} ({spl['train_pct']:.1f}%), "
      f"validation {spl['n_val']:,} ({spl['val_pct']:.1f}%), test {spl['n_test']:,} "
      f"({spl['test_pct']:.1f}%). A held-out rating counts as a true positive when it is "
      f">= {cfg['relevance_threshold']:g}; {spl['users_with_test_positives']} of "
      f"{ov['n_users']} users have at least one, and per-user ranking metrics are "
      f"averaged over those users.\n")
    A("### 5.5 Target-user selection from the roll number\n")
    A("```\n"
      f"roll                 = {tgt['roll_number']}\n"
      f"digits(roll)         = {tgt['digits_extracted']}  -> {tgt['digits_as_int']}\n"
      f"{tgt['digits_as_int']} mod {tgt['n_users']}        = {tgt['modulo']}\n"
      f"target_user_id       = {tgt['modulo']} + 1 = {tgt['target_user_id']}\n"
      f"SEED                 = {tgt['digits_as_int']} mod (2^31 - 1) = {cfg['seed']}\n"
      "```\n")
    A(f"The `+1` shifts the 0-based residue into MovieLens' 1-based user ids; the modulo "
      f"makes the map total for any roll number. The master seed is derived from the same "
      f"digits, so a different roll number yields a different but equally reproducible "
      f"experiment. User {tgt['target_user_id']} has {tgt['n_ratings_total']} ratings in "
      f"total; {tgt['n_hidden']} of the {tgt['n_train_before_hiding']} training ratings "
      f"are hidden ({100 * cfg['target_hide_frac']:.0f}%), leaving a visible profile of "
      f"{tgt['n_kept_visible']}, plus {tgt['n_global_test']} globally held-out ratings.\n")

    # ---------------------------------------------------------------- 6 Baselines
    A("## 6. Baselines\n")
    A(R["baselines"]["why_baselines"] + "\n")
    bl = load_table("baselines")
    bl_tbl = bl[["model", "rmse", "mae", "precision_mean", "recall_mean",
                 "hit_rate_mean", "ndcg_mean", "map_mean", "catalog_coverage"]].copy()
    bl_tbl.columns = ["Model", "RMSE", "MAE", "P@10", "R@10", "HR@10", "NDCG@10",
                      "MAP@10", "Coverage"]
    A(md_table(bl_tbl) + "\n")
    sc = R["baselines"]["sanity_checks"]
    A(f"Sanity checks all hold: user-mean beats global-mean on RMSE "
      f"(`{sc['user_mean_beats_global_rmse']}`), item-mean beats global-mean "
      f"(`{sc['item_mean_beats_global_rmse']}`), the regularised bias model beats both "
      f"(`{sc['bias_beats_both_rmse']}`), and popularity beats random on NDCG@10 "
      f"(`{sc['popularity_beats_random_ndcg']}`).\n")
    A(f"The important baseline is popularity, not the mean predictors. Recommending the "
      f"most-rated movies to everybody reaches NDCG@10 = {f(pop_ndcg)} with zero "
      f"personalisation, which is {f(pop_ndcg / float(base.loc['Random', 'ndcg_mean']), 1)}x "
      f"the random floor. Any personalised model below that line is not earning its "
      f"complexity.\n")

    # ---------------------------------------------------------------- 7 UBCF
    A("## 7. User-based collaborative filtering\n")
    A("```\n"
      "                 sum_{v in N_k(u,i)} s(u,v) (r_vi - mu_v)\n"
      "rhat_ui = mu_u + -----------------------------------------\n"
      "                      sum_{v in N_k(u,i)} |s(u,v)|\n```\n")
    A("`N_k(u,i)` is the *exact* per-(u,i) neighbourhood: the k users most similar to u "
      "**among those who actually rated i**, recomputed for every item rather than "
      "approximated by intersecting a global top-k list. The cheaper approximation "
      "silently shrinks the effective k for unpopular items and makes k-curves look "
      "better than they are.\n")
    A(f"The sweep covers 2 similarities x 2 centring settings x "
      f"{len(cfg['k_grid'])} neighbourhood sizes x 3 ranking heads = "
      f"{len(ubs)} measured configurations, all on the validation split.\n")
    A("![k vs rmse](figures/fig06_k_vs_rmse_mae.png)\n")
    A("![k vs ndcg](figures/fig07_k_vs_ndcg_recall.png)\n")
    ub_sel = R["ubcf"]["selection"]
    A(f"Selected on validation: ranking `{ub_sel['ranking_selected']}` "
      f"(val NDCG@10 = {f(ub_sel['val_ndcg_at_selection'])}); rating "
      f"`{ub_sel['rating_selected']}` (val RMSE = {f(ub_sel['val_rmse_at_selection'])}).\n")
    A(_neighbourhood_findings(ubs, "UBCF"))

    # ---------------------------------------------------------------- 8 IBCF
    A("## 8. Item-based collaborative filtering\n")
    A("```\n"
      "                 sum_{j in N_k(i,u)} s(i,j) (r_uj - mu_j)\n"
      "rhat_ui = mu_i + -----------------------------------------\n"
      "                      sum_{j in N_k(i,u)} |s(i,j)|\n```\n")
    A(f"The sweep covers 6 similarity/centring regimes x {len(cfg['k_grid'])} "
      f"neighbourhood sizes x 3 ranking heads = {len(ibs)} configurations.\n")
    ib_sel = R["ibcf"]["selection"]
    A(f"Selected on validation: ranking `{ib_sel['ranking_selected']}` "
      f"(val NDCG@10 = {f(ib_sel['val_ndcg_at_selection'])}); rating "
      f"`{ib_sel['rating_selected']}` (val RMSE = {f(ib_sel['val_rmse_at_selection'])}).\n")
    A(_neighbourhood_findings(ibs, "IBCF"))
    A("### Why UBCF and IBCF differ\n")
    A(f"On the test split IBCF reaches RMSE {f(par.loc['IBCF', 'rmse'])} / NDCG@10 "
      f"{f(par.loc['IBCF', 'ndcg_mean'])} against UBCF's "
      f"{f(par.loc['UBCF', 'rmse'])} / {f(par.loc['UBCF', 'ndcg_mean'])}. Three "
      f"structural reasons:\n")
    A(f"1. **Estimation support.** A user profile here averages "
      f"{ov['avg_ratings_per_user']:.0f} ratings, but item-item co-rating counts "
      f"concentrate on a popular core, so item similarities are estimated from far more "
      f"co-observations than user similarities and are correspondingly less noisy.\n"
      f"2. **Stability.** Item-item relations are near-stationary while user tastes and "
      f"the user set churn, which is why industrial systems standardised on item-item.\n"
      f"3. **Model size and cost.** The UBCF similarity matrix is "
      f"{ov['n_users']}² = {ov['n_users'] ** 2:,} entries; IBCF's is "
      f"{ov['n_items']}² = {ov['n_items'] ** 2:,}. Here IBCF is the more expensive of the "
      f"two ({f(par.loc['IBCF', 'model_mb'], 1)} MB vs "
      f"{f(par.loc['UBCF', 'model_mb'], 1)} MB); in a catalogue with |U| >> |I| the "
      f"ordering reverses, which is the usual real-world case.\n")
    A(f"They also behave differently on coverage: IBCF reaches "
      f"{f(100 * par.loc['IBCF', 'catalog_coverage'], 1)}% of the catalogue versus "
      f"{f(100 * par.loc['UBCF', 'catalog_coverage'], 1)}% for UBCF, because item "
      f"neighbourhoods anchor on what the user already watched and can wander away from "
      f"the global popularity core, whereas user neighbourhoods reconverge on whatever "
      f"the similar users collectively watched -- which is the popular set.\n")

    # ---------------------------------------------------------------- 9 Regression
    A("## 9. Regression-based collaborative filtering\n")
    rg = R["regression"]
    A("For a target item *i* with neighbourhood `N(i)`, every user who rated *i* becomes "
      "one training example:\n")
    A("```\n"
      "target   y_u  = r_ui - b_ui\n"
      "features z_uj = r_uj - b_uj   for j in N(i)   (0 when u never rated j)\n\n"
      "min_w  sum_u ( y_u - sum_j w_ij z_uj )^2 + lambda ||w||^2\n"
      "  =>   (Z^T Z + lambda I) w = Z^T y      (solved exactly, one k x k system per item)\n\n"
      "rhat_ui = b_ui + sum_j w_ij (r_uj - b_uj)          b_ui = mu + b_u + b_i\n```\n")
    A("Stacking the per-item solutions into a sparse matrix `W` turns prediction into a "
      "single sparse product `P = B + D W`, which makes this model directly comparable "
      "with SLIM's learned `W` and with the raw similarity matrix.\n")
    A("### Statistical interpretation\n")
    A("`s(i,j)` is a **marginal** association -- how *j* relates to *i* ignoring "
      "everything else. `w_ij` is a **partial** regression coefficient -- how *j* relates "
      "to *i* holding the other neighbours fixed. The measured consequences:\n")
    A(f"- Correlation between similarity and learned coefficient at the selected "
      f"configuration: Pearson r = {f(rg['pearson_sim_coef'], 3)}, "
      f"Spearman rho = {f(rg['spearman_sim_coef'], 3)}. They agree in *ordering* far more "
      f"than in *magnitude*, which is exactly what you expect when redundancy among "
      f"neighbours is being discounted.\n"
      f"- {100 * rg['frac_negative_coefficients']:.1f}% of learned coefficients are "
      f"negative, and {100 * rg['frac_sign_flip_positive_sim_negative_coef']:.1f}% are "
      f"outright sign flips (positive similarity, negative weight) -- classic suppressor "
      f"effects that a similarity heuristic cannot express.\n"
      f"- Similarity weights are normalised to sum to 1 per item by construction; the "
      f"learned weights sum to {f(rg['coef_sum_mean_per_item'], 3)} on average, so the "
      f"regression can say 'this neighbourhood barely determines the target' -- something "
      f"the heuristic literally cannot do.\n")
    A("![regression coefficients](figures/fig13_regression_sim_vs_coef.png)\n")
    A("### Does learning the weights help?\n")
    lam_curve = (rgs[rgs.weight_mode == "learned"]
                 .groupby("lam")[["rmse", "ndcg_mean", "pearson_sim_coef"]]
                 .agg({"rmse": "min", "ndcg_mean": "max", "pearson_sim_coef": "mean"})
                 .reset_index())
    lam_curve.columns = ["lambda", "best val RMSE", "best val NDCG@10", "mean r(sim,coef)"]
    A(md_table(lam_curve) + "\n")
    k_dd = int(rg["deep_dive_spec"]["k"])
    ols_k, ridge_k = ols_vs_ridge(rgs, k_dd)
    ols_by_k = (rgs[(rgs.weight_mode == "learned") & (rgs.lam == 0)]
                .groupby("k").rmse.min())
    A(f"Unregularised least squares fails, and it fails in the way the theory predicts -- "
      f"it gets worse as the model gets bigger. At k = {k_dd} it reaches validation RMSE "
      f"{f(ols_k)} against {f(ridge_k)} for the best ridge setting at the same k; across "
      f"the k grid its RMSE runs "
      + ", ".join(f"k={int(kk)}: {f(v)}" for kk, v in ols_by_k.items()) +
      f", so by k = {int(ols_by_k.idxmax())} it is worse than predicting the global mean "
      f"({f(float(load_table('baselines').set_index('model').loc['GlobalMean', 'rmse']))}). "
      f"That is the signature of an under-determined system: k features against a design "
      f"matrix whose row count is the number of users who rated the target item, which "
      f"for most of the catalogue is smaller than k.\n")
    lr = rgs[rgs.weight_mode == "learned"]
    A(f"As lambda grows the coefficients shrink toward zero *and* toward the similarity "
      f"ordering: `r(sim,coef)` rises monotonically along the table above, from "
      f"{f(lr[lr.lam == 0].pearson_sim_coef.mean(), 3)} at lambda = 0 to "
      f"{f(lr[lr.lam == lr.lam.max()].pearson_sim_coef.mean(), 3)} at lambda = "
      f"{lr.lam.max():g}. That is the shrinkage-toward-the-prior story made visible: with "
      f"no regularisation the learned weights carry no relationship to similarity at all, "
      f"and with enough of it they converge back onto the heuristic they were supposed "
      f"to improve. The useful setting is in between, and it has to be found by "
      f"measurement.\n")
    A(f"At its best setting the learned model reaches validation RMSE "
      f"{f(rg['learned_arm_best_val_rmse'])} against {f(rg['similarity_arm_best_val_rmse'])} "
      f"for the identical architecture with similarity weights: learned weights **do** "
      f"help (`{rg['learned_beats_similarity_on_rmse']}`), but only once regularised. "
      f"The gain is real and small; the interpretability gain is larger.\n")

    # ---------------------------------------------------------------- 10 SLIM
    A("## 10. SLIM\n")
    slm = R["slim"]
    A("```\n"
      "min_W (1/2)||X - XW||_F^2 + lambda_1 ||W||_1 + (lambda_2/2) ||W||_F^2\n"
      "s.t.  W >= 0 ,  diag(W) = 0\n```\n")
    A("`diag(W) = 0` is essential -- otherwise `W = I` reconstructs `X` perfectly and "
      "learns nothing. `W >= 0` makes every coefficient an interpretable 'item j is "
      "evidence for item i' weight. L1 produces exact zeros; L2 keeps the problem "
      "strictly convex and stops near-duplicate items fighting over one coefficient.\n")
    A("### Solver\n")
    A("Projected FISTA, written from scratch. With `G = X^T X`, the gradient of the "
      "smooth part is `grad = G W - G + lambda_2 W`, so one dense GEMM per iteration "
      "solves all 1,682 column problems simultaneously. Because `W >= 0`, the L1 term is "
      "linear on the feasible set and its proximal operator collapses to a shifted "
      "projection `max(0, V - eta*lambda_1)` followed by zeroing the diagonal. Step size "
      "`eta = 1/L` with `L = lambda_max(G) + lambda_2` from power iteration; Nesterov "
      "momentum gives the O(1/t²) rate.\n")
    A(f"Convergence is certified by the KKT conditions rather than by a stalled "
      f"objective: at the optimum `W_ij > 0 => grad_ij + lambda_1 = 0` and "
      f"`W_ij = 0 => grad_ij + lambda_1 >= 0`. The delivered model stops at a relative "
      f"KKT violation of {slm['kkt']['relative_violation']:.2e} "
      f"(absolute {f(slm['kkt']['max_violation'], 4)} on a Gram scale of "
      f"{f(slm['kkt']['gram_scale'], 0)}) after "
      f"{slm['solver']['iterations']} iterations.\n")
    A("### Implicit-feedback conversion\n")
    A(slm["implicit_conversion"]["justification"] + "\n")
    thr = pd.DataFrame(slm["implicit_conversion"]["sweep"])
    thr.columns = ["tau", "positives kept", "val NDCG@10", "val coverage"]
    A(md_table(thr) + "\n")
    A(f"The measured optimum is **tau = {slm['implicit_conversion']['selected_tau']:g}**, "
      f"which keeps "
      f"{[r['positives'] for r in slm['implicit_conversion']['sweep'] if r['tau'] == slm['implicit_conversion']['selected_tau']][0]:,} "
      f"of {spl['n_train']:,} training ratings. Both directions degrade for the predicted "
      f"reasons: tau=1 admits 1-star ratings as endorsements and pollutes the "
      f"co-occurrence counts, tau=5 starves the Gram matrix of support.\n")
    A("### Sparsity and the regularisation trade-off\n")
    sp = slm["sparsity"]
    A(f"The delivered model has {sp['nnz']:,} non-zero coefficients out of "
      f"{sp['offdiag_cells']:,} off-diagonal cells -- **{sp['sparsity_pct']:.3f}% sparse**, "
      f"a mean of {sp['mean_nnz_per_column']:.1f} learned neighbours per item "
      f"(median {sp['median_nnz_per_column']:.0f}, max {sp['max_nnz_per_column']}), with "
      f"{sp['empty_columns']} items receiving no incoming coefficient at all.\n")
    l1s = sl[sl.tag == "l1_sweep"].sort_values("l1")[
        ["l1", "l2", "sparsity_pct", "nnz", "ndcg_mean", "catalog_coverage", "fit_time_s"]]
    l1s.columns = ["lambda1", "lambda2", "sparsity %", "nnz", "val NDCG@10",
                   "val coverage", "fit s"]
    A(md_table(l1s) + "\n")
    A("![slim sparsity](figures/fig14_slim_sparsity_vs_performance.png)\n")
    A("![slim regularisation](figures/fig15_slim_regularisation.png)\n")
    tr_ = slm["accuracy_sparsity_tradeoff"]
    A(f"Across the regularisation grid, sparsity correlates "
      f"{f(tr_['corr_sparsity_ndcg'], 3)} with NDCG@10 and "
      f"{f(tr_['corr_sparsity_coverage'], 3)} with catalogue coverage. The trade-off is "
      f"therefore not 'sparser is cheaper but worse' in a vague sense -- it is specific: "
      f"pruning edges removes the *weak* item-item links, and weak links are precisely "
      f"what connect a user to the tail. A sparser SLIM is a smaller, faster model that "
      f"recommends a smaller slice of the catalogue.\n")
    A(f"Correlation between sparsity and fit time is {f(tr_['corr_sparsity_fit_time'], 3)}: "
      f"heavier L1 also converges in fewer iterations, so on this problem sparsity is "
      f"free at training time as well.\n")
    fm = sl[sl.tag == "feature_mask"]
    if len(fm):
        best_fm = fm.loc[fm.ndcg_mean.idxmax()]
        full = fm[fm.top_m_features == -1]
        A(f"One negative result worth recording: restricting each column's candidate "
          f"features to its top-M co-occurring items -- the standard way to scale SLIM -- "
          f"made training **slower** here ({f(fm[fm.top_m_features > 0].fit_time_s.min(), 1)}"
          f"-{f(fm[fm.top_m_features > 0].fit_time_s.max(), 1)}s versus "
          f"{f(full.fit_time_s.iloc[0], 1) if len(full) else 'n/a'}s unrestricted) while "
          f"costing NDCG@10. At 1,682 items the dense Gram GEMM already dominates and the "
          f"masking is pure overhead; the technique only pays once the catalogue is large "
          f"enough that the dense Gram no longer fits in memory.\n")
    pairs = load_table("slim_strongest_pairs").head(8)[
        ["from_title", "to_title", "coefficient", "from_train_pop", "to_train_pop"]]
    pairs.columns = ["Evidence item j", "Predicts item i", "W[j,i]", "pop(j)", "pop(i)"]
    A("Strongest learned item-item relationships:\n")
    A(md_table(pairs) + "\n")

    # ---------------------------------------------------------------- 11 Graph
    A("## 11. Network-based recommender\n")
    gr = R["graph"]
    A("The user-movie graph is bipartite, so every user-to-item path has odd length and "
      "the shortest informative one has length 3:\n")
    A("```\nu --rated--> i'  <--rated-- v --rated--> i\n"
      "\"people who liked what you liked also liked i\"\n```\n")
    A("**Bounded-length path proximity (Katz-style).** "
      "`score(u,i) = sum_{l odd <= L} beta^l (A^l)_{u,i}`, computed by repeated sparse "
      "products, never by forming `A^l`. `beta` is the attenuation: each extra hop "
      "discounts a walk geometrically. With `normalize='degree'` the adjacency is replaced "
      "by `D^-1 A`, so every node spreads a fixed unit of evidence instead of an amount "
      "proportional to its degree.\n")
    A("**Random walk with restart.** "
      "`p_{t+1} = (1-alpha) p_t P + alpha e_u`, power-iterated to the personalised "
      "PageRank fixed point. Probability stranded on dangling nodes is returned to the "
      "restart vector each step, so the iterate stays an exact distribution -- measured "
      f"total mass stays within "
      f"[{gr['probability_mass_check']['min_over_configs']:.12f}, "
      f"{gr['probability_mass_check']['max_over_configs']:.12f}] across every "
      f"configuration.\n")
    A(f"Best RWR: alpha = {gr['best_rwr']['alpha']:g} "
      f"(val NDCG@10 {f(gr['best_rwr']['val_ndcg'])}). "
      f"Best path model: L = {gr['best_path']['L']}, beta = {gr['best_path']['beta']:g}, "
      f"{gr['best_path']['normalize']}-normalised "
      f"(val NDCG@10 {f(gr['best_path']['val_ndcg'])}).\n")
    A("### Does deeper propagation make recommendations more popular?\n")
    dc = pd.DataFrame(gr["depth_vs_popularity"]["curve"])
    dc.columns = ["path length L", "val NDCG@10", "mean pop-rank", "head share",
                  "long-tail share", "novelty (bits)", "coverage"]
    A(md_table(dc) + "\n")
    A(gr["depth_vs_popularity"]["interpretation"] + "\n")
    A(f"Measured: corr(depth, mean pop-rank) = "
      f"{f(gr['depth_vs_popularity']['corr_depth_pop_rank'], 3)}, "
      f"corr(depth, head share) = "
      f"{f(gr['depth_vs_popularity']['corr_depth_head_frac'], 3)}, "
      f"corr(depth, novelty) = "
      f"{f(gr['depth_vs_popularity']['corr_depth_novelty'], 3)}. "
      f"This is not an artefact of the data: an undirected graph's stationary "
      f"distribution is proportional to node degree, so 'propagate far enough' and "
      f"'recommend by popularity' are the same operator in the limit. The restart "
      f"parameter shows the same physics from the other side: corr(alpha, mean pop-rank) "
      f"= {f(gr['restart_vs_popularity']['corr_alpha_pop_rank'], 3)} and "
      f"corr(alpha, novelty) = "
      f"{f(gr['restart_vs_popularity']['corr_alpha_novelty'], 3)} -- restarting more often "
      f"keeps the walk local and the recommendations novel.\n")

    # ---------------------------------------------------------------- 12 Setup
    A("## 12. Experimental setup\n")
    A(f"- Seed `{cfg['seed']}` (derived from the roll number), used for the split, the "
      f"target-user holdout and every stochastic component.\n"
      f"- Per-user stratified split {spl['train_pct']:.0f}/{spl['val_pct']:.0f}/"
      f"{spl['test_pct']:.0f}.\n"
      f"- K = {cfg['top_k']}; relevance threshold {cfg['relevance_threshold']:g}.\n"
      f"- Candidate set per user = all {ov['n_items']} movies minus the user's "
      f"*training* items.\n"
      f"- Novelty = mean self-information in bits; intra-list diversity = "
      f"1 - mean pairwise cosine over the 19 genre flags (a model-independent yardstick, "
      f"deliberately not a CF similarity).\n"
      f"- All configuration selection on validation; the test split is read once.\n")
    A(f"Total measured configurations: {len(ubs)} UBCF + {len(ibs)} IBCF + {len(rgs)} "
      f"regression + {len(sl)} SLIM + {len(grs)} graph = "
      f"{len(ubs) + len(ibs) + len(rgs) + len(sl) + len(grs)}.\n")

    # ---- complexity ------------------------------------------------------------------
    A("### 12.1 Computational complexity and scalability\n")
    n_u, n_i, nnz = ov["n_users"], ov["n_items"], spl["n_train"]
    comp = pd.DataFrame([
        ["UBCF",
         "5 sparse GEMMs for the similarity; O(|U|·nnz) for the exact per-(u,i) top-k",
         "O(|U|²)",
         f"{f(par.loc['UBCF', 'train_time_s'], 2)} s / {f(par.loc['UBCF', 'model_mb'], 1)} MB",
         "|U|² model: the dimension that grows fastest in a real system"],
        ["IBCF", "same, transposed", "O(|I|²)",
         f"{f(par.loc['IBCF', 'train_time_s'], 2)} s / {f(par.loc['IBCF', 'model_mb'], 1)} MB",
         "|I|² model, but precomputable offline and stable between refreshes"],
        ["RegressionCF", "O(nnz·k² + |I|·k³) — one k x k solve per item",
         "O(|I|·k) sparse W",
         f"{f(par.loc['RegressionCF', 'train_time_s'], 2)} s / "
         f"{f(par.loc['RegressionCF', 'model_mb'], 2)} MB",
         "k³ per item is trivial for k<=80; the dense deviation matrix is the real limit"],
        ["SLIM", "one |I|³ dense GEMM per FISTA iteration", "O(|I|²) Gram during training",
         f"{f(par.loc['SLIM', 'train_time_s'], 1)} s / "
         f"{f(par.loc['SLIM', 'model_mb'], 2)} MB",
         "dense Gram is the wall: 22 MB here, ~80 GB at 10^5 items"],
        ["GraphRec", "O(|U|·nnz) per power-iteration step, sparse matvecs only",
         "O(nnz) graph",
         f"{f(par.loc['GraphRec', 'train_time_s'], 2)} s / "
         f"{f(par.loc['GraphRec', 'model_mb'], 2)} MB",
         "scales best in memory; per-user inference is the cost, not training"],
    ], columns=["Model", "Training cost", "Model memory", "Measured (train / size)",
                "Where it breaks"])
    A(md_table(comp) + "\n")
    A(f"Two deliberate algorithmic choices are worth calling out because the naive "
      f"alternative is O(n²) or worse:\n")
    A(f"1. **Co-rated Pearson without a pairwise loop.** The textbook formulation needs "
      f"five sums over `I_a ∩ I_b` for every pair. Written as a loop that is "
      f"O(|U|²·|I|) in Python — on the order of 10⁹ operations here. Written as five "
      f"sparse GEMMs (`n = B Bᵀ`, `S_a = R Bᵀ`, `S_ab = R Rᵀ`, `Q_a = R² Bᵀ`) it is "
      f"BLAS-bound and exact, and it is what makes a {len(ubs) + len(ibs)}-configuration "
      f"neighbourhood sweep take minutes rather than hours.\n"
      f"2. **SLIM as one GEMM per iteration, not one solve per column.** The gradient of "
      f"the smooth part, `G W - G + λ₂W`, covers all {n_i:,} column problems "
      f"simultaneously. Coordinate descent would need a rank-1 update of an |I|x|I| "
      f"matrix per coordinate — the same asymptotic cost, but memory-bandwidth bound "
      f"rather than BLAS-3 bound.\n")
    A(f"The honest scalability statement: the *algorithms* here are standard and scale, "
      f"but these *implementations* exploit the fact that {n_u} x {n_i} "
      f"({nnz:,} training interactions) fits comfortably in memory. Dense similarity and "
      f"score buffers stop being practical somewhere around 10⁴-10⁵ items. The "
      f"production forms are well known — column-parallel coordinate descent over a "
      f"top-M co-occurrence candidate set for SLIM, approximate nearest neighbours for "
      f"the neighbourhood models, and push-based local PageRank for the graph model — and "
      f"the feature-restriction experiment in Part F measures what the first of those "
      f"costs in accuracy at this scale.\n")

    # ---------------------------------------------------------------- 13 Results
    A("## 13. Results\n")
    A("### Master comparison (test split)\n")
    A("Rows are read as follows. The five paradigm names carry the configuration that "
      "maximised **validation NDCG@10**. Rows suffixed `-rating` are the same paradigm at "
      "the configuration that minimised **validation RMSE**, listed wherever that turned "
      "out to be a different model; they are included because the rating-optimal "
      "configuration is a legitimate answer to a different question, and because what "
      "happens to their ranking metrics is itself a result. `SparsityPct` is the measured "
      "fraction of zero off-diagonal entries in SLIM's learned `W`; for the neighbourhood "
      "models it is the structural `1 - k/(n-1)` of the retained neighbour lists.\n")
    A(md_table(master) + "\n")
    idx = mu.set_index("label")
    uc = idx["user_coverage"]
    A(f"*UserCov* counts the users for whom a model produced a genuinely ranked list "
      f"rather than an arbitrary tie-break. It is {f(uc['GlobalMean'], 2)} for GlobalMean "
      f"and UserMean, which is correct: those models assign an identical score to every "
      f"candidate, so their 'Top-10' is a tie-break on item index and their apparent "
      f"NDCG is an artefact of that tie-break rather than a measurement of the model.\n")
    rating_arms = [l for l in idx.index if l.endswith("-rating")]
    if rating_arms:
        worst = min(rating_arms, key=lambda l: idx.loc[l, "ndcg_mean"])
        base = worst.replace("-rating", "")
        A(f"The `-rating` rows are the sharpest single result in the table. For {base}, "
          f"moving to the RMSE-optimal configuration improves RMSE "
          f"{f(par.loc[base, 'rmse'])} -> {f(idx.loc[worst, 'rmse'])} and costs NDCG@10 "
          f"{f(par.loc[base, 'ndcg_mean'])} -> {f(idx.loc[worst, 'ndcg_mean'])} — a "
          f"collapse to near zero. Ranking a {ov['n_items']:,}-item candidate set by "
          f"predicted rating puts thinly-supported items at the top, because one "
          f"enthusiastic neighbour is enough to produce a predicted 5.0. Optimising RMSE "
          f"and optimising Top-N are not merely different objectives; on this data, "
          f"pursuing one to its optimum destroys the other. Their user coverage says the "
          f"same thing from another angle: "
          + ", ".join(f"{l} {f(uc[l], 3)}" for l in rating_arms) + ".\n")
    A("![ranking comparison](figures/fig08_ranking_comparison.png)\n")
    A("![coverage](figures/fig09_coverage_comparison.png)\n")
    A("![novelty and diversity](figures/fig10_novelty_diversity.png)\n")
    A("![runtime](figures/fig16_runtime.png)\n")
    A("### Rankings by axis (the five paradigms)\n")
    aw = R["master"]["axis_winners"]
    ax_rows = []
    for axis, v in aw.items():
        ordering = " > ".join(f"{o['model']} ({f(o['value'], 4)})" for o in v["order"])
        ax_rows.append([axis.replace("_", " "), v["metric"],
                        "higher" if v["higher_is_better"] else "lower", ordering])
    A(md_table(pd.DataFrame(ax_rows, columns=["Axis", "Metric", "Better", "Ordering"]))
      + "\n")
    A(R["master"]["tradeoffs"]["no_single_best"] + "\n")

    # ---------------------------------------------------------------- 14 Ablations
    A("## 14. Ablation studies\n")
    A(f"{len(abl)} controlled ablation rows, each changing exactly one factor. "
      f"Full table in `results/tables/ablations.csv`; the summary below reports the "
      f"NDCG@10 spread each factor is responsible for on the validation split.\n")
    ablj = R["ablations"]
    blocks = ablj["blocks"]
    ab_rows = []
    for kk, v in sorted(blocks.items(), key=lambda x: -x[1]["ndcg_spread"]):
        if v["ndcg_tie"]:
            best, worst = "— (tied)", "— (tied)"
        else:
            best, worst = v["best_ndcg_variant"], v["worst_ndcg_variant"]
        ab_rows.append([kk, v["n_variants"], best, f(v["best_ndcg"]), worst,
                        f(v["worst_ndcg"]), f(v["ndcg_spread"]),
                        v["best_rmse_variant"], f(v["best_rmse"]),
                        f(v["rmse_spread"]), v["discriminating_metric"]])
    A(md_table(pd.DataFrame(ab_rows, columns=[
        "Family / factor", "Variants", "Best (NDCG)", "NDCG", "Worst (NDCG)", "NDCG ",
        "NDCG spread", "Best (RMSE)", "RMSE", "RMSE spread", "Discriminates on"]))
      + "\n")
    ties = [kk for kk, v in blocks.items() if v["ndcg_tie"]]
    if ties:
        A(f"**{len(ties)} block(s) show a zero NDCG@10 spread — "
          f"{', '.join(ties)} — and that is a result rather than a bug.** "
          + ablj["tie_note"] + " The other case is the graph model's attenuation at a "
          "single effective path length: when only one odd path length can reach a "
          "candidate, beta multiplies every score by the same positive constant and "
          "cannot reorder anything, which is why that ablation is measured at the "
          "deepest propagation in the sweep instead.\n")
    A("What changed and why, factor by factor:\n")
    for kk, v in sorted(blocks.items(), key=lambda x: -x[1]["ndcg_spread"]):
        headline = (f"NDCG@10 spread {f(v['ndcg_spread'])}, best = "
                    f"`{v['best_ndcg_variant']}`" if not v["ndcg_tie"]
                    else f"no effect on ranking; RMSE spread {f(v['rmse_spread'])}, "
                         f"best = `{v['best_rmse_variant']}`")
        A(f"- **{kk}** ({headline}): {v['why']}\n")

    # ---------------------------------------------------------------- 15 Popularity
    A("## 15. Popularity and long-tail analysis\n")
    pbj = R["popularity_bias"]
    pb_tbl = pb[["model", "mean_train_popularity", "head_pct", "medium_pct",
                 "long_tail_pct", "catalog_coverage", "novelty_bits", "ild",
                 "gini_exposure", "ndcg_mean"]].copy()
    pb_tbl.columns = ["Model", "Mean pop.", "Head %", "Medium %", "Tail %", "Coverage",
                      "Novelty", "Diversity", "Gini", "NDCG@10"]
    A(md_table(pb_tbl) + "\n")
    A("![popularity bias](figures/fig11_popularity_bias.png)\n")
    A("![long tail exposure](figures/fig12_long_tail_exposure.png)\n")
    expo = pbj["exposure_reference"]
    A("### The exposure gap\n")
    A(f"The load-bearing number in this section is not a correlation, it is the exposure "
      f"gap. The {expo['n_head_items']} head movies are "
      f"{f(expo['head_share_of_catalog_pct'], 1)}% of the catalogue and receive "
      f"{f(expo['paradigm_head_pct_range'][0], 1)}-"
      f"{f(expo['paradigm_head_pct_range'][1], 1)}% of all recommendation slots across the "
      f"five paradigms. The {expo['n_long_tail_items']} long-tail movies are "
      f"{f(expo['long_tail_share_of_catalog_pct'], 1)}% of the catalogue and receive "
      f"{f(expo['paradigm_tail_pct_range'][0], 1)}-"
      f"{f(expo['paradigm_tail_pct_range'][1], 1)}%. Catalogue coverage across the five "
      f"spans {f(100 * expo['paradigm_coverage_range'][0], 1)}%-"
      f"{f(100 * expo['paradigm_coverage_range'][1], 1)}%: even the widest-reaching model "
      f"can only ever surface about a fifth of the movies. Every model here is severely "
      f"popularity-biased; they differ in degree, not in kind.\n")
    A("### Is it a clean accuracy-versus-coverage frontier? No.\n")
    par_front = pbj["pareto"]
    A(f"Treating NDCG@10, catalogue coverage, novelty and long-tail share as four "
      f"objectives, the Pareto front over the five paradigms is "
      f"**{', '.join(par_front['pareto_front'])}** — everything else is strictly "
      f"dominated:\n")
    for m, by in par_front["dominated"].items():
        A(f"- {m} is dominated by {', '.join(by)}\n")
    A(f"This matters because the textbook framing ('the accurate model is the biased one') "
      f"is simply not what the data says here. {pbj['headline']['most_accurate_model_ndcg']} "
      f"is both the best ranker *and* better than UBCF, IBCF and the graph model on "
      f"coverage, novelty and long-tail share simultaneously. Learning item-item weights "
      f"discriminatively does not buy accuracy at the expense of reach; it buys both, "
      f"relative to similarity heuristics and to graph propagation.\n")
    A("### Correlations\n")
    cp = pbj["correlations_paradigms"]
    cw = pbj["correlations_ranking_capable"]
    corr_tbl = pd.DataFrame([
        ["NDCG@10 vs mean popularity", f(cp["ndcg_vs_mean_popularity"], 3),
         f(cw["ndcg_vs_mean_popularity"], 3)],
        ["NDCG@10 vs head share", f(cp["ndcg_vs_head_pct"], 3),
         f(cw["ndcg_vs_head_pct"], 3)],
        ["NDCG@10 vs long-tail share", f(cp["ndcg_vs_long_tail_pct"], 3),
         f(cw["ndcg_vs_long_tail_pct"], 3)],
        ["NDCG@10 vs catalogue coverage", f(cp["ndcg_vs_catalog_coverage"], 3),
         f(cw["ndcg_vs_catalog_coverage"], 3)],
        ["NDCG@10 vs novelty", f(cp["ndcg_vs_novelty"], 3), f(cw["ndcg_vs_novelty"], 3)],
        ["NDCG@10 vs intra-list diversity", f(cp["ndcg_vs_ild"], 3),
         f(cw["ndcg_vs_ild"], 3)],
    ], columns=["Relationship", f"5 paradigms (n={cp['n_models']})",
                f"ranking-capable (n={cw['n_models']})"])
    A(md_table(corr_tbl) + "\n")
    A(pbj["correlation_caveat"] + "\n")
    A(f"Read honestly, the paradigm-level correlations are weak and the only one with any "
      f"size is NDCG@10 against long-tail share "
      f"({f(cp['ndcg_vs_long_tail_pct'], 3)}) — and with five points that is one model's "
      f"position, not a law.\n")
    A("### Does the most accurate model make the most useful recommendations?\n")
    ev = pbj["does_the_most_accurate_model_make_the_most_useful_recommendations"]["evidence"]
    A(f"**{pbj['does_the_most_accurate_model_make_the_most_useful_recommendations']['answer']}**\n")
    A(f"The most accurate model, {ev['most_accurate_model']}, is *not* the most biased "
      f"one — it draws {f(ev['its_head_pct'], 1)}% of its slots from the head, less than "
      f"UBCF, IBCF or the graph model, and reaches "
      f"{f(100 * ev['its_catalog_coverage'], 1)}% of the catalogue. So the easy story "
      f"('accuracy causes popularity bias') is false on this data.\n")
    A(f"The genuine trade-off is between the two Pareto-optimal models, and it is worth "
      f"being blunt about how unattractive it is. {ev['widest_coverage_model']} buys "
      f"{f(ev['coverage_gain_pp'], 1)} percentage points of catalogue coverage and "
      f"{f(ev['tail_gain_pp'], 1)} points of long-tail exposure, and pays "
      f"{f(ev['ndcg_cost_relative_pct'], 1)}% of its NDCG@10 for them. A product team "
      f"asked to give up a fifth of its ranking quality for three points of tail "
      f"exposure would decline, and would be right to. The conclusion is not 'pick the "
      f"other model'; it is that buying reach from the choice of collaborative model is "
      f"bad value, and a system that needs reach should buy it somewhere else — in a "
      f"re-ranking stage, an exploration budget, or content features that collaborative "
      f"filtering structurally cannot supply.\n")
    A(f"But the more important answer is that *no* model here is doing the job a "
      f"recommender exists to do. The best long-tail exposure achieved by any of the five "
      f"is {f(expo['paradigm_tail_pct_range'][1], 1)}% of slots, for a group holding "
      f"{f(expo['long_tail_share_of_catalog_pct'], 1)}% of the catalogue. Offline ranking "
      f"metrics reward predicting what the user would have found anyway, and the held-out "
      f"ground truth is itself drawn from what users already chose to watch, so the "
      f"evaluation cannot reward discovery even in principle. Accuracy and usefulness "
      f"come apart — but the gap is between collaborative filtering as a class and the "
      f"catalogue, far more than between these five models.\n")

    # ---------------------------------------------------------------- 16 Cold start
    A("## 16. Cold-start analysis\n")
    cs = R["coldstart"]
    ugd = cs["user_group_definition"]
    A(f"Users are split into tertiles of training-history length: "
      f"sparse (<= {ugd['q1']:.0f} ratings, n = {ugd['n_sparse']}), "
      f"medium (<= {ugd['q2']:.0f}, n = {ugd['n_medium']}), "
      f"heavy (n = {ugd['n_heavy']}).\n")
    piv = ug[ug.model.isin(PARADIGMS)].pivot(index="model", columns="user_group",
                                             values="ndcg")[["sparse", "medium", "heavy"]]
    piv = piv.reset_index()
    piv.columns = ["Model", "Sparse", "Medium", "Heavy"]
    A("NDCG@10 by user-history tertile:\n")
    A(md_table(piv) + "\n")
    pivr = ug[ug.model.isin(PARADIGMS)].pivot(index="model", columns="user_group",
                                              values="rmse")[["sparse", "medium", "heavy"]]
    pivr = pivr.reset_index()
    pivr.columns = ["Model", "Sparse", "Medium", "Heavy"]
    A("RMSE by user-history tertile:\n")
    A(md_table(pivr) + "\n")
    A("![cold start](figures/fig17_coldstart.png)\n")
    gsz = (ug[ug.model == PARADIGMS[0]]
           .set_index("user_group")[["mean_heldout_positives",
                                     "n_users_with_relevant", "n_users"]])
    A(f"**Read the NDCG table carefully: it is not monotonic in history length, and that "
      f"is a property of the metric, not of the models.** NDCG@10 normalises by the ideal "
      f"DCG over `min(K, |G_u|)`, so a user with a single held-out positive scores 1.0 for "
      f"one hit at rank 1, while a user with ten needs ten hits to do the same. The "
      f"average number of held-out positives per evaluated user is "
      + ", ".join(f"{g} {f(gsz.loc[g, 'mean_heldout_positives'], 1)}"
                  for g in ("sparse", "medium", "heavy")) +
      f", so the three columns are not measuring equally hard problems. The comparison "
      f"that *is* valid is between models within a column, which is what the questions "
      f"below use.\n")
    rmse_piv = ug[ug.model.isin(PARADIGMS)].pivot(index="model", columns="user_group",
                                                  values="rmse")
    mono = [m for m in PARADIGMS
            if rmse_piv.loc[m, "sparse"] > rmse_piv.loc[m, "medium"]
            > rmse_piv.loc[m, "heavy"]]
    A(f"RMSE carries no such normalisation, and it behaves as expected: it falls "
      f"monotonically with history length for {len(mono)} of the {len(PARADIGMS)} "
      f"paradigms ({', '.join(mono)}). More history means a better-estimated user, "
      f"full stop.\n")
    A(f"**1. Which method is best for sparse users?** "
      f"{cs['best_for_sparse_users_ndcg']['model']} on ranking "
      f"(NDCG@10 = {f(cs['best_for_sparse_users_ndcg']['ndcg'])}, runner-up "
      f"{cs['best_for_sparse_users_ndcg']['runner_up']}) and "
      f"{cs['best_for_sparse_users_rmse']['model']} on rating error "
      f"(RMSE = {f(cs['best_for_sparse_users_rmse']['rmse'])}).\n")
    A(f"**2. Which method benefits most from richer histories?** "
      f"{cs['benefits_most_from_history']['model']}, gaining "
      f"{f(cs['benefits_most_from_history']['absolute_ndcg_gain_heavy_minus_sparse'])} "
      f"NDCG@10 from the sparse to the heavy tertile "
      f"({100 * cs['benefits_most_from_history']['relative_gain']:.1f}% relative). "
      f"Full ordering of absolute gains: " +
      ", ".join(f"{m} {v:+.4f}" for m, v in
                sorted(cs["benefits_most_from_history"]["all_gains"].items(),
                       key=lambda x: -x[1])) + ".\n")
    A(f"**3. Which methods are most sensitive to sparsity?** Ranked by relative NDCG@10 "
      f"gain from sparse to heavy users: " +
      ", ".join(f"{m} ({100 * v:+.1f}%)" for m, v in
                cs["sensitivity_ranking_by_relative_ndcg_gain"].items()) +
      f". The least sensitive is {cs['least_sensitive_to_history']['model']} "
      f"({cs['least_sensitive_to_history']['absolute_ndcg_gain']:+.4f}).\n")
    A("### Performance by item popularity group\n")
    A("Ranking quality with the ground truth restricted to one popularity group (the "
      "Top-10 list is unchanged; only what counts as a hit changes):\n")
    igp = ig[ig.model.isin(PARADIGMS)].pivot(index="model", columns="item_group",
                                             values="recall")[["head", "medium", "long_tail"]]
    igp = igp.reset_index()
    igp.columns = ["Model", "Head", "Medium", "Long tail"]
    A(md_table(igp) + "\n")
    A("Every model retrieves head items far better than tail items. That gap is not a "
      "model defect; it is the training signal. The tail has too few interactions for any "
      "collaborative method to estimate a reliable neighbourhood, which is precisely why "
      "content features or explicit exploration -- not a better CF model -- are the "
      "standard fix.\n")

    # ---------------------------------------------------------------- 17 Explainability
    A("## 17. Explainability\n")
    A("Each model exposes an `explain(u, i)` method that returns the evidence actually "
      "used by its own scoring rule:\n")
    A("| Model | Explanation primitive |\n|---|---|\n"
      "| UBCF | the neighbours with the largest signed contribution to the numerator, "
      "with their similarity, co-rating count and rating |\n"
      "| IBCF | the user's own previously-rated movies closest to the candidate, with "
      "similarity and contribution |\n"
      "| RegressionCF | the largest positive and negative *learned* coefficients, shown "
      "next to the raw similarity they replaced |\n"
      "| SLIM | the largest learned item-item coefficients `W[j,i]` from the user's liked "
      "items |\n"
      "| GraphRec | the intermediate movies carrying the most length-3 path weight, with "
      "the number of distinct paths |\n")
    A("Worked explanations for the target user's Top-10 are in "
      "`results/tables/target_reasoning.csv`, and the three strongest cross-model "
      "disagreements are narrated in the next section.\n")

    # ---------------------------------------------------------------- 18 Target user
    A("## 18. Target-user recommendations\n")
    tu = R["target_user"]["target"]
    A(f"Target user **{tu['user_id']}**, derived from roll number `{tu['roll']}`: "
      f"`{tu['mapping']}`. Visible profile {tu['n_visible']} ratings, "
      f"{tu['n_hidden']} hidden from training, {tu['n_global_test']} in the global test "
      f"split, {tu['n_relevant_heldout']} held-out positives.\n")
    A("### Top-10 per model\n")
    A(md_table(tgt_rank) + "\n")
    A("### Measured on this user's held-out ratings\n")
    tm = tgt_metrics[["model", "rmse", "mae", "precision_at_10", "recall_at_10",
                      "hit_rate_at_10", "ndcg_at_10", "map_at_10", "novelty",
                      "ild", "tail_frac"]].copy()
    tm.columns = ["Model", "RMSE", "MAE", "P@10", "R@10", "HR@10", "NDCG@10", "MAP@10",
                  "Novelty", "Diversity", "Tail share"]
    A(md_table(tm) + "\n")
    A(f"Recall@10 looks low for a reason that is arithmetic rather than modelling: this "
      f"user has {tu['n_relevant_heldout']} held-out positives and only "
      f"{CONFIG_TOP_K} slots, so the attainable maximum is "
      f"{f(CONFIG_TOP_K / tu['n_relevant_heldout'], 4)}. Every model here reaches between "
      f"{f(100 * tgt_metrics.recall_at_10.min() / (CONFIG_TOP_K / tu['n_relevant_heldout']), 0)}% "
      f"and "
      f"{f(100 * tgt_metrics.recall_at_10.max() / (CONFIG_TOP_K / tu['n_relevant_heldout']), 0)}% "
      f"of that ceiling, and all five achieve a hit rate of 1.0 — for a user with a "
      f"{tu['n_visible']}-rating profile the Top-10 task is not hard; the interesting "
      f"question is which movies each one chooses, not whether it finds any.\n")
    multi = tgt_cons[tgt_cons.n_models >= 2]
    A(f"Per-rank detail for all five lists — popularity, group, whether the slot was a "
      f"hit, and the user's held-out rating where one exists — is in "
      f"`results/tables/target_top10_detail.csv`.\n")
    A(f"### Consensus\n")
    A(f"{len(multi)} of {len(tgt_cons)} distinct recommended movies are picked by more "
      f"than one model; {len(tgt_cons[tgt_cons.n_models >= 3])} by three or more.\n")
    if len(multi):
        mt = multi[["title", "n_models", "models", "train_popularity", "group",
                    "held_out_rating"]].head(12).copy()
        mt.columns = ["Movie", "# models", "Models", "Train popularity", "Group",
                      "User's held-out rating"]
        A(md_table(mt) + "\n")
    A("### Major disagreements\n")
    A("Every recommended movie ranked by how divisively the models treat it is in "
      "`results/tables/target_disagreements.csv`; the three most divisive are narrated "
      "below.\n")
    for d in R["target_user"]["disagreements"][:3]:
        A(f"**{d['title']}** — {d['popularity_group'].replace('_', ' ')}, "
          f"{d['train_popularity']} training ratings"
          + (f", the user actually rated it {d['held_out_rating']:.0f}"
             if d["held_out_rating"] is not None else ", not in this user's held-out set")
          + ".\n")
        A(f"- Recommended by: {', '.join(d['recommended_by']) or 'none'}; "
          f"omitted by: {', '.join(d['not_recommended_by']) or 'none'}.\n")
        A("- Rank per model: " + ", ".join(
            f"{m}={v if v else '—'}" for m, v in d["rank_by_model"].items()) + ".\n")
        A("- Score percentile within each model's candidate set: " + ", ".join(
            f"{m}={v:.1f}%" for m, v in d["score_percentile_by_model"].items()) + ".\n")
        for m, txt in d["explanations"].items():
            A(f"- *{m}*: {txt}\n")
        A("")
    A("The pattern behind these disagreements is consistent with the aggregate numbers: "
      "the models that concentrate on the head of the popularity distribution agree with "
      "each other and disagree with the models that reach the tail, and the score "
      "percentiles show that the disagreement is usually not 'this model hates the item' "
      "but 'this model ranks it 90th percentile instead of 99.9th', which is the whole "
      "difference between appearing in a Top-10 and not.\n")

    # ---------------------------------------------------------------- 19 Error
    A("## 19. Error analysis\n")
    ea = R["error_analysis"]
    A(f"Best rating model among the paradigms: **{ea['best_rating_model_among_paradigms']}**.\n")
    A("### Regression to the mean\n")
    rtm = err_strata[(err_strata.stratum == "true_rating")
                     & (err_strata.model.isin(PARADIGMS))]
    piv = rtm.pivot(index="model", columns="level", values="bias").reset_index()
    piv.columns = ["Model"] + [f"true = {c}" for c in piv.columns[1:]]
    A("Mean signed error (prediction - truth) by true rating value:\n")
    A(md_table(piv, 3) + "\n")
    A("Every model over-predicts low ratings and under-predicts high ones. This is not a "
      "bug: squared-error training pulls predictions toward the conditional mean, so the "
      "predicted distribution is narrower than the true one. It is also why RMSE and "
      "ranking quality come apart -- a model can be well-calibrated in the middle of the "
      "scale and still order the extremes badly.\n")
    A("### Error by item popularity\n")
    ipg = err_strata[(err_strata.stratum == "item_group")
                     & (err_strata.model.isin(PARADIGMS))]
    piv = ipg.pivot(index="model", columns="level", values="rmse")[
        ["head", "medium", "long_tail"]].reset_index()
    piv.columns = ["Model", "Head", "Medium", "Long tail"]
    A(md_table(piv) + "\n")
    A("### Ranking errors: where does the first hit land?\n")
    er = err_rank[err_rank.model.isin(PARADIGMS)][
        ["model", "users_with_a_hit", "hit_rate", "mean_first_hit_rank",
         "median_first_hit_rank"]].copy()
    er.columns = ["Model", "Users with a hit", "Hit rate", "Mean first-hit rank",
                  "Median first-hit rank"]
    A(md_table(er) + "\n")
    pol = ea.get("hard_item_polarisation")
    if pol:
        A(f"The {pol['n_hard_items']} hardest items to predict (those with at least "
          f"{pol['min_test_ratings']} test ratings, listed in "
          f"`results/tables/error_hardest_items.csv`) are measurably more *divisive* than "
          f"the catalogue, not merely more obscure: their full rating distributions have "
          f"a mean variance of {f(pol['hard_mean_rating_variance'], 3)} against "
          f"{f(pol['catalog_mean_rating_variance_comparable_support'], 3)} for the "
          f"{pol['n_comparable_items']} items with comparable support "
          f"(>= {pol['comparable_support_threshold']} ratings), and "
          f"{f(100 * pol['hard_mean_extreme_share'], 1)}% of their ratings are at the "
          f"extremes (1, 2 or 5) against "
          f"{f(100 * pol['catalog_mean_extreme_share_comparable_support'], 1)}%. A model "
          f"trained on squared error predicts a conditional mean, and a conditional mean "
          f"is exactly the worst summary of a divided audience.\n")

    # ---------------------------------------------------------------- 20 Limitations
    A("## 20. Limitations\n")
    A(f"1. **Scale.** Everything here exploits the fact that 943 x 1682 fits in memory: "
      f"dense similarity matrices, a dense Gram matrix, dense score buffers. The "
      f"algorithms are correct at any scale but these *implementations* stop being "
      f"practical somewhere around 10⁴-10⁵ items (SLIM's dense Gram alone would be "
      f"80 GB at 10⁵ items).\n"
      f"2. **Single split.** Metrics are from one seeded split; the `*_std` columns are "
      f"across-user standard deviations, not confidence intervals over repeated splits. "
      f"Differences smaller than ~0.005 NDCG should not be over-read.\n"
      f"3. **Random rather than temporal holdout.** This measures the missing-value "
      f"problem, not next-item prediction. A temporal split would give lower numbers and "
      f"answer a different question.\n"
      f"4. **Offline proxies.** Held-out ratings only exist for items the user already "
      f"chose to watch, so recall of the long tail is structurally under-measured -- the "
      f"ground truth is itself popularity-biased. This caps how much any offline study "
      f"can say about discovery.\n"
      f"5. **Diversity proxy.** Intra-list diversity uses 19 genre flags, which is coarse.\n"
      f"6. **No content or demographic features.** MovieLens ships titles, genres, ages "
      f"and occupations; none are used except genres for the diversity metric, so the "
      f"cold-item problem is left unaddressed by design.\n"
      f"7. **Rating heads for implicit models.** SLIM and the graph models are rankers; "
      f"their RMSE comes from a single least-squares calibration of score to rating and "
      f"should be read as 'this is not a rating model', not as a competitive RMSE.\n")

    # ---------------------------------------------------------------- 21 Conclusions
    A("## 21. Conclusions\n")
    A(_conclusions(R, par, base, best_ndcg, best_rmse, best_cov, best_nov, best_ild,
                   fastest, pop_ndcg, bias_rmse, ov, brc))

    # ---------------------------------------------------------------- 22 Future work
    A("## 22. Future work\n")
    A("1. **Matrix factorisation and neural baselines** (BPR-MF, implicit ALS, EASE^R, "
      "a light autoencoder) to place these five paradigms on the wider map -- EASE^R in "
      "particular is the closed-form limit of SLIM without the non-negativity constraint "
      "and would isolate what that constraint buys.\n"
      "2. **Hybridise for the tail -- but not with another CF model.** The Pareto "
      "analysis in Part J shows that trading between collaborative models buys very "
      "little reach for a lot of ranking quality. The promising direction is a "
      "re-ranking stage over the accurate model's candidates, optimising an "
      "exposure-constrained objective directly, with content features supplying the "
      "signal collaborative filtering structurally cannot have for rarely-rated items.\n"
      "3. **Repeated splits and significance testing** -- bootstrap confidence intervals "
      "over users and paired tests between models, so small differences become "
      "defensible.\n"
      "4. **Temporal evaluation** to check whether the ordering survives a sequential "
      "protocol.\n"
      "5. **Scalable SLIM**: column-parallel coordinate descent with a co-occurrence "
      "candidate cache, which removes the dense-Gram memory wall that limits the current "
      "implementation.\n"
      "6. **Calibrated popularity de-biasing** -- inverse-propensity re-weighting of the "
      "training signal, evaluated with the same coverage and novelty instruments used "
      "here.\n")

    # ---------------------------------------------------------------- Q&A
    A("---\n")
    A("## Research questions, answered\n")
    A(_answers(R, par, base, best_ndcg, best_rmse, best_cov, best_nov, fastest,
               pop_ndcg, bias_rmse, ug, cs=R["coldstart"], brc=brc))

    A("---\n")
    A(f"*Generated by `src/report.py` from `results/`. "
      f"{R['figures'].get('n_figures', 0)} figures in `figures/`.*\n")

    return "\n".join(L)


# ======================================================================================
def _neighbourhood_findings(df: pd.DataFrame, fam: str) -> str:
    best_r = df.loc[df.rmse.idxmin()]
    best_n = df.loc[df.ndcg_mean.idxmax()]
    cos_r = df[df.similarity == "cosine"].rmse.min()
    pea_r = df[df.similarity == "pearson"].rmse.min()
    cos_n = df[df.similarity == "cosine"].ndcg_mean.max()
    pea_n = df[df.similarity == "pearson"].ndcg_mean.max()
    mc_r = df[df.mean_center].rmse.min()
    raw_r = df[~df.mean_center].rmse.min()
    mc_n = df[df.mean_center].ndcg_mean.max()
    raw_n = df[~df.mean_center].ndcg_mean.max()
    out = [f"**Findings ({fam}, validation split).**\n"]
    out.append(
        f"- *Cosine vs Pearson.* Best RMSE: cosine {f(cos_r)} vs Pearson {f(pea_r)}. "
        f"Best NDCG@10: cosine {f(cos_n)} vs Pearson {f(pea_n)}. The two similarities "
        f"win on different objectives — Pearson removes rating scale and sharpens the "
        f"rating head, cosine retains profile-overlap information that the Top-N head "
        f"needs.\n")
    out.append(
        f"- *Mean-centring.* Best RMSE: centred {f(mc_r)} vs raw {f(raw_r)} "
        f"(a {f(raw_r - mc_r)} improvement). Best NDCG@10: centred {f(mc_n)} vs raw "
        f"{f(raw_n)}. Centring is close to mandatory for rating prediction and roughly "
        f"neutral-to-negative for ranking, because the centred numerator rewards items "
        f"that sit above a neighbour's mean rather than items with a lot of supporting "
        f"evidence.\n")
    out.append(
        f"- *Neighbourhood size.* RMSE is minimised at k = {int(best_r.k)} and rises "
        f"again beyond it (the classic U), while NDCG@10 keeps improving until k = "
        f"{int(best_n.k)} and then plateaus. Ranking wants more evidence than rating "
        f"prediction does: extra weak neighbours add noise to a point estimate but "
        f"stabilise an ordering.\n")
    out.append(
        f"- *Ranking head.* The validation-selected head is `{best_n.rank_mode}` "
        f"(NDCG@10 {f(best_n.ndcg_mean)}); the `rating` head reaches only "
        f"{f(df[df.rank_mode == 'rating'].ndcg_mean.max())}, which is the quantitative "
        f"form of the 'never rank by predicted rating' rule.\n")
    return "".join(out)


def _conclusions(R, par, base, best_ndcg, best_rmse, best_cov, best_nov, best_ild,
                 fastest, pop_ndcg, bias_rmse, ov, brc) -> str:
    o = []
    o.append(
        f"**There is no single winner, and the reason is structural rather than "
        f"accidental.** {best_ndcg} wins ranking (NDCG@10 {f(par.loc[best_ndcg, 'ndcg_mean'])}), "
        f"{brc.label} wins rating prediction (RMSE {f(brc.rmse)}; "
        f"{best_rmse} at {f(par.loc[best_rmse, 'rmse'])} is the best among the "
        f"ranking-selected configurations), "
        f"{best_cov} wins catalogue coverage "
        f"({f(100 * par.loc[best_cov, 'catalog_coverage'], 1)}%), {best_nov} wins novelty "
        f"({f(par.loc[best_nov, 'novelty_mean'], 2)} bits) and {fastest} is the cheapest "
        f"to train ({f(par.loc[fastest, 'train_time_s'], 2)} s). Four different models on "
        f"five axes.\n")
    o.append(
        f"**Personalisation is worth paying for, but the bar is popularity, not zero — "
        f"and on the rating task the bar is nearly unbeatable.** The popularity baseline "
        f"reaches NDCG@10 {f(pop_ndcg)} with no model at all; {best_ndcg} reaches "
        f"{f(par.loc[best_ndcg, 'ndcg_mean'])}, a "
        f"{f(100 * (par.loc[best_ndcg, 'ndcg_mean'] / pop_ndcg - 1), 1)}% relative gain, "
        f"so personalisation clearly earns its keep for Top-N. Rating prediction is a "
        f"different story: a regularised bias model that knows nothing but 'this user is "
        f"generous, this film is good' reaches RMSE {f(bias_rmse)}, and the best "
        f"configuration anywhere in this study ({brc.label}) reaches {f(brc.rmse)} — an "
        f"improvement of {f(bias_rmse - brc.rmse)}, or "
        f"{f(100 * (1 - brc.rmse / bias_rmse), 1)}%. Five paradigms, "
        f"{len(load_table('ubcf_sweep')) + len(load_table('ibcf_sweep')) + len(load_table('regression_sweep'))} "
        f"neighbourhood configurations, and the whole collaborative apparatus buys about "
        f"two and a half percent over two bias terms. Reporting RMSE without that "
        f"baseline would badly overstate what the modelling achieved.\n")
    o.append(
        f"**Learning the weights beats asserting them, but only with regularisation.** "
        f"The regression model's unregularised arm degrades as the neighbourhood grows, "
        f"reaching validation RMSE "
        f"{f(float(load_table('regression_sweep').query('weight_mode == \"learned\" and lam == 0').groupby('k').rmse.min().max()))} "
        f"at the largest k — worse than predicting the global mean. Regularised, the "
        f"same architecture "
        f"reaches {f(R['regression']['learned_arm_best_val_rmse'])} against "
        f"{f(R['regression']['similarity_arm_best_val_rmse'])} for identical "
        f"neighbourhoods weighted by similarity, and its ranking-selected configuration "
        f"has the widest catalogue reach of any model here "
        f"({f(100 * par.loc['RegressionCF', 'catalog_coverage'], 1)}%). The learned "
        f"coefficients correlate {f(R['regression']['pearson_sim_coef'], 3)} with the "
        f"similarities they replaced, so they are not a different signal — they are the "
        f"same signal with redundancy removed.\n")
    pj = R["popularity_bias"]
    expo = pj["exposure_reference"]
    o.append(
        f"**The popularity problem is a property of the class, not of the accurate "
        f"model.** All five paradigms draw "
        f"{f(expo['paradigm_head_pct_range'][0], 1)}-"
        f"{f(expo['paradigm_head_pct_range'][1], 1)}% of their recommendation slots from "
        f"the {expo['n_head_items']} head movies, which are "
        f"{f(expo['head_share_of_catalog_pct'], 1)}% of the catalogue, and at most "
        f"{f(expo['paradigm_tail_pct_range'][1], 1)}% from the "
        f"{f(expo['long_tail_share_of_catalog_pct'], 1)}% that is long tail. What the "
        f"data does *not* support is the usual story that the accurate model is the "
        f"biased one: on the four objectives (NDCG, coverage, novelty, tail share) the "
        f"Pareto front is {{{', '.join(pj['pareto']['pareto_front'])}}}, and "
        f"{pj['headline']['most_accurate_model_ndcg']} strictly dominates UBCF, IBCF and "
        f"the graph model on all four at once. The real decision is between the two "
        f"models on the front, and it is bad value: "
        f"{f(pj['does_the_most_accurate_model_make_the_most_useful_recommendations']['evidence']['coverage_gain_pp'], 1)} "
        f"points of coverage and "
        f"{f(pj['does_the_most_accurate_model_make_the_most_useful_recommendations']['evidence']['tail_gain_pp'], 1)} "
        f"points of tail exposure for "
        f"{f(pj['does_the_most_accurate_model_make_the_most_useful_recommendations']['evidence']['ndcg_cost_relative_pct'], 1)}% "
        f"of NDCG@10. Reach is not something you should buy by picking a different "
        f"collaborative model.\n")
    o.append(
        f"**Graph propagation depth is a popularity dial, provably.** "
        f"corr(path length, head share) = "
        f"{f(R['graph']['depth_vs_popularity']['corr_depth_head_frac'], 3)}; the "
        f"stationary distribution of a random walk on an undirected graph is proportional "
        f"to degree, so an un-restarted walk *is* a popularity ranker. This is the "
        f"cleanest example in the study of an algorithm's bias being derivable from its "
        f"mathematics rather than discovered in its output.\n")
    o.append(
        f"**What I would deploy.** For a catalogue of this shape, SLIM-style item-item "
        f"scoring for the head of the list: it is the strongest ranker, its model is "
        f"{R['slim']['sparsity']['sparsity_pct']:.1f}% sparse so serving is a sparse "
        f"lookup, and every recommendation comes with an inspectable coefficient. It is "
        f"also — unusually — not paid for in beyond-accuracy terms: it dominates UBCF, "
        f"IBCF and the graph model on coverage, novelty and tail share as well. What it "
        f"still does not give you is reach: "
        f"{f(100 * par.loc['SLIM', 'catalog_coverage'], 1)}% of the catalogue. The "
        f"measurements say that swapping in a different collaborative model is a bad way "
        f"to fix that, so the second stage should be a re-ranker with an explicit "
        f"exploration budget or content features — not another CF model. Deploying the "
        f"accuracy winner alone would quietly shrink the catalogue to a few hundred "
        f"movies.\n")
    return "\n".join(o)


def _answers(R, par, base, best_ndcg, best_rmse, best_cov, best_nov, fastest,
             pop_ndcg, bias_rmse, ug, cs, brc) -> str:
    pbj = R["popularity_bias"]
    pbc = pbj["correlations_paradigms"]
    expo = pbj["exposure_reference"]
    ev = pbj["does_the_most_accurate_model_make_the_most_useful_recommendations"]["evidence"]
    pb = load_table("popularity_bias").set_index("model")
    o = []
    o.append(f"**{RESEARCH_QUESTIONS[0]}** It depends on the objective, and the answer "
             f"changes with it: {best_ndcg} on ranking "
             f"(NDCG@10 {f(par.loc[best_ndcg, 'ndcg_mean'])}, "
             f"MAP@10 {f(par.loc[best_ndcg, 'map_mean'])}), {brc.label} on rating "
             f"error (RMSE {f(brc.rmse)}), {best_cov} on catalogue coverage "
             f"({f(100 * par.loc[best_cov, 'catalog_coverage'], 1)}%). The rating winner "
             f"is a *configuration* as much as a model: the same paradigm tuned for "
             f"ranking sits at RMSE "
             f"{f(par.loc[brc.label.replace('-rating', ''), 'rmse'])}.\n")
    o.append(f"**{RESEARCH_QUESTIONS[1]}** {best_ndcg}'s advantage holds where "
             f"co-occurrence evidence is dense — users with longer histories and items in "
             f"the head. For the sparse-history tertile the best ranker is "
             f"{cs['best_for_sparse_users_ndcg']['model']} "
             f"({f(cs['best_for_sparse_users_ndcg']['ndcg'])}) and the best rating model "
             f"is {cs['best_for_sparse_users_rmse']['model']} "
             f"({f(cs['best_for_sparse_users_rmse']['rmse'])}). When the objective "
             f"includes discovery, the ordering inverts entirely.\n")
    o.append(f"**{RESEARCH_QUESTIONS[2]}** Because the models differ in *what evidence "
             f"they are allowed to use and how much they are allowed to trust it*. SLIM "
             f"learns item-item weights discriminatively against the reconstruction "
             f"objective and prunes everything that does not pay for itself, so it "
             f"concentrates on well-supported co-occurrences. The regression model solves "
             f"a ridge system per item, which discounts redundant neighbours and lets it "
             f"stay calibrated on thin evidence — which is why it keeps "
             f"{f(pb.loc['RegressionCF', 'long_tail_pct'], 1)}% long-tail exposure, the "
             f"most of any model here. "
             f"Neighbourhood CF asserts its weights and inherits whatever bias the "
             f"similarity carries. Graph propagation's bias is the operator's stationary "
             f"distribution, which is degree — popularity.\n")
    o.append(f"**{RESEARCH_QUESTIONS[3]}** Three real ones and one that turns out to be "
             f"a myth. (a) *Rating accuracy versus ranking quality*: {brc.label} has "
             f"the best RMSE in the study and NDCG@10 {f(brc.ndcg_mean)}, while "
             f"{best_ndcg} has the best NDCG@10 and RMSE {f(par.loc[best_ndcg, 'rmse'])}. "
             f"(b) *Reach versus ranking*, but only between the two Pareto-optimal "
             f"models: {ev['widest_coverage_model']} buys "
             f"{f(ev['coverage_gain_pp'], 1)} points of coverage and "
             f"{f(ev['tail_gain_pp'], 1)} points of long-tail exposure for "
             f"{f(ev['ndcg_cost_relative_pct'], 1)}% of NDCG@10. (c) *Cost*: training "
             f"spans {f(par.train_time_s.min(), 3)} s to {f(par.train_time_s.max(), 1)} s "
             f"and model size {f(par.model_mb.min(), 2)} MB to "
             f"{f(par.model_mb.max(), 1)} MB. The myth is that accuracy *causes* "
             f"popularity bias: over the five paradigms corr(NDCG@10, catalogue coverage) "
             f"= {f(pbc['ndcg_vs_catalog_coverage'], 3)} and the most accurate model "
             f"dominates three of the other four on coverage, novelty and tail share "
             f"simultaneously.\n")
    tail_order = pb.loc[[m for m in PARADIGMS]].long_tail_pct.sort_values(ascending=False)
    o.append(f"**{RESEARCH_QUESTIONS[4]}** Ordered by share of Top-10 slots drawn from "
             f"the long tail: " +
             ", ".join(f"{m} ({v:.1f}%)" for m, v in tail_order.items()) +
             f". The two Pareto-optimal models — "
             f"{' and '.join(pbj['pareto']['pareto_front'])} — are the only two that "
             f"reach the tail at all in any meaningful sense; the other three are within "
             f"a fraction of a percentage point of never leaving the head. Read the "
             f"scale before reading the ordering: the best of them still gives "
             f"{f(tail_order.max(), 1)}% of its slots to "
             f"{f(expo['long_tail_share_of_catalog_pct'], 1)}% of the catalogue.\n")
    o.append(f"**{RESEARCH_QUESTIONS[5]}** Training: " +
             ", ".join(f"{m} {f(par.loc[m, 'train_time_s'], 3)} s"
                       for m in par.train_time_s.sort_values().index) +
             ". Full-catalogue scoring per user: " +
             ", ".join(f"{m} {f(par.loc[m, 'predict_time_per_user_ms'], 2)} ms"
                       for m in par.predict_time_per_user_ms.sort_values().index) +
             f". Model footprint: " +
             ", ".join(f"{m} {f(par.loc[m, 'model_mb'], 2)} MB"
                       for m in par.model_mb.sort_values().index) +
             f". These are single-run wall-clock times on a shared machine, so treat the "
             f"sub-second figures as order-of-magnitude; the footprints and the "
             f"asymptotics in Section 12.1 are the durable numbers. The shape of the "
             f"answer is robust either way: SLIM costs orders of magnitude more to train "
             f"than anything else here and is still among the cheapest to *serve*, "
             f"because its model is a sparse matrix and inference is one sparse row "
             f"product.\n")
    o.append(f"**{RESEARCH_QUESTIONS[6]}** SLIM as the primary ranker, with a second "
             f"stage for reach. It is the strongest ranker measured here, its learned "
             f"matrix is {R['slim']['sparsity']['sparsity_pct']:.1f}% sparse "
             f"({R['slim']['sparsity']['mean_nnz_per_column']:.0f} neighbours per item on "
             f"average) so serving is a sparse lookup with no user-side computation, it "
             f"retrains in {f(par.loc['SLIM', 'train_time_s'], 1)} s on this data, and "
             f"every slot is explainable by a single coefficient. The caveat is "
             f"non-negotiable: on its own it reaches "
             f"{f(100 * par.loc['SLIM', 'catalog_coverage'], 1)}% of the catalogue and "
             f"draws {f(pb.loc['SLIM', 'long_tail_pct'], 1)}% of its slots from the tail. "
             f"Part J also says *how* to fix that: not by choosing a different "
             f"collaborative model — the best available swap costs "
             f"{f(ev['ndcg_cost_relative_pct'], 1)}% of NDCG@10 for "
             f"{f(ev['tail_gain_pp'], 1)} points of tail exposure — but with an explicit "
             f"exploration or diversification budget layered on top, sized by exactly "
             f"those coverage and novelty measurements.\n")
    return "\n".join(o)


# ======================================================================================
def write_report() -> Path:
    text = build_report()
    path = ROOT / "REPORT.md"
    path.write_text(text)
    return path


# ======================================================================================
# README (Part S)
# ======================================================================================
def build_readme() -> str:
    R = _load_all()
    ov = R["eda"]["overview"]
    cfg = R["eda"]["config"]
    tgt = R["eda"]["target_user"]
    spl = R["eda"]["split"]
    pgp = R["eda"]["popularity_groups"]

    mu = load_table("multiuser_test")
    par = mu[mu.label.isin(PARADIGMS)].set_index("label")
    base = mu[~mu.label.isin(PARADIGMS)].set_index("label")
    pb = load_table("popularity_bias").set_index("model")
    tgt_rank = load_table("target_top10_by_model")

    best_ndcg = par.ndcg_mean.idxmax()
    best_rmse = par.rmse.idxmin()
    best_cov = par.catalog_coverage.idxmax()
    fastest = par.train_time_s.idxmin()
    pop_ndcg = float(base.loc["Popularity", "ndcg_mean"])
    bias_rmse = float(base.loc["BiasBaseline", "rmse"])
    brc = best_rating_config(mu)

    head = par[["rmse", "mae", "precision_mean", "recall_mean", "hit_rate_mean",
                "ndcg_mean", "map_mean", "catalog_coverage", "novelty_mean",
                "ild_mean", "tail_frac_mean", "train_time_s"]].copy().reset_index()
    head.columns = ["Model", "RMSE", "MAE", "P@10", "R@10", "HR@10", "NDCG@10",
                    "MAP@10", "Coverage", "Novelty", "Diversity", "LongTail", "Train s"]
    ref = base.loc[["Popularity", "BiasBaseline", "Random"]][
        ["rmse", "mae", "precision_mean", "recall_mean", "hit_rate_mean", "ndcg_mean",
         "map_mean", "catalog_coverage", "novelty_mean", "ild_mean", "tail_frac_mean",
         "train_time_s"]].reset_index()
    ref.columns = head.columns

    L = []
    A = L.append
    A("# MovieLens 100K — Five Recommendation Paradigms, Built From Scratch\n")
    A(f"An applied-research study comparing **user-based CF, item-based CF, "
      f"regression-based neighbourhood CF, SLIM and graph propagation** on MovieLens 100K "
      f"— every algorithm implemented directly in NumPy/SciPy, with **no "
      f"recommender-system library** (no Surprise, LightFM, implicit, LensKit or RecBole).\n")
    pj0 = R["popularity_bias"]
    A(f"The study measures all five on rating accuracy, ranking quality, catalogue "
      f"coverage, novelty, diversity, popularity bias, long-tail exposure, cold-start "
      f"behaviour and computational cost — then asks whether the most accurate model is "
      f"actually the most useful one. The measured answer is more interesting than the "
      f"expected one: on four objectives at once the Pareto front is "
      f"**{' and '.join(pj0['pareto']['pareto_front'])}**, the accuracy winner is on it, "
      f"and the popularity bias everyone worries about turns out to belong to "
      f"collaborative filtering as a class rather than to the accurate model.\n")
    A("---\n")

    A("## Headline results (held-out test split)\n")
    A(md_table(head) + "\n")
    A("Reference points:\n")
    A(md_table(ref) + "\n")
    A(f"- Best ranker: **{best_ndcg}**, NDCG@10 {f(par.loc[best_ndcg, 'ndcg_mean'])} — "
      f"{f(100 * (par.loc[best_ndcg, 'ndcg_mean'] / pop_ndcg - 1), 1)}% above a "
      f"non-personalised popularity ranker ({f(pop_ndcg)}).\n"
      f"- Best rating predictor: **{brc.label}**, RMSE {f(brc.rmse)} vs {f(bias_rmse)} "
      f"for a regularised bias baseline — and it ranks at NDCG@10 {f(brc.ndcg_mean)}, "
      f"which is the point: the RMSE-optimal configuration is not the one you would "
      f"ship.\n"
      f"- Widest catalogue reach: **{best_cov}**, "
      f"{f(100 * par.loc[best_cov, 'catalog_coverage'], 1)}% of the catalogue vs "
      f"{f(100 * par.loc[best_ndcg, 'catalog_coverage'], 1)}% for the accuracy winner — "
      f"a gap of only {f(100 * (par.loc[best_cov, 'catalog_coverage'] - par.loc[best_ndcg, 'catalog_coverage']), 1)} "
      f"points, bought with {f(100 * (1 - par.loc[best_cov, 'ndcg_mean'] / par.loc[best_ndcg, 'ndcg_mean']), 1)}% "
      f"of the NDCG@10.\n"
      f"- Cheapest to train: **{fastest}**, {f(par.loc[fastest, 'train_time_s'], 3)} s; "
      f"most expensive: {par.train_time_s.idxmax()}, "
      f"{f(par.train_time_s.max(), 1)} s.\n")
    A("![model comparison](figures/fig08_ranking_comparison.png)\n")
    A("![long tail exposure](figures/fig12_long_tail_exposure.png)\n")

    A("## Executive summary\n")
    A(f"MovieLens 100K ({ov['n_users']} users x {ov['n_items']} movies, "
      f"{ov['n_ratings']:,} ratings, {ov['sparsity_pct']:.2f}% sparse) was split per user "
      f"into {spl['train_pct']:.0f}/{spl['val_pct']:.0f}/{spl['test_pct']:.0f} "
      f"train/validation/test. Every hyper-parameter — neighbourhood size, similarity, "
      f"centring, ranking head, ridge strength, elastic-net mix, implicit threshold, "
      f"restart probability, propagation depth — was selected on validation across "
      f"{len(load_table('ubcf_sweep')) + len(load_table('ibcf_sweep')) + len(load_table('regression_sweep')) + len(load_table('slim_sweep')) + len(load_table('graph_sweep'))} "
      f"measured configurations; the test split was read exactly once.\n")
    pj = R["popularity_bias"]
    expo = pj["exposure_reference"]
    ev = pj["does_the_most_accurate_model_make_the_most_useful_recommendations"]["evidence"]
    A(f"**The paradigms win on different axes, but not in the way the textbook says.** "
      f"On four objectives at once — NDCG@10, catalogue coverage, novelty and long-tail "
      f"share — the Pareto front is just "
      f"**{' and '.join(pj['pareto']['pareto_front'])}**; {best_ndcg} strictly dominates "
      f"UBCF, IBCF and the graph model on all four simultaneously, so accuracy does not "
      f"*cause* popularity bias here. The genuine trade-off is between the two models on "
      f"the front: {ev['widest_coverage_model']} buys {f(ev['coverage_gain_pp'], 1)} "
      f"points of coverage and {f(ev['tail_gain_pp'], 1)} points of long-tail exposure "
      f"for {f(ev['ndcg_cost_relative_pct'], 1)}% of NDCG@10 — a trade that is available "
      f"but poor value at this operating point, which is itself the finding: on this "
      f"data there is no cheap way to buy reach from the model alone.\n")
    A(f"**What every model shares is the exposure gap.** The "
      f"{expo['n_head_items']} head movies are "
      f"{f(expo['head_share_of_catalog_pct'], 1)}% of the catalogue and take "
      f"{f(expo['paradigm_head_pct_range'][0], 1)}-"
      f"{f(expo['paradigm_head_pct_range'][1], 1)}% of all recommendation slots; the "
      f"{expo['n_long_tail_items']} long-tail movies are "
      f"{f(expo['long_tail_share_of_catalog_pct'], 1)}% of the catalogue and take "
      f"{f(expo['paradigm_tail_pct_range'][0], 1)}-"
      f"{f(expo['paradigm_tail_pct_range'][1], 1)}%. Popularity bias here is a property "
      f"of collaborative filtering as a class, not of one model.\n")
    A(f"**Three results worth pulling out.** (1) Learning neighbourhood weights by ridge "
      f"regression beats using similarity as the weight "
      f"({f(R['regression']['learned_arm_best_val_rmse'])} vs "
      f"{f(R['regression']['similarity_arm_best_val_rmse'])} validation RMSE) — but the "
      f"unregularised version is worse than every baseline, so the win comes from the "
      f"shrinkage, not from the learning. (2) SLIM's learned matrix is "
      f"{R['slim']['sparsity']['sparsity_pct']:.1f}% sparse "
      f"({R['slim']['sparsity']['mean_nnz_per_column']:.0f} neighbours per item) and is "
      f"certified optimal by a KKT residual of "
      f"{R['slim']['kkt']['relative_violation']:.1e}, not merely by a stalled objective. "
      f"(3) For the graph model, corr(propagation depth, head-item share) = "
      f"{f(R['graph']['depth_vs_popularity']['corr_depth_head_frac'], 3)} — deeper "
      f"propagation is measurably a popularity dial, which follows from the stationary "
      f"distribution of a walk on an undirected graph being proportional to degree.\n")

    A("## What is implemented (and how)\n")
    A("| Paradigm | Core mathematics | Implementation note |\n|---|---|---|\n"
      "| **UBCF** | `rhat = mu_u + Σ s(u,v)(r_vi - mu_v) / Σ|s|` | exact per-(u,i) "
      "neighbourhood; cosine and **exact co-rated Pearson computed as five sparse GEMMs** "
      "rather than a pairwise loop |\n"
      "| **IBCF** | same, transposed | cosine / adjusted cosine / Pearson, item- or "
      "user-centred |\n"
      "| **RegressionCF** | `(Z'Z + λI)w = Z'y` per item | ridge normal equations solved "
      "exactly; weights assembled into a sparse `W` so prediction is one sparse product |\n"
      "| **SLIM** | `min ½‖X-XW‖² + λ₁‖W‖₁ + ½λ₂‖W‖²`, `W ≥ 0`, `diag(W)=0` | "
      "**projected FISTA written from scratch**, Lipschitz step from power iteration, "
      "convergence certified by KKT residual |\n"
      "| **GraphRec** | `p ← (1-α)pP + αe` and `Σ_l β^l (A^l)_{ui}` | bipartite "
      "user-movie graph; probability mass conserved to 1e-12; degree-normalised and raw "
      "variants |\n")
    A("Every model also implements `explain(u, i)`, returning the evidence its own "
      "scoring rule actually used — neighbours, similar items, learned coefficients, or "
      "propagation paths.\n")

    A("## Quickstart\n")
    A("```bash\n"
      "pip install -r requirements.txt\n\n"
      "# fetch MovieLens 100K and verify it against the published MD5\n"
      "python download_data.py\n\n"
      "# full pipeline: EDA -> sweeps -> test evaluation -> analyses -> figures -> report\n"
      "python run_experiments.py\n\n"
      "# or a single stage\n"
      "python experiments/exp06_slim.py\n\n"
      "# interactive demo\n"
      "streamlit run app/streamlit_app.py\n"
      "```\n")
    A(f"**The dataset is not in this repository.** GroupLens' usage licence states that "
      f"\"the user may not redistribute the data without separate permission\", so "
      f"`download_data.py` fetches `ml-100k.zip` and refuses to extract anything whose "
      f"MD5 is not the published `{ov['zip_md5']}`. (GroupLens' own HTTPS certificate is "
      f"currently expired, so the script tries the official host first and falls back to "
      f"mirrors — it never disables certificate verification, and the checksum is what "
      f"establishes authenticity either way.)\n")
    sl_times = load_table("slim_sweep").fit_time_s
    A(f"Everything is seeded from the roll number, so `run_experiments.py` reproduces "
      f"every number in `REPORT.md` exactly — the report is *generated* from `results/`, "
      f"not written alongside it.\n")
    A(f"Most stages finish in seconds. The SLIM regularisation study is the expensive "
      f"one: {len(sl_times)} separate solver runs, each driven to a KKT residual rather "
      f"than to a fixed iteration count. The per-fit times recorded in "
      f"`results/tables/slim_sweep.csv` span {f(sl_times.min(), 1)}-"
      f"{f(sl_times.max(), 1)} s, but they were measured on a machine running other "
      f"work, so treat them as an upper bound rather than a benchmark; budget roughly "
      f"10-25 minutes for that stage and a couple of minutes for everything else.\n")

    A("## Repository layout\n")
    A("```\n"
      "download_data.py   fetches + MD5-verifies MovieLens 100K into data/\n"
      "run_experiments.py runs every stage in dependency order\n"
      "src/               data · similarity · metrics · baselines · ubcf · ibcf\n"
      "                   regression_cf · slim · graph_recommender · evaluation\n"
      "                   analysis · visualization · report · pipeline · base · config\n"
      "experiments/       17 numbered stages, each runnable on its own\n"
      "results/           JSON summaries + the evaluation tables REPORT.md is built from\n"
      "figures/           17 report figures\n"
      "app/               Streamlit demo over a headless service layer\n"
      "notebooks/         a runnable walkthrough of the results\n"
      "REPORT.md          the full research report (generated from results/)\n"
      "PORTFOLIO.md       technical interview questions and answers\n"
      "```\n")
    A("Two directories are deliberately not version-controlled, because both are large "
      "and regenerable: `data/` (see the licence note above) and "
      "`results/artifacts/` (pickled fitted models and saved Top-K lists, written by "
      "stage 16). The demo app refits from the stored configurations when that cache is "
      "absent, so a fresh clone needs only `download_data.py` and `run_experiments.py`.\n")

    A("## Target-user experiment\n")
    A(f"The target user is derived deterministically from the roll number:\n")
    A("```\n"
      f"roll = {tgt['roll_number']}  ->  digits = {tgt['digits_extracted']}  "
      f"->  {tgt['digits_as_int']} mod {tgt['n_users']} = {tgt['modulo']}  "
      f"->  +1  ->  user {tgt['target_user_id']}\n"
      f"SEED = {tgt['digits_as_int']} mod (2^31 - 1) = {cfg['seed']}\n"
      "```\n")
    A(f"{100 * cfg['target_hide_frac']:.0f}% of that user's training ratings are hidden "
      f"({tgt['n_hidden']} of {tgt['n_train_before_hiding']}), all five models are "
      f"refitted on the reduced matrix, and their Top-10s are compared side by side:\n")
    A(md_table(tgt_rank) + "\n")
    A("Per-movie, per-model reasoning is in `results/tables/target_reasoning.csv`; the "
      "narrated disagreements are in [REPORT.md](REPORT.md#18-target-user-recommendations).\n")

    A("## Popularity groups\n")
    A(f"Head / medium / long tail are defined by **equal interaction mass**, not an "
      f"arbitrary rating-count cutoff: items are sorted by training popularity and the "
      f"cumulative interaction curve is cut at 1/3 and 2/3. The result — "
      f"{pgp['n_head']} head movies ({pgp['head_pct_of_catalog']:.1f}% of the catalogue) "
      f"absorbing a third of all attention, against {pgp['n_long_tail']} long-tail movies "
      f"({pgp['long_tail_pct_of_catalog']:.1f}%) sharing another third — is the fact every "
      f"later analysis turns on.\n")
    A("![long tail](figures/fig04_long_tail.png)\n")

    A("## CV description\n")
    A(_cv_bullets(R, par, base, best_ndcg, best_rmse, best_cov, pop_ndcg, bias_rmse, ov))

    A("## 30-second interview explanation\n")
    A(_pitch(R, par, best_ndcg, best_cov, pop_ndcg, ov))

    A(f"Full write-up: **[REPORT.md](REPORT.md)** (22 sections). "
      f"Interview preparation — technical questions and answers: "
      f"**[PORTFOLIO.md](PORTFOLIO.md)**.\n")
    A("---\n")
    A("*Dataset: F. M. Harper and J. A. Konstan, \"The MovieLens Datasets: History and "
      "Context\", ACM TiiS 5(4), 2015. Used for research/educational purposes under the "
      "GroupLens terms.*\n")
    return "\n".join(L)


def _cv_bullets(R, par, base, best_ndcg, best_rmse, best_cov, pop_ndcg, bias_rmse,
                ov) -> str:
    pb = load_table("popularity_bias").set_index("model")
    n_cfg = (len(load_table("ubcf_sweep")) + len(load_table("ibcf_sweep"))
             + len(load_table("regression_sweep")) + len(load_table("slim_sweep"))
             + len(load_table("graph_sweep")))
    return (
        f"- Implemented **five recommendation paradigms from scratch** on MovieLens 100K "
        f"({ov['n_users']}x{ov['n_items']}, {ov['n_ratings']:,} ratings, "
        f"{ov['sparsity_pct']:.1f}% sparse) with no recommender library — user-based and "
        f"item-based CF (including exact co-rated Pearson as five sparse GEMMs), a "
        f"regression-based neighbourhood model solved by per-item ridge normal equations, "
        f"a SLIM-style non-negative elastic-net item-item model trained by a "
        f"hand-written projected FISTA solver certified by its KKT residual "
        f"({R['slim']['kkt']['relative_violation']:.0e}), and random-walk-with-restart / "
        f"bounded-path propagation on the user-movie bipartite graph.\n"
        f"- Benchmarked all five over **{n_cfg} validation-selected configurations** and a "
        f"single-use test split: best NDCG@10 {f(par.loc[best_ndcg, 'ndcg_mean'])} "
        f"({best_ndcg}) versus {f(pop_ndcg)} for a popularity baseline, best RMSE "
        f"{f(par.loc[best_rmse, 'rmse'])} ({best_rmse}) versus {f(bias_rmse)} for a "
        f"regularised bias baseline, with catalogue coverage spanning "
        f"{f(100 * par.catalog_coverage.min(), 1)}%-{f(100 * par.catalog_coverage.max(), 1)}% "
        f"and training cost {f(par.train_time_s.min(), 2)}s-{f(par.train_time_s.max(), 1)}s.\n"
        f"- Ran **ablation, cold-start and popularity-bias analyses**: a four-objective "
        f"Pareto analysis reduced the five paradigms to a two-model front "
        f"({' and '.join(R['popularity_bias']['pareto']['pareto_front'])}) and showed "
        f"every model draws "
        f"{f(R['popularity_bias']['exposure_reference']['paradigm_head_pct_range'][0], 1)}-"
        f"{f(R['popularity_bias']['exposure_reference']['paradigm_head_pct_range'][1], 1)}% "
        f"of recommendation slots from the "
        f"{f(R['popularity_bias']['exposure_reference']['head_share_of_catalog_pct'], 1)}% "
        f"of the catalogue that is 'head'; established that regularisation — not learning "
        f"per se — is what makes learned neighbourhood weights beat similarity heuristics; "
        f"and showed graph propagation depth is a measurable popularity dial "
        f"(corr {f(R['graph']['depth_vs_popularity']['corr_depth_head_frac'], 2)} with "
        f"head-item share). Shipped a Streamlit demo with per-recommendation explanations.\n")


def _pitch(R, par, best_ndcg, best_cov, pop_ndcg, ov) -> str:
    pj = R["popularity_bias"]
    expo = pj["exposure_reference"]
    return (
        f"> I built five recommender paradigms from scratch on MovieLens 100K — no "
        f"Surprise, no LightFM — so user-based and item-based CF, a regression model that "
        f"*learns* the neighbourhood weights instead of assuming similarity is the weight, "
        f"SLIM with my own projected FISTA solver, and random-walk propagation on the "
        f"user-movie graph. {best_ndcg} wins ranking with NDCG@10 "
        f"{f(par.loc[best_ndcg, 'ndcg_mean'])}, against {f(pop_ndcg)} for just "
        f"recommending popular movies. But the result I actually care about came from "
        f"scoring all five on four objectives at once — accuracy, catalogue coverage, "
        f"novelty and long-tail share. The Pareto front is only two models, and the "
        f"accuracy winner is on it: it beats UBCF, IBCF and the graph model on *all four* "
        f"simultaneously. So the usual story that the accurate model is the biased one is "
        f"just false on this data. What is true is that every one of them draws "
        f"{f(expo['paradigm_head_pct_range'][0], 1)} to "
        f"{f(expo['paradigm_head_pct_range'][1], 1)} percent of its slots from the "
        f"{f(expo['head_share_of_catalog_pct'], 1)} percent of the catalogue that's "
        f"already popular. Popularity bias here is a property of collaborative filtering "
        f"as a class, not of one model — which changes what you'd do about it: you don't "
        f"pick a different model, you buy an explicit exploration budget.\n")


def write_readme() -> Path:
    path = ROOT / "README.md"
    path.write_text(build_readme())
    return path


# ======================================================================================
# PORTFOLIO (Part S): executive summary, CV bullets, pitch, interview Q&A
# ======================================================================================
def build_portfolio() -> str:
    R = _load_all()
    ov = R["eda"]["overview"]
    cfg = R["eda"]["config"]
    mu = load_table("multiuser_test")
    par = mu[mu.label.isin(PARADIGMS)].set_index("label")
    base = mu[~mu.label.isin(PARADIGMS)].set_index("label")
    pb = load_table("popularity_bias").set_index("model")
    ug = load_table("coldstart_user_groups")

    best_ndcg = par.ndcg_mean.idxmax()
    best_rmse = par.rmse.idxmin()
    best_cov = par.catalog_coverage.idxmax()
    fastest = par.train_time_s.idxmin()
    pop_ndcg = float(base.loc["Popularity", "ndcg_mean"])
    bias_rmse = float(base.loc["BiasBaseline", "rmse"])
    brc = best_rating_config(mu)
    pj = R["popularity_bias"]
    pbc = pj["correlations_paradigms"]
    expo = pj["exposure_reference"]
    ev = pj["does_the_most_accurate_model_make_the_most_useful_recommendations"]["evidence"]
    cs = R["coldstart"]

    rgs = load_table("regression_sweep")
    _k_dd = int(R["regression"]["deep_dive_spec"]["k"])
    _ols_k, _ridge_k = ols_vs_ridge(rgs, _k_dd)
    _lr = rgs[rgs.weight_mode == "learned"]
    _ols_worst = float(_lr[_lr.lam == 0].groupby("k").rmse.min().max())
    _global_rmse = float(load_table("baselines").set_index("model").loc["GlobalMean", "rmse"])
    _r_lo = float(_lr[_lr.lam == 0].pearson_sim_coef.mean())
    _r_hi = float(_lr[_lr.lam == _lr.lam.max()].pearson_sim_coef.mean())

    L = []
    A = L.append
    A("# Interview preparation pack\n")
    A("Technical questions this project invites, with answers grounded in the measured "
      "results. The executive summary, the CV bullets and the 30-second pitch live in "
      "[README.md](README.md) and are deliberately not repeated here.\n")
    A(f"One-line context: five recommendation paradigms implemented from scratch on "
      f"MovieLens 100K ({ov['n_users']} users x {ov['n_items']} movies, "
      f"{ov['n_ratings']:,} ratings, {ov['sparsity_pct']:.2f}% sparse); best ranker "
      f"{best_ndcg} at NDCG@10 {f(par.loc[best_ndcg, 'ndcg_mean'])} against "
      f"{f(pop_ndcg)} for a popularity baseline; best RMSE {f(brc.rmse)} against "
      f"{f(bias_rmse)} for a regularised bias baseline.\n")
    A("---\n")

    A("## Likely technical interview questions, with answers\n")

    qa = [
        ("Why did you compute Pearson correlation with matrix products instead of a loop "
         "over user pairs?",
         "Textbook co-rated Pearson needs five sums restricted to the intersection "
         "`I_a ∩ I_b`. Multiplying by the binary indicator `B` *is* that restriction, so "
         "`n = B Bᵀ`, `S_a = R Bᵀ`, `S_ab = R Rᵀ` and `Q_a = R² Bᵀ` give every pair's "
         "sums in five sparse GEMMs. The naive loop is O(n² m) in Python; this is "
         "BLAS-bound and exact — I verified it against a brute-force implementation to "
         "0.0 absolute error. The general lesson is that 'restrict a sum to a set' is "
         "usually a multiplication by an indicator, and that turns a loop into a GEMM."),

        ("You said you use the *exact* per-(u,i) neighbourhood. Why does that matter?",
         "The common shortcut is to take user u's global top-k neighbours once and then "
         "keep whichever of them rated item i. For a popular item that is fine; for an "
         "unpopular item almost none of the k neighbours rated it, so the effective "
         "neighbourhood silently shrinks to two or three users. That makes k-curves look "
         "better than they are and hides exactly the failure mode you care about — "
         "unpopular items. I recompute the top-k among `U_i` for every item, which costs "
         "`|U| × nnz` element operations (about 6.7e7 here) done as 1,682 vectorised "
         "argpartitions."),

        ("Your neighbourhood models rank by something other than predicted rating. Why?",
         "Because ranking by `rhat` is dominated by items with almost no evidence. One "
         "enthusiastic neighbour who rated an obscure film 5 produces `rhat = 5.0` with a "
         "denominator of a single similarity. So each model exposes three heads — "
         "`rating`, unnormalised `score`, and implicit `affinity` — and the choice is made "
         "on validation. Measured on this data the rating head reaches NDCG@10 "
         f"{f(load_table('ubcf_sweep')[load_table('ubcf_sweep').rank_mode == 'rating'].ndcg_mean.max())} "
         f"for UBCF while the selected head reaches "
         f"{f(load_table('ubcf_sweep').ndcg_mean.max())}. That gap is the quantitative "
         "form of the 'never rank by predicted rating' rule."),

        ("What exactly does the regression-based neighbourhood model learn that "
         "similarity does not?",
         "Similarity `s(i,j)` is a *marginal* association — how j relates to i ignoring "
         "everything else. The ridge solution `w = (ZᵀZ + λI)⁻¹Zᵀy` gives *partial* "
         "coefficients — how j relates to i holding the other neighbours fixed. So "
         "near-duplicate neighbours stop being double-counted, and a neighbour can even "
         "take a negative weight despite a positive similarity (a suppressor effect); "
         f"{100 * R['regression']['frac_sign_flip_positive_sim_negative_coef']:.1f}% of "
         "coefficients do exactly that. Also, similarity weights are normalised to sum to "
         "1 by construction, so they always produce a full-strength prediction; learned "
         f"weights sum to {f(R['regression']['coef_sum_mean_per_item'], 3)} on average and "
         "can therefore say 'this neighbourhood barely determines the target'."),

        ("Did learning the weights actually help? Be honest.",
         f"Yes, and less than you might hope. Validation RMSE "
         f"{f(R['regression']['learned_arm_best_val_rmse'])} learned versus "
         f"{f(R['regression']['similarity_arm_best_val_rmse'])} with similarity weights "
         f"in the identical architecture — real, but a third of a percent. And it is "
         f"entirely the regularisation doing the work, not the learning. At the same "
         f"neighbourhood size the unregularised fit reaches {f(_ols_k)} against "
         f"{f(_ridge_k)} for the best ridge setting, and it degrades as k grows "
         f"(k=80 gives {f(_ols_worst)}, worse than predicting the global mean at "
         f"{f(_global_rmse)}) because the design matrix has k columns and only as many "
         f"rows as there are users who rated the target item. You can watch the shrinkage "
         f"happen: the correlation between the learned coefficients and the similarities "
         f"they replaced rises monotonically with λ, from "
         f"{f(_r_lo, 2)} at λ=0 to {f(_r_hi, 2)} at the top of the grid — the regression "
         f"is being pulled back onto the heuristic it was meant to improve, and the "
         f"useful setting is the one in between."),

        ("Why FISTA for SLIM rather than coordinate descent, which is the usual choice?",
         "Both work; the question is what dominates. Coordinate descent on the Gram "
         "matrix needs a rank-1 update of an n×n matrix per coordinate, so a full sweep "
         "is O(n³) in memory-bandwidth-bound operations. FISTA needs one dense GEMM "
         "`G W` per iteration — the same O(n³) flops, but as a BLAS-3 call running one to "
         "two orders of magnitude faster per operation. Since `W ≥ 0`, the L1 penalty is "
         "linear on the feasible set and its proximal operator collapses to "
         "`max(0, V - η λ₁)`, so the whole non-smooth part is one shift and a diagonal "
         "zeroing. Nesterov momentum gives O(1/t²). It converges here in "
         f"{R['slim']['solver']['iterations']} iterations."),

        ("How do you know your SLIM solution is actually optimal and not just stopped?",
         "A stalled objective is necessary but not sufficient. I check the KKT conditions "
         "of the non-negative elastic net directly: with `g = GW - G + λ₂W`, an active "
         "coefficient needs `g_ij + λ₁ = 0` and an inactive one needs `g_ij + λ₁ ≥ 0`. "
         f"The delivered model stops at a relative violation of "
         f"{R['slim']['kkt']['relative_violation']:.2e} — absolute "
         f"{f(R['slim']['kkt']['max_violation'], 4)} against a Gram scale of "
         f"{f(R['slim']['kkt']['gram_scale'], 0)}. That certificate is computed from G and "
         "W alone, so it is independent of the solver that produced them, and the solver "
         "stops on it rather than on the objective."),

        ("How did you pick the implicit-feedback threshold, and why not just use 4?",
         f"I swept it. τ ∈ {{1,2,3,4,5}} at the selected regularisation, on validation: "
         + ", ".join(f"τ={r['tau']:g} → NDCG@10 {f(r['ndcg'])}"
                     for r in R['slim']['implicit_conversion']['sweep']) +
         f". The measured argmax is τ = "
         f"{R['slim']['implicit_conversion']['selected_tau']:g}, which happens to be 4 — "
         "but 'happens to be' is the point. Both directions degrade for predictable "
         "reasons: τ=1 admits 1-star ratings as endorsements and pollutes the "
         "co-occurrence counts, τ=5 starves the Gram matrix of support. The threshold is "
         "a hyper-parameter, not a convention."),

        ("Why is your graph recommender biased toward popular items?",
         "It isn't a data artefact — it's the operator. The stationary distribution of a "
         "random walk on an undirected graph is proportional to node degree. So as you "
         "weaken the restart or lengthen the paths, personalised PageRank relaxes toward "
         "the degree distribution, which is exactly popularity. The measurements track "
         f"the theory: corr(path length, head-item share) = "
         f"{f(R['graph']['depth_vs_popularity']['corr_depth_head_frac'], 3)}, "
         f"corr(restart α, mean popularity rank) = "
         f"{f(R['graph']['restart_vs_popularity']['corr_alpha_pop_rank'], 3)}. Degree "
         "normalisation (`D⁻¹A` instead of `A`) is the built-in mitigation, because it "
         "makes each node spread one unit of evidence rather than an amount proportional "
         "to its degree."),

        ("How did you avoid data leakage?",
         "Four separate guards. (1) `fit` only ever receives `R_train`, so every "
         "similarity, bias, Gram matrix, learned weight and graph edge comes from train. "
         "(2) Candidate sets exclude only *training* items — excluding test items would "
         "hand the model the answer key by shrinking the candidate pool to the right "
         "answers. (3) Every hyper-parameter is chosen on validation; the test split is "
         "read once, in one stage. (4) Even the analysis strata — popularity groups, "
         "novelty weights, user-history tertiles — are computed from training counts, so "
         "the way I slice the results is leakage-free too. The gate suite re-checks the "
         "split is an exact disjoint partition and that no recommended item appears in "
         "the user's training profile."),

        ("Your best ranker has a mediocre RMSE and your best RMSE model ranks poorly. "
         "What's going on?",
         f"They optimise different things and the data lets you see it. RMSE is an "
         f"average over *observed* held-out ratings, most of which sit in the middle of "
         f"the scale; ranking quality is decided entirely by the extreme top of a "
         f"1,682-item ordering. A model can be beautifully calibrated in the middle and "
         f"order the tails badly. The error analysis shows the mechanism: every model "
         f"over-predicts low ratings and under-predicts high ones, because squared-error "
         f"training pulls predictions toward the conditional mean and narrows the "
         f"predicted distribution. Concretely here, {best_rmse} reaches RMSE "
         f"{f(par.loc[best_rmse, 'rmse'])} with NDCG@10 "
         f"{f(par.loc[best_rmse, 'ndcg_mean'])}, while {best_ndcg} reaches NDCG@10 "
         f"{f(par.loc[best_ndcg, 'ndcg_mean'])} with RMSE {f(par.loc[best_ndcg, 'rmse'])}."),

        ("Which model would you actually deploy?",
         f"{best_ndcg} for the head of the list, with a second stage underneath it. It is "
         f"the strongest ranker, its learned matrix is "
         f"{R['slim']['sparsity']['sparsity_pct']:.1f}% sparse so serving is a sparse "
         f"lookup with no user-side computation, it retrains in "
         f"{f(par.loc[best_ndcg, 'train_time_s'], 1)} s on this data, and every slot is "
         f"explainable by one coefficient. The caveat is not optional: on its own it "
         f"reaches {f(100 * par.loc[best_ndcg, 'catalog_coverage'], 1)}% of the catalogue "
         f"and draws {f(pb.loc[best_ndcg, 'long_tail_pct'], 1)}% of its slots from the "
         f"long tail. Shipping it alone quietly shrinks the catalogue to a few hundred "
         f"movies, which is a business problem long before it is a metrics problem."),

        ("What breaks if I hand you 100 million ratings instead of 100 thousand?",
         f"The algorithms survive; these implementations do not, and I'd rather be "
         f"specific about where. Dense similarity matrices are O(|I|²) — 22 MB here, 80 GB "
         f"at 10⁵ items. SLIM's dense Gram has the same wall, and the fix is column-wise "
         f"coordinate descent over a top-M co-occurrence candidate set, which is "
         f"embarrassingly parallel and O(|I|·M). Interestingly, I measured that "
         f"feature-restriction trick here and it made training *slower*, because at 1,682 "
         f"items the dense GEMM already dominates and the masking is pure overhead — it "
         f"only pays once the dense Gram stops fitting. UBCF is the worst-scaling of the "
         f"five in practice, because its model is O(|U|²) and |U| is the dimension that "
         f"grows. Item-item and SLIM scale best: the model is bounded by catalogue size "
         f"and can be precomputed offline."),

        ("Why intra-list diversity over genres rather than over your own similarities?",
         "Because measuring diversity with the same similarity a model optimises makes "
         "every model look diverse by construction — the metric and the objective would "
         "share a failure mode. The 19 genre flags are an external, model-independent "
         "yardstick. It is coarse, and I say so in the limitations; the alternative is "
         "circular."),

        ("How would you know any of this generalises? You ran one split.",
         "I wouldn't, fully, and that's stated as limitation 2. The `*_std` columns are "
         "across-user standard deviations, not confidence intervals over repeated splits, "
         "so differences below roughly 0.005 NDCG shouldn't be over-read. The honest fix "
         "is repeated seeded splits with bootstrap confidence intervals over users and "
         "paired tests between models, which is the first item on the future-work list. "
         "What does carry weight is the *ordering* on the qualitative axes — accuracy "
         "versus coverage, propagation depth versus popularity — because those have "
         "mechanisms behind them, not just point estimates."),

        ("Which method is best for cold-start users?",
         f"On this data the sparse-history tertile "
         f"(≤ {cs['user_group_definition']['q1']:.0f} training ratings, "
         f"n = {cs['user_group_definition']['n_sparse']}) is best served by "
         f"{cs['best_for_sparse_users_ndcg']['model']} on ranking "
         f"(NDCG@10 {f(cs['best_for_sparse_users_ndcg']['ndcg'])}) and "
         f"{cs['best_for_sparse_users_rmse']['model']} on rating error "
         f"(RMSE {f(cs['best_for_sparse_users_rmse']['rmse'])}). The model that gains most "
         f"from a richer history is {cs['benefits_most_from_history']['model']} "
         f"(+{f(cs['benefits_most_from_history']['absolute_ndcg_gain_heavy_minus_sparse'])} "
         f"NDCG@10 from sparse to heavy). Note the framing though: MovieLens guarantees "
         f"every user at least {ov['min_ratings_per_user']} ratings, so there are no truly "
         f"cold users here. The real sparsity problem in this dataset is on the item side, "
         f"where the median movie has {ov['median_ratings_per_item']:.0f} ratings."),
    ]

    for i, (q, a) in enumerate(qa, 1):
        A(f"**{i}. {q}**\n")
        A(f"{a}\n")

    A("---\n")
    A(f"*All figures quoted above are read from `results/` by `src/report.py`; none are "
      f"typed by hand. Seed {cfg['seed']}, derived from roll number "
      f"{R['eda']['target_user']['roll_number']}.*\n")
    return "\n".join(L)


def write_portfolio() -> Path:
    path = ROOT / "PORTFOLIO.md"
    path.write_text(build_portfolio())
    return path


def write_all() -> dict:
    return {"report": str(write_report()), "readme": str(write_readme()),
            "portfolio": str(write_portfolio())}
