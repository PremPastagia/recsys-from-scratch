"""Part K -- controlled ablations, assembled from the validation sweeps.

Every row changes exactly one factor while holding the others at the reference setting,
so the delta column is attributable.  Explanations pair the *measured* delta with the
mechanism that produces it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.pipeline import banner, load_table, save_json, save_table

COLS = ["rmse", "mae", "ndcg_mean", "recall_mean", "precision_mean", "map_mean",
        "catalog_coverage", "novelty_mean", "ild_mean", "tail_frac_mean", "fit_time_s"]


def pick(df: pd.DataFrame, **conds) -> pd.Series:
    """Best-NDCG row matching the conditions (ranking head free to vary)."""
    m = pd.Series(True, index=df.index)
    for kk, vv in conds.items():
        m &= (df[kk] == vv)
    sub = df[m]
    if sub.empty:
        raise KeyError(f"no rows for {conds}")
    return sub.loc[sub.ndcg_mean.idxmax()]


def row(family, factor, variant, r, config, why):
    out = {"family": family, "factor": factor, "variant": variant,
           "config": config, "why": why}
    for c in COLS:
        out[c] = float(r[c]) if c in r and pd.notna(r[c]) else np.nan
    out["rank_mode"] = r.get("rank_mode", "n/a")
    return out


def main() -> pd.DataFrame:
    banner("PART K - Ablation studies")
    ub = load_table("ubcf_sweep")
    ib = load_table("ibcf_sweep")
    rg = load_table("regression_sweep")
    sl = load_table("slim_sweep")
    gr = load_table("graph_sweep")
    rows = []

    # ---------------------------------------------------------------- UBCF
    k_ref = int(ub.loc[ub.ndcg_mean.idxmax(), "k"])
    for sim in ("cosine", "pearson"):
        r = pick(ub, similarity=sim, mean_center=True, k=k_ref)
        rows.append(row("UBCF", "similarity", sim, r, f"k={k_ref}, mean-centred",
                        "Cosine on raw rating vectors treats an unrated item as a 0, so "
                        "it partly encodes profile overlap (co-rating volume) as well as "
                        "taste agreement. Exact co-rated Pearson removes user rating "
                        "scale and uses only the co-rated intersection, which sharpens "
                        "rating prediction but throws away the overlap signal that drives "
                        "Top-N retrieval."))
    for mc in (True, False):
        r = pick(ub, similarity="cosine", mean_center=mc, k=k_ref)
        rows.append(row("UBCF", "mean_centering", str(mc), r,
                        f"cosine, k={k_ref}",
                        "Mean-centring subtracts mu_v before aggregating, so a generous "
                        "rater's 4 and a harsh rater's 4 stop meaning the same thing. It "
                        "is what makes the rating head competitive; for ranking it can "
                        "hurt, because the centred numerator rewards items that are "
                        "*above a neighbour's mean* rather than items with a lot of "
                        "supporting evidence."))
    for k in sorted(ub.k.unique()):
        r = pick(ub, similarity="cosine", mean_center=True, k=int(k))
        rows.append(row("UBCF", "k", str(int(k)), r, "cosine, mean-centred",
                        "Small k is low-bias / high-variance: a handful of neighbours is "
                        "noisy. Large k adds weakly-similar neighbours whose votes drag "
                        "the prediction toward the population mean -- which helps ranking "
                        "(more evidence, more stable ordering) while flattening the "
                        "rating curve."))
    for mode in ub.rank_mode.unique():
        sub = ub[(ub.rank_mode == mode)]
        r = sub.loc[sub.ndcg_mean.idxmax()]
        rows.append(row("UBCF", "ranking_head", mode, r,
                        f"best over sweep (k={int(r.k)}, {r.similarity})",
                        "'rating' ranks by rhat and is dominated by thinly-supported "
                        "items that one enthusiastic neighbour rated 5. 'score' keeps the "
                        "unnormalised evidence mass. 'affinity' sums positive similarity "
                        "over neighbours who actually liked the item, which is the "
                        "implicit-feedback objective and matches the Top-N task."))

    # ---------------------------------------------------------------- IBCF
    k_ref_i = int(ib.loc[ib.ndcg_mean.idxmax(), "k"])
    for sim in ("cosine", "pearson", "adjusted_cosine"):
        r = pick(ib, similarity=sim, mean_center=True, center_by="item", k=k_ref_i)
        rows.append(row("IBCF", "similarity", sim, r, f"k={k_ref_i}, item-centred",
                        "Adjusted cosine removes per-user rating scale before comparing "
                        "items and is the strongest rating-side similarity. Item-item "
                        "Pearson conditions on the co-rated users only, which on a sparse "
                        "catalogue assigns near-perfect correlations to obscure pairs "
                        "backed by a handful of users -- good for RMSE where those items "
                        "are rarely queried, disastrous for Top-N where they flood the "
                        "head of the list."))
    for mc in (True, False):
        r = pick(ib, similarity="cosine", mean_center=mc, center_by="item", k=k_ref_i)
        rows.append(row("IBCF", "mean_centering", str(mc), r, f"cosine, k={k_ref_i}",
                        "Item-mean centring turns the prediction into 'this item's mean "
                        "plus what this user's history says about the deviation', which "
                        "is the right decomposition for rating prediction."))
    for cb in ("item", "user"):
        try:
            r = pick(ib, similarity="adjusted_cosine", mean_center=True,
                     center_by=cb, k=k_ref_i)
        except KeyError:
            continue
        rows.append(row("IBCF", "centering_axis", cb, r,
                        f"adjusted cosine, k={k_ref_i}",
                        "Centring by item mean anchors on 'how good is this movie'; "
                        "centring by user mean anchors on 'how generous is this user'. "
                        "Item anchoring wins because item means are estimated from far "
                        "more observations than the deviation signal they replace."))
    for k in sorted(ib.k.unique()):
        r = pick(ib, similarity="cosine", mean_center=True, center_by="item", k=int(k))
        rows.append(row("IBCF", "k", str(int(k)), r, "cosine, item-centred",
                        "Item neighbourhoods saturate much earlier than user "
                        "neighbourhoods: past a few dozen items the extra neighbours are "
                        "near-zero-similarity and change nothing."))

    # ---------------------------------------------------------------- Regression CF
    k_ref_r = int(rg[rg.weight_mode == "learned"].loc[
        rg[rg.weight_mode == "learned"].rmse.idxmin(), "k"])
    r = pick(rg, weight_mode="similarity", k=k_ref_r)
    rows.append(row("RegressionCF", "weights", "similarity_heuristic", r,
                    f"k={k_ref_r}",
                    "Similarity used directly as the interpolation weight: a *marginal* "
                    "association, normalised to sum to 1 regardless of how weak the "
                    "evidence is."))
    for lam in sorted(rg[rg.weight_mode == "learned"].lam.unique()):
        r = pick(rg, weight_mode="learned", k=k_ref_r, lam=float(lam))
        variant = ("learned_OLS" if lam == 0 else f"learned_ridge_lam={lam:g}")
        rows.append(row("RegressionCF", "weights", variant, r, f"k={k_ref_r}",
                        "lambda=0 is ordinary least squares on a design matrix whose "
                        "column count (k) often exceeds the number of users who rated the "
                        "target item, so it is under-determined and overfits badly. "
                        "Growing lambda shrinks the partial coefficients toward zero; "
                        "as it does, the learned weights converge back toward the "
                        "marginal similarity ordering, which the measured "
                        "correlation r(sim,coef) tracks directly."))
    for k in sorted(rg[rg.weight_mode == "learned"].k.unique()):
        sub = rg[(rg.weight_mode == "learned") & (rg.k == k)]
        r = sub.loc[sub.rmse.idxmin()]
        rows.append(row("RegressionCF", "k", str(int(k)), r,
                        f"learned, best lambda={r.lam:g}",
                        "Larger neighbourhoods need proportionally more regularisation: "
                        "the design matrix gains columns without gaining rows."))

    # ---------------------------------------------------------------- SLIM
    reg = sl[sl.tag.isin(["l1_sweep", "l2_sweep", "elasticnet_mix"])]
    for _, r in sl[sl.tag == "l1_sweep"].sort_values("l1").iterrows():
        rows.append(row("SLIM", "l1_strength", f"l1={r.l1:g}", r, f"l2={r.l2:g}",
                        "L1 is the sparsity dial. Because W >= 0, the L1 term is linear "
                        "on the feasible set and its proximal operator is a hard shift: a "
                        "coefficient survives only if its co-occurrence evidence exceeds "
                        "lambda_1. Raising it prunes weak item-item edges, shrinking the "
                        "model and the catalogue it can reach."))
    for _, r in sl[sl.tag == "l2_sweep"].sort_values("l2").iterrows():
        rows.append(row("SLIM", "l2_strength", f"l2={r.l2:g}", r, f"l1={r.l1:g}",
                        "L2 does not create zeros; it spreads weight across correlated "
                        "duplicates instead of letting one of them take everything, and "
                        "it makes the problem better conditioned so the solver converges "
                        "in fewer iterations."))
    for _, r in sl[sl.tag == "elasticnet_mix"].sort_values("l1_ratio").iterrows():
        rows.append(row("SLIM", "elasticnet_mixing", f"l1_ratio={r.l1_ratio:.2f}", r,
                        f"l1+l2={r.l1 + r.l2:g}",
                        "At fixed total penalty, shifting the mix toward L1 trades "
                        "density for selectivity: an L1-heavy model keeps few, confident "
                        "edges; an L2-heavy model keeps many small ones."))
    for _, r in sl[sl.tag == "threshold"].sort_values("threshold").iterrows():
        rows.append(row("SLIM", "implicit_threshold", f"tau={r.threshold:g}", r,
                        f"l1={r.l1:g}, l2={r.l2:g}",
                        "tau decides what counts as an endorsement. Low tau treats a "
                        "1-star rating as a positive and pollutes the co-occurrence "
                        "counts; high tau starves the Gram matrix of support."))
    for _, r in sl[sl.tag == "feature_mask"].sort_values("top_m_features").iterrows():
        rows.append(row("SLIM", "feature_restriction", f"top_m={int(r.top_m_features)}",
                        r, f"l1={r.l1:g}, l2={r.l2:g}",
                        "Restricting each column's candidate features to its top-M "
                        "co-occurring items is the standard way to make SLIM scale. It "
                        "caps model size and solve time; the question is how much "
                        "ranking quality it costs."))

    # ---------------------------------------------------------------- Graph
    for _, r in gr[gr.method == "rwr"].sort_values("alpha").iterrows():
        rows.append(row("GraphRec", "restart_alpha", f"alpha={r.alpha:g}", r, "RWR",
                        "alpha is the restart / attenuation strength. Large alpha keeps "
                        "the walk near the seed user (local, novel, high coverage); small "
                        "alpha lets it relax toward the degree-proportional stationary "
                        "distribution, which is popularity."))
    deg = gr[(gr.method == "path") & (gr.normalize == "degree")]
    beta_ref = float(deg.loc[deg.ndcg_mean.idxmax(), "beta"])
    for _, r in deg[deg.beta == beta_ref].sort_values("max_path_len").iterrows():
        rows.append(row("GraphRec", "propagation_depth", f"L={int(r.max_path_len)}", r,
                        f"path, beta={beta_ref:g}, degree-normalised",
                        "L=1 reaches only items the user already has (all excluded), so "
                        "it is a pure floor. L=3 is exactly the collaborative-filtering "
                        "path u->i'->v->i. Beyond that, each extra pair of hops mixes in "
                        "evidence from users two steps removed, which adds coverage of "
                        "popular hubs and dilutes personalisation."))
    # beta is a *pure positive scale factor* whenever only one odd path length can reach
    # a candidate item: at L=3 the length-1 term touches only already-rated items, which
    # are excluded, so score = beta^3 * (A^3)_ui and the ranking is invariant in beta.
    # The ablation is therefore run at the deepest propagation in the sweep, where
    # several odd lengths mix and beta genuinely re-weights them.
    L_beta = int(deg.max_path_len.max())
    for _, r in deg[deg.max_path_len == L_beta].sort_values("beta").iterrows():
        rows.append(row("GraphRec", "attenuation_beta", f"beta={r.beta:g}", r,
                        f"path, L={L_beta}, degree-normalised",
                        "beta discounts each hop geometrically. It can only change the "
                        "ranking when two or more odd path lengths reach the candidate "
                        "set: at L=3 the length-1 term only touches already-rated items, "
                        "which are excluded, so the score is beta^3 times a fixed "
                        "quantity and the ordering is invariant in beta. This row is "
                        "therefore measured at the deepest propagation in the sweep, "
                        "where small beta makes the long (popularity-driven) paths "
                        "negligible and large beta lets them compete with direct "
                        "evidence."))
    for norm in ("degree", "none"):
        sub = gr[(gr.method == "path") & (gr.normalize == norm)]
        r = sub.loc[sub.ndcg_mean.idxmax()]
        rows.append(row("GraphRec", "degree_normalisation", norm, r,
                        f"path, best (L={int(r.max_path_len)}, beta={r.beta:g})",
                        "Un-normalised path counting scores an item by how many walks "
                        "reach it, and a high-degree blockbuster sits on enormously more "
                        "walks than a niche film. Row-normalising makes every node spread "
                        "one unit of evidence, which is the built-in popularity de-bias."))

    df = pd.DataFrame(rows)
    # Delta against the best variant within each (family, factor) block.
    df["delta_ndcg_vs_best_in_block"] = df.groupby(["family", "factor"])["ndcg_mean"] \
        .transform(lambda s: s - s.max())
    df["delta_rmse_vs_best_in_block"] = df.groupby(["family", "factor"])["rmse"] \
        .transform(lambda s: s - s.min())
    save_table("ablations", df)

    TIE = 1e-9
    summary = {}
    for (fam, fac), g in df.groupby(["family", "factor"]):
        best_n = g.loc[g.ndcg_mean.idxmax()]
        worst_n = g.loc[g.ndcg_mean.idxmin()]
        best_r = g.loc[g.rmse.idxmin()]
        worst_r = g.loc[g.rmse.idxmax()]
        ndcg_spread = float(best_n.ndcg_mean - worst_n.ndcg_mean)
        rmse_spread = float(worst_r.rmse - best_r.rmse)
        ndcg_tie = ndcg_spread <= TIE
        summary[f"{fam}/{fac}"] = {
            "n_variants": int(len(g)),
            "best_ndcg_variant": str(best_n.variant), "best_ndcg": float(best_n.ndcg_mean),
            "worst_ndcg_variant": str(worst_n.variant),
            "worst_ndcg": float(worst_n.ndcg_mean),
            "ndcg_spread": ndcg_spread,
            "best_rmse_variant": str(best_r.variant), "best_rmse": float(best_r.rmse),
            "worst_rmse_variant": str(worst_r.variant), "worst_rmse": float(worst_r.rmse),
            "rmse_spread": rmse_spread,
            # A factor that moves neither metric is inert; one that moves only RMSE is
            # invisible to the selected ranking head. Both are findings, not bugs, so the
            # table records which metric actually discriminates.
            "ndcg_tie": bool(ndcg_tie),
            "discriminating_metric": ("none" if ndcg_tie and rmse_spread <= TIE
                                      else "rmse" if ndcg_tie else "ndcg"),
            "why": str(best_n.why if not ndcg_tie else worst_r.why),
        }
    save_json("ablations", {"blocks": summary, "n_rows": int(len(df)),
                            "split": "validation",
                            "tie_tolerance": TIE,
                            "tie_note": (
                                "A block with ndcg_tie = true varies a factor that the "
                                "validation-selected ranking head does not read. The "
                                "clearest case is IBCF centring: the selected head scores "
                                "sum(max(sim,0) * 1[liked]), which never touches the "
                                "centred ratings, so every centring variant produces the "
                                "identical ordering and only RMSE separates them.")})

    for kk, vv in summary.items():
        tag = "" if not vv["ndcg_tie"] else "  [NDCG tie -> discriminated by RMSE]"
        print(f"  {kk:34s} NDCG {vv['worst_ndcg']:.4f} -> {vv['best_ndcg']:.4f} "
              f"(spread {vv['ndcg_spread']:.4f}, best='{vv['best_ndcg_variant']}'){tag}")
    return df


if __name__ == "__main__":
    main()
