"""
Part O -- report figures.

Every figure is built from a saved results table, never from a model held in memory, so
the figures are reproducible from `results/` alone and can never silently disagree with
the numbers in the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

PALETTE = {
    "UBCF": "#4C72B0", "IBCF": "#DD8452", "RegressionCF": "#55A868",
    "SLIM": "#C44E52", "GraphRec": "#8172B3", "Popularity": "#937860",
    "PopularityPositive": "#DA8BC3", "BiasBaseline": "#8C8C8C",
    "ItemMean": "#CCB974", "UserMean": "#64B5CD", "GlobalMean": "#B0B0B0",
    "Random": "#777777",
}
PARADIGMS = ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec"]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.titlesize": 12, "axes.titleweight": "600", "axes.labelsize": 10,
        "legend.frameon": False, "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.3, "axes.edgecolor": "#444444",
    })


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _colors(labels) -> list:
    fallback = sns.color_palette("colorblind", n_colors=max(len(labels), 1))
    return [PALETTE.get(str(l), fallback[i % len(fallback)]) for i, l in enumerate(labels)]


def _bar(ax, labels, values, title, ylabel, fmt="{:.3f}", rotate=30):
    bars = ax.bar(range(len(labels)), values, color=_colors(labels),
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    span = (max(values) - min(min(values), 0)) or 1
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015 * span, fmt.format(v),
                ha="center", va="bottom", fontsize=7.5)
    ax.margins(y=0.16)


# ======================================================================================
# 1-5  Exploratory
# ======================================================================================
def fig01_rating_distribution(rating_dist: pd.DataFrame, overview: dict):
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    total = rating_dist["count"].sum()
    bars = ax.bar(rating_dist.rating, rating_dist["count"], color="#4C72B0",
                  edgecolor="white", width=0.72)
    for b, c in zip(bars, rating_dist["count"]):
        ax.text(b.get_x() + b.get_width() / 2, c + total * 0.008,
                f"{100 * c / total:.1f}%", ha="center", fontsize=9)
    ax.axvline(overview["mean_rating"], color="#C44E52", ls="--", lw=1.5,
               label=f"mean = {overview['mean_rating']:.2f}")
    ax.set_xlabel("rating"); ax.set_ylabel("count")
    ax.set_title(f"Rating distribution ({total:,} ratings, scale "
                 f"{overview['rating_scale_min']:g}-{overview['rating_scale_max']:g})")
    ax.legend()
    return _save(fig, "fig01_rating_distribution")


def fig02_user_activity(per_user: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = per_user.n_ratings
    axes[0].hist(n, bins=45, color="#4C72B0", edgecolor="white")
    axes[0].axvline(n.mean(), color="#C44E52", ls="--",
                    label=f"mean = {n.mean():.0f}")
    axes[0].axvline(n.median(), color="#55A868", ls=":",
                    label=f"median = {n.median():.0f}")
    axes[0].set_xlabel("ratings per user"); axes[0].set_ylabel("number of users")
    axes[0].set_title("User activity distribution"); axes[0].legend()

    s = np.sort(n)[::-1]
    axes[1].plot(np.arange(1, len(s) + 1), s, color="#4C72B0")
    axes[1].set_yscale("log"); axes[1].set_xlabel("user rank (most active first)")
    axes[1].set_ylabel("ratings (log)")
    axes[1].set_title(f"User activity, ranked  (min {s.min()}, max {s.max()})")
    return _save(fig, "fig02_user_activity")


def fig03_item_popularity(per_item: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = per_item.n_ratings
    axes[0].hist(n, bins=60, color="#DD8452", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].axvline(n.mean(), color="#C44E52", ls="--", label=f"mean = {n.mean():.0f}")
    axes[0].set_xlabel("ratings per movie"); axes[0].set_ylabel("number of movies (log)")
    axes[0].set_title("Item popularity distribution"); axes[0].legend()

    s = np.sort(n)[::-1]
    axes[1].loglog(np.arange(1, len(s) + 1), np.maximum(s, 0.5), color="#DD8452")
    axes[1].set_xlabel("movie rank (log)"); axes[1].set_ylabel("ratings (log)")
    axes[1].set_title("Popularity rank-frequency (log-log)")
    return _save(fig, "fig03_item_popularity")


def fig04_long_tail(long_tail: pd.DataFrame, groups: dict):
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = long_tail["rank"]
    ax.plot(x, 100 * long_tail.cum_share, color="#333333", lw=2,
            label="cumulative share of interactions")
    colors = {"head": "#C44E52", "medium": "#DD8452", "long_tail": "#55A868"}
    for g, c in colors.items():
        m = long_tail.group == g
        ax.fill_between(x[m], 0, 100 * long_tail.cum_share[m], color=c, alpha=0.25)
    nh, nm = groups["n_head"], groups["n_medium"]
    ax.axvline(nh, color="#C44E52", ls="--", lw=1.2)
    ax.axvline(nh + nm, color="#DD8452", ls="--", lw=1.2)
    ax.text(nh, 8, f" head: {nh} items\n ({100 * nh / len(long_tail):.1f}% of catalog)",
            fontsize=8.5, color="#C44E52", va="bottom")
    ax.text(nh + nm, 42, f" medium: {nm}", fontsize=8.5, color="#DD8452", va="bottom")
    ax.text(len(long_tail) * 0.55, 78,
            f"long tail: {groups['n_long_tail']} items "
            f"({100 * groups['n_long_tail'] / len(long_tail):.1f}% of catalog,\n"
            f"only {100 - 100 * long_tail.cum_share.iloc[nh + nm - 1]:.1f}% of interactions)",
            fontsize=8.5, color="#2d6a4f")
    ax.set_xlabel("movie rank by training popularity")
    ax.set_ylabel("cumulative % of training interactions")
    ax.set_title("Long-tail curve and equal-interaction-mass popularity groups")
    ax.set_ylim(0, 103); ax.legend(loc="lower right")
    return _save(fig, "fig04_long_tail")


def fig05_sparsity(R, overview: dict, rng_seed: int = 0):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    coo = R.tocoo()
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(coo.row), size=min(30000, len(coo.row)), replace=False)
    axes[0].scatter(coo.col[idx], coo.row[idx], s=0.35, alpha=0.28, color="#4C72B0",
                    linewidths=0)
    axes[0].set_xlabel("movie index"); axes[0].set_ylabel("user index")
    axes[0].invert_yaxis()
    axes[0].set_title(f"Observed cells of R  ({overview['n_users']}x{overview['n_items']}, "
                      f"{overview['sparsity_pct']:.2f}% sparse)")
    axes[0].grid(False)

    order_u = np.argsort(-np.asarray(R.getnnz(axis=1)).ravel())
    order_i = np.argsort(-np.asarray(R.getnnz(axis=0)).ravel())
    sub = R[order_u][:, order_i][:300, :400].toarray()
    axes[1].imshow(sub > 0, cmap="Blues", aspect="auto", interpolation="nearest")
    axes[1].set_xlabel("movie (sorted by popularity)")
    axes[1].set_ylabel("user (sorted by activity)")
    axes[1].set_title("Top 300 users x top 400 movies, sorted\n(the dense core of a sparse matrix)")
    axes[1].grid(False)
    return _save(fig, "fig05_sparsity")


# ======================================================================================
# 6-7  Neighbourhood sweeps
# ======================================================================================
def _best_per_k(df: pd.DataFrame, group_cols: list[str], metric: str, maximise: bool):
    idx = (df.groupby(group_cols + ["k"])[metric].idxmax() if maximise
           else df.groupby(group_cols + ["k"])[metric].idxmin())
    return df.loc[idx]


def fig06_k_vs_error(ubcf: pd.DataFrame, ibcf: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for col, metric, name in ((0, "rmse", "RMSE"), (1, "mae", "MAE")):
        for r, (df, fam) in enumerate(((ubcf, "UBCF"), (ibcf, "IBCF"))):
            ax = axes[r][col]
            keys = ["similarity", "mean_center"] + (
                ["center_by"] if "center_by" in df.columns else [])
            best = _best_per_k(df, keys, metric, maximise=False)
            for kv, g in best.groupby(keys):
                kv = kv if isinstance(kv, tuple) else (kv,)
                lab = f"{kv[0]}, {'centred' if kv[1] else 'raw'}"
                if len(kv) > 2 and kv[1]:
                    lab += f"/{kv[2]}"
                g = g.sort_values("k")
                ax.plot(g.k, g[metric], marker="o", ms=4, lw=1.6, label=lab)
            ax.set_xscale("log")
            ax.set_title(f"{fam}: k vs {name} (validation)")
            ax.set_ylabel(name)
            if r == 1:
                ax.set_xlabel("neighbourhood size k")
            ax.legend(fontsize=7.5, ncol=1)
    return _save(fig, "fig06_k_vs_rmse_mae")


def fig07_k_vs_ranking(ubcf: pd.DataFrame, ibcf: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for col, metric, name in ((0, "ndcg_mean", "NDCG@10"),
                              (1, "recall_mean", "Recall@10")):
        for r, (df, fam) in enumerate(((ubcf, "UBCF"), (ibcf, "IBCF"))):
            ax = axes[r][col]
            keys = ["similarity", "mean_center"] + (
                ["center_by"] if "center_by" in df.columns else [])
            best = _best_per_k(df, keys, metric, maximise=True)
            for kv, g in best.groupby(keys):
                kv = kv if isinstance(kv, tuple) else (kv,)
                lab = f"{kv[0]}, {'centred' if kv[1] else 'raw'}"
                if len(kv) > 2 and kv[1]:
                    lab += f"/{kv[2]}"
                g = g.sort_values("k")
                ax.plot(g.k, g[metric], marker="o", ms=4, lw=1.6, label=lab)
            ax.set_xscale("log")
            ax.set_title(f"{fam}: k vs {name} (validation)")
            ax.set_ylabel(name)
            if r == 1:
                ax.set_xlabel("neighbourhood size k")
            ax.legend(fontsize=7.5)
    return _save(fig, "fig07_k_vs_ndcg_recall")


# ======================================================================================
# 8-12  Model comparison
# ======================================================================================
def fig08_ranking_comparison(mu: pd.DataFrame):
    _style()
    metrics = [("precision_mean", "Precision@10"), ("recall_mean", "Recall@10"),
               ("hit_rate_mean", "HitRate@10"), ("ndcg_mean", "NDCG@10"),
               ("map_mean", "MAP@10"), ("rmse", "RMSE (lower better)")]
    df = mu.sort_values("ndcg_mean", ascending=False)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (col, name) in zip(axes.ravel(), metrics):
        _bar(ax, df.label.tolist(), df[col].tolist(), name, name,
             fmt="{:.3f}" if col != "rmse" else "{:.3f}")
    fig.suptitle("Model comparison on rating and ranking metrics (test split)",
                 y=1.01, fontsize=13, weight="600")
    fig.tight_layout()
    return _save(fig, "fig08_ranking_comparison")


def fig09_coverage_comparison(mu: pd.DataFrame):
    _style()
    df = mu.sort_values("catalog_coverage", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    _bar(axes[0], df.label.tolist(), (100 * df.catalog_coverage).tolist(),
         "Catalog coverage: % of the 1,682 movies that ever appear in a Top-10",
         "% of catalog", fmt="{:.1f}")
    _bar(axes[1], df.label.tolist(), (100 * df.user_coverage).tolist(),
         "User coverage: % of users given a genuinely ranked (non-tied) list",
         "% of users", fmt="{:.1f}")
    fig.tight_layout()
    return _save(fig, "fig09_coverage_comparison")


def fig10_novelty_diversity(mu: pd.DataFrame):
    _style()
    df = mu.sort_values("novelty_mean", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _bar(axes[0], df.label.tolist(), df.novelty_mean.tolist(),
         "Novelty (mean self-information, bits)", "bits", fmt="{:.2f}")
    _bar(axes[1], df.label.tolist(), df.ild_mean.tolist(),
         "Intra-list diversity (1 - mean genre cosine)", "ILD", fmt="{:.3f}")
    ax = axes[2]
    for _, r in mu.iterrows():
        ax.scatter(r.novelty_mean, r.ndcg_mean, s=85,
                   color=_colors([r.label])[0], edgecolor="white", zorder=3)
        ax.annotate(r.label, (r.novelty_mean, r.ndcg_mean), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("novelty (bits)"); ax.set_ylabel("NDCG@10")
    ax.set_title("The accuracy / novelty frontier")
    fig.tight_layout()
    return _save(fig, "fig10_novelty_diversity")


def fig11_popularity_bias(pb: pd.DataFrame):
    _style()
    df = pb.sort_values("mean_train_popularity", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    _bar(axes[0], df.model.tolist(), df.mean_train_popularity.tolist(),
         "Mean training popularity of recommended items", "ratings in train",
         fmt="{:.0f}")
    _bar(axes[1], df.model.tolist(), df.gini_exposure.tolist(),
         "Gini of exposure across the catalog\n(0 = uniform, 1 = winner-take-all)",
         "Gini", fmt="{:.3f}")
    ax = axes[2]
    for _, r in pb.iterrows():
        ax.scatter(r.mean_train_popularity, r.ndcg_mean, s=85,
                   color=_colors([r.model])[0], edgecolor="white", zorder=3)
        ax.annotate(r.model, (r.mean_train_popularity, r.ndcg_mean), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean training popularity of Top-10")
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 vs recommendation popularity")
    fig.tight_layout()
    return _save(fig, "fig11_popularity_bias")


def fig12_long_tail_exposure(pb: pd.DataFrame):
    _style()
    df = pb.sort_values("long_tail_pct", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax = axes[0]
    bottom = np.zeros(len(df))
    for col, c, lab in (("head_pct", "#C44E52", "head"),
                        ("medium_pct", "#DD8452", "medium"),
                        ("long_tail_pct", "#55A868", "long tail")):
        ax.bar(range(len(df)), df[col], bottom=bottom, color=c, label=lab,
               edgecolor="white", linewidth=0.6)
        bottom += df[col].to_numpy()
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.model, rotation=30, ha="right")
    ax.set_ylabel("% of recommended slots")
    ax.set_title("Where recommendations come from")
    ax.legend(ncol=3, fontsize=8)

    _bar(axes[1], df.model.tolist(), df.long_tail_pct.tolist(),
         "Long-tail exposure", "% of Top-10 slots", fmt="{:.1f}")

    ax = axes[2]
    for _, r in pb.iterrows():
        ax.scatter(r.long_tail_pct, r.ndcg_mean, s=85,
                   color=_colors([r.model])[0], edgecolor="white", zorder=3)
        ax.annotate(r.model, (r.long_tail_pct, r.ndcg_mean), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("% of recommendations from the long tail")
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 vs long-tail exposure")
    fig.tight_layout()
    return _save(fig, "fig12_long_tail_exposure")


# ======================================================================================
# 13-15  Model internals
# ======================================================================================
def fig13_regression_coefficients(sc: pd.DataFrame, summary: dict):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.scatter(sc.similarity, sc.coefficient, s=3, alpha=0.16, color="#4C72B0",
               linewidths=0)
    lim = np.percentile(np.abs(sc.coefficient), 99.5)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axvline(0, color="#888888", lw=0.8)
    z = np.polyfit(sc.similarity, sc.coefficient, 1)
    xs = np.linspace(sc.similarity.min(), sc.similarity.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#C44E52", lw=1.8,
            label=f"OLS fit  (Pearson r = {summary['pearson_sim_coef']:+.3f},\n"
                  f"Spearman rho = {summary['spearman_sim_coef']:+.3f})")
    ax.set_xlabel("item-item similarity (adjusted cosine)")
    ax.set_ylabel("learned ridge coefficient")
    ax.set_title("Similarity vs learned coefficient")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(sc.coefficient, bins=120, color="#55A868", edgecolor="none")
    ax.set_xlim(-lim, lim); ax.set_yscale("log")
    ax.axvline(0, color="#C44E52", lw=1.2)
    ax.set_xlabel("learned coefficient"); ax.set_ylabel("count (log)")
    ax.set_title(f"Coefficient distribution\n"
                 f"({100 * summary['frac_negative_coefficients']:.1f}% negative, "
                 f"{100 * summary['frac_sign_flip_positive_sim_negative_coef']:.1f}% sign flips)")

    ax = axes[2]
    pos = sc[sc.similarity > 0.05]
    ax.hexbin(pos.similarity, pos.coefficient, gridsize=45, cmap="Blues", mincnt=1,
              bins="log")
    ax.axhline(0, color="#C44E52", lw=1.2)
    ax.set_xlabel("similarity (> 0.05 only)"); ax.set_ylabel("learned coefficient")
    ax.set_ylim(-lim, lim)
    ax.set_title("Where the regression overrides similarity")
    fig.tight_layout()
    return _save(fig, "fig13_regression_sim_vs_coef")


def fig14_slim_sparsity_performance(sl: pd.DataFrame):
    _style()
    reg = sl[sl.tag.isin(["l1_sweep", "l2_sweep", "elasticnet_mix"])]
    fm = sl[sl.tag == "feature_mask"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    for tag, g in reg.groupby("tag"):
        g = g.sort_values("sparsity_pct")
        ax.plot(g.sparsity_pct, g.ndcg_mean, marker="o", ms=5, lw=1.6, label=tag)
    ax.set_xlabel("model sparsity (% zero off-diagonal entries of W)")
    ax.set_ylabel("NDCG@10 (validation)")
    ax.set_title("Sparsity vs recommendation quality")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(reg.sparsity_pct, reg.fit_time_s, s=55, color="#C44E52",
               edgecolor="white", label="regularisation sweep", zorder=3)
    if len(fm):
        ax.scatter(fm.sparsity_pct, fm.fit_time_s, s=55, color="#4C72B0",
                   marker="s", edgecolor="white", label="feature restriction", zorder=3)
    ax.set_xlabel("model sparsity (%)"); ax.set_ylabel("training time (s)")
    ax.set_title("Sparsity vs runtime"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(reg.sparsity_pct, 100 * reg.catalog_coverage, s=55, color="#55A868",
               edgecolor="white", zorder=3)
    ax.set_xlabel("model sparsity (%)"); ax.set_ylabel("catalog coverage (%)")
    ax.set_title("Sparsity vs catalog reach")
    fig.tight_layout()
    return _save(fig, "fig14_slim_sparsity_vs_performance")


def fig15_slim_regularisation(sl: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    l1 = sl[sl.tag == "l1_sweep"].sort_values("l1")
    l2 = sl[sl.tag == "l2_sweep"].sort_values("l2")
    mix = sl[sl.tag == "elasticnet_mix"].sort_values("l1_ratio")

    ax = axes[0]
    ax.plot(l1.l1, l1.sparsity_pct, marker="o", color="#C44E52", label="sparsity %")
    ax.set_xscale("log"); ax.set_xlabel(r"$\lambda_1$ (L1 strength)")
    ax.set_ylabel("sparsity of W (%)", color="#C44E52")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(l1.l1, l1.ndcg_mean, marker="s", color="#4C72B0")
    ax2.set_ylabel("NDCG@10", color="#4C72B0")
    ax.set_title(r"L1 strength vs sparsity and NDCG@10")

    ax = axes[1]
    ax.plot(np.maximum(l2.l2, 0.3), l2.sparsity_pct, marker="o", color="#C44E52")
    ax.set_xscale("log"); ax.set_xlabel(r"$\lambda_2$ (L2 strength, 0 plotted at 0.3)")
    ax.set_ylabel("sparsity of W (%)", color="#C44E52")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(np.maximum(l2.l2, 0.3), l2.ndcg_mean, marker="s", color="#4C72B0")
    ax2.set_ylabel("NDCG@10", color="#4C72B0")
    ax.set_title(r"L2 strength vs sparsity and NDCG@10")

    ax = axes[2]
    ax.plot(mix.l1_ratio, mix.sparsity_pct, marker="o", color="#C44E52")
    ax.set_xlabel(r"elastic-net mixing $\rho=\lambda_1/(\lambda_1+\lambda_2)$")
    ax.set_ylabel("sparsity of W (%)", color="#C44E52")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(mix.l1_ratio, mix.ndcg_mean, marker="s", color="#4C72B0")
    ax2.set_ylabel("NDCG@10", color="#4C72B0")
    ax.set_title("Elastic-net mixing at fixed total penalty")
    fig.tight_layout()
    return _save(fig, "fig15_slim_regularisation")


# ======================================================================================
# 16-17
# ======================================================================================
def fig16_runtime(mu: pd.DataFrame):
    _style()
    df = mu.sort_values("train_time_s")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    _bar(axes[0], df.label.tolist(), np.maximum(df.train_time_s, 1e-4).tolist(),
         "Training time", "seconds (log)", fmt="{:.3f}")
    axes[0].set_yscale("log")
    _bar(axes[1], df.label.tolist(),
         np.maximum(df.predict_time_per_user_ms, 1e-4).tolist(),
         "Full-catalog scoring time per user", "milliseconds (log)", fmt="{:.2f}")
    axes[1].set_yscale("log")
    _bar(axes[2], df.label.tolist(), np.maximum(df.model_mb, 1e-6).tolist(),
         "Learned-model footprint", "MB (log)", fmt="{:.2f}")
    axes[2].set_yscale("log")
    fig.tight_layout()
    return _save(fig, "fig16_runtime")


def fig17_coldstart(ug: pd.DataFrame, ig: pd.DataFrame, models: list[str]):
    _style()
    order = ["sparse", "medium", "heavy"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    piv = ug[ug.model.isin(models)].pivot(index="user_group", columns="model",
                                          values="ndcg").loc[order]
    piv[models].plot(kind="bar", ax=ax, color=_colors(models), edgecolor="white",
                     width=0.78, legend=False)
    ax.set_xlabel("user-history tertile"); ax.set_ylabel("NDCG@10")
    ax.set_title("Ranking quality by user history length")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[1]
    piv = ug[ug.model.isin(models)].pivot(index="user_group", columns="model",
                                          values="rmse").loc[order]
    piv[models].plot(kind="bar", ax=ax, color=_colors(models), edgecolor="white",
                     width=0.78, legend=False)
    ax.set_xlabel("user-history tertile"); ax.set_ylabel("RMSE (lower better)")
    ax.set_title("Rating error by user history length")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[2]
    iorder = ["head", "medium", "long_tail"]
    piv = ig[ig.model.isin(models)].pivot(index="item_group", columns="model",
                                          values="recall").loc[iorder]
    piv[models].plot(kind="bar", ax=ax, color=_colors(models), edgecolor="white",
                     width=0.78, legend=False)
    ax.set_xlabel("item popularity group of the held-out positive")
    ax.set_ylabel("Recall@10")
    ax.set_title("Retrieval by item popularity group")
    ax.tick_params(axis="x", rotation=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _colors(models)]
    fig.legend(handles, models, ncol=len(models), loc="lower center",
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    return _save(fig, "fig17_coldstart")
