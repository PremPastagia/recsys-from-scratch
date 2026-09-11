"""
Part A: data loading, the user-item rating matrix R, popularity groups and the
leakage-free train / validation / test protocol.

Design notes
------------
*Why a random per-user stratified split rather than a temporal one?*
    MovieLens 100K is a *rating-prediction* / missing-value benchmark: ratings were
    collected in a single session-driven campaign and the timestamps reflect when a
    user happened to sit down, not a genuine sequential consumption order. The
    classical evaluation for the five paradigms compared here (neighbourhood CF,
    regression CF, SLIM, graph propagation) is the random-holdout protocol, so we use
    a per-user stratified random split. This keeps every user present in train, which
    is required for user-based CF to have a profile to work from at all. The temporal
    alternative answers a different (next-item) question and is listed in the
    limitations section of the report.

*Why stratify per user?*
    A globally random split would hand heavy users almost all of the test mass and
    could leave light users with zero test ratings, making per-user averaged ranking
    metrics unstable and biased toward heavy users. Per-user stratification fixes the
    holdout *proportion* for everybody.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import CONFIG, ML100K_DIR, ExperimentConfig, target_user_id

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ======================================================================================
# Raw loading
# ======================================================================================
def load_ratings(data_dir=ML100K_DIR) -> pd.DataFrame:
    """Load u.data -> DataFrame[user_id, item_id, rating, timestamp]."""
    df = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "item_id": np.int32,
               "rating": np.float64, "timestamp": np.int64},
        engine="c",
    )
    return df


def load_items(data_dir=ML100K_DIR) -> pd.DataFrame:
    """Load u.item -> DataFrame[item_id, title, release_date, <19 genre flags>]."""
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    df = pd.read_csv(
        data_dir / "u.item", sep="|", names=cols, encoding="latin-1", engine="c",
    )
    return df.drop(columns=["video_release_date"])


def load_users(data_dir=ML100K_DIR) -> pd.DataFrame:
    return pd.read_csv(
        data_dir / "u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"], encoding="latin-1",
    )


# ======================================================================================
# Dataset container
# ======================================================================================
@dataclass
class Dataset:
    """Everything downstream code needs, with contiguous 0-based internal indices.

    MovieLens ids are 1..n and dense, so the internal index is simply ``id - 1``; we
    still keep the maps explicit so nothing silently depends on that coincidence.
    """

    ratings: pd.DataFrame          # user_id, item_id, rating, timestamp, u, i
    items: pd.DataFrame
    users: pd.DataFrame
    n_users: int
    n_items: int
    user_ids: np.ndarray           # internal index -> external id
    item_ids: np.ndarray
    genre_matrix: np.ndarray       # (n_items, 19) binary content features
    titles: np.ndarray             # internal item index -> title

    @property
    def rating_scale(self) -> tuple[float, float]:
        return float(self.ratings.rating.min()), float(self.ratings.rating.max())

    def item_title(self, i: int) -> str:
        return str(self.titles[i])


def load_dataset(data_dir=ML100K_DIR) -> Dataset:
    ratings = load_ratings(data_dir)
    items = load_items(data_dir)
    users = load_users(data_dir)

    user_ids = np.sort(users.user_id.to_numpy())
    item_ids = np.sort(items.item_id.to_numpy())
    u_index = {uid: k for k, uid in enumerate(user_ids)}
    i_index = {iid: k for k, iid in enumerate(item_ids)}

    ratings = ratings.assign(
        u=ratings.user_id.map(u_index).astype(np.int32),
        i=ratings.item_id.map(i_index).astype(np.int32),
    )
    assert ratings.u.notna().all() and ratings.i.notna().all()

    items = items.set_index("item_id").loc[item_ids].reset_index()
    genre_matrix = items[GENRES].to_numpy(dtype=np.float64)
    titles = items.title.to_numpy()

    return Dataset(
        ratings=ratings, items=items, users=users,
        n_users=len(user_ids), n_items=len(item_ids),
        user_ids=user_ids, item_ids=item_ids,
        genre_matrix=genre_matrix, titles=titles,
    )


# ======================================================================================
# Rating matrix
# ======================================================================================
def build_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Build the sparse user-item rating matrix R (CSR).

    R[u, i] = r_ui for observed ratings; *structurally absent* otherwise.  Storing
    missing ratings as structural zeros (rather than dense zeros) is what keeps every
    later algorithm honest: a zero can never be mistaken for "rated 0".
    """
    R = sp.csr_matrix(
        (df.rating.to_numpy(np.float64), (df.u.to_numpy(), df.i.to_numpy())),
        shape=(n_users, n_items),
    )
    R.sum_duplicates()
    return R


def binarize(R: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    """Implicit-feedback conversion: X[u,i] = 1 iff r_ui >= threshold."""
    X = R.copy().tocoo()
    keep = X.data >= threshold
    X = sp.csr_matrix(
        (np.ones(keep.sum()), (X.row[keep], X.col[keep])), shape=R.shape
    )
    X.sum_duplicates()
    return X


def sparsity_stats(R: sp.csr_matrix) -> dict:
    n_users, n_items = R.shape
    cells = n_users * n_items
    nnz = int(R.nnz)
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "matrix_cells": int(cells),
        "observed_entries": nnz,
        "density_pct": 100.0 * nnz / cells,
        "sparsity_pct": 100.0 * (1.0 - nnz / cells),
    }


# ======================================================================================
# Splitting
# ======================================================================================
@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    R_train: sp.csr_matrix
    R_val: sp.csr_matrix
    R_test: sp.csr_matrix
    seed: int

    def summary(self) -> dict:
        return {
            "n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test),
            "train_pct": 100 * len(self.train) / (len(self.train) + len(self.val) + len(self.test)),
            "val_pct": 100 * len(self.val) / (len(self.train) + len(self.val) + len(self.test)),
            "test_pct": 100 * len(self.test) / (len(self.train) + len(self.val) + len(self.test)),
            "seed": self.seed,
        }


def _user_rng(seed: int, u: int) -> np.random.Generator:
    """Independent, order-invariant RNG stream per user.

    Deriving each user's stream from a hash of (seed, user) rather than from a single
    sequentially-consumed global generator means the split for user u does not depend
    on how many users were processed before it, so the split is stable under any
    reordering or parallelisation of the loop.
    """
    h = hashlib.sha256(f"{seed}:{u}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def stratified_split(ds: Dataset, cfg: ExperimentConfig = CONFIG) -> Split:
    """Per-user stratified random split into train / validation / test.

    Guarantees
    ----------
    * The three parts are a *partition*: disjoint and covering all 100,000 ratings.
    * Every user keeps at least one training rating (so no user is cold by construction).
    * Fully determined by ``cfg.seed`` and the user id.
    """
    df = ds.ratings
    test_idx, val_idx = [], []

    for u, grp in df.groupby("u", sort=True):
        idx = grp.index.to_numpy()
        rng = _user_rng(cfg.seed, int(u))
        perm = rng.permutation(len(idx))
        idx = idx[perm]

        n = len(idx)
        n_test = int(np.floor(n * cfg.test_frac))
        n_val = int(np.floor(n * cfg.val_frac))
        # Never let holdout starve a user's profile.
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - n_test - 1))

        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test:n_test + n_val])

    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    held = np.concatenate([test_idx, val_idx])

    mask = np.ones(len(df), dtype=bool)
    mask[df.index.get_indexer(held)] = False

    train = df.loc[mask].copy()
    val = df.loc[val_idx].copy()
    test = df.loc[test_idx].copy()

    return Split(
        train=train, val=val, test=test,
        R_train=build_matrix(train, ds.n_users, ds.n_items),
        R_val=build_matrix(val, ds.n_users, ds.n_items),
        R_test=build_matrix(test, ds.n_users, ds.n_items),
        seed=cfg.seed,
    )


# ======================================================================================
# Target-user experiment (Part A.8 / Part M)
# ======================================================================================
@dataclass
class TargetUserExperiment:
    user_id: int          # external MovieLens id
    u: int                # internal index
    roll: str
    mapping_explanation: str
    kept: pd.DataFrame    # visible profile
    hidden: pd.DataFrame  # held-out ground truth
    R_profile: sp.csr_matrix  # full training matrix with the target's hidden cells removed


def make_target_experiment(ds: Dataset, split: Split,
                           cfg: ExperimentConfig = CONFIG) -> TargetUserExperiment:
    """Hide a reproducible subset of the target user's *training* ratings.

    The target user's global test ratings are already hidden from every model.  For the
    course-required single-user experiment we additionally hide ``target_hide_frac`` of
    what remains, so the target has a genuinely reduced profile while every other user's
    training data is untouched.  The recommender therefore sees
    ``R_profile`` = R_train with the target's hidden cells zeroed out, and is scored on
    those hidden cells plus the target's global test ratings.
    """
    uid = target_user_id(ds.n_users, cfg.roll)
    u = int(np.searchsorted(ds.user_ids, uid))

    prof = split.train[split.train.u == u]
    rng = _user_rng(cfg.seed ^ 0x5EED, u)
    perm = rng.permutation(len(prof))
    n_hide = int(round(len(prof) * cfg.target_hide_frac))
    n_hide = min(n_hide, max(0, len(prof) - 5))  # always leave a usable profile

    hidden = prof.iloc[perm[:n_hide]].copy()
    kept = prof.iloc[perm[n_hide:]].copy()

    R_profile = split.R_train.tolil(copy=True)
    for i in hidden.i.to_numpy():
        R_profile[u, int(i)] = 0.0
    R_profile = R_profile.tocsr()
    R_profile.eliminate_zeros()

    expl = (
        f"roll='{cfg.roll}' -> digits='{''.join(c for c in cfg.roll if c.isdigit())}' "
        f"-> {int(''.join(c for c in cfg.roll if c.isdigit()))} mod {ds.n_users} = "
        f"{int(''.join(c for c in cfg.roll if c.isdigit())) % ds.n_users} "
        f"-> +1 -> user_id {uid}"
    )
    return TargetUserExperiment(
        user_id=int(uid), u=u, roll=cfg.roll, mapping_explanation=expl,
        kept=kept, hidden=hidden, R_profile=R_profile,
    )


# ======================================================================================
# Popularity groups (Part A.6)
# ======================================================================================
def popularity_groups(R_train: sp.csr_matrix,
                      cuts: tuple[float, float] = CONFIG.popularity_mass_cuts) -> dict:
    """Partition the catalog into head / medium / long-tail by *equal interaction mass*.

    Rationale for the threshold choice
    ----------------------------------
    A raw count cutoff ("head = items with >= 100 ratings") is arbitrary and does not
    transfer between datasets.  Instead we sort items by training popularity and cut
    the cumulative interaction curve at 1/3 and 2/3 of total interaction *mass*:

        head   = the smallest set of items that absorbs the first 33.3% of all ratings
        medium = the items absorbing the next 33.3%
        tail   = everything else (including items with zero training ratings)

    This is the exposure-weighted reading of the long tail: each group is equally
    important to the *observed* traffic, so "long tail" literally means "the many items
    that together only receive as much attention as the few head items".  It is
    scale-free and reproduces the classic Pareto picture without hand-set constants.
    """
    counts = np.asarray(R_train.getnnz(axis=0)).ravel().astype(np.int64)
    order = np.argsort(-counts, kind="stable")
    sorted_counts = counts[order]
    total = sorted_counts.sum()
    cum = np.cumsum(sorted_counts) / total

    c1, c2 = cuts
    n_head = int(np.searchsorted(cum, c1) + 1)
    n_med = int(np.searchsorted(cum, c2) + 1) - n_head

    group = np.empty(len(counts), dtype=object)
    group[order[:n_head]] = "head"
    group[order[n_head:n_head + n_med]] = "medium"
    group[order[n_head + n_med:]] = "long_tail"

    return {
        "counts": counts,
        "group": group,
        "order": order,
        "cum_mass": cum,
        "n_head": n_head,
        "n_medium": n_med,
        "n_long_tail": int(len(counts) - n_head - n_med),
        "head_min_count": int(sorted_counts[n_head - 1]),
        "medium_min_count": int(sorted_counts[n_head + n_med - 1]),
        "cuts": list(cuts),
        "total_interactions": int(total),
    }


def user_history_groups(R_train: sp.csr_matrix) -> dict:
    """Tertile split of users by training-history length (Part I cold-start groups)."""
    hist = np.asarray(R_train.getnnz(axis=1)).ravel().astype(np.int64)
    q1, q2 = np.quantile(hist, [1 / 3, 2 / 3])
    group = np.where(hist <= q1, "sparse", np.where(hist <= q2, "medium", "heavy"))
    return {"history": hist, "group": group, "q1": float(q1), "q2": float(q2)}
