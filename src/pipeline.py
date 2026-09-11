"""Shared plumbing: context construction, model specs and results I/O.

A *spec* is a small JSON-serialisable dict that fully determines a model, e.g.

    {"family": "ibcf", "k": 200, "similarity": "cosine", "mean_center": true,
     "center_by": "item", "rank_mode": "affinity"}

Specs are what the sweep stages persist and what later stages re-instantiate, so the
selected configuration travels through the pipeline as data rather than as a hard-coded
constant, and every stage can be rerun independently and reproducibly.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import (BiasBaseline, GlobalMean, ItemMean, Popularity,
                        PopularityPositive, RandomRec, UserMean)
from .config import CONFIG, RESULTS_DIR, TABLES_DIR
from .data import Dataset, Split, load_dataset, make_target_experiment, stratified_split
from .evaluation import EvalContext
from .graph_recommender import GraphRecommender
from .ibcf import IBCF
from .regression_cf import RegressionCF
from .slim import SLIM
from .ubcf import UBCF


# ======================================================================================
# Context
# ======================================================================================
@lru_cache(maxsize=1)
def get_context() -> tuple[Dataset, Split, EvalContext]:
    ds = load_dataset()
    split = stratified_split(ds, CONFIG)
    ctx = EvalContext(ds, split, CONFIG)
    return ds, split, ctx


@lru_cache(maxsize=1)
def get_target():
    ds, split, _ = get_context()
    return make_target_experiment(ds, split, CONFIG)


# ======================================================================================
# Model construction
# ======================================================================================
def build_model(spec: dict):
    """Instantiate a model from its spec dict."""
    s = dict(spec)
    fam = s.pop("family")
    s.pop("_label", None)
    if fam == "global_mean":
        return GlobalMean()
    if fam == "user_mean":
        return UserMean()
    if fam == "item_mean":
        return ItemMean()
    if fam == "bias":
        return BiasBaseline(**s)
    if fam == "popularity":
        return Popularity()
    if fam == "popularity_positive":
        return PopularityPositive(**s)
    if fam == "random":
        return RandomRec(**s)
    if fam == "ubcf":
        return UBCF(**s)
    if fam == "ibcf":
        return IBCF(**s)
    if fam == "regression":
        return RegressionCF(**s)
    if fam == "slim":
        return SLIM(**s)
    if fam == "graph":
        return GraphRecommender(**s)
    raise ValueError(f"unknown model family: {fam!r}")


def fit_model(spec: dict, R_train, calibration=None):
    """Fit a model on R_train and calibrate its rating head on a held-out set.

    `calibration` is the validation DataFrame.  Only the implicit-feedback models
    (SLIM, graph) use it; everybody else ignores the call.  The test split is never
    passed here.
    """
    model = build_model(spec).fit(R_train)
    if calibration is not None and len(calibration):
        model.calibrate(calibration.u.to_numpy(), calibration.i.to_numpy(),
                        calibration.rating.to_numpy())
    return model


# ======================================================================================
# Results I/O
# ======================================================================================
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(name: str, payload) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    return path


def load_json(name: str):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def save_table(name: str, df: pd.DataFrame) -> Path:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv")


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)
