"""
Headless service layer for the demo application.

The Streamlit UI is a thin shell over this class, which means the whole application can
be exercised in a test or by the gate suite without a browser -- the gate for Part Q
calls exactly these methods.
"""
from __future__ import annotations

import pickle
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.base import topk_from_scores
from src.config import ARTIFACTS_DIR, CONFIG, TABLES_DIR
from src.pipeline import build_model, fit_model, get_context, load_json

MODEL_ORDER = ["UBCF", "IBCF", "RegressionCF", "SLIM", "GraphRec",
               "Popularity", "BiasBaseline"]

# Baselines are cheap enough to always fit on the fly.
FALLBACK_SPECS = {
    "Popularity": {"family": "popularity"},
    "BiasBaseline": {"family": "bias", "lam_u": 10.0, "lam_i": 10.0},
}

# The five paradigms are normally loaded from results/artifacts/models/*.pkl, which stage
# 16 writes. That cache is large and regenerable, so it is not version-controlled; when it
# is absent we fall back to refitting from the same validation-selected spec the pipeline
# used, which costs seconds for everything except SLIM.
PARADIGM_RESULT_KEY = {
    "UBCF": "ubcf", "IBCF": "ibcf", "RegressionCF": "regression",
    "SLIM": "slim", "GraphRec": "graph",
}

REASON_PREFIX = {
    "UBCF": "Users most similar to you rated it highly",
    "IBCF": "It is closest to movies you already rated highly",
    "RegressionCF": "Learned regression weights on your rated neighbours",
    "SLIM": "Learned sparse item-item coefficients from your liked movies",
    "GraphRec": "Short paths reach it through the user-movie graph",
    "Popularity": "It is one of the most-rated movies in the catalog",
    "BiasBaseline": "Its item bias plus your user bias is high",
}


class RecommenderService:
    """Loads persisted models (or fits them on demand) and serves recommendations."""

    def __init__(self, models: list[str] | None = None):
        self.ds, self.split, self.ctx = get_context()
        self.titles = self.ds.titles
        self.models: dict[str, object] = {}
        self.specs: dict[str, dict] = {}
        self._load(models or MODEL_ORDER)
        self.metrics = self._load_metrics()

    # ------------------------------------------------------------------ loading
    def _load(self, wanted: list[str]) -> None:
        mdir = ARTIFACTS_DIR / "models"
        for label in wanted:
            path = mdir / f"{label}.pkl"
            if path.exists():
                with open(path, "rb") as fh:
                    blob = pickle.load(fh)
                self.models[label] = blob["model"]
                self.specs[label] = blob["spec"]
            elif label in FALLBACK_SPECS:
                self.models[label] = build_model(FALLBACK_SPECS[label]).fit(
                    self.split.R_train)
                self.specs[label] = FALLBACK_SPECS[label]
            elif label in PARADIGM_RESULT_KEY:
                spec = self._selected_spec(PARADIGM_RESULT_KEY[label])
                if spec is None:
                    continue
                # Same fit path as the pipeline, including the held-out calibration of
                # the implicit models' rating head, so the app cannot disagree with the
                # published metrics.
                self.models[label] = fit_model(spec, self.split.R_train, self.split.val)
                self.specs[label] = spec
        if not self.models:
            raise RuntimeError(
                "no models available -- run `python run_experiments.py` first")

    @staticmethod
    def _selected_spec(key: str) -> dict | None:
        try:
            sel = load_json(key)
        except FileNotFoundError:
            return None
        return (sel["selection"]["ranking_selected"] if "selection" in sel
                else sel.get("ranking_selected"))

    @staticmethod
    def _load_metrics() -> pd.DataFrame:
        path = TABLES_DIR / "multiuser_test.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    # ------------------------------------------------------------------ API
    def available_models(self) -> list[str]:
        return [m for m in MODEL_ORDER if m in self.models]

    def user_ids(self) -> list[int]:
        return [int(x) for x in self.ds.user_ids]

    def _internal(self, user_id: int) -> int:
        idx = int(np.searchsorted(self.ds.user_ids, user_id))
        if idx >= self.ds.n_users or self.ds.user_ids[idx] != user_id:
            raise KeyError(f"unknown user id {user_id}")
        return idx

    def user_profile(self, user_id: int, limit: int | None = None) -> pd.DataFrame:
        """The ratings the models were allowed to see for this user (training data)."""
        u = self._internal(user_id)
        tr = self.split.train
        prof = tr[tr.u == u][["i", "rating", "timestamp"]].copy()
        prof["title"] = self.titles[prof.i.to_numpy()]
        prof["train_popularity"] = self.ctx.item_pop[prof.i.to_numpy()].astype(int)
        prof["popularity_group"] = self.ctx.item_group[prof.i.to_numpy()]
        prof = prof.sort_values(["rating", "train_popularity"], ascending=[False, False])
        return prof.head(limit) if limit else prof

    def user_summary(self, user_id: int) -> dict:
        u = self._internal(user_id)
        prof = self.user_profile(user_id)
        return {
            "user_id": int(user_id), "internal_index": u,
            "n_known_ratings": int(len(prof)),
            "mean_rating": float(prof.rating.mean()) if len(prof) else float("nan"),
            "history_group": str(self.ctx.user_group[u]),
            "n_heldout_test_ratings": int((self.split.test.u == u).sum()),
            "is_roll_target": bool(user_id == self._target_id()),
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def _target_id() -> int:
        try:
            return int(load_json("eda")["target_user"]["target_user_id"])
        except Exception:
            return -1

    def recommend(self, user_id: int, model: str, k: int = 10) -> list[dict]:
        if model not in self.models:
            raise KeyError(f"model {model!r} not loaded")
        u = self._internal(user_id)
        m = self.models[model]
        S = m.score_all()
        topk = topk_from_scores(S, self.split.R_train, k)[u]
        P = np.clip(m.predict_all(), *self.ds.rating_scale)

        out = []
        for rank, i in enumerate(topk, start=1):
            i = int(i)
            out.append({
                "rank": rank, "item": i, "title": str(self.titles[i]),
                "score": float(S[u, i]),
                "predicted_rating": float(P[u, i]),
                "train_popularity": int(self.ctx.item_pop[i]),
                "popularity_group": str(self.ctx.item_group[i]),
                "genres": self._genres(i),
                "explanation": self.explain(user_id, model, i),
            })
        return out

    def _genres(self, i: int) -> str:
        from src.data import GENRES
        g = self.ds.genre_matrix[i]
        names = [GENRES[j] for j in np.nonzero(g)[0] if GENRES[j] != "unknown"]
        return ", ".join(names) if names else "unknown"

    def explain(self, user_id: int, model: str, item: int, top_n: int = 3) -> dict:
        u = self._internal(user_id)
        m = self.models[model]
        raw = []
        if hasattr(m, "explain"):
            try:
                raw = m.explain(u, int(item), top_n)
            except Exception:
                raw = []
        return {"headline": REASON_PREFIX.get(model, "model score"),
                "evidence": self._render(model, raw)}

    def _render(self, model: str, raw: list[dict]) -> list[str]:
        out = []
        for e in raw:
            if model == "UBCF":
                out.append(f"user {e['neighbor_u']} (similarity {e['similarity']:+.3f}, "
                           f"{e['co_ratings']} movies in common) rated it "
                           f"{e['neighbor_rating']:.0f}")
            elif model == "IBCF":
                out.append(f"'{self.titles[e['neighbor_item']]}' - you rated it "
                           f"{e['user_rating']:.0f}, similarity {e['similarity']:+.3f}")
            elif model == "RegressionCF":
                out.append(f"'{self.titles[e['neighbor_item']]}' - learned weight "
                           f"{e['learned_coef']:+.3f} (raw similarity "
                           f"{e['similarity']:+.3f}, contribution "
                           f"{e['contribution']:+.3f})")
            elif model == "SLIM":
                out.append(f"'{self.titles[e['source_item']]}' - learned coefficient "
                           f"{e['coefficient']:.4f}")
            elif model == "GraphRec":
                out.append(f"via '{self.titles[e['via_item']]}' - {e['n_paths']} "
                           f"length-3 paths, weight {e['total_path_weight']:.2e}")
            else:
                out.append(str(e))
        return out

    def model_metrics(self, model: str) -> dict:
        if self.metrics.empty or model not in set(self.metrics.label):
            return {"note": "run the experiment pipeline to populate test metrics"}
        r = self.metrics[self.metrics.label == model].iloc[0]
        return {
            "spec": self.specs.get(model, {}),
            "RMSE": float(r.rmse), "MAE": float(r.mae),
            "Precision@10": float(r.precision_mean), "Recall@10": float(r.recall_mean),
            "HitRate@10": float(r.hit_rate_mean), "NDCG@10": float(r.ndcg_mean),
            "MAP@10": float(r.map_mean),
            "Catalog coverage": float(r.catalog_coverage),
            "Novelty (bits)": float(r.novelty_mean),
            "Intra-list diversity": float(r.ild_mean),
            "Long-tail share": float(r.tail_frac_mean),
            "Training time (s)": float(r.train_time_s),
        }

    def all_model_metrics(self) -> pd.DataFrame:
        if self.metrics.empty:
            return pd.DataFrame()
        cols = ["label", "rmse", "mae", "precision_mean", "recall_mean",
                "hit_rate_mean", "ndcg_mean", "map_mean", "catalog_coverage",
                "novelty_mean", "ild_mean", "tail_frac_mean", "train_time_s"]
        df = self.metrics[[c for c in cols if c in self.metrics.columns]].copy()
        df.columns = ["Model", "RMSE", "MAE", "P@10", "R@10", "HR@10", "NDCG@10",
                      "MAP@10", "Coverage", "Novelty", "Diversity", "LongTail",
                      "TrainSec"][:len(df.columns)]
        return df

    def compare(self, user_id: int, k: int = 10) -> pd.DataFrame:
        """Side-by-side Top-K across every loaded model."""
        cols = {}
        for m in self.available_models():
            cols[m] = [r["title"] for r in self.recommend(user_id, m, k)]
        return pd.DataFrame(cols, index=[f"Rank {i}" for i in range(1, k + 1)])
