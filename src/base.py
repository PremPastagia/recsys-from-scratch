"""Common recommender interface.

Every model in this project exposes the same tiny surface:

    fit(R_train)      -> self          learn everything from the training matrix only
    predict_all()     -> (n_users, n_items) dense array of *rating* predictions
    score_all()       -> (n_users, n_items) dense array of *ranking* scores
    model_bytes()     -> int           in-memory footprint of the learned artifacts

MovieLens 100K is 943 x 1682, so a dense float64 score matrix is ~12.7 MB.  Materialising
it once is both faster and far simpler than per-user recomputation, and it lets the
evaluation harness treat all models identically.  Every model internally still uses
*sparse* algebra for the expensive steps; the dense array is only the output buffer.

Rating prediction vs. ranking score
-----------------------------------
Ranking a candidate set by predicted rating is known to be fragile for neighbourhood
models: an item supported by a single enthusiastic neighbour can reach a predicted 5.0
with essentially no evidence.  We therefore let each model expose a separate *ranking
score* that keeps the evidence mass in the numerator (an unnormalised similarity-weighted
sum), and we select between the two modes on the validation split rather than assuming.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


class Recommender(ABC):
    name: str = "recommender"

    def __init__(self) -> None:
        self.fit_time_s: float = float("nan")
        self._R: sp.csr_matrix | None = None

    # -- lifecycle ---------------------------------------------------------------
    @abstractmethod
    def _fit(self, R: sp.csr_matrix) -> None: ...

    def fit(self, R: sp.csr_matrix) -> "Recommender":
        t0 = time.perf_counter()
        self._R = R.tocsr()
        self._fit(self._R)
        self.fit_time_s = time.perf_counter() - t0
        return self

    # -- inference ---------------------------------------------------------------
    @abstractmethod
    def predict_all(self) -> np.ndarray: ...

    def score_all(self) -> np.ndarray:
        return self.predict_all()

    def model_bytes(self) -> int:
        return 0

    # -- rating-head calibration -------------------------------------------------
    def calibrate(self, u: np.ndarray, i: np.ndarray, r: np.ndarray) -> "Recommender":
        """Fit the score -> rating head on a held-out calibration set.

        Implicit-feedback models (SLIM, the graph recommenders) produce a ranking score,
        not a rating.  Mapping that score onto the 1-5 scale needs a calibration, and
        fitting it on *training* pairs is wrong: for a training pair the user's own
        interaction with the item inflates the score (directly for a random walk), so the
        fitted line is estimated in a regime the model never sees at inference time and
        extrapolates badly onto unseen items.

        Calibrating on the validation split matches the inference condition exactly --
        those items are absent from R_train, just like the items being scored in
        production.  Validation is a legitimate place to fit a calibration; the test split
        is still untouched.  Models that already predict ratings ignore this call.
        """
        return self

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _nbytes(*objs) -> int:
        total = 0
        for o in objs:
            if o is None:
                continue
            if sp.issparse(o):
                total += o.data.nbytes + o.indices.nbytes + o.indptr.nbytes
            elif isinstance(o, np.ndarray):
                total += o.nbytes
        return int(total)


def topk_from_scores(scores: np.ndarray, exclude: sp.csr_matrix, k: int) -> np.ndarray:
    """Top-k item indices per user, excluding every item the user already has in `exclude`.

    `exclude` is always the model's *visible* interaction matrix (training profile), never
    the held-out data -- excluding test items would leak the answer into the candidate set.

    Uses argpartition (O(n_items) per user) rather than a full sort.
    """
    S = scores.copy()
    ex = exclude.tocsr()
    rows = np.repeat(np.arange(ex.shape[0]), np.diff(ex.indptr))
    S[rows, ex.indices] = -np.inf
    S[~np.isfinite(S)] = -np.inf

    n_users, n_items = S.shape
    kk = min(k, n_items)
    part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    part_scores = np.take_along_axis(S, part, axis=1)
    # Deterministic tie-break: higher score first, then smaller item index.  np.lexsort
    # sorts by the LAST key first, so the primary key (-score) is given last.
    order = np.lexsort((part, -part_scores), axis=1)
    return np.take_along_axis(part, order, axis=1)
