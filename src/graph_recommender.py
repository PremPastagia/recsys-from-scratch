"""
Part G: Network-based recommendation on the user-movie bipartite graph.

Graph
-----
    V = U u I            943 user nodes + 1682 movie nodes
    E = {(u,i) : u interacted with i}      (implicit edges, r >= tau)

The adjacency is block-antidiagonal, which is the whole point: there are no user-user or
item-item edges, so *every* path from a user to an item has odd length, and the shortest
informative one has length 3:

    u --rated--> i' <--rated-- v --rated--> i
    "people who liked what you liked also liked i"

That length-3 path is literally the collaborative-filtering signal; the graph view
generalises it to arbitrary depth and lets us dial how far the evidence may travel.

--------------------------------------------------------------------------------------
1. Bounded-length path proximity  (Katz-style)
--------------------------------------------------------------------------------------
    score(u, i) = sum_{l odd, l <= L}  beta^l  (A^l)_{u,i}

``(A^l)_{u,i}`` counts walks of length l from u to i; ``beta in (0,1)`` is the
*attenuation*: each extra hop multiplies a walk's contribution by beta, so long
(weak, indirect) evidence is discounted geometrically.  Computed by repeated sparse
matrix products, never by forming A^l.

Degree normalisation (``normalize="degree"``) replaces A by the random-walk-normalised
adjacency D^-1 A.  Un-normalised path counting is *structurally* popularity-biased: a
blockbuster sits on thousands of walks purely because it has a huge degree.  Dividing by
degree makes each node spread a fixed unit of evidence over its neighbours, which is the
single most effective popularity de-bias available inside the graph formulation.

--------------------------------------------------------------------------------------
2. Random walk with restart (personalised PageRank)
--------------------------------------------------------------------------------------
A surfer starts at u and at each step either teleports back to u (probability alpha) or
follows a uniformly random incident edge (probability 1 - alpha):

    p_{t+1} = (1 - alpha) p_t P  +  alpha e_u ,     P = D^{-1} A  (row-stochastic)

The limit p* is the stationary personalised-PageRank vector; its item entries are the
recommendation scores.  ``alpha`` is the restart / attenuation strength and is the knob
that controls locality:

    alpha -> 1   the walk barely leaves u        -> very local, low coverage, high novelty
    alpha -> 0   the walk forgets where it began -> p* converges to the *global* stationary
                 distribution, which on an undirected graph is exactly proportional to
                 node degree.  In recommender terms: pure popularity.

So the popularity bias of a graph recommender is not an accident of the data -- it is the
limiting behaviour of the operator itself, and ``deeper propagation provably means more
popular recommendations``.  ``depth_popularity_curve`` measures exactly that.

Dangling nodes (items with no training edges) would leak probability mass; the leaked
mass is returned to the restart vector each step, so the iterate stays an exact
probability distribution -- a property the verifier checks.

Complexity
----------
    per power-iteration step : 2 sparse GEMMs, O(|U| * nnz) = 943 * 55k ~ 5e7
    memory                   : dense |U| x |I| + |U| x |U| score buffers (~20 MB)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


def _row_normalize(M: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Row-stochastic version of M plus the row-sum vector (0 marks a dangling node)."""
    M = M.tocsr().astype(np.float64)
    deg = np.asarray(M.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    return sp.diags(inv) @ M, deg


class GraphRecommender(Recommender):
    """Bipartite-graph recommender: ``method`` selects RWR or bounded path proximity."""

    def __init__(self, method: str = "rwr", alpha: float = 0.15, n_iter: int = 200,
                 tol: float = 1e-9, max_path_len: int = 3, beta: float = 0.5,
                 normalize: str = "degree", threshold: float = 4.0,
                 use_implicit: bool = True):
        super().__init__()
        self.method = method
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.max_path_len = int(max_path_len)
        self.beta = float(beta)
        self.normalize = normalize
        self.threshold = float(threshold)
        self.use_implicit = use_implicit
        self.name = (f"Graph-RWR[a={alpha:g}]" if method == "rwr"
                     else f"Graph-Path[L={max_path_len},b={beta:g},{normalize}]")

    # ---------------------------------------------------------------- fit
    def _fit(self, R: sp.csr_matrix) -> None:
        if self.use_implicit:
            coo = R.tocoo()
            keep = coo.data >= self.threshold
            B = sp.csr_matrix((np.ones(int(keep.sum())), (coo.row[keep], coo.col[keep])),
                              shape=R.shape)
        else:
            B = R.copy()
            B.data = np.ones_like(B.data)
        self.B = B.tocsr()

        self.B_T = self.B.T.tocsr()
        self.P_ui, self.deg_u = _row_normalize(self.B)        # user -> item
        self.P_iu, self.deg_i = _row_normalize(self.B_T)      # item -> user

        self._scores = None
        self.diagnostics: dict = {}
        self._fit_rating_head(R)

    # ---------------------------------------------------------------- propagation
    def _rwr_scores(self) -> np.ndarray:
        """All-user personalised PageRank by simultaneous power iteration."""
        n_users, n_items = self.B.shape
        a = self.alpha
        restart = np.eye(n_users)                 # e_u for every seed user u (rows)
        pu = restart.copy()                        # mass on user nodes
        pi = np.zeros((n_users, n_items))          # mass on item nodes

        deltas = []
        for _ in range(self.n_iter):
            new_pi = (1 - a) * (pu @ self.P_ui)
            new_pu = (1 - a) * (pi @ self.P_iu)
            # Everything the (1-alpha)-scaled transition does not carry forward -- the
            # alpha teleport share *and* any mass stranded on a dangling node -- is
            # returned to the seed, so the iterate stays an exact probability
            # distribution (sum = 1) at every step.
            leaked = 1.0 - (new_pu.sum(axis=1) + new_pi.sum(axis=1))
            new_pu = new_pu + np.maximum(leaked, 0.0)[:, None] * restart

            delta = np.abs(new_pi - pi).sum(axis=1).max()
            pu, pi = new_pu, new_pi
            deltas.append(float(delta))
            if delta < self.tol:
                break

        self.diagnostics = {
            "iterations": len(deltas),
            "final_delta": deltas[-1],
            "mass_min": float((pu.sum(axis=1) + pi.sum(axis=1)).min()),
            "mass_max": float((pu.sum(axis=1) + pi.sum(axis=1)).max()),
            "delta_trace": deltas,
        }
        self.p_user, self.p_item = pu, pi
        return pi

    def _path_scores(self) -> np.ndarray:
        """sum over odd path lengths l <= L of beta^l * (A^l)_{u,i}."""
        n_users, n_items = self.B.shape
        if self.normalize == "degree":
            A_ui, A_iu = self.P_ui, self.P_iu
        else:
            A_ui, A_iu = self.B, self.B_T

        scores = np.zeros((n_users, n_items))
        state = np.eye(n_users)                    # length-0 walks, on the user side
        contributions = {}
        for l in range(1, self.max_path_len + 1):
            state = state @ (A_ui if l % 2 == 1 else A_iu)
            if l % 2 == 1:                         # odd length -> we are on the item side
                inc = (self.beta ** l) * state
                scores = scores + inc
                contributions[l] = float(np.abs(inc).sum())
        self.diagnostics = {"path_mass_by_length": contributions,
                            "max_path_len": self.max_path_len, "beta": self.beta}
        return scores

    def score_all(self) -> np.ndarray:
        if self._scores is None:
            self._scores = (self._rwr_scores() if self.method == "rwr"
                            else self._path_scores())
        return self._scores

    # ---------------------------------------------------------------- rating head
    @staticmethod
    def _feature(x: np.ndarray) -> np.ndarray:
        """Scores span orders of magnitude; a log transform makes a linear head sane."""
        return np.log1p(np.maximum(np.asarray(x, float), 0.0) * 1e4)

    def _fit_rating_head(self, R: sp.csr_matrix) -> None:
        coo = R.tocoo()
        self.calibrate(coo.row, coo.col, coo.data)

    def calibrate(self, u, i, r):
        S = self.score_all()
        x = self._feature(S[np.asarray(u, int), np.asarray(i, int)])
        y = np.asarray(r, float)
        A = np.column_stack([np.ones_like(x), x])
        self.rating_head, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict_all(self) -> np.ndarray:
        a, b = self.rating_head
        return a + b * self._feature(self.score_all())

    def model_bytes(self) -> int:
        return self._nbytes(self.B, self._scores)

    # ---------------------------------------------------------------- explainability
    def top_paths(self, u: int, i: int, top_n: int = 5) -> list[dict]:
        """Strongest length-3 paths u -> i' -> v -> i, ranked by their walk probability.

        The contribution of one intermediate pair (i', v) to the degree-normalised
        length-3 walk probability is

            (1/deg(u)) * (1/deg(i')) * (1/deg(v))       for i' in I_u, v in U_{i'} n U_i
        """
        Iu = self.B[u].indices
        Ui = self.B_T[i].indices
        if Iu.size == 0 or Ui.size == 0:
            return []
        paths = []
        Ui_set = set(Ui.tolist())
        for ip in Iu:
            if ip == i:          # a path that passes through the target itself explains nothing
                continue
            raters = self.B_T[ip].indices
            for v in raters:
                if v == u or v not in Ui_set:
                    continue
                w = (1.0 / max(self.deg_u[u], 1)) * (1.0 / max(self.deg_i[ip], 1)) \
                    * (1.0 / max(self.deg_u[v], 1))
                paths.append({"via_item": int(ip), "via_user": int(v),
                              "path_weight": float(w)})
        paths.sort(key=lambda d: -d["path_weight"])
        # Aggregate by intermediate item: that is the humanly meaningful explanation.
        agg: dict[int, float] = {}
        cnt: dict[int, int] = {}
        for p in paths:
            agg[p["via_item"]] = agg.get(p["via_item"], 0.0) + p["path_weight"]
            cnt[p["via_item"]] = cnt.get(p["via_item"], 0) + 1
        out = [{"via_item": k, "total_path_weight": v, "n_paths": cnt[k]}
               for k, v in agg.items()]
        out.sort(key=lambda d: -d["total_path_weight"])
        return out[:top_n]

    def explain(self, u: int, i: int, top_n: int = 5) -> list[dict]:
        return self.top_paths(u, i, top_n)
