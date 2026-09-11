"""
Global configuration: paths, reproducible seeds and the roll-number -> target-user mapping.

Every stochastic step in this project draws from a seed derived from the roll number,
so the entire pipeline is reproducible from a single documented constant.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ML100K_DIR = DATA_DIR / "ml-100k"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = ROOT / "figures"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"

for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Published MD5 of the official GroupLens ml-100k.zip release.
ML100K_ZIP_MD5 = "0e33842e24a9c977be4e0107933c0723"

# --------------------------------------------------------------------------------------
# Roll number -> target user  (documented, hand-verifiable, deterministic)
# --------------------------------------------------------------------------------------
ROLL_NUMBER = "23IM10049"


def roll_digits(roll: str = ROLL_NUMBER) -> int:
    """Concatenate the decimal digits of the roll number, in order, into one integer.

    '23IM10049' -> '23' + '10049' -> '2310049'
    """
    digits = "".join(ch for ch in roll if ch.isdigit())
    if not digits:
        raise ValueError(f"roll number {roll!r} contains no digits")
    return int(digits)


def target_user_id(n_users: int, roll: str = ROLL_NUMBER) -> int:
    """Map a roll number to a 1-based MovieLens user id.

        target = (digits(roll) mod n_users) + 1

    The modulo makes the map total over any roll number; the ``+1`` shifts the
    result from the 0-based residue class into MovieLens' 1-based user ids.
    For roll 23IM10049 and n_users = 943 this is (2310049 mod 943) + 1 = 642 + 1 = 643.
    """
    return (roll_digits(roll) % n_users) + 1


# The master seed is itself derived from the roll number so that two students with
# different roll numbers get genuinely different (but individually reproducible) runs.
SEED = roll_digits() % (2**31 - 1)  # 2310049


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen experiment configuration shared by every stage of the pipeline."""

    seed: int = SEED
    roll: str = ROLL_NUMBER

    # Global rating split (per-user stratified).
    test_frac: float = 0.20
    val_frac: float = 0.10

    # Top-N protocol.
    top_k: int = 10
    relevance_threshold: float = 4.0  # a held-out rating >= 4 counts as a true positive

    # Implicit-feedback conversion for SLIM / graph models (justified in Part F).
    implicit_threshold: float = 4.0

    # Neighbourhood grid used by UBCF / IBCF sweeps.
    k_grid: tuple[int, ...] = (5, 10, 20, 30, 50, 80, 120, 200, 300)

    # Minimum co-ratings required before a similarity is trusted at all.
    min_support: int = 3
    # Significance-weighting (shrinkage) constant: s' = s * min(n_uv, beta) / beta.
    shrinkage_beta: float = 25.0

    # Target-user experiment: fraction of the target's ratings to hide.
    target_hide_frac: float = 0.30

    # Popularity-group partition: equal-interaction-mass tertiles.
    popularity_mass_cuts: tuple[float, float] = (1 / 3, 2 / 3)

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "roll": self.roll,
            "test_frac": self.test_frac,
            "val_frac": self.val_frac,
            "top_k": self.top_k,
            "relevance_threshold": self.relevance_threshold,
            "implicit_threshold": self.implicit_threshold,
            "k_grid": list(self.k_grid),
            "min_support": self.min_support,
            "shrinkage_beta": self.shrinkage_beta,
            "target_hide_frac": self.target_hide_frac,
            "popularity_mass_cuts": list(self.popularity_mass_cuts),
        }


CONFIG = ExperimentConfig()
