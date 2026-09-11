"""Persist the fitted, validation-selected models so the demo app starts instantly."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTIFACTS_DIR
from src.pipeline import banner, fit_model, get_context, load_json

PARADIGMS = [("UBCF", "ubcf"), ("IBCF", "ibcf"), ("RegressionCF", "regression"),
             ("SLIM", "slim"), ("GraphRec", "graph")]


def main() -> dict:
    banner("ARTIFACTS - persist fitted models for the demo app")
    _, split, _ = get_context()
    out_dir = ARTIFACTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for label, key in PARADIGMS:
        sel = load_json(key)
        spec = sel["selection"]["ranking_selected"] if "selection" in sel \
            else sel["ranking_selected"]
        model = fit_model(spec, split.R_train, split.val)
        # Materialise the score matrix before pickling so the app never recomputes it.
        model.score_all()
        model.predict_all()
        path = out_dir / f"{label}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"label": label, "spec": spec, "model": model}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        manifest[label] = {"spec": spec, "file": path.name,
                           "bytes": path.stat().st_size,
                           "fit_time_s": model.fit_time_s}
        print(f"  {label:14s} {path.stat().st_size / 1e6:7.2f} MB  "
              f"fit={model.fit_time_s:5.2f}s  {spec}")

    return manifest


if __name__ == "__main__":
    main()
