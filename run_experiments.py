#!/usr/bin/env python3
"""
Full reproducible pipeline.

    python run_experiments.py                 # everything, in order
    python run_experiments.py --only 06 08    # just those stages
    python run_experiments.py --from 08       # resume from a stage onward
    python run_experiments.py --list          # show the stage list

Stages are ordered by data dependency, not by part letter: the figures stage needs the
cold-start tables, and the report stage needs everything.  Each stage is also runnable on
its own as `python experiments/expNN_*.py`.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.similarity import clear_similarity_cache  # noqa: E402

STAGES = [
    ("01", "exp01_eda", "Part A - data exploration, split, popularity groups, target user"),
    ("02", "exp02_baselines", "Part B - baselines"),
    ("03", "exp03_ubcf", "Part C - user-based CF sweep"),
    ("04", "exp04_ibcf", "Part D - item-based CF sweep"),
    ("05", "exp05_regression", "Part E - regression-based neighbourhood CF"),
    ("06", "exp06_slim", "Part F - SLIM regularisation and threshold study"),
    ("07", "exp07_graph", "Part G - graph recommenders"),
    ("08", "exp08_multiuser", "Part H - multi-user evaluation on TEST"),
    ("09", "exp09_coldstart", "Part I - cold-start and sparsity analysis"),
    ("10", "exp10_popbias", "Part J - popularity bias and long tail"),
    ("11", "exp11_ablations", "Part K - ablation studies"),
    ("12", "exp12_target_user", "Parts L & M - explainability and target user"),
    ("13", "exp13_master", "Part N - master comparison"),
    ("15", "exp15_error_analysis", "Error analysis"),
    ("16", "exp16_artifacts", "Persist fitted models for the demo app"),
    ("14", "exp14_figures", "Part O - figures"),
    ("17", "exp17_report", "Parts R & S - report, README, portfolio"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ID", help="run only these stage ids")
    ap.add_argument("--from", dest="start", metavar="ID", help="resume from this stage")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for sid, mod, desc in STAGES:
            print(f"  {sid}  {mod:24s} {desc}")
        return 0

    stages = STAGES
    if args.only:
        want = set(args.only)
        stages = [s for s in STAGES if s[0] in want]
    elif args.start:
        ids = [s[0] for s in STAGES]
        if args.start not in ids:
            print(f"unknown stage {args.start!r}", file=sys.stderr)
            return 2
        stages = STAGES[ids.index(args.start):]

    t_all = time.perf_counter()
    failures = []
    for sid, mod, desc in stages:
        print(f"\n\n>>> [{sid}] {desc}", flush=True)
        t0 = time.perf_counter()
        try:
            m = importlib.import_module(f"experiments.{mod}")
            m.main()
            # Stages are independent; drop the memoised similarity matrices so peak
            # memory is bounded by one stage rather than by the whole pipeline.
            clear_similarity_cache()
            print(f"<<< [{sid}] done in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception:
            traceback.print_exc()
            failures.append(sid)
            print(f"<<< [{sid}] FAILED after {time.perf_counter() - t0:.1f}s", flush=True)

    total = time.perf_counter() - t_all
    print(f"\n{'=' * 78}")
    print(f"pipeline finished in {total / 60:.1f} min "
          f"({len(stages) - len(failures)}/{len(stages)} stages OK)")
    if failures:
        print(f"FAILED stages: {', '.join(failures)}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
