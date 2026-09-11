"""Parts R & S -- generate REPORT.md, README.md and PORTFOLIO.md from results/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import banner
from src.report import write_all


def main() -> dict:
    banner("PARTS R & S - Report, README and portfolio pack")
    paths = write_all()
    sizes = {kk: Path(vv).stat().st_size for kk, vv in paths.items()}
    for kk, vv in paths.items():
        print(f"  {kk:10s} {vv}  ({sizes[kk] / 1024:.1f} KB)")
    return paths


if __name__ == "__main__":
    main()
