#!/usr/bin/env python3
"""
Fetch and verify the MovieLens 100K dataset.

The dataset is **not** committed to this repository. GroupLens' usage licence states
that "the user may not redistribute the data without separate permission", so the repo
ships the code that reproduces the study and this script fetches the data instead.

    python download_data.py

The archive is checked against the MD5 published by GroupLens before anything is
extracted, so a mirror is as trustworthy as the original host for this purpose -- the
checksum, not the hostname, is what establishes authenticity.

Note on the primary host: https://files.grouplens.org has, at time of writing, an expired
TLS certificate. This script therefore tries the official URL first and falls back to
mirrors, but it *never* disables certificate verification and it *always* refuses an
archive whose MD5 does not match. If every source fails, download ml-100k.zip by hand
from https://grouplens.org/datasets/movielens/100k/ and drop it in data/.
"""
from __future__ import annotations

import hashlib
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARCHIVE = DATA / "ml-100k.zip"
EXTRACTED = DATA / "ml-100k"

# MD5 published by GroupLens for the official ml-100k.zip release.
EXPECTED_MD5 = "0e33842e24a9c977be4e0107933c0723"
EXPECTED_SIZE = 4_924_029

SOURCES = [
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "https://raw.githubusercontent.com/rudrasingh21/Data-ML-100k-/master/ml-100k.zip",
]

REQUIRED_FILES = ["u.data", "u.item", "u.user", "u.genre", "u.info", "README"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> bool:
    try:
        print(f"  trying {url}")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=120, context=ctx) as r, \
                open(dest, "wb") as out:
            shutil.copyfileobj(r, out)
        return True
    except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
        print(f"    failed: {exc}")
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)

    if ARCHIVE.exists() and md5(ARCHIVE) == EXPECTED_MD5:
        print(f"archive already present and verified: {ARCHIVE}")
    else:
        if ARCHIVE.exists():
            print("existing archive failed its checksum; re-downloading")
            ARCHIVE.unlink()
        print("downloading MovieLens 100K…")
        got = any(fetch(url, ARCHIVE) for url in SOURCES)
        if not got:
            print("\nERROR: could not download from any source.\n"
                  "Download ml-100k.zip manually from\n"
                  "  https://grouplens.org/datasets/movielens/100k/\n"
                  f"and save it to {ARCHIVE}", file=sys.stderr)
            return 1

        actual = md5(ARCHIVE)
        if actual != EXPECTED_MD5:
            ARCHIVE.unlink(missing_ok=True)
            print(f"\nERROR: checksum mismatch.\n  expected {EXPECTED_MD5}\n"
                  f"  got      {actual}\nThe file was not what it claims to be; "
                  "it has been deleted.", file=sys.stderr)
            return 1
        print(f"  MD5 verified: {actual}")

    size = ARCHIVE.stat().st_size
    if size != EXPECTED_SIZE:
        print(f"warning: archive is {size} bytes, expected {EXPECTED_SIZE}")

    print(f"extracting to {EXTRACTED}…")
    with zipfile.ZipFile(ARCHIVE) as z:
        # Refuse path traversal rather than trusting the archive's member names.
        for name in z.namelist():
            target = (DATA / name).resolve()
            if not str(target).startswith(str(DATA.resolve())):
                print(f"ERROR: archive member escapes data/: {name}", file=sys.stderr)
                return 1
        z.extractall(DATA)

    missing = [f for f in REQUIRED_FILES if not (EXTRACTED / f).exists()]
    if missing:
        print(f"ERROR: extraction incomplete, missing {missing}", file=sys.stderr)
        return 1

    n_ratings = sum(1 for _ in open(EXTRACTED / "u.data"))
    print(f"\nready: {EXTRACTED}  ({n_ratings:,} ratings)")
    print("next:  python run_experiments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
