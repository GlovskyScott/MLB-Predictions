#!/usr/bin/env python3
"""Register a reconstructed legacy model version into the prediction archive.

Takes the per-date prediction cores produced by `import_legacy_predictions.py`
(run in a legacy worktree) plus the release's model pkls, derives a version id
from those pkls, copies the cores into `data/predictions/<version>/`, and adds a
registry entry so the model shows up on the /archive pages.

Usage (from the current checkout, .venv active):

    python scripts/register_legacy_version.py \
        --pkl-dir /tmp/oldmodels/v1.0.0 \
        --staging /tmp/legacy_staging \
        --release v1.0.0 --created-at 2026-06-26T07:10:49Z \
        --features 32 --n-games 5921
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from src import predictions as P  # noqa: E402

_DATA_DIR = _REPO_ROOT / "data"


def legacy_version_id(pkl_dir: Path) -> str:
    """12-hex id over whatever model_*.pkl files the release shipped (sorted)."""
    h = hashlib.sha256()
    for name in sorted(p.name for p in pkl_dir.glob("model_*.pkl")):
        h.update((pkl_dir / name).read_bytes())
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", required=True, type=Path)
    ap.add_argument("--staging", required=True, type=Path)
    ap.add_argument("--release", required=True)
    ap.add_argument("--created-at", required=True)
    ap.add_argument("--features", type=int, required=True)
    ap.add_argument("--n-games", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    args = ap.parse_args()

    version = legacy_version_id(args.pkl_dir)
    dest = args.data_dir / "predictions" / version
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(args.staging.glob("*.json")):
        shutil.copy2(f, dest / f.name)
        n += 1

    P.append_version(args.data_dir, {
        "version": version,
        "created_at": args.created_at,
        "release": args.release,
        "feature_version": None,      # predates FEATURE_VERSION numbering
        "features": args.features,
        "n_games": args.n_games,
        "training_years": [2024, 2025, 2026],
    })
    print(f"Registered {args.release} as version {version} with {n} dates.")


if __name__ == "__main__":
    main()
