"""Versioned, on-disk prediction store.

Predictions are immutable artifacts keyed by model version. A version is a hash
of the model .pkl files, so any retrain produces a new version. Within a version
a date's prediction is written once and served verbatim forever; a model change
forks a new version folder and leaves the old one as an archive.

Layout under data/:
    predictions/versions.json          registry of known versions (ordered)
    predictions/<version>/<date>.json   frozen per-game prediction core
"""
import hashlib
import json
from pathlib import Path

_PKL_NAMES = ('model_win.pkl', 'model_runs_home.pkl',
              'model_runs_away.pkl', 'model_inning.pkl')

# The simulation core persisted per game. No actual scores (joined on read) and
# no presentation fields (logos/colors/weather/features, re-derived at render).
PREDICTION_FIELDS = (
    'game_id', 'game_date', 'home_id', 'away_id', 'home_name', 'away_name',
    'venue_id', 'venue_name', 'home_win_pct', 'away_win_pct',
    'median_home_score', 'median_away_score', 'modal_home_score', 'modal_away_score',
    'predicted_score', 'score_distribution',
    'home_innings_scoring_pct', 'away_innings_scoring_pct',
    'home_innings', 'away_innings', 'n_simulations',
)


def _pred_dir(data_dir) -> Path:
    return Path(data_dir) / 'predictions'


def model_version(data_dir):
    """Return the 12-hex version id for the current model pkls, or None if any
    pkl is missing. Deterministic: same bytes -> same id; any change -> new id."""
    h = hashlib.sha256()
    for name in _PKL_NAMES:
        p = Path(data_dir) / name
        if not p.exists():
            return None
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _versions_file(data_dir) -> Path:
    return _pred_dir(data_dir) / 'versions.json'


def read_versions(data_dir) -> list:
    f = _versions_file(data_dir)
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception:
        return []


def append_version(data_dir, entry: dict) -> None:
    """Append a version entry to the registry, idempotent on entry['version']."""
    versions = read_versions(data_dir)
    if any(v.get('version') == entry.get('version') for v in versions):
        return
    versions.append(entry)
    f = _versions_file(data_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(versions, indent=2))


def load_prediction(data_dir, version: str, date: str):
    """Return the stored prediction core list for (version, date), or None."""
    f = _pred_dir(data_dir) / version / f'{date}.json'
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def save_prediction(data_dir, version: str, date: str, games: list) -> None:
    """Write the prediction core for (version, date) atomically."""
    d = _pred_dir(data_dir) / version
    d.mkdir(parents=True, exist_ok=True)
    core = [extract_core(g) for g in games]
    tmp = d / f'{date}.json.tmp'
    tmp.write_text(json.dumps(core, default=str))
    tmp.replace(d / f'{date}.json')


def list_dates(data_dir, version: str) -> list:
    """Return sorted dates (YYYY-MM-DD) that have a stored prediction for version."""
    d = _pred_dir(data_dir) / version
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob('*.json'))


def extract_core(game: dict) -> dict:
    """Slim a full sim/result dict down to the persisted PREDICTION_FIELDS."""
    return {k: game[k] for k in PREDICTION_FIELDS if k in game}
