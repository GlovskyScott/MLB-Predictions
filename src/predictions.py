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
import threading
from pathlib import Path

_PKL_NAMES = ('model_win.pkl', 'model_runs_home.pkl',
              'model_runs_away.pkl', 'model_inning.pkl')

_registry_lock = threading.Lock()
_version_memo: dict = {}  # pkl signature -> 12-hex hash (avoids re-hashing 1.5 MB per call)

# The simulation core persisted per game. No actual scores (joined on read) and
# no presentation fields (logos/colors/weather/features, re-derived at render).
PREDICTION_FIELDS = (
    'game_id', 'game_date', 'home_id', 'away_id', 'home_name', 'away_name',
    'venue_id', 'venue_name', 'home_win_pct', 'away_win_pct',
    'median_home_score', 'median_away_score', 'modal_home_score', 'modal_away_score',
    'predicted_score', 'score_distribution',
    'home_innings_dist', 'away_innings_dist', 'combined_innings_dist',
    'n_simulations',
)


def _pred_dir(data_dir) -> Path:
    return Path(data_dir) / 'predictions'


# ---- market-odds snapshot store --------------------------------------------
# De-vigged inputs for the market-blended ("Consensus") moneyline. Version-
# independent (the book line is the same regardless of model) and write-once: the
# first line captured for a game is preserved (a closing-ish line); later captures
# only add games not yet seen. Absent line -> the game falls back to the model.

_market_odds_lock = threading.Lock()


def _market_odds_dir(data_dir) -> Path:
    return Path(data_dir) / 'market_odds'


def load_market_odds(data_dir, date: str) -> dict:
    """Return {game_id(str): {ml_home, ml_away, captured_at}} for date, or {}."""
    f = _market_odds_dir(data_dir) / f'{date}.json'
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def save_market_odds(data_dir, date: str, odds: dict) -> None:
    """Merge moneyline snapshots for date, write-once per game.

    ``odds`` maps game_id -> {'ml_home': float|None, 'ml_away': float|None}.
    Games missing either price are skipped; games already on disk keep their
    first-captured line and timestamp.
    """
    import datetime
    with _market_odds_lock:
        existing = load_market_odds(data_dir, date)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        changed = False
        for gid, line in odds.items():
            key = str(gid)
            mlh, mla = (line or {}).get('ml_home'), (line or {}).get('ml_away')
            if mlh is None or mla is None or key in existing:
                continue
            existing[key] = {'ml_home': float(mlh), 'ml_away': float(mla),
                             'captured_at': now}
            changed = True
        if not changed and (_market_odds_dir(data_dir) / f'{date}.json').exists():
            return
        d = _market_odds_dir(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f'{date}.json.tmp'
        tmp.write_text(json.dumps(existing, default=str))
        tmp.replace(d / f'{date}.json')


# ---- closing-odds snapshot store -------------------------------------------
# True closing lines captured near first pitch (The Odds API, via
# fetcher.get_closing_moneylines). Unlike the write-once opener store above this
# UPSERTS — the closing line is the *latest* snapshot before a game starts, so a
# capture run near game time overwrites the earlier one. Accumulates forward; the
# real-closing-line backtest reads it when present.

_closing_odds_lock = threading.Lock()


def _closing_odds_dir(data_dir) -> Path:
    return Path(data_dir) / 'market_closing'


def load_closing_odds(data_dir, date: str) -> dict:
    """Return {game_id(str): {ml_home, ml_away, book, captured_at}} for date, or {}."""
    f = _closing_odds_dir(data_dir) / f'{date}.json'
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def save_closing_odds(data_dir, date: str, odds: dict) -> None:
    """Upsert per-game closing snapshots for date (latest capture wins).

    ``odds`` maps game_id -> {'ml_home', 'ml_away', 'book'(optional)}. Games
    missing either price are skipped. Re-running closer to first pitch overwrites
    a game's prior snapshot with the fresher (more 'closing') line.
    """
    import datetime
    with _closing_odds_lock:
        existing = load_closing_odds(data_dir, date)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        changed = False
        for gid, line in odds.items():
            mlh, mla = (line or {}).get('ml_home'), (line or {}).get('ml_away')
            if mlh is None or mla is None:
                continue
            existing[str(gid)] = {'ml_home': float(mlh), 'ml_away': float(mla),
                                  'book': (line or {}).get('book'), 'captured_at': now}
            changed = True
        if not changed:
            return
        d = _closing_odds_dir(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f'{date}.json.tmp'
        tmp.write_text(json.dumps(existing, default=str))
        tmp.replace(d / f'{date}.json')


def model_version(data_dir):
    """Return the 12-hex version id for the current model pkls, or None if any
    pkl is missing. Deterministic: same bytes -> same id; any change -> new id.

    Memoized on the pkls' (path, mtime, size) signature so callers (every
    prediction/grade path hits this) don't re-read and SHA-256 ~1.5 MB each
    call. A retrain rewrites the pkls -> new mtime -> the hash recomputes.
    """
    paths = [Path(data_dir) / name for name in _PKL_NAMES]
    if not all(p.exists() for p in paths):
        return None
    sig = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    cached = _version_memo.get(sig)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    v = h.hexdigest()[:12]
    _version_memo[sig] = v
    return v


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
    """Append a version entry to the registry, idempotent on entry['version'].

    Locked: the backfill runs parallel workers that all call _current_version on
    a new version's first sight, which would otherwise race read-modify-write.
    """
    with _registry_lock:
        versions = read_versions(data_dir)
        if any(v.get('version') == entry.get('version') for v in versions):
            return
        versions.append(entry)
        f = _versions_file(data_dir)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(versions, indent=2))
        tmp.replace(f)


def next_build(data_dir, major) -> int:
    """Next build index for a feature generation: the count of already-registered
    versions whose major matches. Drives the v<major>.<build> naming rule."""
    return sum(1 for v in read_versions(data_dir) if v.get('major') == major)


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
