from pathlib import Path
from src import predictions as P

PKLS = ['model_win.pkl', 'model_runs_home.pkl', 'model_runs_away.pkl', 'model_inning.pkl']


def _write_pkls(d: Path, payloads):
    d.mkdir(parents=True, exist_ok=True)
    for name, b in payloads.items():
        (d / name).write_bytes(b)


def test_model_version_stable_and_changes(tmp_path):
    _write_pkls(tmp_path, {n: b'a' for n in PKLS})
    v1 = P.model_version(tmp_path)
    assert isinstance(v1, str) and len(v1) == 12
    assert P.model_version(tmp_path) == v1            # stable
    (tmp_path / 'model_win.pkl').write_bytes(b'different')
    assert P.model_version(tmp_path) != v1            # changes on pkl change


def test_model_version_none_when_missing(tmp_path):
    assert P.model_version(tmp_path) is None


def test_append_version_idempotent(tmp_path):
    P.append_version(tmp_path, {'version': 'abc', 'n_games': 1})
    P.append_version(tmp_path, {'version': 'abc', 'n_games': 1})
    P.append_version(tmp_path, {'version': 'def', 'n_games': 2})
    vs = P.read_versions(tmp_path)
    assert [v['version'] for v in vs] == ['abc', 'def']


def test_save_load_roundtrip(tmp_path):
    games = [{'game_id': 1, 'home_win_pct': 60.0, 'extra': 'dropped'}]
    P.save_prediction(tmp_path, 'v1', '2026-06-26', games)
    loaded = P.load_prediction(tmp_path, 'v1', '2026-06-26')
    assert loaded[0]['game_id'] == 1 and loaded[0]['home_win_pct'] == 60.0
    assert 'extra' not in loaded[0]


def test_load_missing_returns_none(tmp_path):
    assert P.load_prediction(tmp_path, 'v1', '2026-06-26') is None


def test_extract_core_keeps_only_fields():
    core = P.extract_core({'game_id': 1, 'home_win_pct': 60.0, 'home_logo': 'x', 'weather': {}})
    assert 'home_logo' not in core and 'weather' not in core
    assert core['game_id'] == 1
