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


def test_next_build_counts_per_major(tmp_path):
    assert P.next_build(tmp_path, 4) == 0
    P.append_version(tmp_path, {'version': 'a', 'major': 4})
    P.append_version(tmp_path, {'version': 'b', 'major': 4})
    P.append_version(tmp_path, {'version': 'c', 'major': 1})
    assert P.next_build(tmp_path, 4) == 2   # -> next would be v4.2
    assert P.next_build(tmp_path, 1) == 1   # -> next would be v1.1
    assert P.next_build(tmp_path, 5) == 0   # new feature gen starts at .0


# ---- market-odds (moneyline) snapshot store --------------------------------

def test_market_odds_round_trip(tmp_path):
    P.save_market_odds(tmp_path, '2026-06-28',
                       {1: {'ml_home': -130, 'ml_away': 110},
                        2: {'ml_home': 105, 'ml_away': -125}})
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert got['1']['ml_home'] == -130 and got['1']['ml_away'] == 110
    assert 'captured_at' in got['1']


def test_market_odds_missing_returns_empty(tmp_path):
    assert P.load_market_odds(tmp_path, '2026-01-01') == {}


def test_market_odds_write_once(tmp_path):
    P.save_market_odds(tmp_path, '2026-06-28', {1: {'ml_home': -130, 'ml_away': 110}})
    ts = P.load_market_odds(tmp_path, '2026-06-28')['1']['captured_at']
    P.save_market_odds(tmp_path, '2026-06-28',
                       {1: {'ml_home': -200, 'ml_away': 170},
                        3: {'ml_home': 100, 'ml_away': -120}})
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert got['1']['ml_home'] == -130 and got['1']['captured_at'] == ts
    assert got['3']['ml_home'] == 100


def test_market_odds_skips_incomplete(tmp_path):
    P.save_market_odds(tmp_path, '2026-06-28',
                       {1: {'ml_home': None, 'ml_away': 110},
                        2: {'ml_home': -120, 'ml_away': 100}})
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert '1' not in got and got['2']['ml_home'] == -120
