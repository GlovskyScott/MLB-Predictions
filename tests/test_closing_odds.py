"""Tests for the closing-odds store + The Odds API capture (Part 2)."""
from unittest import mock

from src import fetcher as F
from src import predictions as P


def test_closing_store_upserts_latest(tmp_path):
    P.save_closing_odds(tmp_path, "2026-06-28",
                        {"1": {"ml_home": -120, "ml_away": 110, "book": "pinnacle"}})
    # A later capture (closer to first pitch) overwrites the earlier one.
    P.save_closing_odds(tmp_path, "2026-06-28",
                        {"1": {"ml_home": -135, "ml_away": 125, "book": "pinnacle"}})
    got = P.load_closing_odds(tmp_path, "2026-06-28")["1"]
    assert got["ml_home"] == -135 and got["ml_away"] == 125


def test_closing_store_skips_incomplete(tmp_path):
    P.save_closing_odds(tmp_path, "2026-06-28", {"1": {"ml_home": -135, "ml_away": None}})
    assert P.load_closing_odds(tmp_path, "2026-06-28") == {}


def test_pick_h2h_prefers_sharp_book():
    bms = [
        {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
            {"name": "NYY", "price": -140}, {"name": "BOS", "price": 120}]}]},
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "NYY", "price": -150}, {"name": "BOS", "price": 135}]}]},
    ]
    assert F._pick_h2h(bms, "NYY", "BOS") == (-150.0, 135.0, "pinnacle")


def test_pick_h2h_falls_back_to_any_book():
    bms = [{"key": "somebook", "markets": [{"key": "h2h", "outcomes": [
        {"name": "NYY", "price": -110}, {"name": "BOS", "price": -110}]}]}]
    assert F._pick_h2h(bms, "NYY", "BOS") == (-110.0, -110.0, "somebook")


def test_get_closing_moneylines_no_key_is_noop(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert F.get_closing_moneylines() == {}


def test_get_closing_moneylines_parses_payload():
    payload = [{
        "home_team": "New York Yankees", "away_team": "Boston Red Sox",
        "commence_time": "2026-06-28T23:05:00Z",
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "New York Yankees", "price": -150},
            {"name": "Boston Red Sox", "price": 135}]}]}],
    }]
    with mock.patch.object(F.requests, "get", return_value=mock.Mock(json=lambda: payload)):
        out = F.get_closing_moneylines(api_key="x")
    key = F.market_key("Boston Red Sox", "New York Yankees")
    assert out[key] == {"ml_home": -150.0, "ml_away": 135.0, "book": "pinnacle",
                        "commence_time": "2026-06-28T23:05:00Z"}


def test_get_closing_moneylines_handles_api_error():
    with mock.patch.object(F.requests, "get", side_effect=Exception("boom")):
        assert F.get_closing_moneylines(api_key="x") == {}


def test_get_historical_moneylines_parses_snapshot_and_quota():
    body = {"timestamp": "2024-05-01T17:00:00Z", "data": [{
        "home_team": "Detroit Tigers", "away_team": "St. Louis Cardinals",
        "commence_time": "2024-05-01T17:10:00Z",
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Detroit Tigers", "price": -125},
            {"name": "St. Louis Cardinals", "price": 115}]}]}]}]}
    resp = mock.Mock(status_code=200, json=lambda: body,
                     headers={"x-requests-remaining": "1234"})
    with mock.patch.object(F.requests, "get", return_value=resp):
        lines, remaining, ok = F.get_historical_moneylines("2024-05-01T17:10:00Z", api_key="x")
    key = F.market_key("St. Louis Cardinals", "Detroit Tigers")
    assert lines[key]["ml_home"] == -125 and lines[key]["ml_away"] == 115
    assert remaining == 1234 and ok is True


def test_get_historical_moneylines_no_key_is_noop(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert F.get_historical_moneylines("2024-05-01T17:10:00Z") == ({}, None, False)


def test_get_historical_moneylines_fatal_status_not_ok():
    resp = mock.Mock(status_code=401, json=lambda: {}, headers={})
    with mock.patch.object(F.requests, "get", return_value=resp):
        lines, remaining, ok = F.get_historical_moneylines("2024-05-01T17:10:00Z", api_key="bad")
    assert lines == {} and ok is False


def test_backfill_cluster_caps_snapshots_per_day():
    from scripts.backfill_closing_odds import cluster
    # 6 distinct start times spread over 9h, cap at 3 -> must collapse to <= 3 snapshots.
    games = [(f"2024-05-01T{h:02d}:05:00Z", i, "A", "B")
             for i, h in enumerate((17, 18, 20, 21, 23, 1))]  # last wraps next day via sort
    cl = cluster(games, tol_min=60, cap=3)
    assert len(cl) <= 3
    # every game lands in exactly one cluster
    assert sum(len(members) for _, members in cl) == len(games)
