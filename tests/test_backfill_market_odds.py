from scripts import backfill_market_odds as BF
from src import predictions as P
from src.fetcher import market_key


def test_collect_market_odds_writes_by_game_id(tmp_path, mocker):
    mocker.patch('scripts.backfill_market_odds.get_schedule', return_value=[
        {'game_id': 11, 'away_name': 'Boston Red Sox', 'home_name': 'New York Yankees'},
        {'game_id': 12, 'away_name': 'Chicago Cubs', 'home_name': 'St. Louis Cardinals'},
    ])
    mocker.patch('scripts.backfill_market_odds.get_market_odds', return_value={
        market_key('Boston Red Sox', 'New York Yankees'): {'ml_home': -130, 'ml_away': 110},
        # game 12 has no odds -> skipped
    })
    n = BF.collect_market_odds('2026-06-28', tmp_path)
    assert n == 1
    got = P.load_market_odds(tmp_path, '2026-06-28')
    assert got['11']['ml_home'] == -130 and '12' not in got


def test_collect_market_odds_empty_when_no_odds(tmp_path, mocker):
    mocker.patch('scripts.backfill_market_odds.get_market_odds', return_value={})
    assert BF.collect_market_odds('2026-06-28', tmp_path) == 0
