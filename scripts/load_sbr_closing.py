"""Load real CLOSING moneylines from the SportsbookReviewsOnline archive
(free, no auth) and join them to our game_ids → data/sbr_closing_<years>.parquet.

The SBR .xlsx is two rows per game (V=away, H=home) with an integer American
`Close` column. We pair the rows, parse the M-DD date, map SBR's team abbrev to
the statsapi team id, and join (date, away_id, home_id) to the season schedule to
attach our game_id — the key the walk-forward backtest uses.

Run:  python -m scripts.load_sbr_closing 2019 2021
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.fetcher import get_season_schedule  # noqa: E402

_DATA = _REPO / "data"
_SBR_DIR = _DATA / "sbr"
_URL = ("https://www.sportsbookreviewsonline.com/wp-content/uploads/"
        "sportsbookreviewsonline_com_737/mlb-odds-{year}.xlsx")

# SBR team abbreviation -> statsapi team id (stable; survives name changes).
SBR_TO_ID = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CIN": 113, "CLE": 114,
    "COL": 115, "CUB": 112, "CWS": 145, "DET": 116, "HOU": 117, "KAN": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "OAK": 133, "PHI": 143, "PIT": 134, "SDG": 135, "SEA": 136,
    "SFO": 137, "STL": 138, "TAM": 139, "TEX": 140, "TOR": 141, "WAS": 120,
}


def _download(year: int) -> Path:
    _SBR_DIR.mkdir(parents=True, exist_ok=True)
    path = _SBR_DIR / f"mlb-odds-{year}.xlsx"
    if path.exists() and path.stat().st_size > 10000:
        return path
    r = requests.get(_URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def _parse_date(raw: int, year: int) -> str:
    s = int(raw)
    day, month = s % 100, s // 100
    return f"{year}-{month:02d}-{day:02d}"


def parse_year(year: int) -> pd.DataFrame:
    """One row per game: game_date, away_id, home_id, away_close, home_close."""
    df = pd.read_excel(_download(year))
    df = df[df["VH"].isin(["V", "H"])].reset_index(drop=True)
    games = []
    for i in range(0, len(df) - 1, 2):
        v, h = df.iloc[i], df.iloc[i + 1]
        if v["VH"] != "V" or h["VH"] != "H":   # stay aligned if a row is malformed
            continue
        aid, hid = SBR_TO_ID.get(str(v["Team"])), SBR_TO_ID.get(str(h["Team"]))
        if aid is None or hid is None:
            continue
        try:
            ac, hc = int(v["Close"]), int(h["Close"])
        except (ValueError, TypeError):
            continue
        games.append({"game_date": _parse_date(v["Date"], year),
                      "away_id": aid, "home_id": hid,
                      "away_close": ac, "home_close": hc})
    return pd.DataFrame(games)


def schedule_map(year: int) -> dict:
    """(date, away_id, home_id) -> game_id from the season schedule (skip ambiguous DH)."""
    seen, out = {}, {}
    for g in get_season_schedule(year):
        key = (str(g.get("game_date")), g.get("away_id"), g.get("home_id"))
        seen[key] = seen.get(key, 0) + 1
        out[key] = g.get("game_id")
    return {k: v for k, v in out.items() if seen[k] == 1}   # drop doubleheaders


def main():
    years = [int(a) for a in sys.argv[1:]] or [2019, 2021]
    rows = []
    for y in years:
        sbr = parse_year(y)
        smap = schedule_map(y)
        matched = 0
        for r in sbr.itertuples(index=False):
            gid = smap.get((r.game_date, r.away_id, r.home_id))
            if gid is None:
                continue
            matched += 1
            rows.append({"game_id": int(gid), "game_date": r.game_date,
                         "ml_home": r.home_close, "ml_away": r.away_close})
        print(f"{y}: {len(sbr)} SBR games, {matched} joined to a game_id "
              f"({100*matched/max(len(sbr),1):.0f}%)", flush=True)
    out_df = pd.DataFrame(rows)
    tag = "_".join(str(y) for y in years)
    out = _DATA / f"sbr_closing_{tag}.parquet"
    out_df.to_parquet(out)
    print(f"Wrote {len(out_df)} closing lines -> {out}", flush=True)


if __name__ == "__main__":
    main()
