"""Reconstruct model training features for past seasons (for the real-closing-line
backtest). Writes data/backtest_features_<years>.parquet.

The training path uses neutral weather (features.build_game_features with
_NEUTRAL_WEATHER), so no historical weather reconstruction is needed — the pull is
statsapi schedules + Baseball-Reference rate stats, which support arbitrary years.

Run:  python -m scripts.build_historical_features 2019 2021
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.training import build_training_df  # noqa: E402

_DATA = _REPO / "data"


def main():
    years = [int(a) for a in sys.argv[1:]] or [2019, 2021]
    tag = "_".join(str(y) for y in years)
    out = _DATA / f"backtest_features_{tag}.parquet"
    print(f"Building features for {years} -> {out}", flush=True)
    t = time.time()
    df = build_training_df(years)
    if df.empty:
        print("EMPTY — no games built", flush=True)
        sys.exit(1)
    df.to_parquet(out)
    print(f"DONE {len(df)} games in {time.time()-t:.0f}s "
          f"({df['game_date'].min()}..{df['game_date'].max()}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
