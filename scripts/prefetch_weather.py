"""Bulk-fetch historical weather for every completed game in 2024-2026.

Strategy: one Open-Meteo API call per (stadium, month, year) — 12 calls per
stadium per year — so every calendar month is independently covered regardless
of schedule quirks (spring training, postseason, doubleheaders). With 30
outdoor stadiums × 12 months × 3 years that is ~1,080 API calls total, each
returning a full month of hourly data at no cost.

Run this before retraining to populate weather_cache.csv with real values for
temperature, wind, precipitation, and humidity for all historical games.

Usage:
    python scripts/prefetch_weather.py
"""
import sys
import time
import calendar
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetcher import (
    get_season_schedule,
    OPEN_METEO_HISTORICAL_URL,
    _WEATHER_CACHE_FILE,
    _weather_cache,
    _load_weather_cache,
)
from src.stadiums import get_stadium

YEARS = [2024, 2025, 2026]
MONTHS = range(1, 13)   # January–December, all 12
REQUEST_DELAY = 0.25    # seconds between API calls — be a good citizen


def _game_hour(game_datetime: str) -> int:
    if game_datetime and 'T' in game_datetime and len(game_datetime) > 13:
        try:
            return int(game_datetime[11:13])
        except ValueError:
            pass
    return 19


def fetch_month(lat: float, lon: float, year: int, month: int) -> tuple[dict, dict] | tuple[None, None]:
    """One Open-Meteo call covering the entire calendar month at (lat, lon).

    Returns (hourly_arrays, time→index_map) or (None, None) on error.
    """
    last_day = calendar.monthrange(year, month)[1]
    start = f'{year}-{month:02d}-01'
    end   = f'{year}-{month:02d}-{last_day:02d}'

    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': (
            'temperature_2m,windspeed_10m,winddirection_10m,'
            'precipitation,relative_humidity_2m'
        ),
        'temperature_unit': 'fahrenheit',
        'windspeed_unit': 'mph',
        'start_date': start,
        'end_date': end,
        'timezone': 'UTC',
    }
    try:
        resp = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()['hourly']
        time_to_idx = {t: i for i, t in enumerate(hourly['time'])}
        return hourly, time_to_idx
    except Exception as exc:
        print(f"    WARN: {exc}")
        return None, None


def extract_weather(hourly: dict, time_to_idx: dict, game_date: str, hour: int) -> dict | None:
    key = f'{game_date}T{hour:02d}:00'
    idx = time_to_idx.get(key)
    if idx is None:
        return None
    return {
        'temperature_f':    float(hourly['temperature_2m'][idx]),
        'wind_speed_mph':   float(hourly['windspeed_10m'][idx]),
        'wind_direction_deg': float(hourly['winddirection_10m'][idx]),
        'precipitation_mm': float(hourly['precipitation'][idx]),
        'humidity_pct':     float(hourly['relative_humidity_2m'][idx]),
        'is_dome':          False,
    }


def main():
    _load_weather_cache()
    print(f"Cache loaded: {len(_weather_cache)} entries already on disk\n")

    # Collect all outdoor stadiums across every year's schedule
    # stadium_games[(lat, lon, year)] = [game, ...]
    stadium_games: dict[tuple, list] = defaultdict(list)

    for year in YEARS:
        schedule = get_season_schedule(year)
        completed = [g for g in schedule if g.get('status') == 'Final']
        print(f"{year}: {len(completed)} completed games")
        for g in completed:
            home_id = g.get('home_id')
            if not home_id:
                continue
            stadium = get_stadium(int(home_id)) or {}
            if stadium.get('roof') == 'dome':
                continue
            lat = round(float(stadium.get('lat', 0)), 2)
            lon = round(float(stadium.get('lon', 0)), 2)
            if lat == 0 and lon == 0:
                continue
            stadium_games[(lat, lon, year)].append(g)

    # Unique outdoor stadium-year combos
    combos = sorted(stadium_games.keys())
    print(f"\n{len(combos)} outdoor stadium-year combos × 12 months = "
          f"{len(combos) * 12} API calls\n")

    new_entries: list[dict] = []
    total_filled = 0
    call_count = 0

    for (lat, lon, year) in combos:
        games = stadium_games[(lat, lon, year)]
        venue = games[0].get('venue_name', f'{lat},{lon}')

        # Group this stadium's games by month
        by_month: dict[int, list] = defaultdict(list)
        for g in games:
            try:
                m = int(g['game_date'][5:7])
                by_month[m].append(g)
            except (KeyError, ValueError):
                pass

        venue_filled = 0
        for month in MONTHS:
            month_games = by_month.get(month, [])
            # Only games missing from cache need a fetch, but we fetch the whole
            # month regardless so we have complete coverage for that (lat, lon, month, year).
            missing = [
                g for g in month_games
                if (g['game_date'], lat, lon) not in _weather_cache
            ]
            if not missing:
                continue  # all games this month already cached

            call_count += 1
            hourly, time_to_idx = fetch_month(lat, lon, year, month)
            time.sleep(REQUEST_DELAY)

            if hourly is None:
                continue

            for g in missing:
                hour = _game_hour(g.get('game_datetime', ''))
                weather = extract_weather(hourly, time_to_idx, g['game_date'], hour)
                if weather:
                    cache_key = (g['game_date'], lat, lon)
                    _weather_cache[cache_key] = weather
                    new_entries.append({
                        'date': g['game_date'],
                        'lat': lat,
                        'lon': lon,
                        **{k: v for k, v in weather.items() if k != 'is_dome'},
                    })
                    venue_filled += 1
                    total_filled += 1

        if venue_filled:
            print(f"  {venue} {year}: +{venue_filled} games cached")

    # Write all new entries at once
    if new_entries:
        df_new = pd.DataFrame(new_entries)
        write_header = not _WEATHER_CACHE_FILE.exists()
        df_new.to_csv(_WEATHER_CACHE_FILE, mode='a', header=write_header, index=False)
        print(f"\nWrote {len(new_entries)} new entries → {_WEATHER_CACHE_FILE}")
    else:
        print("\nAll games already cached — nothing to write.")

    print(f"API calls made: {call_count}  |  Games filled: {total_filled}")
    print("Run /retrain (or click Retrain in the UI) to rebuild models with real weather data.")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f}s")
