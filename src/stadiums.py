import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_STADIUMS_FILE = _DATA_DIR / "stadiums.json"

WIND_DIRECTIONS = {
    "in_from_cf": (135, 225),
    "out_to_cf": (315, 45),
    "crosswind": (45, 135),
}

def get_all_stadiums() -> dict:
    with open(_STADIUMS_FILE) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def get_stadium(team_id: int) -> dict | None:
    return get_all_stadiums().get(team_id)

def classify_wind(wind_direction_deg: float, home_team_id: int) -> str:
    """Classify wind as in_from_cf, out_to_cf, or crosswind."""
    d = wind_direction_deg % 360
    if 135 <= d <= 225:
        return "in_from_cf"
    elif d <= 45 or d >= 315:
        return "out_to_cf"
    return "crosswind"
