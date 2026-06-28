"""AI game explanations via a local Ollama model (llama3.1:8b).

Explanations are streamed to the browser over SSE and cached to memory + disk
(data/explanations/<date>/<game_id>.txt) so repeat loads are instant and a
background pre-generator can fill them in after each simulation run.
"""
import json
from datetime import date
from pathlib import Path

import requests as _requests

_DATA_DIR = Path(__file__).parent.parent / "data"
_EXPLANATIONS_DIR = _DATA_DIR / "explanations"

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.1:8b"
_explanation_cache: dict = {}  # game_id → full explanation text


def _load_disk_explanations(game_date: str) -> None:
    """Load any saved explanations for game_date from disk into memory cache."""
    day_dir = _EXPLANATIONS_DIR / game_date
    if not day_dir.exists():
        return
    for f in day_dir.glob("*.txt"):
        try:
            gid = int(f.stem)
            if gid not in _explanation_cache:
                _explanation_cache[gid] = f.read_text()
        except (ValueError, OSError):
            pass


def _save_disk_explanation(game_date: str, game_id: int, text: str) -> None:
    """Persist one explanation to disk under data/explanations/{date}/{game_id}.txt."""
    day_dir = _EXPLANATIONS_DIR / game_date
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / f"{game_id}.txt").write_text(text)


def _build_explain_prompt(game: dict) -> str:
    f = game.get('features', {})
    away = game.get('away_name', 'Away')
    home = game.get('home_name', 'Home')
    venue = game.get('venue_name', 'the ballpark')
    away_p = game.get('away_pitcher', 'TBD')
    home_p = game.get('home_pitcher', 'TBD')
    away_win = game.get('away_win_pct', 50)
    home_win = game.get('home_win_pct', 50)
    modal_away = game.get('modal_away_score', '?')
    modal_home = game.get('modal_home_score', '?')

    away_hand = 'LHP' if f.get('away_sp_is_lhp', 0) > 0.5 else 'RHP'
    home_hand = 'LHP' if f.get('home_sp_is_lhp', 0) > 0.5 else 'RHP'

    weather = game.get('weather') or {}
    is_dome = f.get('is_dome', 0) > 0.5
    if is_dome:
        wx = 'indoor dome — weather not a factor'
    else:
        wx = f"{weather.get('temperature_f', '?'):.0f}°F, {weather.get('wind_speed_mph', 0):.0f} mph"
        if f.get('wind_out', 0) > 0.5:
            wx += ' blowing out (hitter-friendly)'
        elif f.get('wind_in', 0) > 0.5:
            wx += ' blowing in (pitcher-friendly)'

    return f"""You are a sharp baseball analyst. Write 2-3 tight paragraphs explaining why the model predicts this outcome. Be specific, cite the numbers, and lead with the most decisive factors. No bullet points. Confident, present-tense analyst voice. Keep it under 200 words.

{away} @ {home} — {venue}
Win probability: {away} {away_win}% | {home} {home_win}%
Most likely score: {away} {modal_away} – {home} {modal_home}

Starters:
  {away}: {away_p} ({away_hand}) ERA {f.get('away_sp_era',0):.2f} FIP {f.get('away_sp_fip',0):.2f} WHIP {f.get('away_sp_whip',0):.2f} — {f.get('away_sp_days_rest',5):.0f}d rest
  {home}: {home_p} ({home_hand}) ERA {f.get('home_sp_era',0):.2f} FIP {f.get('home_sp_fip',0):.2f} WHIP {f.get('home_sp_whip',0):.2f} — {f.get('home_sp_days_rest',5):.0f}d rest

Offense (wOBA / OPS / R/G last 15):
  {away}: {f.get('away_team_woba',0):.3f} / {f.get('away_team_ops',0):.3f} / {f.get('away_runs_l15',0):.1f}
  {home}: {f.get('home_team_woba',0):.3f} / {f.get('home_team_ops',0):.3f} / {f.get('home_runs_l15',0):.1f}

Bullpen (ERA / L3 stress):
  {away}: {f.get('away_bullpen_era',0):.2f} ERA / {f.get('away_bullpen_stress_l3',0):.1f} stress
  {home}: {f.get('home_bullpen_era',0):.2f} ERA / {f.get('home_bullpen_stress_l3',0):.1f} stress

Handedness OPS edge:
  {away} vs {home_hand}: {f.get('away_bat_ops_vs_sp_hand',0):.3f}
  {home} vs {away_hand}: {f.get('home_bat_ops_vs_sp_hand',0):.3f}

Park runs factor: {f.get('park_runs_factor',1.0):.3f}  Elevation: {f.get('elevation_ft',500):.0f} ft  Weather: {wx}  Humidity: {f.get('humidity_pct',50):.0f}%

Analysis:"""


def _stream_ollama(prompt: str, game_id: int = None, game_date: str = None):
    """Stream Ollama response as SSE chunks, caching the full text when done."""
    buf = []
    try:
        resp = _requests.post(
            _OLLAMA_URL,
            json={'model': _OLLAMA_MODEL, 'prompt': prompt, 'stream': True,
                  'options': {'num_predict': 350, 'temperature': 0.7}},
            stream=True,
            timeout=90,
        )
        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            text = chunk.get('response', '')
            if text:
                buf.append(text)
                yield f"data: {json.dumps({'text': text})}\n\n"
            if chunk.get('done'):
                if game_id is not None and buf:
                    full = ''.join(buf)
                    _explanation_cache[game_id] = full
                    if game_date:
                        _save_disk_explanation(game_date, game_id, full)
                yield "data: [DONE]\n\n"
                return
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


def _generate_explanation_sync(game: dict) -> str:
    """Call Ollama synchronously (no streaming). Used by the background pre-generator."""
    prompt = _build_explain_prompt(game)
    try:
        resp = _requests.post(
            _OLLAMA_URL,
            json={'model': _OLLAMA_MODEL, 'prompt': prompt, 'stream': False,
                  'options': {'num_predict': 350, 'temperature': 0.7}},
            timeout=120,
        )
        return resp.json().get('response', '').strip()
    except Exception:
        return ''


def _pregenerate_explanations(games: list, game_date: str) -> None:
    """Background: generate and persist explanations for all valid games sequentially."""
    for game in games:
        gid = game.get('game_id')
        if not gid or game.get('error') or gid in _explanation_cache:
            continue
        text = _generate_explanation_sync(game)
        if text:
            _explanation_cache[gid] = text
            _save_disk_explanation(game_date, gid, text)
