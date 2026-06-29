"""AI game explanations via a local Ollama model (default qwen2.5:14b).

Explanations are streamed to the browser over SSE and cached to memory + disk
(data/explanations/<date>/<game_id>.txt) so repeat loads are instant and a
background pre-generator can fill them in after each simulation run.
"""
import json
import os
from datetime import date
from pathlib import Path

import requests as _requests

_DATA_DIR = Path(__file__).parent.parent / "data"
_EXPLANATIONS_DIR = _DATA_DIR / "explanations"

_OLLAMA_URL = "http://localhost:11434/api/generate"
# Override with OLLAMA_MODEL (e.g. qwen2.5:7b / llama3.1:8b).
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
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

    return f"""You are a sharp baseball analyst. In 2-3 tight paragraphs, explain why the model predicts this outcome. Lead with the most decisive factors and cite the specific numbers below.

RULES — follow exactly:
- Use ONLY the numbers given below. Never invent statistics, records, player history, or ballpark dimensions, and never cite a number that is not listed.
- Describe the park ONLY via its runs factor (1.00 = neutral, above = favors hitters, below = favors pitchers). Do not mention field dimensions or the Green Monster.
- The score below is the model's single most-likely FULL-GAME final. Do not invent any other score, "median", or "average", and report it for the correct team.
- Stay internally consistent: if the projected runs are low, call it low-scoring — never also call it high-scoring.
- No bullet points. Confident present-tense analyst voice. Under 180 words.

{away} @ {home} — {venue}
Win probability: {away} {away_win}% | {home} {home_win}%
Model's projected final score: {away} {modal_away}, {home} {modal_home}

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
                  'options': {'num_predict': 350, 'temperature': 0.2}},
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
                  'options': {'num_predict': 350, 'temperature': 0.2}},
            timeout=120,
        )
        return resp.json().get('response', '').strip()
    except Exception:
        return ''


_edge_summary_cache: dict = {}  # game_id → short (<=12 word) edge angle


def _build_edge_prompt(game: dict) -> str:
    """Prompt for a one-sentence (<=12 word) angle on why to take the model's
    edge pick — the reasoning, not a restatement of the win% or odds."""
    f = game.get('features', {})
    aw, hw = game.get('away_abbr', 'AWAY'), game.get('home_abbr', 'HOME')
    fav_home = game.get('home_win_pct', 50) >= game.get('away_win_pct', 50)
    pick = hw if fav_home else aw
    pick_p = game.get('home_pitcher' if fav_home else 'away_pitcher', 'the starter')
    opp = aw if fav_home else hw
    mk = game.get('market') or {}
    return (
        f"You are a sharp MLB betting analyst. The model likes {pick} over {opp} "
        f"(+{mk.get('edge_ml_pct', 0)}% edge vs the market), behind {pick_p}.\n"
        f"Give the bettor the KEY ANGLE for taking {pick} in ONE punchy, complete "
        f"sentence — aim for about 8 words, no more than 14. Do NOT restate the "
        f"win %, the odds, or the projected score — only the reasoning. Plain text, "
        f"no quotes.\nAngle:"
    )


def _first_sentence_break(s: str) -> int:
    """Index of the first sentence-ending terminator followed by a space (so a
    decimal like 3.82 doesn't count), or -1. Used to stop after one sentence."""
    best = -1
    for sep in ('. ', '! ', '? '):
        i = s.find(sep)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def _stream_edge_summary(game: dict, game_id: int = None):
    """Stream the edge angle from Ollama as SSE deltas so it types in as it
    generates. Emits the first complete sentence only (no mid-thought chopping);
    brevity is encouraged by the prompt. Caches the final text. Yields
    {'error': ...} on failure so the client keeps its deterministic fallback."""
    sent = ''  # cleaned text already emitted to the client
    try:
        resp = _requests.post(
            _OLLAMA_URL,
            json={'model': _OLLAMA_MODEL, 'prompt': _build_edge_prompt(game),
                  'stream': True, 'options': {'num_predict': 48, 'temperature': 0.3}},
            stream=True, timeout=60,
        )
        acc = ''
        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            acc += chunk.get('response', '')
            done = bool(chunk.get('done'))
            cleaned = acc.replace('\n', ' ').strip().lstrip('"')
            brk = _first_sentence_break(cleaned)
            if brk != -1:           # stop at the end of the first sentence
                cleaned = cleaned[:brk]
                done = True
            delta = cleaned[len(sent):]
            if delta:
                sent = cleaned
                yield f"data: {json.dumps({'text': delta})}\n\n"
            if done:
                final = sent.rstrip(' .,;:')
                if game_id is not None and final:
                    _edge_summary_cache[game_id] = final
                yield "data: [DONE]\n\n"
                return
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


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

