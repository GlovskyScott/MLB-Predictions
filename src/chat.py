"""Conversational assistant backed by the local Ollama model.

`stream_chat` proxies a conversation to Ollama's /api/chat with a system prompt
(the day's data + history + model facts, built by the app) and streams the
reply back as plain-text deltas. The dashboard renders these in a chat widget.
"""
import json

import requests as _requests

_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
_OLLAMA_MODEL = "llama3.1:8b"

_SYSTEM_PREAMBLE = (
    "You are the assistant built into an MLB prediction dashboard. The data block "
    "below contains everything the site serves for today: each game's win "
    "probabilities, predicted and most-likely scores, predicted run totals, the "
    "full per-inning scoring probabilities (away / home / either team, innings "
    "1-9), starting pitchers, venue/weather, the model's fair betting lines, the "
    "average ESPN sportsbook lines, the model's edge vs the market, recent "
    "accuracy, and model facts. You DO have all of this — use it freely and cite "
    "the specific numbers (including the inning-by-inning breakdown) when asked. "
    "Answer from this data; if something truly isn't in it, say so rather than "
    "guessing. Be concise and concrete. You are not a licensed advisor; the "
    "betting edges are the model's view vs the market, not guaranteed bets."
)


def stream_chat(messages: list, context: str):
    """Yield assistant text deltas for a conversation, given a system context.

    messages: [{role: 'user'|'assistant', content: str}, ...]
    """
    payload = {
        "model": _OLLAMA_MODEL,
        "messages": [{"role": "system", "content": f"{_SYSTEM_PREAMBLE}\n\n{context}"}] + messages,
        "stream": True,
        "options": {"num_ctx": 8192, "temperature": 0.6},
    }
    try:
        resp = _requests.post(_OLLAMA_CHAT_URL, json=payload, stream=True, timeout=180)
        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta
            if chunk.get("done"):
                return
    except Exception as exc:
        yield f"\n\nCouldn't reach Ollama ({exc}). Make sure it's running (`ollama serve`)."
