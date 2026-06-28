"""Conversational assistant backed by the local Ollama model.

`stream_chat` proxies a conversation to Ollama's /api/chat with a system prompt
(the day's data + history + model facts, built by the app) and streams the
reply back as plain-text deltas. The dashboard renders these in a chat widget.
"""
import json
import os

import requests as _requests

_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# qwen2.5:14b reasons noticeably better than llama3.1:8b on these structured
# betting questions and fits 16GB. Override with OLLAMA_MODEL to fall back
# (e.g. qwen2.5:7b for less memory pressure, or llama3.1:8b for speed).
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

_SYSTEM_PREAMBLE = (
    "You are the assistant built into an MLB prediction dashboard. The data block "
    "below contains everything the site serves for today: each game's win "
    "probabilities, predicted and most-likely scores, predicted run totals, the "
    "full per-inning scoring probabilities (away / home / either team, innings "
    "1-9), starting pitchers, venue/weather, the model's fair betting lines, the "
    "average ESPN sportsbook lines, the model's edge vs the market, recent "
    "accuracy, and model facts. You DO have all of this — use it freely and cite "
    "the specific numbers (including the inning-by-inning breakdown) when asked.\n\n"
    "RULES — follow exactly:\n"
    "- Use ONLY the numbers in the data block. NEVER invent, round, or guess an "
    "odds price — quote the exact market price shown. If a number isn't in the "
    "data, say so rather than guessing.\n"
    "- For any bet question, follow each game's COMPUTED VERDICT verbatim: it names "
    "the value side and its exact price, or says NO BET. Never recommend a side or "
    "price the verdict doesn't give, and never reverse the side. When asked WHY, use "
    "the verdict's WHY text as-is.\n"
    "- Moneyline signs are literal: a favorite is NEGATIVE (e.g. -138), an underdog "
    "POSITIVE (e.g. +120). Quote every price exactly as written, sign included. NEVER "
    "flip a sign, convert -X to +X, or say a price 'should be' a number not in the "
    "data. The fair price and market price are both given — don't invent a third.\n"
    "- Stay consistent across the whole conversation. Do not flip-flop your pick; if "
    "the user corrects a number, re-read the data block rather than inventing a new "
    "stance.\n"
    "- Be concise and concrete. You are not a licensed advisor; the edges are the "
    "model's view vs the market, not guaranteed bets."
)


def stream_chat(messages: list, context: str):
    """Yield assistant text deltas for a conversation, given a system context.

    messages: [{role: 'user'|'assistant', content: str}, ...]
    """
    payload = {
        "model": _OLLAMA_MODEL,
        "messages": [{"role": "system", "content": f"{_SYSTEM_PREAMBLE}\n\n{context}"}] + messages,
        "stream": True,
        "options": {"num_ctx": 8192, "temperature": 0.2},
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
