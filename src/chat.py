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
    "You are the assistant built into an MLB prediction dashboard. Answer using "
    "ONLY the data provided below — today's model predictions, betting lines and "
    "edges, recent accuracy, and model facts. If the answer isn't in the data, "
    "say you don't have it rather than guessing. Be concise, concrete, and cite "
    "the numbers. You are not a licensed advisor; the betting edges are the "
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
