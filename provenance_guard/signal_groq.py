"""Signal 1 — Groq LLM judge (semantic signal).

Prompts Llama-3.3-70b-versatile with a structured JSON contract that
asks the model to return a single float `ai_probability` in [0, 1].
The function parses the model's response, clamps the value, and returns
it. If parsing fails for any reason (network, JSON, malformed float),
the function returns 0.5 (maximum uncertainty) so the rest of the
pipeline degrades gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are an AI-vs-human writing classifier. Read the text and return
ONLY a JSON object of the form {{"ai_probability": <float between 0 and 1>}}
with no commentary, where 0.0 means certainly human-written and 1.0
means certainly AI-generated. Consider hedging phrases, generic
structure, uniform register, and whether the text looks like a polished
LLM draft.

Text:
\"\"\"
{text}
\"\"\"
"""


def _extract_float(raw: str) -> Optional[float]:
    """Find the first plausible float in a model response."""
    match = re.search(r"-?\d+\.\d+|-?\d+", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def groq_signal(text: str) -> float:
    """Return the Groq LLM judge's ai_probability in [0.0, 1.0].

    Falls back to 0.5 on any error so the rest of the pipeline sees
    a maximum-uncertainty value rather than crashing or pretending
    the signal is unavailable.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        log.warning("GROQ_API_KEY not set; returning neutral signal value 0.5")
        return 0.5

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You respond only with valid JSON. No prose.",
                },
                {
                    "role": "user",
                    "content": _PROMPT_TEMPLATE.format(text=text),
                },
            ],
            temperature=0.0,
            max_tokens=32,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("Groq call failed (%s); returning neutral value 0.5", exc)
        return 0.5

    value: Optional[float] = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "ai_probability" in parsed:
            value = float(parsed["ai_probability"])
    except (json.JSONDecodeError, ValueError, TypeError):
        value = _extract_float(raw)

    if value is None:
        log.warning("Could not parse AI-probability from Groq response: %r", raw[:80])
        return 0.5

    return max(0.0, min(1.0, value))
