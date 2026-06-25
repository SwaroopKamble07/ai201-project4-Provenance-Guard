"""Signal 2 — Stylometric heuristics (structural signal).

Computes three metrics in pure Python per the planning.md spec:

1. Sentence-length variance (SLV): std-dev of sentence word lengths,
   normalized by clamp(slv / 12.0, 0, 1). Higher variance -> lower
   AI-likelihood, so the *score* (0=human, 1=AI) is slv_score directly.

2. Type-token ratio (TTR): unique_words / total_words, case-insensitive
   and punctuation-stripped. We invert so low TTR (more repetitive)
   scores closer to 1: clamp(1.0 - (ttr - 0.30) / 0.30, 0, 1).

3. Punctuation density: punctuation marks per 100 words across the
   set . , ; : ! ? - —. AI text -> ~4/100; human text -> ~7/100.
   pd_score = clamp(1.0 - (pd - 3.0) / 5.0, 0, 1).

The averaged score is (slv_score + ttr_score + pd_score) / 3.0.

Edge case: text with fewer than 3 sentences or fewer than 30 total
words returns 0.5 (no opinion) — structural statistics aren't
meaningful on fragments, so a neutral value is safer than a guess.
"""

from __future__ import annotations

import re
import statistics
import string
from typing import List

# Sentence terminators used by the splitter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Word = a run of letters or apostrophes (handles contractions, hyphenated surnames, etc.).
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Punctuation set from the spec.
_PUNCT_CHARS = set(". , ; : ! ? - —".replace(" ", ""))
_PUNCT_RE = re.compile(f"[{re.escape(''.join(sorted(_PUNCT_CHARS)))}]")


def _split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    return raw or ([text.strip()] if text.strip() else [])


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _slv_score(sentences: List[str]) -> float:
    """Lower SLV (more uniform) -> closer to 1 (AI-like). Higher SLV -> 0 (human-like)."""
    lengths = [len(_words(s)) for s in sentences if _words(s)]
    if len(lengths) < 2:
        return 0.5
    try:
        sd = statistics.pstdev(lengths)
    except statistics.StatisticsError:
        return 0.5
    slv_score = max(0.0, min(1.0, sd / 12.0))
    # slv_score already is "high SD -> high value", but per spec higher-variance = human,
    # so the *AI-likelihood* score inverts: high SD -> low AI score.
    return 1.0 - slv_score


def _ttr_score(words: List[str]) -> float:
    if not words:
        return 0.5
    lowered = [w.lower() for w in words]
    unique = len(set(lowered))
    total = len(lowered)
    ttr = unique / total
    # Per spec: TTR <= 0.30 -> 1.0 (max AI); TTR >= 0.60 -> 0.0
    return max(0.0, min(1.0, 1.0 - (ttr - 0.30) / 0.30))


def _punctuation_density_score(text: str, words: List[str]) -> float:
    if not words:
        return 0.5
    punct_count = sum(1 for ch in text if ch in _PUNCT_CHARS)
    per_100 = punct_count * 100.0 / len(words)
    return max(0.0, min(1.0, 1.0 - (per_100 - 3.0) / 5.0))


def stylometric_signal(text: str) -> float:
    """Return the structural AI-likelihood score in [0.0, 1.0].

    Uses three metrics per the planning.md spec, averaged uniformly.
    Returns 0.5 if the text is too short (<3 sentences or <30 words)
    for structural statistics to be meaningful.
    """
    if not text or not text.strip():
        return 0.5

    sentences = _split_sentences(text)
    words = _words(text)

    if len(sentences) < 3 or len(words) < 30:
        return 0.5

    slv = _slv_score(sentences)
    ttr = _ttr_score(words)
    pd = _punctuation_density_score(text, words)

    score = (slv + ttr + pd) / 3.0
    return max(0.0, min(1.0, score))
