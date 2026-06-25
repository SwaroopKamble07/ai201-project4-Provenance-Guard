"""Three-variant transparency label generator.

The exact headline + body strings from planning.md are encoded as
module constants. The `label_for()` function maps a confidence score
to the correct variant using the same asymmetric thresholds as the
combiner in `scoring.py` so the attribution and the label agree.

A score > 0.70 → likely AI label
A score < 0.30 → likely human label
Else            → uncertain label
"""

from __future__ import annotations

from typing import TypedDict


class Label(TypedDict):
    headline: str
    body: str


LABELS: dict[str, Label] = {
    "likely_ai": {
        "headline": "Likely AI-generated",
        "body": (
            "Independent analysis of this submission suggests it was very "
            "likely produced by an AI writing assistant. Signals used: "
            "stylometric analysis and a language-model assessment. If you "
            "are the creator and believe this is incorrect, you may submit "
            "an appeal from your dashboard."
        ),
    },
    "likely_human": {
        "headline": "Likely human-written",
        "body": (
            "Both the language-model assessment and stylometric analysis "
            "suggest this text was written by a human. This is a "
            "probabilistic judgment, not a guarantee - AI-assisted writing "
            "is not always detectable."
        ),
    },
    "uncertain": {
        "headline": "Uncertain - verification recommended",
        "body": (
            "The two analysis signals used by Provenance Guard disagree on "
            "this submission, or both are weak. We cannot make a confident "
            "attribution in either direction. Readers should treat the "
            "authorship of this piece as unverified."
        ),
    },
}


def label_for(confidence: float) -> Label:
    """Return the appropriate transparency label for a confidence score.

    Thresholds match the attribution bands from scoring.combine_signals
    so that the response's `attribution` matches the chosen label.
    """
    if confidence > 0.70:
        return LABELS["likely_ai"]
    if confidence < 0.30:
        return LABELS["likely_human"]
    return LABELS["uncertain"]


def all_variants() -> dict[str, Label]:
    """Return all three labels keyed by attribution band (for tests/README)."""
    return {k: dict(v) for k, v in LABELS.items()}
