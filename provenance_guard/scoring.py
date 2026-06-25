"""Confidence scoring — combines the two signal scores per planning.md.

The full combiner is implemented here (Milestone 4); the placeholder
in earlier milestones routed the LLM score through directly.

Algorithm per spec:

    raw = 0.6 * llm_score + 0.4 * struct_score
    if |llm_score - struct_score| > 0.3:
        raw = raw * 0.85 + 0.5 * 0.15    # pull 15% toward 0.5 on disagreement
    confidence = clamp(raw, 0, 1)

    if confidence > 0.70:           attribution = "likely_ai"
    elif confidence < 0.30:         attribution = "likely_human"
    else:                           attribution = "uncertain"

The LLM judge gets more weight because semantic evidence carries more
information than three structural metrics. The thresholds are
asymmetric (0.30 / 0.70) to reflect the false-positive asymmetry.
"""

from __future__ import annotations


def combine_signals(llm_score: float, stylometric_score: float) -> dict:
    """Combine two signal scores into confidence + attribution.

    Lifted verbatim from planning.md so future spec edits to the
    threshold logic have a single source of truth.
    """
    raw = 0.6 * llm_score + 0.4 * stylometric_score

    if abs(llm_score - stylometric_score) > 0.3:
        # disagreement penalty: pull 15% toward 0.5
        raw = raw * 0.85 + 0.5 * 0.15

    confidence = max(0.0, min(1.0, raw))

    if confidence > 0.70:
        attribution = "likely_ai"
    elif confidence < 0.30:
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    return {"confidence": confidence, "attribution": attribution}
