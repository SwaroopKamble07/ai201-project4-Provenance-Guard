"""Confidence scoring — combines the two signal scores per spec.

Milestone 3 uses a tiny placeholder that returns llm_score directly so
the submission flow can be exercised end-to-end. The full asymmetric
weighted-average combiner with disagreement penalty ships in Milestone 4.
"""

from __future__ import annotations


def combine_signals(llm_score: float, stylometric_score: float) -> dict:
    """Combine two signal scores into a final confidence + attribution.

    M3 placeholder: passes the LLM score through verbatim; Milestone 4
    wires this to the spec's full combiner.
    """
    confidence = max(0.0, min(1.0, llm_score))
    if confidence > 0.70:
        attribution = "likely_ai"
    elif confidence < 0.30:
        attribution = "likely_human"
    else:
        attribution = "uncertain"
    return {"confidence": confidence, "attribution": attribution}
