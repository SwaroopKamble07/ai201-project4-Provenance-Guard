"""Detection signals for Provenance Guard.

Each signal exposes a single function that accepts a string of text
and returns a float in [0.0, 1.0] where 0.0 = confidently human-written
and 1.0 = confidently AI-generated.
"""

from .signal_groq import groq_signal
from .signal_stylometry import stylometric_signal

__all__ = ["groq_signal", "stylometric_signal"]
