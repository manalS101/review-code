"""Cholecystitis verifier factory.

All verifier logic lives in `src/verifier.py`, dispatched on the framework type
declared by the KB. This module is the named entry point for the cholecystitis
KB, whose framework is the WSES 2020 integrated assessment.
"""

from __future__ import annotations

from src.kb_loader import KBRegistry
from src.verifier import Verifier


def make_cholecystitis_verifier(registry: KBRegistry) -> Verifier:
    """Construct a Verifier bound to the cholecystitis KB."""
    return Verifier(registry.kb("cholecystitis"))
