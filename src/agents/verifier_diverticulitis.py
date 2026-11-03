"""Diverticulitis verifier factory.

All verifier logic lives in `src/verifier.py`, dispatched on the framework type
declared by the KB. This module is the named entry point for the diverticulitis
KB, whose framework is the WSES 2020 multi-evidence assessment.
"""

from __future__ import annotations

from src.kb_loader import KBRegistry
from src.verifier import Verifier


def make_diverticulitis_verifier(registry: KBRegistry) -> Verifier:
    """Construct a Verifier bound to the diverticulitis KB."""
    return Verifier(registry.kb("diverticulitis"))
