"""Pancreatitis verifier factory.

All verifier logic lives in `src/verifier.py`, dispatched on the framework type
declared by the KB. This module is the named entry point for the pancreatitis
KB, whose framework is the Revised Atlanta two-of-three rule.
"""

from __future__ import annotations

from src.kb_loader import KBRegistry
from src.verifier import Verifier


def make_pancreatitis_verifier(registry: KBRegistry) -> Verifier:
    """Construct a Verifier bound to the pancreatitis KB."""
    return Verifier(registry.kb("pancreatitis"))
