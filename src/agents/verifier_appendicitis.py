"""Appendicitis verifier factory.

All verifier logic lives in `src/verifier.py`, dispatched on the framework type
declared by the KB. This module is the named entry point for the appendicitis
KB, whose framework is the WSES 2020 tailored stepwise risk stratification.
"""

from __future__ import annotations

from src.kb_loader import KBRegistry
from src.verifier import Verifier


def make_appendicitis_verifier(registry: KBRegistry) -> Verifier:
    """Construct a Verifier bound to the appendicitis KB."""
    return Verifier(registry.kb("appendicitis"))
