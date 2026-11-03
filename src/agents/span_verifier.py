"""Evidence grounding gate.

Every finding the model reports carries a quoted `evidence_span`. This module
checks each span against the record text the model has actually been shown
before the ledger is allowed to merge the iteration.

The gate exists because the check is not hypothetical. On a pancreatitis case
whose examination described only diffuse, non-localized tenderness, and whose
record contained no occurrence of "RLQ", "right lower", "McBurney" or "rebound",
the 8B model reported right-lower-quadrant tenderness with rebound and quoted a
span the record did not contain. Without a deterministic gate the appendicitis
verifier would have confirmed on it.

Findings are downgraded, never deleted, so the audit trail keeps both the claim
and the reason it was not trusted:

  Present findings with an unmatched span lose their assertion and drop to
  uncertain. An empty span alongside a claim is recorded but left standing,
  since a model can have legitimate reasons not to quote.

  Numeric findings get the span rule plus a value check. A number that appears
  nowhere in the record under any common formatting zeroes the whole leaf, since
  a fabricated value is worse than a missing one.

  Imaging findings are boolean, so only the span rule applies.

Composite branches are walked recursively for the leaves inside them; bare
scalars pass through untouched.

    verify_findings_delta(findings_delta, record_text, fuzzy_threshold=0.85)
        -> (verified_findings, audit_entries)

Each audit entry records the criterion path, the span, a status, whether the
finding was downgraded, and the leaf shape.
"""

from __future__ import annotations

import re
from typing import Any

from src.record import value_in_text, verify_source_span


# Status codes

STATUS_VERIFIED = "verified"
STATUS_SPAN_UNVERIFIED = "SPAN_UNVERIFIED"
STATUS_EMPTY_SPAN_WITH_CLAIM = "EMPTY_SPAN_WITH_CLAIM"
STATUS_NUMERIC_FABRICATION = "NUMERIC_FABRICATION_RISK"
STATUS_SPAN_OK_VALUE_BAD = "SPAN_VERIFIED_VALUE_UNVERIFIED"
STATUS_VALUE_UNIT_RECONCILED = "SPAN_VERIFIED_VALUE_UNIT_RECONCILED"


# Unit-aware value reconciliation (WBC K/uL <-> /uL, x1000)

# Factors applied to numbers parsed from a VERIFIED evidence_span when
# reconciling against the LLM's normalized `value`. Scope:
# WBC K/uL -> /uL is x1000 and is the only empirically load-bearing
# conversion (blocks ~2/7 cholecystitis confirms; also bites appendicitis
# WBC). CRP mg/dL -> mg/L (x10) is a deliberate one-line add the day a case
# needs it -- omitted now to avoid an untested path. This is arithmetic
# (1 K/uL == 1000 /uL), NOT a clinical threshold, so it needs no guideline
# citation. See finding-value-verifier-unit-conversion-downgrade.
_UNIT_CONVERSION_FACTORS: tuple[float, ...] = (1000.0,)

_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+")


def _value_reconciles_with_span(value: float, span: str) -> bool:
    """True iff `value` equals some numeric token in `span` times one of the
    allowed unit-conversion factors (within 0.5% relative tolerance).

    Caller MUST only invoke this when the span itself has been verified
    against the record (span_ok is True): span-derived numbers are trusted
    only after the span text is confirmed present. A fabricated `value` that
    bears no x-factor relationship to its quoted span still returns False and
    is zeroed by the normal numeric guard, so fabrication detection is intact.
    """
    if not span:
        return False
    for tok in _NUMERIC_TOKEN_RE.findall(span):
        try:
            base = float(tok)
        except ValueError:
            continue
        for factor in _UNIT_CONVERSION_FACTORS:
            tol = max(1.0, abs(value) * 0.005)
            if abs(value - base * factor) <= tol:
                return True
    return False


# Shape detection

def _is_present_leaf(v: Any) -> bool:
    return (
        isinstance(v, dict)
        and "present" in v
        and "value" not in v  # not NumericFinding
    )


def _is_numeric_leaf(v: Any) -> bool:
    return isinstance(v, dict) and "value" in v and not isinstance(v.get("value"), bool)


def _is_ct_leaf(v: Any) -> bool:
    # CtFinding: value is bool|None; no "present" key.
    return (
        isinstance(v, dict)
        and "value" in v
        and (isinstance(v.get("value"), bool) or v.get("value") is None)
        and "present" not in v
    )


def _is_any_leaf(v: Any) -> bool:
    return _is_present_leaf(v) or _is_numeric_leaf(v) or _is_ct_leaf(v)


# Public API

def verify_findings_delta(
    findings_delta: dict[str, Any],
    record_text: str,
    *,
    fuzzy_threshold: float = 0.85,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (verified_findings, audit) per the policy in the module docstring.

    The input dict is NOT mutated; a deep copy is returned with downgrades
    applied in place on the copy. This way the Agent's IterationRecord
    can keep the raw LLM emission AND the verified post-gate version separately
    if needed (we currently only persist the verified version into the EvidenceLedger,
    and the raw is in iter_NNN_parsed.json -- so the audit lets you trace any
    downgrade back to its origin).
    """
    if not isinstance(findings_delta, dict):
        return findings_delta, []

    import copy
    verified = copy.deepcopy(findings_delta)
    audit: list[dict[str, Any]] = []
    _walk(verified, record_text, audit, path="", fuzzy_threshold=fuzzy_threshold)
    return verified, audit


# Walker (internal)

def _walk(
    node: dict[str, Any],
    record_text: str,
    audit: list[dict[str, Any]],
    *,
    path: str,
    fuzzy_threshold: float,
) -> None:
    for key, val in list(node.items()):
        sub_path = f"{path}.{key}" if path else key
        if _is_any_leaf(val):
            _verify_leaf(node, key, sub_path, record_text, audit, fuzzy_threshold)
        elif isinstance(val, dict):
            # Composite (imaging, scores, ct_findings, us_findings, mri_findings)
            _walk(val, record_text, audit, path=sub_path, fuzzy_threshold=fuzzy_threshold)
        # Scalars (population_context, age_lt_40, imaging.ct_done): no shape to verify.


def _verify_leaf(
    parent: dict[str, Any],
    key: str,
    full_path: str,
    record_text: str,
    audit: list[dict[str, Any]],
    fuzzy_threshold: float,
) -> None:
    leaf = parent[key]
    span = (leaf.get("evidence_span") or "")
    is_present = _is_present_leaf(leaf)
    is_numeric = _is_numeric_leaf(leaf)
    is_ct = _is_ct_leaf(leaf)
    shape = "present" if is_present else ("numeric" if is_numeric else ("ct" if is_ct else "scalar"))

    # Step 1: span check (universal across shapes).
    span_ok = True
    span_status = STATUS_VERIFIED
    if span:
        span_ok = verify_source_span(span, record_text, fuzzy_threshold=fuzzy_threshold)
        if not span_ok:
            span_status = STATUS_SPAN_UNVERIFIED
    else:
        # No span at all. Distinguish "empty-default" (present is None, value is None)
        # from "claim-without-span" (present True/False OR value not None).
        is_claiming = False
        if is_present and leaf.get("present") is not None:
            is_claiming = True
        if is_numeric and leaf.get("value") is not None:
            is_claiming = True
        if is_ct and leaf.get("value") is not None:
            is_claiming = True
        if is_claiming:
            span_status = STATUS_EMPTY_SPAN_WITH_CLAIM
            # We don't downgrade here -- the LLM might have legitimate reasons
            # to make a claim from observation context. Audit only.

    if not span_ok:
        # Downgrade by shape.
        if is_present:
            leaf["present"] = None
        elif is_numeric:
            # Don't fully zero on span-fail-only; preserve value if value
            # itself is verifiable (handled next step).
            leaf["present"] = None
            # leave value untouched here -- the numeric check below will
            # decide whether to zero it
        elif is_ct:
            leaf["value"] = None
        leaf["confidence"] = "uncertain"
        leaf["span_verified"] = False

    # Step 2: numeric value-support check (NumericFinding only).
    final_status = span_status
    downgraded = not span_ok
    if is_numeric:
        v = leaf.get("value")
        if v is not None:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                vf = None
            if vf is not None and not value_in_text(vf, record_text):
                if span_ok and _value_reconciles_with_span(vf, span):
                    # Value is grounded as a unit-conversion of the verified
                    # span (e.g. value=11500 /uL from record text "11.5 K/uL").
                    # Keep the leaf intact; record the reconciliation so the
                    # analysis can count how often it fires.
                    final_status = STATUS_VALUE_UNIT_RECONCILED
                else:
                    # Numeric fabrication: zero the entire leaf.
                    leaf["value"] = None
                    leaf["ref_upper"] = None
                    leaf["unit"] = ""
                    leaf["present"] = None
                    leaf["evidence_span"] = ""
                    leaf["confidence"] = "uncertain"
                    leaf["span_verified"] = False
                    downgraded = True
                    if span_ok:
                        final_status = STATUS_SPAN_OK_VALUE_BAD
                    else:
                        final_status = STATUS_NUMERIC_FABRICATION

    audit.append({
        "criterion_path": full_path,
        "evidence_span":  span,
        "status":         final_status,
        "downgraded":     downgraded,
        "shape":          shape,
    })
