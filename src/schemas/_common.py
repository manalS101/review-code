"""Shared schema primitives.

Findings are rich by design: each one carries the quoted `evidence_span`, a
`span_verified` flag and a confidence level alongside the value itself, so the
audit trail survives all the way to the report. A thin
`to_verifier_evidence()` on each model strips that down to the minimal dict the
verifier reads.

Three finding shapes cover everything the KBs need: `PresentFinding` for binary
criteria, where `None` means indeterminate rather than absent; `NumericFinding`
for lab values and scores, with an optional upper reference bound used by the
pancreatitis enzyme threshold; and `CtFinding` for imaging.

Every model sets `extra="ignore"`. Small models add keys the schema never
declared, and dropping them is better than failing validation on an otherwise
usable extraction. The verifier reads only the keys it knows about, so its
contract is unaffected.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError as _PydanticValidationError


# Coercion logging
#
# When the shape-coercion validators silently coerce a value (dict -> None, list
# -> None, uncoercible string -> None, etc.), there is currently NO signal
# that something was lost. One appendicitis case passed with six coerced
# CtFinding values while the audit reported a clean extraction.
#
#  fixes this by routing each coercion event through a ContextVar that
# the Extractor sets up before `schema_cls.model_validate(payload)` and
# reads after. The captured list is then surfaced through SpanVerificationStats
# (-> per-case JSON's `span_verification` block -> audit pipeline).
#
# A ContextVar rather than a global: it is standard library, and it scopes
# cleanly to one extraction through context push and pop.
#
# Each log entry is a dict with keys:
#   - "class":    the schema class name (e.g., "CtFinding")
#   - "field":    the field name (e.g., "value")
#   - "mode":     which validator fired -- "before" (non-primitive) or
#                 "wrap" (uncoercible primitive)
#   - "raw_repr": truncated repr of the raw value the LLM emitted (<=200 chars)
#
# When the ContextVar is None (default), validators silently coerce as before
# WITHOUT logging -- this preserves test isolation and means the coercion log
# is opt-in. The Extractor opts in by initializing it per-extraction.

_COERCION_LOG: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "_coercion_log", default=None,
)


def _log_coercion(*, class_name: str, field_name: str, mode: str, raw_value: Any) -> None:
    """Push one coercion entry into the active ContextVar (no-op if inactive).

    Safe to call from validators that run with no surrounding context (e.g.,
    in unit tests). Truncates raw_value repr to <=200 chars to avoid log
    bloat from deeply nested LLM payloads.
    """
    log = _COERCION_LOG.get()
    if log is None:
        return
    try:
        r = repr(raw_value)
    except Exception:  # noqa: BLE001 -- raw could be a weird object
        r = f"<unrepr-able {type(raw_value).__name__}>"
    if len(r) > 200:
        r = r[:197] + "..."
    log.append({
        "class": class_name,
        "field": field_name,
        "mode": mode,
        "raw_repr": r,
    })


def _coerce_schema_fragment_to_none(v: Any) -> Any:
    """Coerce any non-primitive value to None so the per-field type coercion
    has a clean input to work with.

    Catches several distinct model failure modes seen during development:

    1. Full JSON Schema fragments copied verbatim from the inlined schema:
    `{"default": false, "enum": [true, false, null]}` or
    `{"anyOf": [{"type": "boolean"}, {"type": "null"}]}`. This killed a
    diverticulitis extraction on `pericolic_abscess.value`.

    2. Bare `enum` lists copied without the wrapping dict, e.g.
    `[true, false, null]` written directly as the `value`. Suspected
    cause of a repeated `bool_parsing` failure on
    `appendix_outer_diameter_gt_6_to_9_mm.value`: same field, same error,
    three separate runs.

    3. Nested wrapper dicts like `{"value": true}` for a leaf field.

    4. Any other complex shape (tuples, sets, custom objects) the LLM
    might invent under stress.

    The v2 of this helper returns None for ANY value that is
    not a primitive (None, bool, int, float, str). Primitives flow through
    unchanged so pydantic's normal coercion can handle them (e.g., "true"
    -> True, "60.0" -> 60.0). Strings that pydantic CANNOT coerce will still
    raise the usual ValidationError -- that is the correct behaviour, since
    a string like "8 mm" written into a bool field is a model error worth
    surfacing, not silently hiding.

    Corresponding prompt edit: the "LITERAL VALUES ONLY" rule in
    the agent prompt tells the model not to emit non-primitives in the first
    place; this validator is the belt-and-suspenders defense.

    Note: this is the mode='before' fast-path that handles non-primitive
    shapes (dict/list/tuple/...). A complementary mode='wrap' validator
    (see `_lenient_coerce_or_none`) catches the remaining failure mode:
    primitive STRINGS that pydantic itself cannot coerce to the target
    type (e.g., "8 mm" or ">=6mm" written into a bool field by the LLM
    interpreting a measurement-named binary field literally). The two
    layers together produce extraction that never aborts on a single bad
    leaf field -- the field becomes "indeterminate" instead.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    # dict, list, tuple, set, frozenset, or any other complex type -> None.
    # The caller will also call _log_coercion(...) to surface this.
    return None


def coerce_done_flag(v: Any) -> Any:
    """Coerce a dict-wrapped scalar to a bool for imaging ``*_done`` flags.

    The 8B sometimes emits imaging modality-done flags (``ct_done`` / ``us_done``
    / ``hida_done`` / ``mri_done``) as a FINDING-shaped dict instead of a bare
    boolean, e.g. ``{"present": false, "evidence_span":..., "confidence":...}``
    or ``{"value": true, "confidence":...}``. These fields are declared plain
    ``bool``, so the dict reaches pydantic and raises ``bool_type`` -- which,
    uncaught at the final ``current_evidence`` call, crashed whole cases on the
    review run. This
    extracts the inner bool from ``value``/``present``; a dict carrying no usable
    bool collapses to ``False`` (the field default). Non-dict values pass through
    to pydantic's normal coercion (so "true"/1 still work). Generalized: applies
    to every imaging ``*_done`` flag across all disease schemas.

    ``None`` -> ``False``: a null done-flag means "modality
    not done". Without this, a ``None`` reaches the plain ``bool`` field and raises
    ``bool_type``, which -- uncaught at the final ``current_evidence`` call --
    forces the whole agent onto the defensive fallback (insufficient_evidence).
    Validated no-false-confirm on the fallback cases checked for this change. (The fallback
    still catches OTHER malformed-field modes; this removes only the None one so a
    true-disease agent is not silently lost to a stray null done-flag.)
    """
    if v is None:
        return False
    if isinstance(v, dict):
        for k in ("value", "present"):
            inner = v.get(k)
            if isinstance(inner, bool):
                return inner
        return False
    return v


def _lenient_coerce_or_none(v: Any, handler) -> Any:
    """mode='wrap' validator body: try pydantic's normal coercion; on any
    ValidationError, return None.

    Catches primitive strings that pydantic cannot coerce to the target type,
    the confirmed cause of a repeated appendicitis failure on
    `appendix_outer_diameter_gt_6_to_9_mm.value`. mode='before' handles
    dict/list shapes but a string like "8 mm" passes through as a string
    and bool_parsing fails on it.

    Tradeoff: this also silently zeros out genuine type errors (e.g., a
    typo'd "1z2" written into a float field). That is acceptable for the
    L2 Extractor's contract -- a single bad leaf becoming "indeterminate"
    is the same semantic as "I couldn't extract this", which is exactly
    the desired behaviour for noisy LLM output. The corresponding prompt
    edit ("LITERAL VALUES ONLY") still tells the LLM not to emit bad
    shapes in the first place; this validator only catches the residual.
    """
    try:
        return handler(v)
    except _PydanticValidationError:
        return None


class Confidence(str, Enum):
    """when source-span verification fails, confidence is
    force-downgraded to UNCERTAIN regardless of what the LLM reported.
    """

    EXPLICIT  = "explicit"     # verbatim or near-verbatim in record
    INFERRED  = "inferred"     # reasonable inference from record text
    UNCERTAIN = "uncertain"    # weak, span unverifiable, or LLM unsure


class _AuditBase(BaseModel):
    """Common audit fields for every extracted finding.

    `extra="ignore"`: LLMs sometimes add extra keys (e.g., putting `present`
    on a CtFinding). Silently drop them rather than failing validation.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    evidence_span: str = Field(
        default="",
        description="Verbatim or fuzzy-matched record span supporting this finding.",
    )
    confidence: Confidence = Field(default=Confidence.UNCERTAIN)
    span_verified: bool = Field(
        default=True,
        description=(
            "Set False by the Extractor's source-span verifier if "
            "evidence_span was not found in the record text. "
            "confidence is also forced to UNCERTAIN on span verification failure."
        ),
    )

    # A model reporting honestly that it has no data sometimes passes
    # `evidence_span: null`, which crashed Pydantic
    # with `string_type` errors and lost a whole iteration of findings.
    # Coerce None to "".
    @field_validator("evidence_span", mode="before")
    @classmethod
    def _coerce_none_evidence_span(cls, v: Any) -> Any:
        return "" if v is None else v


class PresentFinding(_AuditBase):
    """Binary present/absent finding with audit fields.

    `present` may be None to indicate the field wasn't extracted (Verifier
    treats None as indeterminate).
    """

    present: bool | None = Field(
        default=None,
        description="True / False / None (None = not extracted / indeterminate).",
    )

    #  defense +  logging:
    #   - mode='before' catches non-primitive shapes (dict/list/tuple)
    #   - mode='wrap'   catches uncoercible primitives (e.g., "8 mm")
    # Both modes call _log_coercion when they actually coerce, so the
    # Extractor's ContextVar-collected log surfaces them via .
    @field_validator("present", mode="before")
    @classmethod
    def _coerce_schema_fragment(cls, v: Any) -> Any:
        coerced = _coerce_schema_fragment_to_none(v)
        if coerced is None and v is not None:
            _log_coercion(class_name="PresentFinding", field_name="present",
                          mode="before", raw_value=v)
        return coerced

    @field_validator("present", mode="wrap")
    @classmethod
    def _lenient_coerce(cls, v: Any, handler) -> Any:
        try:
            return handler(v)
        except _PydanticValidationError:
            _log_coercion(class_name="PresentFinding", field_name="present",
                          mode="wrap", raw_value=v)
            return None


class NumericFinding(_AuditBase):
    """Numeric finding (lab values, scores) with optional present-flag.

    Some lab keys are checked by the Verifier via `_present_bool` (e.g.,
    cholecystitis `elevated_wbc`) which prefers `present` over `value`. This
    model carries BOTH so the to_verifier_evidence method can include
    whichever the Verifier needs.

    `ref_upper` (used by the AP_C2_ENZYMES numeric cross-check): optional
    upper bound of the lab's normal reference range, as extracted verbatim
    from record context (e.g., "ref 0-60" -> ref_upper=60.0). Currently only
    consumed by the pancreatitis Verifier for the 3x ULN check on lipase /
    amylase. Other diseases inherit the field but ignore it. Per locked
    decision LLM populates from record text; never from clinical
    knowledge.
    """

    value: float | None = Field(default=None)
    unit: str = Field(default="")

    # Coerce None to "" so an honest "no unit" emission does not crash
    # validation. Mirror of evidence_span coercion in _AuditBase.
    @field_validator("unit", mode="before")
    @classmethod
    def _coerce_none_unit(cls, v: Any) -> Any:
        return "" if v is None else v

    ref_upper: float | None = Field(
        default=None,
        description=(
            "Optional. Upper bound of the lab's normal reference range, "
            "extracted verbatim from record text. Used by Verifier the numeric cross-check "
            "threshold cross-checks (e.g., AP_C2_ENZYMES 3x ULN)."
        ),
    )
    present: bool | None = Field(
        default=None,
        description=(
            "Optional. Used by Verifier checks that read `present` rather "
            "than `value` (e.g., cholecystitis elevated_wbc). When set, "
            "should be consistent with the numeric value if known."
        ),
    )

    #  defense +  logging.
    @field_validator("value", "ref_upper", "present", mode="before")
    @classmethod
    def _coerce_schema_fragment(cls, v: Any, info) -> Any:
        coerced = _coerce_schema_fragment_to_none(v)
        if coerced is None and v is not None:
            _log_coercion(class_name="NumericFinding",
                          field_name=info.field_name, mode="before", raw_value=v)
        return coerced

    @field_validator("value", "ref_upper", "present", mode="wrap")
    @classmethod
    def _lenient_coerce(cls, v: Any, handler, info) -> Any:
        try:
            return handler(v)
        except _PydanticValidationError:
            _log_coercion(class_name="NumericFinding",
                          field_name=info.field_name, mode="wrap", raw_value=v)
            return None


class CtFinding(_AuditBase):
    """Imaging finding (True/False/None) with confidence.

    Mirrors the Verifier's `_normalize_ct_finding` accepted shape exactly:
    `{value: bool|None, confidence: str}` plus the audit extras inherited from
    `_AuditBase`. The Verifier's confidence-gated branches read the
    `confidence` field directly.
    """

    value: bool | None = Field(default=None)

    #  defense +  logging.
    # Examples seen during development:
    #   - `pericolic_abscess.value` got a JSON Schema fragment (dict)
    #     and killed a diverticulitis extraction
    #   - `appendix_outer_diameter_gt_6_to_9_mm.value` got "8 mm" or similar
    #     uncoercible string, which killed an appendicitis extraction until
    #     the mode='wrap' layer was added
    @field_validator("value", mode="before")
    @classmethod
    def _coerce_schema_fragment(cls, v: Any) -> Any:
        coerced = _coerce_schema_fragment_to_none(v)
        if coerced is None and v is not None:
            _log_coercion(class_name="CtFinding", field_name="value",
                          mode="before", raw_value=v)
        return coerced

    @field_validator("value", mode="wrap")
    @classmethod
    def _lenient_coerce(cls, v: Any, handler) -> Any:
        try:
            return handler(v)
        except _PydanticValidationError:
            _log_coercion(class_name="CtFinding", field_name="value",
                          mode="wrap", raw_value=v)
            return None


class CaseProvenance(BaseModel):
    """Per-case provenance for L2 traceability.

    `extra="ignore"` for robustness; provenance is always injected by the
    Extractor itself, so the LLM shouldn't be writing this -- but tolerate it
    if it tries.
    """

    model_config = ConfigDict(extra="ignore")

    hadm_id: int
    extraction_model: str            # e.g., "stub" | "meta-llama/Meta-Llama-3.1-8B-Instruct"
    kb_tag: str                      # e.g., "kb-v4-frozen"
    extraction_timestamp: str        # ISO 8601 UTC

    @classmethod
    def now(
        cls,
        *,
        hadm_id: int,
        extraction_model: str,
        kb_tag: str,
    ) -> "CaseProvenance":
        return cls(
            hadm_id=hadm_id,
            extraction_model=extraction_model,
            kb_tag=kb_tag,
            extraction_timestamp=datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        )


# Default-factory helpers (used by per-disease extraction schemas to
# default any unprovided field to an "empty" Finding rather than failing)

def empty_present() -> PresentFinding:
    """Default factory: an unextracted PresentFinding (present=None)."""
    return PresentFinding()


def empty_numeric() -> NumericFinding:
    """Default factory: an unextracted NumericFinding (value=None)."""
    return NumericFinding()


def empty_ct() -> CtFinding:
    """Default factory: an unextracted CtFinding (value=None)."""
    return CtFinding()


# Helpers for to_verifier_evidence() implementations

def present_to_verifier(f: PresentFinding) -> dict[str, Any]:
    """Reduce a PresentFinding to the verifier-input shape: {present, evidence_span, confidence}."""
    return {
        "present": f.present,
        "evidence_span": f.evidence_span,
        "confidence": f.confidence.value,
    }


def numeric_to_verifier(f: NumericFinding) -> dict[str, Any]:
    """Reduce a NumericFinding to the verifier-input shape.

    `ref_upper` is included unconditionally (may be None). The Verifier's
    numeric cross-check for AP_C2_ENZYMES requires both `value` and `ref_upper`
    to fire; absence of either falls back to the LLM's `present` flag.
    """
    out: dict[str, Any] = {
        "value": f.value,
        "unit": f.unit,
        "ref_upper": f.ref_upper,
        "evidence_span": f.evidence_span,
        "confidence": f.confidence.value,
    }
    if f.present is not None:
        out["present"] = f.present
    return out


def ct_to_verifier(f: CtFinding) -> dict[str, Any]:
    """Reduce a CtFinding to the verifier-input shape: {value, confidence}."""
    return {
        "value": f.value,
        "confidence": f.confidence.value,
    }
