"""Mapping knowledge base outputs to the evaluation label space.

Each KB emits disease-specific outputs such as `confirmed_acute_pancreatitis`
or `probable_ACC`. Evaluation happens in a five-outcome space: the four
conditions plus `insufficient_evidence`. This module holds that mapping table
and the abstention flag that goes with each entry.

The mapping is strict. Only the formal diagnostic outputs are mapped; severity
grades, complication flags and treatment recommendations are metadata and raise
rather than being silently coerced into a diagnosis. At construction the mapper
cross-checks itself against the loaded KBs in both directions, so an output
added to a KB without a mapping, or a mapping left behind after a KB edit, is
caught at startup.

    LabelMapper(registry).map(disease, kb_output) -> MappedLabel
    LabelMapper(registry).to_outcome_label(disease, kb_output) -> str
    LabelMapper(registry).is_abstention(disease, kb_output) -> bool
"""

from __future__ import annotations

from typing import NamedTuple

from src.kb_loader import KBRegistry


# The five-outcome evaluation label space.

OUTCOME_LABELS = frozenset(
    {
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
        "insufficient_evidence",
    }
)


# The mapping table
# Format: { (disease, kb_output): (outcome_label, abstention_flag) }
# Locked against kb-v4-frozen diagnostic_outputs (verified at LabelMapper
# construction via cross-check against KBRegistry).
LABEL_MAP: dict[tuple[str, str], tuple[str, bool]] = {
    # ---- Appendicitis (5 outputs) ----
    ("appendicitis", "confirmed_or_highly_likely_acute_appendicitis"):
        ("appendicitis", False),
    ("appendicitis", "possible_complicated_appendicitis"):
        ("appendicitis", False),
    ("appendicitis", "intermediate_risk_suspected_appendicitis_needs_imaging_or_observation"):
        ("insufficient_evidence", True),
    ("appendicitis", "low_risk_appendicitis_unlikely"):
        ("insufficient_evidence", True),
    ("appendicitis", "unsupported_or_alternative_diagnosis_more_likely"):
        ("insufficient_evidence", True),

    # ---- Cholecystitis (5 outputs) ----
    ("cholecystitis", "confirmed_or_highly_supported_ACC"):
        ("cholecystitis", False),
    ("cholecystitis", "probable_ACC"):
        ("cholecystitis", False),
    ("cholecystitis", "possible_ACC_needs_more_workup"):
        ("insufficient_evidence", True),
    ("cholecystitis", "alternative_diagnosis_more_likely"):
        ("insufficient_evidence", True),
    ("cholecystitis", "insufficient_evidence"):
        ("insufficient_evidence", True),

    # ---- Diverticulitis (6 outputs) ----
    ("diverticulitis", "ct_confirmed_diverticulitis_uncomplicated"):
        ("diverticulitis", False),
    ("diverticulitis", "ct_confirmed_diverticulitis_complicated"):
        ("diverticulitis", False),
    ("diverticulitis", "high_clinical_probability_imaging_pending"):
        ("insufficient_evidence", True),
    ("diverticulitis", "suspected_diverticulitis_needs_imaging"):
        ("insufficient_evidence", True),
    ("diverticulitis", "unsupported_or_alternative_diagnosis_more_likely"):
        ("insufficient_evidence", True),
    ("diverticulitis", "uncertain_insufficient_evidence"):
        ("insufficient_evidence", True),

    # ---- Pancreatitis (4 outputs) ----
    ("pancreatitis", "confirmed_acute_pancreatitis"):
        ("pancreatitis", False),
    ("pancreatitis", "possible_acute_pancreatitis_insufficient_evidence"):
        ("insufficient_evidence", True),
    ("pancreatitis", "not_supported_by_available_evidence"):
        ("insufficient_evidence", True),
    ("pancreatitis", "uncertain_due_to_missing_data"):
        ("insufficient_evidence", True),
}


# Types

class MappedLabel(NamedTuple):
    """Result of mapping a KB diagnostic_output to the five-outcome label space."""

    outcome_label: str          # one of OUTCOME_LABELS
    abstention: bool         # True if five-outcome == 'insufficient_evidence'
    source_disease: str      # the KB that produced the original label
    source_label: str        # the original KB-native label


class LabelMapperError(KeyError):
    """Raised when a (disease, kb_output) tuple has no entry in LABEL_MAP.

    Under strict mapping this covers:
    - Typos in the KB output label
    - Extension labels (severity / CBDS / complication / action) -- these are
    NOT diagnostic outputs and must be handled separately by resolution
    logic, not mapped to a five-outcome label.
    """


# LabelMapper

class LabelMapper:
    """Maps KB diagnostic_outputs to the five-outcome evaluation label space.

    Strict mode: only the 20 entries in LABEL_MAP are recognized. Extension
    labels (e.g., 'severe_acute_pancreatitis_severity_label',
    'complicated_ACC_red_flag', 'mandatory_ct_per_wses') raise
    LabelMapperError on attempted mapping.

    Cross-check: If a KBRegistry is supplied at construction, the
    mapper verifies every KB diagnostic_output has a LABEL_MAP entry AND
    every LABEL_MAP entry references a real KB output. Catches drift in
    both directions.

    Example:
    registry = KBRegistry("diagnostic_extracts")
    mapper = LabelMapper(registry)
    result = mapper.map("pancreatitis", "confirmed_acute_pancreatitis")
    assert result.outcome_label == "pancreatitis"
    assert result.abstention is False
    """

    LABEL_MAP = LABEL_MAP  # expose as class attribute for easy introspection

    def __init__(self, registry: KBRegistry | None = None) -> None:
        if registry is not None:
            self._cross_check_completeness(registry)

    # ---- core mapping API ----

    def map(self, disease: str, kb_output: str) -> MappedLabel:
        """Map a (disease, kb_output) pair to a MappedLabel.

        Raises:
            LabelMapperError: if the pair is not in LABEL_MAP (strict mode --
            includes any extension label).
        """
        key = (disease, kb_output)
        if key not in LABEL_MAP:
            raise LabelMapperError(
                f"no LABEL_MAP entry for ({disease!r}, {kb_output!r}). "
                "Either this is an extension label (severity / CBDS / "
                "complication / action -- not a diagnostic output) and "
                "should be handled by resolution logic, or the label space "
                "needs updating."
            )
        outcome_label, abstention = LABEL_MAP[key]
        return MappedLabel(
            outcome_label=outcome_label,
            abstention=abstention,
            source_disease=disease,
            source_label=kb_output,
        )

    def to_outcome_label(self, disease: str, kb_output: str) -> str:
        """Shortcut: return only the five-outcome label."""
        return self.map(disease, kb_output).outcome_label

    def is_abstention(self, disease: str, kb_output: str) -> bool:
        """Shortcut: return only the abstention flag."""
        return self.map(disease, kb_output).abstention

    # ---- introspection ----

    @staticmethod
    def all_mappings() -> dict[tuple[str, str], tuple[str, bool]]:
        """Return a copy of the full mapping table."""
        return dict(LABEL_MAP)

    @staticmethod
    def outcome_label_space() -> frozenset[str]:
        """The 5 valid output labels."""
        return OUTCOME_LABELS

    # -- cross-check --

    @staticmethod
    def _cross_check_completeness(registry: KBRegistry) -> None:
        """Verify LABEL_MAP completeness in BOTH directions:
          - Every KB diagnostic_output has a LABEL_MAP entry
          - Every LABEL_MAP entry references a real KB output

        Raises LabelMapperError listing all drift if any found.
        """
        kb_outputs: set[tuple[str, str]] = set()
        for disease in registry.diseases():
            kb = registry.kb(disease)
            for output in kb.diagnostic_outputs:
                kb_outputs.add((disease, output))

        map_keys = set(LABEL_MAP.keys())

        kb_not_mapped = sorted(kb_outputs - map_keys)
        map_no_kb = sorted(map_keys - kb_outputs)

        problems: list[str] = []
        if kb_not_mapped:
            problems.append(
                f"KB outputs missing from LABEL_MAP: {kb_not_mapped}"
            )
        if map_no_kb:
            problems.append(
                f"LABEL_MAP entries that don't match any KB output: {map_no_kb}"
            )
        if problems:
            raise LabelMapperError(
                "LabelMapper / KBRegistry drift detected:\n  "
                + "\n  ".join(problems)
                + "\nFix: update the LABEL_MAP in src/label_mapper.py to match "
                "the knowledge bases, or update the knowledge bases to match it."
            )

        # Also verify every mapped label is in OUTCOME_LABELS
        for (disease, kb_output), (outcome_label, _) in LABEL_MAP.items():
            if outcome_label not in OUTCOME_LABELS:
                raise LabelMapperError(
                    f"LABEL_MAP[({disease!r}, {kb_output!r})] maps to {outcome_label!r} "
                    f"which is not in OUTCOME_LABELS {sorted(OUTCOME_LABELS)}"
                )
