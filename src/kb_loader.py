"""Knowledge base loading and validation.

Reads the four guideline knowledge bases from disk, validates their structure
at load time, and exposes them as typed objects to the verifier and the
agents. Validation is not optional: a KB missing a required section or
declaring an unexpected framework type raises immediately rather than failing
silently mid-run.

    KBRegistry(kb_dir).kb(disease) -> KB
    KBRegistry(kb_dir).diseases() -> list[str]
    KB.static_backbone() -> str   # the criteria block injected into prompts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Locked invariants

EXPECTED_FRAMEWORK_TYPES = {
    "appendicitis": "tailored_stepwise_risk_stratification",
    "pancreatitis": "n_of_m_rule",
    "cholecystitis": "integrated_assessment",
    "diverticulitis": "multi_evidence_assessment",
}

# Required top-level sections (mirrors scripts/validate_kb.py REQUIRED_TOP_LEVEL).
# The Loader's accessors require these to be present and non-empty where applicable.
REQUIRED_TOP_LEVEL = [
    "kb_id",
    "schema_version",
    "disease",
    "sources",
    "scientific_guardrails",
    "diagnostic_framework",
    "agent_verifier",
    "clinical_concepts",
    "laboratory_evidence",
    "imaging_evidence",
    "evidence_interpretation_layer",
    "contradictory_and_uncertainty_layer",
    "clinical_nlp_context_layer",
    "source_audit",
]

# One JSON file per disease, relative to kb_dir.
KB_FILENAMES = {disease: f"{disease}.json" for disease in EXPECTED_FRAMEWORK_TYPES}


# Exceptions

class KBValidationError(ValueError):
    """Raised when a KB JSON fails the minimum structural checks the Loader
    requires. Triggered at KBRegistry construction time (eager + always-validate)."""


# KB (one disease)

class KB:
    """One validated knowledge base wrapping the raw JSON dict.

    Use typed accessors (`kb.decision_algorithm`, `kb.patterns`, etc.) rather
    than reaching into `kb.raw` directly -- the accessors give type safety
    and IDE autocomplete.
    """

    def __init__(self, disease: str, raw: dict[str, Any]) -> None:
        self._disease = disease
        self._raw = raw
        self._static_backbone_cache: str | None = None
        self._validate_minimum_structure()

    # ---- Top-level identifiers ----

    @property
    def disease(self) -> str:
        """Disease project_label (e.g., 'pancreatitis')."""
        return self._disease

    @property
    def kb_id(self) -> str:
        return str(self._raw["kb_id"])

    @property
    def schema_version(self) -> str:
        return str(self._raw["schema_version"])

    @property
    def canonical_name(self) -> str:
        """Medical canonical name (e.g., 'acute_pancreatitis')."""
        return str(self._raw.get("disease", {}).get("canonical_name", ""))

    @property
    def raw(self) -> dict[str, Any]:
        """The full underlying KB dict. Escape hatch for sections without
        typed accessors."""
        return self._raw

    # ---- diagnostic_framework ----

    @property
    def framework_type(self) -> str:
        """One of n_of_m_rule | tailored_stepwise_risk_stratification |
        integrated_assessment | multi_evidence_assessment."""
        return str(self._raw["diagnostic_framework"]["framework_type"])

    @property
    def primary_rule(self) -> dict[str, Any]:
        """The disease-specific primary diagnostic rule dict."""
        return dict(self._raw["diagnostic_framework"]["primary_rule"])

    @property
    def diagnostic_outputs(self) -> list[str]:
        """Valid output labels this KB can emit (e.g.,
        ['confirmed_acute_pancreatitis', 'possible_acute_pancreatitis_insufficient_evidence', ...])."""
        return list(self._raw["diagnostic_framework"]["diagnostic_outputs"])

    @property
    def minimum_agent_output_fields(self) -> list[str]:
        """Field names every agent output must include."""
        return list(self._raw["diagnostic_framework"]["minimum_agent_output_fields"])

    # ---- agent_verifier (most critical -- consumed by deterministic Verifier) ----

    @property
    def decision_algorithm(self) -> list[dict[str, Any]]:
        """Ordered list of decision steps for the Verifier to walk.
        Each step has {step, name, logic, source}."""
        return list(self._raw["agent_verifier"]["decision_algorithm"])

    @property
    def abstention_rules(self) -> list[str]:
        return list(self._raw["agent_verifier"]["abstention_rules"])

    @property
    def minimum_safe_output_contract(self) -> dict[str, Any]:
        return dict(self._raw["agent_verifier"]["minimum_safe_output_contract"])

    @property
    def verifier_input_schema(self) -> dict[str, Any]:
        return dict(self._raw["agent_verifier"]["input_schema"])

    # ---- evidence_interpretation_layer ----

    @property
    def patterns(self) -> list[dict[str, Any]]:
        """Evidence patterns used by conviction scoring."""
        return list(self._raw["evidence_interpretation_layer"]["patterns"])

    @property
    def patterns_global_rules(self) -> list[str]:
        return list(
            self._raw["evidence_interpretation_layer"].get("global_rules", [])
        )

    # ---- contradictory_and_uncertainty_layer ----

    @property
    def contradictions(self) -> list[dict[str, Any]]:
        """Features that argue against this disease, used by disconfirmation."""
        return list(
            self._raw["contradictory_and_uncertainty_layer"][
                "contradictory_or_confidence_lowering_features"
            ]
        )

    @property
    def required_agent_behavior(self) -> list[str]:
        return list(
            self._raw["contradictory_and_uncertainty_layer"].get(
                "required_agent_behavior", []
            )
        )

    # ---- Reference content (used by Retriever later) ----

    @property
    def clinical_concepts(self) -> dict[str, Any]:
        return dict(self._raw["clinical_concepts"])

    @property
    def laboratory_evidence(self) -> dict[str, Any]:
        return dict(self._raw["laboratory_evidence"])

    @property
    def imaging_evidence(self) -> dict[str, Any]:
        return dict(self._raw["imaging_evidence"])

    @property
    def scientific_guardrails(self) -> dict[str, Any]:
        return dict(self._raw["scientific_guardrails"])

    @property
    def clinical_nlp_context_layer(self) -> dict[str, Any]:
        return dict(self._raw["clinical_nlp_context_layer"])

    # ---- Optional disease-specific extras ----

    def has(self, section: str) -> bool:
        """Check if an optional top-level section exists in this KB."""
        return section in self._raw

    def optional(self, section: str) -> dict[str, Any] | None:
        """Return an optional section's dict if present, else None."""
        v = self._raw.get(section)
        return dict(v) if isinstance(v, dict) else None

    # ---- Static backbone (~3,400-token prompt-injection block) ----

    def static_backbone(self, *, include_outputs: bool = True) -> str:
        """Markdown-style text block ready for LLM prompt injection.

        the static backbone includes:
        - diagnostic_framework.framework_type + primary_rule
        - diagnostic_framework.diagnostic_outputs
        - diagnostic_framework.minimum_agent_output_fields
        - agent_verifier.decision_algorithm
        - agent_verifier.abstention_rules
        - scientific_guardrails

        ``include_outputs`` (default True = deployed behaviour): when False, the
        "Diagnostic outputs (valid labels this KB can emit)" block is omitted. Used
        ONLY by the clean LLM-only ablation so the model is not primed to parrot a KB
        output key (e.g. ``confirmed_acute_pancreatitis``) as its free-text Final
        Diagnosis. The diagnostic *criteria* the
        agent reasons over are unchanged; only the output-label menu is hidden.

        The default (full) variant is cached on first call. Approximate length:
        ~3,000-4,000 tokens depending on KB content size.
        """
        # Only the deployed full variant is cached; the suppressed variant is fresh.
        if include_outputs and self._static_backbone_cache is not None:
            return self._static_backbone_cache

        lines: list[str] = [
            f"# Knowledge Base -- {self.disease}",
            f"(kb_id: {self.kb_id}, schema_version: {self.schema_version},",
            f" canonical_name: {self.canonical_name})",
            "",
            "## Diagnostic Framework",
            f"Framework type: {self.framework_type}",
            "",
        ]

        # Primary rule -- compact JSON dump for the LLM to parse the structure
        lines.append("### Primary rule")
        lines.append("```json")
        lines.append(json.dumps(self.primary_rule, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        # Valid diagnostic outputs (label vocabulary). Suppressed in the clean
        # LLM-only ablation (include_outputs=False) so the model is not primed to
        # emit a KB output key as its free-text Final Diagnosis.
        if include_outputs:
            lines.append("### Diagnostic outputs (valid labels this KB can emit)")
            for label in self.diagnostic_outputs:
                lines.append(f"- {label}")
            lines.append("")

        # Minimum agent output fields
        lines.append("### Minimum agent output fields (required keys in every agent output)")
        for field in self.minimum_agent_output_fields:
            lines.append(f"- {field}")
        lines.append("")

        # Agent verifier -- decision algorithm
        lines.append("## Agent Verifier")
        lines.append("")
        lines.append("### Decision algorithm (Verifier walks these in order)")
        for step in self.decision_algorithm:
            step_num = step.get("step", "?")
            step_name = step.get("name", "<unnamed>")
            step_logic = step.get("logic", "")
            step_source = step.get("source", "")
            lines.append(f"Step {step_num} -- {step_name}")
            lines.append(f"  Logic: {step_logic}")
            if step_source:
                lines.append(f"  Source: {step_source}")
            lines.append("")

        # Abstention rules
        lines.append("### Abstention rules")
        for rule in self.abstention_rules:
            lines.append(f"- {rule}")
        lines.append("")

        # Minimum safe output contract
        lines.append("### Minimum safe output contract")
        contract = self.minimum_safe_output_contract
        lines.append("Required output fields:")
        for f in contract.get("required_fields", []):
            lines.append(f"- {f}")
        lines.append("")
        lines.append("Forbidden outputs:")
        for f in contract.get("forbidden_outputs", []):
            lines.append(f"- {f}")
        lines.append("")

        # Scientific guardrails
        lines.append("## Scientific guardrails")
        guardrails = self.scientific_guardrails
        for k, v in guardrails.items():
            if isinstance(v, str):
                lines.append(f"- **{k}**: {v}")
            elif isinstance(v, list):
                lines.append(f"- **{k}**:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
        lines.append("")

        backbone = "\n".join(lines)
        if include_outputs:
            self._static_backbone_cache = backbone
        return backbone

    # ---- internal ----

    def _validate_minimum_structure(self) -> None:
        """Lightweight structural validation -- verifies the KB has what the
        Loader's accessors need. Heavier full schema check is in
        scripts/validate_kb.py.

        Raises:
            KBValidationError: with a clear message naming the missing key.
        """
        # Tier 1: required top-level keys
        for key in REQUIRED_TOP_LEVEL:
            if key not in self._raw:
                raise KBValidationError(
                    f"KB {self._disease!r} missing required top-level key {key!r}"
                )

        # Tier 2: diagnostic_framework substructure
        df = self._raw["diagnostic_framework"]
        for k in ("framework_type", "primary_rule", "diagnostic_outputs",
                 "minimum_agent_output_fields"):
            if k not in df:
                raise KBValidationError(
                    f"KB {self._disease!r}.diagnostic_framework missing {k!r}"
                )

        # framework_type lock per
        expected_ft = EXPECTED_FRAMEWORK_TYPES.get(self._disease)
        actual_ft = df.get("framework_type")
        if expected_ft and actual_ft != expected_ft:
            raise KBValidationError(
                f"KB {self._disease!r} framework_type is {actual_ft!r}, "
                f"expected {expected_ft!r}"
            )

        if not isinstance(df["diagnostic_outputs"], list) or not df["diagnostic_outputs"]:
            raise KBValidationError(
                f"KB {self._disease!r}.diagnostic_framework.diagnostic_outputs "
                "must be a non-empty list"
            )

        # Tier 3: agent_verifier substructure
        av = self._raw["agent_verifier"]
        for k in ("decision_algorithm", "abstention_rules",
                 "minimum_safe_output_contract", "input_schema"):
            if k not in av:
                raise KBValidationError(
                    f"KB {self._disease!r}.agent_verifier missing {k!r}"
                )

        da = av["decision_algorithm"]
        if not isinstance(da, list) or not da:
            raise KBValidationError(
                f"KB {self._disease!r}.agent_verifier.decision_algorithm "
                "must be a non-empty list"
            )
        for i, step in enumerate(da):
            if not isinstance(step, dict):
                raise KBValidationError(
                    f"KB {self._disease!r}.agent_verifier.decision_algorithm[{i}] "
                    "must be a dict, got a string"
                )
            for k in ("step", "name", "logic"):
                if k not in step:
                    raise KBValidationError(
                        f"KB {self._disease!r}.agent_verifier.decision_algorithm[{i}] "
                        f"missing required key {k!r}"
                    )

        # Tier 4: patterns + contradictions
        eil = self._raw["evidence_interpretation_layer"]
        if not isinstance(eil.get("patterns"), list) or not eil["patterns"]:
            raise KBValidationError(
                f"KB {self._disease!r}.evidence_interpretation_layer.patterns "
                "must be a non-empty list"
            )

        cul = self._raw["contradictory_and_uncertainty_layer"]
        contrads = cul.get("contradictory_or_confidence_lowering_features", [])
        if not isinstance(contrads, list) or not contrads:
            raise KBValidationError(
                f"KB {self._disease!r}.contradictory_and_uncertainty_layer."
                "contradictory_or_confidence_lowering_features must be a non-empty list"
            )


# KBRegistry (all 4 KBs)

class KBRegistry:
    """Registry of the four disease knowledge bases.

    Eager loads all 4 KBs at construction and always validates each
    against the Loader's minimum structural requirements. If any KB
    is malformed, construction fails with KBValidationError.

        registry = KBRegistry("knowledge_bases/")
        kb = registry.kb("pancreatitis")
        kb.framework_type      # 'n_of_m_rule'
        registry.diseases      # ['appendicitis', ...]
    """

    def __init__(self, kb_dir: str | Path) -> None:
        self._kb_dir = Path(kb_dir)
        if not self._kb_dir.exists():
            raise FileNotFoundError(f"kb_dir does not exist: {self._kb_dir}")

        # Eager load + validate all 4 ()
        self._kbs: dict[str, KB] = {}
        for disease, filename in KB_FILENAMES.items():
            path = self._kb_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"KB file not found for {disease!r}: {path}"
                )
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self._kbs[disease] = KB(disease, raw)

    # ---- public API ----

    def kb(self, disease: str) -> KB:
        """Get the KB for a given disease."""
        if disease not in self._kbs:
            raise KeyError(
                f"unknown disease {disease!r}; known: {sorted(self._kbs.keys())}"
            )
        return self._kbs[disease]

    def diseases(self) -> list[str]:
        """Return the 4 disease names in canonical order."""
        return list(self._kbs.keys())

    def all_diagnostic_outputs(self) -> dict[str, list[str]]:
        """Return mapping of disease -> list of diagnostic_outputs labels."""
        return {d: kb.diagnostic_outputs for d, kb in self._kbs.items()}

    def kb_ids(self) -> dict[str, str]:
        """Return mapping of disease -> kb_id string."""
        return {d: kb.kb_id for d, kb in self._kbs.items()}

    def __len__(self) -> int:
        return len(self._kbs)

    def __contains__(self, disease: str) -> bool:
        return disease in self._kbs
