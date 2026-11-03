"""Adversarial disconfirmation.

Runs after all four agents have committed and before resolution. For each
agent that positively claimed its disease, the evidence it confirmed on is
challenged; challenges that survive are applied and that agent's verifier is
re-run unchanged. An agent that no longer claims its disease drops out, and
resolution decides over whoever is left. The model only surfaces candidate
problems; Python re-decides.

The challenge implemented here is anchored retraction. It generalizes the
per-iteration anatomical anchor from imaging findings to the flat clinical and
criterion-level confirming findings: a confirming finding listed in the anchor
table whose evidence span carries no term from the required territory has its
support neutralized before the verifier re-runs. The step is fully
deterministic.

Anchor vocabularies follow each condition's defining anatomy in the source
guidelines and are not derived from the benchmark.

    Disconfirmer(kb_registry, adapter).disconfirm(multi_result)
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from src.adapter import Adapter
from src.agents.disease_agent_base import (
    AgentResult,
    _serialize_verifier_result,
)
from src.record import verify_source_span
from src.kb_loader import KBRegistry
from src.label_mapper import LabelMapper, LabelMapperError
from src.evidence_ledger import _deep_merge_with_confidence
from src.llm_client import LLMClient
from src.multi_agent import MultiAgentResult
from src.verifier import Verifier


# Localizing-anchor table (guideline-grounded; flat confirming-finding keys)
#
# For each disease, the confirming findings that anatomically localize, with two
# lowercase-substring term sets:
#   required    -- region/organ terms consistent with this finding's criterion
#   conflicting -- region/organ terms that belong to a DIFFERENT region/organ
#
# Retraction rule (conservative): a confirming finding (present=True) is
# retracted iff its evidence_span contains a `conflicting` term AND no `required`
# term -- i.e. the span POSITIVELY localizes to the wrong place. A span with no
# region token at all (vague or even self-contradictory, e.g. "soft, non
# tender") is NOT retracted: span-grounding cannot prove misattribution without
# a competing localization, and retracting vague spans was observed to nuke a
# correct-but-evidence-poor true positive. It still catches the clear
# cross-organ misattributions: RUQ credited as RLQ, and colonic CT or
# lower-abdominal pain credited as pancreatic.
#
# Only flat present-based criterion keys are anchored here (nested imaging
# findings are organ-anchored elsewhere: diverticulitis CT per-iteration anchor;
# cholecystitis CT gallbladder-anchor in the Verifier). Region vocabularies are
# anatomical (WSES/Atlanta disease definitions), NOT MIMIC-tuned.
LOCALIZING_ANCHORS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    # WSES appendicitis: RLQ / McBurney / right iliac fossa.
    "appendicitis": {
        k: {
            "required":    ("rlq", "right lower", "right iliac", "mcburney"),
            "conflicting": ("ruq", "right upper", "right hypochond", "llq",
                            "left lower", "left iliac", "luq", "left upper",
                            "epigastr", "upper abdom"),
        }
        for k in ("pain_migrates_to_rlq", "rlq_tenderness",
                  "localized_guarding_rlq", "rebound_tenderness_mcburney")
    },
    # Revised Atlanta / WSES pancreatitis: epigastric/upper-abdominal pain
    # radiating to the back; AP_C3 imaging is of the pancreas.
    "pancreatitis": {
        "AP_C1_PAIN": {
            "required":    ("epigastr", "upper abdom", "periumbilical", "back",
                            "band-like", "band like"),
            "conflicting": ("rlq", "right lower", "right iliac", "llq",
                            "left lower", "left iliac", "lower abdom",
                            "pelvic", "suprapubic", "mcburney"),
        },
        "AP_C3_IMAGING": {
            "required":    ("pancrea", "peripancreatic"),
            "conflicting": ("colon", "colonic", "sigmoid", "pericolic",
                            "rectosig", "appendix", "periappendiceal",
                            "gallbladder", "gall bladder", "cholecyst",
                            "diverticul", "renal", "ureter"),
        },
    },
    # WSES diverticulitis: LLQ / sigmoid / left iliac fossa.
    "diverticulitis": {
        k: {
            "required":    ("llq", "left lower", "left iliac", "sigmoid",
                            "descending colon"),
            "conflicting": ("rlq", "right lower", "right iliac", "ruq",
                            "right upper", "luq", "left upper", "epigastr",
                            "upper abdom", "mcburney"),
        }
        for k in ("left_lower_quadrant_pain", "llq_tenderness", "llq_tenderness_only")
    },
    # WSES ACC: RUQ / right hypochondrium. (Murphy is a named sign, not a region,
    # so it is intentionally not region-anchored here. Epigastric is NOT marked
    # conflicting -- early ACC pain can be epigastric.)
    "cholecystitis": {
        "ruq_pain_or_tenderness": {
            "required":    ("ruq", "right upper", "right hypochond", "subcostal"),
            "conflicting": ("llq", "left lower", "left iliac", "rlq",
                            "right lower", "right iliac", "luq", "left upper",
                            "lower abdom", "pelvic", "suprapubic", "mcburney"),
        },
    },
}


STATUS_ANCHORED = "anchored"
STATUS_RETRACTED = "RETRACTED_ANCHOR_MISSING"


# Contradictory features surfaced by the model (pass 2)
#
# Per-disease PREDICATE strong-differential features the Verifier consumes to
# trigger branch A ("alternative_diagnosis"). The disconfirmer LLM may assert
# one ONLY with a verbatim record span; verified assertions are injected as
# present=True and the unchanged Verifier re-decides (the LLM never decides).
# Scope: only diseases whose framework has a strong-differential gate
# (cholecystitis, appendicitis). Numeric-threshold features (e.g.
# lipase_3x_uln_pattern) are EXCLUDED -- they need value/ref_upper, not a bare
# present flag. n_of_m (pancreatitis) and multi_evidence (diverticulitis) have
# no contradictory slot -> pass 2 does not apply, and anchored retraction covers them.
# Descriptions are grounded in each KB's contradictory_and_uncertainty_layer.
_CONTRADICTORY_SURFACE: dict[str, dict[str, str]] = {
    "cholecystitis": {
        "epigastric_rather_than_ruq":
            "pain is described as EPIGASTRIC and explicitly NOT right-upper-quadrant",
        "charcot_triad_pattern":
            "Charcot triad (RUQ pain + jaundice + fever) suggesting cholangitis, not ACC",
        "normal_us_with_persistent_pain":
            "ultrasound is explicitly normal/negative despite persistent pain",
    },
    "appendicitis": {
        "dominant_watery_diarrhea_or_viral_syndrome":
            "dominant watery diarrhea or a viral/gastroenteritis syndrome",
        "flank_pain_hematuria_or_colicky_urinary_pain":
            "flank pain, hematuria, or colicky urinary pain (urologic cause)",
        "pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass":
            "pelvic pain with cervical-motion tenderness, positive pregnancy test, or adnexal mass (gynecologic cause)",
        "known_crohn_or_terminal_ileitis_features":
            "known Crohn disease or terminal-ileitis features",
    },
}


# Relevance and negation guard
#
# The pass-2 model call span-verifies PRESENCE of text, not its POLARITY/relevance, so it
# surfaced contradictory features from spans that say the opposite (chronic
# constipation read as watery diarrhoea), that are incidental (an unrelated
# renal cyst read as flank pain with haematuria), or that are explicitly
# negative (a report of no adnexal abnormality read as an adnexal mass), which
# wrongly eliminated a true appendicitis case. The guard requires the span to (a) contain >=1 feature-relevant
# keyword AND (b) for POSITIVE-presence features, NOT be negated. The one
# inherently-negative feature (normal_us_with_persistent_pain) sets
# negation_rejects=False (its span legitimately describes a normal/negative US).
_C2_NEGATION_CUES: tuple[str, ...] = (
    "no ", "not ", "without", "denies", "denied", "negative", "unremarkable",
    " absent", "no evidence", "ruled out", "rule out", "r/o",
)
_C2_FEATURE_GUARD: dict[str, dict[str, Any]] = {
    # 'epigastric rather than RUQ' legitimately negates RUQ in its span, so the
    # negation reject must be off (require the positive 'epigastric' keyword).
    "epigastric_rather_than_ruq":
        {"require": ("epigastr",), "negation_rejects": False},
    "charcot_triad_pattern":
        {"require": ("charcot", "jaundice"), "negation_rejects": True},
    "normal_us_with_persistent_pain":
        {"require": ("ultrasound", "sonograph", "u/s"), "negation_rejects": False},
    "dominant_watery_diarrhea_or_viral_syndrome":
        {"require": ("diarrhea", "loose stool", "watery", "viral", "gastroenter"),
         "negation_rejects": True},
    "flank_pain_hematuria_or_colicky_urinary_pain":
        {"require": ("flank", "hematuria", "colicky", "dysuria", "costovertebral",
                     "urinary"), "negation_rejects": True},
    "pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass":
        {"require": ("pelvic", "cervical motion", "adnexal", "pregnan"),
         "negation_rejects": True},
    "known_crohn_or_terminal_ileitis_features":
        {"require": ("crohn", "terminal ileitis", "terminal ileum",
                     "inflammatory bowel"), "negation_rejects": True},
}


def c2_feature_supported(feature: str, span: str) -> bool:
    """Does `span` genuinely support `feature` (relevant keyword
    present, and not negated for positive-presence features)? Unknown features
    are rejected (conservative)."""
    guard = _C2_FEATURE_GUARD.get(feature)
    if guard is None:
        return False
    s = (span or "").lower()
    if not any(t in s for t in guard["require"]):
        return False
    if guard["negation_rejects"] and any(c in s for c in _C2_NEGATION_CUES):
        return False
    return True


# Result type

@dataclasses.dataclass
class DisconfirmationAudit:
    """Per-candidate record of what disconfirmation did."""

    disease: str
    label_before: str
    label_after: str
    eliminated: bool
    retractions: list[dict[str, Any]]            # {finding_key, evidence_span, status}
    injected_features: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # {feature, evidence_span, span_verified}


# Disconfirmer

class Disconfirmer:
    """Adversarial disconfirmation over the agent results."""

    def __init__(
        self,
        kb_registry: KBRegistry,
        adapter: Adapter,
        *,
        llm_client: LLMClient | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.kb_registry = kb_registry
        self.adapter = adapter
        self.label_mapper = LabelMapper(kb_registry)
        # pass 2 (Pass 2) runs only when an LLM client is supplied. Without it the
        # Disconfirmer performs deterministic anchored retraction only.
        self.llm = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def disconfirm(
        self, multi_result: MultiAgentResult,
    ) -> tuple[MultiAgentResult, list[DisconfirmationAudit]]:
        """Return a new MultiAgentResult with candidates re-verified after
        anchored retraction, plus a per-candidate audit list.

        Non-candidates are passed through unchanged. Candidates whose
        localizing confirming findings fail their anchors are re-verified and
        may be eliminated (label -> insufficient).
        """
        new_results: dict[str, AgentResult] = dict(multi_result.results)
        audits: list[DisconfirmationAudit] = []

        for disease in multi_result.candidates:
            result = multi_result.results[disease]
            updated, audit = self._disconfirm_candidate(disease, result)
            new_results[disease] = updated
            audits.append(audit)

        surviving_candidates = [
            d for d in multi_result.agent_order
            if new_results[d].outcome_label == d
        ]
        new_multi = MultiAgentResult(
            hadm_id=multi_result.hadm_id,
            results=new_results,
            candidates=surviving_candidates,
            agent_order=list(multi_result.agent_order),
        )
        return new_multi, audits

    # ---- internals ----

    def _disconfirm_candidate(
        self, disease: str, result: AgentResult,
    ) -> tuple[AgentResult, DisconfirmationAudit]:
        anchors = LOCALIZING_ANCHORS.get(disease, {})
        merged_findings = self._merged_findings_with_spans(result)

        # Decide retractions from the spans of confirming localizing findings.
        # Retract iff the span positively localizes to a CONFLICTING region and
        # carries no REQUIRED region term (see LOCALIZING_ANCHORS docstring).
        retractions: list[dict[str, Any]] = []
        retract_keys: list[str] = []
        for key, spec in anchors.items():
            leaf = merged_findings.get(key)
            if not isinstance(leaf, dict):
                continue
            if leaf.get("present") is not True:
                continue  # only retract findings that positively support the confirm
            span = (leaf.get("evidence_span") or "").lower()
            has_required = any(t in span for t in spec["required"])
            has_conflict = any(t in span for t in spec["conflicting"])
            if has_conflict and not has_required:
                retract_keys.append(key)
                status = STATUS_RETRACTED
            else:
                status = STATUS_ANCHORED
            retractions.append({
                "finding_key":  key,
                "evidence_span": leaf.get("evidence_span") or "",
                "status":       status,
            })

        # Pass 2 (pass 2): LLM-surfaced contradictory features (only if a client is
        # configured and this disease has a strong-differential gate).
        injected: list[dict[str, Any]] = []
        if self.llm is not None and disease in _CONTRADICTORY_SURFACE:
            injected = self._surface_contradictions(disease, result)

        label_before = result.outcome_label
        if not retract_keys and not injected:
            return result, DisconfirmationAudit(
                disease=disease, label_before=label_before, label_after=label_before,
                eliminated=False, retractions=retractions, injected_features=injected,
            )

        # Apply both passes to the Verifier-input evidence and re-verify ONCE.
        evidence = dict(result.ledger.current_evidence())
        for key in retract_keys:
            evidence[key] = {"present": None, "evidence_span": "", "confidence": "uncertain"}
        for feat in injected:
            evidence[feat["feature"]] = {
                "present": True,
                "evidence_span": feat["evidence_span"],
                "confidence": "explicit",
            }

        kb = self.kb_registry.kb(disease)
        vresult = Verifier(kb).verify(evidence)
        vdict = _serialize_verifier_result(vresult)
        try:
            label_after = self.label_mapper.to_outcome_label(disease, vresult.diagnostic_output)
        except LabelMapperError:
            label_after = "insufficient_evidence"

        updated = dataclasses.replace(
            result, final_verifier_result=vdict, outcome_label=label_after,
        )
        return updated, DisconfirmationAudit(
            disease=disease, label_before=label_before, label_after=label_after,
            eliminated=(label_after != disease), retractions=retractions,
            injected_features=injected,
        )

    # ---- Pass 2: contradictory features surfaced by the model ----

    def _surface_contradictions(
        self, disease: str, result: AgentResult,
    ) -> list[dict[str, Any]]:
        """Ask the LLM which predicate strong-differential features the record
        supports; keep only those with a record-verifiable evidence_span.
        Returns [{feature, evidence_span, span_verified}].
        """
        catalog = _CONTRADICTORY_SURFACE[disease]
        record_text = self._reconstruct_record_text(result)
        prompt = self._build_disconfirm_prompt(disease, catalog, record_text)
        system = (
            "You are an adversarial clinical reviewer. Your job is to DISCONFIRM a "
            f"tentative diagnosis of {disease} by finding record evidence for specific "
            "contradictory features. Assert a feature ONLY if the record EXPLICITLY "
            "supports it, and quote a VERBATIM substring of the record as evidence. "
            "If none apply, return an empty list. Respond with JSON only."
        )
        try:
            resp = self.llm.complete(
                prompt, system_prompt=system, max_tokens=self.max_tokens,
                temperature=self.temperature, json_only=True,
            )
        except Exception:  # noqa: BLE001 -- pass 2 is best-effort; a failed LLM call
            # must never sink an otherwise-good case (agent + resolution
            # results already stand). Surface no features and continue.
            return []
        parsed = _extract_json_features(resp.text or "")
        out: list[dict[str, Any]] = []
        for item in parsed:
            feat = item.get("feature")
            span = (item.get("evidence_span") or "").strip()
            if feat not in catalog or not span:
                continue
            if not verify_source_span(span, record_text):
                continue  # span not record-grounded
            if not c2_feature_supported(feat, span):
                continue  # Span doesn't genuinely support the feature
                          # (irrelevant / opposite-polarity / negated)
            out.append({"feature": feat, "evidence_span": span, "span_verified": True})
        return out

    def _build_disconfirm_prompt(
        self, disease: str, catalog: dict[str, str], record_text: str,
    ) -> str:
        feats = "\n".join(f'  - "{k}": {v}' for k, v in catalog.items())
        return (
            f"Tentative diagnosis under review: {disease}.\n\n"
            "Contradictory features to look for (assert only if explicitly supported):\n"
            f"{feats}\n\n"
            "=== Patient record ===\n"
            f"{record_text}\n\n"
            "=== Output ===\n"
            "Return JSON: {\"features\": [{\"feature\": \"<one key above>\", "
            "\"evidence_span\": \"<verbatim record quote>\"}]}. "
            "Only include features the record EXPLICITLY supports; otherwise return "
            "{\"features\": []}."
        )

    def _reconstruct_record_text(self, result: AgentResult) -> str:
        """Record text the candidate was shown: HPI + every tool observation."""
        parts: list[str] = []
        try:
            case = self.adapter.load_case(result.hadm_id)
            hpi = case.get("Patient History") or ""
            if hpi:
                parts.append(hpi)
        except Exception:  # noqa: BLE001
            pass
        for rec in result.ledger.iterations:
            if rec.observation:
                parts.append(rec.observation)
        return "\n".join(parts)

    def _merged_findings_with_spans(self, result: AgentResult) -> dict[str, Any]:
        """Replay the EvidenceLedger's confidence-ranked merge over raw findings_delta to
        recover merged leaves WITH evidence_span (current_evidence() drops spans).
        """
        merged: dict[str, Any] = {}
        for rec in result.ledger.iterations:
            if rec.findings_delta:
                _deep_merge_with_confidence(merged, rec.findings_delta)
        return merged


# Module-level helpers

def _extract_json_features(text: str) -> list[dict[str, Any]]:
    """Tolerantly pull the contradiction list out of an LLM JSON response.

    Accepts ``{"features": [...]}`` or a bare ``[...]``; tolerates code fences
    and surrounding prose. Returns [] on any parse failure (no LLM text ever
    decides anything -- a malformed response simply surfaces no features).
    """
    if not text:
        return []
    s = text.strip()
    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    def _coerce(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, dict):
            feats = obj.get("features")
            return [x for x in feats if isinstance(x, dict)] if isinstance(feats, list) else []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []

    try:
        return _coerce(json.loads(s))
    except (ValueError, TypeError):
        pass
    # Fallback: grab the first {...} or [...] block.
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, s, flags=re.DOTALL)
        if m:
            try:
                return _coerce(json.loads(m.group(0)))
            except (ValueError, TypeError):
                continue
    return []
