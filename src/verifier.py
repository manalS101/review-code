"""Deterministic verifier.

Given a knowledge base and a set of already-extracted findings, this walks the
KB's decision algorithm and primary rule and returns one of the KB's declared
diagnostic outputs together with a step-by-step audit trace. No language model
is involved: everything the verifier reads has been extracted and span-checked
upstream by the agent.

Four framework types are supported, one per condition:

    n_of_m_rule                            pancreatitis
    multi_evidence_assessment              diverticulitis
    integrated_assessment                  cholecystitis
    tailored_stepwise_risk_stratification  appendicitis

    Verifier(kb).verify(evidence) -> VerifierResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.kb_loader import KB


# Result types

@dataclass
class StepResult:
    """One row in the decision_algorithm trace."""

    step_num: int | str
    step_name: str
    logic: str
    status: str  # 'recorded' | 'passed' | 'failed' | 'indeterminate' | 'skipped'
    detail: str = ""


@dataclass
class VerifierResult:
    """Full Verifier output. Returned by Verifier.verify(evidence)."""

    diagnostic_output: str               # one of kb.diagnostic_outputs
    abstention_triggered: bool           # True if an abstention rule fired
    abstention_reason: str | None        # which rule, if any
    primary_rule_outcome: dict[str, Any] # framework-specific
    step_trace: list[StepResult] = field(default_factory=list)
    criteria_evaluated: dict[str, str] = field(default_factory=dict)
    kb_id: str = ""
    disease: str = ""
    framework_type: str = ""


class VerifierError(RuntimeError):
    """Raised for unrecoverable Verifier errors."""


# Helpers (module-level, framework-agnostic)

def _present_bool(raw: Any) -> bool | None:
    """Extract a present-status boolean from a finding shape."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        p = raw.get("present")
        if isinstance(p, bool):
            return p
        v = raw.get("value")
        if isinstance(v, bool):
            return v
    return None


def _numeric_value(raw: Any) -> float | None:
    """Extract a numeric value from a finding shape."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, dict):
        v = raw.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _confidence_of(raw: Any) -> str:
    """Return the 'confidence' label of a finding shape."""
    valid = ("explicit", "inferred", "uncertain")
    if raw is None:
        return "uncertain"
    if isinstance(raw, dict):
        c = raw.get("confidence", "explicit")
        return c if c in valid else "uncertain"
    return "explicit"


def _status_word(v: bool | None) -> str:
    """Map a tri-state present-bool to satisfied/absent/indeterminate."""
    if v is True:
        return "satisfied"
    if v is False:
        return "absent"
    return "indeterminate"


# Negation tokens for free-form finding KEY names. The 8B sometimes keys a
# NEGATIVE finding ("no_gallbladder_inflammation", "gallbladder_normal") and
# sets value=True -- i.e. value=True asserts the *negative* ("yes, there is no
# inflammation"). A substring inflammation match on such a key would credit it
# as positive imaging: on one case `no_gallbladder_inflammation=True` drove
# the imaging stream true. Token-based (split on
# non-alphanumerics) so "no" matches "no_gallbladder_inflammation" but NOT
# "nodule"/"necrosis", and "normal" matches "gallbladder_normal" but NOT
# "abnormal". "free" is deliberately EXCLUDED ("free fluid" is a positive sign).
_KEY_NEGATION_TOKENS: frozenset[str] = frozenset({
    "no", "non", "not", "without", "negative", "neg", "absent", "absence",
    "unremarkable", "normal", "resolved", "none", "ruled", "wnl",
})


def _alnum_tokens(text: str) -> list[str]:
    """Split a key on any non-alphanumeric run (underscore/space/hyphen/slash)."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _key_encodes_negation(key_lower: str) -> bool:
    """True iff a free-form finding key name encodes a negation/normal token."""
    return any(t in _KEY_NEGATION_TOKENS for t in _alnum_tokens(key_lower))


def _normalize_ct_finding(raw: Any) -> dict[str, Any]:
    """Normalize a single CT finding to {'value': bool|None, 'confidence': str}."""
    valid = ("explicit", "inferred", "uncertain")
    if raw is None:
        return {"value": None, "confidence": "uncertain"}
    if isinstance(raw, bool):
        return {"value": raw, "confidence": "explicit"}
    if isinstance(raw, dict):
        v = raw.get("value")
        if not isinstance(v, bool):
            v = None
        c = raw.get("confidence", "explicit")
        if c not in valid:
            c = "uncertain"
        return {"value": v, "confidence": c}
    return {"value": None, "confidence": "uncertain"}


def _or_indeterminate(a: bool | None, b: bool | None) -> bool | None:
    """OR with explicit indeterminate handling.

    Truth table:
    True if either input is True
    False if both inputs are False
    None otherwise (any other combination involving None)

    Used to combine the independent numeric verdicts on AST and ALT into the
    derived `marked_transaminitis` signal. The model never claims
    `marked_transaminitis` directly; the verifier derives it from a numeric
    cross-check on each lab, preserving uncertainty when either is missing.
    """
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return None


def _derive_wses_stage_hint(complicated_features_present: list[str]) -> str:
    """Coarse WSES CT stage hint per diverticulitis design."""
    s = set(complicated_features_present)
    has_fluid = "intraabdominal_free_fluid" in s
    has_gas = "extraluminal_gas" in s
    has_abscess = "pericolic_abscess" in s
    has_fistula = "fistula" in s
    has_phlegmon = "phlegmon" in s
    if has_fluid and has_gas: return "4"
    if has_fluid: return "3"
    if has_abscess: return "1B_or_2A"
    if has_gas: return "1A_or_2B"
    if has_fistula: return "fistula_no_stage"
    if has_phlegmon: return "unspecified_complicated"
    return "unspecified_complicated"


# numeric cross-check constants

# Atlanta 2-of-3 criterion AP_C2_ENZYMES requires lipase or amylase >= 3x ULN.
# KB source: knowledge_bases/pancreatitis.json
#   laboratory_evidence.thresholds[0]:
#     id    = "lipase_or_amylase_ge_3x_uln"
#     value = "3x upper limit of normal"
# Note: this constant could be replaced by a structured KB field
#   laboratory_evidence.thresholds[*].multiplier (and remove this constant).
AP_C2_ENZYMES_THRESHOLD_MULTIPLIER: float = 3.0

# Markedly elevated lipase pattern in the cholecystitis differential context
# (lipase >= 3x ULN -> suggests acute pancreatitis rather than ACC, drives
# integrated_assessment branch A "alternative_diagnosis_more_likely").
# KB source: cholecystitis strong-differential signal "lipase_3x_uln_pattern";
# clinically grounded in WSES ACC 2020 + Atlanta 2012 (same 3x ULN bar as
# AP_C2_ENZYMES; it is the same lab and same threshold viewed from the
# cholecystitis side).
# Note: could be replaced by a structured laboratory_evidence.thresholds[*].multiplier.
LIPASE_3X_ULN_PATTERN_MULTIPLIER: float = 3.0

# Markedly elevated transaminases pattern (AST and/or ALT >= 3x ULN ->
# suggests choledocholithiasis / cholangitis / hepatic obstruction rather
# than uncomplicated ACC). Per WSES ACC 2020 differential framing
# (`knowledge_bases/cholecystitis.json`
# strong_differential_signals -> marked_transaminitis). The 3x ULN bar
# mirrors the Atlanta lipase threshold used elsewhere in the verifier and is
# the same multiplier applied to AP_C2_ENZYMES and lipase_3x_uln_pattern.
# Derived inside the verifier: the constant is applied to `ast` and `alt`
# separately and OR'd (see `_or_indeterminate` and
# `_evaluate_integrated_assessment`). The model never claims
# `marked_transaminitis` directly.
# Note: could be replaced by a structured laboratory_evidence.thresholds[*].multiplier.
MARKED_TRANSAMINITIS_MULTIPLIER: float = 3.0

# Elevated inflammatory-marker patterns (cholecystitis lab-stream). The
# cholecystitis KB frames these as "elevated_wbc" / "elevated_crp" without
# a marked/mild distinction -- the literal predicate is "value above ULN",
# i.e. multiplier = 1.0. The verifier cross-checks these two fields so the
# lab stream (stream_L) verdict
# stops trusting the LLM's `present` claim when the numeric value disagrees.
# KB source: knowledge_bases/cholecystitis.json
#   laboratory_evidence -> "elevated_wbc" / "elevated_crp" (qualitative).
# Per-field constants kept parallel to LIPASE/MARKED_TRANSAMINITIS for
# future tuning (e.g., a "marked_leukocytosis" follow-up could change WBC
# to 2.0); they are intentionally identical today.
# Note: could be replaced by a structured laboratory_evidence.thresholds[*].multiplier.
ELEVATED_WBC_MULTIPLIER: float = 1.0
ELEVATED_CRP_MULTIPLIER: float = 1.0


# Verifier

class Verifier:
    """Deterministic Verifier wrapping one KB."""

    # ---- Pancreatitis (n_of_m_rule) ----
    _N_OF_M_LABEL_MAP: dict[str, dict[str, str]] = {
        "pancreatitis": {
            "confirmed":     "confirmed_acute_pancreatitis",
            "possible":      "possible_acute_pancreatitis_insufficient_evidence",
            "not_supported": "not_supported_by_available_evidence",
            "uncertain":     "uncertain_due_to_missing_data",
        }
    }

    # ---- Diverticulitis (multi_evidence_assessment) ----
    _DIVERTICULITIS_LABEL_MAP: dict[str, str] = {
        "ct_complicated":         "ct_confirmed_diverticulitis_complicated",
        "ct_uncomplicated":       "ct_confirmed_diverticulitis_uncomplicated",
        "ct_negative_explicit":   "unsupported_or_alternative_diagnosis_more_likely",
        "ct_negative_ambiguous":  "uncertain_insufficient_evidence",
        "no_ct_lameris":          "high_clinical_probability_imaging_pending",
        "no_ct_classic_clinical": "suspected_diverticulitis_needs_imaging",
        "no_ct_no_clinical":      "unsupported_or_alternative_diagnosis_more_likely",
        "fallback":               "uncertain_insufficient_evidence",
    }
    _DIVERTICULITIS_COMPLICATED_FEATURES: tuple[str, ...] = (
        "pericolic_abscess", "extraluminal_gas", "intraabdominal_free_fluid",
        "fistula", "phlegmon",
    )
    # Non-specific complicated features. Free intraabdominal
    # fluid and extraluminal gas are not diverticulitis-specific (free fluid is
    # often physiologic/pelvic; gas can arise from any source, e.g. a gastric
    # fistula near the ascending colon). WSES "complicated diverticulitis" is
    # diverticulitis (colonic diverticular inflammation) PLUS a complication, so
    # these count toward 'complicated' ONLY when a primary diverticular finding
    # is also present (a stage-0 colonic finding, or a colon-specific
    # complication: pericolic_abscess / fistula / phlegmon). This prevents
    # free-fluid/gas-ALONE from confirming complicated diverticulitis on records
    # that explicitly show no diverticulitis, including cholecystitis,
    # appendicitis and post-colectomy records. The anatomical anchor used to mask
    # this by retracting non-colonic CT findings; the conservative anchor does not.
    _DIVERTICULITIS_NONSPECIFIC_COMPLICATED: tuple[str, ...] = (
        "extraluminal_gas", "intraabdominal_free_fluid",
    )
    _DIVERTICULITIS_UNCOMPLICATED_STAGE0_FEATURES: tuple[str, ...] = (
        "diverticula_with_wall_thickening", "pericolic_fat_stranding",
    )

    # ---- Cholecystitis (integrated_assessment) ----
    _CHOLECYSTITIS_LABEL_MAP: dict[str, str] = {
        "confirmed":   "confirmed_or_highly_supported_ACC",
        "probable":    "probable_ACC",
        "possible":    "possible_ACC_needs_more_workup",
        "alternative": "alternative_diagnosis_more_likely",
        "insufficient":"insufficient_evidence",
    }
    # `marked_transaminitis` is not on the model-claim whitelist. The signal is
    # synthesized inside `_evaluate_integrated_assessment` from `ast` and `alt`
    # (see MARKED_TRANSAMINITIS_MULTIPLIER) and joined via `_or_indeterminate`.
    # `strong_differential_fires` still treats `marked_transaminitis` as a
    # whitelisted strong-differential cause -- the membership check below
    # OR-includes the derived signal explicitly.
    _CHOLECYSTITIS_STRONG_DIFFERENTIAL_FEATURES: tuple[str, ...] = (
        "lipase_3x_uln_pattern", "charcot_triad_pattern",
        "epigastric_rather_than_ruq",
        "normal_us_with_persistent_pain",
    )
    # `marked_transaminitis` removed (the numeric-threshold work); synthesized in
    # `_evaluate_integrated_assessment` from the numeric check on `ast` and `alt`. Removing it
    # from this tuple means the strong-differential loop no longer reads
    # `evidence["marked_transaminitis"]`; the derived signal is appended to
    # `strong_differential_signals` independently after the loop completes.
    _CHOLECYSTITIS_ALL_CONTRADICTORY_FEATURES: tuple[str, ...] = (
        "no_ruq_pain_or_tenderness", "normal_inflammatory_markers_with_brief_pain",
        "epigastric_rather_than_ruq", "gallstones_without_inflammatory_imaging_features",
        "normal_us_with_persistent_pain", "lipase_3x_uln_pattern",
        "charcot_triad_pattern",
        "elevated_bilirubin_alone_for_cbds",
    )
    _CHOLECYSTITIS_COMPLICATION_FEATURES: tuple[str, ...] = (
        "gangrenous_cholecystitis", "gallbladder_perforation",
        "emphysematous_cholecystitis", "intraabdominal_abscess", "sepsis",
    )
    _CHOLECYSTITIS_US_INFLAMMATION_FEATURES: tuple[str, ...] = (
        "ultrasound_wall_thickening_gt_3mm", "pericholecystic_fluid",
        "sonographic_murphy_sign", "gallbladder_distention",
    )

    # CT signs of gallbladder inflammation per WSES_ACC_2020 + StatPearls,
    # taken from the KB (imaging_evidence.normalization_notes.ct_positive_features:
    # "gallbladder wall thickening", "pericholecystic fluid", "distention",
    # "surrounding inflammation"). GALLSTONES are EXCLUDED here per the KB
    # caution: "Gallstones alone may represent gallstone disease without active
    # ACC." Matched as substrings against the free-form imaging.ct_findings keys
    # the 8B emits (naming varies: wall_thickening/_gt_3mm/wall_edema;
    # pericholecystic_fluid|collection|stranding; distention|distended). The
    # complicated variants (gangrene/perforation/emphysematous/necrosis) are
    # also CT signs of active inflammation. NOT derived from MIMIC -- grounded
    # in the KB / source guidelines, so it generalizes to any radiology phrasing.
    _CHOLECYSTITIS_CT_INFLAMMATION_KEYWORDS: tuple[str, ...] = (
        "wall_thick", "wall thick", "wall_edema", "wall edema",
        "pericholecystic", "fat_strand", "fat strand",
        "surrounding_inflamm", "surrounding inflamm",
        "distention", "distended",
        "gangren", "perforation", "perfor", "emphysematous", "necro",
        # extra sign tokens the 8B uses for the same concepts
        "fluid", "collection", "stranding", "inflamm", "edema",
    )

    # NON-SPECIFIC distension tokens. Gallbladder distention is
    # listed among the KB ct/us positive_features, but it is a distension /
    # outflow sign (present with fasting, hyperalimentation, obstruction absent
    # ACC), NOT a sign of gallbladder INFLAMMATION. The KB imaging caution
    # anchors ACC on "imaging signs of gallbladder inflammation" and already
    # demotes CBD dilation as "not an ACC-confirming inflammation feature by
    # itself"; WSES states no single feature establishes ACC. A distention-only
    # CT finding is therefore RECORDED (criteria_eval) but cannot ALONE make the
    # imaging stream positive -- it must co-occur with a specific inflammation
    # sign (wall thickening/edema, pericholecystic fluid/stranding, surrounding
    # inflammation, gangrene/perforation/emphysematous/necrosis). Closes the
    # false positive where a true diverticulitis was confirmed on branch B from
    # ct_gallbladder_distention alone, on a CT reading "no evidence ... to suggest
    # cholecystitis". Every true cholecystitis confirm in the review sample carries
    # a specific sign independent of distention, so the guard costs no recall.
    _CHOLECYSTITIS_CT_NONSPECIFIC_TOKENS: tuple[str, ...] = (
        "distention", "distended",
    )

    # A CT finding counts as GALLBLADDER inflammation only if its key BOTH
    # references the gallbladder (an anchor below) AND carries an inflammation
    # sign (a keyword above). Without the anchor, a cholecystitis-agent run
    # over a CT that happens to show OTHER-organ condition mis-credits it as GB
    # inflammation and yields a false probable_ACC. In the confounder gate an
    # appendicitis case emitted ct_findings.appendix_wall_thickening=True and was
    # wrongly labelled cholecystitis. The "pericholecystic*" tokens are
    # intrinsically GB-specific and satisfy both the anchor and the sign.
    _CHOLECYSTITIS_CT_GB_ANCHORS: tuple[str, ...] = (
        "gallbladder", "gall_bladder", "gallblader", "cholecyst",
        "pericholecystic", "gb_", "_gb",
    )

    # ---- Appendicitis (tailored_stepwise_risk_stratification) ----
    _APPENDICITIS_LABEL_MAP: dict[str, str] = {
        "confirmed":   "confirmed_or_highly_likely_acute_appendicitis",
        "complicated": "possible_complicated_appendicitis",
        "intermediate":"intermediate_risk_suspected_appendicitis_needs_imaging_or_observation",
        "low_risk":    "low_risk_appendicitis_unlikely",
        "alternative": "unsupported_or_alternative_diagnosis_more_likely",
    }
    # Whitelist: strong differentials that trigger branch A.
    _APPENDICITIS_STRONG_DIFFERENTIAL_FEATURES: tuple[str, ...] = (
        "dominant_watery_diarrhea_or_viral_syndrome",
        "flank_pain_hematuria_or_colicky_urinary_pain",
        "pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass",
        "known_crohn_or_terminal_ileitis_features",
    )
    # Full 8-feature audit list (all collected, only the 4 above trigger branch A).
    _APPENDICITIS_ALL_CONTRADICTORY_FEATURES: tuple[str, ...] = (
        "absence_of_rlq_pain_or_tenderness",
        "dominant_watery_diarrhea_or_viral_syndrome",
        "flank_pain_hematuria_or_colicky_urinary_pain",
        "pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass",
        "known_crohn_or_terminal_ileitis_features",
        "normal_wbc_and_normal_crp",
        "negative_imaging_with_resolving_symptoms",
        "negative_imaging_but_persistent_or_progressive_pain",
    )
    # Complication features per KB pattern (any_of fires complication signal X).
    _APPENDICITIS_COMPLICATION_FEATURES: tuple[str, ...] = (
        "appendiceal_perforation", "periappendiceal_abscess", "phlegmon",
        "gangrenous_appendicitis", "diffuse_peritonitis", "sepsis",
        "prolonged_symptoms_beyond_48h",
    )
    # GENUINE complications -- imaging/operative findings -- as
    # distinct from mere ASSOCIATIONS (prolonged symptoms, WBC >= 17k). Per WSES
    # Jerusalem 2020, complicated appendicitis (perforation/abscess/phlegmon) is an
    # imaging/operative diagnosis; a lab/duration association must NOT by itself
    # drive a CONFIRMING "complicated" label via the imaging-free branch E. Closes
    # a case where a WBC of 22.9 set signal_X and branch E confirmed complicated
    # appendicitis on a perforated sigmoid diverticulitis whose appendix was
    # normal. The
    # branches that require positive imaging (B/C/D) are unaffected and still use
    # the broader signal_X.
    _APPENDICITIS_GENUINE_COMPLICATION_FEATURES: tuple[str, ...] = (
        "appendiceal_perforation", "periappendiceal_abscess", "phlegmon",
        "gangrenous_appendicitis", "diffuse_peritonitis", "sepsis",
    )
    # US findings sets.
    _APPENDICITIS_US_POSITIVE_FEATURES: tuple[str, ...] = (
        "us_appendix_diameter_gt_6mm", "us_appendicolith",
        "us_echogenic_periappendiceal_fat",
    )
    _APPENDICITIS_US_NEGATIVE_FEATURE: str = "us_diameter_lt_5mm_and_compressibility"
    # CT findings sets.
    _APPENDICITIS_CT_UNCOMPLICATED_FEATURES: tuple[str, ...] = (
        "appendix_outer_diameter_gt_6_to_9_mm",
        "appendiceal_wall_thickening_gt_2_to_3_mm",
        "periappendiceal_fat_stranding",
        "appendiceal_wall_enhancement",
        "appendicolith",
    )
    _APPENDICITIS_CT_COMPLICATED_FEATURES: tuple[str, ...] = (
        "developing_abscess_fluid_collection",
    )
    # Free-form CT-findings classification (mirrors the cholecystitis
    # free-form ct_findings handling). The canonical feature names above are
    # always honored; ADDITIONALLY any non-canonical key carrying an APPENDIX
    # anchor + a feature keyword is classified, so a positive CT appendix emitted
    # under a naming variant is not dropped (the review sample signal_I under-read).
    # The per-iteration anatomical anchor has already retracted any CT
    # finding whose SPAN localizes to a non-appendiceal organ, so a wrong-organ
    # key cannot reach here with value=True. Vocab grounded in WSES appendicitis
    # imaging criteria (appendix diameter >6mm, wall thickening, periappendiceal
    # fat stranding, wall enhancement, appendicolith; complicated: abscess /
    # fluid collection / phlegmon / perforation).
    _APPENDICITIS_CT_APPENDIX_ANCHORS: tuple[str, ...] = (
        "appendix", "appendice", "appendico", "periappendic",
    )
    _APPENDICITIS_CT_UNCOMPLICATED_KEYWORDS: tuple[str, ...] = (
        "diameter", "dilat", "thicken", "fat strand", "fat_strand", "stranding",
        "enhanc", "appendicolith",
    )
    _APPENDICITIS_CT_COMPLICATED_KEYWORDS: tuple[str, ...] = (
        "abscess", "fluid collection", "fluid_collection", "phlegmon",
        "perforation", "extraluminal",
    )
    # MRI findings (no complicated vs uncomplicated distinction in KB).
    _APPENDICITIS_MRI_FEATURES: tuple[str, ...] = (
        "mri_luminal_distension_dilation",
        "mri_wall_thickening",
        "mri_free_periappendiceal_fluid",
    )

    # -- Cross-KB contradictory demotion --
    # A disease's confirm is demoted to a non-confirming output when a RIVAL
    # disease's SOURCE-VERIFIED DEFINING criterion is independently documented in
    # the record -- even if the rival agent did not itself confirm. This is the
    # case for it: one agent hitting its extraction ceiling, with pancreatitis
    # missing its own lipase at 54 times the upper limit, must not let another
    # agent emit a confident wrong label. The gate is record-driven via
    # evidence["_record_rival_criteria"], populated deterministically by the
    # agent from the record it already built -- NOT from an LLM-populated field
    # (the cholecystitis `lipase_3x_uln_pattern` finding was left empty by the 8B
    # there, so the existing branch-A path never fired; it is also
    # suppressed whenever stream_C == "true", which RUQ/epigastric pain -- shared
    # by ACC and pancreatitis -- always satisfies).
    #
    # Currently one rival criterion: acute pancreatitis Revised Atlanta AP_C2 --
    # serum amylase and/or lipase >= 3x ULN. Source-verified:
    #   WSES_pancreatitis_2019: "the diagnosis of AP requires at least the presence
    #     of two of the three following criteria: ... (ii) biochemical evidence of
    #     pancreatitis (serum amylase and/or lipase greater than three times the
    #     upper limit of normal) ..."; StatPearls Acute Pancreatitis: "3 or more
    #     times the upper limit of normal."
    #   Cross-grounded on the host side: the cholecystitis KB independently lists
    #     `lipase_3x_uln_pattern` (contradictory_and_uncertainty_layer) and
    #     StatPearls Acute Cholecystitis directs "assessment of amylase and lipase
    #     levels to rule out pancreatitis."
    # Pancreatitis is exempt (this is its OWN defining criterion). The threshold
    # mirrors AP_C2_ENZYMES_THRESHOLD_MULTIPLIER (3.0). NOT MIMIC-derived.
    _CONFIRMING_OUTPUTS: dict[str, frozenset[str]] = {
        "appendicitis": frozenset({
            "confirmed_or_highly_likely_acute_appendicitis",
            "possible_complicated_appendicitis",
        }),
        "cholecystitis": frozenset({
            "confirmed_or_highly_supported_ACC", "probable_ACC",
        }),
        "diverticulitis": frozenset({
            "ct_confirmed_diverticulitis_uncomplicated",
            "ct_confirmed_diverticulitis_complicated",
        }),
        "pancreatitis": frozenset({"confirmed_acute_pancreatitis"}),
    }
    _CONTRADICTORY_DEMOTION_TARGET: dict[str, str] = {
        "cholecystitis": "alternative_diagnosis_more_likely",
        "appendicitis":  "unsupported_or_alternative_diagnosis_more_likely",
        "diverticulitis": "unsupported_or_alternative_diagnosis_more_likely",
    }
    _RIVAL_PANCREATIC_ENZYME_X_ULN: float = 3.0

    def __init__(self, kb: KB) -> None:
        self.kb = kb

    # ---- public API ----

    def verify(self, evidence: dict[str, dict[str, Any]]) -> VerifierResult:
        """Apply the KB's decision_algorithm + primary_rule to evidence."""
        step_trace = self._walk_decision_algorithm()

        ft = self.kb.framework_type
        if ft == "n_of_m_rule":
            label, outcome, criteria_eval = self._evaluate_n_of_m(evidence)
        elif ft == "multi_evidence_assessment":
            label, outcome, criteria_eval = self._evaluate_multi_evidence_assessment(evidence)
        elif ft == "integrated_assessment":
            label, outcome, criteria_eval = self._evaluate_integrated_assessment(evidence)
        elif ft == "tailored_stepwise_risk_stratification":
            label, outcome, criteria_eval = self._evaluate_tailored_stepwise_risk_stratification(evidence)
        else:
            raise VerifierError(
                f"unknown framework_type {ft!r} on KB {self.kb.disease!r}"
            )

        # Framework-aware abstention assessment.
        if ft == "multi_evidence_assessment":
            branch = str(outcome.get("branch", ""))
            if branch in ("1a", "1b"):
                abstention_triggered, abstention_reason = False, None
            else:
                abstention_triggered = True
                abstention_reason = (
                    f"branch {branch} ({outcome.get('outcome_key', '')}): "
                    f"no CT-confirmed diagnosis"
                )
        elif ft == "integrated_assessment":
            branch = str(outcome.get("branch", ""))
            if branch in ("B", "C", "D"):
                abstention_triggered, abstention_reason = False, None
            else:
                abstention_triggered = True
                abstention_reason = (
                    f"branch {branch} ({outcome.get('outcome_key', '')}): "
                    f"integrated_assessment did not reach confirmed/probable"
                )
        elif ft == "tailored_stepwise_risk_stratification":
            branch = str(outcome.get("branch", ""))
            # Non-abstention: B/C/D/E (-> complicated), F/F2/G (-> confirmed).
            # F2 is the imaging-anchored confirm (specific non-diameter sign).
            if branch in ("B", "C", "D", "E", "F", "F2", "G"):
                abstention_triggered, abstention_reason = False, None
            else:
                abstention_triggered = True
                abstention_reason = (
                    f"branch {branch} ({outcome.get('outcome_key', '')}): "
                    f"stepwise risk stratification did not reach confirmed/complicated"
                )
        else:
            abstention_triggered, abstention_reason = self._check_abstentions_simple(criteria_eval)

        # Cross-KB contradictory demotion (record-driven rival defining
        # criterion). Applied LAST so it overrides the framework label +
        # abstention uniformly across all four frameworks. Only acts on an actual
        # confirm; records the demotion in primary_rule_outcome for the audit.
        demotion = self._apply_contradictory_demotion(label, evidence)
        if demotion is not None:
            new_label, demotion_reason = demotion
            outcome["contradictory_demotion"] = {
                "from": label, "to": new_label, "reason": demotion_reason,
            }
            label = new_label
            abstention_triggered = True
            abstention_reason = demotion_reason

        return VerifierResult(
            diagnostic_output=label,
            abstention_triggered=abstention_triggered,
            abstention_reason=abstention_reason,
            primary_rule_outcome=outcome,
            step_trace=step_trace,
            criteria_evaluated=criteria_eval,
            kb_id=self.kb.kb_id,
            disease=self.kb.disease,
            framework_type=ft,
        )

    def _apply_contradictory_demotion(
        self, label: str, evidence: dict[str, Any],
    ) -> tuple[str, str] | None:
        """Demote a confirm when a rival's defining criterion is in the record.

        Returns ``(demoted_label, reason)`` to override the framework verdict, or
        ``None`` when no demotion applies. Record-driven: reads
        ``evidence["_record_rival_criteria"]`` (deterministically populated by the
        agent from the record text -- see DiseaseAgentBase), never an
        LLM-populated field. Only demotes an actual confirming output; abstaining
        outputs are left unchanged. Pancreatitis is exempt for the pancreatic-
        enzyme criterion (it is pancreatitis' own AP_C2 criterion).
        """
        disease = self.kb.disease
        if disease == "pancreatitis":
            return None
        if label not in self._CONFIRMING_OUTPUTS.get(disease, frozenset()):
            return None

        rc = evidence.get("_record_rival_criteria") or {}
        ratio = rc.get("pancreatic_enzyme_max_x_uln")
        try:
            fired = ratio is not None and float(ratio) >= self._RIVAL_PANCREATIC_ENZYME_X_ULN
        except (TypeError, ValueError):
            return None
        if not fired:
            return None

        target = self._CONTRADICTORY_DEMOTION_TARGET.get(disease)
        if target is None or target not in self.kb.diagnostic_outputs:
            return None

        reason = (
            f"contradictory_demotion: serum amylase/lipase "
            f"~{float(ratio):.1f}x ULN (>= {self._RIVAL_PANCREATIC_ENZYME_X_ULN:g}x) "
            f"documents the Revised Atlanta AP_C2 defining criterion of acute "
            f"pancreatitis (WSES_pancreatitis_2019; StatPearls); a {disease} "
            f"confirm is demoted to {target!r} because a rival disease's defining "
            f"criterion is present in the record."
        )
        return (target, reason)

    # ---- internals ----

    def _walk_decision_algorithm(self) -> list[StepResult]:
        """Records each decision_algorithm step into the trace (no evaluation)."""
        results: list[StepResult] = []
        for step in self.kb.decision_algorithm:
            results.append(
                StepResult(
                    step_num=step.get("step", "?"),
                    step_name=str(step.get("name", "<unnamed>")),
                    logic=str(step.get("logic", "")),
                    status="recorded",
                    detail="recorded by generic walker; evaluation in framework-specific primary_rule logic",
                )
            )
        return results

    def _numeric_threshold_check(
        self,
        finding: dict[str, Any],
        *,
        multiplier: float,
        criterion: str,
    ) -> tuple[bool | None, dict[str, Any] | None]:
        """Generalized numeric threshold cross-check.

        When both `value` and `ref_upper` are present on a numeric finding, the
        Verifier computes whether `value >= multiplier * ref_upper` and treats
        that as the source of truth, regardless of the LLM's `present` flag.
        Originally introduced for pancreatitis AP_C2_ENZYMES (3x ULN);
        generalized for the cholecystitis lab-threshold patterns starting
        Shared by every numeric threshold criterion.

        Args:
            finding: the per-criterion evidence dict; expected to carry
                `value`, `ref_upper`, and optionally `present`.
            multiplier: the threshold multiplier (e.g., 3.0 for the Atlanta
                3x ULN bar; 1.0 for "any value above ULN").
            criterion: the criterion name to record in the disagreement entry.

        Returns:
            (present_r3, disagreement_entry)

            present_r3:
                - True/False when the Verifier has enough data to decide.
                - None when value or ref_upper is missing -> caller falls
                  back to the LLM's `present` flag (legacy path).

            disagreement_entry:
                - None when there is no disagreement (either agreement or the
                  LLM was indeterminate).
                - A dict with the audit fields when the LLM's `present`
                  conflicts with the Verifier's numeric cross-check. This is the
                  payload appended to `primary_rule_outcome.disagreements`
                  (always-on list, possibly empty).
        """
        value = finding.get("value")
        ref_upper = finding.get("ref_upper")
        # Need numeric value and a strictly positive ULN to do the check.
        try:
            value_f = float(value) if value is not None else None
            ref_upper_f = float(ref_upper) if ref_upper is not None else None
        except (TypeError, ValueError):
            return None, None
        if value_f is None or ref_upper_f is None or ref_upper_f <= 0:
            return None, None

        threshold = multiplier * ref_upper_f
        present_r3 = bool(value_f >= threshold)

        llm_present = finding.get("present")
        # LLM indeterminate -> Verifier just provides its answer, no disagreement.
        if llm_present is None:
            return present_r3, None
        # Agreement -> no disagreement.
        if bool(llm_present) == present_r3:
            return present_r3, None
        # Disagreement -> log it.
        return present_r3, {
            "criterion": criterion,
            "llm_present": bool(llm_present),
            "r3_present": present_r3,
            "value": value_f,
            "ref_upper": ref_upper_f,
            "threshold": threshold,
            "multiplier": multiplier,
        }

    def _enzyme_threshold_check(
        self, finding: dict[str, Any]
    ) -> tuple[bool | None, dict[str, Any] | None]:
        """Named entry point for AP_C2_ENZYMES. The implementation is the
        generic `_numeric_threshold_check`."""
        return self._numeric_threshold_check(
            finding,
            multiplier=AP_C2_ENZYMES_THRESHOLD_MULTIPLIER,
            criterion="AP_C2_ENZYMES",
        )

    def _apply_threshold_dispatch(
        self,
        finding: dict[str, Any],
        *,
        multiplier: float,
        criterion: str,
        disagreements: list[dict[str, Any]],
    ) -> bool | None:
        """numeric-cross-check dispatch with symmetric override (symmetric override).

        Returns the effective `present` value for the criterion:
          - if the numeric check has data AND LLM-True + numeric-False -> None (downgrade)
          - if the numeric check has data AND (LLM-False OR LLM-None) + numeric-True -> True (upgrade)
          - if the numeric check has data AND LLM agrees with the numeric check -> numeric (same as the model)
          - if the numeric check has data AND LLM-None + the numeric check has an answer -> numeric (silent acceptance)
          - if the numeric check has no data -> _present_bool(finding) (legacy LLM-present path)

        Any disagreement entry returned by `_numeric_threshold_check` is appended
        to the caller-provided `disagreements` list. Caller is responsible
        for keeping that list alive across all dispatches within a single
        verifier invocation, so the outcome can serialize the full audit.
        """
        r3_present, dis = self._numeric_threshold_check(
            finding, multiplier=multiplier, criterion=criterion
        )
        if dis is not None:
            disagreements.append(dis)
        if r3_present is None:
            return _present_bool(finding)
        llm_present = finding.get("present")
        if llm_present is True and r3_present is False:
            return None
        if (llm_present is False or llm_present is None) and r3_present is True:
            return True
        return r3_present

    def _evaluate_n_of_m(
        self, evidence: dict[str, dict[str, Any]]
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Evaluate n_of_m_rule (currently only pancreatitis).

        For the pancreatitis AP_C2_ENZYMES criterion, this method applies the
        numeric cross-check defined in `_enzyme_threshold_check`:
        - `present=True` with a value below 3x ULN is downgraded to
          indeterminate.
        - `present=False` or None with a value at or above 3x ULN is upgraded
          to satisfied.
        - When value or ref_upper is missing, the legacy path (trust the
        LLM's `present`) is used unchanged.
        Disagreements are appended to `outcome["disagreements"]`, which is
        always present in the outcome dict and is often empty.
        """
        rule = self.kb.primary_rule
        required = int(rule.get("required_count", 2))
        criteria_ids: list[str] = list(rule.get("criteria", []))

        satisfied: list[str] = []
        absent: list[str] = []
        indeterminate: list[str] = []
        criteria_eval: dict[str, str] = {}
        disagreements: list[dict[str, Any]] = []  # Always emitted

        for cid in criteria_ids:
            finding = evidence.get(cid, {}) or {}
            present = finding.get("present")

            # Scope: numeric cross-check only for pancreatitis AP_C2_ENZYMES.
            if self.kb.disease == "pancreatitis" and cid == "AP_C2_ENZYMES":
                r3_present, dis = self._enzyme_threshold_check(finding)
                if dis is not None:
                    disagreements.append(dis)
                if r3_present is not None:
                    # Apply rules:
                    #   LLM True + numeric False  -> downgrade to indeterminate (None)
                    #   LLM False/None + numeric True -> upgrade to True
                    #   Otherwise (agreement) -> use the numeric result directly
                    if present is True and r3_present is False:
                        present = None
                    elif (present is False or present is None) and r3_present is True:
                        present = True
                    else:
                        present = r3_present

            if present is True:
                satisfied.append(cid)
                criteria_eval[cid] = "satisfied"
            elif present is False:
                absent.append(cid)
                criteria_eval[cid] = "absent"
            else:
                indeterminate.append(cid)
                criteria_eval[cid] = "indeterminate"

        n_sat = len(satisfied)
        n_indet = len(indeterminate)

        outcome: dict[str, Any] = {
            "rule_type": "n_of_m",
            "rule_id": rule.get("id", ""),
            "required_count": required,
            "total_count": int(rule.get("total_count", len(criteria_ids))),
            "n_satisfied": n_sat,
            "n_absent": len(absent),
            "n_indeterminate": n_indet,
            "satisfied_criteria": satisfied,
            "absent_criteria": absent,
            "indeterminate_criteria": indeterminate,
            "gate_passed": n_sat >= required,
            "disagreements": disagreements,  # Always present
        }

        label_map = self._N_OF_M_LABEL_MAP.get(self.kb.disease)
        if label_map is None:
            raise NotImplementedError(
                f"n_of_m label mapping not defined for disease {self.kb.disease!r}"
            )

        if n_sat >= required:
            outcome_key = "confirmed"
        elif n_indet >= 1 and n_sat < required:
            outcome_key = "uncertain"
        elif n_sat >= 1:
            outcome_key = "possible"
        else:
            outcome_key = "not_supported"

        label = label_map[outcome_key]
        outcome["outcome_key"] = outcome_key

        if label not in self.kb.diagnostic_outputs:
            raise VerifierError(
                f"emitted label {label!r} not in {self.kb.disease!r} diagnostic_outputs"
            )
        return label, outcome, criteria_eval

    def _evaluate_multi_evidence_assessment(
        self, evidence: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Evaluate multi_evidence_assessment (currently only diverticulitis)."""
        if self.kb.disease != "diverticulitis":
            raise NotImplementedError(
                f"multi_evidence_assessment label mapping not defined for "
                f"disease {self.kb.disease!r}"
            )

        label_map = self._DIVERTICULITIS_LABEL_MAP
        criteria_eval: dict[str, str] = {}
        flags: list[str] = []
        patterns_fired: list[str] = []

        llq_pain      = _present_bool(evidence.get("left_lower_quadrant_pain"))
        llq_tend      = _present_bool(evidence.get("llq_tenderness"))
        llq_tend_only = _present_bool(evidence.get("llq_tenderness_only"))
        absence_vom   = _present_bool(evidence.get("absence_of_vomiting"))
        peritoneal    = _present_bool(evidence.get("peritoneal_signs"))
        fever         = _present_bool(evidence.get("fever"))
        change_bowel  = _present_bool(evidence.get("change_in_bowel_habits"))

        criteria_eval["left_lower_quadrant_pain"] = _status_word(llq_pain)
        criteria_eval["llq_tenderness"]           = _status_word(llq_tend)
        criteria_eval["llq_tenderness_only"]      = _status_word(llq_tend_only)
        criteria_eval["absence_of_vomiting"]      = _status_word(absence_vom)
        criteria_eval["peritoneal_signs"]         = _status_word(peritoneal)
        criteria_eval["fever"]                    = _status_word(fever)
        criteria_eval["change_in_bowel_habits"]   = _status_word(change_bowel)

        crp_val = _numeric_value(evidence.get("crp"))
        wbc_val = _numeric_value(evidence.get("wbc"))
        crp_gt_50  = (crp_val is not None and crp_val > 50.0)
        crp_gt_150 = (crp_val is not None and crp_val > 150.0)
        criteria_eval["crp_gt_50_lameris"] = (
            "satisfied" if crp_gt_50
            else ("absent" if crp_val is not None else "indeterminate")
        )
        criteria_eval["crp_gt_150_complicated"] = (
            "satisfied" if crp_gt_150
            else ("absent" if crp_val is not None else "indeterminate")
        )

        imaging_raw = evidence.get("imaging")
        imaging: dict[str, Any] = imaging_raw if isinstance(imaging_raw, dict) else {}
        ct_done = bool(imaging.get("ct_done", False))
        ct_findings_raw = imaging.get("ct_findings") or {}
        if not isinstance(ct_findings_raw, dict):
            ct_findings_raw = {}

        ct_findings: dict[str, dict[str, Any]] = {}
        for fname in (
            *self._DIVERTICULITIS_COMPLICATED_FEATURES,
            *self._DIVERTICULITIS_UNCOMPLICATED_STAGE0_FEATURES,
        ):
            ct_findings[fname] = _normalize_ct_finding(ct_findings_raw.get(fname))
            criteria_eval[f"ct_{fname}"] = _status_word(ct_findings[fname]["value"])

        complicated_findings_present = [
            f for f in self._DIVERTICULITIS_COMPLICATED_FEATURES
            if ct_findings[f]["value"] is True
        ]
        uncomplicated_findings_present = [
            f for f in self._DIVERTICULITIS_UNCOMPLICATED_STAGE0_FEATURES
            if ct_findings[f]["value"] is True
        ]
        # A non-specific complicated feature (free fluid / extraluminal
        # gas) only counts as 'complicated' when accompanied by a primary
        # diverticular finding -- a stage-0 colonic finding, or a colon-specific
        # complication (pericolic_abscess / fistula / phlegmon). Otherwise it is
        # not attributable to diverticulitis (and the same CT often states no
        # diverticulitis is present).
        _specific_complicated = [
            f for f in complicated_findings_present
            if f not in self._DIVERTICULITIS_NONSPECIFIC_COMPLICATED
        ]
        _nonspecific_complicated = [
            f for f in complicated_findings_present
            if f in self._DIVERTICULITIS_NONSPECIFIC_COMPLICATED
        ]
        _has_primary_diverticular = bool(uncomplicated_findings_present)
        any_complicated = bool(_specific_complicated) or (
            bool(_nonspecific_complicated) and _has_primary_diverticular
        )
        if _nonspecific_complicated and not any_complicated:
            flags.append("nonspecific_ct_finding_without_diverticular_inflammation")
        any_uncomplicated = (
            len(uncomplicated_findings_present)
            == len(self._DIVERTICULITIS_UNCOMPLICATED_STAGE0_FEATURES)
        )

        if crp_gt_150:
            patterns_fired.append("severe_disease_crp_pattern")
            flags.append("mandatory_ct_per_wses")
        if peritoneal is True:
            patterns_fired.append("red_flag_peritonitis_pattern")
            flags.append("possible_complicated_diverticulitis_or_perforation")

        classic_clinical = (llq_pain is True and llq_tend is True)
        if classic_clinical:
            patterns_fired.append("classic_clinical_diverticulitis_pattern")
        lameris = (
            llq_tend_only is True and crp_gt_50 and absence_vom is True
        )
        if lameris:
            patterns_fired.append("lameris_high_probability_pattern")

        no_clinical_pattern_explicit = (llq_pain is False) and (llq_tend is False)

        wses_stage_hint: str | None = None
        outcome_key: str
        branch: str

        if ct_done:
            if any_complicated:
                patterns_fired.append("ct_confirmed_complicated_pattern")
                wses_stage_hint = _derive_wses_stage_hint(complicated_findings_present)
                outcome_key = "ct_complicated"
                branch = "1a"
            elif any_uncomplicated:
                patterns_fired.append("ct_confirmed_uncomplicated_stage0")
                wses_stage_hint = "0"
                outcome_key = "ct_uncomplicated"
                branch = "1b"
            else:
                definite_confidences = [
                    f["confidence"] for f in ct_findings.values()
                    if f["value"] is not None
                ]
                if definite_confidences and all(c == "explicit" for c in definite_confidences):
                    patterns_fired.append("alternative_more_likely_pattern")
                    outcome_key = "ct_negative_explicit"
                    branch = "1c"
                else:
                    outcome_key = "ct_negative_ambiguous"
                    branch = "1d"
        elif lameris:
            outcome_key = "no_ct_lameris"
            branch = "2"
        elif classic_clinical:
            outcome_key = "no_ct_classic_clinical"
            branch = "3"
        elif no_clinical_pattern_explicit:
            patterns_fired.append("alternative_more_likely_pattern")
            outcome_key = "no_ct_no_clinical"
            branch = "4"
        else:
            outcome_key = "fallback"
            branch = "5"

        label = label_map[outcome_key]

        outcome: dict[str, Any] = {
            "rule_type": "multi_evidence",
            "rule_id": str(self.kb.primary_rule.get("id", "")),
            "branch": branch,
            "outcome_key": outcome_key,
            "imaging_done": ct_done,
            "any_complicated_ct_feature": any_complicated,
            "any_uncomplicated_ct_feature": any_uncomplicated,
            "complicated_findings_present": complicated_findings_present,
            "uncomplicated_findings_present": uncomplicated_findings_present,
            "wses_stage_hint": wses_stage_hint,
            "lameris_pattern_satisfied": lameris,
            "classic_clinical_pattern_satisfied": classic_clinical,
            "patterns_fired": patterns_fired,
            "flags": flags,
            "crp_value": crp_val,
            "wbc_value": wbc_val,
        }

        if label not in self.kb.diagnostic_outputs:
            raise VerifierError(
                f"emitted label {label!r} not in {self.kb.disease!r} diagnostic_outputs"
            )
        return label, outcome, criteria_eval

    def _evaluate_integrated_assessment(
        self, evidence: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Evaluate integrated_assessment (currently only cholecystitis).: the `lipase_3x_uln_pattern`
        contradictory feature is now dispatched through `_numeric_threshold_check`
        (LIPASE_3X_ULN_PATTERN_MULTIPLIER=3.0). Disagreements between the
        model's `present` claim and the verifier's numeric verdict are
        appended to `outcome["disagreements"]`, which is always present and
        may be empty, in the same shape as the AP_C2_ENZYMES payload.
        """
        if self.kb.disease != "cholecystitis":
            raise NotImplementedError(
                f"integrated_assessment label mapping not defined for disease "
                f"{self.kb.disease!r}"
            )

        label_map = self._CHOLECYSTITIS_LABEL_MAP
        criteria_eval: dict[str, str] = {}
        flags: list[str] = []
        patterns_fired: list[str] = []
        disagreements: list[dict[str, Any]] = []   # the numeric-threshold work; mirrors AP_C2 shape

        # Clinical (stream C)
        ruq_pain   = _present_bool(evidence.get("ruq_pain_or_tenderness"))
        murphy     = _present_bool(evidence.get("murphy_sign"))
        fever      = _present_bool(evidence.get("fever"))
        vomiting   = _present_bool(evidence.get("vomiting_or_food_intolerance"))
        radiation  = _present_bool(evidence.get("radiation_to_right_shoulder_or_midback"))
        criteria_eval["ruq_pain_or_tenderness"] = _status_word(ruq_pain)
        criteria_eval["murphy_sign"]            = _status_word(murphy)
        criteria_eval["fever"]                  = _status_word(fever)
        criteria_eval["vomiting_or_food_intolerance"] = _status_word(vomiting)
        criteria_eval["radiation_to_right_shoulder_or_midback"] = _status_word(radiation)

        clinical_pattern = (ruq_pain is True and murphy is True)
        if clinical_pattern:
            patterns_fired.append("classic_clinical_acc_pattern")
        # RUQ pain or tenderness is the documentable local anchor and sets stream_C
        # "true" on its own; Murphy is additive (it strengthens via
        # classic_clinical_acc_pattern) but is NOT required. Grounded in WSES ACC
        # 2020 rec 1.2 (clinical features are a combination-menu, not a required
        # RUQ+Murphy pair) and rec 1.1 ("do not rely on a single finding").
        # Safety: confirm (branch B/C) still requires all three streams, so RUQ
        # pain alone reaches at most possible_ACC -- never confirmed. Murphy-alone
        # (no documented RUQ sign) stays "weak" (conservative; near-absent in data).
        if ruq_pain is True:
            stream_C = "true"
        elif murphy is True:
            stream_C = "weak"
        elif (ruq_pain is False) and (murphy is False):
            stream_C = "false"
        else:
            stream_C = "absent"

        # Lab (stream L) -- the numeric-threshold work: numeric-cross-check dispatch
        # for elevated_wbc + elevated_crp via _apply_threshold_dispatch.
        # Replaces the previous trust-the-LLM `_present_bool(...)` calls.
        # `disagreements` was initialized at the top of this method;
        # both the numeric cross-check dispatches append to the same list so the outcome's audit
        # surface captures any LLM/the numeric cross-check contradictions on WBC or CRP claims.
        wbc_present = self._apply_threshold_dispatch(
            evidence.get("elevated_wbc", {}) or {},
            multiplier=ELEVATED_WBC_MULTIPLIER,
            criterion="elevated_wbc",
            disagreements=disagreements,
        )
        crp_present = self._apply_threshold_dispatch(
            evidence.get("elevated_crp", {}) or {},
            multiplier=ELEVATED_CRP_MULTIPLIER,
            criterion="elevated_crp",
            disagreements=disagreements,
        )
        wbc_value = _numeric_value(evidence.get("elevated_wbc"))
        crp_value = _numeric_value(evidence.get("elevated_crp"))
        bilirubin_value = _numeric_value(evidence.get("total_bilirubin"))
        criteria_eval["elevated_wbc"]   = _status_word(wbc_present)
        criteria_eval["elevated_crp"]   = _status_word(crp_present)
        criteria_eval["total_bilirubin"] = (
            "satisfied" if bilirubin_value is not None else "indeterminate"
        )

        if wbc_present is True or crp_present is True:
            stream_L = "true"
        elif wbc_present is False and crp_present is False:
            stream_L = "false"
        else:
            stream_L = "absent"

        # Imaging (stream I)
        imaging_raw = evidence.get("imaging")
        imaging: dict[str, Any] = imaging_raw if isinstance(imaging_raw, dict) else {}
        us_done   = bool(imaging.get("us_done", False))
        ct_done   = bool(imaging.get("ct_done", False))
        hida_done = bool(imaging.get("hida_done", False))
        any_imaging_done = us_done or ct_done or hida_done

        us_findings_raw = imaging.get("us_findings") or {}
        if not isinstance(us_findings_raw, dict):
            us_findings_raw = {}
        hida_findings_raw = imaging.get("hida_findings") or {}
        if not isinstance(hida_findings_raw, dict):
            hida_findings_raw = {}
        ct_supportive_raw = imaging.get("ct_supportive_findings")

        us_findings: dict[str, dict[str, Any]] = {}
        for fname in ("ultrasound_gallstones", *self._CHOLECYSTITIS_US_INFLAMMATION_FEATURES):
            us_findings[fname] = _normalize_ct_finding(us_findings_raw.get(fname))
            criteria_eval[f"us_{fname}"] = _status_word(us_findings[fname]["value"])

        hida_nonfilling = _normalize_ct_finding(
            hida_findings_raw.get("hidascan_nonfilling_gallbladder")
        )
        criteria_eval["hida_hidascan_nonfilling_gallbladder"] = _status_word(
            hida_nonfilling["value"]
        )
        ct_supportive = _normalize_ct_finding(ct_supportive_raw)
        criteria_eval["ct_supportive_findings"] = _status_word(ct_supportive["value"])

        # Nested free-form CT findings: classify each emitted CT
        # finding key against the guideline CT-inflammation vocabulary. ANY
        # inflammation sign True makes CT supportive (WSES rec 1.2 lists "signs
        # of GB inflammation" as a diagnostic category; a single CT inflammation
        # sign is sufficient for the imaging STREAM, while three-stream confirm
        # is still required by WSES rec 1.1 "do not rely on a single finding").
        ct_findings_raw = imaging.get("ct_findings") or {}
        if not isinstance(ct_findings_raw, dict):
            ct_findings_raw = {}
        ct_inflammation_keys_true: list[str] = []
        ct_findings_evaluated: list[dict[str, Any]] = []
        for fname, fraw in ct_findings_raw.items():
            norm = _normalize_ct_finding(fraw)
            key_l = str(fname).lower()
            has_gb_anchor = any(a in key_l for a in self._CHOLECYSTITIS_CT_GB_ANCHORS)
            matched_kw = [
                kw for kw in self._CHOLECYSTITIS_CT_INFLAMMATION_KEYWORDS
                if kw in key_l
            ]
            # a non-specific distension token (distention/distended) is
            # recorded but does NOT by itself constitute an inflammation sign --
            # a SPECIFIC inflammation keyword must also match for the finding to
            # make CT supportive. (A key that pairs distention with a specific
            # sign, e.g. "wall_thickening_and_distention", still counts.)
            specific_kw = [
                kw for kw in matched_kw
                if kw not in self._CHOLECYSTITIS_CT_NONSPECIFIC_TOKENS
            ]
            # #1b: a key that NAMES a negation ("no_gallbladder_inflammation",
            # "gallbladder_normal") asserts the NEGATIVE even at value=True, so
            # it must never be credited as a positive inflammation sign. Recorded
            # for audit and excluded from ct_inflammation_keys_true, which closes the
            # spurious imaging stream driven by no_gallbladder_inflammation=True.
            negated_key = _key_encodes_negation(key_l)
            if has_gb_anchor and matched_kw:
                criteria_eval[f"ct_{fname}"] = _status_word(norm["value"])
                ct_findings_evaluated.append(norm)
                if norm["value"] is True and specific_kw and not negated_key:
                    ct_inflammation_keys_true.append(fname)
        ct_inflammation_True = len(ct_inflammation_keys_true) > 0
        if ct_inflammation_True:
            patterns_fired.append("ct_supported_acc_pattern")

        imaging_supported_acc = (
            us_findings["ultrasound_wall_thickening_gt_3mm"]["value"] is True
            and us_findings["ultrasound_gallstones"]["value"] is True
        )
        if imaging_supported_acc:
            patterns_fired.append("imaging_supported_acc_pattern")
        hida_nonfilling_True = hida_nonfilling["value"] is True
        if hida_nonfilling_True:
            patterns_fired.append("hida_supported_pattern")
        ct_supportive_True = (ct_supportive["value"] is True) or ct_inflammation_True

        any_imaging_supportive = (
            imaging_supported_acc or hida_nonfilling_True or ct_supportive_True
        )

        if not any_imaging_done:
            stream_I = "absent"
        elif any_imaging_supportive:
            stream_I = "true"
        else:
            evaluated_confidences: list[str] = []
            for f in us_findings.values():
                if f["value"] is not None:
                    evaluated_confidences.append(f["confidence"])
            if hida_done and hida_nonfilling["value"] is not None:
                evaluated_confidences.append(hida_nonfilling["confidence"])
            if ct_done and ct_supportive["value"] is not None:
                evaluated_confidences.append(ct_supportive["confidence"])
            if ct_done:
                for norm in ct_findings_evaluated:
                    if norm["value"] is not None:
                        evaluated_confidences.append(norm["confidence"])
            if evaluated_confidences and all(c == "explicit" for c in evaluated_confidences):
                stream_I = "false"
            else:
                stream_I = "unclear"

        gallstones_only = (
            us_findings["ultrasound_gallstones"]["value"] is True
            and not any(
                us_findings[f]["value"] is True
                for f in self._CHOLECYSTITIS_US_INFLAMMATION_FEATURES
            )
            and not ct_inflammation_True
        )
        if gallstones_only:
            patterns_fired.append("gallstones_only_pattern")
            flags.append("biliary_disease_likely_not_active_ACC")

        # CBDS
        cbds_features_raw = evidence.get("cbds_features")
        cbds_features: dict[str, Any] = (
            cbds_features_raw if isinstance(cbds_features_raw, dict) else {}
        )
        cbd_stone = _present_bool(cbds_features.get("cbd_stone_on_us"))
        cbd_dilation = _present_bool(cbds_features.get("cbd_dilation"))
        criteria_eval["cbd_stone_on_us"] = _status_word(cbd_stone)
        criteria_eval["cbd_dilation"]    = _status_word(cbd_dilation)

        if cbd_stone is True:
            cbds_risk_band = "high"
            patterns_fired.append("cbds_high_risk_pattern")
            flags.append("high_cbds_risk")
        elif bilirubin_value is not None and bilirubin_value > 4.0:
            cbds_risk_band = "moderate"
            patterns_fired.append("cbds_moderate_risk_pattern")
            flags.append("moderate_cbds_risk")
        elif (
            bilirubin_value is not None
            and 1.8 < bilirubin_value <= 4.0
            and cbd_dilation is True
        ):
            cbds_risk_band = "moderate"
            patterns_fired.append("cbds_moderate_risk_pattern")
            flags.append("moderate_cbds_risk")
        elif bilirubin_value is None and cbd_stone is None and cbd_dilation is None:
            cbds_risk_band = "unknown"
        else:
            cbds_risk_band = "low"

        # Complications
        complication_flags: list[str] = []
        for cf in self._CHOLECYSTITIS_COMPLICATION_FEATURES:
            v = _present_bool(evidence.get(cf))
            criteria_eval[cf] = _status_word(v)
            if v is True:
                complication_flags.append(cf)
        if complication_flags:
            patterns_fired.append("complicated_acc_pattern")
            flags.append("complicated_ACC_red_flag")

        # Strong-differential whitelist -- the numeric-threshold work:
        # `lipase_3x_uln_pattern` is dispatched through `_apply_threshold_dispatch`
        # to enforce the 3x ULN threshold via numeric cross-check (stage 1); all
        # other features in the contradictory tuple stay on the legacy
        # LLM-`present` path. `marked_transaminitis` is no longer in the
        # contradictory tuple (stage 2); it is synthesized below from the numeric cross-check of
        # the `ast` and `alt` NumericFindings.
        strong_differential_signals: list[str] = []
        for cf in self._CHOLECYSTITIS_ALL_CONTRADICTORY_FEATURES:
            if cf == "lipase_3x_uln_pattern":
                v = self._apply_threshold_dispatch(
                    evidence.get(cf, {}) or {},
                    multiplier=LIPASE_3X_ULN_PATTERN_MULTIPLIER,
                    criterion="lipase_3x_uln_pattern",
                    disagreements=disagreements,
                )
            else:
                v = _present_bool(evidence.get(cf))
            criteria_eval[cf] = _status_word(v)
            if v is True:
                strong_differential_signals.append(cf)

        # Derive `marked_transaminitis`
        # Verifier-side from the numeric check on `ast` and `alt` (OR semantics; either lab
        # >= MARKED_TRANSAMINITIS_MULTIPLIER * ref_upper -> True).
        #
        # The LLM no longer claims `marked_transaminitis` directly; the
        # agent populates `ast.value+ref_upper` and `alt.value+ref_upper`
        # and the Verifier is the source of truth on the threshold. We use
        # `_numeric_threshold_check` directly (not `_apply_threshold_dispatch`)
        # because there is no LLM `present` claim on `ast`/`alt` to override
        # -- the schema only carries value/ref_upper/unit/audit. If a
        # future extraction path does volunteer `present`, the
        # disagreement record is still logged.
        ast_finding = evidence.get("ast", {}) or {}
        alt_finding = evidence.get("alt", {}) or {}
        ast_r3, ast_dis = self._numeric_threshold_check(
            ast_finding,
            multiplier=MARKED_TRANSAMINITIS_MULTIPLIER,
            criterion="marked_transaminitis_ast",
        )
        alt_r3, alt_dis = self._numeric_threshold_check(
            alt_finding,
            multiplier=MARKED_TRANSAMINITIS_MULTIPLIER,
            criterion="marked_transaminitis_alt",
        )
        if ast_dis is not None:
            disagreements.append(ast_dis)
        if alt_dis is not None:
            disagreements.append(alt_dis)
        mt_present = _or_indeterminate(ast_r3, alt_r3)
        criteria_eval["marked_transaminitis"] = _status_word(mt_present)
        if mt_present is True:
            strong_differential_signals.append("marked_transaminitis")

        # `marked_transaminitis` is not in the constant tuple, but the derived
        # signal still triggers branch A, so it is OR-ed into the check.
        strong_differential_fires = any(
            cf in strong_differential_signals
            for cf in (
                *self._CHOLECYSTITIS_STRONG_DIFFERENTIAL_FEATURES,
                "marked_transaminitis",
            )
        )

        outcome_key: str
        branch: str

        if strong_differential_fires and stream_C != "true":
            outcome_key, branch = "alternative", "A"
        elif stream_C == "true" and stream_L == "true" and stream_I == "true":
            outcome_key, branch = "confirmed", "B"
        elif stream_C == "true" and hida_nonfilling_True:
            outcome_key, branch = "confirmed", "C"
        elif stream_C == "true" and stream_I == "true":
            outcome_key, branch = "probable", "D"
        elif stream_C == "true" and stream_L == "true" and stream_I in ("absent", "unclear"):
            outcome_key, branch = "possible", "E"
        elif gallstones_only and stream_C != "true":
            outcome_key, branch = "alternative", "F"
        elif stream_C == "true" and stream_L != "true" and stream_I == "absent":
            outcome_key, branch = "possible", "G"
        elif stream_I == "true" and stream_C != "true":
            outcome_key, branch = "possible", "H"
        elif (
            stream_C in ("weak", "false", "absent")
            and stream_L in ("absent", "false")
            and stream_I in ("absent", "false")
        ):
            outcome_key, branch = "insufficient", "I"
        else:
            outcome_key, branch = "insufficient", "J"

        label = label_map[outcome_key]

        outcome: dict[str, Any] = {
            "rule_type": "integrated_assessment",
            "rule_id": str(self.kb.primary_rule.get("id", "")),
            "branch": branch,
            "outcome_key": outcome_key,
            "stream_C_clinical": stream_C,
            "stream_L_lab": stream_L,
            "stream_I_imaging": stream_I,
            "patterns_fired": patterns_fired,
            "gallstones_only": gallstones_only,
            "strong_differential_signals": strong_differential_signals,
            "cbds_risk_band": cbds_risk_band,
            "complication_flags": complication_flags,
            "flags": flags,
            "imaging_modalities_done": {"us": us_done, "ct": ct_done, "hida": hida_done},
            "wbc_value": wbc_value,
            "crp_value": crp_value,
            "bilirubin_value": bilirubin_value,
            "disagreements": disagreements,   # the numeric-threshold work (mirrors AP_C2 shape)
        }

        if label not in self.kb.diagnostic_outputs:
            raise VerifierError(
                f"emitted label {label!r} not in {self.kb.disease!r} diagnostic_outputs"
            )
        return label, outcome, criteria_eval

    def _compute_alvarado_from_components(
        self, evidence: dict[str, Any], wbc_value: float | None
    ) -> int | None:
        """Compute Alvarado score (0-9) from 7 components.

        Components (StatPearls):
        - RLQ tenderness +2
        - Leukocytosis (WBC > 10000) +2
        - Migratory RLQ pain +1
        - Rebound tenderness (McBurney) +1
        - Fever +1
        - Nausea or vomiting +1
        - Anorexia +1
        Returns None if no components are extractable (avoids 0 false negative).
        """
        components: list[tuple[str, int, bool | None]] = [
            ("rlq_tenderness", 2, _present_bool(evidence.get("rlq_tenderness"))),
            ("pain_migrates_to_rlq", 1, _present_bool(evidence.get("pain_migrates_to_rlq"))),
            ("rebound_tenderness_mcburney", 1, _present_bool(evidence.get("rebound_tenderness_mcburney"))),
            ("fever", 1, _present_bool(evidence.get("fever"))),
            ("nausea_or_vomiting", 1, _present_bool(evidence.get("nausea_or_vomiting"))),
            ("anorexia", 1, _present_bool(evidence.get("anorexia"))),
        ]
        # Leukocytosis is derived from raw WBC value, not a feature flag.
        leukocytosis = (wbc_value is not None and wbc_value > 10000)
        components.append(("leukocytosis_wbc_gt_10000", 2, leukocytosis if wbc_value is not None else None))

        any_known = any(v is not None for _, _, v in components)
        if not any_known:
            return None
        # Sum points for components that are True; None and False contribute 0.
        return sum(pts for _, pts, val in components if val is True)

    def _evaluate_tailored_stepwise_risk_stratification(
        self, evidence: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Evaluate tailored_stepwise_risk_stratification (only appendicitis).

        Five signals (C, L, I, S, X) plus strong_differential and
        progressive_pain feed a fifteen-branch priority cascade.
        """
        if self.kb.disease != "appendicitis":
            raise NotImplementedError(
                f"tailored_stepwise_risk_stratification label mapping not defined for "
                f"disease {self.kb.disease!r}"
            )

        label_map = self._APPENDICITIS_LABEL_MAP
        criteria_eval: dict[str, str] = {}
        flags: list[str] = []
        patterns_fired: list[str] = []

        # ---------- Population context ----------
        pop_ctx_raw = evidence.get("population_context")
        population_context = pop_ctx_raw if isinstance(pop_ctx_raw, str) else "adult"
        age_lt_40_raw = evidence.get("age_lt_40")
        if isinstance(age_lt_40_raw, bool):
            age_lt_40 = age_lt_40_raw
        elif isinstance(age_lt_40_raw, dict):
            v = age_lt_40_raw.get("value")
            age_lt_40 = v if isinstance(v, bool) else True
        else:
            age_lt_40 = True  # Default permissive for branch G

        # ---------- Signal C (clinical) ----------
        pain_periumbilical = _present_bool(evidence.get("pain_periumbilical_or_diffuse_initial"))
        pain_migrates_to_rlq = _present_bool(evidence.get("pain_migrates_to_rlq"))
        rlq_tenderness = _present_bool(evidence.get("rlq_tenderness"))
        rebound = _present_bool(evidence.get("rebound_tenderness_mcburney"))
        anorexia = _present_bool(evidence.get("anorexia"))
        nausea_vomiting = _present_bool(evidence.get("nausea_or_vomiting"))
        fever = _present_bool(evidence.get("fever"))
        guarding = _present_bool(evidence.get("localized_guarding_rlq"))

        for k, v in (
            ("pain_periumbilical_or_diffuse_initial", pain_periumbilical),
            ("pain_migrates_to_rlq", pain_migrates_to_rlq),
            ("rlq_tenderness", rlq_tenderness),
            ("rebound_tenderness_mcburney", rebound),
            ("anorexia", anorexia),
            ("nausea_or_vomiting", nausea_vomiting),
            ("fever", fever),
            ("localized_guarding_rlq", guarding),
        ):
            criteria_eval[k] = _status_word(v)

        classic_clinical = (pain_migrates_to_rlq is True and rlq_tenderness is True)
        if classic_clinical:
            patterns_fired.append("classic_clinical_appendicitis_pattern")
        any_clinical_signal = (
            pain_migrates_to_rlq is True or rlq_tenderness is True or rebound is True
        )
        # Localized RLQ tenderness is
        # the documentable cardinal sign of appendicitis (WSES Jerusalem 2020) and
        # sets signal_C "true" on its own; migratory pain / rebound / Rovsing are
        # ADDITIVE (they strengthen via classic_clinical_appendicitis_pattern) but
        # are NOT required, mirroring the cholecystitis clinical stream, where RUQ
        # pain anchors and Murphy is additive. Confirmation
        # still requires positive imaging (branch F) or a HIGH score + age<40
        # (branch G), so RLQ tenderness ALONE never reaches "confirmed" -- it only
        # promotes CT/score-supported cases out of "intermediate". Previously this
        # required BOTH migration AND RLQ tenderness, blocking CT-positive cases
        # with documented RLQ tenderness but no migration history (the dominant
        # appendicitis recall miss on the review sample review).
        if rlq_tenderness is True:
            signal_C = "true"
        elif any_clinical_signal:
            signal_C = "weak"
        elif pain_migrates_to_rlq is False and rlq_tenderness is False:
            signal_C = "false"
        else:
            signal_C = "absent"

        # ---------- Signal L (lab) ----------
        wbc_value = _numeric_value(evidence.get("wbc"))
        crp_value = _numeric_value(evidence.get("crp"))
        anc_value = _numeric_value(evidence.get("anc"))
        wbc_gt_10000 = (wbc_value is not None and wbc_value > 10000)
        wbc_ge_17000 = (wbc_value is not None and wbc_value >= 17000)
        wbc_ge_16000_peds = (wbc_value is not None and wbc_value >= 16000)
        crp_elevated = (crp_value is not None and crp_value > 10.0)
        criteria_eval["wbc"] = "satisfied" if wbc_value is not None else "indeterminate"
        criteria_eval["crp"] = "satisfied" if crp_value is not None else "indeterminate"
        criteria_eval["wbc_gt_10000_common"] = (
            "satisfied" if wbc_gt_10000
            else ("absent" if wbc_value is not None else "indeterminate")
        )
        criteria_eval["wbc_ge_17000_complicated_association"] = (
            "satisfied" if wbc_ge_17000
            else ("absent" if wbc_value is not None else "indeterminate")
        )
        if wbc_gt_10000 or crp_elevated:
            signal_L = "true"
        elif (
            wbc_value is not None and not wbc_gt_10000
            and crp_value is not None and not crp_elevated
        ):
            signal_L = "false"
        else:
            signal_L = "absent"

        # ---------- Signal I (imaging) ----------
        imaging_raw = evidence.get("imaging")
        imaging: dict[str, Any] = imaging_raw if isinstance(imaging_raw, dict) else {}
        us_done  = bool(imaging.get("us_done", False))
        ct_done  = bool(imaging.get("ct_done", False))
        mri_done = bool(imaging.get("mri_done", False))
        any_imaging_done = us_done or ct_done or mri_done

        us_findings_raw  = imaging.get("us_findings") or {}
        ct_findings_raw  = imaging.get("ct_findings") or {}
        mri_findings_raw = imaging.get("mri_findings") or {}
        if not isinstance(us_findings_raw, dict): us_findings_raw = {}
        if not isinstance(ct_findings_raw, dict): ct_findings_raw = {}
        if not isinstance(mri_findings_raw, dict): mri_findings_raw = {}

        us_features: dict[str, dict[str, Any]] = {}
        for fname in (*self._APPENDICITIS_US_POSITIVE_FEATURES,
                      self._APPENDICITIS_US_NEGATIVE_FEATURE):
            us_features[fname] = _normalize_ct_finding(us_findings_raw.get(fname))
            criteria_eval[f"us_{fname}"] = _status_word(us_features[fname]["value"])

        ct_features: dict[str, dict[str, Any]] = {}
        for fname in (*self._APPENDICITIS_CT_UNCOMPLICATED_FEATURES,
                      *self._APPENDICITIS_CT_COMPLICATED_FEATURES):
            ct_features[fname] = _normalize_ct_finding(ct_findings_raw.get(fname))
            criteria_eval[f"ct_{fname}"] = _status_word(ct_features[fname]["value"])

        # ALSO classify non-canonical free-form CT keys. A positive
        # (value=True) key with an appendix anchor + a feature keyword is bucketed
        # as complicated or uncomplicated, so CT findings emitted under naming
        # variants are not dropped. Canonical keys are handled above; here we only
        # scan the extras.
        _app_canonical_ct = (set(self._APPENDICITIS_CT_UNCOMPLICATED_FEATURES)
                             | set(self._APPENDICITIS_CT_COMPLICATED_FEATURES))
        extra_ct_uncomplicated_true: list[str] = []
        extra_ct_complicated_true: list[str] = []
        for fname, fraw in ct_findings_raw.items():
            if fname in _app_canonical_ct:
                continue
            norm = _normalize_ct_finding(fraw)
            if norm["value"] is not True:
                continue
            key_l = str(fname).lower()
            if not any(a in key_l for a in self._APPENDICITIS_CT_APPENDIX_ANCHORS):
                continue
            if any(kw in key_l for kw in self._APPENDICITIS_CT_COMPLICATED_KEYWORDS):
                extra_ct_complicated_true.append(fname)
                criteria_eval[f"ct_{fname}"] = _status_word(True)
            elif any(kw in key_l for kw in self._APPENDICITIS_CT_UNCOMPLICATED_KEYWORDS):
                extra_ct_uncomplicated_true.append(fname)
                criteria_eval[f"ct_{fname}"] = _status_word(True)

        mri_features: dict[str, dict[str, Any]] = {}
        for fname in self._APPENDICITIS_MRI_FEATURES:
            mri_features[fname] = _normalize_ct_finding(mri_findings_raw.get(fname))
            criteria_eval[f"mri_{fname}"] = _status_word(mri_features[fname]["value"])

        any_us_positive = any(
            us_features[f]["value"] is True
            for f in self._APPENDICITIS_US_POSITIVE_FEATURES
        )
        us_definitively_negative = (
            us_features[self._APPENDICITIS_US_NEGATIVE_FEATURE]["value"] is True
        )
        any_ct_uncomplicated = (
            any(ct_features[f]["value"] is True
                for f in self._APPENDICITIS_CT_UNCOMPLICATED_FEATURES)
            or len(extra_ct_uncomplicated_true) > 0
        )
        any_ct_complicated = (
            any(ct_features[f]["value"] is True
                for f in self._APPENDICITIS_CT_COMPLICATED_FEATURES)
            or len(extra_ct_complicated_true) > 0
        )
        any_mri_positive = any(
            mri_features[f]["value"] is True
            for f in self._APPENDICITIS_MRI_FEATURES
        )
        any_imaging_uncomplicated_positive = (
            any_us_positive or any_ct_uncomplicated or any_mri_positive
        )

        # A NON-DIAMETER inflammatory imaging sign of appendicitis.
        # StatPearls Appendicitis: "an increased appendiceal diameter ALONE is not a
        # conclusive finding and should be interpreted in conjunction with other ...
        # features"; the diagnostic CT criteria also list wall thickening >2-3 mm,
        # periappendiceal fat stranding, mural enhancement, and appendicolith. So a
        # positive appendix imaging carrying at least one SPECIFIC inflammatory sign
        # BEYOND bare diameter/dilation is diagnostic and may confirm even when the
        # clinical band is weak, while diameter or dilation alone stays intermediate,
        # which is the caveat the source guideline makes.
        _ct_diameter_feature = "appendix_outer_diameter_gt_6_to_9_mm"
        _ct_diameter_kw = ("diameter", "dilat")
        _canon_ct_nondiameter = any(
            ct_features[f]["value"] is True
            for f in self._APPENDICITIS_CT_UNCOMPLICATED_FEATURES
            if f != _ct_diameter_feature
        )
        _extra_ct_nondiameter = any(
            any(kw in str(f).lower()
                for kw in self._APPENDICITIS_CT_UNCOMPLICATED_KEYWORDS
                if kw not in _ct_diameter_kw)
            for f in extra_ct_uncomplicated_true
        )
        _us_nondiameter = (
            us_features["us_appendicolith"]["value"] is True
            or us_features["us_echogenic_periappendiceal_fat"]["value"] is True
        )
        _mri_nondiameter = (
            mri_features["mri_wall_thickening"]["value"] is True
            or mri_features["mri_free_periappendiceal_fluid"]["value"] is True
        )
        imaging_nondiameter_inflammatory_sign = (
            _canon_ct_nondiameter or _extra_ct_nondiameter
            or _us_nondiameter or _mri_nondiameter
        )

        if any_us_positive:
            patterns_fired.append("imaging_supported_appendicitis_pattern")
        if any_ct_complicated:
            patterns_fired.append("imaging_complicated_finding")
        if not any_imaging_done:
            signal_I = "absent"
        elif any_ct_complicated:
            signal_I = "positive_complicated"
        elif any_imaging_uncomplicated_positive:
            signal_I = "positive_uncomplicated"
        elif us_definitively_negative:
            signal_I = "negative"
        else:
            # imaging done, no positive features, no definitive US negative;
            # gate by confidence
            definite_confidences: list[str] = []
            for f in (*us_features.values(), *ct_features.values(), *mri_features.values()):
                if f["value"] is not None:
                    definite_confidences.append(f["confidence"])
            if definite_confidences and all(c == "explicit" for c in definite_confidences):
                signal_I = "negative"
            else:
                signal_I = "unclear"

        # A radiology report that explicitly documents a NORMAL /
        # non-dilated appendix rules out appendicitis -> a positive imaging signal
        # cannot stand (it is reactive peri-appendiceal change, not appendicitis).
        # Record-driven, read from the full record, independent of what the
        # 8B surfaced. It closes a cecal diverticulitis whose CT read appendix "appears
        # normal ... not dilated", with only adjacent stranding that the 8B credited
        # as periappendiceal_fat_stranding and confirmed on branch F. A
        # true appendicitis always describes a dilated/inflamed/thickened appendix,
        # so the record flag is never set for it.
        appendix_documented_normal = bool(evidence.get("_record_appendix_normal"))
        if appendix_documented_normal and signal_I in (
            "positive_uncomplicated", "positive_complicated"
        ):
            signal_I = "negative"
            flags.append("appendix_documented_normal_imaging_veto")

        # ---------- Signal S (score) ----------
        # Schema-gated: AppendicitisExtraction intentionally has no
        # AIR_score/AAS_score/Alvarado_score/PAS_score fields, so these four reads are always
        # None and signal_S is ALWAYS driven by the deterministically *computed* Alvarado
        # (_compute_alvarado_from_components). This is by design -- the Verifier computes the
        # score itself from extracted components rather than trusting an LLM-extracted score
        # number (more auditable; MIMIC ED records rarely pre-state a named score). The AIR/AAS/
        # PAS/Alvarado-extracted branches below are retained for completeness but are currently
        # unreachable; add the schema fields if record-stated scores should ever be captured.
        air_score      = _numeric_value(evidence.get("AIR_score"))
        aas_score      = _numeric_value(evidence.get("AAS_score"))
        alvarado_extracted = _numeric_value(evidence.get("Alvarado_score"))
        pas_score      = _numeric_value(evidence.get("PAS_score"))
        alvarado_computed = self._compute_alvarado_from_components(evidence, wbc_value)

        # Priority: AIR > AAS > Alvarado_extracted > Alvarado_computed > PAS
        # PAS preferred only when population_context == 'pediatric'
        if population_context == "pediatric" and pas_score is not None:
            score_used = "PAS"
            score_value: float | None = pas_score
        elif air_score is not None:
            score_used = "AIR"
            score_value = air_score
        elif aas_score is not None:
            score_used = "AAS"
            score_value = aas_score
        elif alvarado_extracted is not None:
            score_used = "Alvarado_extracted"
            score_value = alvarado_extracted
        elif alvarado_computed is not None:
            score_used = "Alvarado_computed"
            score_value = float(alvarado_computed)
        elif pas_score is not None:
            score_used = "PAS"
            score_value = pas_score
        else:
            score_used = "none"
            score_value = None

        # Banding
        if score_value is None:
            signal_S = "undetermined"
        elif score_used == "AIR":
            if score_value >= 9: signal_S = "high"
            elif score_value >= 5: signal_S = "intermediate"
            else: signal_S = "low"
        elif score_used == "AAS":
            if score_value >= 16: signal_S = "high"
            elif score_value >= 11: signal_S = "intermediate"
            else: signal_S = "low"
        elif score_used in ("Alvarado_extracted", "Alvarado_computed"):
            if score_value >= 9: signal_S = "high"
            elif score_value >= 4: signal_S = "intermediate"
            else: signal_S = "low"
        elif score_used == "PAS":
            if score_value >= 7: signal_S = "high"
            elif score_value >= 4: signal_S = "intermediate"
            else: signal_S = "low"
        else:
            signal_S = "undetermined"

        # ---------- Signal X (complications) ----------
        complication_flags: list[str] = []
        for cf in self._APPENDICITIS_COMPLICATION_FEATURES:
            v = _present_bool(evidence.get(cf))
            criteria_eval[cf] = _status_word(v)
            if v is True:
                complication_flags.append(cf)
        if wbc_ge_17000:
            complication_flags.append("wbc_ge_17000")
        if complication_flags:
            patterns_fired.append("possible_complicated_appendicitis_pattern")
            flags.append("complicated_appendicitis_red_flag")
            signal_X = "present"
        else:
            signal_X = "absent"
        # A GENUINE (imaging/operative) complication, excluding association-only
        # flags (wbc_ge_17000, prolonged_symptoms_beyond_48h). Gates the imaging-free
        # complicated confirm (branch E) so a lab/duration association cannot confirm
        # "complicated appendicitis" without imaging.
        signal_X_genuine = (
            "present"
            if any(_present_bool(evidence.get(cf)) is True
                   for cf in self._APPENDICITIS_GENUINE_COMPLICATION_FEATURES)
            else "absent"
        )

        # Appendicolith as separate flag (not strictly a complication but a risk marker)
        appendicolith_present = (
            us_features["us_appendicolith"]["value"] is True
            or ct_features["appendicolith"]["value"] is True
        )
        if appendicolith_present:
            flags.append("appendicolith_present")

        # -------- Strong-differential ( whitelist) --------
        strong_differential_signals: list[str] = []
        for cf in self._APPENDICITIS_ALL_CONTRADICTORY_FEATURES:
            v = _present_bool(evidence.get(cf))
            criteria_eval[cf] = _status_word(v)
            if v is True:
                strong_differential_signals.append(cf)
        strong_differential_fires = any(
            cf in strong_differential_signals
            for cf in self._APPENDICITIS_STRONG_DIFFERENTIAL_FEATURES
        )

        # Aux signal: progressive pain (per abstention rule 5).
        progressive_pain = (
            _present_bool(
                evidence.get("negative_imaging_but_persistent_or_progressive_pain")
            ) is True
        )

        # Pregnancy flag
        if population_context == "pregnant" and signal_I == "absent":
            flags.append("pregnancy_requires_imaging_pathway")

        # ---------- Decision flow (priority cascade) ----------
        outcome_key: str
        branch: str

        # A documented PRIOR appendectomy makes acute appendicitis
        # anatomically impossible (no appendix). Hard-veto BEFORE any confirming
        # branch -- record-driven (the discharge-dx "s/p appendectomy" of a CURRENT
        # appendicitis is excluded by the detector). It closes a true diverticulitis
        # listing "Appendectomy" in past surgical history, where the 8B confirmed
        # branch E from a pelvic abscess mis-mapped to periappendiceal_abscess.
        if bool(evidence.get("_record_prior_appendectomy")):
            flags.append("prior_appendectomy_excludes_appendicitis")
            outcome_key, branch = "alternative", "P"
        elif strong_differential_fires and signal_C != "true":
            outcome_key, branch = "alternative", "A"
        elif signal_X == "present" and signal_I == "positive_complicated":
            outcome_key, branch = "complicated", "B"
        elif signal_X == "present" and signal_I == "positive_uncomplicated":
            outcome_key, branch = "complicated", "C"
        elif signal_I == "positive_complicated":
            outcome_key, branch = "complicated", "D"
        elif signal_X_genuine == "present" and signal_C == "true":
            # Imaging-free complicated confirm requires a GENUINE complication
            # (perforation/abscess/phlegmon/gangrene/peritonitis/sepsis), not a
            # lab/duration association alone.
            outcome_key, branch = "complicated", "E"
        elif signal_I in ("positive_uncomplicated", "positive_complicated") and signal_C == "true":
            outcome_key, branch = "confirmed", "F"
        elif signal_C == "true" and signal_S == "high" and age_lt_40:
            outcome_key, branch = "confirmed", "G"
        elif (
            signal_I == "positive_uncomplicated"
            and imaging_nondiameter_inflammatory_sign
        ):
            # A diagnostic appendix CT/US/MRI carrying a SPECIFIC inflammatory
            # sign beyond bare diameter confirms regardless of the clinical/score
            # band (signal_C is necessarily != "true" here -- branch F caught that).
            outcome_key, branch = "confirmed", "F2"
        elif signal_I in ("positive_uncomplicated", "positive_complicated") and signal_C != "true":
            outcome_key, branch = "intermediate", "H"
        elif signal_I == "negative" and progressive_pain:
            outcome_key, branch = "intermediate", "I"
        elif signal_I == "negative" and signal_C != "true" and not progressive_pain:
            outcome_key, branch = "low_risk", "J"
        elif (
            signal_C == "true"
            and (signal_L == "true" or signal_S == "intermediate")
            and signal_I in ("absent", "unclear")
        ):
            outcome_key, branch = "intermediate", "K"
        elif (
            signal_C == "weak"
            and (signal_L == "true" or signal_S in ("intermediate", "high"))
        ):
            outcome_key, branch = "intermediate", "L"
        elif signal_C == "false" and signal_L == "false":
            outcome_key, branch = "low_risk", "M"
        elif (
            signal_C in ("absent", "false")
            and signal_L == "absent"
            and signal_I == "absent"
            and signal_S == "undetermined"
        ):
            outcome_key, branch = "alternative", "N"
        else:
            outcome_key, branch = "intermediate", "O"

        label = label_map[outcome_key]

        outcome: dict[str, Any] = {
            "rule_type": "tailored_stepwise_risk_stratification",
            "rule_id": str(self.kb.primary_rule.get("id", "")),
            "branch": branch,
            "outcome_key": outcome_key,
            "signal_C_clinical":      signal_C,
            "signal_L_lab":           signal_L,
            "signal_I_imaging":       signal_I,
            "signal_S_score_band":    signal_S,
            "signal_X_complication":  signal_X,
            "signal_X_genuine":       signal_X_genuine,
            "scores_used": {
                "AIR_score":              air_score,
                "AAS_score":              aas_score,
                "Alvarado_score_extracted": alvarado_extracted,
                "Alvarado_score_computed":  alvarado_computed,
                "PAS_score":              pas_score,
                "score_used_for_band":    score_used,
                "score_value":            score_value,
            },
            "patterns_fired":   patterns_fired,
            "complication_flags": complication_flags,
            "strong_differential_signals": strong_differential_signals,
            "progressive_pain": progressive_pain,
            "population_context": population_context,
            "age_lt_40":        age_lt_40,
            "appendicolith_present": appendicolith_present,
            "imaging_modalities_done": {"us": us_done, "ct": ct_done, "mri": mri_done},
            "flags":  flags,
            "wbc_value":  wbc_value,
            "crp_value":  crp_value,
            "anc_value":  anc_value,
        }

        if label not in self.kb.diagnostic_outputs:
            raise VerifierError(
                f"emitted label {label!r} not in {self.kb.disease!r} diagnostic_outputs"
            )
        return label, outcome, criteria_eval

    def _check_abstentions_simple(
        self, criteria_eval: dict[str, str]
    ) -> tuple[bool, str | None]:
        """Lightweight abstention check used by n_of_m_rule."""
        if not criteria_eval:
            return True, "no criteria evaluated"
        n_indet = sum(1 for v in criteria_eval.values() if v == "indeterminate")
        n_total = len(criteria_eval)
        if n_indet > n_total / 2:
            return True, f"{n_indet}/{n_total} criteria are indeterminate"
        return False, None
