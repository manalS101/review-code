"""Extraction schema for cholecystitis.

Mirrors the evidence dict the cholecystitis verifier reads
(WSES 2020 integrated assessment). Every field defaults to an empty finding and unknown keys are
ignored.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas._common import (
    CaseProvenance,
    CtFinding,
    NumericFinding,
    PresentFinding,
    coerce_done_flag,
    ct_to_verifier,
    empty_ct,
    empty_numeric,
    empty_present,
    numeric_to_verifier,
    present_to_verifier,
)


class CholecystitisUSFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ultrasound_gallstones:             CtFinding = Field(default_factory=empty_ct)
    ultrasound_wall_thickening_gt_3mm: CtFinding = Field(default_factory=empty_ct)
    pericholecystic_fluid:             CtFinding = Field(default_factory=empty_ct)
    sonographic_murphy_sign:           CtFinding = Field(default_factory=empty_ct)
    gallbladder_distention:            CtFinding = Field(default_factory=empty_ct)


class CholecystitisHidaFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hidascan_nonfilling_gallbladder: CtFinding = Field(default_factory=empty_ct)


class CholecystitisImaging(BaseModel):
    model_config = ConfigDict(extra="ignore")

    us_done:   bool = False
    ct_done:   bool = False
    hida_done: bool = False
    us_findings:   CholecystitisUSFindings   = Field(default_factory=CholecystitisUSFindings)
    hida_findings: CholecystitisHidaFindings = Field(default_factory=CholecystitisHidaFindings)
    # Legacy single-leaf CT support, kept for backward compatibility with older
    # per-case JSONs and the Verifier's legacy read path.
    ct_supportive_findings: CtFinding = Field(default_factory=empty_ct)
    # Nested, FREE-FORM CT findings. The 8B emits per-sign CT
    # findings under imaging.ct_findings.<key>, but the key naming varies widely
    # (gallbladder_wall_thickening / _gt_3mm / wall_edema; pericholecystic_
    # fluid|collection|stranding; distention|distended; gangrene|perforation).
    # Keeping this a free-form dict[str, CtFinding] means NO finding is dropped
    # to naming variance (the old fixed single-leaf contract silently dropped
    # the whole nested block via extra="ignore"). The Verifier classifies the
    # keys against the WSES/StatPearls CT-positive-feature vocabulary
    # (KB imaging_evidence.normalization_notes.ct_positive_features).
    ct_findings: dict[str, CtFinding] = Field(default_factory=dict)

    @field_validator("us_done", "ct_done", "hida_done", mode="before")
    @classmethod
    def _coerce_done(cls, v: Any) -> Any:
        # 8B sometimes wraps a modality-done flag in a finding dict; extract the
        # inner bool so a plain bool field never crashes validation. See
        # _common.coerce_done_flag.
        return coerce_done_flag(v)

    @field_validator("ct_findings", mode="before")
    @classmethod
    def _coerce_ct_findings(cls, v: Any) -> Any:
        """Robustly coerce the free-form ct_findings mapping. Non-dict -> {};
        bare-bool leaf -> {value: bool}; dict leaf kept; anything else dropped.
        Mirrors the lenient coercion used elsewhere so a malformed CT emission
        never crashes extraction (it just yields no CT support)."""
        if not isinstance(v, dict):
            return {}
        out: dict[str, Any] = {}
        for k, val in v.items():
            if isinstance(val, bool):
                out[str(k)] = {"value": val, "confidence": "explicit"}
            elif isinstance(val, dict):
                out[str(k)] = val
        return out


class CholecystitisCBDS(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cbd_stone_on_us: PresentFinding = Field(default_factory=empty_present)
    cbd_dilation:    PresentFinding = Field(default_factory=empty_present)


class CholecystitisExtraction(BaseModel):
    """Cholecystitis extraction (integrated_assessment framework)."""

    model_config = ConfigDict(extra="ignore")

    # Clinical
    ruq_pain_or_tenderness:                 PresentFinding = Field(default_factory=empty_present)
    murphy_sign:                            PresentFinding = Field(default_factory=empty_present)
    fever:                                  PresentFinding = Field(default_factory=empty_present)
    vomiting_or_food_intolerance:           PresentFinding = Field(default_factory=empty_present)
    radiation_to_right_shoulder_or_midback: PresentFinding = Field(default_factory=empty_present)

    # Labs (numeric + present flag; Verifier reads `present` for elevated_* keys)
    elevated_wbc:           NumericFinding = Field(default_factory=empty_numeric)
    elevated_crp:           NumericFinding = Field(default_factory=empty_numeric)
    total_bilirubin:        NumericFinding = Field(default_factory=empty_numeric)
    abnormal_liver_enzymes: PresentFinding = Field(default_factory=empty_present)
    # `marked_transaminitis` removed as a
    # standalone PresentFinding; replaced by per-lab NumericFindings `ast` and
    # `alt`. The Verifier derives the marked-transaminitis pattern internally
    # (OR over the numeric check on either lab >= MARKED_TRANSAMINITIS_MULTIPLIER * ref_upper).
    # Removing the LLM-level `marked_transaminitis` rubber-stamp surface was
    # one of the highest-leverage moves for cholecystitis confirmability per
    # v5 (marked_transaminitis dominated the residual branch-A FPs).
    # Backward-compat: old per-case JSONs with `marked_transaminitis:{present:...}`
    # parse silently (extra="ignore" drops the unknown field); Verifier sees no
    # ast/alt, falls back to indeterminate -> not added to strong_differential.
    # See `
    ast: NumericFinding = Field(default_factory=empty_numeric)   # AST/SGOT, U/L
    alt: NumericFinding = Field(default_factory=empty_numeric)   # ALT/SGPT, U/L

    # Imaging + CBDS
    imaging:       CholecystitisImaging = Field(default_factory=CholecystitisImaging)
    cbds_features: CholecystitisCBDS    = Field(default_factory=CholecystitisCBDS)

    # Complications (each PresentFinding)
    gangrenous_cholecystitis:    PresentFinding = Field(default_factory=empty_present)
    gallbladder_perforation:     PresentFinding = Field(default_factory=empty_present)
    emphysematous_cholecystitis: PresentFinding = Field(default_factory=empty_present)
    intraabdominal_abscess:      PresentFinding = Field(default_factory=empty_present)
    sepsis:                      PresentFinding = Field(default_factory=empty_present)

    # Strong-differential signals, all optional.
    # lipase_3x_uln_pattern is a NumericFinding rather than a PresentFinding so it
    # can carry value and ref_upper for the verifier's 3x ULN cross-check. An old
    # PresentFinding-shaped payload still parses, with value and ref_upper left
    # None, which falls back to reading the model's `present` claim.
    lipase_3x_uln_pattern:                            NumericFinding = Field(default_factory=empty_numeric)
    charcot_triad_pattern:                            PresentFinding = Field(default_factory=empty_present)
    epigastric_rather_than_ruq:                       PresentFinding = Field(default_factory=empty_present)
    # `marked_transaminitis` is not listed here; the verifier derives it from
    # the `ast` and `alt` findings above.
    normal_us_with_persistent_pain:                   PresentFinding = Field(default_factory=empty_present)
    no_ruq_pain_or_tenderness:                        PresentFinding | None = None
    normal_inflammatory_markers_with_brief_pain:      PresentFinding | None = None
    gallstones_without_inflammatory_imaging_features: PresentFinding | None = None
    elevated_bilirubin_alone_for_cbds:                PresentFinding | None = None

    provenance: CaseProvenance

    def to_verifier_evidence(self) -> dict[str, Any]:
        """Produce the dict shape Verifier(chol_kb).verify(...) accepts."""
        usf = self.imaging.us_findings
        hidaf = self.imaging.hida_findings
        out: dict[str, Any] = {
            # Clinical
            "ruq_pain_or_tenderness":                 present_to_verifier(self.ruq_pain_or_tenderness),
            "murphy_sign":                            present_to_verifier(self.murphy_sign),
            "fever":                                  present_to_verifier(self.fever),
            "vomiting_or_food_intolerance":           present_to_verifier(self.vomiting_or_food_intolerance),
            "radiation_to_right_shoulder_or_midback": present_to_verifier(self.radiation_to_right_shoulder_or_midback),
            # Labs
            "elevated_wbc":           numeric_to_verifier(self.elevated_wbc),
            "elevated_crp":           numeric_to_verifier(self.elevated_crp),
            "total_bilirubin":        numeric_to_verifier(self.total_bilirubin),
            "abnormal_liver_enzymes": present_to_verifier(self.abnormal_liver_enzymes),
            # `ast`/`alt` feed Verifier-internal
            # marked-transaminitis derivation (OR over the numeric check on each lab).
            "ast":                    numeric_to_verifier(self.ast),
            "alt":                    numeric_to_verifier(self.alt),
            # Imaging
            "imaging": {
                "us_done":   self.imaging.us_done,
                "ct_done":   self.imaging.ct_done,
                "hida_done": self.imaging.hida_done,
                "us_findings": {
                    "ultrasound_gallstones":             ct_to_verifier(usf.ultrasound_gallstones),
                    "ultrasound_wall_thickening_gt_3mm": ct_to_verifier(usf.ultrasound_wall_thickening_gt_3mm),
                    "pericholecystic_fluid":             ct_to_verifier(usf.pericholecystic_fluid),
                    "sonographic_murphy_sign":           ct_to_verifier(usf.sonographic_murphy_sign),
                    "gallbladder_distention":            ct_to_verifier(usf.gallbladder_distention),
                },
                "hida_findings": {
                    "hidascan_nonfilling_gallbladder": ct_to_verifier(hidaf.hidascan_nonfilling_gallbladder),
                },
                "ct_supportive_findings": ct_to_verifier(self.imaging.ct_supportive_findings),
                "ct_findings": {
                    k: ct_to_verifier(f) for k, f in self.imaging.ct_findings.items()
                },
            },
            # CBDS
            "cbds_features": {
                "cbd_stone_on_us": present_to_verifier(self.cbds_features.cbd_stone_on_us),
                "cbd_dilation":    present_to_verifier(self.cbds_features.cbd_dilation),
            },
            # Complications
            "gangrenous_cholecystitis":    present_to_verifier(self.gangrenous_cholecystitis),
            "gallbladder_perforation":     present_to_verifier(self.gallbladder_perforation),
            "emphysematous_cholecystitis": present_to_verifier(self.emphysematous_cholecystitis),
            "intraabdominal_abscess":      present_to_verifier(self.intraabdominal_abscess),
            "sepsis":                      present_to_verifier(self.sepsis),
            # Strong-differential signals (the numeric-threshold work: lipase_3x_uln_pattern is
            # now a NumericFinding; verifier reads value+ref_upper for numeric-cross-check dispatch).
            "lipase_3x_uln_pattern":           numeric_to_verifier(self.lipase_3x_uln_pattern),
            "charcot_triad_pattern":           present_to_verifier(self.charcot_triad_pattern),
            "epigastric_rather_than_ruq":      present_to_verifier(self.epigastric_rather_than_ruq),
            # `marked_transaminitis` removed (the numeric-threshold work); derived
            # Verifier-side from the `ast`/`alt` keys emitted above.
            "normal_us_with_persistent_pain":  present_to_verifier(self.normal_us_with_persistent_pain),
        }
        # Optional contradictory features (only emit if extracted)
        for attr in (
            "no_ruq_pain_or_tenderness",
            "normal_inflammatory_markers_with_brief_pain",
            "gallstones_without_inflammatory_imaging_features",
            "elevated_bilirubin_alone_for_cbds",
        ):
            v = getattr(self, attr)
            if v is not None:
                out[attr] = present_to_verifier(v)
        return out
