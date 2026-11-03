"""Extraction schema for appendicitis.

Mirrors the evidence dict the appendicitis verifier reads
(WSES 2020 tailored stepwise risk stratification). Every field defaults to an empty finding and unknown keys are
ignored.
"""

from __future__ import annotations

from typing import Any, Literal

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


class AppendicitisUSFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    us_appendix_diameter_gt_6mm:            CtFinding = Field(default_factory=empty_ct)
    us_appendicolith:                       CtFinding = Field(default_factory=empty_ct)
    us_echogenic_periappendiceal_fat:       CtFinding = Field(default_factory=empty_ct)
    us_diameter_lt_5mm_and_compressibility: CtFinding = Field(default_factory=empty_ct)


class AppendicitisCTFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    appendix_outer_diameter_gt_6_to_9_mm:      CtFinding = Field(default_factory=empty_ct)
    appendiceal_wall_thickening_gt_2_to_3_mm:  CtFinding = Field(default_factory=empty_ct)
    periappendiceal_fat_stranding:             CtFinding = Field(default_factory=empty_ct)
    appendiceal_wall_enhancement:              CtFinding = Field(default_factory=empty_ct)
    appendicolith:                             CtFinding = Field(default_factory=empty_ct)
    developing_abscess_fluid_collection:       CtFinding = Field(default_factory=empty_ct)


class AppendicitisMRIFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mri_luminal_distension_dilation: CtFinding = Field(default_factory=empty_ct)
    mri_wall_thickening:             CtFinding = Field(default_factory=empty_ct)
    mri_free_periappendiceal_fluid:  CtFinding = Field(default_factory=empty_ct)


class AppendicitisImaging(BaseModel):
    model_config = ConfigDict(extra="ignore")

    us_done:  bool = False
    ct_done:  bool = False
    mri_done: bool = False
    us_findings:  AppendicitisUSFindings  = Field(default_factory=AppendicitisUSFindings)
    # FREE-FORM CT findings (mirrors cholecystitis). The 8B
    # emits per-sign appendiceal CT findings under widely-varying keys; a fixed
    # model silently dropped any non-canonical key via extra="ignore", causing
    # the Verifier's signal_I to under-read a positive CT appendix (review sample
    # appendicitis recall misses). Keeping ct_findings free-form means no finding
    # is lost to naming variance; the Verifier classifies the keys against the
    # appendiceal CT-positive-feature vocabulary.
    ct_findings:  dict[str, CtFinding] = Field(default_factory=dict)
    mri_findings: AppendicitisMRIFindings = Field(default_factory=AppendicitisMRIFindings)

    @field_validator("us_done", "ct_done", "mri_done", mode="before")
    @classmethod
    def _coerce_done(cls, v):
        # Coerce dict-wrapped modality-done flags to bool (see _common.coerce_done_flag).
        return coerce_done_flag(v)

    @field_validator("ct_findings", mode="before")
    @classmethod
    def _coerce_ct_findings(cls, v: Any) -> Any:
        """Robustly coerce the free-form ct_findings mapping. Non-dict -> {};
        bare-bool leaf -> {value: bool}; dict leaf kept; anything else dropped
        (mirrors cholecystitis, so a malformed CT emission never crashes)."""
        if not isinstance(v, dict):
            return {}
        out: dict[str, Any] = {}
        for k, val in v.items():
            if isinstance(val, bool):
                out[str(k)] = {"value": val, "confidence": "explicit"}
            elif isinstance(val, dict):
                out[str(k)] = val
        return out


class AppendicitisScores(BaseModel):
    """Pre-computed clinical scores. None means not extracted / not computable.

    The Verifier prefers AIR > AAS > Alvarado_extracted > Alvarado_computed > PAS
    (with PAS preferred when population_context=='pediatric'). The Verifier
    will also compute Alvarado on-the-fly from individual components if no
    pre-computed score is given.
    """

    model_config = ConfigDict(extra="ignore")

    AIR_score:      NumericFinding | None = None    # 0-12
    AAS_score:      NumericFinding | None = None    # 0-25+
    Alvarado_score: NumericFinding | None = None    # 0-9
    PAS_score:      NumericFinding | None = None    # 0-10 (pediatric)


PopulationContext = Literal["adult", "pediatric", "pregnant", "elderly", "immunocompromised"]


class AppendicitisExtraction(BaseModel):
    """Appendicitis extraction (tailored_stepwise_risk_stratification)."""

    model_config = ConfigDict(extra="ignore")

    # Clinical symptoms
    pain_periumbilical_or_diffuse_initial:     PresentFinding = Field(default_factory=empty_present)
    pain_migrates_to_rlq:                      PresentFinding = Field(default_factory=empty_present)
    pain_worse_with_coughing_walking_movement: PresentFinding = Field(default_factory=empty_present)
    anorexia:                                  PresentFinding = Field(default_factory=empty_present)
    nausea_or_vomiting:                        PresentFinding = Field(default_factory=empty_present)
    fever:                                     PresentFinding = Field(default_factory=empty_present)

    # Physical exam
    rlq_tenderness:                            PresentFinding = Field(default_factory=empty_present)
    localized_guarding_rlq:                    PresentFinding = Field(default_factory=empty_present)
    rebound_tenderness_mcburney:               PresentFinding = Field(default_factory=empty_present)

    # Labs
    wbc: NumericFinding = Field(default_factory=empty_numeric)
    crp: NumericFinding = Field(default_factory=empty_numeric)
    anc: NumericFinding = Field(default_factory=empty_numeric)

    # Scores (composite block)
    scores: AppendicitisScores = Field(default_factory=AppendicitisScores)

    # Imaging (composite block)
    imaging: AppendicitisImaging = Field(default_factory=AppendicitisImaging)

    # Complication features
    appendiceal_perforation:        PresentFinding = Field(default_factory=empty_present)
    periappendiceal_abscess:        PresentFinding = Field(default_factory=empty_present)
    phlegmon:                       PresentFinding = Field(default_factory=empty_present)
    gangrenous_appendicitis:        PresentFinding = Field(default_factory=empty_present)
    diffuse_peritonitis:            PresentFinding = Field(default_factory=empty_present)
    sepsis:                         PresentFinding = Field(default_factory=empty_present)
    prolonged_symptoms_beyond_48h:  PresentFinding = Field(default_factory=empty_present)

    # Population context
    population_context: PopulationContext = "adult"
    age_lt_40:          bool | None = None     # None defaults to True

    # ----------------------------------------------------------------
    # The 8B sometimes emits population_context
    # and age_lt_40 wrapped in a NumericFinding-shaped dict instead of as a
    # bare scalar (e.g. ``{"value": "adult", "unit": "", ...}``), which crashes
    # pydantic with `literal_error`. Coerce
    # dict-wrapped scalars to their inner value before validation.
    # ----------------------------------------------------------------
    @field_validator("population_context", mode="before")
    @classmethod
    def _unwrap_population_context(cls, v: Any) -> Any:
        if isinstance(v, dict):
            # Prefer "value" (Shape B / NumericFinding key); fall back to
            # "present" (Shape A) or a nested "population_context" key.
            for k in ("value", "present", "population_context"):
                inner = v.get(k)
                if isinstance(inner, str):
                    return inner
            return "adult"   # malformed dict -> safe default
        return v

    @field_validator("age_lt_40", mode="before")
    @classmethod
    def _unwrap_age_lt_40(cls, v: Any) -> Any:
        if isinstance(v, dict):
            for k in ("value", "present", "age_lt_40"):
                inner = v.get(k)
                if isinstance(inner, bool):
                    return inner
            return None    # let the schema apply the default downstream
        return v

    # Strong-differential signals.
    dominant_watery_diarrhea_or_viral_syndrome:                                     PresentFinding = Field(default_factory=empty_present)
    flank_pain_hematuria_or_colicky_urinary_pain:                                   PresentFinding = Field(default_factory=empty_present)
    pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass: PresentFinding = Field(default_factory=empty_present)
    known_crohn_or_terminal_ileitis_features:                                       PresentFinding = Field(default_factory=empty_present)
    # Abstention rule 5 signal (NOT a contradiction; drives progressive_pain aux)
    negative_imaging_but_persistent_or_progressive_pain:                            PresentFinding = Field(default_factory=empty_present)

    provenance: CaseProvenance

    def to_verifier_evidence(self) -> dict[str, Any]:
        """Produce the dict shape Verifier(app_kb).verify(...) accepts."""
        usf = self.imaging.us_findings
        ctf = self.imaging.ct_findings
        mrif = self.imaging.mri_findings
        out: dict[str, Any] = {
            # Clinical symptoms
            "pain_periumbilical_or_diffuse_initial":     present_to_verifier(self.pain_periumbilical_or_diffuse_initial),
            "pain_migrates_to_rlq":                      present_to_verifier(self.pain_migrates_to_rlq),
            "pain_worse_with_coughing_walking_movement": present_to_verifier(self.pain_worse_with_coughing_walking_movement),
            "anorexia":                                  present_to_verifier(self.anorexia),
            "nausea_or_vomiting":                        present_to_verifier(self.nausea_or_vomiting),
            "fever":                                     present_to_verifier(self.fever),
            # PE
            "rlq_tenderness":                            present_to_verifier(self.rlq_tenderness),
            "localized_guarding_rlq":                    present_to_verifier(self.localized_guarding_rlq),
            "rebound_tenderness_mcburney":               present_to_verifier(self.rebound_tenderness_mcburney),
            # Labs
            "wbc": numeric_to_verifier(self.wbc),
            "crp": numeric_to_verifier(self.crp),
            "anc": numeric_to_verifier(self.anc),
            # Imaging
            "imaging": {
                "us_done":  self.imaging.us_done,
                "ct_done":  self.imaging.ct_done,
                "mri_done": self.imaging.mri_done,
                "us_findings": {
                    "us_appendix_diameter_gt_6mm":            ct_to_verifier(usf.us_appendix_diameter_gt_6mm),
                    "us_appendicolith":                       ct_to_verifier(usf.us_appendicolith),
                    "us_echogenic_periappendiceal_fat":       ct_to_verifier(usf.us_echogenic_periappendiceal_fat),
                    "us_diameter_lt_5mm_and_compressibility": ct_to_verifier(usf.us_diameter_lt_5mm_and_compressibility),
                },
                "ct_findings": {
                    k: ct_to_verifier(f) for k, f in ctf.items()
                },
                "mri_findings": {
                    "mri_luminal_distension_dilation": ct_to_verifier(mrif.mri_luminal_distension_dilation),
                    "mri_wall_thickening":             ct_to_verifier(mrif.mri_wall_thickening),
                    "mri_free_periappendiceal_fluid":  ct_to_verifier(mrif.mri_free_periappendiceal_fluid),
                },
            },
            # Complications
            "appendiceal_perforation":       present_to_verifier(self.appendiceal_perforation),
            "periappendiceal_abscess":       present_to_verifier(self.periappendiceal_abscess),
            "phlegmon":                      present_to_verifier(self.phlegmon),
            "gangrenous_appendicitis":       present_to_verifier(self.gangrenous_appendicitis),
            "diffuse_peritonitis":           present_to_verifier(self.diffuse_peritonitis),
            "sepsis":                        present_to_verifier(self.sepsis),
            "prolonged_symptoms_beyond_48h": present_to_verifier(self.prolonged_symptoms_beyond_48h),
            # Population context
            "population_context": self.population_context,
            "age_lt_40":          self.age_lt_40 if self.age_lt_40 is not None else True,
            # Strong-differential signals
            "dominant_watery_diarrhea_or_viral_syndrome": present_to_verifier(self.dominant_watery_diarrhea_or_viral_syndrome),
            "flank_pain_hematuria_or_colicky_urinary_pain": present_to_verifier(self.flank_pain_hematuria_or_colicky_urinary_pain),
            "pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass":
                present_to_verifier(self.pelvic_pain_cervical_motion_tenderness_positive_pregnancy_test_or_adnexal_mass),
            "known_crohn_or_terminal_ileitis_features": present_to_verifier(self.known_crohn_or_terminal_ileitis_features),
            "negative_imaging_but_persistent_or_progressive_pain":
                present_to_verifier(self.negative_imaging_but_persistent_or_progressive_pain),
        }
        # Pre-computed scores (only emit if extracted)
        for attr in ("AIR_score", "AAS_score", "Alvarado_score", "PAS_score"):
            v = getattr(self.scores, attr)
            if v is not None:
                out[attr] = numeric_to_verifier(v)
        return out
