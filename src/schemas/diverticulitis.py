"""Extraction schema for diverticulitis.

Mirrors the evidence dict the diverticulitis verifier reads
(WSES 2020 multi-evidence assessment). Every field defaults to an empty finding and unknown keys are
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


class DiverticulitisCTFindings(BaseModel):
    """7 CT findings the verifier evaluates (5 complicated, 2 stage-0)."""

    model_config = ConfigDict(extra="ignore")

    pericolic_abscess:                CtFinding = Field(default_factory=empty_ct)
    extraluminal_gas:                 CtFinding = Field(default_factory=empty_ct)
    intraabdominal_free_fluid:        CtFinding = Field(default_factory=empty_ct)
    fistula:                          CtFinding = Field(default_factory=empty_ct)
    phlegmon:                         CtFinding = Field(default_factory=empty_ct)
    diverticula_with_wall_thickening: CtFinding = Field(default_factory=empty_ct)
    pericolic_fat_stranding:          CtFinding = Field(default_factory=empty_ct)


class DiverticulitisImaging(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ct_done:     bool = False
    ct_findings: DiverticulitisCTFindings = Field(default_factory=DiverticulitisCTFindings)

    @field_validator("ct_done", mode="before")
    @classmethod
    def _coerce_done(cls, v):
        # Coerce dict-wrapped modality-done flag to bool (see _common.coerce_done_flag).
        return coerce_done_flag(v)


class DiverticulitisExtraction(BaseModel):
    """Extraction for diverticulitis (multi_evidence_assessment framework)."""

    model_config = ConfigDict(extra="ignore")

    # Clinical
    left_lower_quadrant_pain: PresentFinding = Field(default_factory=empty_present)
    llq_tenderness:           PresentFinding = Field(default_factory=empty_present)
    llq_tenderness_only:      PresentFinding = Field(default_factory=empty_present)
    absence_of_vomiting:      PresentFinding = Field(default_factory=empty_present)
    peritoneal_signs:         PresentFinding = Field(default_factory=empty_present)
    fever:                    PresentFinding = Field(default_factory=empty_present)
    change_in_bowel_habits:   PresentFinding = Field(default_factory=empty_present)
    # Labs
    crp: NumericFinding = Field(default_factory=empty_numeric)
    wbc: NumericFinding = Field(default_factory=empty_numeric)
    # Imaging
    imaging: DiverticulitisImaging = Field(default_factory=DiverticulitisImaging)
    provenance: CaseProvenance

    def to_verifier_evidence(self) -> dict[str, Any]:
        """Produce the dict shape Verifier(div_kb).verify(...) accepts."""
        ct = self.imaging.ct_findings
        return {
            "left_lower_quadrant_pain": present_to_verifier(self.left_lower_quadrant_pain),
            "llq_tenderness":           present_to_verifier(self.llq_tenderness),
            "llq_tenderness_only":      present_to_verifier(self.llq_tenderness_only),
            "absence_of_vomiting":      present_to_verifier(self.absence_of_vomiting),
            "peritoneal_signs":         present_to_verifier(self.peritoneal_signs),
            "fever":                    present_to_verifier(self.fever),
            "change_in_bowel_habits":   present_to_verifier(self.change_in_bowel_habits),
            "crp":                      numeric_to_verifier(self.crp),
            "wbc":                      numeric_to_verifier(self.wbc),
            "imaging": {
                "ct_done": self.imaging.ct_done,
                "ct_findings": {
                    "pericolic_abscess":                ct_to_verifier(ct.pericolic_abscess),
                    "extraluminal_gas":                 ct_to_verifier(ct.extraluminal_gas),
                    "intraabdominal_free_fluid":        ct_to_verifier(ct.intraabdominal_free_fluid),
                    "fistula":                          ct_to_verifier(ct.fistula),
                    "phlegmon":                         ct_to_verifier(ct.phlegmon),
                    "diverticula_with_wall_thickening": ct_to_verifier(ct.diverticula_with_wall_thickening),
                    "pericolic_fat_stranding":          ct_to_verifier(ct.pericolic_fat_stranding),
                },
            },
        }
