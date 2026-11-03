"""Extraction schema for pancreatitis.

Mirrors the evidence dict the pancreatitis verifier reads
(Revised Atlanta two-of-three rule). Every field defaults to an empty finding and unknown keys are
ignored.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.schemas._common import (
    CaseProvenance,
    NumericFinding,
    PresentFinding,
    empty_numeric,
    empty_present,
    numeric_to_verifier,
    present_to_verifier,
)


class PancreatitisExtraction(BaseModel):
    """Extraction for pancreatitis. The Verifier checks 3 criteria (n_of_m).

    AP_C1_PAIN     : characteristic epigastric pain (often radiating to back)
    AP_C2_ENZYMES  : lipase or amylase > 3x ULN (carries numeric value)
    AP_C3_IMAGING : characteristic imaging finding (CT/MRI/US)
    """

    model_config = ConfigDict(extra="ignore")

    AP_C1_PAIN:    PresentFinding = Field(default_factory=empty_present)
    AP_C2_ENZYMES: NumericFinding = Field(default_factory=empty_numeric)
    AP_C3_IMAGING: PresentFinding = Field(default_factory=empty_present)
    provenance:    CaseProvenance

    def to_verifier_evidence(self) -> dict[str, Any]:
        """Produce the dict shape Verifier(panc_kb).verify(...) accepts."""
        return {
            "AP_C1_PAIN":    present_to_verifier(self.AP_C1_PAIN),
            "AP_C2_ENZYMES": numeric_to_verifier(self.AP_C2_ENZYMES),
            "AP_C3_IMAGING": present_to_verifier(self.AP_C3_IMAGING),
        }
