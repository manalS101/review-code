"""Per-disease extraction schemas.

Each `<Disease>Extraction` model mirrors the evidence dict its verifier expects
and adds the audit fields shared by every finding. Calling
`.to_verifier_evidence()` produces exactly the dict shape that
`Verifier(<disease>_kb).verify()` accepts.

`DISEASE_TO_SCHEMA` maps a disease name to its extraction class.
"""

from __future__ import annotations

from src.schemas._common import (
    CaseProvenance,
    Confidence,
    CtFinding,
    NumericFinding,
    PresentFinding,
    _AuditBase,
)
from src.schemas.appendicitis import (
    AppendicitisCTFindings,
    AppendicitisExtraction,
    AppendicitisImaging,
    AppendicitisMRIFindings,
    AppendicitisScores,
    AppendicitisUSFindings,
)
from src.schemas.cholecystitis import (
    CholecystitisCBDS,
    CholecystitisExtraction,
    CholecystitisHidaFindings,
    CholecystitisImaging,
    CholecystitisUSFindings,
)
from src.schemas.diverticulitis import (
    DiverticulitisCTFindings,
    DiverticulitisExtraction,
    DiverticulitisImaging,
)
from src.schemas.pancreatitis import PancreatitisExtraction

# Registry: disease name -> Extraction class
DISEASE_TO_SCHEMA: dict[str, type] = {
    "appendicitis":   AppendicitisExtraction,
    "cholecystitis":  CholecystitisExtraction,
    "diverticulitis": DiverticulitisExtraction,
    "pancreatitis":   PancreatitisExtraction,
}

__all__ = [
    # Common
    "Confidence",
    "PresentFinding",
    "NumericFinding",
    "CtFinding",
    "CaseProvenance",
    # Per-disease
    "AppendicitisExtraction",
    "CholecystitisExtraction",
    "DiverticulitisExtraction",
    "PancreatitisExtraction",
    # Sub-models (handy for tests)
    "AppendicitisImaging",
    "AppendicitisUSFindings",
    "AppendicitisCTFindings",
    "AppendicitisMRIFindings",
    "AppendicitisScores",
    "CholecystitisImaging",
    "CholecystitisUSFindings",
    "CholecystitisHidaFindings",
    "CholecystitisCBDS",
    "DiverticulitisImaging",
    "DiverticulitisCTFindings",
    # Registry
    "DISEASE_TO_SCHEMA",
]
