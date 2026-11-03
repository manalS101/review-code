"""Case loading from the MIMIC-CDM CSV export.

Assembles one admission (by hadm_id) into a dict with Hager-compatible section
keys, so a case built here can be dropped into the baseline framework
unchanged. Labs come from the normalized table rather than the basic one: same
rows, but it carries reference ranges and disease-relevance flags that the
verifier cross-checks against the model's numeric claims.

Missing sections are filled with empty defaults and logged, never faked. The
source CSVs carry no timestamps, so any guideline criterion with a temporal
component cannot be enforced and is treated as unavailable.

    Adapter(data_dir).load_case(hadm_id) -> dict
    Adapter(data_dir).reference_label_for(hadm_id) -> str | None
    Adapter(data_dir).all_hadm_ids() -> dict[str, list[int]]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "1.0"


class Adapter:
    """Loads MIMIC-CDM cases into the framework case dict format.

    All source CSVs are loaded ONCE at construction time. Subsequent
    .load_case(hadm_id) calls are O(boolean filter) on cached DataFrames.

    Example:
        adapter = Adapter("data/")
        case = adapter.load_case(hadm_id)
        print(case["_case_meta"]["reference_label"])
    """

    # Required source files (relative to data_dir)
    REQUIRED_FILES = [
        "history_of_present_illness.csv",
        "physical_examination.csv",
        "normalized_laboratory_tests.csv",
        "microbiology.csv",
        "radiology_reports.csv",
        "discharge_diagnosis.csv",
        "discharge_procedures.csv",
        "icd_diagnosis.csv",
        "icd_procedures.csv",
        "pathology_ids.json",
    ]

    def __init__(
        self,
        data_dir: str | Path,
        kb_tag: str = "kb-v4-frozen",
        warnings_log_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.kb_tag = kb_tag
        self.warnings_log_path = (
            Path(warnings_log_path) if warnings_log_path else None
        )

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        # Verify all required files are present
        missing = [
            f for f in self.REQUIRED_FILES if not (self.data_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"data_dir is missing required files: {missing}"
            )

        # Load all CSVs once (low_memory=False suppresses mixed-dtype warnings)
        self.hpi_df = pd.read_csv(self.data_dir / "history_of_present_illness.csv")
        self.pe_df = pd.read_csv(self.data_dir / "physical_examination.csv")
        self.labs_df = pd.read_csv(
            self.data_dir / "normalized_laboratory_tests.csv", low_memory=False
        )
        self.micro_df = pd.read_csv(self.data_dir / "microbiology.csv")
        self.rad_df = pd.read_csv(self.data_dir / "radiology_reports.csv")
        self.disch_diag_df = pd.read_csv(self.data_dir / "discharge_diagnosis.csv")
        self.disch_proc_df = pd.read_csv(self.data_dir / "discharge_procedures.csv")
        self.icd_diag_df = pd.read_csv(self.data_dir / "icd_diagnosis.csv")
        self.icd_proc_df = pd.read_csv(self.data_dir / "icd_procedures.csv")

        with open(self.data_dir / "pathology_ids.json", encoding="utf-8") as f:
            self.pathology_ids: dict[str, list[int]] = json.load(f)

        # Reverse lookup: hadm_id -> disease (for fast reference_label_for)
        self._hadm_to_disease: dict[int, str] = {}
        for disease, ids in self.pathology_ids.items():
            for hadm_id in ids:
                self._hadm_to_disease[int(hadm_id)] = disease

        # Cache the set of all known hadm_ids (any case present in HPI)
        self._all_hadm_ids = set(int(x) for x in self.hpi_df["hadm_id"].tolist())

    # PUBLIC API

    def load_case(self, hadm_id: int) -> dict[str, Any]:
        """Return the case dict for one hadm_id.

        Raises:
            KeyError: if hadm_id is not present in source CSVs.
        """
        hadm_id = int(hadm_id)
        if hadm_id not in self._all_hadm_ids:
            raise KeyError(f"hadm_id {hadm_id} not found in source CSVs")

        warnings: list[dict] = []

        hpi = self._get_hpi(hadm_id, warnings)
        pe = self._get_pe(hadm_id, warnings)
        labs = self._get_labs(hadm_id, warnings)
        micro_rows = self._get_microbiology_rows(hadm_id)
        rad = self._get_radiology(hadm_id, warnings)
        disch_diag = self._get_discharge_diagnosis(hadm_id, warnings)
        disch_proc = self._get_discharge_procedures(hadm_id)
        icd_diag = self._get_icd_diagnosis(hadm_id, warnings)
        (icd9, icd9_titles, icd10, icd10_titles) = self._get_icd_procedures(hadm_id)
        reference_label = self._hadm_to_disease.get(hadm_id)

        case: dict[str, Any] = {
            # --- Hager-compatible Title Case keys ---
            "Patient History": hpi,
            "Physical Examination": pe,
            "Laboratory Tests": labs,  # {itemid: <full normalized record>}
            "Microbiology": {
                int(row["test_itemid"]): str(row["valuestr"]) for row in micro_rows
            },
            "Microbiology Spec": {
                int(row["test_itemid"]): (
                    int(row["spec_itemid"]) if pd.notna(row["spec_itemid"]) else None
                )
                for row in micro_rows
            },
            "Radiology": rad,
            "Discharge Diagnosis": disch_diag,
            "ICD Diagnosis": icd_diag,
            "Procedures Discharge": disch_proc,
            "Procedures ICD9": icd9,
            "Procedures ICD9 Title": icd9_titles,
            "Procedures ICD10": icd10,
            "Procedures ICD10 Title": icd10_titles,
            # --- Pipeline metadata ---
            "_case_meta": {
                "hadm_id": hadm_id,
                "reference_label": reference_label,
                "kb_tag": self.kb_tag,
                "adapter_version": ADAPTER_VERSION,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "completeness": {
                    "has_hpi": bool(hpi),
                    "has_pe": bool(pe),
                    "n_lab_results": len(labs),
                    "n_microbiology": len(micro_rows),
                    "n_radiology_reports": len(rad),
                    "has_discharge_diagnosis": bool(disch_diag),
                    "n_discharge_procedures": len(disch_proc),
                    "n_icd_diagnosis": len(icd_diag),
                    "n_icd9_procedures": len(icd9),
                    "n_icd10_procedures": len(icd10),
                },
                "warnings": warnings,
            },
        }

        if warnings and self.warnings_log_path:
            self._write_warnings(hadm_id, warnings)

        return case

    def reference_label_for(self, hadm_id: int) -> str | None:
        """Return the reference label for an hadm_id (per pathology_ids.json)
        or None if unknown."""
        return self._hadm_to_disease.get(int(hadm_id))

    def all_hadm_ids(self) -> dict[str, list[int]]:
        """Return mapping of disease -> list of hadm_ids."""
        return {disease: [int(i) for i in ids] for disease, ids in self.pathology_ids.items()}

    def load_corpus(
        self, diseases: list[str] | None = None
    ) -> dict[int, dict[str, Any]]:
        """Load all cases for given diseases (or all 4 if None).

        Use sparingly. `load_case` is the primary API for downstream code;
        this is for batch operations over a whole disease at once.
        """
        target_diseases = diseases if diseases is not None else list(self.pathology_ids.keys())
        cases: dict[int, dict[str, Any]] = {}
        for disease in target_diseases:
            if disease not in self.pathology_ids:
                raise ValueError(f"unknown disease: {disease!r}")
            for hadm_id in self.pathology_ids[disease]:
                cases[int(hadm_id)] = self.load_case(hadm_id)
        return cases

    def write_sample_cases(
        self, out_dir: str | Path, n_per_disease: int = 1
    ) -> list[Path]:
        """Write `case_<hadm_id>.json` files (n_per_disease per disease).

        Used for spot-checking the conversion. Picks the first n_per_disease
        hadm_ids per disease per pathology_ids.json. Returns list of written
        paths.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for disease, ids in self.pathology_ids.items():
            for hadm_id in ids[:n_per_disease]:
                case = self.load_case(int(hadm_id))
                p = out_path / f"case_{int(hadm_id)}.json"
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(case, f, indent=2, default=_json_default, ensure_ascii=False)
                written.append(p)
        return written

    # PRIVATE HELPERS

    def _get_hpi(self, hadm_id: int, warnings: list) -> str:
        rows = self.hpi_df[self.hpi_df["hadm_id"] == hadm_id]
        if len(rows) == 0:
            warnings.append({"field": "Patient History", "issue": "no HPI row"})
            return ""
        return str(rows.iloc[0]["hpi"])

    def _get_pe(self, hadm_id: int, warnings: list) -> str:
        rows = self.pe_df[self.pe_df["hadm_id"] == hadm_id]
        if len(rows) == 0:
            warnings.append({"field": "Physical Examination", "issue": "no PE row"})
            return ""
        return str(rows.iloc[0]["pe"])

    def _get_labs(self, hadm_id: int, warnings: list) -> dict[int, dict[str, Any]]:
        """Return labs as {itemid: <full normalized record dict>}.

        Each record is the complete row from normalized_laboratory_tests.csv
        with NaN replaced by None for JSON cleanliness. The record includes
        pre-computed disease-relevance flags (rule_is_*_marker,
        is_pancreatitis_enzyme_gt_3x_uln_fixed, etc.) that the Verifier can
        consume directly (and cross-check against raw numeric_value_fixed).
        """
        rows = self.labs_df[self.labs_df["hadm_id"] == hadm_id]
        if len(rows) == 0:
            warnings.append({"field": "Laboratory Tests", "issue": "no lab rows"})
            return {}
        result: dict[int, dict[str, Any]] = {}
        for _, row in rows.iterrows():
            rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            # Coerce numpy scalar types to Python native for JSON serialization
            rec = {k: _native(v) for k, v in rec.items()}
            itemid = int(rec["itemid"])
            result[itemid] = rec
        return result

    def _get_microbiology_rows(self, hadm_id: int) -> list[dict[str, Any]]:
        rows = self.micro_df[self.micro_df["hadm_id"] == hadm_id]
        return rows.to_dict(orient="records") if len(rows) > 0 else []

    def _get_radiology(self, hadm_id: int, warnings: list) -> list[dict[str, Any]]:
        rows = self.rad_df[self.rad_df["hadm_id"] == hadm_id]
        if len(rows) == 0:
            warnings.append({"field": "Radiology", "issue": "no radiology reports"})
            return []
        return [
            {
                "Note ID": str(row["note_id"]),
                "Modality": str(row["modality"]),
                "Region": str(row["region"]),
                "Exam Name": str(row["exam_name"]),
                "Report": str(row["text"]),
            }
            for _, row in rows.iterrows()
        ]

    def _get_discharge_diagnosis(self, hadm_id: int, warnings: list) -> str:
        rows = self.disch_diag_df[self.disch_diag_df["hadm_id"] == hadm_id]
        if len(rows) == 0:
            warnings.append({"field": "Discharge Diagnosis", "issue": "no row"})
            return ""
        return str(rows.iloc[0]["discharge_diagnosis"])

    def _get_discharge_procedures(self, hadm_id: int) -> list[str]:
        rows = self.disch_proc_df[self.disch_proc_df["hadm_id"] == hadm_id]
        return [str(p) for p in rows["discharge_procedure"].tolist()]

    def _get_icd_diagnosis(self, hadm_id: int, warnings: list) -> list[str]:
        rows = self.icd_diag_df[self.icd_diag_df["hadm_id"] == hadm_id]
        if len(rows) == 0:
            warnings.append({"field": "ICD Diagnosis", "issue": "no ICD rows"})
            return []
        return [str(d) for d in rows["icd_diagnosis"].tolist()]

    def _get_icd_procedures(
        self, hadm_id: int
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        rows = self.icd_proc_df[self.icd_proc_df["hadm_id"] == hadm_id]
        icd9 = rows[rows["icd_version"] == 9]
        icd10 = rows[rows["icd_version"] == 10]
        return (
            [str(c) for c in icd9["icd_code"].tolist()],
            [str(t) for t in icd9["icd_title"].tolist()],
            [str(c) for c in icd10["icd_code"].tolist()],
            [str(t) for t in icd10["icd_title"].tolist()],
        )

    def _write_warnings(self, hadm_id: int, warnings: list[dict]) -> None:
        assert self.warnings_log_path is not None
        self.warnings_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.warnings_log_path, "a", encoding="utf-8") as f:
            for w in warnings:
                f.write(
                    json.dumps(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "hadm_id": hadm_id,
                            **w,
                        }
                    )
                    + "\n"
                )


# Module-level helpers

def _native(v: Any) -> Any:
    """Coerce numpy scalar / pandas types to JSON-friendly Python natives."""
    if v is None:
        return None
    # Use duck-typing for numpy / pandas without hard-importing numpy here
    # (pandas already imports numpy)
    import numpy as np

    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if pd.isna(f) else f
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    if isinstance(v, dict):
        return {k: _native(x) for k, x in v.items()}
    return str(v)


def _json_default(o: Any) -> Any:
    """Fallback JSON serializer for numpy scalars, datetimes, paths."""
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if pd.isna(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)
