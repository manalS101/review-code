"""Evidence tools available to an agent.

Wraps the case dict with Hager-aligned tool semantics. The agent calls
`dispatch()` once per iteration and gets back the text that becomes the next
iteration's observation.

Physical Examination returns the whole examination and ignores the action
input. This follows the baseline framework, on the grounds that splitting a
free-text examination into the parts a caller asked for is unreliable, and
withholding the rest would not reflect clinical practice.

Laboratory Tests returns only what was asked for. The input is free text, so
the dispatcher tokenizes it, expands panel names such as CBC or BMP into their
member labels, maps labels to canonical itemids through the lab mapping table,
and formats each hit with its value, unit and reference range. Itemids already
served are tracked per case so the same result is not returned twice.

Imaging returns one modality and region per call, moving to the next
chronological report when the same pair is requested again. The abdomen filter
matches the record rendering used elsewhere, so runs stay comparable.

    ToolDispatcher(case, lab_mapping_csv).dispatch(action, action_input, ctx)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.record import apply_impression_filter


# Errors

class ToolDispatchError(RuntimeError):
    """Raised when ToolDispatcher.dispatch() is given an unknown action.

    Specifically NOT raised for malformed action_input -- those degrade
    gracefully to "no matching X" observations so the Agent can recover
    on the next iteration.
    """


# Hager-aligned semantic dictionaries

# Modality keyword (lowercase) -> canonical Modality value in
# case["Radiology"][*]["Modality"]. Empirical canonicalization based on
# data/radiology_reports.csv unique-value scan (Modality values in our data:
# CT, Ultrasound, Radiograph, MRI, CTU, Drainage). Drainage / CTU are rare
# and not exposed as keyword aliases -- the LLM would have to type them
# exactly.
_MODALITY_KEYWORDS: dict[str, str] = {
    # CT
    "ct":                    "CT",
    "ct scan":               "CT",
    "cat scan":              "CT",
    "computed tomography":   "CT",
    "ctu":                   "CTU",                # CT Urogram; explicit only
    # MRI
    "mri":                   "MRI",
    "mr":                    "MRI",
    "magnetic resonance":    "MRI",
    # Ultrasound
    "ultrasound":            "Ultrasound",
    "us":                    "Ultrasound",
    "sonogram":              "Ultrasound",
    "ultrasonography":       "Ultrasound",
    "doppler":               "Ultrasound",
    # Plain film
    "radiograph":            "Radiograph",
    "x-ray":                 "Radiograph",
    "xray":                  "Radiograph",
    "plain film":            "Radiograph",
    "kub":                   "Radiograph",
}

# Modality subsumption: a requested modality also matches these specialized variants
# stored under a different ``Modality`` value in the record's Radiology table. A CT
# urogram (CTU) / angiogram (CTA) is still a CT of the abdomen; an MRCP is an MRI.
# Closes a true sigmoid diverticulitis whose diagnostic "CT ABDOMEN PELVIS"
# report is filed under Modality "CTU" and never matched a plain "Abdominal CT"
# request -> the agent never received the diagnostic CT). An explicit request for
# the variant itself (e.g. "CTU") still resolves to only that variant.
_MODALITY_SUBSUMES: dict[str, tuple[str, ...]] = {
    "CT":  ("CT", "CTU", "CTA"),
    "MRI": ("MRI", "MRCP"),
}

# Region keyword (lowercase) -> canonical Region value in
# case["Radiology"][*]["Region"]. Matches the build_record_text rendering and
# Hager filter, only "Abdomen" is served; pelvic queries route to Abdomen
# (abdomen-CTs in MIMIC routinely cover the pelvis).
_REGION_KEYWORDS: dict[str, str] = {
    "abdomen":    "Abdomen",
    "abdominal":  "Abdomen",
    "belly":      "Abdomen",
    "pelvis":     "Abdomen",   # pelvic content lives inside abdomen CTs
    "pelvic":     "Abdomen",
    "rlq":        "Abdomen",
    "llq":        "Abdomen",
    "ruq":        "Abdomen",
    "luq":        "Abdomen",
    "right lower quadrant": "Abdomen",
    "right upper quadrant": "Abdomen",
    "left lower quadrant":  "Abdomen",
    "left upper quadrant":  "Abdomen",
    "appendix":   "Abdomen",
    "gallbladder": "Abdomen",
    "biliary":    "Abdomen",
    "pancreas":   "Abdomen",
    "colon":      "Abdomen",
    "sigmoid":    "Abdomen",
    # Non-abdominal regions we recognize but explicitly DON'T serve (per
    # the abdomen-only filter). Listed so the keyword matcher still picks
    # them up; _handle_imaging() then returns a "no available reports"
    # observation because the region isn't in _ABDOMINAL_REGIONS.
    "chest":      "Chest",
    "thorax":     "Chest",
    "thoracic":   "Chest",
    "lung":       "Chest",
    "head":       "Head",
    "brain":      "Head",
    "skull":      "Head",
}

# Regions we actually return reports for.
_ABDOMINAL_REGIONS: set[str] = {"Abdomen"}


# Lab panel expansions
#
# Maps lowercased panel name -> list of canonical lab labels (matching the
# "label" column in data/lab_test_mapping.csv). Resolved at lab-handler time
# against the label index. Empirically picked: panels commonly requested in
# acute abdominal workups (Hager + WSES guidance).

_PANEL_EXPANSIONS: dict[str, list[str]] = {
    "cbc": [
        "White Blood Cells", "Hemoglobin", "Hematocrit",
        "Platelet Count", "MCV", "MCH", "MCHC", "RDW",
    ],
    "complete blood count": [
        "White Blood Cells", "Hemoglobin", "Hematocrit",
        "Platelet Count", "MCV", "MCH", "MCHC", "RDW",
    ],
    "bmp": [
        "Sodium", "Potassium", "Chloride", "Bicarbonate",
        "Urea Nitrogen", "Creatinine", "Glucose", "Calcium, Total",
    ],
    "basic metabolic panel": [
        "Sodium", "Potassium", "Chloride", "Bicarbonate",
        "Urea Nitrogen", "Creatinine", "Glucose", "Calcium, Total",
    ],
    "chem 7": [
        "Sodium", "Potassium", "Chloride", "Bicarbonate",
        "Urea Nitrogen", "Creatinine", "Glucose",
    ],
    "cmp": [
        "Sodium", "Potassium", "Chloride", "Bicarbonate",
        "Urea Nitrogen", "Creatinine", "Glucose", "Calcium, Total",
        "Bilirubin, Total", "Alkaline Phosphatase",
        "Asparate Aminotransferase (AST)", "Alanine Aminotransferase (ALT)",
        "Albumin", "Total Protein",
    ],
    "comprehensive metabolic panel": [
        "Sodium", "Potassium", "Chloride", "Bicarbonate",
        "Urea Nitrogen", "Creatinine", "Glucose", "Calcium, Total",
        "Bilirubin, Total", "Alkaline Phosphatase",
        "Asparate Aminotransferase (AST)", "Alanine Aminotransferase (ALT)",
        "Albumin", "Total Protein",
    ],
    "lft": [
        "Bilirubin, Total", "Bilirubin, Direct", "Alkaline Phosphatase",
        "Asparate Aminotransferase (AST)", "Alanine Aminotransferase (ALT)",
        "Albumin",
    ],
    "liver function tests": [
        "Bilirubin, Total", "Bilirubin, Direct", "Alkaline Phosphatase",
        "Asparate Aminotransferase (AST)", "Alanine Aminotransferase (ALT)",
        "Albumin",
    ],
    "lfts": [
        "Bilirubin, Total", "Bilirubin, Direct", "Alkaline Phosphatase",
        "Asparate Aminotransferase (AST)", "Alanine Aminotransferase (ALT)",
        "Albumin",
    ],
    "pancreatic enzymes": ["Lipase", "Amylase"],
    "coags": ["PT", "PTT", "INR"],
    "coagulation": ["PT", "PTT", "INR"],
    "inflammatory markers": ["White Blood Cells", "C-Reactive Protein"],
}

# Common shorthand -> canonical label mapping for single labs the LLM is
# likely to abbreviate. Lowercase keys.
_LAB_SHORTHAND: dict[str, str] = {
    "wbc":        "White Blood Cells",
    "hgb":        "Hemoglobin",
    "hb":         "Hemoglobin",
    "hct":        "Hematocrit",
    "plt":        "Platelet Count",
    "platelets":  "Platelet Count",
    "na":         "Sodium",
    "k":          "Potassium",
    "cl":         "Chloride",
    "hco3":       "Bicarbonate",
    "bicarb":     "Bicarbonate",
    "bun":        "Urea Nitrogen",
    "cr":         "Creatinine",
    "creat":      "Creatinine",
    "ca":         "Calcium, Total",
    "calcium":    "Calcium, Total",
    "tbili":      "Bilirubin, Total",
    "total bilirubin": "Bilirubin, Total",
    "direct bilirubin": "Bilirubin, Direct",
    "alp":        "Alkaline Phosphatase",
    "alkphos":    "Alkaline Phosphatase",
    "ast":        "Asparate Aminotransferase (AST)",
    "alt":        "Alanine Aminotransferase (ALT)",
    "sgot":       "Asparate Aminotransferase (AST)",
    "sgpt":       "Alanine Aminotransferase (ALT)",
    "crp":        "C-Reactive Protein",
    "c-reactive protein": "C-Reactive Protein",
    "lipase":     "Lipase",
    "amylase":    "Amylase",
    "lactate":    "Lactate",
}


# State carried across iterations

@dataclass
class ToolContext:
    """Mutable per-Agent-run state.

    Imaging next-chronological semantics: ``served_imaging[(modality, region)]``
    holds the COUNT of reports we've already served for that pair. The handler
    returns the report at index = served_imaging[pair] (0-indexed) and then
    increments.

    Labs dedup: ``served_lab_itemids`` keeps track of itemids we've already
    surfaced so a second "Action: Laboratory Tests / Action Input: CBC" call
    a few iterations later doesn't dump the same WBC value back into the
    prompt.
    """

    served_imaging: dict[tuple[str, str], int] = field(default_factory=dict)
    served_lab_itemids: set[int] = field(default_factory=set)


# Helpers

def _load_lab_label_index(csv_path: Path) -> dict[str, list[int]]:
    """Load data/lab_test_mapping.csv into lowercase_label -> [canonical itemids].

    Returns a dict where multiple itemids per label are possible (e.g., the
    same human label "White Blood Cells" maps to both itemid 51301 (CBC panel)
    and 51755 (manual diff)). All matching itemids are returned so downstream
    case-lookup can find any of them.
    """
    index: dict[str, list[int]] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            itemid_raw = row.get("itemid", "").strip()
            if not itemid_raw:
                continue
            try:
                itemid = int(float(itemid_raw))
            except ValueError:
                continue
            label = (row.get("label") or "").strip().lower()
            if not label:
                continue
            index.setdefault(label, []).append(itemid)
    return index


def _tokenize_lab_input(action_input: str) -> list[str]:
    """Split raw lab action_input into per-test tokens.

    Hager-aligned (DiagnosisWorkflowParser.parse_lab_tests_action_input):
      - drop "order", "run", "level(s)", "repeat", "check" wrappers
      - replace " and " / newline with ","
      - split on commas (respecting parentheses)
      - strip whitespace; drop empties
    Stop-word removal is deliberately omitted; it eats useful tokens such as
    the "of" inside "Test of X".
    """
    s = action_input or ""
    for filler in (r"\border\b", r"\brun\b", r"\blevel[s]?\b",
                   r"\brepeat\b", r"\bcheck\b"):
        s = re.sub(filler, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\band\b", ",", s, flags=re.IGNORECASE)
    s = s.replace("\n", ",")
    s = s.replace(";", ",")     # Treat semicolons as separators too
    parts = re.split(r",\s*(?![^()]*\))", s)
    return [p.strip() for p in parts if p and p.strip()]


def _resolve_lab_token(
    token: str,
    label_index: dict[str, list[int]],
) -> list[int]:
    """Resolve one user-supplied lab/panel token to canonical itemids.

    Lookup order:
      1. Panel expansion (CBC, BMP, ...) -> each canonical label -> itemids
      2. Single-lab shorthand (WBC, ALT, ...) -> canonical label -> itemids
      3. Direct label match (case-insensitive) against the CSV's label column
      4. Substring match: prefer the SHORTEST matching label so "CRP" doesn't
         match "Serum C-Reactive Protein Hi-Sens" if a plain "C-Reactive Protein"
         label exists.

    Returns [] if nothing resolved.
    """
    if not token:
        return []
    t = token.strip().lower()
    out: list[int] = []
    seen: set[int] = set()

    def _extend(itemids: list[int]) -> None:
        for i in itemids:
            if i not in seen:
                seen.add(i)
                out.append(i)

    # 1. panel
    if t in _PANEL_EXPANSIONS:
        for canonical_label in _PANEL_EXPANSIONS[t]:
            _extend(label_index.get(canonical_label.lower(), []))
        return out

    # 2. shorthand
    if t in _LAB_SHORTHAND:
        canonical_label = _LAB_SHORTHAND[t].lower()
        _extend(label_index.get(canonical_label, []))
        return out

    # 3. direct match
    if t in label_index:
        _extend(label_index[t])
        return out

    # 4. shortest-substring fallback
    candidates: list[tuple[int, str]] = [
        (len(lbl), lbl) for lbl in label_index if t in lbl
    ]
    if candidates:
        candidates.sort()
        shortest = candidates[0][1]
        _extend(label_index[shortest])
    return out


def _parse_imaging_keywords(action_input: str) -> tuple[str | None, str | None]:
    """Find the first modality + first region keyword in ``action_input``.

    Returns (modality, region). Either may be None if not detected. Keyword
    matching is greedy on the longest phrase (so "computed tomography" beats
    a bare "ct" inside it). Case-insensitive.
    """
    s = (action_input or "").lower()
    modality = _greedy_match(s, _MODALITY_KEYWORDS)
    region = _greedy_match(s, _REGION_KEYWORDS)
    return modality, region


def _greedy_match(text: str, table: dict[str, str]) -> str | None:
    """Return the table value for the longest-keyword that appears in ``text``.

    Ties broken by earliest position. Returns None if no keyword found.
    """
    best: tuple[int, int, str] | None = None  # (-len, pos, value)
    for kw, val in table.items():
        pos = text.find(kw)
        if pos < 0:
            continue
        key = (-len(kw), pos)
        if best is None or key < (best[0], best[1]):
            best = (-len(kw), pos, val)
    return best[2] if best else None


def _format_lab_record(rec: dict[str, Any]) -> str:
    """Format one row from case["Laboratory Tests"] as a single-line observation.

    Pattern: ``<Label> = <valuestr> <unit> (ref <low>-<high>)``

    Mirrors the lab rendering in src/record.py, so an agent sees lab content
    in the same shape wherever it comes from. Missing fields are elided.
    """
    label = (rec.get("label") or "").strip()
    # Prefer numeric_value_fixed, a clean number, over valuestr, to avoid
    # "9.9 K/uL K/uL" duplication when valuestr already carries the unit. Fall back to valuestr for non-numeric labs
    # (e.g., "POSITIVE", "NEGATIVE", culture growth strings).
    numeric = rec.get("numeric_value_fixed")
    if numeric is None or (isinstance(numeric, float) and numeric != numeric):
        raw = rec.get("valuestr")
        if raw is None or (isinstance(raw, float) and raw != raw):
            value_str = ""
        else:
            value_str = str(raw).strip()
    else:
        # numeric path: format cleanly without trailing zeros
        try:
            value_str = f"{float(numeric):g}"
        except (TypeError, ValueError):
            value_str = str(numeric).strip()

    # Prefer the unit_suffix_fixed column from the normalized lab table.
    unit = rec.get("unit_suffix_fixed") or ""
    if not isinstance(unit, str):
        unit = ""
    unit = unit.strip()

    ref_lo = rec.get("ref_range_lower")
    ref_hi = rec.get("ref_range_upper")

    def _fmt_ref(x: Any) -> str:
        if x is None:
            return ""
        try:
            xf = float(x)
            if xf != xf:  # NaN
                return ""
            return f"{xf:g}"
        except (TypeError, ValueError):
            return str(x).strip()

    ref_lo_s = _fmt_ref(ref_lo)
    ref_hi_s = _fmt_ref(ref_hi)

    parts = [f"{label} = {value_str}"]
    if unit:
        parts.append(unit)
    parts_str = " ".join(parts).strip()

    if ref_lo_s or ref_hi_s:
        parts_str += f" (ref {ref_lo_s}-{ref_hi_s})"
    return parts_str


# Dispatcher

class ToolDispatcher:
    """Routes (action, action_input) pairs to per-tool handlers.

    The Agent constructs ONE dispatcher per case (so the loaded case dict
    + lab mapping stay in scope for the duration of the run) and passes a
    fresh ToolContext through every iteration.
    """

    def __init__(
        self,
        case: dict[str, Any],
        *,
        lab_mapping_csv: Path | str = "data/lab_test_mapping.csv",
    ) -> None:
        self._case = case
        self._lab_mapping_csv = Path(lab_mapping_csv)
        if not self._lab_mapping_csv.exists():
            raise FileNotFoundError(
                f"lab_test_mapping.csv not found at {self._lab_mapping_csv}; "
                f"required for Laboratory Tests action expansion"
            )
        self._lab_label_index = _load_lab_label_index(self._lab_mapping_csv)

    # ---- public API ----

    def dispatch(
        self,
        action: str,
        action_input: str,
        ctx: ToolContext,
    ) -> str:
        if action == "Physical Examination":
            return self._handle_pe(ctx)
        if action == "Laboratory Tests":
            return self._handle_labs(action_input, ctx)
        if action == "Imaging":
            return self._handle_imaging(action_input, ctx)
        raise ToolDispatchError(
            f"unknown action {action!r}; valid actions are "
            f"'Physical Examination', 'Laboratory Tests', 'Imaging'"
        )

    # ---- handlers ----

    def _handle_pe(self, ctx: ToolContext) -> str:
        """Returns the entire Physical Examination blob verbatim.

        The examination is returned whole regardless of action_input,
        following the baseline framework. ctx is unused here but kept in the
        signature so every handler has the same shape.
        """
        _ = ctx
        pe = self._case.get("Physical Examination") or ""
        if not pe.strip():
            return "Physical Examination: not available."
        return f"Physical Examination:\n{pe}"

    def _handle_labs(self, action_input: str, ctx: ToolContext) -> str:
        tokens = _tokenize_lab_input(action_input)
        if not tokens:
            return ("Laboratory Tests: no tests specified. "
                    "Please name specific lab tests or panels (e.g., 'CBC, CRP').")

        # Resolve each token to itemids; collect with order preserved + dedup.
        requested_itemids: list[int] = []
        seen: set[int] = set()
        unresolved: list[str] = []
        for tok in tokens:
            ids = _resolve_lab_token(tok, self._lab_label_index)
            if not ids:
                unresolved.append(tok)
                continue
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    requested_itemids.append(i)

        # Look up each itemid in the case; honor dedup vs prior iterations.
        case_labs = self._case.get("Laboratory Tests", {}) or {}
        lines: list[str] = []
        served_now: list[int] = []
        for iid in requested_itemids:
            if iid in ctx.served_lab_itemids:
                continue
            rec = case_labs.get(iid) or case_labs.get(str(iid))
            if not rec:
                continue
            lines.append(_format_lab_record(rec))
            served_now.append(iid)

        # Update context after the loop so we don't accidentally include
        # later iterations in a single call.
        ctx.served_lab_itemids.update(served_now)

        if not lines:
            note = (
                "no matching laboratory results available for the requested tests"
            )
            if unresolved:
                note += f" (unrecognized terms: {', '.join(unresolved)})"
            return f"Laboratory Tests: {note}."

        header = "Laboratory Tests:"
        body = "\n".join(lines)
        if unresolved:
            body += f"\n(unrecognized terms not served: {', '.join(unresolved)})"
        return f"{header}\n{body}"

    def _handle_imaging(self, action_input: str, ctx: ToolContext) -> str:
        modality, region = _parse_imaging_keywords(action_input)

        if not modality and not region:
            return ("Imaging: could not parse a modality or region from the "
                    "Action Input. Specify e.g. 'Abdominal CT' or "
                    "'Ultrasound of the abdomen'.")
        if not modality:
            return ("Imaging: a region was specified but no modality. "
                    "Specify e.g. 'CT' or 'Ultrasound' along with the region.")
        if not region:
            return ("Imaging: a modality was specified but no region. "
                    "Specify e.g. 'Abdomen' or 'Pelvis' along with the modality.")

        # Abdomen-only filter, matching build_record_text and the baseline.
        if region not in _ABDOMINAL_REGIONS:
            return (
                f"Imaging: {region} {modality} not available "
                f"(this Agent serves abdominal imaging only)."
            )

        accepted_modalities = _MODALITY_SUBSUMES.get(modality, (modality,))
        reports = [
            r for r in (self._case.get("Radiology") or [])
            if str(r.get("Modality", "")).strip() in accepted_modalities
               and str(r.get("Region", "")).strip() == region
        ]
        if not reports:
            return (
                f"Imaging: no {region} {modality} report on file for this patient. "
                f"Try a different imaging modality."
            )

        # Stateful next-chronological serving. Reports are already in the
        # adapter's iteration order; assume that is chronological enough
        # for our purposes (MIMIC notes are typically ordered by time).
        idx = ctx.served_imaging.get((modality, region), 0)
        if idx >= len(reports):
            return (
                f"Imaging: no further {region} {modality} reports available; "
                f"you already requested {idx} of {len(reports)}. "
                f"Try a different imaging modality."
            )
        report = reports[idx]
        ctx.served_imaging[(modality, region)] = idx + 1

        header = f"Imaging: {modality} {region}"
        if idx > 0:
            header += f" (report {idx + 1} of {len(reports)})"
        body = (report.get("Report") or "").strip()
        if not body:
            body = "(report text empty)"
        else:
            # Withhold the radiologist's IMPRESSION/HISTORY/etc.
            # so the Agent reasons from FINDINGS, not the stated conclusion
            # (baseline parity). See record.apply_impression_filter.
            body = apply_impression_filter(body)
            if not body.strip():
                body = ("(no objective FINDINGS section available; "
                        "interpretive sections withheld)")
        return f"{header}\n{body}"
