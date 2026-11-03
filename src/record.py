"""Record rendering and evidence-span checking.

Two jobs, both shared by every agent.

`build_record_text` renders a case dict into the single block of text that
grounding is checked against. Radiology is rendered the way the baseline
framework renders it, so a record built here and a record built there contain the
same words.

`verify_source_span` and `value_in_text` answer whether a quoted span, or a
numeric value, actually occurs in that text. Matching is deliberately tolerant:
exact substring first, then a normalized comparison, then token coverage with
stopwords dropped, so that a model quoting "RLQ tenderness" against a record
saying "tenderness in the RLQ" is not punished for word order. Numbers are
matched under the common record formattings, including thousands separators and
the unit conventions labs are written in.

`apply_impression_filter` removes the radiologist's IMPRESSION and similar
sections from a report, leaving the FINDINGS. The agent is meant to reason
from observations, not from another clinician's conclusion.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from typing import Any


_SPAN_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "with", "for", "that", "this", "was", "were", "are", "has",
    "had", "his", "her", "its", "from", "into", "onto", "than", "then", "but",
    "not", "also", "which", "who", "she", "him", "their", "they",
})

_TOKEN_COVERAGE_THRESHOLD: float = 0.80

def _alnum_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric words of length >= 3 (drops to/of/in/is punctuation)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3]

def _content_tokens(text: str) -> list[str]:
    """`_alnum_tokens` minus stopwords -- the discriminating words of a span."""
    return [w for w in _alnum_tokens(text) if w not in _SPAN_STOPWORDS]

def _token_matches(tok: str, window_tokens: set[str]) -> bool:
    """A span token matches a window token by exact equality, a shared >=5-char
    prefix (so 'radiating' == 'radiation', 'tenderness' == 'tender'), or a
    word-level fuzzy ratio >= 0.82. Short tokens (<5 chars) require exact match."""
    if tok in window_tokens:
        return True
    if len(tok) < 5:
        return False
    pre = tok[:5]
    for w in window_tokens:
        if len(w) >= 5 and (
            w[:5] == pre
            or difflib.SequenceMatcher(a=tok, b=w, autojunk=False).ratio() >= 0.82
        ):
            return True
    return False

def verify_source_span(
    span: str, record_text: str, *, fuzzy_threshold: float = 0.85
) -> bool:
    """Return True if `span` (or a close fuzzy match) appears in record_text.

    Empty `span` returns True (treated as "no claim about the source"; the
    finding may still carry a value extracted by the LLM, but it's not
    asserting a specific verbatim record location).

    Match strategy (each tier only runs if the prior fails -- purely additive,
    so nothing already verified changes):
      1. Case-insensitive substring containment -> True.
      2. Fuzzy substring match (difflib.SequenceMatcher .ratio() over a
         sliding window of comparable length) >= fuzzy_threshold -> True.
      3. Token-coverage: >= 80% of the span's CONTENT tokens (stopwords dropped)
         match -- exact / >=5-char prefix / word-fuzzy -- inside ONE record window
         of ~2x the span length -> True. Catches a faithful paraphrase or a fusion
         of verbatim fragments (e.g. the LLM joins "midepigastric abdominal pain"
         and "radiation to the back", both verbatim in the record) that the
         character-ratio rejects. The single-window CLUSTER constraint keeps it
         anti-fabrication: a made-up finding's words do not co-occur locally.
      4. Otherwise -> False.
    """
    if not span or not record_text:
        return True if not span else False

    s = span.strip().lower()
    record = record_text.lower()
    if s in record:
        return True

    # Fuzzy: pick a window in record of comparable length to span, scan in steps.
    n = len(s)
    if n == 0:
        return True
    step = max(1, n // 4)
    best = 0.0
    matcher = difflib.SequenceMatcher(a=s, autojunk=False)
    for i in range(0, max(1, len(record) - n + 1), step):
        window = record[i:i + int(n * 1.2)]
        matcher.set_seq2(window)
        r = matcher.ratio()
        if r > best:
            best = r
            if best >= fuzzy_threshold:
                return True

    # Tier 3: token-coverage within a ~2x-span window.
    span_tokens = _content_tokens(s)
    if len(span_tokens) >= 2:
        win_len = max(2 * n, n + 40)
        need = _TOKEN_COVERAGE_THRESHOLD * len(span_tokens)
        for i in range(0, max(1, len(record)), step):
            wtoks = set(_alnum_tokens(record[i:i + win_len]))
            if not wtoks:
                continue
            matched = sum(1 for t in span_tokens if _token_matches(t, wtoks))
            if matched >= need:
                return True
    return False

_INTERPRETIVE_RAD_SECTIONS: tuple[str, ...] = (
    "CLINICAL HISTORY", "MEDICAL HISTORY", "CLINICAL INFORMATION",
    "COMPARISON", "COMPARISONS", "COMMENT", "CONCLUSION", "HISTORY",
    "IMPRESSION", "CLINICAL INDICATION", "INDICATION", "OPERATORS",
    "REASON", "REFERENCE", "DATE",
)

_RAD_SECTION_HEADER_RE = re.compile(r"(?m)^([A-Z][A-Z \t,._-]*):[ \t]*")

def apply_impression_filter(report_text: str) -> str:
    """Remove radiologist-interpretation / clinical-prior sections from a
    radiology report, keeping FINDINGS and other objective sections.

    Generalized + section-header driven (no report- or dataset-specific
    strings): a section runs from one ALL-CAPS ``HEADER:`` line to the next.
    Any section whose header prefix-matches ``_INTERPRETIVE_RAD_SECTIONS`` is
    dropped; everything else (including text before the first header) is kept.

    Conservative on unstructured input: if no section headers are found, the
    report is returned unchanged rather than blanked.
    """
    if not report_text or not report_text.strip():
        return report_text

    matches = list(_RAD_SECTION_HEADER_RE.finditer(report_text))
    if not matches:
        return report_text

    kept: list[str] = []
    preamble = report_text[: matches[0].start()].strip()
    if preamble:
        kept.append(preamble)

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report_text)
        body = report_text[m.end():end].strip()
        if any(header.startswith(bad) for bad in _INTERPRETIVE_RAD_SECTIONS):
            continue
        kept.append(f"{header}:\n{body}" if body else f"{header}:")

    return "\n\n".join(kept).strip()

def value_in_text(value: float | int | None, text: str) -> bool:
    """True if the numeric `value` appears in `text`
    in any common numeric formatting (str, %g, %.1f, integer-if-integral).

    Shared between the runtime value-support guard in this module and the
    audit script `scripts/audit_extraction_vs_record.py`; the latter imports
    this function so the two layers use IDENTICAL fabrication semantics.
    A drift between the two would mean the audit flags HALLUCINATION on
    cases the runtime guard let through (or vice versa), reintroducing the
    audit-versus-runtime gap.

    Args:
        value: the numeric value emitted by the LLM (float, int, or None)
        text:  the full record text the LLM was given

    Returns:
        True if `value` appears anywhere in `text` under at least one of
        the candidate formattings; False otherwise. `value is None` -> False.
    """
    if value is None:
        return False
    candidates = {
        str(value),
        f"{value:g}",
        f"{value:.1f}",
    }
    if value == int(value):
        candidates.add(str(int(value)))
    for c in candidates:
        if c and c in text:
            return True
    return False


def build_record_text(case: dict[str, Any]) -> str:
    """Flatten the Adapter's case dict into a single record text string.

    Sections (when present): Patient History, Physical Examination,
    Laboratory Tests, Microbiology, Radiology, Discharge Diagnosis.

    Lab values are emitted as `<label> = <value> <unit>` lines from the
    normalized record; missing fields are skipped.

    This is the model's view of the record. Span verification runs against this
    same string, so anything the model rewords fails to verify and the finding
    is downgraded.
    """
    parts: list[str] = []

    hpi = case.get("Patient History")
    if isinstance(hpi, str) and hpi.strip():
        parts.append("=== Patient History ===\n" + hpi.strip())

    pe = case.get("Physical Examination")
    if isinstance(pe, str) and pe.strip():
        parts.append("=== Physical Examination ===\n" + pe.strip())

    labs = case.get("Laboratory Tests")
    if isinstance(labs, dict) and labs:
        # Lab normalization: group labs by (fluid, category)
        # and emit one section per group, instead of a single flat
        # "=== Laboratory Tests ===" block. A single flat block let the model
        # confuse blood "White Blood Cells" (itemid 51301, fluid=Blood)
        # with urinalysis "WBC" (itemid 51516, fluid=Urine) because both
        # lines were structurally equivalent within the same section.
        # Splitting by fluid first, then category, makes cross-fluid
        # mistakes structurally impossible. The category subgrouping
        # additionally separates urinalysis dipstick "Bilirubin = NEG"
        # (Urine Labs Hematology) from blood Total Bilirubin (Blood Labs
        # Chemistry) which otherwise share the bare "Bilirubin" label.
        # Cross-references: data/lab_test_mapping.csv (Hager's
        # fluid/category columns from paulhager/MIMIC-CDM dataset);
        # scripts/audit_extraction_vs_record.py parse_lab_section now
        # recognizes the new headers via regex.
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for itemid, record in labs.items():
            if not isinstance(record, dict):
                continue
            label = (
                record.get("label")
                or record.get("test_name")
                or f"itemid={itemid}"
            )
            val = (
                record.get("numeric_value_fixed")
                or record.get("valuenum")
                or record.get("valuestr")
                or record.get("value")
            )
            # Lab normalization: also fall back to
            # `unit_suffix_fixed`, which is the column Hager's converted
            # records actually populate (paulhager/MIMIC-CDM dataset). The
            # legacy `valueuom`/`unit` lookups returned "" on every converted
            # record, so the record text printed the value with no unit at all.
            # That was the root cause
            # of the lab mismaps: the 8B could not tell a neutrophil
            # percentage from an absolute neutrophil count in K/uL, and read a
            # percentage as an ANC, which looks like life-threatening
            # neutropenia. With this fallback the unit is printed, which gives
            # the model the disambiguator.
            unit = (
                record.get("valueuom")
                or record.get("unit")
                or record.get("unit_suffix_fixed")
                or ""
            )
            # Reference range -- exposed so the LLM can fill NumericFinding.ref_upper
            # from record context (needed for Verifier numeric cross-check on AP_C2_ENZYMES).
            ref_lo = record.get("ref_lower_numeric")
            ref_hi = record.get("ref_upper_numeric")
            if val is None:
                continue
            line = f"{label} = {val} {unit}".strip()
            if ref_lo is not None and ref_hi is not None:
                line += f" (ref: {ref_lo}-{ref_hi})"
            elif ref_hi is not None:
                line += f" (ref: <={ref_hi})"
            fluid = record.get("fluid") or "Other"
            category = record.get("category") or "Other"
            groups[(fluid, category)].append(line)
        # Section ordering: Blood fluid first (largest, most diagnostically
        # relevant for our 4 abdominal-condition diseases), Urine second,
        # then other fluids alphabetically. Within fluid: Chemistry,
        # Hematology, Blood Gas, then others alphabetically. Within each
        # section: labs sorted alphabetically by full line (deterministic).
        _FLUID_PRIORITY = {"Blood": 0, "Urine": 1}
        _CATEGORY_PRIORITY = {"Chemistry": 0, "Hematology": 1, "Blood Gas": 2}
        sorted_groups = sorted(
            groups.keys(),
            key=lambda fc: (
                _FLUID_PRIORITY.get(fc[0], 2), fc[0],
                _CATEGORY_PRIORITY.get(fc[1], 3), fc[1],
            ),
        )
        for fluid, category in sorted_groups:
            section_lines = sorted(groups[(fluid, category)])
            parts.append(
                f"=== {fluid} Labs ({category}) ===\n"
                + "\n".join(section_lines)
            )

    radiology = case.get("Radiology")
    if isinstance(radiology, list) and radiology:
        # Hager-aligned radiology rendering.
        # Fixes silent radiology drop bug (key mismatch): the Adapter writes
        # each report dict with capital "Report" (see src/adapter.py
        # _get_radiology), but prior code here looked for lowercase
        # "report_text"/"text"/"report" only -- none of which the adapter
        # ever writes -- so radiology was silently absent from record_text
        # for every run: not one transcript in a 200-case 70B run contained a
        # Radiology section.
        # Three Hager-aligned changes:
        #   1. Read "Report" first (adapter's actual key); fall back to the
        #      legacy lowercase variants for backward-compat with stub tests.
        #   2. Filter to Region == "Abdomen" only -- mirrors Hager
        #      run_full_info.add_rad_reports which drops chest x-rays and
        #      other non-target-region scans that dilute LLM attention on
        #      our 4 abdominal conditions. Skipped when region info is
        #      absent (backward-compat with tests using simple report_text
        #      dicts that have no Region field).
        #   3. Prefix each report with "<Modality> <Region>" so the LLM
        #      knows which scan it is reading -- mirrors Hager's per-report
        #      header style.
        rad_lines: list[str] = []
        for r in radiology:
            if isinstance(r, dict):
                txt = (
                    r.get("Report")
                    or r.get("report_text")
                    or r.get("text")
                    or r.get("report")
                    or ""
                )
                region = r.get("Region") or r.get("region") or ""
                modality = r.get("Modality") or r.get("modality") or ""
                # Abdomen-only filter when region info is present
                if region and region != "Abdomen":
                    continue
                if txt:
                    header_bits = " ".join(b for b in (modality, region) if b).strip()
                    if header_bits:
                        rad_lines.append(f"=== {header_bits} ===\n{str(txt).strip()}")
                    else:
                        rad_lines.append(str(txt).strip())
            elif isinstance(r, str):
                rad_lines.append(r.strip())
        if rad_lines:
            parts.append("=== Radiology ===\n" + "\n\n".join(rad_lines))

    microbio = case.get("Microbiology")
    if isinstance(microbio, dict) and microbio:
        parts.append("=== Microbiology ===\n" + json.dumps(microbio, default=str))

    dx = case.get("Discharge Diagnosis")
    if isinstance(dx, str) and dx.strip():
        # Included for completeness, but the agent prompt instructs the model
        # not to draw extraction from it. It is visible to the model, so it is
        # labelled in the rendered text.
        parts.append("=== Discharge Diagnosis (do not use for extraction) ===\n" + dx.strip())

    return "\n\n".join(parts)
