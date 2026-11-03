"""Anatomical anchoring of localizing findings.

The span gate proves that a quoted span is real. It cannot prove that the span
belongs to the organ the model credited it to. Small models produce findings
that are span-real but misattributed: right-upper-quadrant tenderness credited
as right-lower, diverticular stranding credited as periappendiceal,
peripancreatic fluid credited as pericolic, unlocalized lower abdominal pain
credited as pancreatic. Those are what drive wrong-disease confirmations.

This gate runs once per iteration, before the ledger merge, for every agent,
over both clinical localizing findings and imaging findings. The rule is
deliberately conservative: a finding is retracted only when its evidence span
positively localizes to a conflicting region and carries no term from the
finding's own territory. A vague or unlocalized span is kept, because
misattribution cannot be demonstrated without a competing localization, and
retracting on vagueness alone would destroy correct findings on records that are
simply written loosely.

The territory vocabularies come from each condition's defining anatomy in the
source guidelines: right lower quadrant, McBurney and appendix for
appendicitis; epigastrium, back and pancreas for pancreatitis; left lower
quadrant, sigmoid and colon, including right-sided cecal disease, for
diverticulitis; right upper quadrant and gallbladder for cholecystitis. They
are not derived from the benchmark.

    LOCALIZING_TERRITORY
    anchor_localizing_findings(findings_delta, territory) -> (anchored, audit)
"""

from __future__ import annotations

import copy
from typing import Any


STATUS_ANCHORED = "anchored"
STATUS_RETRACTED = "RETRACTED_WRONG_REGION"


# Region / organ term groups (lowercase substrings)

_RLQ = ("rlq", "right lower", "right iliac", "mcburney")
_RUQ = ("ruq", "right upper", "right hypochond", "subcostal")
_LLQ = ("llq", "left lower", "left iliac")
_LUQ = ("luq", "left upper")
_EPIGASTRIC = ("epigastr", "upper abdom", "periumbilical", "back", "band-like", "band like")
_LOWER = ("lower abdom",)
_PELVIC = ("pelvic", "suprapubic")
_COLON = ("colon", "colonic", "sigmoid", "rectosig", "pericolic", "paracolic",
          "diverticul", "descending colon", "cecal", "cecum")
_PANCREAS = ("pancrea", "peripancreatic")
_APPENDIX = ("appendix", "periappendice", "appendice")
# Gallbladder/biliary territory. Includes GALLSTONE + BILIARY vocabulary so a terse
# stone span ("cholelithiasis") is recognized as biliary -- otherwise it carries no
# conflicting term and a misattributed appendix US finding survives (one 70B run
# emitted us_appendicolith with span "cholelithiasis", a gallstone, and the
# anchor could not retract it because "cholelithiasis" matched none of the prior
# gallbladder terms).
_GALLBLADDER = ("gallbladder", "gall bladder", "cholecyst", "pericholecystic",
                "cholelith", "gallstone", "biliary", "common bile duct")


def _spec(
    required: tuple[str, ...],
    *conflicting_groups: tuple[str, ...],
    override: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    """Build an anchoring spec.

    required:    span must carry one of these to be kept (the conservative guard).
    conflicting: a span localizes to a wrong region IFF it carries one of these
                 AND none of `required` (masked by the has_required guard).
    override:    hard conflicting terms that retract even when a `required`
                 term is also present. Used at the appendix to right-colon seam:
                 a span whose own locus is the appendiceal tip must not be
                 credited as a colonic finding merely because it also mentions
                 the pericolic gutter, which is a peritoneal recess.
    """
    conflicting: list[str] = []
    for g in conflicting_groups:
        conflicting.extend(g)
    return {"required": required, "conflicting": tuple(conflicting),
            "override": tuple(override)}


# Per-disease territory table
#
# Shape per disease:
#   "clinical": { flat_present_key: {required, conflicting} }
#   "imaging_ct_findings": {required, conflicting}   # applied to every True
#                                                      imaging.ct_findings leaf
LOCALIZING_TERRITORY: dict[str, dict[str, Any]] = {
    "appendicitis": {
        "clinical": {
            k: _spec(_RLQ, _RUQ, _LLQ, _LUQ, ("epigastr", "upper abdom"))
            for k in ("pain_migrates_to_rlq", "rlq_tenderness",
                      "localized_guarding_rlq", "rebound_tenderness_mcburney")
        },
        "imaging_ct_findings": _spec(
            _APPENDIX, _COLON, _GALLBLADDER, _PANCREAS,
        ),
        # US stream: valid territory = appendix; a us_* finding localizing to the
        # gallbladder/colon/pancreas is a misattribution.
        "imaging_us_findings": _spec(
            _APPENDIX, _GALLBLADDER, _COLON, _PANCREAS,
        ),
    },
    "pancreatitis": {
        "clinical": {
            # AP_C1_PAIN: characteristic epigastric/upper/back pain.
            "AP_C1_PAIN":    _spec(_EPIGASTRIC, _RLQ, _LLQ, _LOWER, _PELVIC),
            # AP_C3_IMAGING is a flat present-finding -> treat as clinical-shaped;
            # valid territory is the pancreas.
            "AP_C3_IMAGING": _spec(_PANCREAS, _COLON, _APPENDIX, _GALLBLADDER,
                                   ("renal", "ureter")),
        },
        # No nested imaging.ct_findings in the pancreatitis schema.
    },
    "diverticulitis": {
        "clinical": {
            k: _spec(_LLQ + ("sigmoid", "descending colon"),
                     _RLQ, _RUQ, _LUQ, ("epigastr", "upper abdom"))
            for k in ("left_lower_quadrant_pain", "llq_tenderness", "llq_tenderness_only")
        },
        # Mirror the clinical conflicting REGION/QUADRANT terms
        # into the imaging spec. Previously imaging conflicting carried only ORGAN
        # names, so a CT span localizing to a quadrant WITHOUT naming an organ --
        # a span naming inflammatory fat stranding in the right upper quadrant,
        # on a true cholecystitis, slipped through and credited
        # pericolic_fat_stranding.
        # The conservative `has_required` guard still keeps any span carrying a
        # colonic anchor (colon/sigmoid/cecal/pericolic/diverticul...), so a
        # legitimately right-sided (cecal) diverticulitis finding is unaffected.
        # An explicit appendiceal locus is an OVERRIDE conflict
        # it retracts a colonic finding even though "pericolic" satisfies has_required.
        # Closes a perforated appendicitis case where the 8B credited
        # pericolic_abscess and pericolic_fat_stranding from a span describing
        # change around the appendiceal tip extending into the right pericolic
        # gutter, which is a peritoneal recess rather than the colon wall.
        # True cecal/sigmoid diverticulitis finding-spans name the colon, not the
        # appendiceal tip, so they are unaffected (confounder-verified).
        "imaging_ct_findings": _spec(
            _COLON, _PANCREAS, _GALLBLADDER, _APPENDIX, ("renal",),
            _RUQ, _RLQ, _LUQ, ("epigastr", "upper abdom"),
            override=_APPENDIX,
        ),
    },
    "cholecystitis": {
        "clinical": {
            "ruq_pain_or_tenderness": _spec(_RUQ, _LLQ, _RLQ, _LUQ, _LOWER, _PELVIC),
            # Right-shoulder/midback referral: a LEFT shoulder or lower-quadrant
            # span is a misattribution.
            "radiation_to_right_shoulder_or_midback": _spec(
                ("right shoulder", "right scapul", "midback", "mid back",
                 "interscapular", "subscapular"),
                ("left shoulder", "rlq", "right lower", "llq", "left lower"),
            ),
        },
        "imaging_ct_findings": _spec(
            _GALLBLADDER, _COLON, _PANCREAS, _APPENDIX,
        ),
        # US stream: valid territory = gallbladder; a us_* finding localizing to the
        # appendix/colon/pancreas is a misattribution (symmetric to the CT spec).
        "imaging_us_findings": _spec(
            _GALLBLADDER, _APPENDIX, _COLON, _PANCREAS,
        ),
    },
}


# Core

def _is_present_leaf(v: Any) -> bool:
    return isinstance(v, dict) and "present" in v and "value" not in v


def _is_ct_leaf(v: Any) -> bool:
    return (
        isinstance(v, dict) and "value" in v
        and (isinstance(v.get("value"), bool) or v.get("value") is None)
        and "present" not in v
    )


def _conflicting_wrong_region(span: str, spec: dict[str, tuple[str, ...]]) -> bool:
    """True iff the span positively localizes to a CONFLICTING region/organ AND
    carries NO term from the finding's valid territory (conservative rule)."""
    s = (span or "").lower()
    if not s:
        return False
    # Override terms are a HARD conflict -- they retract even when a required
    # term is also present (the appendix<->right-colon seam: an explicit
    # appendiceal locus overrides a "pericolic gutter" keyword).
    if any(t in s for t in spec.get("override", ())):
        return True
    has_required = any(t in s for t in spec["required"])
    if has_required:
        return False
    return any(t in s for t in spec["conflicting"])


def anchor_localizing_findings(
    findings_delta: dict[str, Any],
    territory: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retract localizing findings whose evidence_span localizes to the wrong
    region/organ. The input is NOT mutated; a deep copy is returned. Audit
    entries mirror the span_verifier shape.
    """
    audit: list[dict[str, Any]] = []
    if not isinstance(findings_delta, dict) or not territory:
        return findings_delta, audit

    anchored = copy.deepcopy(findings_delta)

    # ---- Clinical (flat present-keyed) findings ----
    for key, spec in (territory.get("clinical") or {}).items():
        leaf = anchored.get(key)
        if not _is_present_leaf(leaf) or leaf.get("present") is not True:
            continue
        span = leaf.get("evidence_span") or ""
        if _conflicting_wrong_region(span, spec):
            leaf["present"] = None
            leaf["confidence"] = "uncertain"
            leaf["span_verified"] = False
            status = STATUS_RETRACTED
        else:
            status = STATUS_ANCHORED
        audit.append({"criterion_path": key, "evidence_span": span,
                      "status": status, "downgraded": status == STATUS_RETRACTED,
                      "shape": "present"})

    # ---- Imaging: nested imaging.<stream>.<finding> leaves ----
    # Walk every imaging sub-stream that has a territory spec. Originally CT-only;
    # the ULTRASOUND stream is anchored too (us_findings), because the 8B mis-maps
    # US findings across organs just like CT -- on one true cholecystitis it
    # credited `us_appendicolith` from "grayscale images of the GALLBLADDER
    # demonstrate cholelithiasis", confirming appendicitis on a gallbladder US.
    imaging = anchored.get("imaging")
    if isinstance(imaging, dict):
        for stream_key, spec_key in (
            ("ct_findings", "imaging_ct_findings"),
            ("us_findings", "imaging_us_findings"),
        ):
            spec = territory.get(spec_key)
            if not spec:
                continue
            findings = imaging.get(stream_key)
            if not isinstance(findings, dict):
                continue
            for key, leaf in findings.items():
                if not _is_ct_leaf(leaf) or leaf.get("value") is not True:
                    continue
                span = leaf.get("evidence_span") or ""
                if _conflicting_wrong_region(span, spec):
                    leaf["value"] = None
                    leaf["confidence"] = "uncertain"
                    leaf["span_verified"] = False
                    status = STATUS_RETRACTED
                else:
                    status = STATUS_ANCHORED
                audit.append({"criterion_path": f"imaging.{stream_key}.{key}",
                              "evidence_span": span, "status": status,
                              "downgraded": status == STATUS_RETRACTED, "shape": "ct"})

    return anchored, audit
