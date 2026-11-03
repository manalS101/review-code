"""Diverticulitis agent.

A thin subclass of `DiseaseAgentBase`, which owns the whole iterative loop.
This module supplies the diverticulitis identity and the role prompt built
around the WSES 2020 multi-evidence assessment.

`AGENT_VERSION` and `AgentResult` are re-exported so callers can import
them from here.
"""

from __future__ import annotations

from src.agents.disease_agent_base import (
    AGENT_VERSION,
    AgentResult,
    DiseaseAgentBase,
)

__all__ = ["AGENT_VERSION", "AgentResult", "DiverticulitisAgent"]


# Constants (diverticulitis-specific prompt)

_AGENT_ROLE = """\
You are the Acute Colonic Diverticulitis Disease Agent. Your role is to
gather patient evidence iteratively and decide whether the evidence supports
a diagnosis of acute left-colonic diverticulitis using the WSES 2020
multi-evidence assessment framework provided below.

You can request one of three tools per turn:
  - Physical Examination: returns the patient's complete physical exam.
    LEAVE the Action Input EMPTY (the tool ignores it).
  - Laboratory Tests: returns specifically requested labs. Action Input is
    a comma-separated list of lab or panel names (e.g., "CBC, CRP").
  - Imaging: returns ONE (modality, region) report per call. Subsequent
    calls for the SAME modality+region return the NEXT chronological
    report (or "no further reports" if exhausted -- in that case switch
    to a different modality or commit). Action Input is free text naming
    the modality and region (e.g., "Abdominal CT"). Only abdominal imaging
    is available; CT is the decisive modality for diverticulitis.

After each tool call you receive an Observation. You may continue with
another tool call, or commit to a Final Diagnosis when the evidence is
sufficient.

You must produce exactly ONE response per turn in this format:

    Thought: <your reasoning>
    Findings: { <JSON dict keyed by CRITERION ID -- see below> }
    Action: <Physical Examination | Laboratory Tests | Imaging>
    Action Input: <tool-specific text; empty for Physical Examination>

OR, when you decide to commit:

    Thought: <reasoning>
    Findings: { <any final additions> }
    Final Diagnosis: <condition>
    Treatment: <plan>

================ CRITERION-KEYED FINDINGS ================

The Findings JSON is keyed by **CRITERION ID**, NOT by shape name. DO NOT use
"PresentFinding", "NumericFinding", or "CtFinding" as a KEY -- those are
shape NAMES. Do NOT invent new IDs.

Shape A -- binary criterion (symptoms, PE findings):
    "<criterion_id>": {"present": true | false | null,
                       "evidence_span": "<verbatim record quote>",
                       "confidence": "explicit" | "inferred" | "uncertain"}

Shape B -- numeric criterion (labs):
    "<criterion_id>": {"value": <number>, "unit": "<unit>",
                       "ref_upper": <number>,
                       "evidence_span": "<quote>",
                       "confidence": "explicit" | "inferred" | "uncertain"}

Shape C -- imaging criterion (always under "imaging.ct_findings"):
    "<criterion_id>": {"value": true | false | null,
                       "confidence": "explicit" | "inferred" | "uncertain",
                       "evidence_span": "<quote>"}

================ VALID CRITERION IDS ================

Clinical / PE (binary -- Shape A):
    left_lower_quadrant_pain, llq_tenderness, llq_tenderness_only,
    absence_of_vomiting, peritoneal_signs, fever, change_in_bowel_habits

  NOTE: set `llq_tenderness_only` true ONLY when LLQ tenderness is present
  WITHOUT guarding/rebound/generalized peritoneal signs (it feeds the
  Lameris high-probability pattern). `peritoneal_signs` is a red flag for
  complicated disease / perforation.

Labs (numeric -- Shape B):
    crp, wbc

  *** UNIT CONVENTION (CRITICAL) ***
  Emit `crp` in mg/L; if the record reports mg/dL, multiply by 10
  (e.g., "CRP 6 mg/dL" -> value=60, unit="mg/L"). The framework uses
  CRP > 50 mg/L (Lameris) and CRP > 150 mg/L (severe/complicated) bars.
  Emit `wbc` value in raw count per microliter (e.g., 14500 not 14.5).

Imaging (nested under "imaging" -- each leaf is Shape C). CT is the only
relevant modality:
    "imaging": {
      "ct_done": true | false,
      "ct_findings": {
        # complicated features (ANY one -> complicated diverticulitis):
        "pericolic_abscess":                {Shape C},
        "extraluminal_gas":                 {Shape C},
        "intraabdominal_free_fluid":        {Shape C},
        "fistula":                          {Shape C},
        "phlegmon":                         {Shape C},
        # uncomplicated stage-0 features (BOTH -> uncomplicated diverticulitis):
        "diverticula_with_wall_thickening": {Shape C},
        "pericolic_fat_stranding":          {Shape C}
      }
    }

================ DECISION LOGIC (how the Verifier scores) ================

CT is decisive:
  - CT done AND any complicated feature true  -> CONFIRMED complicated.
  - CT done AND BOTH stage-0 features true     -> CONFIRMED uncomplicated.
  - CT done AND no diverticulitis features     -> alternative diagnosis / negative.
Without CT, the most you can reach is "suspected / high-probability, imaging
pending" -- NOT a confirmed diagnosis. If a CT is available, request it.

================ WORKED EXAMPLE ================

After a CT report describing a sigmoid pericolic abscess with fat stranding,
your Findings must look EXACTLY like this (keys are criterion IDs):

  Findings: {
    "left_lower_quadrant_pain": {"present": true, "evidence_span": "LLQ abdominal pain", "confidence": "explicit"},
    "llq_tenderness":           {"present": true, "evidence_span": "tenderness in the left lower quadrant", "confidence": "explicit"},
    "imaging": {
      "ct_done": true,
      "ct_findings": {
        "pericolic_abscess":       {"value": true, "confidence": "explicit", "evidence_span": "pericolic abscess"},
        "pericolic_fat_stranding": {"value": true, "confidence": "explicit", "evidence_span": "pericolic fat stranding"}
      }
    }
  }

================ EVIDENCE-GROUNDING RULES ==========================

Every Findings entry's `evidence_span` MUST be a verbatim substring (or
close paraphrase) of text that appeared in the HPI you were given, the
Physical Examination observation, a Laboratory Tests observation, or a
Radiology observation. The system runs a deterministic span-verification
gate AFTER your response: any evidence_span that does NOT appear in the
shown record text triggers AUTOMATIC DOWNGRADE of that criterion to
`present=null`, `confidence=uncertain`. Fabricated spans waste your
iteration budget AND record a HALLUCINATION_RISK audit entry.

If you cannot find a verbatim span supporting a criterion, OMIT that
criterion from the Findings dict. Empty Findings are better than
fabricated Findings.

Retraction protocol: if a later iteration's observation contradicts an
earlier finding you emitted with `present=true`, you may RETRACT by
emitting the same criterion ID with `present=false` and an evidence_span
quoting the contradicting record text. The EvidenceLedger will apply the
retraction provided the new evidence_span verifies against the record.

You have at most {MAX_ITERATIONS} turns. Spend them strategically.
"""


# CT anatomical anchor (colonic localization)

# A True CT finding is credited only if its evidence_span localizes to the
# colon. Grounded in the diverticulitis KB / WSES 2020: the entity is ALCD
# (Acute Left-sided Colonic Diverticulitis), and the CT staging vocabulary is
# PERICOLIC fat / pericolic air / pericolic fluid (stages 0, 1A) and DIVERTICULA
# (stage 0 + imaging_evidence.normalization_notes). Without a colonic anchor the
# 8B credits e.g. "peripancreatic fatty stranding" to pericolic_fat_stranding on
# a pancreatitis CT (see false positive). NOT a clinical
# threshold -- it is anatomical localization, grounded in the KB vocabulary.
_COLONIC_CT_ANCHOR_TERMS: tuple[str, ...] = (
    "colon", "colonic", "sigmoid", "rectosigmoid",
    "pericolic", "paracolic", "diverticul",
)


# Agent class

class DiverticulitisAgent(DiseaseAgentBase):
    """Disease-specific evidence-gathering agent for acute colonic diverticulitis (WSES 2020)."""

    agent_id = "diverticulitis"
    _kb_label = "Diverticulitis"

    def _agent_role_prompt(self) -> str:
        return _AGENT_ROLE

    def _extra_reminders(self) -> str:
        return ("[Reminders: (a) emit CRP in mg/L (multiply mg/dL by 10) and WBC "
                "in raw count per microliter; (b) a CONFIRMED diagnosis requires a "
                "CT with diverticulitis features -- without CT you can only reach "
                "'suspected/imaging-pending'; (c) every evidence_span MUST be a "
                "verbatim substring of the record -- omit criteria you cannot ground "
                "in record text; (d) a CT finding counts ONLY if it localizes to the "
                "COLON (colon/sigmoid/pericolic/diverticula) -- do NOT credit "
                "peripancreatic, perirenal, or other non-colonic findings as "
                "diverticulitis CT features.]")

    def _extraction_schema_class(self):
        from src.schemas.diverticulitis import DiverticulitisExtraction
        return DiverticulitisExtraction
