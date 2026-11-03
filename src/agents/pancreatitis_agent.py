"""Pancreatitis agent.

A thin subclass of `DiseaseAgentBase`, which owns the whole iterative loop.
This module supplies the pancreatitis identity and the role prompt built
around the Revised Atlanta two-of-three rule.

`AGENT_VERSION` and `AgentResult` are re-exported so callers can import
them from here.
"""

from __future__ import annotations

from src.agents.disease_agent_base import (
    AGENT_VERSION,
    AgentResult,
    DiseaseAgentBase,
)

__all__ = ["AGENT_VERSION", "AgentResult", "PancreatitisAgent"]


# Constants (pancreatitis-specific prompt)

_AGENT_ROLE = """\
You are the Acute Pancreatitis Disease Agent. Your role is to gather
patient evidence iteratively and decide whether the evidence supports a
diagnosis of acute pancreatitis using the Revised Atlanta / WSES 2019
"2 of 3 criteria" diagnostic framework provided below.

You can request one of three tools per turn:
  - Physical Examination: returns the patient's complete physical exam.
    LEAVE the Action Input EMPTY (the tool ignores it).
  - Laboratory Tests: returns specifically requested labs. Action Input is
    a comma-separated list of lab or panel names (e.g., "lipase, amylase,
    LFTs").
  - Imaging: returns ONE (modality, region) report per call. Subsequent
    calls for the SAME modality+region return the NEXT chronological
    report (or "no further reports" if exhausted -- in that case switch
    to a different modality or commit). Action Input is free text naming
    the modality and region (e.g., "Abdominal CT", "Ultrasound of the
    abdomen"). Only abdominal imaging is available.

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

The Findings JSON is keyed by **CRITERION ID**. There are exactly THREE
criteria. Diagnosis of acute pancreatitis requires ANY 2 of the 3.
DO NOT invent new IDs. DO NOT use shape names ("PresentFinding",
"NumericFinding") as keys.

AP_C1_PAIN  -- Shape A (binary): characteristic acute-pancreatitis abdominal
pain (persistent, severe epigastric pain, often radiating to the back).
    "AP_C1_PAIN": {"present": true | false | null,
                   "evidence_span": "<verbatim record quote>",
                   "confidence": "explicit" | "inferred" | "uncertain"}

AP_C2_ENZYMES -- Shape B (numeric): serum lipase OR amylase. The diagnostic
bar is a value at least 3x the upper limit of normal (ULN).
    "AP_C2_ENZYMES": {"value": <number>, "unit": "<unit>",
                      "ref_upper": <ULN number>,
                      "evidence_span": "<quote>",
                      "confidence": "explicit" | "inferred" | "uncertain"}

  *** ENZYME CONVENTION (CRITICAL) ***
  - Prefer LIPASE over amylase when both are reported.
  - Emit BOTH the measured `value` AND the lab's `ref_upper` (ULN) in the
    SAME unit, exactly as documented. Do NOT convert units.
  - Do NOT pre-judge the 3x threshold yourself. Report the raw value and
    ref_upper; the system deterministically computes value >= 3 x ref_upper.

AP_C3_IMAGING -- Shape A (binary): characteristic imaging finding of acute
pancreatitis on contrast-enhanced CT, MRI, or transabdominal ultrasound
(e.g., pancreatic enlargement / edema, peripancreatic fat stranding,
peripancreatic fluid collection, or necrosis).
    "AP_C3_IMAGING": {"present": true | false | null,
                      "evidence_span": "<quote>",
                      "confidence": "explicit" | "inferred" | "uncertain"}

  NOTE: AP_C3_IMAGING is a TOP-LEVEL binary criterion. Do NOT nest it under
  an "imaging" block and do NOT break it into per-sign sub-findings.

================ WORKED EXAMPLE ================

After observing an HPI describing severe epigastric pain radiating to the
back and a lipase of 600 U/L (ULN 60), your Findings must look EXACTLY like
this (keys are the criterion IDs above):

  Findings: {
    "AP_C1_PAIN":    {"present": true, "evidence_span": "severe epigastric pain radiating to the back", "confidence": "explicit"},
    "AP_C2_ENZYMES": {"value": 600, "unit": "U/L", "ref_upper": 60, "evidence_span": "Lipase 600 U/L", "confidence": "explicit"}
  }

That is already 2 of 3 criteria (pain + enzymes >= 3x ULN) and is sufficient
to commit to acute pancreatitis.

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


# Agent class

class PancreatitisAgent(DiseaseAgentBase):
    """Disease-specific evidence-gathering agent for acute pancreatitis (Revised Atlanta / WSES 2019)."""

    agent_id = "pancreatitis"
    _kb_label = "Pancreatitis"

    def _agent_role_prompt(self) -> str:
        return _AGENT_ROLE

    def _extra_reminders(self) -> str:
        return ("[Reminders: (a) for AP_C2_ENZYMES emit BOTH value and ref_upper "
                "in the same unit -- do NOT convert and do NOT pre-judge the 3x ULN "
                "bar (the system computes value >= 3x ref_upper); (b) every "
                "evidence_span MUST be a verbatim substring of the record -- omit "
                "criteria you cannot ground in record text.]")

    def _extraction_schema_class(self):
        from src.schemas.pancreatitis import PancreatitisExtraction
        return PancreatitisExtraction
