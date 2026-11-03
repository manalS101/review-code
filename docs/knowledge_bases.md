# The knowledge bases

`knowledge_bases/` holds one JSON file per condition. Each is a structured
encoding of the diagnostic criteria from that condition's published guideline.
Two components read them: the verifier, which executes the decision framework,
and the disease-specific agents, which are shown a rendered summary of the
criteria so they know
what evidence is worth gathering.

They are the part of this work that is worth reading closely, because they are
what the diagnosis is actually made from. Nothing in them is fitted to the
benchmark.

## Sources

| Condition | Primary source | Secondary |
|---|---|---|
| Appendicitis | WSES Jerusalem guidelines 2020 | StatPearls |
| Cholecystitis | WSES acute calculous cholecystitis 2020 | StatPearls |
| Diverticulitis | WSES acute left-sided colonic diverticulitis 2020 | StatPearls |
| Pancreatitis | WSES severe acute pancreatitis 2019, Revised Atlanta classification | StatPearls |

Every knowledge base carries a `sources` block naming its documents with DOIs
and access terms, and a `source_audit` block recording which criteria came from
where. The guideline PDFs themselves are not redistributed here; the DOIs will
take you to them.

## Structure

The verifier requires these top-level sections and refuses to load a file that
is missing any of them:

| Section | What it holds |
|---|---|
| `kb_id`, `schema_version`, `disease` | identity |
| `sources` | the guideline documents, with DOI and license |
| `scientific_guardrails` | what this knowledge base does not claim to cover |
| `diagnostic_framework` | the framework type, the primary rule, and the valid outputs |
| `agent_verifier` | the decision algorithm and the abstention rules |
| `clinical_concepts` | history and examination criteria, with the phrasings that count |
| `laboratory_evidence` | laboratory criteria and their thresholds |
| `imaging_evidence` | imaging signs and the modalities that can establish them |
| `evidence_interpretation_layer` | how partial or conflicting evidence is read |
| `contradictory_and_uncertainty_layer` | findings that argue against this diagnosis |
| `clinical_nlp_context_layer` | negation and hedging handling for record text |
| `source_audit` | criterion-level provenance |

`diagnostic_framework.framework_type` selects the verifier implementation, and
must be one of `n_of_m_rule`, `multi_evidence_assessment`,
`integrated_assessment` or `tailored_stepwise_risk_stratification`. A mismatch
between the declared type and the expected one for that condition is a load
error.

## A worked example

The pancreatitis primary rule is the Revised Atlanta two-of-three:

```json
{
  "framework_type": "n_of_m_rule",
  "primary_rule": {
    "id": "AP_DX_REVISED_ATLANTA_WSES_N_OF_3",
    "rule_type": "n_of_m",
    "required_count": 2,
    "total_count": 3,
    "criteria": ["AP_C1_PAIN", "AP_C2_ENZYMES", "AP_C3_IMAGING"]
  }
}
```

Those criterion ids are what the agents report findings under, what the
verifier evaluates, and what appears in the step trace. Each has a definition
and a list of the record phrasings that satisfy it, so that "burning epigastric
pain radiating to the back" is recognised as `AP_C1_PAIN` without the model
being asked to make that judgement for itself.

## Outputs and the label space

Each knowledge base declares the outputs it can emit, such as
`confirmed_acute_pancreatitis` or `probable_ACC`. `src/label_mapper.py` maps
those to the five-outcome evaluation space and marks which count as abstention.

The mapping is strict. Severity grades, complication flags and treatment
recommendations are metadata, not diagnoses, and attempting to map one raises.
At construction the mapper checks itself against the loaded knowledge bases in
both directions, so an output added to a knowledge base without a mapping, or a
mapping left behind after an edit, fails at startup rather than mid-run.

## If you modify a knowledge base

Two rules keep the result meaningful.

**Every threshold and every criterion needs a citation.** Not a citation to this
benchmark, and not a value that improved a score. If a change cannot be traced
to a guideline statement, it is fitting to the evaluation set and the result
stops meaning anything. Arithmetic and unit identities are not thresholds and
need no citation.

**Adding an output means updating the mapping.** `LABEL_MAP` in
`src/label_mapper.py` and the knowledge base outputs must agree; the loader will
tell you immediately if they do not.
