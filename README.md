# Evidence Before Diagnosis

**A guideline-grounded multi-agent framework for acute abdominal conditions.**

A diagnostic framework for appendicitis, cholecystitis, diverticulitis and
pancreatitis, evaluated on the MIMIC-CDM dataset. It separates evidence
acquisition from diagnostic commitment: language-model agents gather and ground
the evidence, and a deterministic verifier decides.

The label comes from the verifier, not from the model. Each committed diagnosis
and each abstention traces to a specific guideline criterion and a quoted span
of the patient record.

This repository contains **no patient data**. See [docs/data.md](docs/data.md)
for how to obtain the dataset.

## How it works

**Phase 1, multi-agent evidence gathering.** Four disease-specific agents work
the same record independently, one per condition, without exchanging evidence.
Each requests the physical examination, specific laboratory tests and specific
imaging through three tools, up to ten iterations, and extracts structured
findings with the record text supporting them. An impression filter strips the
Impression, History, Conclusion and Comparison sections from imaging reports, so
extraction rests on reported findings rather than the radiologist's conclusion.

Two deterministic gates run before a finding reaches the evidence ledger. Both
downgrade rather than delete, so the trace keeps the claim and the reason it was
not trusted. **Span verification** requires the quoted `evidence_span` to occur
in the record the agent was shown, by exact substring or fuzzy match at 0.85; an
ungrounded finding cannot satisfy any criterion. **Anatomical anchoring**
retracts a finding whose supporting text names a conflicting territory without a
valid condition-specific anchor, since a span can be real yet attributed to the
wrong organ.

Each agent's verifier then evaluates the accumulated evidence against its
knowledge base:

| Condition | Framework | Source |
|---|---|---|
| Appendicitis | Tailored stepwise risk stratification | WSES 2020 |
| Cholecystitis | Integrated assessment | WSES ACC 2020 |
| Diverticulitis | Multi-evidence assessment | WSES 2020 |
| Pancreatitis | Two-of-three rule | Revised Atlanta |

**Phase 2, knowledge-based grounded diagnosis.** Disconfirmation challenges the
confirming findings of each agent that claimed its condition and re-runs that
agent's verifier unchanged; an agent that no longer confirms drops out. The
resolution cascade decides over the survivors: no candidate returns
*insufficient evidence*, one candidate is selected, and two or more go to an
ordered cascade of full over partial confirmation, then cross-condition
contradiction rules, then the number of independent supporting criteria. An
unresolved tie returns *insufficient evidence*. Phase 2 performs no
language-model inference.

Two scoring modes differ only on unresolved cases. **Mode A** keeps
*insufficient evidence*. **Mode B** applies a deterministic forced-choice
fallback, for comparison with non-abstaining baselines.

## Setup and run

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Obtain the MIMIC-CDM dataset into `data/` following
[docs/data.md](docs/data.md). Serve the model with vLLM, or any
OpenAI-compatible server, and give its address to `--base-url`. The prompts
contain record text from a credentialed dataset, so whichever endpoint you use
must be permitted by your own data use agreement.

```bash
python run_pipeline.py --cases full --model meta-llama/Meta-Llama-3.1-8B-Instruct --base-url http://localhost:8000/v1 --out-dir runs/first
```

The 8B configuration is `meta-llama/Meta-Llama-3.1-8B-Instruct` and the 70B configuration is
`meta-llama/Llama-3.3-70B-Instruct`; both are the instruction-tuned checkpoints, used
without fine-tuning or adaptation.

`--cases full` runs the whole dataset, `--cases subset` a seeded stratified
subset for a quick check, `--cases custom --hadm-ids 1,2,3` specific admissions.
`--resume` continues an interrupted run without repeating finished cases.

## Output

Each run writes `summary.csv` with one row per case, `run_manifest.json` with
everything needed to repeat it, a readable `<hadm>_<condition>.md` trace per
case, and under `artifacts/` the per-case evidence ledger, verifier output and
per-iteration transcripts.

`summary.csv` carries `mode_a`, `mode_b` and the disconfirmation ablation
`mode_a_no_disconfirmation`, each agent's own verdict, a cross-label
confirmation column, and the per-case model calls, tool calls and tokens.
Accuracy for any arm is the mean of its `*_correct` column.

## GPU and runtime

Per case the framework makes about 35 model calls and 190,000 tokens at 8B, and
about 30 calls and 155,000 tokens at 70B.

| Model | GPU | Measured, one request at a time |
|---|---|---|
| 8B | NVIDIA RTX 4090, 24 GB | about 3.5 minutes per case |
| 70B | NVIDIA A100, 80 GB | about 10 minutes per case |

Every call is made at temperature 0 and the four agents run sequentially.
Run-to-run agreement depends on server batching, since a changing batch
composition changes the floating-point reduction order and therefore the sampled
tokens. One request at a time gives identical labels across repeat runs; two
concurrent requests roughly halve the wall-clock and drop agreement to about
85%. The verifier and the resolution cascade are reproducible by construction.

## Layout

`run_pipeline.py` is the entry point. `knowledge_bases/` holds one guideline
knowledge base per condition. Under `src/`: `adapter.py` and `record.py` load
and render a record, `case_selection.py` picks the cases, `kb_loader.py`,
`verifier.py` and `label_mapper.py` are the deterministic decision path,
`evidence_ledger.py` records it, `multi_agent.py` and `resolution.py` are the
two phases, and `agents/` holds the per-condition agents, the three tools, the
two gates and disconfirmation.

## Limitations

Four acute abdominal conditions, single diagnosis, retrospective. Patients with
none of the target conditions are not tested. The primary discharge diagnosis is
the single reference label, so disagreement with it is not always clinical
error. The source files carry no timestamps, so criteria with a temporal
component cannot be enforced. The knowledge bases encode published adult
criteria and are not fitted to this dataset. This is research software and is
not a medical device.

## Citation

This is an anonymized copy prepared for peer review. Author and affiliation
details are withheld, and the citation for this work will be added on
publication.

The MIMIC-CDM dataset and the autonomous comparator framework are separate work
by Hager et al., and should be cited alongside it:

> Hager, P., Jungmann, F., Holland, R. et al. Evaluation and mitigation of the
> limitations of large language models in clinical decision-making. *Nature
> Medicine* (2024). https://doi.org/10.1038/s41591-024-03097-1

> Hager, P., Jungmann, F., & Rueckert, D. MIMIC-IV-Ext Clinical Decision Making
> (version 1.0). *PhysioNet* (2024). https://doi.org/10.13026/2pfq-5b68

## License

MIT, see [LICENSE](LICENSE). The knowledge bases are structured encodings of
criteria from published clinical guidelines, with per-criterion provenance to
the source. The guideline documents themselves are not redistributed here.
