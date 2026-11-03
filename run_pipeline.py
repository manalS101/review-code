r"""Run the framework over a set of cases.

Phase 1 runs the four disease-specific agents against a vLLM endpoint. Phase 2
applies disconfirmation and the resolution cascade. Both scoring modes are
recorded per case, along with the disconfirmation ablation:

    mode_a                      abstaining: unresolved cases stay
                                insufficient_evidence
    mode_b                      forced choice over the same evidence
    mode_a_no_disconfirmation   Mode A with disconfirmation disabled

Outputs land under --out-dir:

    summary.csv                    one row per case, machine readable
    run_manifest.json              everything needed to repeat the run
    <hadm>_<condition>.md          a readable trace of how the case was decided
    artifacts/<hadm>_<condition>/  evidence ledgers and per-iteration transcripts

Example:

    python run_pipeline.py --cases subset \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --base-url http://localhost:8000/v1 \
        --out-dir runs/subset50
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapter import Adapter
from src.agents.disconfirmation import Disconfirmer
from src.resolution import Resolver
from src.case_selection import DISEASE_ORDER, SUBSET_SEED, select_cases
from src.kb_loader import KBRegistry
from src.llm_client import VLLMClient
from src.multi_agent import MultiAgentRunner

SUMMARY_FIELDS = [
    "hadm_id", "reference_label",
    "mode_a_no_disconfirmation", "mode_a", "mode_b",
    "mode_a_no_disconfirmation_correct", "mode_a_correct", "mode_b_correct",
    # Each agent's own verdict, which is what cross-label confirmation reads.
    "app_label", "chol_label", "div_label", "panc_label", "cross_label_confirmation",
    # Cost, summed over the four agents.
    "n_llm_calls", "n_tool_calls", "latency_ms", "case_seconds",
    "parser_retries", "n_agent_errors",
    "prompt_tokens", "completion_tokens",
    "termination_reasons", "candidates_before", "candidates_after",
]


def write_text(path: Path, text: str) -> None:
    """Write to a temporary file and rename, so a kill mid-write cannot
    truncate a result that is already on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def trim(text: str | None, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def case_report(hadm, reference_label, phase1, audits, phase1_disc,
                mode_a_no_disc, mode_a) -> str:
    """A readable account of one case: what each agent concluded, what
    disconfirmation removed, and how resolution landed."""
    out = [f"# Case {hadm} (reference diagnosis: {reference_label})", "",
           f"- Mode A: **{mode_a.final_label}**",
           f"- Mode A without disconfirmation: **{mode_a_no_disc.final_label}**",
           f"- Decision path: `{mode_a.decision_path}`", "",
           "## Phase 1: agents", ""]
    for disease in DISEASE_ORDER:
        r = phase1.results[disease]
        vr = r.final_verifier_result or {}
        out += [f"### {disease}", "",
                f"- Verifier output: `{vr.get('diagnostic_output', 'n/a')}`",
                f"- Mapped label: `{r.outcome_label}`",
                f"- Ended on `{r.termination_reason}` after {r.total_llm_calls} "
                f"model calls and {r.total_tool_calls} tool calls", "",
                "Final diagnosis text from the model:", "",
                f"> {trim(r.final_diagnosis_text)}", ""]

    removed = [d for d in phase1.candidates if d not in phase1_disc.candidates]
    out += ["## Phase 2: disconfirmation", "",
            f"- Candidates before: {', '.join(phase1.candidates) or 'none'}",
            f"- Candidates after: {', '.join(phase1_disc.candidates) or 'none'}",
            f"- Eliminated: {', '.join(removed) or 'none'}", ""]
    for a in audits or []:
        out.append(f"- {a.disease}: `{a.label_before}` -> `{a.label_after}`"
                   f"{' (eliminated)' if a.eliminated else ''}")
        for r in a.retractions:
            out.append(f"  - retracted `{r.get('finding_key', '?')}` "
                       f"from span {r.get('evidence_span', '')!r}")
        for f in a.injected_features:
            out.append(f"  - contradictory feature `{f.get('feature', '?')}` "
                       f"from span {f.get('evidence_span', '')!r}")

    out += ["", "## Resolution cascade", "",
            f"- Candidates: {', '.join(mode_a.candidates) or 'none'}",
            f"- Winner: {mode_a.winner or 'none'}",
            f"- Convictions: `{json.dumps(mode_a.convictions)}`",
            f"- Final label: **{mode_a.final_label}**"]
    return "\n".join(out) + "\n"


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_FIELDS)
    w.writeheader()
    w.writerows(rows)
    write_text(out_dir / "summary.csv", buf.getvalue())


def load_completed(out_dir: Path) -> dict[int, dict]:
    """Finished rows from an earlier run of the same command, so --resume can
    carry them forward instead of paying for them again."""
    path = out_dir / "summary.csv"
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode_a") in ("", "ERROR", None):
                continue
            for key in ("mode_a_no_disconfirmation_correct", "mode_a_correct", "mode_b_correct"):
                row[key] = str(row.get(key, "")).strip().lower() == "true"
            done[int(row["hadm_id"])] = row
    return done


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="subset",
                    choices=("subset", "full", "custom"),
                    help="which cases to run: the complete dataset, a seeded "
                         "stratified subset, or an explicit list")
    ap.add_argument("--hadm-ids", help="comma-separated ids, use with --cases custom")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="must match the vLLM --served-model-name")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out-dir", default="runs/latest")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--kb-dir", default="knowledge_bases")
    ap.add_argument("--lab-mapping-csv", default="data/lab_test_mapping.csv")
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--limit", type=int, help="stop after the first N cases")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already finished in <out-dir>/summary.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    print(f"[info] checking {args.base_url}")
    client = VLLMClient(model_id=args.model, base_url=args.base_url)
    client.health_check()

    registry = KBRegistry(Path(args.kb_dir))
    adapter = Adapter(Path(args.data_dir))
    resolver = Resolver(registry)

    hadm_ids = [int(x) for x in args.hadm_ids.split(",")] if args.hadm_ids else None
    pairs = select_cases(adapter, cases=args.cases, hadm_ids=hadm_ids)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[info] cases={args.cases} cases={len(pairs)} -> {out_dir}")

    done = load_completed(out_dir) if args.resume else {}
    if done:
        print(f"[info] resuming, {len(done)} case(s) already finished")

    started_at, run_t0 = now_iso(), time.monotonic()
    rows: list[dict] = []

    for i, pair in enumerate(pairs, 1):
        hadm, reference_label = pair.hadm_id, pair.reference_label
        if hadm in done:
            rows.append(done[hadm])
            print(f"[{i}/{len(pairs)}] {hadm} -- already done")
            continue

        print(f"[{i}/{len(pairs)}] {hadm} (reference: {reference_label})", flush=True)
        case_t0 = time.monotonic()
        try:
            runner = MultiAgentRunner(
                llm_client=client, kb_registry=registry, adapter=adapter,
                max_iterations=args.max_iterations,
                log_dir=out_dir / "artifacts",
                lab_mapping_csv=Path(args.lab_mapping_csv),
            )
            phase1 = runner.run(hadm)
            adv = phase1.results

            disc = Disconfirmer(registry, adapter, llm_client=client)
            phase1_disc, audits = disc.disconfirm(phase1)
            mode_a_no_disc = resolver.resolve(phase1)
            mode_a = resolver.resolve(phase1_disc)
            mode_b = resolver.resolve(phase1_disc, forced_choice=True)

            write_text(out_dir / f"{hadm}_{reference_label}.md",
                       case_report(hadm, reference_label, phase1, audits, phase1_disc,
                                   mode_a_no_disc, mode_a))

            labels = {d: adv[d].outcome_label for d in DISEASE_ORDER}
            rows.append({
                "hadm_id": hadm, "reference_label": reference_label,
                "mode_a_no_disconfirmation": mode_a_no_disc.final_label,
                "mode_a": mode_a.final_label,
                "mode_b": mode_b.final_label,
                "mode_a_no_disconfirmation_correct": mode_a_no_disc.final_label == reference_label,
                "mode_a_correct": mode_a.final_label == reference_label,
                "mode_b_correct": mode_b.final_label == reference_label,
                "app_label": labels["appendicitis"],
                "chol_label": labels["cholecystitis"],
                "div_label": labels["diverticulitis"],
                "panc_label": labels["pancreatitis"],
                # An agent claiming a disease the patient does not have.
                "cross_label_confirmation": "|".join(d for d in DISEASE_ORDER
                                          if d != reference_label and labels[d] == d),
                "n_llm_calls": sum(adv[d].total_llm_calls for d in DISEASE_ORDER),
                "n_tool_calls": sum(adv[d].total_tool_calls for d in DISEASE_ORDER),
                "latency_ms": round(sum(adv[d].total_latency_ms
                                        for d in DISEASE_ORDER), 1),
                "case_seconds": round(time.monotonic() - case_t0, 1),
                "parser_retries": sum(adv[d].parser_retries_consumed
                                      for d in DISEASE_ORDER),
                "n_agent_errors": sum(
                    1 for d in DISEASE_ORDER
                    if adv[d].termination_reason in ("llm_error", "parser_error")),
                "prompt_tokens": sum(adv[d].total_prompt_tokens
                                     for d in DISEASE_ORDER),
                "completion_tokens": sum(adv[d].total_completion_tokens
                                         for d in DISEASE_ORDER),
                "termination_reasons": "|".join(adv[d].termination_reason or ""
                                                for d in DISEASE_ORDER),
                "candidates_before": "|".join(phase1.candidates),
                "candidates_after": "|".join(phase1_disc.candidates),
            })
        except Exception as e:  # one bad case must not abort the run
            print(f"    error: {type(e).__name__}: {e}", flush=True)
            row = {f: "" for f in SUMMARY_FIELDS}
            row.update({"hadm_id": hadm, "reference_label": reference_label,
                        "mode_a_no_disconfirmation": "ERROR", "mode_a": "ERROR", "mode_b": "ERROR",
                        "mode_a_no_disconfirmation_correct": False, "mode_a_correct": False,
                        "mode_b_correct": False,
                        "case_seconds": round(time.monotonic() - case_t0, 1)})
            rows.append(row)

        write_summary(out_dir, rows)
        print(f"    {rows[-1].get('case_seconds', '?')}s", flush=True)

    write_summary(out_dir, rows)
    write_text(out_dir / "run_manifest.json", json.dumps({
        "cases": args.cases,
        "subset_seed": SUBSET_SEED,
        "n_cases": len(pairs),
        "hadm_ids": [p.hadm_id for p in pairs],
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0.0,
        "max_iterations": args.max_iterations,
        "git_sha": git_sha(),
        "started_at_utc": started_at,
        "ended_at_utc": now_iso(),
        "duration_seconds": round(time.monotonic() - run_t0, 1),
    }, indent=2))

    n = len(rows)
    print(f"\nMode A {sum(1 for r in rows if r['mode_a_correct'])}/{n}   "
          f"Mode B {sum(1 for r in rows if r['mode_b_correct'])}/{n}   "
          f"(without disconfirmation "
          f"{sum(1 for r in rows if r['mode_a_no_disconfirmation_correct'])}/{n})")
    print(f"Results in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
