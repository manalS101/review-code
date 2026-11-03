"""Append-only evidence ledger.

One ledger per agent per case. It records every iteration in full (thought,
action, findings, observation, verifier verdict, transcript metadata, parse
warnings) and maintains the merged evidence dict the verifier reads between
iterations.

Two invariants matter. Records are append-only, so a later iteration can
supersede an earlier finding but never erase the fact that it was claimed. And
the ledger is flushed to disk by atomic rename after every iteration, so a run
killed mid-case leaves a readable partial trace rather than a truncated file.

When two iterations report the same finding, the merge keeps the
higher-confidence one and breaks ties in favour of the later observation.

    EvidenceLedger(agent_id, hadm_id, kb_tag).append_iteration(record)
    EvidenceLedger(...).current_evidence()          # merged dict for Verifier.verify()
    EvidenceLedger(...).attach_final_diagnosis(...)
    EvidenceLedger.load(path)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

# Heavy schema imports happen lazily inside EvidenceLedger.current_evidence() so that
# importing this module does NOT require pydantic / kb_loader if a caller only
# needs append/load mechanics. Tests benefit from the lighter import surface.


LEDGER_FILE_NAME = "ledger.json"
LEDGER_VERSION = "1.0"


# IterationRecord (per design D1)

@dataclass
class IterationRecord:
    """One row in the append-only Evidence EvidenceLedger.

    All fields are JSON-serializable. The Agent constructs one of these
    per iteration after the (LLM call -> parse -> tool dispatch -> Verifier)
    chain completes; the EvidenceLedger then appends + persists.

    Fields are deliberately verbose: this is the canonical audit surface.
    Reducing fields here means losing data for post-hoc investigation.
    """

    iteration: int                              # 1-indexed
    timestamp_utc: str                          # ISO 8601, e.g. "2026-05-26T14:30:21Z"

    # The four parsed blocks (per DiagnosisParser output) ----
    thought: str                                # verbatim Thought: text
    findings_delta: dict[str, Any]              # criterion-keyed; merged via D3
    action: str | None                          # PE / Labs / Imaging / None (Final Diag)
    action_input: str | None                    # raw tool input string (None for PE/terminal)
    observation: str | None                     # tool output string (None for terminal iter)

    # Verifier snapshot AFTER this iteration's evidence merged in ----
    verifier_result: dict[str, Any]             # serialized VerifierResult; see _serialize_verifier_result

    # LLM call audit ----
    llm_request: dict[str, Any]                 # {system_prompt_hash, user_prompt_len, max_tokens, temperature, json_only}
    llm_response_meta: dict[str, Any]           # {model_id, finish_reason, prompt_tokens, completion_tokens, latency_ms}
    raw_response_text: str                      # full LLM output BEFORE parsing

    # Parser audit ----
    parse_errors: list[str] = field(default_factory=list)   # non-fatal warnings from parser

    # Span-verification audit ( F1) -- populated by SpanVerifier in
    # the Agent after the parser yields findings_delta but BEFORE the
    # ledger merge. Each entry: {criterion_path, evidence_span, status,
    # downgraded, shape}. See src/agents/span_verifier.py for status codes.
    span_audit: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IterationRecord":
        # Tolerate extra keys (future-proofing) while requiring the documented ones.
        kwargs = {k: d[k] for k in (
            "iteration", "timestamp_utc", "thought", "findings_delta",
            "action", "action_input", "observation", "verifier_result",
            "llm_request", "llm_response_meta", "raw_response_text",
        )}
        kwargs["parse_errors"] = list(d.get("parse_errors", []))
        kwargs["span_audit"] = list(d.get("span_audit", []))
        return cls(**kwargs)


# Errors

class LedgerError(RuntimeError):
    """Base class for EvidenceLedger violations."""


class AppendOnlyViolation(LedgerError):
    """Raised on attempts to mutate past iterations or skip iteration numbers."""


# EvidenceLedger

class EvidenceLedger:
    """Per-Agent-per-case append-only Evidence EvidenceLedger.

    Constructor params:
      agent_id  -- "appendicitis" / "pancreatitis" / "cholecystitis" / "diverticulitis"
      hadm_id      -- MIMIC-CDM admission id
      kb_tag       -- KB version string ("kb-v4-frozen" today)
      log_dir      -- if set, ledger.json is persisted here after every iteration.
                      None = in-memory only (tests).
      schema_class -- Pydantic schema for current_evidence() validation.
                      Defaults to AppendicitisExtraction (Phase 2). Pass a
                      different class for future Agents.

    Append-only invariant:
      Once append_iteration(rec) is called for iteration N, rec.iteration must
      equal len(self.iterations) before the call + 1. Out-of-order or duplicate
      iteration numbers raise AppendOnlyViolation.
    """

    # Class-level constant: confidence ranking.
    _CONFIDENCE_RANK: ClassVar[dict[str, int]] = {
        "explicit":  3,
        "inferred":  2,
        "uncertain": 1,
    }

    def __init__(
        self,
        *,
        agent_id: str,
        hadm_id: int,
        kb_tag: str,
        log_dir: Path | str | None = None,
        schema_class: Any | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.hadm_id = int(hadm_id)
        self.kb_tag = kb_tag
        self.iterations: list[IterationRecord] = []
        self.log_dir: Path | None = Path(log_dir) if log_dir is not None else None

        # Terminal state (set by attach_final_diagnosis)
        self.final_diagnosis_text: str | None = None
        self.final_verifier_result: dict[str, Any] | None = None
        self.outcome_label: str | None = None
        self.termination_reason: str | None = None

        # Lazy-resolved Pydantic schema class. Default AppendicitisExtraction for Phase 2.
        self._schema_class = schema_class
        if self._schema_class is None:
            from src.schemas.appendicitis import AppendicitisExtraction
            self._schema_class = AppendicitisExtraction

        self.created_at_utc = _now_iso()

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- Append-only mutation API ----

    def append_iteration(self, rec: IterationRecord) -> None:
        """Append a new iteration. Enforces append-only invariant.

        Also persists ledger.json atomically when log_dir is set. Only
        ledger.json: the per-iteration transcript files are written by the
        agent itself.
        """
        expected = len(self.iterations) + 1
        if rec.iteration != expected:
            raise AppendOnlyViolation(
                f"iteration number out of order: got {rec.iteration}, "
                f"expected {expected} (ledger currently has {len(self.iterations)} iters)"
            )
        if self.final_diagnosis_text is not None:
            raise AppendOnlyViolation(
                f"cannot append iteration {rec.iteration} after final diagnosis already attached"
            )
        self.iterations.append(rec)
        self._persist_if_configured()

    def attach_final_diagnosis(
        self,
        *,
        final_diagnosis_text: str,
        verifier_result: dict[str, Any],
        outcome_label: str,
        termination_reason: str,
    ) -> None:
        """Mark the trajectory terminated and persist final-state metadata.

        Keeps both the model's free-text final diagnosis and the
        deterministic Verifier's diagnostic_output. Disagreements are surfaced
        downstream (the Agent sets termination_reason="agent_finish_verifier_disagrees"
        when they conflict).
        """
        if self.final_diagnosis_text is not None:
            raise AppendOnlyViolation(
                "final diagnosis already attached; cannot overwrite"
            )
        self.final_diagnosis_text = final_diagnosis_text
        self.final_verifier_result = verifier_result
        self.outcome_label = outcome_label
        self.termination_reason = termination_reason
        self._persist_if_configured()

    # ---- Evidence-merge API (drives Verifier) ----

    def current_evidence(self) -> dict[str, Any]:
        """Return the merged evidence dict in Verifier-input shape.

        Merge policy: confidence-ranked at the leaf level (explicit beats
        inferred beats uncertain), tie-break = latest iteration wins. Plain-value
        leaves (booleans like imaging.ct_done) follow latest-wins. Non-leaf
        dicts (imaging, scores, ct_findings) are deep-merged recursively.

        The merged raw dict is then validated through the per-Agent Pydantic
        schema (defaulting to AppendicitisExtraction); the schema's lenient
        coercion handles malformed LLM output gracefully (see schemas/_common.py),
        and .to_verifier_evidence() produces the exact shape Verifier.verify()
        accepts.

        Returns an empty (default-filled) evidence dict when no iterations have
        been appended.
        """
        merged: dict[str, Any] = {}
        for rec in self.iterations:
            if not rec.findings_delta:
                continue
            _deep_merge_with_confidence(merged, rec.findings_delta)

        # Inject required provenance so AppendicitisExtraction (and siblings)
        # can validate. The actual model_id varies per iteration and is
        # captured in IterationRecord.llm_response_meta; we use a coarse tag
        # here just to satisfy the schema's CaseProvenance.
        merged.setdefault("provenance", {
            "hadm_id":             self.hadm_id,
            "extraction_model":    "evidence-gathering-agent",
            "kb_tag":              self.kb_tag,
            "extraction_timestamp": _now_iso(),
        })

        # Validate via Pydantic schema -> emit Verifier-input dict.
        # If validation produces a hard error (very rare given _common.py's
        # lenient coercion), surface it as a LedgerError so the Agent can
        # log + terminate cleanly.
        try:
            validated = self._schema_class.model_validate(merged)
        except Exception as e:  # noqa: BLE001 -- pydantic's ValidationError shape varies
            raise LedgerError(
                f"current_evidence() failed schema validation against "
                f"{self._schema_class.__name__}: {type(e).__name__}: {e}"
            ) from e

        return validated.to_verifier_evidence()

    def current_evidence_with_provisional(self, rec: "IterationRecord") -> dict[str, Any]:
        """Return merged evidence as if ``rec`` had already been appended.

        Used by the Agent's per-iteration Verifier pass, which needs the merge
        to include the iteration about to be appended (otherwise the Verifier sees
        stale evidence and the early-commit nudge can never fire on the FIRST
        confirming iteration). Implementation: temporarily append ``rec``, compute
        evidence, then pop it back off -- purely in-memory (the iterations list is
        restored BEFORE any persist path could fire, so the on-disk ledger is never
        touched).
        """
        self.iterations.append(rec)
        try:
            return self.current_evidence()
        finally:
            self.iterations.pop()

    # ---- Early-commit nudge ----

    # The nudge fires on any non-abstaining verifier verdict, not only on a
    # confirmation. Cases were observed stalling at "intermediate"
    # (branch H or I) for six or seven iterations with no
    # nudge -- the LLM had no signal that the verdict had stabilized.
    # Non-abstaining = abstention_triggered is False.
    def has_pending_confirmation_nudge(self) -> bool:
        """True iff the last iteration's Verifier emitted a NON-ABSTAINING
        verdict AND the Agent has not yet committed.

        This fires on any non-abstaining verdict, not only a confirmation:
        "intermediate" and "probable" verdicts also warrant a
        commit-or-continue nudge.
        """
        if self.final_diagnosis_text is not None:
            return False
        if not self.iterations:
            return False
        last = self.iterations[-1].verifier_result
        if not isinstance(last, dict):
            return False
        # Prefer the explicit abstention flag if present (post-Verifier serialization).
        abst = last.get("abstention_triggered")
        if isinstance(abst, bool):
            return not abst
        # Fallback: confirmed_* startswith (legacy / partial-result iterations).
        diag = last.get("diagnostic_output", "")
        return isinstance(diag, str) and diag.startswith("confirmed_")

    # -- Token-budget signal (D2 / ) --

    def approximate_token_count(self) -> int:
        """Coarse upper-bound token count for the ledger's prompt-relevant content.

        Uses the standard ~4 chars/token heuristic across the Thought, Action,
        Action Input, and Observation fields of every iteration. Does NOT count
        Findings (rarely echoed back to the LLM) or Verifier results (only the
        confirmed_* label is surfaced via the nudge, which is small).

        Used to decide whether to summarize
        prior observations. The stub LedgerSummarizer never triggers; the real
        impl uses this signal against the model's max_model_len.
        """
        chars = 0
        for rec in self.iterations:
            chars += len(rec.thought or "")
            chars += len(rec.action or "")
            chars += len(rec.action_input or "")
            chars += len(rec.observation or "")
        return chars // 4

    # ---- Serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable snapshot. Round-trips via EvidenceLedger.load(path)."""
        return {
            "ledger_version":          LEDGER_VERSION,
            "agent_id":             self.agent_id,
            "hadm_id":                 self.hadm_id,
            "kb_tag":                  self.kb_tag,
            "created_at_utc":          self.created_at_utc,
            "iterations":              [r.to_dict() for r in self.iterations],
            "final_diagnosis_text":    self.final_diagnosis_text,
            "final_verifier_result":   self.final_verifier_result,
            "outcome_label":            self.outcome_label,
            "termination_reason":      self.termination_reason,
        }

    @classmethod
    def load(cls, path: Path | str, *, schema_class: Any | None = None) -> "EvidenceLedger":
        """Reconstruct a EvidenceLedger from disk (e.g. for crash recovery or audit).

        The reconstructed EvidenceLedger has log_dir=None to avoid accidental overwrites;
        callers that want to resume writing should set ledger.log_dir manually
        after loading.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        led = cls(
            agent_id=data["agent_id"],
            hadm_id=data["hadm_id"],
            kb_tag=data["kb_tag"],
            log_dir=None,
            schema_class=schema_class,
        )
        led.created_at_utc = data.get("created_at_utc", led.created_at_utc)
        led.iterations = [IterationRecord.from_dict(r) for r in data.get("iterations", [])]
        led.final_diagnosis_text = data.get("final_diagnosis_text")
        led.final_verifier_result = data.get("final_verifier_result")
        led.outcome_label = data.get("outcome_label")
        led.termination_reason = data.get("termination_reason")
        return led

    # ---- Internal ----

    def _persist_if_configured(self) -> None:
        """Atomic write: temporary file then rename.

        Write to <log_dir>/ledger.json.tmp then os.replace() to ledger.json so
        that even if the process is killed mid-write, the on-disk ledger.json
        is either fully old or fully new -- never partial.
        """
        if self.log_dir is None:
            return
        path = self.log_dir / LEDGER_FILE_NAME
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


# Module-level helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_finding_leaf(v: Any) -> bool:
    """A leaf is either:
      - a non-dict value (bool, int, float, str, None, list)
      - a dict that carries a 'confidence' field (PresentFinding / NumericFinding /
        CtFinding shape)
    Non-leaf dicts are recursed into (e.g. `imaging`, `imaging.ct_findings`).
    """
    if not isinstance(v, dict):
        return True
    return "confidence" in v


def _confidence_of(v: Any) -> str | None:
    if isinstance(v, dict):
        c = v.get("confidence")
        if isinstance(c, str):
            return c
    return None


def _rank(c: str | None) -> int:
    return EvidenceLedger._CONFIDENCE_RANK.get(c or "", 0)


def _deep_merge_with_confidence(
    acc: dict[str, Any],
    delta: dict[str, Any],
) -> None:
    """Mutate `acc` by deep-merging `delta` under the confidence and retraction rules.

    Rules (priority order):
      1. If a key is new in acc, copy it from delta.
      2. If both values are non-leaf dicts (no 'confidence'), recurse.
      3. **Explicit retraction**: if the existing leaf has
         `present=True` and the new leaf has `present=False` AND the new leaf
         passed span verification (span_verified=True, set by SpanVerifier;
         absence of the field is treated as verified for backward compat),
         REPLACE existing -- the LLM is retracting an earlier claim with
         evidence-grounded support.
      4. Otherwise apply confidence-ranked rule: new wins if rank >= existing.
         (Plain-value leaves -- no dict at all -- fall into rank-0 vs rank-0
         and follow latest-wins semantics.)

    The retraction protocol means a later iteration can undo an earlier
    hallucinated `present=True` IFF the LLM explicitly emits `present=False`
    with a record-verifiable evidence_span. Without the verification gate, a
    naked `present=False` at low confidence cannot retract a high-confidence
    `True` -- so we keep some protection against the LLM oscillating without
    evidence.

    Note: when `delta[k]` is a non-leaf dict but `acc[k]` is a leaf (or vice
    versa), we treat it as a leaf-replacement using the latest value. This is
    defensive against malformed LLM outputs that change a field's shape mid-run.
    """
    for k, v_new in delta.items():
        if k not in acc:
            acc[k] = v_new
            continue
        v_old = acc[k]
        old_is_leaf = _is_finding_leaf(v_old)
        new_is_leaf = _is_finding_leaf(v_new)
        if not old_is_leaf and not new_is_leaf:
            _deep_merge_with_confidence(v_old, v_new)
            continue

        # Explicit retraction protocol. Fires on PresentFinding-shaped
        # leaves (i.e. leaves with a `present` key) when the LLM transitions
        # True -> False with a verifiable evidence_span.
        retracted = False
        if (isinstance(v_old, dict) and isinstance(v_new, dict)
                and "present" in v_old and "present" in v_new):
            old_present = v_old.get("present")
            new_present = v_new.get("present")
            # span_verified defaults to True for legacy / non-walked records.
            new_span_ok = v_new.get("span_verified", True)
            if old_present is True and new_present is False and new_span_ok:
                acc[k] = v_new
                retracted = True
        if retracted:
            continue

        if _rank(_confidence_of(v_new)) >= _rank(_confidence_of(v_old)):
            acc[k] = v_new
        # else: keep existing
