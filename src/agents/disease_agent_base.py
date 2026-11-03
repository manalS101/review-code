"""The iterative disease agent loop.

All four condition agents share this loop verbatim. A subclass supplies only
what is genuinely disease-specific:

    agent_id             the KB registry key
    _kb_label               the heading used in the system prompt
    _agent_role_prompt() role and format text: criterion ids, unit
                            conventions, a worked example
    _extra_reminders()      an optional per-disease reminder appended to each
                            user prompt

Everything else lives here: the request-parse-verify-merge cycle, parser
retries, the span-grounding gate, tool dispatch, ledger appends, the verifier
call, the early-commit nudge, repetition detection, termination classification,
and persistence by atomic rename.

    class AppendicitisAgent(DiseaseAgentBase):
        agent_id = "appendicitis"
        _kb_label = "Appendicitis"
        def _agent_role_prompt(self) -> str:
            return _AGENT_ROLE

    AppendicitisAgent(llm_client=..., kb_registry=..., adapter=...).run(hadm_id)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.adapter import Adapter
from src.agents.anatomical_anchor import LOCALIZING_TERRITORY, anchor_localizing_findings
from src.agents.tools import ToolContext, ToolDispatcher
from src.agents.span_verifier import verify_findings_delta
from src.record import build_record_text
from src.diagnosis_parser import (
    AgentAction,
    AgentFinish,
    DiagnosisParser,
    ParserError,
)
from src.kb_loader import KBRegistry
from src.label_mapper import LabelMapper, LabelMapperError
from src.evidence_ledger import IterationRecord, EvidenceLedger
from src.llm_client import LLMClient, LLMError, LLMResponse
from src.verifier import Verifier, VerifierResult


AGENT_VERSION = "1.2"   # Span verification + retraction protocol + broadened nudge


# Result type

@dataclass
class AgentResult:
    """Top-level summary returned by one agent's run()."""

    hadm_id: int
    agent_id: str
    final_diagnosis_text: str
    final_verifier_result: dict[str, Any]       # serialized VerifierResult
    outcome_label: str                            # one of OUTCOME_LABELS
    ledger: EvidenceLedger
    termination_reason: str
    total_llm_calls: int
    total_tool_calls: int
    total_latency_ms: float
    parser_retries_consumed: int                 # cumulative across all iterations
    # Aggregation-only instrumentation (no effect on the diagnosis): token usage
    # summed across every LLM call, and a count of top-level findings keys the LLM
    # emitted that are NOT in the extraction schema (the extra="ignore" silent-drop
    # footgun). Defaults keep older AgentResult constructions valid.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_schema_drops: int = 0


# Constants (generic prompts)

_NUDGE_TEMPLATE = """\
[Verifier note from your accumulated evidence]
Your accumulated evidence supports the diagnostic finding: {diagnostic_output}
(Verifier branch {branch}, outcome key {outcome_key}).
If you have sufficient evidence to commit, emit:
  Thought: <reasoning>
  Findings: <any final additions>
  Final Diagnosis: <condition>
  Treatment: <plan>
Otherwise, continue investigating with one more Action."""


# Base Agent

class DiseaseAgentBase:
    """Iterative evidence-gathering loop shared across the four conditions.

    Subclasses MUST set ``agent_id`` and ``_kb_label`` class attributes and
    override ``_agent_role_prompt()``.
    """

    # ---- Subclass contract (must override) ----
    agent_id: str = ""
    _kb_label: str = ""
    agent_version: str = AGENT_VERSION

    def _agent_role_prompt(self) -> str:
        """Return the disease-specific role + format prompt.

        The base templates ``{MAX_ITERATIONS}`` into this text; subclasses may
        include that placeholder verbatim.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _agent_role_prompt()"
        )

    def _anatomical_territory(self) -> dict[str, Any] | None:
        """The per-disease anatomical territory used to retract
        span-real-but-misattributed localizing findings (clinical + imaging)
        per-iteration, BEFORE the EvidenceLedger merge. Keyed automatically on
        ``agent_id`` from ``LOCALIZING_TERRITORY`` so every agent is
        covered; returns None if the disease has no territory defined.
        See src/agents/anatomical_anchor.py.
        """
        return LOCALIZING_TERRITORY.get(self.agent_id)

    def _extraction_schema_class(self) -> Any:
        """Pydantic schema the EvidenceLedger uses to validate merged evidence into
        Verifier-input shape (.to_verifier_evidence()). Subclasses MUST
        override -- the schema's criterion fields determine which findings
        survive the merge. Returning the wrong schema silently drops every
        criterion (extra="ignore"), so there is no safe default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _extraction_schema_class()"
        )

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        kb_registry: KBRegistry,
        adapter: Adapter,
        max_iterations: int = 10,
        log_dir: Path | str | None = None,
        lab_mapping_csv: Path | str = "data/lab_test_mapping.csv",
        max_parser_retries: int = 2,
        max_response_tokens: int = 4096,   # large enough for a full Findings block
        temperature: float = 0.0,
        enable_span_verifier: bool = True,
        enable_anatomical_anchor: bool = True,
        auto_commit_stable_verdict: int | None = 2,
        enable_commit_nudge: bool = True,
        force_final_diagnosis: bool = False,
        suppress_kb_output_menu: bool = False,
    ) -> None:
        if not self.agent_id:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty agent_id class attribute"
            )
        self.llm = llm_client
        self.adapter = adapter
        self.kb_registry = kb_registry
        self.kb = kb_registry.kb(self.agent_id)
        self.verifier = Verifier(self.kb)
        self.parser = DiagnosisParser()
        self.label_mapper = LabelMapper(kb_registry)
        self.max_iterations = max_iterations
        self.max_parser_retries = max_parser_retries
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.log_dir_root = Path(log_dir) if log_dir is not None else None
        self.lab_mapping_csv = Path(lab_mapping_csv)
        # Span-verification gate (can be disabled for canned-LLM
        # tests that intentionally use non-record-grounded fixture spans).
        self.enable_span_verifier = enable_span_verifier
        # Per-iteration anatomical/semantic anchoring of localizing findings
        # (clinical + imaging). Can be disabled for canned-LLM tests that
        # intentionally use region-misattributed fixture spans to exercise
        # downstream components in isolation.
        self.enable_anatomical_anchor = enable_anatomical_anchor
        # When the Verifier holds a POSITIVE
        # verdict (outcome_label != insufficient_evidence) for this many
        # CONSECUTIVE iterations, the loop auto-terminates instead of waiting
        # for the 8B to emit Final Diagnosis (which it often never does, burning
        # all max_iterations). The final label is unchanged (final Verifier pass
        # is authoritative); only termination timing and reason change. Set to
        # None to disable.
        self.auto_commit_stable_verdict = auto_commit_stable_verdict
        # Verifier-ablation knobs. For a "no verifier, the model decides"
        # baseline:
        #   enable_commit_nudge=False  -> the verifier nudge is NOT injected into
        #       the prompt, so no verifier signal leaks into the LLM's decision.
        #   force_final_diagnosis=True -> if the loop ends without the LLM emitting
        #       its own Final Diagnosis (and auto-commit is off), make ONE post-loop
        #       forced call so the model commits in its own words (Hager parity: his
        #       single-agent baseline always ends on a Final Diagnosis). The emitted
        #       free text is Hager-fuzzy-scored upstream -- the verifier never decides.
        # Defaults keep the deployed behaviour unchanged.
        self.enable_commit_nudge = enable_commit_nudge
        self.force_final_diagnosis = force_final_diagnosis
        # Clean LLM-only ablation: suppress the KB diagnostic_outputs label
        # menu from the system prompt so the model cannot parrot a KB output key as its
        # free-text Final Diagnosis. Default False = deployed behaviour unchanged. The
        # diagnostic CRITERIA the agent reasons over are unaffected.
        self.suppress_kb_output_menu = suppress_kb_output_menu
        # Repetition detection across iterations.
        # _recent_action_signatures stores (action, normalized_input) for every
        # successfully-parsed AgentAction; _pending_repetition_hint is the next
        # user prompt's injection slot (consumed on read).
        self._recent_action_signatures: list[tuple[str, str]] = []
        self._pending_repetition_hint: str | None = None

    # Public entry point

    def run(self, hadm_id: int) -> AgentResult:
        hadm_id = int(hadm_id)

        # Reset per-run state so repeated.run calls don't leak.
        self._recent_action_signatures = []
        self._pending_repetition_hint = None

        case = self.adapter.load_case(hadm_id)
        hpi = case.get("Patient History") or ""

        # Accumulate the record text the LLM has actually been
        # SHOWN so far. HPI is in every prompt; observations from tool calls
        # are appended after each iteration. SpanVerifier checks evidence_spans
        # only against this accumulator (NOT against the full case dict) --
        # otherwise the LLM could "cheat" by citing a record section it has not
        # observed.
        record_text_seen = hpi + "\n"
        # Full-record rival channel for the deterministic contradictory-demotion
        # gate reads a RIVAL disease's DEFINING lab criterion (Atlanta AP_C2:
        # lipase/amylase >= 3x ULN) from the FULL record -- NOT record_text_seen.
        # This is a guideline safety gate over structured labs, not an LLM-grounding
        # check, so the anti-cheat seen-text scoping (which guards span verification)
        # must not apply: the criterion must fire even when THIS agent never
        # requested the enzyme lab. Without this the cholecystitis agent can
        # confirm a true pancreatitis via branch D while never surfacing a lipase
        # 35 times the upper limit. Span verification below still uses record_text_seen.
        full_record_text = build_record_text(case)
        kb_tag = case.get("_case_meta", {}).get("kb_tag") or "kb-v4-frozen"

        # Per-case log directory (the Agent writes; the EvidenceLedger writes its
        # own file inside log_dir).
        case_dir: Path | None = None
        if self.log_dir_root is not None:
            case_dir = self.log_dir_root / f"{hadm_id}_{self.agent_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "transcripts").mkdir(exist_ok=True)

        system_prompt = self._build_system_prompt()

        ledger = EvidenceLedger(
            agent_id=self.agent_id,
            hadm_id=hadm_id,
            kb_tag=kb_tag,
            log_dir=case_dir,
            schema_class=self._extraction_schema_class(),
        )
        tool_dispatcher = ToolDispatcher(case, lab_mapping_csv=self.lab_mapping_csv)
        tool_ctx = ToolContext()

        # Persist immutable initial artifacts (case + system prompt).
        self._persist_initial_artifacts(case_dir, case, system_prompt, ledger.created_at_utc)

        run_started_at = datetime.now(timezone.utc)
        total_llm_calls = 0
        total_tool_calls = 0
        total_latency_ms = 0.0
        parser_retries_consumed = 0
        # Aggregation-only counters (do not influence any decision).
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_schema_drops = 0
        _schema_fields = _schema_field_names(self._extraction_schema_class())

        # Iteration loop
        terminal_record: IterationRecord | None = None
        terminal_finish: AgentFinish | None = None
        termination_reason: str | None = None
        last_llm_error: str | None = None
        last_parser_error: str | None = None
        consecutive_positive_verdicts = 0
        auto_committed_dx: str | None = None

        for iter_num in range(1, self.max_iterations + 1):
            user_prompt = self._build_user_prompt(hpi=hpi, ledger=ledger)
            self._persist_prompt_text(case_dir, iter_num, user_prompt)

            # ---- Model call, with parser retries ----
            parsed: AgentAction | AgentFinish | None = None
            raw_text: str = ""
            llm_response: LLMResponse | None = None
            attempt_prompt = user_prompt
            iter_parser_retries = 0
            iter_llm_calls = 0
            try:
                for attempt in range(self.max_parser_retries + 1):
                    iter_llm_calls += 1
                    llm_response = self.llm.complete(
                        attempt_prompt,
                        system_prompt=system_prompt,
                        max_tokens=self.max_response_tokens,
                        temperature=self.temperature,
                        json_only=False,
                    )
                    raw_text = llm_response.text or ""
                    if llm_response.latency_ms is not None:
                        total_latency_ms += llm_response.latency_ms
                    total_prompt_tokens += llm_response.prompt_tokens or 0
                    total_completion_tokens += llm_response.completion_tokens or 0
                    try:
                        parsed = self.parser.parse(raw_text)
                        break
                    except ParserError as pe:
                        last_parser_error = str(pe)
                        if attempt >= self.max_parser_retries:
                            raise
                        iter_parser_retries += 1
                        attempt_prompt = self._build_parser_retry_prompt(
                            base_user_prompt=user_prompt,
                            previous_response=raw_text,
                            parser_error=str(pe),
                        )
            except ParserError:
                termination_reason = "parser_error"
                total_llm_calls += iter_llm_calls
                parser_retries_consumed += iter_parser_retries
                break
            except LLMError as le:
                last_llm_error = str(le)
                termination_reason = "llm_error"
                total_llm_calls += iter_llm_calls
                parser_retries_consumed += iter_parser_retries
                break

            total_llm_calls += iter_llm_calls
            parser_retries_consumed += iter_parser_retries
            # Count top-level findings keys the schema would silently drop (measure
            # only -- the actual drop already happens downstream in the EvidenceLedger).
            total_schema_drops += sum(
                1 for k in (parsed.findings_delta or {}) if k not in _schema_fields
            )

            # ---- Span verification, before tool dispatch ----
            # We verify against record_text_seen (HPI + prior observations).
            # The tool observation for THIS iteration is not yet available,
            # so a Findings entry that quotes the tool output for the current
            # iter cannot be verified yet -- those typically come in iter N+1.
            # When enable_span_verifier=False (canned-LLM tests), skip the
            # gate and pass findings through unchanged.
            if self.enable_span_verifier:
                verified_findings, span_audit = verify_findings_delta(
                    parsed.findings_delta, record_text_seen,
                )
                parsed.findings_delta = verified_findings
            else:
                span_audit = []

            # Per-iteration anatomical anchoring (e.g. diverticulitis colonic
            # anchor). Disease-parameterized via the hook below; a no-op unless
            # the subclass supplies anchor terms. Runs BEFORE the EvidenceLedger merge
            # so a correctly-anchored emission of a CT key in another iteration
            # still survives the confidence-ranked merge.
            territory = self._anatomical_territory()
            if self.enable_anatomical_anchor and territory:
                parsed.findings_delta, anchor_audit = anchor_localizing_findings(
                    parsed.findings_delta, territory,
                )
                span_audit = list(span_audit) + anchor_audit

            # ---- Tool dispatch (only if AgentAction) ----
            tool_observation: str | None = None
            tool_record: dict[str, Any] = {}
            if isinstance(parsed, AgentAction):
                tool_observation = tool_dispatcher.dispatch(
                    parsed.action, parsed.action_input, tool_ctx,
                )
                total_tool_calls += 1
                tool_record = {
                    "action":       parsed.action,
                    "action_input": parsed.action_input,
                    "observation":  tool_observation,
                }
                # Extend record_text_seen so the NEXT iteration's
                # span verification can match against this observation too.
                if tool_observation:
                    record_text_seen = record_text_seen + tool_observation + "\n"

            # ---- Build + append iteration record ----
            assert llm_response is not None  # noqa: S101
            rec = IterationRecord(
                iteration=iter_num,
                timestamp_utc=_now_iso(),
                thought=parsed.thought,
                findings_delta=parsed.findings_delta,
                action=parsed.action if isinstance(parsed, AgentAction) else None,
                action_input=parsed.action_input if isinstance(parsed, AgentAction) else None,
                observation=tool_observation,
                verifier_result={},   # filled below
                llm_request=_serialize_llm_request(
                    user_prompt=attempt_prompt,
                    system_prompt=system_prompt,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                ),
                llm_response_meta=_serialize_llm_response_meta(llm_response),
                raw_response_text=raw_text,
                parse_errors=list(parsed.parse_warnings),
                span_audit=span_audit,
            )

            # ---- Verifier on accumulated evidence ----
            try:
                evidence = ledger.current_evidence_with_provisional(rec)
                # Record-driven rival-criteria channel (deterministic; not LLM).
                # Reads the FULL record, not record_text_seen (see full_record_text above).
                evidence["_record_rival_criteria"] = _extract_record_rival_criteria(
                    full_record_text
                )
                # Record-driven normal-appendix veto (deterministic; not LLM).
                evidence["_record_appendix_normal"] = _extract_record_appendix_normal(
                    full_record_text
                )
                # Record-driven prior-appendectomy veto (deterministic; not LLM).
                evidence["_record_prior_appendectomy"] = _extract_record_prior_appendectomy(
                    full_record_text
                )
                vresult = self.verifier.verify(evidence)
                rec.verifier_result = _serialize_verifier_result(vresult)
            except Exception as ve:  # noqa: BLE001 -- verifier should never crash, but be defensive
                rec.verifier_result = {"error": f"{type(ve).__name__}: {ve}"}
                vresult = None  # type: ignore[assignment]

            ledger.append_iteration(rec)
            # Repetition detector: signature is (action, normalized input).
            # Fires only on Imaging because PE / Labs have natural variation.
            if isinstance(parsed, AgentAction):
                sig = (parsed.action, (parsed.action_input or "").strip().lower())
                self._recent_action_signatures.append(sig)
                if parsed.action == "Imaging":
                    same = sum(1 for s in self._recent_action_signatures if s == sig)
                    if same >= 2:
                        self._pending_repetition_hint = (
                            f"You have already requested {parsed.action!r} with input "
                            f"{(parsed.action_input or '')!r} {same} times. If the "
                            "observation keeps showing 'no further reports' or the same "
                            "content, switch to a different modality (CT vs Ultrasound vs MRI), "
                            "try a different region, or commit with Final Diagnosis."
                        )
            self._persist_iteration_transcripts(
                case_dir=case_dir,
                iter_num=iter_num,
                user_prompt=attempt_prompt,
                llm_response=llm_response,
                parsed=parsed,
                tool_record=tool_record,
                verifier_result_dict=rec.verifier_result,
                span_audit=span_audit,
            )

            # ---- Terminal? ----
            if isinstance(parsed, AgentFinish):
                terminal_record = rec
                terminal_finish = parsed
                break

            # ---- Auto-commit on a stable positive verdict ----
            # Only reached when the LLM did NOT itself finish this iteration.
            if self.auto_commit_stable_verdict is not None:
                iter_label: str | None = None
                try:
                    _do = (rec.verifier_result or {}).get("diagnostic_output")
                    if _do:
                        iter_label = self.label_mapper.to_outcome_label(self.agent_id, _do)
                except Exception:  # noqa: BLE001 -- label mapper must never break the loop
                    iter_label = None
                if iter_label is not None and iter_label != "insufficient_evidence":
                    consecutive_positive_verdicts += 1
                else:
                    consecutive_positive_verdicts = 0
                if consecutive_positive_verdicts >= self.auto_commit_stable_verdict:
                    termination_reason = "auto_committed_stable_verdict"
                    auto_committed_dx = (
                        (rec.verifier_result or {}).get("diagnostic_output")
                        or self.kb.disease
                    )
                    break

        # ---- LLM-only ablation: post-loop forced Final Diagnosis ----
        # If the model never committed on its own (no AgentFinish) and auto-commit
        # is off, force ONE final-diagnosis call so the LLM states a diagnosis in
        # its own words (Hager parity). Pure LLM channel -- the verifier result is
        # NOT consulted to build this. Skipped entirely unless force_final_diagnosis.
        forced_final_text: str | None = None
        if (self.force_final_diagnosis and terminal_finish is None
                and auto_committed_dx is None):
            forced_final_text, _fcalls, _flat, _fpt, _fct = self._force_final_diagnosis(
                system_prompt=system_prompt, hpi=hpi, ledger=ledger,
            )
            total_llm_calls += _fcalls
            total_latency_ms += _flat
            total_prompt_tokens += _fpt
            total_completion_tokens += _fct
            if termination_reason is None:
                termination_reason = "forced_final_diagnosis"

        # ---- Determine final state ----
        # current_evidence() can itself raise (LedgerError on a Pydantic schema
        # validation failure over malformed accumulated 8B findings). Keep it
        # INSIDE the guard so a bad final merge degrades to an indeterminate
        # verdict instead of crashing the whole case. Every crash seen on the
        # review sample came from dict-wrapped imaging *_done flags, which the
        # schemas now coerce.
        try:
            final_evidence = ledger.current_evidence()
            # Record-driven rival-criteria channel (deterministic; not LLM).
            # Reads the FULL record, not record_text_seen (see full_record_text above).
            final_evidence["_record_rival_criteria"] = _extract_record_rival_criteria(
                full_record_text
            )
            # Record-driven normal-appendix veto (deterministic; not LLM).
            final_evidence["_record_appendix_normal"] = _extract_record_appendix_normal(
                full_record_text
            )
            # Record-driven prior-appendectomy veto (deterministic; not LLM).
            final_evidence["_record_prior_appendectomy"] = _extract_record_prior_appendectomy(
                full_record_text
            )
            final_vresult = self.verifier.verify(final_evidence)
        except Exception:  # noqa: BLE001
            final_vresult = _build_fallback_verifier_result(self.kb.disease)

        final_dx_text = (
            terminal_finish.final_diagnosis if terminal_finish
            else (forced_final_text if forced_final_text is not None
                  else (auto_committed_dx or ""))
        )
        try:
            outcome_label = self.label_mapper.to_outcome_label(
                self.agent_id, final_vresult.diagnostic_output,
            )
        except LabelMapperError:
            outcome_label = "insufficient_evidence"

        if termination_reason is None:
            if terminal_finish is not None:
                termination_reason = self._classify_finish_termination(
                    final_dx_text=final_dx_text,
                    outcome_label=outcome_label,
                )
            else:
                termination_reason = (
                    "max_iterations_confirmed"
                    if outcome_label != "insufficient_evidence"
                    else "max_iterations_unconfirmed"
                )

        final_vresult_dict = _serialize_verifier_result(final_vresult)
        ledger.attach_final_diagnosis(
            final_diagnosis_text=final_dx_text,
            verifier_result=final_vresult_dict,
            outcome_label=outcome_label,
            termination_reason=termination_reason,
        )

        result = AgentResult(
            hadm_id=hadm_id,
            agent_id=self.agent_id,
            final_diagnosis_text=final_dx_text,
            final_verifier_result=final_vresult_dict,
            outcome_label=outcome_label,
            ledger=ledger,
            termination_reason=termination_reason,
            total_llm_calls=total_llm_calls,
            total_tool_calls=total_tool_calls,
            total_latency_ms=total_latency_ms,
            parser_retries_consumed=parser_retries_consumed,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_schema_drops=total_schema_drops,
        )

        self._persist_final_artifacts(
            case_dir=case_dir,
            result=result,
            run_started_at=run_started_at,
            last_llm_error=last_llm_error,
            last_parser_error=last_parser_error,
        )
        return result

    # Prompt builders

    def _build_system_prompt(self) -> str:
        head = self._agent_role_prompt().replace("{MAX_ITERATIONS}", str(self.max_iterations))
        kb_backbone = self.kb.static_backbone(
            include_outputs=not self.suppress_kb_output_menu
        )
        return f"{head}\n\n=== Knowledge Base ({self._kb_label}) ===\n{kb_backbone}\n"

    def _build_user_prompt(self, *, hpi: str, ledger: EvidenceLedger) -> str:
        sections: list[str] = []
        sections.append("=== Case ===")
        sections.append("Patient History:")
        sections.append(hpi.strip() if hpi.strip() else "(not available)")

        if ledger.iterations:
            sections.append("\n=== Trajectory so far ===")
            for r in ledger.iterations:
                sections.append(f"Iteration {r.iteration}:")
                if r.thought:
                    sections.append(f"  Thought: {r.thought}")
                if r.action:
                    sections.append(f"  Action: {r.action}")
                    sections.append(f"  Action Input: {r.action_input or ''}")
                if r.observation:
                    obs_indented = "\n".join(f"    {line}" for line in r.observation.splitlines())
                    sections.append(f"  Observation:\n{obs_indented}")

            # Early-commit nudge (only if last iter Verifier saw a confirmed_*).
            # Suppressed entirely in the LLM-only ablation (enable_commit_nudge=False)
            # so the verifier's verdict never reaches the model.
            if self.enable_commit_nudge and ledger.has_pending_confirmation_nudge():
                last_vr = ledger.iterations[-1].verifier_result or {}
                outcome = last_vr.get("primary_rule_outcome", {}) or {}
                sections.append("\n" + _NUDGE_TEMPLATE.format(
                    diagnostic_output=last_vr.get("diagnostic_output", "?"),
                    branch=outcome.get("branch", "?"),
                    outcome_key=outcome.get("outcome_key", "?"),
                ))

        # Consume any pending repetition hint into the next prompt.
        if self._pending_repetition_hint:
            sections.append(f"\n[Hint] {self._pending_repetition_hint}")
            self._pending_repetition_hint = None

        sections.append("\n=== Your next turn ===")
        sections.append("Emit one Thought / Findings / Action / Action Input block, "
                        "OR commit with Thought / Findings / Final Diagnosis / Treatment.")
        reminders = self._extra_reminders()
        if reminders:
            sections.append(reminders)
        return "\n".join(sections)

    def _extra_reminders(self) -> str:
        """Per-disease reminder line appended to the end of every user prompt.

        Base default carries only the disease-agnostic span-grounding rule.
        Subclasses override to prepend disease-specific reminders (e.g.
        appendicitis' WBC raw-count unit convention, pancreatitis' enzyme
        ratio rule). Return "" to omit the reminder line entirely.
        """
        return ("[Reminder: every evidence_span MUST be a verbatim substring "
                "of the record -- omit criteria you cannot ground in record text.]")

    def _build_parser_retry_prompt(
        self,
        *,
        base_user_prompt: str,
        previous_response: str,
        parser_error: str,
    ) -> str:
        return (
            base_user_prompt
            + "\n\n=== Parser correction request ===\n"
            + "Your previous response could not be parsed. "
            + f"Reason: {parser_error}\n"
            + "Previous response was:\n---\n"
            + previous_response[:2000]
            + ("\n[...truncated]" if len(previous_response) > 2000 else "")
            + "\n---\nPlease re-emit ONE block in the required format "
              "(Thought / Findings / Action / Action Input, OR "
              "Thought / Findings / Final Diagnosis / Treatment)."
        )

    def _force_final_diagnosis(
        self, *, system_prompt: str, hpi: str, ledger: EvidenceLedger,
    ) -> tuple[str, int, float, int, int]:
        """LLM-only ablation: one post-loop call that forces the model to commit.

        Returns ``(final_dx_free_text, llm_calls, latency_ms, prompt_tokens,
        completion_tokens)``. The free text is whatever the model emits as its
        Final Diagnosis (Hager-fuzzy-scored upstream). On a parse/LLM failure
        returns the raw text (or ""). The verifier is never consulted here.
        """
        # Clean LLM-only ablation: when the KB output-key menu is
        # suppressed, also instruct a plain-English diagnosis so the model does not
        # answer with a label/code (belt-and-suspenders with the hidden menu).
        plain_english = (
            "\nState the diagnosis as a plain-English disease name "
            "(e.g. 'acute appendicitis' or 'no acute surgical diagnosis'), "
            "NOT a label, code, or underscore_key.\n"
            if self.suppress_kb_output_menu else "\n"
        )
        prompt = self._build_user_prompt(hpi=hpi, ledger=ledger) + (
            "\n\n=== FINAL ANSWER REQUIRED ===\n"
            "Do not request more evidence. Based only on what you have gathered, "
            "commit now to the single most likely diagnosis."
            + plain_english +
            "Emit exactly:\n"
            "Thought: <brief reasoning>\n"
            "Findings: <optional>\n"
            "Final Diagnosis: <one diagnosis>\n"
            "Treatment: <plan>"
        )
        try:
            resp = self.llm.complete(
                prompt, system_prompt=system_prompt,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature, json_only=False,
            )
        except LLMError:
            return "", 1, 0.0, 0, 0
        lat = resp.latency_ms or 0.0
        pt = resp.prompt_tokens or 0
        ct = resp.completion_tokens or 0
        text = resp.text or ""
        try:
            parsed = self.parser.parse(text)
            if isinstance(parsed, AgentFinish) and (parsed.final_diagnosis or "").strip():
                return parsed.final_diagnosis, 1, lat, pt, ct
        except ParserError:
            pass
        return text, 1, lat, pt, ct

    # Termination classification

    def _classify_finish_termination(
        self, *, final_dx_text: str, outcome_label: str,
    ) -> str:
        text_lower = (final_dx_text or "").lower()
        verifier_confirmed = outcome_label != "insufficient_evidence"

        # What disease did the LLM name?
        llm_claimed: str | None = None
        for d in ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"):
            if d in text_lower:
                llm_claimed = d
                break

        if llm_claimed is None:
            return "agent_finish_verifier_disagrees" if verifier_confirmed else "agent_finish_unconfirmed"

        if not verifier_confirmed:
            return "agent_finish_unconfirmed"

        if llm_claimed == outcome_label:
            return "agent_finish_confirmed"
        return "agent_finish_verifier_disagrees"

    # Persistence helpers

    def _persist_initial_artifacts(
        self,
        case_dir: Path | None,
        case: dict[str, Any],
        system_prompt: str,
        created_at_utc: str,
    ) -> None:
        if case_dir is None:
            return
        _atomic_write_json(case_dir / "case_snapshot.json", _json_safe(case))
        _atomic_write_text(case_dir / "system_prompt.txt", system_prompt)

    def _persist_prompt_text(
        self,
        case_dir: Path | None,
        iter_num: int,
        prompt: str,
    ) -> None:
        if case_dir is None:
            return
        path = case_dir / "transcripts" / f"iter_{iter_num:03d}_prompt.txt"
        _atomic_write_text(path, prompt)

    def _persist_iteration_transcripts(
        self,
        *,
        case_dir: Path | None,
        iter_num: int,
        user_prompt: str,
        llm_response: LLMResponse,
        parsed: AgentAction | AgentFinish,
        tool_record: dict[str, Any],
        verifier_result_dict: dict[str, Any],
        span_audit: list[dict[str, Any]] | None = None,
    ) -> None:
        if case_dir is None:
            return
        d = case_dir / "transcripts"
        _atomic_write_json(d / f"iter_{iter_num:03d}_request.json", {
            "iteration":       iter_num,
            "user_prompt":     user_prompt,
            "max_tokens":      self.max_response_tokens,
            "temperature":     self.temperature,
            "json_only":       False,
        })
        _atomic_write_json(d / f"iter_{iter_num:03d}_response.json", {
            "iteration":         iter_num,
            "text":              llm_response.text,
            "finish_reason":     llm_response.finish_reason,
            "model_id":          llm_response.model_id,
            "prompt_tokens":     llm_response.prompt_tokens,
            "completion_tokens": llm_response.completion_tokens,
            "latency_ms":        llm_response.latency_ms,
            "raw_metadata":      llm_response.raw_metadata,
        })
        _atomic_write_json(d / f"iter_{iter_num:03d}_parsed.json",
                           _parsed_to_dict(parsed))
        _atomic_write_json(d / f"iter_{iter_num:03d}_tool.json", tool_record)
        _atomic_write_json(d / f"iter_{iter_num:03d}_verifier.json",
                           verifier_result_dict)
        if span_audit is not None:
            _atomic_write_json(d / f"iter_{iter_num:03d}_span_audit.json",
                               list(span_audit))

    def _persist_final_artifacts(
        self,
        *,
        case_dir: Path | None,
        result: AgentResult,
        run_started_at: datetime,
        last_llm_error: str | None,
        last_parser_error: str | None,
    ) -> None:
        if case_dir is None:
            return
        _atomic_write_json(case_dir / "final_diagnosis.json", {
            "hadm_id":                  result.hadm_id,
            "agent_id":              result.agent_id,
            "final_diagnosis_text":     result.final_diagnosis_text,
            "final_verifier_result":    result.final_verifier_result,
            "outcome_label":             result.outcome_label,
            "termination_reason":       result.termination_reason,
        })
        ended_at = datetime.now(timezone.utc)
        _atomic_write_json(case_dir / "run_metadata.json", {
            "agent_id":              result.agent_id,
            "agent_version":         self.agent_version,
            "hadm_id":                  result.hadm_id,
            "kb_tag":                   result.ledger.kb_tag,
            "llm_model_id":             self.llm.model_id,
            "max_iterations":           self.max_iterations,
            "max_parser_retries":       self.max_parser_retries,
            "started_at_utc":           run_started_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "ended_at_utc":             ended_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "termination_reason":       result.termination_reason,
            "total_iterations":         len(result.ledger.iterations),
            "total_llm_calls":          result.total_llm_calls,
            "total_tool_calls":         result.total_tool_calls,
            "total_latency_ms":         result.total_latency_ms,
            "parser_retries_consumed":  result.parser_retries_consumed,
            "total_tokens":             result.total_prompt_tokens + result.total_completion_tokens,
            "total_prompt_tokens":      result.total_prompt_tokens,
            "total_completion_tokens":  result.total_completion_tokens,
            "schema_drops":             result.total_schema_drops,
            "last_llm_error":           last_llm_error,
            "last_parser_error":        last_parser_error,
        })


# Module-level helpers

# Record-driven detection of a RIVAL disease's DEFINING lab
# criterion. Matches the lab lines the tools surface, in both the
# "Lipase = <value> IU/L (ref 0-60)" and "(ref: 0.0-60.0)" renderings.
# group(1) is the value, group(2) the upper reference bound.
_RIVAL_ENZYME_RE = re.compile(
    r"(?:Lipase|Amylase)\s*=\s*([0-9][0-9.,]*)\s*[^()\n]*\("
    r"\s*ref:?\s*[0-9.,]+\s*-\s*([0-9][0-9.,]*)\s*\)",
    re.IGNORECASE,
)


def _extract_record_rival_criteria(record_text: str) -> dict[str, Any]:
    """Deterministically parse RIVAL-disease DEFINING lab criteria from the record.

    Currently the Revised Atlanta AP_C2 criterion for acute pancreatitis -- serum
    amylase and/or lipase >= 3x ULN, a DEFINING diagnostic criterion
    (WSES_pancreatitis_2019; StatPearls Acute Pancreatitis). Returns the maximum
    observed (value / ref_upper) ratio across lipase and amylase as
    ``pancreatic_enzyme_max_x_uln`` so the verifier contradictory-demotion gate
    can demote a non-pancreatitis confirm when a rival's defining criterion is
    documented in the record -- even if the pancreatitis agent failed to surface
    it.

    Record-driven: parses enzyme labs from the FULL record text (build_record_text),
    NOT only the agent's surfaced seen-text -- so the gate fires even when this
    agent never requested the enzyme lab. Does not depend
    on the LLM populating any structured contradictory field. Best-effort -- returns
    {} when no parseable enzyme line (value + "(ref: lo-hi)") is present, so a bare
    narrative mention ("elevated lipase") cannot fire it.
    """
    if not record_text:
        return {}
    ratios: list[float] = []
    for m in _RIVAL_ENZYME_RE.finditer(record_text):
        try:
            val = float(m.group(1).replace(",", ""))
            hi = float(m.group(2).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if hi > 0:
            ratios.append(val / hi)
    if not ratios:
        return {}
    return {"pancreatic_enzyme_max_x_uln": max(ratios)}


# A radiology report that explicitly documents a NORMAL /
# non-dilated appendix rules out appendicitis (StatPearls/WSES: CT appendicitis is
# anchored on an ABNORMAL appendix). Detect it deterministically from the FULL
# record so a positive imaging signal cannot be credited on reactive peri-appendiceal
# stranding when the appendix itself is read as normal, as on a cecal
# diverticulitis whose CT reports the appendix as normal and not dilated.
_APPENDIX_MENTION = ("appendix", "appendice", "appendiceal", "appendico")
_APPENDIX_NORMAL_MARKERS = (
    "normal", "unremarkable", "within normal limits", "not dilated",
    "non-dilated", "nondilated", "normal caliber", "normal in caliber",
)
_APPENDIX_ABNORMAL_MARKERS = (
    "dilated", "inflamed", "inflammation", "thicken", "enlarged", "appendicolith",
    "ruptur", "perforat", "abscess", "hyperemi", "phlegmon", "fecalith",
)
_APPENDIX_NEGATORS = ("not ", "non-", "non ", "no ")


def _extract_record_appendix_normal(record_text: str) -> bool:
    """True iff the record documents a NORMAL / non-dilated appendix and NEVER an
    abnormal one. Conservative: fires only on an explicit normal-appendix statement
    (so a true appendicitis -- which always describes a dilated/inflamed/thickened
    appendix -- is never vetoed). Used by the appendicitis verifier to block a
    positive imaging signal when the appendix itself is read as normal.
    """
    if not record_text:
        return False
    normal_found = False
    abnormal_found = False
    for sent in re.split(r"[.;\n]", record_text):
        s = sent.lower()
        if not any(m in s for m in _APPENDIX_MENTION):
            continue
        sent_abnormal = False
        for term in _APPENDIX_ABNORMAL_MARKERS:
            idx = s.find(term)
            while idx != -1:
                prefix = s[max(0, idx - 10):idx]
                if not any(neg in prefix for neg in _APPENDIX_NEGATORS):
                    sent_abnormal = True
                    break
                idx = s.find(term, idx + 1)
            if sent_abnormal:
                break
        if sent_abnormal:
            abnormal_found = True
        elif any(m in s for m in _APPENDIX_NORMAL_MARKERS):
            normal_found = True
    return normal_found and not abnormal_found


# A documented PRIOR appendectomy makes acute appendicitis
# anatomically impossible (no appendix). Detect it deterministically from the record
# to hard-veto an appendicitis confirm. On a record listing "Appendectomy" under
# past surgical history, the 8B confirmed branch E from a pelvic abscess it had
# mis-mapped to periappendiceal_abscess.
_APPENDECTOMY_TERMS = ("appendectomy", "appendicectomy")


def _extract_record_prior_appendectomy(record_text: str) -> bool:
    """True iff the record documents a PRIOR appendectomy. Excludes the
    discharge-diagnosis section (marked 'do not use for extraction'), which can name
    the CURRENT appendectomy that TREATED this admission's appendicitis -- e.g.
    'acute appendicitis s/p appendectomy', a true appendicitis that must not be
    vetoed. After that exclusion, an appendectomy mention is a past surgery.
    """
    if not record_text:
        return False
    head = re.split(r"===\s*discharge diagnosis", record_text, flags=re.IGNORECASE)[0]
    head_l = head.lower()
    return any(t in head_l for t in _APPENDECTOMY_TERMS)


def _schema_field_names(schema_class: Any) -> frozenset[str]:
    """Top-level field names of a Pydantic extraction schema (for the schema-drop
    counter). Best-effort: returns an empty set if introspection fails, which makes
    the drop count 0 rather than ever affecting behaviour."""
    try:
        return frozenset(schema_class.model_fields.keys())
    except Exception:  # noqa: BLE001
        return frozenset()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic write of a UTF-8 text file. Survives mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomic write of a JSON file (pretty-printed). Survives mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    """Fallback JSON serializer for dataclasses + Paths + datetimes."""
    if dataclasses.is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _json_safe(obj: Any) -> Any:
    """Coerce non-JSON-native types in `obj` recursively. Used for case_snapshot."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return [_json_safe(x) for x in obj]
    if dataclasses.is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy / pandas scalars commonly carry through here -- use repr fallback.
    try:
        import math
        if isinstance(obj, float) and math.isnan(obj):
            return None
    except Exception:  # noqa: BLE001
        pass
    return str(obj)


def _serialize_llm_request(
    *,
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "system_prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16],
        "user_prompt_len":    len(user_prompt),
        "max_tokens":         max_tokens,
        "temperature":        temperature,
        "json_only":          False,
    }


def _serialize_llm_response_meta(resp: LLMResponse) -> dict[str, Any]:
    return {
        "model_id":          resp.model_id,
        "finish_reason":     resp.finish_reason,
        "prompt_tokens":     resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "latency_ms":        resp.latency_ms,
    }


def _serialize_verifier_result(vr: VerifierResult) -> dict[str, Any]:
    return {
        "diagnostic_output":      vr.diagnostic_output,
        "abstention_triggered":   vr.abstention_triggered,
        "abstention_reason":      vr.abstention_reason,
        "primary_rule_outcome":   _json_safe(vr.primary_rule_outcome),
        "criteria_evaluated":     dict(vr.criteria_evaluated),
        "step_trace":             [asdict(s) for s in vr.step_trace],
        "kb_id":                  vr.kb_id,
        "disease":                vr.disease,
        "framework_type":         vr.framework_type,
    }


def _build_fallback_verifier_result(disease: str) -> VerifierResult:
    """Constructed only when the Verifier itself crashes (should never happen).

    Returns an indeterminate VerifierResult so the run can still produce a
    outcome_label without aborting.
    """
    return VerifierResult(
        diagnostic_output="unsupported_or_alternative_diagnosis_more_likely",
        abstention_triggered=True,
        abstention_reason="verifier_exception",
        primary_rule_outcome={"branch": "fallback", "outcome_key": "fallback"},
        criteria_evaluated={},
        step_trace=[],
        kb_id="",
        disease=disease,
        framework_type="",
    )


def _parsed_to_dict(parsed: AgentAction | AgentFinish) -> dict[str, Any]:
    if isinstance(parsed, AgentFinish):
        return {
            "kind":             "AgentFinish",
            "thought":          parsed.thought,
            "findings_delta":   parsed.findings_delta,
            "final_diagnosis":  parsed.final_diagnosis,
            "treatment":        parsed.treatment,
            "parse_warnings":   list(parsed.parse_warnings),
        }
    return {
        "kind":             "AgentAction",
        "thought":          parsed.thought,
        "findings_delta":   parsed.findings_delta,
        "action":           parsed.action,
        "action_input":     parsed.action_input,
        "parse_warnings":   list(parsed.parse_warnings),
    }


# Note: ``EvidenceLedger.current_evidence_with_provisional`` (used by the per-iteration
# Verifier pass below) is now a proper method on the EvidenceLedger class
# (src/evidence_ledger.py) rather than being patched on from here.
