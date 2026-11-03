"""Phase 2: knowledge-based grounded diagnosis.

Takes the confirmed candidates from Phase 1 and deterministically resolves them
into a single outcome. Pure Python, no model call: it reads each agent's mapped
label and its serialized verifier result, never free text the model wrote.

A candidate is an agent whose mapped label equals its own condition, that is,
one that positively claims its condition. No candidate returns insufficient
evidence, exactly one candidate is selected outright, and two or more enter the
ordered resolution cascade.

The cascade is ordered deliberately. `label_strength` decides first: a full
confirmation outranks a partial one, and it is the only axis that means the same
thing across all four frameworks. Candidates still tied are separated by the
knowledge base's cross-condition contradiction rules, which are
guideline-grounded. Only if that is inconclusive does the number of independent
supporting criteria act as a last resort, because a raw count is not comparable
across frameworks: the cholecystitis integrated assessment offers up to three
streams where the pancreatitis two-of-three rule offers two. A tie that survives
all three returns insufficient evidence and is flagged for review rather than
guessed.

Mode A keeps that outcome. Mode B applies a deterministic forced-choice fallback
based on verifier tier, support breadth and a fixed condition ordering, so that
the framework can be compared with non-abstaining baselines. Neither mode makes
any further model call.

    Resolver(kb_registry).resolve(multi_agent_result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.disease_agent_base import AgentResult
from src.kb_loader import KBRegistry
from src.label_mapper import OUTCOME_LABELS
from src.multi_agent import MultiAgentResult


# Confirming outputs that are only a PARTIAL claim (label_strength 0). Every
# other candidate output is a full confirm (label_strength 1). Grounded in the
# KB diagnostic_outputs ordering: cholecystitis branch D (clinical + imaging,
# no lab) yields probable_ACC, weaker than a three-stream confirm.
_PARTIAL_CONFIRM_OUTPUTS: frozenset[str] = frozenset({"probable_ACC"})


RES_NO_CANDIDATES = "no_candidates"
RES_SINGLE_CANDIDATE = "single_candidate"
RES_CONVICTION = "conviction_winner"
RES_KB_CROSSREF = "kb_crossref_winner"
RES_UNRESOLVED = "unresolved_multiple_candidates"
RES_FORCED_CHOICE = "forced_choice_argmax"

# Deterministic disease order for the final forced-choice tiebreak (stable; not
# clinically meaningful -- only breaks exact tier+breadth ties reproducibly).
_FORCED_CHOICE_DISEASE_ORDER: tuple[str, ...] = (
    "pancreatitis", "diverticulitis", "cholecystitis", "appendicitis",
)


# Result type

@dataclass
class ResolutionResult:
    """Final resolution outcome for one case."""

    final_label: str                  # one of OUTCOME_LABELS
    winner: str | None                       # winning disease, or None
    decision_path: str                       # ARB_* constant
    candidates: list[str]
    convictions: dict[str, dict[str, int]]   # disease -> {label_strength, support_breadth}
    resolution_failed: bool
    trace: dict[str, Any] = field(default_factory=dict)


# Resolver

class Resolver:
    """Deterministic, KB-grounded resolution over the four agent results."""

    def __init__(self, kb_registry: KBRegistry) -> None:
        self.kb_registry = kb_registry

    # ---- public API ----

    def resolve(
        self,
        results: MultiAgentResult | dict[str, AgentResult],
        *,
        forced_choice: bool = False,
    ) -> ResolutionResult:
        """Pick the final five-outcome label.

        forced_choice (Mode B, EVALUATION-ONLY): when set, the Resolver NEVER
        abstains -- on what would be an abstention (0 candidates, or an unresolved
        dual-confirmation tie) it instead emits the disease whose agent came
        CLOSEST to confirming, ranked by has-satisfied-criterion then verifier-outcome
        tier then support_breadth (no new LLM call). The DEPLOYED system uses Mode A.
        This is the apples-to-apples arm vs a forced-choice baseline (Hager) that
        cannot abstain; report both numbers and the Mode A<->B gap.
        """
        result_map = self._as_result_map(results)
        candidates = [
            disease for disease, r in result_map.items()
            if r.outcome_label == disease
        ]
        convictions = {
            d: self._conviction_fields(result_map[d]) for d in candidates
        }

        # 0 candidates -> abstain (Mode A) or forced argmax over all four (Mode B).
        if not candidates:
            if forced_choice:
                return self._forced_choice_result(
                    result_map, list(result_map.keys()), candidates, convictions,
                    reason="no candidate -> closest-to-confirming agent",
                )
            return ResolutionResult(
                final_label="insufficient_evidence",
                winner=None,
                decision_path=RES_NO_CANDIDATES,
                candidates=[],
                convictions={},
                resolution_failed=False,
                trace={"reason": "no agent positively claimed its disease"},
            )

        # 1 candidate -> that disease.
        if len(candidates) == 1:
            d = candidates[0]
            return ResolutionResult(
                final_label=d,
                winner=d,
                decision_path=RES_SINGLE_CANDIDATE,
                candidates=candidates,
                convictions=convictions,
                resolution_failed=False,
                trace={"reason": f"sole candidate {d!r}"},
            )

        # >=2 candidates.
        # label_strength is the only CROSS-DISEASE-commensurable
        # axis (full confirm 1 vs partial confirm 0). support_breadth is a raw
        # per-framework criterion count -- chol integrated_assessment yields up to
        # 3 streams while pancreatitis n_of_m caps at 2 -- so it is NOT comparable
        # across diseases and must NOT outrank a guideline signal. Among
        # candidates tied on label_strength we therefore consult the KB
        # contradictory cross-reference (guideline-grounded: a disease whose KB
        # names the rival is argued against) BEFORE falling back to breadth.
        # This closes a false win where cholecystitis breadth 3 beat a clean two-of-
        # three Atlanta pancreatitis: the cholecystitis KB names pancreatitis, so
        # the cross-reference now favours pancreatitis. Breadth still resolves ties
        # the cross-reference cannot, such as cholecystitis against diverticulitis,
        # where neither KB names the other.
        conv_tuple = {d: self._conviction_tuple(result_map[d]) for d in candidates}
        max_strength = max(convictions[d]["label_strength"] for d in candidates)
        strength_top = [
            d for d in candidates if convictions[d]["label_strength"] == max_strength
        ]

        # Unique highest label_strength -> winner.
        if len(strength_top) == 1:
            d = strength_top[0]
            return ResolutionResult(
                final_label=d,
                winner=d,
                decision_path=RES_CONVICTION,
                candidates=candidates,
                convictions=convictions,
                resolution_failed=False,
                trace={"reason": "unique highest label_strength",
                       "conviction_tuples": {k: list(v) for k, v in conv_tuple.items()},
                       "top": strength_top},
            )

        # Tie on label_strength -> KB contradictory cross-reference FIRST
        # (guideline-grounded; outranks the non-commensurable breadth count).
        crossref_winner = self._kb_crossref_winner(strength_top)
        if crossref_winner is not None:
            return ResolutionResult(
                final_label=crossref_winner,
                winner=crossref_winner,
                decision_path=RES_KB_CROSSREF,
                candidates=candidates,
                convictions=convictions,
                resolution_failed=False,
                trace={"reason": "KB contradictory cross-reference resolved label_strength tie",
                       "conviction_tuples": {k: list(v) for k, v in conv_tuple.items()},
                       "tied": strength_top},
            )

        # Cross-reference inconclusive -> support_breadth as a weak last-resort
        # heuristic among the label_strength-tied candidates.
        max_breadth = max(convictions[d]["support_breadth"] for d in strength_top)
        breadth_top = [
            d for d in strength_top if convictions[d]["support_breadth"] == max_breadth
        ]
        if len(breadth_top) == 1:
            d = breadth_top[0]
            return ResolutionResult(
                final_label=d,
                winner=d,
                decision_path=RES_CONVICTION,
                candidates=candidates,
                convictions=convictions,
                resolution_failed=False,
                trace={"reason": "support_breadth tie-break (KB cross-reference inconclusive)",
                       "conviction_tuples": {k: list(v) for k, v in conv_tuple.items()},
                       "top": breadth_top},
            )

        # Unresolved dual/multi confirmation -> abstain (Mode A) or, in Mode B,
        # forced-pick among the tied candidates (highest breadth, then order).
        if forced_choice:
            return self._forced_choice_result(
                result_map, breadth_top, candidates, convictions,
                reason="unresolved dual-confirmation -> highest-support tied candidate",
            )
        return ResolutionResult(
            final_label="insufficient_evidence",
            winner=None,
            decision_path=RES_UNRESOLVED,
            candidates=candidates,
            convictions=convictions,
            resolution_failed=True,
            trace={"reason": "tied label_strength + breadth, KB cross-reference inconclusive",
                   "conviction_tuples": {k: list(v) for k, v in conv_tuple.items()},
                   "tied": breadth_top},
        )

    # ---- forced choice (Mode B, evaluation-only) ----

    # Verifier-outcome tiers: how close a NON-confirming agent is to a confirm.
    # Higher = closer. Classified from the diagnostic_output string so it is
    # framework-agnostic across all four KBs. (Confirming outputs never reach the
    # forced-choice path -- such an agent is already a candidate.)
    @staticmethod
    def _forced_choice_tier(output: str) -> int:
        o = (output or "").lower()
        if any(k in o for k in ("intermediate", "suspected", "needs_imaging",
                                "needs_more_workup", "observation", "possible")):
            return 3   # partial-positive: needs imaging / suspected / possible
        if any(k in o for k in ("uncertain", "missing_data", "insufficient")):
            return 2   # neutral: indeterminate / missing data
        if "low_risk" in o or "low risk" in o:
            return 1   # actively low risk
        if any(k in o for k in ("alternative", "unsupported", "not_supported",
                                "more_likely")):
            return 0   # argued against
        return 2

    def _forced_choice_result(
        self,
        result_map: dict[str, AgentResult],
        choice_pool: list[str],
        candidates: list[str],
        convictions: dict[str, dict[str, int]],
        *,
        reason: str,
    ) -> ResolutionResult:
        """Deterministically pick the closest-to-confirming candidate disease.

        Rank by (verifier-outcome tier, support_breadth), tie-broken by a fixed
        disease order. Never abstains. Records the per-disease tiers in the trace.
        """
        order_idx = {d: i for i, d in enumerate(_FORCED_CHOICE_DISEASE_ORDER)}
        tiers = {
            d: self._forced_choice_tier(
                (result_map[d].final_verifier_result or {}).get("diagnostic_output", "")
            )
            for d in choice_pool
        }
        breadth = {d: self._support_breadth(result_map[d]) for d in choice_pool}
        # Rank key: a disease whose agent satisfied at least one of its
        # own criteria/signals (has_support) is preferred FIRST -- a satisfied hard
        # criterion (e.g. pancreatitis lipase >=3x ULN) is a stronger near-miss than a
        # soft high-tier verdict with NO criterion met (e.g. "suspected appendicitis"
        # on RLQ tenderness alone). The binary has_support is cross-framework
        # commensurable (unlike raw breadth: chol caps at 3 streams, panc at 2 Atlanta
        # criteria); raw breadth only breaks ties among equally-supported diseases.
        # In practice this recovers pancreatitis cases where AP_C2 was met but a
        # soft appendicitis tier outranked it, with no loss elsewhere.
        winner = max(
            choice_pool,
            key=lambda d: (
                1 if breadth[d] >= 1 else 0,
                tiers[d],
                breadth[d],
                -order_idx.get(d, len(order_idx)),
            ),
        )
        return ResolutionResult(
            final_label=winner,
            winner=winner,
            decision_path=RES_FORCED_CHOICE,
            candidates=candidates,
            convictions=convictions,
            resolution_failed=False,
            trace={
                "reason": f"forced_choice (Mode B): {reason}",
                "forced_choice": True,
                "tiers": tiers,
                "support_breadth": breadth,
                "outputs": {
                    d: (result_map[d].final_verifier_result or {}).get("diagnostic_output", "")
                    for d in choice_pool
                },
            },
        )

    # ---- conviction ----

    def _conviction_tuple(self, result: AgentResult) -> tuple[int, int]:
        f = self._conviction_fields(result)
        return (f["label_strength"], f["support_breadth"])

    def _conviction_fields(self, result: AgentResult) -> dict[str, int]:
        vr = result.final_verifier_result or {}
        output = vr.get("diagnostic_output", "")
        label_strength = 0 if output in _PARTIAL_CONFIRM_OUTPUTS else 1
        return {
            "label_strength": label_strength,
            "support_breadth": self._support_breadth(result),
        }

    @staticmethod
    def _support_breadth(result: AgentResult) -> int:
        """Count of independent satisfied streams/signals/criteria (framework-specific)."""
        vr = result.final_verifier_result or {}
        ft = vr.get("framework_type", "")
        o = vr.get("primary_rule_outcome", {}) or {}

        if ft == "integrated_assessment":  # cholecystitis: streams C/L/I
            return sum(
                1 for k in ("stream_C_clinical", "stream_L_lab", "stream_I_imaging")
                if o.get(k) == "true"
            )
        if ft == "tailored_stepwise_risk_stratification":  # appendicitis: signals
            n = 0
            if o.get("signal_C_clinical") == "true":
                n += 1
            if o.get("signal_L_lab") == "true":
                n += 1
            if o.get("signal_I_imaging") in ("positive_uncomplicated", "positive_complicated"):
                n += 1
            if o.get("signal_S_score_band") == "high":
                n += 1
            if o.get("signal_X_complication") == "present":
                n += 1
            return n
        if ft == "multi_evidence_assessment":  # diverticulitis: CT features present
            return (
                len(o.get("complicated_findings_present", []) or [])
                + len(o.get("uncomplicated_findings_present", []) or [])
            )
        if ft == "n_of_m_rule":  # pancreatitis: satisfied criteria
            return int(o.get("n_satisfied", 0) or 0)
        return 0

    # ---- KB contradictory cross-reference ----

    def _kb_crossref_winner(self, tied: list[str]) -> str | None:
        """Resolve a 2-way conviction tie via KB contradictions (coarse).

        If disease X's KB lists a contradictory feature that names disease Y,
        the presence of Y argues against X -> Y is favoured. Resolves only when
        exactly one direction holds. Returns None otherwise (caller abstains).
        Only the 2-candidate case is resolved here; larger ties fall through to
        ``resolution_failed`` (conservative).
        """
        if len(tied) != 2:
            return None
        x, y = tied
        x_weakened_by_y = self._kb_names_rival(x, y)
        y_weakened_by_x = self._kb_names_rival(y, x)
        if x_weakened_by_y and not y_weakened_by_x:
            return y
        if y_weakened_by_x and not x_weakened_by_y:
            return x
        return None

    def _kb_names_rival(self, disease: str, rival: str) -> bool:
        """True iff `disease`'s KB contradictions text references `rival` by name."""
        try:
            kb = self.kb_registry.kb(disease)
        except Exception:  # noqa: BLE001
            return False
        blob = json.dumps(kb.contradictions).lower()
        return rival.lower() in blob

    # ---- helpers ----

    @staticmethod
    def _as_result_map(
        results: MultiAgentResult | dict[str, AgentResult],
    ) -> dict[str, AgentResult]:
        if isinstance(results, MultiAgentResult):
            return results.results
        return results
