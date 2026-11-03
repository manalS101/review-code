"""Phase 1: multi-agent evidence gathering.

Runs the four disease-specific agents over one case. Each agent works the record
independently and never sees another agent's evidence. Execution is sequential
against a single vLLM endpoint, which keeps a run reproducible; concurrent
batching on the GPU is what breaks bit-level determinism.

No verifier semantics are added here. Each agent's verdict is exactly what its
own run produces. This module collects the four verdicts and records which
agents confirmed their own condition, giving Phase 2 between zero and four
confirmed candidates.

    MultiAgentRunner(llm_client=..., kb_registry=..., adapter=...).run(hadm_id)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.adapter import Adapter
from src.agents.appendicitis_agent import AppendicitisAgent
from src.agents.cholecystitis_agent import CholecystitisAgent
from src.agents.disease_agent_base import AgentResult, DiseaseAgentBase
from src.agents.diverticulitis_agent import DiverticulitisAgent
from src.agents.pancreatitis_agent import PancreatitisAgent
from src.kb_loader import KBRegistry
from src.llm_client import LLMClient


# The four disease-specific agent classes, keyed by knowledge base id.
AGENT_CLASSES: dict[str, type[DiseaseAgentBase]] = {
    "appendicitis":   AppendicitisAgent,
    "cholecystitis":  CholecystitisAgent,
    "diverticulitis": DiverticulitisAgent,
    "pancreatitis":   PancreatitisAgent,
}

# Deterministic execution order (alphabetical by disease). Fixed so repeated
# runs are reproducible and the candidate list ordering is stable.
DEFAULT_ORDER: tuple[str, ...] = (
    "appendicitis", "cholecystitis", "diverticulitis", "pancreatitis",
)


# Result type

@dataclass
class MultiAgentResult:
    """All four Agents' results for one case, plus the candidate list."""

    hadm_id: int
    results: dict[str, AgentResult]   # keyed by disease (agent_id)
    candidates: list[str]                # diseases whose outcome_label == disease
    agent_order: list[str]

    def labels(self) -> dict[str, str]:
        """Convenience: {disease: final five-outcome label}."""
        return {d: r.outcome_label for d, r in self.results.items()}


# Runner

class MultiAgentRunner:
    """Runs all four disease-specific agents on one case and collects their results.

    All four agents share one ``llm_client``, ``kb_registry`` and
    ``adapter`` but otherwise run fully independently (no inter-agent
    communication; the ledger is the only shared surface). Constructor kwargs mirror
    ``DiseaseAgentBase`` and are forwarded to every agent.
    """

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
        enable_span_verifier: bool = True,
        enable_anatomical_anchor: bool = True,
        auto_commit_stable_verdict: int | None = 2,
        enable_commit_nudge: bool = True,
        force_final_diagnosis: bool = False,
        suppress_kb_output_menu: bool = False,
        order: tuple[str, ...] = DEFAULT_ORDER,
    ) -> None:
        unknown = [d for d in order if d not in AGENT_CLASSES]
        if unknown:
            raise ValueError(f"unknown agent id(s) in order: {unknown}")
        self.order = list(order)
        self.log_dir_root = Path(log_dir) if log_dir is not None else None
        self._agents: dict[str, DiseaseAgentBase] = {}
        for disease in self.order:
            cls = AGENT_CLASSES[disease]
            self._agents[disease] = cls(
                llm_client=llm_client,
                kb_registry=kb_registry,
                adapter=adapter,
                max_iterations=max_iterations,
                log_dir=log_dir,
                lab_mapping_csv=lab_mapping_csv,
                max_parser_retries=max_parser_retries,
                enable_span_verifier=enable_span_verifier,
                enable_anatomical_anchor=enable_anatomical_anchor,
                auto_commit_stable_verdict=auto_commit_stable_verdict,
                enable_commit_nudge=enable_commit_nudge,
                force_final_diagnosis=force_final_diagnosis,
                suppress_kb_output_menu=suppress_kb_output_menu,
            )

    def run(self, hadm_id: int) -> MultiAgentResult:
        hadm_id = int(hadm_id)
        results: dict[str, AgentResult] = {}
        for disease in self.order:
            results[disease] = self._agents[disease].run(hadm_id)

        candidates = [d for d in self.order if results[d].outcome_label == d]

        result = MultiAgentResult(
            hadm_id=hadm_id,
            results=results,
            candidates=candidates,
            agent_order=list(self.order),
        )
        self._persist_summary(hadm_id, result)
        return result

    # ---- persistence ----

    def _persist_summary(self, hadm_id: int, result: MultiAgentResult) -> None:
        if self.log_dir_root is None:
            return
        d = self.log_dir_root / f"{hadm_id}_phase1"
        d.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            "hadm_id":         hadm_id,
            "agent_order":  result.agent_order,
            "candidates":      result.candidates,
            "per_agent": {
                disease: {
                    "outcome_label":        r.outcome_label,
                    "diagnostic_output":   r.final_verifier_result.get("diagnostic_output"),
                    "termination_reason":  r.termination_reason,
                    "final_diagnosis_text": r.final_diagnosis_text,
                    "total_llm_calls":     r.total_llm_calls,
                    "total_tool_calls":    r.total_tool_calls,
                }
                for disease, r in result.results.items()
            },
        }
        _atomic_write_json(d / "summary.json", summary)


# Helpers

def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomic write of a JSON file (matches the base-agent persistence pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
