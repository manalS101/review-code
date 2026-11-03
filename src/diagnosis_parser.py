"""Parser for the agent's per-iteration output.

Turns one model turn into a structured record the loop can act on. The format
follows Hager's diagnosis workflow parser with three deliberate differences.

A `Findings:` block sits between `Thought:` and `Action:`, carrying the
criterion-keyed findings for the observation in hand. Putting reasoning,
findings and the next action in a single turn lets the deterministic verifier
run between iterations without doubling the number of model calls.

There is no "Diagnostic Criteria" action. The KB criteria are already in the
system prompt, so a tool that returns them again would only burn iterations.

Tool input is not interpreted here. The parser returns `action_input` verbatim
and leaves panel expansion and modality matching to the dispatcher, which keeps
this module purely about format.

Expected shape, one of:

    Thought: <reasoning>
    Findings: { criterion -> finding, optional }
    Action: <Physical Examination | Laboratory Tests | Imaging>
    Action Input: <raw string>

or, terminally:

    Thought: <reasoning>
    Findings: { optional }
    Final Diagnosis: <condition>
    Treatment: <plan, optional>

A missing or unrecognised action with no final diagnosis raises. Malformed
findings JSON is not fatal: the delta is dropped, a warning is recorded, and
the turn still parses. The final diagnosis text is kept verbatim and is not
validated against KB vocabulary, since that mapping happens downstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# Result types

@dataclass
class AgentAction:
    """Non-terminal iteration: agent requests a tool call."""

    thought: str
    findings_delta: dict[str, Any]
    action: str                                 # canonicalized: one of VALID_ACTIONS
    action_input: str                           # raw string; ToolDispatcher parses further
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class AgentFinish:
    """Terminal iteration: agent commits to a final diagnosis."""

    thought: str
    findings_delta: dict[str, Any]
    final_diagnosis: str
    treatment: str | None
    parse_warnings: list[str] = field(default_factory=list)


class ParserError(ValueError):
    """Unrecoverable parse failure.

    The Agent's run loop catches this and may retry the LLM with a
    correction prompt up to 2 times. On the third failure, the case
    terminates with ``termination_reason="parser_error"``.
    """


# Section markers

# Order matters for the alternation regex: longer-prefix alternatives FIRST so
# "Action Input" wins over "Action" at the same position. Each marker is
# anchored at line start (MULTILINE) with optional leading whitespace.
_SECTION_NAMES = (
    "Action Input",
    "Final Diagnosis",
    "Diagnosis",        # tolerant alias for "Final Diagnosis"
    "Thought",
    "Findings",
    "Action",
    "Treatment",
)

# Build a single MULTILINE regex matching any section marker line.
_MARKER_REGEX = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(n) for n in _SECTION_NAMES) + r")[ \t]*:",
    flags=re.IGNORECASE | re.MULTILINE,
)


# Action canonicalization

VALID_ACTIONS: tuple[str, ...] = (
    "Physical Examination",
    "Laboratory Tests",
    "Imaging",
)

# Aliases observed in Hager's data + likely 8B-emitted variants. Mapped to the
# canonical action name. Lowercase keys; lookup is case-insensitive. Kept
# small, and extended only from observed output rather than from guesswork.
_ACTION_ALIASES: dict[str, str] = {
    "physical examination": "Physical Examination",
    "physical exam":        "Physical Examination",
    "examination":          "Physical Examination",
    "pe":                   "Physical Examination",

    "laboratory tests":     "Laboratory Tests",
    "lab tests":            "Laboratory Tests",
    "labs":                 "Laboratory Tests",
    "lab":                  "Laboratory Tests",
    "blood work":           "Laboratory Tests",
    "bloodwork":            "Laboratory Tests",

    "imaging":              "Imaging",
    "image":                "Imaging",
    "scan":                 "Imaging",
    "radiology":            "Imaging",
}


def _canonicalize_action(raw: str) -> str | None:
    """Return the canonical action name for ``raw`` or None if unrecognized."""
    s = raw.strip().lower()
    if not s:
        return None
    if s in _ACTION_ALIASES:
        return _ACTION_ALIASES[s]
    # Tolerate trailing punctuation / minor noise: try the first token + multi-word forms.
    s_no_trailing = re.sub(r"[\s\.\:;,!?]+$", "", s)
    if s_no_trailing in _ACTION_ALIASES:
        return _ACTION_ALIASES[s_no_trailing]
    return None


# DiagnosisParser

class DiagnosisParser:
    """Parse Agent LLM output into AgentAction | AgentFinish.

    Stateless across calls -- ``parse()`` returns a fresh result per invocation.
    Callers (the Agent) hold the conversation state in the Evidence EvidenceLedger,
    not in the parser.
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        if not isinstance(text, str):
            raise ParserError(f"parser input must be str, got {type(text).__name__}")
        if not text.strip():
            raise ParserError("parser input is empty / whitespace-only")

        sections = self._split_sections(text)
        warnings: list[str] = []

        thought = self._extract_thought(sections, warnings)
        findings_delta = self._extract_findings(sections, warnings)

        # Decide terminal vs non-terminal: presence of Final Diagnosis (or
        # the alias "Diagnosis") section signals AgentFinish. Final wins over
        # Action if both are present (the agent committed).
        if "final diagnosis" in sections or "diagnosis" in sections:
            return self._build_agent_finish(sections, thought, findings_delta, warnings)
        return self._build_agent_action(sections, thought, findings_delta, warnings)

    # ---- section discovery ----

    def _split_sections(self, text: str) -> dict[str, str]:
        """Partition ``text`` by line-start section markers.

        Returns a dict keyed by lowercased section name (e.g. "thought",
        "action input", "final diagnosis") with content = the text following
        the marker up to the next marker (or EOF). Whitespace around content
        is preserved EXCEPT for a single leading newline immediately after the
        colon, which is consumed for cleanliness.

        Repeated markers: the LAST occurrence wins (most recent block). We
        record a warning if a section appears more than once.
        """
        matches = list(_MARKER_REGEX.finditer(text))
        if not matches:
            raise ParserError(
                "no recognized section markers in LLM output. Expected at least "
                "'Thought:' and one of 'Action:' or 'Final Diagnosis:'"
            )
        # Build (start, end, name) ranges:
        bounds: list[tuple[int, int, str]] = []
        for i, m in enumerate(matches):
            # Content begins at the character after the colon.
            content_start = m.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            name = m.group(1).strip().lower()
            bounds.append((content_start, content_end, name))

        sections: dict[str, str] = {}
        duplicates: list[str] = []
        for cs, ce, name in bounds:
            content = text[cs:ce]
            # Strip exactly one leading newline / spaces on the colon's line.
            content = re.sub(r"^[ \t]*\n", "", content, count=1)
            content = content.rstrip()
            if name in sections:
                duplicates.append(name)
            sections[name] = content

        # If both "diagnosis" and "final diagnosis" appeared, merge: prefer the
        # explicit "final diagnosis" content.
        if "diagnosis" in sections and "final diagnosis" not in sections:
            sections["final diagnosis"] = sections["diagnosis"]
        # We never emit warnings for duplicate markers via the public API right
        # now (they're rare); the last-wins behaviour is documented above.
        _ = duplicates
        return sections

    # ---- extraction helpers ----

    def _extract_thought(self, sections: dict[str, str], warnings: list[str]) -> str:
        if "thought" not in sections:
            # Thought is REQUIRED per the format spec, but we tolerate absence
            # with a warning -- the Conviction metric will simply see an empty
            # string for this iteration. Better to surface the issue
            # downstream than to abort.
            warnings.append("missing 'Thought:' section -- thought set to empty string")
            return ""
        return sections["thought"].strip()

    def _extract_findings(
        self, sections: dict[str, str], warnings: list[str]
    ) -> dict[str, Any]:
        """Parse the Findings: block as JSON. Malformed JSON gives
        empty dict + warning (we never abort over Findings)."""
        if "findings" not in sections:
            return {}
        raw = sections["findings"].strip()
        if not raw:
            return {}

        # Trim text after the JSON object if the LLM appended commentary.
        json_text = _extract_first_json_object(raw)
        if json_text is None:
            warnings.append(
                "Findings: block was present but did not contain a "
                "balanced JSON object; findings_delta set to {}"
            )
            return {}
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            warnings.append(
                f"Findings: JSON did not parse ({e.msg} at char {e.pos}); "
                "findings_delta set to {}"
            )
            return {}
        if not isinstance(parsed, dict):
            warnings.append(
                f"Findings: parsed JSON is {type(parsed).__name__}, not a dict; "
                "findings_delta set to {}"
            )
            return {}
        return parsed

    # ---- builders ----

    def _build_agent_action(
        self,
        sections: dict[str, str],
        thought: str,
        findings_delta: dict[str, Any],
        warnings: list[str],
    ) -> AgentAction:
        if "action" not in sections:
            raise ParserError(
                "no 'Action:' section and no 'Final Diagnosis:' section -- "
                "Agent must request a tool or commit"
            )
        # Action name lives on a single line in well-formed output. We take
        # the first non-blank line of the Action: block as the action name
        # and tolerate trailing whitespace.
        raw_action = sections["action"].splitlines()[0].strip() if sections["action"] else ""
        canonical = _canonicalize_action(raw_action)
        if canonical is None:
            raise ParserError(
                f"unrecognized Action {raw_action!r}. "
                f"Valid actions: {list(VALID_ACTIONS)}"
            )

        # Action Input: optional in source text, but required by some tools.
        # The parser does NOT validate input content; the ToolDispatcher
        # decides whether the (action, input) pair is workable.
        action_input = sections.get("action input", "").strip()

        # Special case: Physical Examination ignores action_input per Hager
        # semantics ( line 154). If the LLM provided one anyway,
        # we keep it for audit but record a warning -- the tool will ignore it.
        if canonical == "Physical Examination" and action_input:
            warnings.append(
                "Physical Examination ignores action_input per Hager semantics; "
                "input preserved for audit only"
            )
        # For Labs/Imaging, an empty action_input is a soft error -- the tool
        # will return a "please specify which tests/modality" observation
        # rather than erroring. We just warn.
        if canonical in ("Laboratory Tests", "Imaging") and not action_input:
            warnings.append(
                f"{canonical} requested with empty Action Input; "
                "tool will return a no-op observation"
            )

        return AgentAction(
            thought=thought,
            findings_delta=findings_delta,
            action=canonical,
            action_input=action_input,
            parse_warnings=warnings,
        )

    def _build_agent_finish(
        self,
        sections: dict[str, str],
        thought: str,
        findings_delta: dict[str, Any],
        warnings: list[str],
    ) -> AgentFinish:
        diag_block = sections.get("final diagnosis", "")
        final_diagnosis = diag_block.splitlines()[0].strip() if diag_block else ""
        if not final_diagnosis:
            warnings.append(
                "'Final Diagnosis:' section was present but empty; "
                "downstream label mapping will receive empty string"
            )
        treatment = sections.get("treatment", "").strip() or None
        return AgentFinish(
            thought=thought,
            findings_delta=findings_delta,
            final_diagnosis=final_diagnosis,
            treatment=treatment,
            parse_warnings=warnings,
        )


# Module-level helpers

def _extract_first_json_object(text: str) -> str | None:
    """Return the substring corresponding to the first balanced ``{...}`` in
    ``text``, or None if no balanced object is found.

    Tracks brace depth while respecting string literals (so a ``{`` inside a
    JSON string doesn't bump the depth). Returns None if a ``{`` is opened
    but never closed.
    """
    i = text.find("{")
    if i < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    j = i
    while j < len(text):
        ch = text[j]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
        j += 1
    return None
