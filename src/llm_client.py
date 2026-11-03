"""Model client.

`LLMClient` is the abstraction the rest of the framework depends on, and
`VLLMClient` is its only backend. It talks to a vLLM server you run yourself,
over the chat-completions protocol vLLM serves. That protocol is conventionally
called OpenAI-compatible, but the name refers to the wire format alone: no
request leaves the machine or network you point `base_url` at.

All calls are made at temperature 0. Transient connection failures are retried
with backoff, which is safe because an identical request at temperature 0
returns an identical completion.
"""

from __future__ import annotations

import abc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """One completion. Token counts and latency are filled in where the
    backend reports them."""

    text: str                                  # raw model output
    finish_reason: str                         # "stop" | "length" | "error" | "stub" | ...
    model_id: str                              # e.g., "stub", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    prompt_tokens: int | None = None           # for cost accounting
    completion_tokens: int | None = None
    latency_ms: float | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised on unrecoverable LLM errors (backend failure, no canned response, ...)."""


class LLMClient(abc.ABC):
    """Abstract LLM client. Subclasses implement .complete()."""

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        json_only: bool = True,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Send `prompt` (optionally with `system_prompt`) and return the response.

        Args:
            prompt: user-turn content.
            max_tokens: generation cap.
            temperature: sampling temperature (0.0 = greedy).
            json_only: if True, ask the server to constrain decoding to valid
                JSON. Reduces parsing failures.
            system_prompt: system-turn content; backend-specific handling.
        """
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        """Backends override to return their model identifier."""
        return getattr(self, "_model_id", "unknown")


# vLLM backend.
#
#   client = VLLMClient(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
#                       base_url="http://localhost:8000/v1")
#   client.health_check()
#   reply = client.complete("Hello", max_tokens=50)
#
# vLLM does not enforce API keys, so any non-empty string satisfies the openai
# package.


def _import_openai():
    """Lazy import openai so the rest of the module works without it installed."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise LLMError(
            "VLLMClient requires the openai package: pip install 'openai>=1.0'"
        ) from e
    return OpenAI


def _openai_exc_types() -> tuple:
    """Transient exception types worth retrying (connection drops, timeouts,
    server hiccups). A dropped SSH tunnel to a self-hosted vLLM box surfaces as
    APIConnectionError / APITimeoutError; these are safe to retry because at
    temperature 0 the re-sent request is byte-identical and yields the same
    output (determinism-preserving). NON-transient errors (e.g. BadRequestError
    on a context overflow) are deliberately absent so they fail fast."""
    from openai import (  # type: ignore
        APIConnectionError, APITimeoutError, InternalServerError, RateLimitError,
    )
    return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class VLLMClient(LLMClient):
    """Client for a self-hosted vLLM endpoint.

    Args:
        model_id: name sent in the request's `model` field. Must match the
            `--served-model-name` vLLM was started with.
        base_url: endpoint address. The default assumes vLLM on the same
            machine, or reached through a forwarded port.
        api_key: vLLM ignores this, so any non-empty string will do.
        timeout_s: per-request timeout. The default of 120 allows for a 70B
            model on a record that fills the context.
        max_retries: transport-level retries.
        served_model_name: optional override if your vLLM server uses a
            different served-model-name. Defaults to model_id.
    """

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        served_model_name: str | None = None,
        connect_retries: int = 5,
        connect_backoff_s: float = 3.0,
        connect_backoff_max_s: float = 30.0,
    ) -> None:
        OpenAI = _import_openai()
        # Resolve API key:
        #   1. explicit arg
        #   2. OPENAI_API_KEY env var (works for vLLM too -- it just ignores it)
        #   3. fallback "EMPTY" (vLLM accepts any non-empty string)
        key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self._client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        self._model_id = model_id
        self._served_model_name = served_model_name or model_id
        self._base_url = base_url
        # Outer retry loop for TRANSIENT connection failures (e.g. a briefly
        # dropped SSH tunnel to a self-hosted vLLM box). Longer backoff than the
        # openai client's built-in max_retries, so a tunnel down for up to ~a
        # minute is ridden out instead of killing the agent mid-run.
        self._connect_retries = int(connect_retries)
        self._connect_backoff_s = float(connect_backoff_s)
        self._connect_backoff_max_s = float(connect_backoff_max_s)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _with_conn_retry(self, fn, what: str):
        """Call `fn`, retrying on transient connection errors with exponential
        backoff. Wraps a non-transient or exhausted failure in LLMError
        (preserving prior behaviour). Determinism-safe: the retried request is
        byte-identical, so a successful retry returns what an un-dropped call
        would have."""
        retry_types = _openai_exc_types()
        last_exc: Exception | None = None
        for attempt in range(self._connect_retries + 1):
            try:
                return fn()
            except retry_types as e:
                last_exc = e
                if attempt >= self._connect_retries:
                    break
                delay = min(self._connect_backoff_s * (2 ** attempt),
                            self._connect_backoff_max_s)
                print(f"[llm-retry] {what}: {type(e).__name__} "
                      f"(attempt {attempt + 1}/{self._connect_retries + 1}); "
                      f"retrying in {delay:.0f}s (base_url={self._base_url})",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
            except Exception as e:  # noqa: BLE001 -- non-transient: fail fast
                raise LLMError(
                    f"vLLM {what} failed (base_url={self._base_url}, "
                    f"model={self._served_model_name!r}): {type(e).__name__}: {e}"
                ) from e
        raise LLMError(
            f"vLLM {what} failed after {self._connect_retries + 1} attempts "
            f"(base_url={self._base_url}, model={self._served_model_name!r}): "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        json_only: bool = True,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the vLLM endpoint.

        When `json_only=True`, requests `response_format={"type": "json_object"}`,
        which vLLM honours by constraining decoding to valid JSON.

        Raises:
            LLMError on connection / API errors.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self._served_model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if json_only:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        completion = self._with_conn_retry(
            lambda: self._client.chat.completions.create(**kwargs), "request")
        latency_ms = (time.monotonic() - t0) * 1000.0

        try:
            choice = completion.choices[0]
        except (AttributeError, IndexError) as e:
            raise LLMError(
                f"vLLM returned no choices in response: {completion!r}"
            ) from e
        text = (choice.message.content or "") if choice.message else ""
        finish_reason = choice.finish_reason or "stop"

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        return LLMResponse(
            text=text,
            finish_reason=finish_reason,
            model_id=self._model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw_metadata={
                "served_model_name": self._served_model_name,
                "base_url": self._base_url,
                "completion_id": getattr(completion, "id", None),
                "created": getattr(completion, "created", None),
            },
        )

    def health_check(self) -> bool:
        """Quick check that the vLLM endpoint is reachable + serving the model.

        Returns True on success. Raises LLMError on failure.
        """
        models = self._with_conn_retry(
            lambda: self._client.models.list(), "health-check")
        ids = {m.id for m in models.data}
        if self._served_model_name not in ids and not any(
            self._served_model_name in mid for mid in ids
        ):
            raise LLMError(
                f"vLLM endpoint at {self._base_url} is up but does not serve "
                f"{self._served_model_name!r}. Available: {sorted(ids)}"
            )
        return True
