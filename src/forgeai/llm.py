"""OpenAI-compatible LLM client with JSON repair/retry and budget tracking."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from forgeai.schemas import RunBudget

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised when a factory run exceeds its token or latency budget."""


def _extract_json_block(text: str) -> str:
    """Extract a JSON object or array from raw model output."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")
    starts = [i for i in (first_brace, first_bracket) if i != -1]
    if not starts:
        return text

    start = min(starts)
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    end = text.rfind(close_char)
    if end > start:
        return text[start : end + 1]
    return text[start:]


class LLMClient:
    """Thin, resilient wrapper around any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        budget: Optional[RunBudget] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("FORGEAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("FORGEAI_API_KEY", "ollama")
        self.model = model or os.getenv("FORGEAI_MODEL", "qwen2.5:7b-instruct")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.budget = budget or RunBudget(
            max_tokens=int(os.getenv("FORGEAI_MAX_TOKENS", "400000")),
            max_latency_seconds=float(os.getenv("FORGEAI_MAX_LATENCY_SECONDS", "1200")),
        )

    def check_budget(self) -> None:
        """Raise BudgetExceededError if token or latency limits are exceeded."""
        if self.budget.tokens_used >= self.budget.max_tokens:
            self.budget.stopped_early = True
            self.budget.stop_reason = (
                f"Token budget exceeded ({self.budget.tokens_used} >= {self.budget.max_tokens})"
            )
            raise BudgetExceededError(self.budget.stop_reason)
        if self.budget.latency_seconds >= self.budget.max_latency_seconds:
            self.budget.stopped_early = True
            self.budget.stop_reason = (
                f"Latency budget exceeded ({self.budget.latency_seconds:.1f}s >= "
                f"{self.budget.max_latency_seconds:.1f}s)"
            )
            raise BudgetExceededError(self.budget.stop_reason)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        tools: Optional[list[dict[str, Any]]] = None,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> tuple[Any, int, float]:
        """Execute a chat completion and return (message, tokens_used, latency_ms)."""
        self.check_budget()
        t0 = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                err_str = str(exc).lower()
                # Fallback if provider/model rejects response_format={"type": "json_object"}
                if json_mode and "response_format" in err_str:
                    kwargs.pop("response_format", None)
                    resp = self.client.chat.completions.create(**kwargs)
                    break
                is_transient = any(
                    tok in err_str
                    for tok in ("429", "rate limit", "too many requests", "timeout", "502", "503", "504", "connection")
                )
                if is_transient and attempt < 2:
                    time.sleep(1.5 * (2**attempt))
                    continue
                raise

        elapsed_s = time.perf_counter() - t0
        tokens = int(getattr(resp.usage, "total_tokens", 0) or 0)
        self.budget.tokens_used += tokens
        self.budget.latency_seconds += elapsed_s

        return resp.choices[0].message, tokens, elapsed_s * 1000.0

    def structured_chat(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[T],
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> T:
        """Call the LLM and parse/repair output into a validated Pydantic model."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = (
            "Respond ONLY with a valid JSON object matching this JSON Schema. "
            "Do not include markdown fences or commentary.\n"
            f"Schema:\n{schema_json}"
        )

        working_messages = list(messages)
        if working_messages and working_messages[0]["role"] == "system":
            working_messages[0] = {
                "role": "system",
                "content": working_messages[0]["content"] + "\n\n" + schema_instruction,
            }
        else:
            working_messages.insert(0, {"role": "system", "content": schema_instruction})

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            msg, _, _ = self.chat(
                working_messages,
                temperature=temperature,
                json_mode=True,
                max_tokens=2500,
            )
            raw = (msg.content or "").strip()
            candidate = _extract_json_block(raw)
            try:
                data = json.loads(candidate)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as err:
                last_error = err
                if attempt < max_retries:
                    working_messages.append({"role": "assistant", "content": raw})
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous JSON was invalid: {err}\n"
                                "Fix the JSON and return ONLY the valid JSON object matching the schema."
                            ),
                        }
                    )

        raise ValueError(
            f"Failed to parse {response_model.__name__} after {max_retries + 1} attempts: {last_error}"
        )
