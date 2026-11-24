"""Declarative Agent Generator with Dedicated Tool Selection Audit."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from forgeai.llm import LLMClient
from forgeai.schemas import (
    AgentConfig,
    CorpusManifest,
    RequirementsDoc,
    SystemPromptSections,
    ToolSelection,
)
from forgeai.tools import TOOL_DEFINITIONS, format_tool_catalog


class _PromptVariantList(BaseModel):
    variants: list[SystemPromptSections] = Field(min_length=2, max_length=3)


class _ToolDecision(BaseModel):
    name: Literal["csv_lookup", "calculator", "open_meteo_weather", "current_date"]
    needed: bool
    justification: str


class _ToolSelectionAudit(BaseModel):
    decisions: list[_ToolDecision] = Field(min_length=1, max_length=4)


def select_tools(
    requirements: RequirementsDoc,
    manifest: CorpusManifest,
    llm: LLMClient,
) -> list[ToolSelection]:
    """Evaluate every registered tool independently against brief, requirements, and manifest."""
    has_csv = any(d.file_type == "csv" for d in manifest.documents)
    csv_summary = [
        {"filename": d.filename, "columns": d.csv_columns, "row_count": d.csv_row_count}
        for d in manifest.documents
        if d.file_type == "csv"
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are ForgeAI's Tool Selection Auditor.\n"
                "Evaluate EVERY registered tool (`csv_lookup`, `calculator`, `open_meteo_weather`, `current_date`) "
                "independently against the Brief, Requirements, and Corpus Manifest.\n"
                "Rules for each tool:\n"
                "- `csv_lookup`: `needed=true` ONLY when the corpus contains `.csv` files AND the brief asks "
                "to look up or query structured records/codes/costs/inventories. Otherwise `needed=false`.\n"
                "- `calculator`: `needed=true` ONLY when the brief explicitly asks to calculate, compute, or "
                "estimate numeric totals, fees, discounts, or arithmetic. Otherwise `needed=false`.\n"
                "- `open_meteo_weather`: `needed=true` ONLY when the brief explicitly asks for live weather, "
                "weather conditions, or forecasts (and weather is not refused). Otherwise `needed=false`.\n"
                "- `current_date`: `needed=true` ONLY when the brief explicitly asks for today's/current date "
                "or checking dates/expirations relative to today. Otherwise `needed=false`.\n"
                "CRITICAL: BOTH attaching an unnecessary tool AND omitting a requested registered tool are factory defects.\n"
                "Return one decision entry per tool in `decisions` with a 1-sentence `justification`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brief:\n{requirements.brief}\n\n"
                f"Requirements:\n{requirements.model_dump_json(indent=2)}\n\n"
                f"CSV Files in Corpus: {json.dumps(csv_summary, indent=2) if csv_summary else 'None'}\n\n"
                f"Registered Tools:\n{format_tool_catalog()}"
            ),
        },
    ]

    audit = llm.structured_chat(messages, _ToolSelectionAudit, temperature=0.0)
    selected: list[ToolSelection] = []
    seen: set[str] = set()
    for decision in audit.decisions:
        if not decision.needed or decision.name in seen:
            continue
        if decision.name == "csv_lookup" and not has_csv:
            continue
        if decision.name in TOOL_DEFINITIONS:
            selected.append(ToolSelection(name=decision.name, justification=decision.justification))
            seen.add(decision.name)
    return selected


def generate_agent_config(
    requirements: RequirementsDoc,
    manifest: CorpusManifest,
    llm: LLMClient,
) -> AgentConfig:
    """Produce a complete declarative AgentConfig from requirements and corpus manifest."""
    selected = select_tools(requirements, manifest, llm)
    selected_json = json.dumps([t.model_dump() for t in selected], indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are ForgeAI's Agent Generator.\n"
                "Produce a complete declarative `AgentConfig` JSON object.\n"
                "Rules:\n"
                "1. `system_prompt` MUST be modular (`role`, `instructions`, `guardrails`, `refusal_policy`, `few_shot_examples`).\n"
                "2. `role` and `instructions` MUST NOT claim to support or handle any topic in `must_refuse_topics` or `unsupported_gaps`.\n"
                "3. `refusal_policy` MUST explicitly list the topics in `must_refuse_topics` / `unsupported_gaps` and state: "
                "\"For any request outside the documented scope and selected tools (including drafting emails, general ungrounded "
                "advice, or external actions like ticket submission), clearly and firmly refuse using explicit refusal language "
                "(e.g., 'I cannot assist with that request because it is outside my scope.'). Do NOT attempt to fulfill or "
                "partially answer out-of-scope requests, and never mention 'clarifying questions'.\"\n"
                "4. `few_shot_examples` MUST NOT demonstrate performing any refused or unsupported action.\n"
                "5. `selected_tools` MUST match the finalized Selected Tools list provided below. In `instructions` "
                "and `guardrails`, explain how to use those exact tools (e.g., look up records/coordinates via "
                "`csv_lookup`, pass coordinates to `open_meteo_weather`, or compute totals via `calculator`), and "
                "never tell the agent to refuse capabilities supported by those tools.\n"
                "6. `retrieval_config`: Choose sensible `chunk_size` (600-900), `chunk_overlap` (80-150), "
                "`top_k` (5-7), and `retrieval_mode` (`hybrid`).\n"
                "7. `output_contract`: Require citations `[Source: <filename>]`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirements:\n{requirements.model_dump_json(indent=2)}\n\n"
                f"Corpus Manifest:\n{manifest.model_dump_json(indent=2)}\n\n"
                f"Finalized Selected Tools:\n{selected_json}"
            ),
        },
    ]

    config = llm.structured_chat(messages, AgentConfig, temperature=0.1)
    # Authoritatively enforce the audited tool selection
    config.selected_tools = selected
    config.version = 1
    return config


def generate_prompt_candidates(
    base_config: AgentConfig,
    requirements: RequirementsDoc,
    llm: LLMClient,
    num_candidates: int = 3,
) -> list[AgentConfig]:
    """Stretch Goal: Propose 2-3 candidate SystemPromptSections for Prompt Tournament."""
    messages = [
        {
            "role": "system",
            "content": (
                f"Generate {num_candidates} distinct `SystemPromptSections` variants for the same agent:\n"
                "- Variant 1: Concise, strict extractive-grounding style.\n"
                "- Variant 2: Step-by-step verification & explicit citation-first style.\n"
                "- Variant 3: Boundary-hardened style emphasizing refusal clarity and ambiguity handling.\n"
                "Keep all variants faithful to the requirements."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requirements:\n{requirements.model_dump_json(indent=2)}\n\n"
                f"Base System Prompt:\n{base_config.system_prompt.model_dump_json(indent=2)}"
            ),
        },
    ]

    try:
        parsed = llm.structured_chat(messages, _PromptVariantList, temperature=0.3)
        candidates: list[AgentConfig] = []
        for idx, variant in enumerate(parsed.variants[:num_candidates], start=1):
            clone = base_config.model_copy(deep=True)
            clone.system_prompt = variant
            clone.version = idx
            candidates.append(clone)
        return candidates or [base_config]
    except Exception:
        return [base_config]
