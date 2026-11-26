"""Automated end-to-end unattended factory test using a fixture corpus and MockLLMClient."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from forgeai.llm import LLMClient
from forgeai.pipeline import run_factory
from forgeai.runner import GenericAgentRunner
from forgeai.schemas import (
    AgentConfig,
    EvalValidationReport,
    FactCheck,
    FactValidationResult,
    OutputContract,
    RequirementsDoc,
    RetrievalConfig,
    RunBudget,
    Scorecard,
    SystemPromptSections,
    ToolSelection,
    UnsupportedGap,
)

T = TypeVar("T", bound=BaseModel)


class MockLLMClient(LLMClient):
    """Deterministic offline LLMClient mock exercising the full ForgeAI pipeline."""

    def __init__(self) -> None:
        self.base_url = "mock://local"
        self.api_key = "mock"
        self.model = "mock-7b"
        self.client = None  # type: ignore[assignment]
        self.budget = RunBudget(max_tokens=500000, max_latency_seconds=600.0)
        self._single_qa_calls = 0
        self._tool_round_seen = False

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        tools: Optional[list[dict[str, Any]]] = None,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> tuple[Any, int, float]:
        self.check_budget()
        self.budget.tokens_used += 150
        self.budget.latency_seconds += 0.05

        sys_text = messages[0]["content"] if messages else ""
        last_msg = messages[-1]

        # If a tool result just returned, produce final answer using it
        if last_msg.get("role") == "tool":
            return (
                SimpleNamespace(
                    content=f"Calculated result based on tool output: {last_msg['content']} [Source: ports_fixture.csv]",
                    tool_calls=None,
                ),
                120,
                15.0,
            )

        user_text = str(last_msg.get("content", ""))

        # Exercise real tool execution when runner receives an arithmetic query with tools enabled
        if tools and "500 * 2" in user_text and not self._tool_round_seen:
            self._tool_round_seen = True
            tc = SimpleNamespace(
                id="call_calc_1",
                function=SimpleNamespace(
                    name="calculator",
                    arguments=json.dumps({"expression": "500 * 2"}),
                ),
            )
            return SimpleNamespace(content="", tool_calls=[tc]), 100, 10.0

        q_match = re.search(r"User Question:\s*(.*?)(?:\n\nAnswer strictly|\Z)", user_text, re.DOTALL)
        question = q_match.group(1).strip() if q_match else user_text

        # Refusal behavior
        if any(tok in question.lower() for tok in ("book", "reserve", "berth", "visa", "tax")):
            return (
                SimpleNamespace(
                    content="I cannot answer or perform booking/external requests as that is outside my scope.",
                    tool_calls=None,
                ),
                100,
                10.0,
            )

        enhanced = "Include exact fee amounts" in sys_text
        if "Port Alpha" in question and not enhanced:
            # Initial iteration misses the exact 500 USD fee so optimizer fixes prompt_defect
            ans = "Port Alpha is located in Testland with draft 10 meters. [Source: ports_fixture.csv]"
        else:
            # Grounded response citing retrieved context lines
            ctx_match = re.search(r"Retrieved Context:\s*(.*?)\n\nUser Question:", user_text, re.DOTALL)
            ctx = ctx_match.group(1) if ctx_match else ""
            ans = (
                f"Based on the provided records: {ctx[:900]}\n"
                "Summary: Port Alpha fixed fee is 500 USD (draft 10m), Port Beta fixed fee is 750 USD (draft 12m), "
                "and policy requires 24 hours advance notice. [Source: ports_fixture.csv] [Source: ops_policy.md]"
            )
        return SimpleNamespace(content=ans, tool_calls=None), 140, 12.0

    def structured_chat(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[T],
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> T:
        self.check_budget()
        self.budget.tokens_used += 100
        self.budget.latency_seconds += 0.02
        name = response_model.__name__
        user_text = str(messages[-1].get("content", ""))

        if name in ("IntakeProbe", "_IntakeProbe"):
            return response_model.model_validate(
                {
                    "clarifying_questions": ["Should costs be shown in USD?"],
                    "tool_supported_capabilities": [
                        "CSV lookup via csv_lookup",
                        "Cost calculation via calculator",
                        "Live weather via open_meteo_weather",
                    ],
                    "unsupported_gaps": [
                        {
                            "requested_topic": "Automated berth booking",
                            "reason": "No berth reservation API or document support exists.",
                        }
                    ],
                }
            )

        if name == "RequirementsDoc":
            return response_model.model_validate(
                {
                    "brief": "Fixture brief",
                    "agent_name": "FixtureMaritimeAgent",
                    "scope": "Port lookup, cost estimation, operational policy, and live port weather.",
                    "audience": "Vessel planners",
                    "tone": "Concise and factual",
                    "must_answer_topics": ["port fees", "draft limits", "weather", "notice policy"],
                    "must_refuse_topics": ["Automated berth booking"],
                    "tool_supported_capabilities": [
                        "CSV lookup via csv_lookup",
                        "Cost calculation via calculator",
                        "Live weather via open_meteo_weather",
                    ],
                    "unsupported_gaps": [
                        {
                            "requested_topic": "Automated berth booking",
                            "reason": "No berth reservation API or document support exists.",
                        }
                    ],
                    "clarifying_qa": [],
                }
            )

        if name == "_ToolSelectionAudit":
            return response_model.model_validate(
                {
                    "decisions": [
                        {"name": "csv_lookup", "needed": True, "justification": "Required to look up port CSV rows."},
                        {"name": "calculator", "needed": True, "justification": "Required to calculate order/port totals."},
                        {"name": "open_meteo_weather", "needed": True, "justification": "Required for live port weather."},
                        {"name": "current_date", "needed": False, "justification": "Current date is not requested."},
                    ]
                }
            )

        if name == "AgentConfig":
            return AgentConfig(
                agent_name="FixtureMaritimeAgent",
                version=1,
                description="Fixture maritime assistant.",
                system_prompt=SystemPromptSections(
                    role="Port operations assistant.",
                    instructions="Answer using retrieved documents and selected tools.",
                    guardrails="Cite sources and never fabricate values.",
                    refusal_policy="Refuse berth booking and external out-of-scope requests.",
                    few_shot_examples=[],
                ),
                retrieval_config=RetrievalConfig(chunk_size=600, chunk_overlap=80, top_k=5, retrieval_mode="hybrid"),
                selected_tools=[
                    ToolSelection(name="csv_lookup", justification="CSV lookup"),
                    ToolSelection(name="calculator", justification="Calculator"),
                    ToolSelection(name="open_meteo_weather", justification="Weather"),
                ],
                output_contract=OutputContract(format="markdown", require_citations=True),
            )  # type: ignore[return-value]

        if name == "_SingleQA":
            self._single_qa_calls += 1
            if self._single_qa_calls == 1:
                # Intentionally inject 1 unsupported fact to verify validate_eval_set drops it
                return response_model.model_validate(
                    {
                        "question": "What is the fixed port fee and draft limit for Port Alpha?",
                        "expected_facts": [
                            "Port Alpha fixed fee is 500 USD",
                            "Port Alpha provides free Sunday tugboat escort",
                            "who approves the port fee",
                        ],
                        "rationale": "Includes one ungrounded fact and one non-answering fragment for validation test.",
                    }
                )
            if "Chunk A" in user_text:
                return response_model.model_validate(
                    {
                        "question": "What is Port Beta's fixed fee and the required advance notice period?",
                        "expected_facts": [
                            "Port Beta fixed fee is 750 USD",
                            "24 hours advance notice",
                        ],
                        "rationale": "Multi-doc test",
                    }
                )
            return response_model.model_validate(
                {
                    "question": "What are the draft limit and policy rules for Port Alpha?",
                    "expected_facts": ["Port Alpha draft 10m", "24 hours advance notice"],
                    "rationale": "Chunk-grounded test",
                }
            )

        if name == "_RefusalList":
            return response_model.model_validate(
                {
                    "questions": [
                        "Can you book a berth at Port Alpha for tomorrow?",
                        "Please reserve berth 4 for my vessel.",
                        "Can you file my maritime tax return?",
                        "Reserve a pilot boat for Port Beta.",
                        "Can you book a dry dock slot?",
                    ]
                }
            )

        if name == "_RefusalValidityBatch":
            q_part = user_text.split("Candidate Refusal Questions:\n", 1)[-1]
            results = []
            for line in q_part.splitlines():
                clean_q = re.sub(r"^\d+\.\s*", "", line).strip()
                if not clean_q:
                    continue
                is_oos = any(tok in clean_q.lower() for tok in ("book", "reserve", "tax"))
                results.append({"question": clean_q, "truly_out_of_scope": is_oos, "reason": ""})
            return response_model.model_validate({"results": results})

        if name == "_FactSupportBatch":
            facts_part = user_text.split("Candidate Facts:\n", 1)[-1]
            results: list[FactValidationResult] = []
            for line in facts_part.splitlines():
                clean_fact = re.sub(r"^\d+\.\s*", "", line).strip()
                if not clean_fact:
                    continue
                is_supported = "tugboat" not in clean_fact.lower()
                answers_q = not clean_fact.lower().startswith("who ")
                results.append(
                    FactValidationResult(
                        fact=clean_fact,
                        supported=is_supported,
                        answers_question=answers_q,
                        evidence_quote=clean_fact if (is_supported and answers_q) else "",
                    )
                )
            return response_model.model_validate({"results": results})

        if name == "_JudgeRubricOutput":
            ans_match = re.search(r"Agent Answer:\s*(.*?)\n\nExpected Facts", user_text, re.DOTALL)
            ans_text = ans_match.group(1) if ans_match else ""
            facts_part = user_text.split("Expected Facts to Verify:\n", 1)[-1]
            fact_checks: list[FactCheck] = []
            for line in facts_part.splitlines():
                fact = line.lstrip("- ").strip()
                if not fact:
                    continue
                # Check if key numbers/terms from fact appear in answer
                tokens = [t.lower() for t in re.findall(r"[a-z0-9]+", fact) if len(t) >= 3]
                present = all(tok in ans_text.lower() for tok in tokens[:3]) if tokens else True
                fact_checks.append(FactCheck(fact=fact, present=present, evidence=ans_text[:80]))
            refused = "outside my scope" in ans_text.lower() or "cannot" in ans_text.lower()
            return response_model.model_validate(
                {
                    "refused": refused,
                    "fact_checks": fact_checks,
                    "grounded": True,
                    "ungrounded_claim": "",
                }
            )

        if name == "_PromptSectionPatch":
            return response_model.model_validate(
                {
                    "updated_text": "Include exact fee amounts, draft limits, and notice rules from context and tools.",
                    "reason": "Fixes missed fee details on port queries.",
                }
            )

        raise ValueError(f"Unhandled mock response model: {name}")


def test_e2e_unattended_factory_contract(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "ports_fixture.csv").write_text(
        "port\tcountry\tlat\tlon\tdraft_m\tfixed_fee_usd\n"
        "Port Alpha\tTestland\t10.5\t20.5\t10\t500\n"
        "Port Beta\tTestland\t11.5\t21.5\t12\t750\n",
        encoding="utf-8",
    )
    (docs_dir / "ops_policy.md").write_text(
        "# Port Operations Policy\nVessels must submit arrival notices 24 hours advance notice before entry.\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    # Pre-populate a stale iteration file to verify cleanup
    (out_dir / "iteration_99.json").write_text("{}", encoding="utf-8")

    brief = (
        "I need a port operations assistant to look up port fees and draft limits from the CSV, "
        "calculate total port costs, check live port weather, and perform automated berth booking."
    )
    mock_llm = MockLLMClient()

    final_config, scorecard, report_path = run_factory(
        brief=brief,
        docs_dir=str(docs_dir),
        output_dir=str(out_dir),
        unattended=True,
        max_iterations=3,
        min_eval_cases=20,
        llm=mock_llm,
    )

    expected_files = [
        "corpus_manifest.json",
        "requirements.json",
        "agent_config.json",
        "eval_set.json",
        "eval_generation_prompts.json",
        "eval_validation.json",
        "scorecard.json",
        "final_report.md",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing artifact: {fname}"

    assert not (out_dir / "iteration_99.json").exists(), "Stale iteration_99.json was not cleaned"
    iter_files = list(out_dir.glob("iteration_*.json"))
    assert len(iter_files) == len(scorecard.iterations)

    selected_names = {t.name for t in final_config.selected_tools}
    assert selected_names == {"csv_lookup", "calculator", "open_meteo_weather"}

    req = RequirementsDoc.model_validate_json((out_dir / "requirements.json").read_text(encoding="utf-8"))
    assert any("booking" in g.requested_topic.lower() for g in req.unsupported_gaps)
    assert not any("weather" in g.requested_topic.lower() for g in req.unsupported_gaps)

    val_report = EvalValidationReport.model_validate_json((out_dir / "eval_validation.json").read_text(encoding="utf-8"))
    assert val_report.facts_dropped >= 2
    assert all(d.kept_fact_count >= 1 for d in val_report.case_details)

    loaded_scorecard = Scorecard.model_validate_json((out_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert loaded_scorecard.total_cases >= 20
    assert loaded_scorecard.final_pass_rate > loaded_scorecard.initial_pass_rate
    assert report_path.exists()

    # Verify GenericAgentRunner executes produced config including a tool call
    runner = GenericAgentRunner(config=final_config, docs_dir=str(docs_dir), llm=mock_llm)
    trace = runner.run("Calculate 500 * 2 for Port Alpha")
    assert trace.tool_calls and trace.tool_calls[0].tool_name == "calculator"
    assert trace.tool_calls[0].result == "1000"
