"""Hybrid Deterministic + Decomposed Binary LLM Judge."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from forgeai.llm import LLMClient
from forgeai.schemas import AgentTrace, EvalCase, FactCheck, FailureCategory, JudgeResult

_REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bunable to\b",
    r"\bout of scope\b",
    r"\boutside (?:my|the) scope\b",
    r"\bnot covered\b",
    r"\bnot supported\b",
    r"\bdo not have (?:enough )?information\b",
    r"\bdoes not contain\b",
    r"\bno information available\b",
    r"\bnot mentioned in the (?:provided )?documents\b",
    r"\bcannot answer\b",
]


def _detect_refusal_deterministic(answer: str) -> bool:
    lower = answer.lower()
    return any(re.search(pat, lower) for pat in _REFUSAL_PATTERNS)


class _JudgeRubricOutput(BaseModel):
    refused: bool
    fact_checks: list[FactCheck] = Field(default_factory=list)
    grounded: bool
    ungrounded_claim: str = ""


def judge_case(
    case: EvalCase,
    trace: AgentTrace,
    llm: LLMClient,
) -> JudgeResult:
    """Grade a single EvalCase using deterministic checks + decomposed binary LLM verification."""
    retrieved_files = {c.filename for c in trace.retrieved_chunks}
    if case.expected_source_docs:
        retrieval_hit = any(doc in retrieved_files for doc in case.expected_source_docs)
    else:
        retrieval_hit = True

    citation_present = len(trace.cited_sources) > 0
    det_refused = _detect_refusal_deterministic(trace.answer)

    context_summary = "\n\n".join(
        f"[{c.filename} | {c.page_or_row}] {c.text}" for c in trace.retrieved_chunks
    )
    tool_summary = "\n".join(
        f"[Tool {tc.tool_name}({tc.arguments})] -> {tc.result}" for tc in trace.tool_calls
    )

    rubric = llm.structured_chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an impartial QA Judge. Grade ONLY using binary true/false checks.\n"
                    "Do NOT reward verbosity or style. Ignore answer length.\n"
                    "1. `refused`: Did the answer refuse to answer or state the topic is unsupported/out-of-scope?\n"
                    "2. `fact_checks`: For EACH expected fact, set `present: true` ONLY if the answer explicitly "
                    "states that fact accurately.\n"
                    "3. `grounded`: Set `true` by default if the answer is based on the Retrieved Context or Tool Outputs "
                    "(paraphrases, direct deductions, polite refusals, and stating 'not specified' all count as `true`). "
                    "Omissions or missing facts are NOT hallucinations—grade those in `fact_checks`, not `grounded`. "
                    "Set `grounded: false` ONLY if the answer invents a specific fabricated fact, number, or rule with no basis in the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {case.question}\n\n"
                    f"Retrieved Context:\n{context_summary or '(none)'}\n\n"
                    f"Tool Outputs:\n{tool_summary or '(none)'}\n\n"
                    f"Agent Answer:\n{trace.answer}\n\n"
                    f"Expected Facts to Verify:\n"
                    + "\n".join(f"- {f}" for f in case.expected_facts)
                ),
            },
        ],
        response_model=_JudgeRubricOutput,
        temperature=0.0,
    )

    refused = det_refused or rubric.refused
    refusal_correct = refused == case.should_refuse

    if case.should_refuse:
        fact_score = 1.0 if refusal_correct else 0.0
    elif rubric.fact_checks:
        hits = sum(1 for fc in rubric.fact_checks if fc.present)
        fact_score = hits / len(rubric.fact_checks)
    else:
        fact_score = 1.0

    # Determine pass/fail and classify failure into the 4 canonical buckets
    failure_category: FailureCategory | None = None
    failure_reason = ""

    if case.should_refuse:
        passed = refusal_correct
        if not refusal_correct:
            failure_category = "wrong_refusal"
            failure_reason = "Agent answered an out-of-scope/unsupported question instead of refusing."
    else:
        if not retrieval_hit:
            passed = False
            failure_category = "retrieval_miss"
            failure_reason = (
                f"Expected source {case.expected_source_docs} not in retrieved top-k ({sorted(retrieved_files)})."
            )
        elif not refusal_correct:
            passed = False
            failure_category = "wrong_refusal"
            failure_reason = "Agent wrongly refused an in-scope, answerable question."
        elif not rubric.grounded:
            passed = False
            failure_category = "hallucination"
            failure_reason = rubric.ungrounded_claim or "Answer contains claims not supported by retrieved context."
        elif fact_score < (0.5 if case.category == "edge_ambiguous" else 0.65):
            passed = False
            failure_category = "prompt_defect"
            missing = [fc.fact for fc in rubric.fact_checks if not fc.present]
            failure_reason = f"Missed expected facts ({fact_score:.0%}): {missing}"
        else:
            passed = True

    return JudgeResult(
        case_id=case.id,
        category=case.category,
        passed=passed,
        retrieval_hit=retrieval_hit,
        citation_present=citation_present,
        refused=refused,
        refusal_correct=refusal_correct,
        fact_checks=rubric.fact_checks,
        fact_score=fact_score,
        grounded=rubric.grounded,
        failure_category=failure_category,
        failure_reason=failure_reason,
    )
