"""Targeted Critic-Optimizer Loop (max 3 iterations) with budget enforcement."""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Optional

from pydantic import BaseModel

from forgeai.judge import judge_case
from forgeai.llm import BudgetExceededError, LLMClient
from forgeai.runner import GenericAgentRunner
from forgeai.schemas import (
    AgentConfig,
    EvalSet,
    IterationDiff,
    IterationLog,
    JudgeResult,
    Scorecard,
)


class _PromptSectionPatch(BaseModel):
    updated_text: str
    reason: str


def evaluate_agent(
    config: AgentConfig,
    eval_set: EvalSet,
    docs_dir: str,
    llm: LLMClient,
    on_case: Optional[Callable[[int, int, str], None]] = None,
) -> list[JudgeResult]:
    """Run GenericAgentRunner on every EvalCase and grade with the Hybrid Judge."""
    runner = GenericAgentRunner(config=config, docs_dir=docs_dir, llm=llm)
    results: list[JudgeResult] = []
    total = len(eval_set.cases)
    for idx, case in enumerate(eval_set.cases, start=1):
        if on_case is not None:
            on_case(idx, total, case.id)
        trace = runner.run(case.question)
        result = judge_case(case, trace, llm)
        results.append(result)
    return results


def _pass_rate(results: list[JudgeResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def _apply_targeted_patch(
    config: AgentConfig,
    dominant_category: str,
    failed_results: list[JudgeResult],
    eval_set: EvalSet,
    llm: LLMClient,
) -> tuple[AgentConfig, list[IterationDiff]]:
    """Mutate ONLY the targeted component for the dominant failure category."""
    updated = config.model_copy(deep=True)
    updated.version += 1
    diffs: list[IterationDiff] = []

    case_map = {c.id: c for c in eval_set.cases}
    examples_text = "\n".join(
        f"- [{r.case_id}] Q: {case_map[r.case_id].question} | Reason: {r.failure_reason}"
        for r in failed_results[:5]
        if r.case_id in case_map
    )

    if dominant_category == "retrieval_miss":
        before_cfg = updated.retrieval_config.model_dump_json()
        new_k = min(updated.retrieval_config.top_k + 2, 10)
        updated.retrieval_config.top_k = new_k
        updated.retrieval_config.retrieval_mode = "hybrid"
        if updated.retrieval_config.chunk_size < 700:
            updated.retrieval_config.chunk_size = min(updated.retrieval_config.chunk_size + 150, 900)
        after_cfg = updated.retrieval_config.model_dump_json()
        diffs.append(
            IterationDiff(
                component="retrieval_config",
                before=before_cfg,
                after=after_cfg,
                reason=f"Increased top_k to {new_k} and ensured hybrid retrieval to fix retrieval misses.",
            )
        )
        return updated, diffs

    section_map = {
        "hallucination": "guardrails",
        "wrong_refusal": "refusal_policy",
        "prompt_defect": "instructions",
    }
    target_section = section_map.get(dominant_category, "instructions")
    before_text = getattr(updated.system_prompt, target_section)

    patch = llm.structured_chat(
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are ForgeAI's Critic-Optimizer. Patch ONLY the `{target_section}` section "
                    "of the system prompt to fix the listed failures. Keep it concise (no prompt bloat). "
                    "Your `updated_text` MUST meaningfully modify the current text; never return it unchanged."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Failure Category: {dominant_category}\n\n"
                    f"Current `{target_section}`:\n{before_text}\n\n"
                    f"Sample Failures:\n{examples_text}"
                ),
            },
        ],
        response_model=_PromptSectionPatch,
        temperature=0.1,
    )

    updated_text = patch.updated_text.strip()
    patch_reason = patch.reason
    if not updated_text or updated_text == before_text.strip():
        fallback_additions = {
            "wrong_refusal": (
                " Always answer in-scope questions directly whenever relevant facts appear in the retrieved "
                "documents or registered tools; refuse only genuinely out-of-scope or unsupported requests."
            ),
            "hallucination": (
                " Never invent numbers, dates, or policy rules not explicitly present in the retrieved context or tool outputs."
            ),
            "prompt_defect": (
                " Include every relevant numeric threshold, limit, and exception clause verbatim from the retrieved context."
            ),
        }
        addition = fallback_additions.get(
            dominant_category,
            " Address all aspects of the user's question directly using verbatim facts from the retrieved context.",
        )
        updated_text = (before_text.strip() + addition).strip()
        patch_reason = (
            f"Applied targeted `{dominant_category}` guardrail reinforcement because the optimizer returned an unchanged section."
        )

    setattr(updated.system_prompt, target_section, updated_text)
    diffs.append(
        IterationDiff(
            component=f"system_prompt.{target_section}",
            before=before_text,
            after=updated_text,
            reason=patch_reason,
        )
    )
    return updated, diffs


def run_optimization_loop(
    initial_config: AgentConfig,
    eval_set: EvalSet,
    docs_dir: str,
    llm: LLMClient,
    max_iterations: int = 3,
    target_pass_rate: float = 0.90,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> tuple[AgentConfig, Scorecard]:
    """Run up to max_iterations of targeted critic-optimizer improvements."""
    emit = progress_cb or (lambda pct, msg: None)
    current_config = initial_config
    current_results = evaluate_agent(
        current_config,
        eval_set,
        docs_dir,
        llm,
        on_case=lambda i, n, cid: emit(
            0.35 * (i / max(1, n)),
            f"Baseline evaluation: case {i}/{n} ({cid})",
        ),
    )
    initial_rate = _pass_rate(current_results)
    emit(0.38, f"Baseline evaluation finished: {initial_rate:.1%} pass rate")
    iterations: list[IterationLog] = []
    skipped_categories: set[str] = set()

    for it in range(1, max_iterations + 1):
        score_before = _pass_rate(current_results)
        passed_before = sum(1 for r in current_results if r.passed)
        failed = [r for r in current_results if not r.passed]

        if not failed or score_before >= target_pass_rate:
            break

        counts = Counter(r.failure_category or "prompt_defect" for r in failed)
        ranked_cats = [cat for cat, _ in counts.most_common() if cat not in skipped_categories]
        if not ranked_cats:
            break
        dominant_cat = ranked_cats[0]
        addressed_ids = [r.case_id for r in failed if (r.failure_category or "prompt_defect") == dominant_cat]

        t0 = time.perf_counter()
        tok_before = llm.budget.tokens_used
        slot_start = 0.38 + (it - 1) * (0.62 / max(1, max_iterations))
        slot_span = 0.62 / max(1, max_iterations)
        emit(slot_start, f"Iteration {it}/{max_iterations}: patching `{dominant_cat}` failures...")

        try:
            candidate_config, diffs = _apply_targeted_patch(
                current_config, dominant_cat, [r for r in failed if r.case_id in addressed_ids], eval_set, llm
            )
            candidate_results = evaluate_agent(
                candidate_config,
                eval_set,
                docs_dir,
                llm,
                on_case=lambda i, n, cid, _s=slot_start, _sp=slot_span, _it=it: emit(
                    _s + _sp * (0.15 + 0.85 * (i / max(1, n))),
                    f"Iteration {_it}/{max_iterations} re-evaluation: case {i}/{n} ({cid})",
                ),
            )
        except BudgetExceededError:
            break

        score_after = _pass_rate(candidate_results)
        passed_after = sum(1 for r in candidate_results if r.passed)
        tok_used = llm.budget.tokens_used - tok_before
        lat_ms = (time.perf_counter() - t0) * 1000.0

        # Roll back if the patch degraded performance
        if score_after < score_before:
            diffs.append(
                IterationDiff(
                    component="rollback",
                    before=f"v{candidate_config.version}",
                    after=f"v{current_config.version}",
                    reason=f"Rolled back patch because pass rate dropped ({score_before:.1%} -> {score_after:.1%}).",
                )
            )
            iterations.append(
                IterationLog(
                    iteration=it,
                    score_before=score_before,
                    score_after=score_before,
                    passed_before=passed_before,
                    passed_after=passed_before,
                    total_cases=len(eval_set.cases),
                    failure_breakdown=dict(counts),
                    addressed_case_ids=addressed_ids,
                    changes=diffs,
                    tokens_used=tok_used,
                    latency_ms=lat_ms,
                )
            )
            skipped_categories.add(dominant_cat)
            continue

        iterations.append(
            IterationLog(
                iteration=it,
                score_before=score_before,
                score_after=score_after,
                passed_before=passed_before,
                passed_after=passed_after,
                total_cases=len(eval_set.cases),
                failure_breakdown=dict(counts),
                addressed_case_ids=addressed_ids,
                changes=diffs,
                tokens_used=tok_used,
                latency_ms=lat_ms,
            )
        )
        current_config = candidate_config
        current_results = candidate_results

        # Early stop if plateaued
        if score_after == score_before:
            break

    final_rate = _pass_rate(current_results)
    passed_final = sum(1 for r in current_results if r.passed)
    scorecard = Scorecard(
        agent_name=current_config.agent_name,
        total_cases=len(eval_set.cases),
        initial_pass_rate=initial_rate,
        final_pass_rate=final_rate,
        passed_cases=passed_final,
        failed_cases=len(eval_set.cases) - passed_final,
        iterations=iterations,
        final_results=current_results,
        budget=llm.budget,
    )
    return current_config, scorecard
