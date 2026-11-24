"""Honest Quality Reporter."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from forgeai.schemas import (
    AgentConfig,
    EvalSet,
    EvalValidationReport,
    RequirementsDoc,
    Scorecard,
)


def build_honest_report(
    requirements: RequirementsDoc,
    config: AgentConfig,
    eval_set: EvalSet,
    scorecard: Scorecard,
    tournament_notes: str = "",
    eval_validation: Optional[EvalValidationReport] = None,
) -> str:
    """Generate an honest Markdown quality report explaining passes and remaining failures."""
    lines: list[str] = [
        f"# ForgeAI Quality Report — {config.agent_name}",
        "",
        "## 1. Executive Summary",
        f"- **Brief:** {requirements.brief}",
        f"- **Initial Pass Rate:** {scorecard.initial_pass_rate:.1%}",
        f"- **Final Pass Rate:** {scorecard.final_pass_rate:.1%} ({scorecard.passed_cases}/{scorecard.total_cases} cases passed)",
        f"- **Iterations Run:** {len(scorecard.iterations)}",
        f"- **Budget Used:** {scorecard.budget.tokens_used:,} tokens | {scorecard.budget.latency_seconds:.1f}s",
    ]
    if scorecard.budget.stopped_early:
        lines.append(f"- **Early Stop Reason:** {scorecard.budget.stop_reason}")

    lines.extend(["", "## 2. Groundability Guard Findings"])
    if requirements.tool_supported_capabilities:
        lines.append("- **Tool-Supported Capabilities:**")
        for cap in requirements.tool_supported_capabilities:
            lines.append(f"  - {cap}")
    if requirements.unsupported_gaps:
        for g in requirements.unsupported_gaps:
            lines.append(f"- **Unsupported:** `{g.requested_topic}` — {g.reason}")
    else:
        lines.append("- All remaining requested topics in the brief were groundable in the documents or registered tools.")

    lines.extend(["", "## 3. Tool Selection Audit"])
    if config.selected_tools:
        for t in config.selected_tools:
            lines.append(f"- **`{t.name}`**: {t.justification}")
    else:
        lines.append("- **No tools attached** (`[]`) — pure document-grounded QA.")

    if tournament_notes:
        lines.extend(["", "## 4. Prompt Tournament", tournament_notes])

    if eval_validation is not None:
        lines.extend(
            [
                "",
                "## 4b. Eval Grounding Validation Audit",
                f"- **Non-Refusal Cases Validated:** {eval_validation.cases_validated}/{eval_validation.total_non_refusal_cases}",
                f"- **Facts Checked / Kept / Dropped:** {eval_validation.total_facts_checked} / "
                f"{eval_validation.facts_kept} / {eval_validation.facts_dropped}",
                f"- **Cases Regenerated:** {eval_validation.cases_regenerated}",
            ]
        )

    lines.extend(["", "## 5. Critic–Optimizer Iteration Scorecard"])
    if scorecard.iterations:
        lines.append("| Iteration | Score Before | Score After | Dominant Failures | Changes Applied |")
        lines.append("| :--- | :---: | :---: | :--- | :--- |")
        for it in scorecard.iterations:
            changes_str = "; ".join(f"`{c.component}` ({c.reason})" for c in it.changes)
            lines.append(
                f"| {it.iteration} | {it.score_before:.1%} ({it.passed_before}/{it.total_cases}) "
                f"| {it.score_after:.1%} ({it.passed_after}/{it.total_cases}) "
                f"| {it.failure_breakdown} | {changes_str} |"
            )
    else:
        lines.append("- No optimization iterations were required or budget stopped early.")

    failed = [r for r in scorecard.final_results if not r.passed]
    case_map = {c.id: c for c in eval_set.cases}
    lines.extend(["", "## 6. Honest Failure Analysis (What Still Fails & Why)"])
    if not failed:
        lines.append(
            "- All generated eval cases passed in this run. Note: edge cases outside the sampled chunks "
            "or stricter latency constraints may still expose limitations of the small open model."
        )
    else:
        cat_counts = Counter(r.failure_category or "unknown" for r in failed)
        lines.append(f"- **Remaining Failure Breakdown:** {dict(cat_counts)}")
        lines.append("")
        for r in failed:
            q = case_map[r.case_id].question if r.case_id in case_map else ""
            lines.append(
                f"- **`{r.case_id}` ({r.category} → `{r.failure_category}`):** {q}\n"
                f"  - *Why it failed:* {r.failure_reason}"
            )

    return "\n".join(lines) + "\n"
