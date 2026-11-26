"""Honest Quality Reporter."""

from __future__ import annotations

from collections import Counter
import re
from typing import Optional

from forgeai.schemas import (
    AgentConfig,
    CoverageMatrix,
    EvalSet,
    EvalValidationReport,
    RequirementsDoc,
    Scorecard,
    RequirementCoverage,
)


def build_requirement_coverage(
    requirements: RequirementsDoc, config: AgentConfig, eval_set: EvalSet
) -> CoverageMatrix:
    """Build a transparent keyword-overlap map; this is traceability support, not semantic proof."""
    items: list[tuple[str, str]] = []
    items.extend(("must_answer", topic) for topic in requirements.must_answer_topics)
    items.extend(("must_refuse", topic) for topic in requirements.must_refuse_topics)
    items.extend(("tool_capability", cap) for cap in requirements.tool_supported_capabilities)
    items.extend(("unsupported_gap", gap.requested_topic) for gap in requirements.unsupported_gaps)
    eval_index = {
        case.id: (
            case,
            set(re.findall(r"[a-z0-9]+", f"{case.question} {' '.join(case.expected_facts)}".lower())),
        )
        for case in eval_set.cases
    }
    rows: list[RequirementCoverage] = []
    for kind, topic in items:
        expect_refusal = kind in ("must_refuse", "unsupported_gap")
        terms = {t for t in re.findall(r"[a-z0-9]+", topic.lower()) if len(t) > 3}
        matches: list[str] = []
        doc_sources: set[str] = set()
        for cid, (case, case_terms) in eval_index.items():
            if case.should_refuse != expect_refusal:
                continue
            if terms and len(terms & case_terms) / len(terms) >= 0.34:
                matches.append(cid)
                doc_sources.update(case.expected_source_docs)
        tool_sources = [f"tool:{t.name}" for t in config.selected_tools if t.name in topic.lower()]
        evidence = sorted(doc_sources) + tool_sources
        rows.append(RequirementCoverage(
            topic=topic, requirement_type=kind,
            status="covered" if matches else "untested",
            matching_case_ids=matches, evidence_sources=evidence,
        ))
    return CoverageMatrix(requirements=rows)


def build_honest_report(
    requirements: RequirementsDoc,
    config: AgentConfig,
    eval_set: EvalSet,
    scorecard: Scorecard,
    tournament_notes: str = "",
    eval_validation: Optional[EvalValidationReport] = None,
    coverage_matrix: Optional[CoverageMatrix] = None,
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

    if coverage_matrix is not None:
        lines.extend(["", "## Requirement Coverage Map (Heuristic)",
                      "Coverage indicates keyword-linked eval cases, not semantic proof or a passing result.",
                      "| Requirement | Type | Status | Eval cases | Doc / Tool evidence |",
                      "| :--- | :--- | :---: | :--- | :--- |"])
        for row in coverage_matrix.requirements:
            lines.append(f"| {row.topic} | {row.requirement_type} | {row.status} | "
                         f"{', '.join(row.matching_case_ids) or '—'} | {', '.join(row.evidence_sources) or '—'} |")

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


def build_regression_report(
    baseline: Optional[Scorecard],
    regression: Scorecard,
    eval_set: EvalSet,
) -> str:
    """Generate a Markdown report comparing a regression run against baseline scorecard.json."""
    case_map = {c.id: c for c in eval_set.cases}
    lines: list[str] = [
        f"# ForgeAI Regression Report — {regression.agent_name}",
        "",
        "## 1. Summary",
    ]
    if baseline is not None:
        delta = (regression.final_pass_rate - baseline.final_pass_rate) * 100.0
        lines.append(f"- **Baseline Pass Rate:** {baseline.final_pass_rate:.1%} ({baseline.passed_cases}/{baseline.total_cases})")
        lines.append(
            f"- **Regression Pass Rate:** {regression.final_pass_rate:.1%} "
            f"({regression.passed_cases}/{regression.total_cases}) — **Delta: {delta:+.1f}%**"
        )
    else:
        lines.append(
            f"- **Regression Pass Rate:** {regression.final_pass_rate:.1%} "
            f"({regression.passed_cases}/{regression.total_cases})"
        )
    lines.append(
        f"- **Budget Used:** {regression.budget.tokens_used:,} tokens | {regression.budget.latency_seconds:.1f}s"
    )

    base_by_id = {r.case_id: r for r in baseline.final_results} if baseline else {}
    newly_broken = []
    newly_fixed = []
    still_failing = []

    for r in regression.final_results:
        prev = base_by_id.get(r.case_id)
        if not r.passed:
            if prev is not None and prev.passed:
                newly_broken.append(r)
            else:
                still_failing.append(r)
        elif prev is not None and not prev.passed:
            newly_fixed.append(r)

    lines.extend(["", f"## 2. Newly Broken Cases ({len(newly_broken)})"])
    if not newly_broken:
        lines.append("- None.")
    else:
        for r in newly_broken:
            q = case_map[r.case_id].question if r.case_id in case_map else ""
            lines.append(
                f"- **`{r.case_id}` ({r.category} → `{r.failure_category}`):** {q}\n"
                f"  - *What broke:* {r.failure_reason}"
            )

    lines.extend(["", f"## 3. Newly Passing Cases ({len(newly_fixed)})"])
    if not newly_fixed:
        lines.append("- None.")
    else:
        for r in newly_fixed:
            q = case_map[r.case_id].question if r.case_id in case_map else ""
            lines.append(f"- **`{r.case_id}` ({r.category}):** {q}")

    lines.extend(["", f"## 4. Still Failing Cases ({len(still_failing)})"])
    if not still_failing:
        lines.append("- None.")
    else:
        for r in still_failing:
            q = case_map[r.case_id].question if r.case_id in case_map else ""
            lines.append(
                f"- **`{r.case_id}` ({r.category} → `{r.failure_category}`):** {q}\n"
                f"  - *Reason:* {r.failure_reason}"
            )

    return "\n".join(lines) + "\n"


def build_baseline_comparison_report(
    factory_scorecard: Scorecard,
    baseline_scorecard: Scorecard,
    eval_set: EvalSet,
) -> str:
    """Generate an honest Factory vs. Human Baseline evaluation report."""
    case_map = {c.id: c for c in eval_set.cases}
    fac_by_id = {r.case_id: r for r in factory_scorecard.final_results}
    hum_by_id = {r.case_id: r for r in baseline_scorecard.final_results}

    factory_wins = []
    human_wins = []
    both_pass = []
    both_fail = []

    for c in eval_set.cases:
        fr = fac_by_id.get(c.id)
        hr = hum_by_id.get(c.id)
        if fr is None or hr is None:
            continue
        if fr.passed and not hr.passed:
            factory_wins.append((c, fr, hr))
        elif hr.passed and not fr.passed:
            human_wins.append((c, fr, hr))
        elif fr.passed and hr.passed:
            both_pass.append(c.id)
        else:
            both_fail.append((c, fr, hr))

    delta = (factory_scorecard.final_pass_rate - baseline_scorecard.final_pass_rate) * 100.0
    lines: list[str] = [
        f"# ForgeAI Factory vs. Human Baseline — {factory_scorecard.agent_name}",
        "",
        "## 1. Head-to-Head Summary",
        "| Metric | ForgeAI Factory | Human Baseline | Delta (Factory - Human) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Pass Rate** | **{factory_scorecard.final_pass_rate:.1%}** ({factory_scorecard.passed_cases}/{factory_scorecard.total_cases}) "
        f"| **{baseline_scorecard.final_pass_rate:.1%}** ({baseline_scorecard.passed_cases}/{baseline_scorecard.total_cases}) "
        f"| **{delta:+.1f}%** |",
        f"| **Both Pass / Both Fail** | {len(both_pass)} passed both | {len(both_fail)} failed both | "
        f"Factory wins: {len(factory_wins)} · Human wins: {len(human_wins)} |",
        "",
        f"## 2. Where ForgeAI Factory Beats Human Baseline ({len(factory_wins)})",
    ]
    if not factory_wins:
        lines.append("- None on this eval set.")
    else:
        for c, _fr, hr in factory_wins:
            lines.append(
                f"- **`{c.id}` ({c.category}):** {c.question}\n"
                f"  - *Why human baseline failed:* `{hr.failure_category}` — {hr.failure_reason}"
            )

    lines.extend(["", f"## 3. Where Human Baseline Beats ForgeAI Factory ({len(human_wins)})"])
    if not human_wins:
        lines.append("- None on this eval set.")
    else:
        for c, fr, _hr in human_wins:
            lines.append(
                f"- **`{c.id}` ({c.category}):** {c.question}\n"
                f"  - *Why factory failed:* `{fr.failure_category}` — {fr.failure_reason}"
            )

    lines.extend(["", f"## 4. Shared Failure Modes ({len(both_fail)})"])
    if not both_fail:
        lines.append("- None.")
    else:
        for c, fr, hr in both_fail:
            lines.append(
                f"- **`{c.id}` ({c.category}):** {c.question}\n"
                f"  - *Factory:* `{fr.failure_category}` ({fr.failure_reason})\n"
                f"  - *Human:* `{hr.failure_category}` ({hr.failure_reason})"
            )

    return "\n".join(lines) + "\n"
