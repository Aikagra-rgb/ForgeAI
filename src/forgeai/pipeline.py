"""End-to-end ForgeAI Factory Pipeline & Regression Runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from forgeai.eval_gen import GENERATION_PROMPTS, generate_and_validate_eval_set
from forgeai.generator import generate_agent_config, generate_prompt_candidates
from forgeai.ingestion import profile_corpus
from forgeai.interviewer import run_intake_interview
from forgeai.llm import LLMClient
from forgeai.optimizer import evaluate_agent, run_optimization_loop
from forgeai.reporter import (
    build_requirement_coverage,
    build_baseline_comparison_report,
    build_honest_report,
    build_regression_report,
)
from forgeai.schemas import AgentConfig, EvalSet, RequirementsDoc, Scorecard

_STANDARD_OUTPUT_FILES = (
    "corpus_manifest.json",
    "requirements.json",
    "agent_config.json",
    "eval_set.json",
    "eval_set_raw.json",
    "eval_generation_prompts.json",
    "eval_validation.json",
    "coverage_matrix.json",
    "scorecard.json",
    "final_report.md",
)


def _clean_stale_outputs(out: Path) -> None:
    """Remove stale iteration logs and standard factory outputs before a fresh run."""
    if not out.exists():
        return
    for stale_iter in out.glob("iteration_*.json"):
        stale_iter.unlink(missing_ok=True)
    for name in _STANDARD_OUTPUT_FILES:
        (out / name).unlink(missing_ok=True)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump_json"):
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _select_stratified_tournament_slice(eval_set: EvalSet, limit: int = 8) -> EvalSet:
    """Pick a balanced tournament subset across answerable, refusal, multi-doc, and ambiguous cases."""
    quotas = (
        ("answerable", 3),
        ("refusal", 2),
        ("edge_multi_doc", 2),
        ("edge_ambiguous", 1),
    )
    selected: list = []
    seen_ids: set[str] = set()
    for category, quota in quotas:
        count = 0
        for case in eval_set.cases:
            if case.category == category and case.id not in seen_ids:
                selected.append(case)
                seen_ids.add(case.id)
                count += 1
                if count >= quota:
                    break
    for case in eval_set.cases:
        if len(selected) >= limit:
            break
        if case.id not in seen_ids:
            selected.append(case)
            seen_ids.add(case.id)
    return EvalSet(agent_name=eval_set.agent_name, cases=selected[:limit])


def run_factory(
    brief: str,
    docs_dir: str,
    output_dir: str,
    unattended: bool = False,
    tournament: bool = False,
    max_iterations: int = 3,
    min_eval_cases: int = 20,
    llm: Optional[LLMClient] = None,
    answer_fn: Optional[Callable[[str, int, int], str]] = None,
    checkpoint_fn: Optional[Callable[[RequirementsDoc], RequirementsDoc]] = None,
    requirements_override: Optional[RequirementsDoc] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    progress_bar_fn: Optional[Callable[[float, str], None]] = None,
) -> tuple[AgentConfig, Scorecard, Path]:
    """Run the complete ForgeAI pipeline and persist all artifacts to output_dir."""
    log = progress_fn or (lambda msg: None)
    bar = progress_bar_fn or (lambda pct, msg: None)
    client = llm or LLMClient()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clean_stale_outputs(out)

    log("Stage 1/6: Profiling document corpus...")
    bar(0.05, "Stage 1/6: Profiling document corpus...")
    manifest = profile_corpus(docs_dir)
    _write_json(out / "corpus_manifest.json", manifest)

    log("Stage 2/6: Running Intake Interviewer & Tool-Aware Groundability Guard...")
    bar(0.12, "Stage 2/6: Finalizing requirements & groundability checks...")
    requirements = requirements_override or run_intake_interview(
        brief=brief,
        manifest=manifest,
        llm=client,
        unattended=unattended,
        answer_fn=answer_fn,
    )
    if requirements_override is None and checkpoint_fn is not None and not unattended:
        requirements = checkpoint_fn(requirements)
    _write_json(out / "requirements.json", requirements)

    log("Stage 3/6: Auditing tools and generating declarative AgentConfig...")
    bar(0.20, "Stage 3/6: Auditing tools & generating AgentConfig...")
    base_config = generate_agent_config(requirements, manifest, client)

    log("Stage 4/6: Generating and validating chunk-grounded evaluation set (>=20 cases)...")
    bar(0.30, "Stage 4/6: Generating & validating >=20 chunk-grounded eval cases...")
    eval_set, validation_report, raw_eval_set = generate_and_validate_eval_set(
        requirements=requirements,
        docs_dir=docs_dir,
        llm=client,
        min_cases=min_eval_cases,
    )
    _write_json(out / "eval_set_raw.json", raw_eval_set)
    _write_json(out / "eval_set.json", eval_set)
    _write_json(out / "eval_generation_prompts.json", GENERATION_PROMPTS)
    _write_json(out / "eval_validation.json", validation_report)
    coverage_matrix = build_requirement_coverage(requirements, base_config, eval_set)
    _write_json(out / "coverage_matrix.json", coverage_matrix)

    tournament_notes = ""
    opt_start = 0.42
    if tournament:
        log("Stretch Goal: Running Prompt Tournament across candidate prompts...")
        bar(0.42, "Prompt Tournament: generating 3 candidate prompts...")
        candidates = generate_prompt_candidates(base_config, requirements, client, num_candidates=3)
        subset = _select_stratified_tournament_slice(eval_set, limit=8)
        best_cfg = candidates[0]
        best_score = -1.0
        rows = []
        total_cands = max(1, len(candidates))
        for idx, cand in enumerate(candidates, start=1):
            c_start = 0.44 + (idx - 1) * (0.14 / total_cands)
            c_span = 0.14 / total_cands
            res = evaluate_agent(
                cand,
                subset,
                docs_dir,
                client,
                on_case=lambda i, n, cid, _cs=c_start, _csp=c_span, _idx=idx: bar(
                    _cs + _csp * (i / max(1, n)),
                    f"Prompt Tournament — Candidate {_idx}/{total_cands}: case {i}/{n} ({cid})",
                ),
            )
            score = sum(1 for r in res if r.passed) / max(1, len(res))
            rows.append(f"- Candidate {idx}: {score:.1%} on {len(subset.cases)}-case tournament slice")
            if score > best_score:
                best_score = score
                best_cfg = cand
        base_config = best_cfg
        tournament_notes = "\n".join(rows)
        opt_start = 0.58

    log("Stage 5/6: Running Critic-Optimizer loop (max 3 iterations)...")
    opt_span = 0.95 - opt_start
    final_config, scorecard = run_optimization_loop(
        initial_config=base_config,
        eval_set=eval_set,
        docs_dir=docs_dir,
        llm=client,
        max_iterations=max_iterations,
        progress_cb=lambda frac, msg: bar(
            min(0.95, opt_start + opt_span * frac),
            f"Stage 5/6: {msg}",
        ),
    )
    _write_json(out / "agent_config.json", final_config)
    _write_json(out / "scorecard.json", scorecard)
    for it_log in scorecard.iterations:
        _write_json(out / f"iteration_{it_log.iteration}.json", it_log)

    log("Stage 6/6: Writing Honest Quality Report...")
    bar(0.97, "Stage 6/6: Writing Honest Quality Report...")
    report_md = build_honest_report(
        requirements=requirements,
        config=final_config,
        eval_set=eval_set,
        scorecard=scorecard,
        tournament_notes=tournament_notes,
        eval_validation=validation_report,
        coverage_matrix=coverage_matrix,
    )
    report_path = out / "final_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    bar(1.0, "Build complete!")

    return final_config, scorecard, report_path


def run_regression(
    docs_dir: str,
    artifacts_dir: str,
    llm: Optional[LLMClient] = None,
    progress_bar_fn: Optional[Callable[[float, str], None]] = None,
) -> Scorecard:
    """Stretch Goal: Re-evaluate regenerated agent against existing eval_set.json after doc changes."""
    bar = progress_bar_fn or (lambda pct, msg: None)
    client = llm or LLMClient()
    art = Path(artifacts_dir)
    bar(0.08, "Loading existing requirements & saved eval set...")
    requirements = RequirementsDoc.model_validate_json((art / "requirements.json").read_text(encoding="utf-8"))
    eval_set = EvalSet.model_validate_json((art / "eval_set.json").read_text(encoding="utf-8"))

    bar(0.18, "Profiling updated document corpus...")
    manifest = profile_corpus(docs_dir)
    bar(0.30, "Regenerating AgentConfig for updated corpus...")
    new_config = generate_agent_config(requirements, manifest, client)
    results = evaluate_agent(
        new_config,
        eval_set,
        docs_dir,
        client,
        on_case=lambda i, n, cid: bar(
            0.35 + 0.60 * (i / max(1, n)),
            f"Regression eval: case {i}/{n} ({cid})",
        ),
    )

    passed = sum(1 for r in results if r.passed)
    rate = passed / max(1, len(results))
    scorecard = Scorecard(
        agent_name=new_config.agent_name,
        total_cases=len(results),
        initial_pass_rate=rate,
        final_pass_rate=rate,
        passed_cases=passed,
        failed_cases=len(results) - passed,
        iterations=[],
        final_results=results,
        budget=client.budget,
    )
    baseline_sc: Optional[Scorecard] = None
    baseline_path = art / "scorecard.json"
    if baseline_path.exists():
        try:
            baseline_sc = Scorecard.model_validate_json(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            baseline_sc = None

    _write_json(art / "regression_agent_config.json", new_config)
    _write_json(art / "regression_scorecard.json", scorecard)
    reg_report = build_regression_report(baseline=baseline_sc, regression=scorecard, eval_set=eval_set)
    (art / "regression_report.md").write_text(reg_report, encoding="utf-8")
    bar(1.0, f"Regression complete: {rate:.1%} ({passed}/{len(results)} passed)")
    return scorecard


def run_baseline_comparison(
    baseline_config_path: str,
    docs_dir: str,
    artifacts_dir: str,
    llm: Optional[LLMClient] = None,
    progress_bar_fn: Optional[Callable[[float, str], None]] = None,
) -> tuple[Scorecard, Scorecard, Path]:
    """Stretch Goal: Evaluate a hand-written human baseline config on the exact same eval_set.json."""
    bar = progress_bar_fn or (lambda pct, msg: None)
    client = llm or LLMClient()
    art = Path(artifacts_dir)
    bar(0.05, "Loading saved eval set and factory scorecard...")
    eval_set = EvalSet.model_validate_json((art / "eval_set.json").read_text(encoding="utf-8"))
    baseline_cfg = AgentConfig.model_validate_json(Path(baseline_config_path).read_text(encoding="utf-8"))

    factory_scorecard_path = art / "scorecard.json"
    if factory_scorecard_path.exists():
        factory_sc = Scorecard.model_validate_json(factory_scorecard_path.read_text(encoding="utf-8"))
    else:
        factory_cfg = AgentConfig.model_validate_json((art / "agent_config.json").read_text(encoding="utf-8"))
        fac_results = evaluate_agent(factory_cfg, eval_set, docs_dir, client)
        fac_passed = sum(1 for r in fac_results if r.passed)
        fac_rate = fac_passed / max(1, len(fac_results))
        factory_sc = Scorecard(
            agent_name=factory_cfg.agent_name,
            total_cases=len(fac_results),
            initial_pass_rate=fac_rate,
            final_pass_rate=fac_rate,
            passed_cases=fac_passed,
            failed_cases=len(fac_results) - fac_passed,
            iterations=[],
            final_results=fac_results,
            budget=client.budget,
        )

    bar(0.15, f"Evaluating human baseline ({baseline_cfg.agent_name}) on {len(eval_set.cases)} cases...")
    hum_results = evaluate_agent(
        baseline_cfg,
        eval_set,
        docs_dir,
        client,
        on_case=lambda i, n, cid: bar(
            0.15 + 0.80 * (i / max(1, n)),
            f"Human baseline eval: case {i}/{n} ({cid})",
        ),
    )
    hum_passed = sum(1 for r in hum_results if r.passed)
    hum_rate = hum_passed / max(1, len(hum_results))
    baseline_sc = Scorecard(
        agent_name=baseline_cfg.agent_name,
        total_cases=len(hum_results),
        initial_pass_rate=hum_rate,
        final_pass_rate=hum_rate,
        passed_cases=hum_passed,
        failed_cases=len(hum_results) - hum_passed,
        iterations=[],
        final_results=hum_results,
        budget=client.budget,
    )

    _write_json(art / "baseline_scorecard.json", baseline_sc)
    report_md = build_baseline_comparison_report(factory_sc, baseline_sc, eval_set)
    report_path = art / "baseline_comparison.md"
    report_path.write_text(report_md, encoding="utf-8")
    bar(
        1.0,
        f"Comparison complete — Factory {factory_sc.final_pass_rate:.1%} vs Human {baseline_sc.final_pass_rate:.1%}",
    )
    return factory_sc, baseline_sc, report_path
