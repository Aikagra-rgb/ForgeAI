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
from forgeai.reporter import build_honest_report
from forgeai.schemas import AgentConfig, EvalSet, RequirementsDoc, Scorecard

_STANDARD_OUTPUT_FILES = (
    "corpus_manifest.json",
    "requirements.json",
    "agent_config.json",
    "eval_set.json",
    "eval_set_raw.json",
    "eval_generation_prompts.json",
    "eval_validation.json",
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
) -> tuple[AgentConfig, Scorecard, Path]:
    """Run the complete ForgeAI pipeline and persist all artifacts to output_dir."""
    log = progress_fn or (lambda msg: None)
    client = llm or LLMClient()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clean_stale_outputs(out)

    log("Stage 1/6: Profiling document corpus...")
    manifest = profile_corpus(docs_dir)
    _write_json(out / "corpus_manifest.json", manifest)

    log("Stage 2/6: Running Intake Interviewer & Tool-Aware Groundability Guard...")
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
    base_config = generate_agent_config(requirements, manifest, client)

    log("Stage 4/6: Generating and validating chunk-grounded evaluation set (>=20 cases)...")
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

    tournament_notes = ""
    if tournament:
        log("Stretch Goal: Running Prompt Tournament across candidate prompts...")
        candidates = generate_prompt_candidates(base_config, requirements, client, num_candidates=3)
        subset = EvalSet(agent_name=eval_set.agent_name, cases=eval_set.cases[:8])
        best_cfg = candidates[0]
        best_score = -1.0
        rows = []
        for idx, cand in enumerate(candidates, start=1):
            res = evaluate_agent(cand, subset, docs_dir, client)
            score = sum(1 for r in res if r.passed) / max(1, len(res))
            rows.append(f"- Candidate {idx}: {score:.1%} on {len(subset.cases)}-case tournament slice")
            if score > best_score:
                best_score = score
                best_cfg = cand
        base_config = best_cfg
        tournament_notes = "\n".join(rows)

    log("Stage 5/6: Running Critic-Optimizer loop (max 3 iterations)...")
    final_config, scorecard = run_optimization_loop(
        initial_config=base_config,
        eval_set=eval_set,
        docs_dir=docs_dir,
        llm=client,
        max_iterations=max_iterations,
    )
    _write_json(out / "agent_config.json", final_config)
    _write_json(out / "scorecard.json", scorecard)
    for it_log in scorecard.iterations:
        _write_json(out / f"iteration_{it_log.iteration}.json", it_log)

    log("Stage 6/6: Writing Honest Quality Report...")
    report_md = build_honest_report(
        requirements=requirements,
        config=final_config,
        eval_set=eval_set,
        scorecard=scorecard,
        tournament_notes=tournament_notes,
        eval_validation=validation_report,
    )
    report_path = out / "final_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    return final_config, scorecard, report_path


def run_regression(
    docs_dir: str,
    artifacts_dir: str,
    llm: Optional[LLMClient] = None,
) -> Scorecard:
    """Stretch Goal: Re-evaluate regenerated agent against existing eval_set.json after doc changes."""
    client = llm or LLMClient()
    art = Path(artifacts_dir)
    requirements = RequirementsDoc.model_validate_json((art / "requirements.json").read_text(encoding="utf-8"))
    eval_set = EvalSet.model_validate_json((art / "eval_set.json").read_text(encoding="utf-8"))

    manifest = profile_corpus(docs_dir)
    new_config = generate_agent_config(requirements, manifest, client)
    results = evaluate_agent(new_config, eval_set, docs_dir, client)

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
    _write_json(art / "regression_agent_config.json", new_config)
    _write_json(art / "regression_scorecard.json", scorecard)
    return scorecard
