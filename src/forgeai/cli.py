"""Typer CLI for ForgeAI Factory, Generic Chat Runner, and Regression Mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from forgeai.pipeline import run_baseline_comparison, run_factory, run_regression
from forgeai.runner import GenericAgentRunner
from forgeai.schemas import AgentConfig, RequirementsDoc

app = typer.Typer(help="ForgeAI: The Agent That Builds Agents")
console = Console()


@app.command()
def build(
    brief: str = typer.Option(..., "--brief", "-b", help="One-paragraph brief or path to a brief .txt/.md file."),
    docs_dir: str = typer.Option(..., "--docs", "-d", help="Path to the documents folder."),
    output_dir: str = typer.Option("./output", "--out", "-o", help="Directory for generated artifacts."),
    unattended: bool = typer.Option(False, "--unattended", help="Run without interactive prompts."),
    tournament: bool = typer.Option(False, "--tournament", help="Run Prompt Tournament across candidate prompts."),
    max_iterations: int = typer.Option(3, "--max-iters", help="Maximum critic-optimizer iterations."),
) -> None:
    """Run the ForgeAI factory end-to-end on a brief and document folder."""
    brief_path = Path(brief)
    if brief_path.exists() and brief_path.is_file():
        brief_text = brief_path.read_text(encoding="utf-8").strip()
    else:
        brief_text = brief

    def _cli_checkpoint(req: RequirementsDoc) -> RequirementsDoc:
        draft_path = Path(output_dir) / "requirements_draft.json"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
        console.print(Panel.fit(f"[bold yellow]Human Checkpoint[/bold yellow]\nDraft saved to: {draft_path}"))
        console.print(req.model_dump_json(indent=2))
        choice = console.input(
            "\nPress [bold]Enter[/bold] to approve, type [bold]r[/bold] after editing the draft file to reload, "
            "or [bold]q[/bold] to cancel: "
        ).strip().lower()
        if choice == "q":
            raise typer.Exit(code=1)
        if choice == "r":
            return RequirementsDoc.model_validate_json(draft_path.read_text(encoding="utf-8"))
        return req

    console.print(Panel.fit("[bold cyan]ForgeAI Factory[/bold cyan] — Building & Testing Agent"))
    final_config, scorecard, report_path = run_factory(
        brief=brief_text,
        docs_dir=docs_dir,
        output_dir=output_dir,
        unattended=unattended,
        tournament=tournament,
        max_iterations=max_iterations,
        checkpoint_fn=None if unattended else _cli_checkpoint,
        progress_fn=lambda msg: console.print(f"[green]•[/green] {msg}"),
    )
    console.print(
        f"\n[bold green]Done![/bold green] Agent: [cyan]{final_config.agent_name}[/cyan] | "
        f"Pass rate: [bold]{scorecard.final_pass_rate:.1%}[/bold] "
        f"({scorecard.passed_cases}/{scorecard.total_cases}) | Report: [underline]{report_path}[/underline]"
    )


@app.command()
def chat(
    config_path: str = typer.Option(..., "--config", "-c", help="Path to agent_config.json."),
    docs_dir: str = typer.Option(..., "--docs", "-d", help="Path to the documents folder."),
) -> None:
    """Interactive generic chat runner for any produced agent_config.json."""
    cfg = AgentConfig.model_validate_json(Path(config_path).read_text(encoding="utf-8"))
    runner = GenericAgentRunner(config=cfg, docs_dir=docs_dir)
    console.print(
        Panel.fit(
            f"[bold cyan]{cfg.agent_name}[/bold cyan]\n{cfg.description}\n"
            f"Tools: {[t.name for t in cfg.selected_tools]} | Type 'exit' to quit."
        )
    )

    history: list[dict[str, str]] = []
    while True:
        query = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        trace = runner.run(query, history=history[-6:])
        console.print("\n[bold green]Agent:[/bold green]")
        console.print(Markdown(trace.answer))
        if trace.tool_calls:
            console.print(f"[dim]Tools used: {[tc.tool_name for tc in trace.tool_calls]}[/dim]")
        console.print(f"[dim]Cited: {trace.cited_sources} | {trace.latency_ms:.0f}ms[/dim]")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": trace.answer})


@app.command()
def regress(
    docs_dir: str = typer.Option(..., "--docs", "-d", help="Updated documents folder."),
    artifacts_dir: str = typer.Option(..., "--artifacts", "-a", help="Existing factory output directory."),
) -> None:
    """Re-run existing eval_set.json after documents change."""
    console.print(Panel.fit("[bold cyan]ForgeAI Regression[/bold cyan] — Checking Policy/Corpus Drift"))
    scorecard = run_regression(
        docs_dir=docs_dir,
        artifacts_dir=artifacts_dir,
        progress_bar_fn=lambda pct, msg: console.print(f"[green][{int(pct * 100):3d}%][/green] {msg}"),
    )
    console.print(
        f"\n[bold]Regression Result:[/bold] {scorecard.final_pass_rate:.1%} "
        f"({scorecard.passed_cases}/{scorecard.total_cases} passed) | "
        f"Report: [underline]{Path(artifacts_dir) / 'regression_report.md'}[/underline]"
    )


@app.command()
def compare(
    baseline: str = typer.Option(..., "--baseline", "-b", help="Path to hand-written human baseline agent_config.json."),
    docs_dir: str = typer.Option(..., "--docs", "-d", help="Path to the documents folder."),
    artifacts_dir: str = typer.Option(..., "--artifacts", "-a", help="Existing factory output directory."),
) -> None:
    """Compare the factory-generated agent against a hand-written human baseline on the same eval_set.json."""
    console.print(Panel.fit("[bold cyan]ForgeAI Factory vs. Human Baseline[/bold cyan]"))
    factory_sc, baseline_sc, report_path = run_baseline_comparison(
        baseline_config_path=baseline,
        docs_dir=docs_dir,
        artifacts_dir=artifacts_dir,
        progress_bar_fn=lambda pct, msg: console.print(f"[green][{int(pct * 100):3d}%][/green] {msg}"),
    )
    console.print(
        f"\n[bold]Comparison Result:[/bold] Factory {factory_sc.final_pass_rate:.1%} "
        f"({factory_sc.passed_cases}/{factory_sc.total_cases}) vs. Human {baseline_sc.final_pass_rate:.1%} "
        f"({baseline_sc.passed_cases}/{baseline_sc.total_cases}) | "
        f"Report: [underline]{report_path}[/underline]"
    )


if __name__ == "__main__":
    app()
