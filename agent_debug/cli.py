"""CLI for agent-debug: analyze traces and scan prompts for risk."""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app = typer.Typer(
    name="agent-debug",
    help="Diagnose why your AI agent failed. Root cause analysis + fix suggestions.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ─── analyze command ─────────────────────────────────────────────────────────

@app.command()
def analyze(
    trace_file: Annotated[Path, typer.Argument(help="Path to trace JSON file")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write JSON report to this file"),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print raw JSON report to stdout")
    ] = False,
) -> None:
    """Analyze a failed agent trace. Outputs root cause + fix suggestions."""
    if not trace_file.exists():
        err_console.print(f"[red]Error:[/red] File not found: {trace_file}")
        raise typer.Exit(1)

    raw_text = trace_file.read_text()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        err_console.print(f"[red]Error:[/red] Invalid JSON in {trace_file}: {e}")
        raise typer.Exit(1)

    console.print(f"[dim]Analyzing trace from {trace_file.name}...[/dim]")

    try:
        from agent_debug.pipeline import DiagnosisPipeline
        pipeline = DiagnosisPipeline()
        report = pipeline.run(raw)
    except ValueError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(report, indent=2))
        return

    if output:
        output.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Report saved to {output}[/green]")

    _print_report(report)


# ─── scan command ─────────────────────────────────────────────────────────────

@app.command()
def scan(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to JSON file with system_prompt and tool_definitions"
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write JSON report to this file"),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print raw JSON report to stdout")
    ] = False,
) -> None:
    """Scan a system prompt + tool definitions for failure risk before deployment."""
    if not config_file.exists():
        err_console.print(f"[red]Error:[/red] File not found: {config_file}")
        raise typer.Exit(1)

    raw_text = config_file.read_text()
    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as e:
        err_console.print(f"[red]Error:[/red] Invalid JSON in {config_file}: {e}")
        raise typer.Exit(1)

    if "system_prompt" not in config and "tool_definitions" not in config:
        err_console.print(
            "[red]Error:[/red] Config file must have at least one of: "
            "system_prompt, tool_definitions"
        )
        raise typer.Exit(1)

    console.print(f"[dim]Scanning {config_file.name}...[/dim]")

    try:
        from agent_debug.agents.risk_scorer import RiskScorer
        from agent_debug.models.types import RiskInput
        scorer = RiskScorer()
        result = scorer.score(
            RiskInput(
                system_prompt=config.get("system_prompt", ""),
                tool_definitions=config.get("tool_definitions", []),
            )
        )
    except ValueError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(result, indent=2))
        return

    if output:
        output.write_text(json.dumps(result, indent=2))
        console.print(f"[green]Report saved to {output}[/green]")

    _print_risk_report(result)


# ─── Rich output helpers ──────────────────────────────────────────────────────

def _severity_color(severity: int) -> str:
    return {1: "green", 2: "yellow", 3: "yellow", 4: "red", 5: "bold red"}.get(
        severity, "white"
    )


def _risk_color(score: int) -> str:
    if score <= 3:
        return "green"
    if score <= 6:
        return "yellow"
    return "red"


def _print_report(report: dict) -> None:
    classification = report["classification"]
    severity = report["severity"]
    root_cause = report["root_cause"]
    fixes = report["fixes"]

    # Header
    sev_val = severity["severity"]
    sev_color = _severity_color(sev_val)
    console.print()
    console.print(
        Panel(
            f"[bold]{classification['subcategory']}[/bold]  "
            f"[{sev_color}]severity {sev_val}/5[/{sev_color}]  "
            f"[dim]confidence {classification['confidence']:.0%}[/dim]\n\n"
            f"{severity['rationale']}",
            title="[bold blue]Failure Classification[/bold blue]",
            border_style="blue",
        )
    )

    # Root cause
    console.print(
        Panel(
            f"[bold]Step {root_cause['failing_step_index']}[/bold] "
            f"({root_cause['failing_step_type']})\n\n"
            f"{root_cause['root_cause_explanation']}\n\n"
            f"[dim italic]Evidence: {root_cause['evidence_quote'][:300]}[/dim italic]",
            title="[bold yellow]Root Cause[/bold yellow]",
            border_style="yellow",
        )
    )

    # Fix suggestions
    suggestions = fixes.get("suggestions", [])
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            conf_color = "green" if s["confidence"] >= 0.7 else "yellow"
            console.print(
                Panel(
                    f"[bold]Target:[/bold] {s['target']}\n\n"
                    f"[bold]Before:[/bold]\n[red]{s['before']}[/red]\n\n"
                    f"[bold]After:[/bold]\n[green]{s['after']}[/green]\n\n"
                    f"[dim]{s['explanation']}[/dim]\n"
                    f"[{conf_color}]confidence {s['confidence']:.0%}[/{conf_color}]",
                    title=f"[bold green]Fix #{i}[/bold green]",
                    border_style="green",
                )
            )
        console.print(f"[dim italic]{fixes['disclaimer']}[/dim italic]")

    cost = report.get("cost_usd", 0)
    console.print(f"\n[dim]Analysis cost: ${cost:.4f}[/dim]\n")


def _print_risk_report(result: dict) -> None:
    score = result["overall_score"]
    color = _risk_color(score)
    findings = result.get("findings", [])

    console.print()
    console.print(
        Panel(
            f"[{color}]Risk score: {score}/10[/{color}]  "
            f"[dim]confidence {result['confidence']:.0%}[/dim]\n"
            f"{len(findings)} issue(s) found",
            title="[bold blue]Pre-deploy Risk Scan[/bold blue]",
            border_style="blue",
        )
    )

    if not findings:
        console.print("[green]No issues found. Looks good to deploy.[/green]\n")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Check", style="dim")
    table.add_column("Severity")
    table.add_column("Excerpt")
    table.add_column("Suggestion")

    sev_colors = {"high": "red", "medium": "yellow", "low": "green"}
    for f in findings:
        sev = f["severity"]
        table.add_row(
            f["check"],
            f"[{sev_colors.get(sev, 'white')}]{sev}[/{sev_colors.get(sev, 'white')}]",
            f["excerpt"][:60] + ("..." if len(f["excerpt"]) > 60 else ""),
            f["suggestion"][:80] + ("..." if len(f["suggestion"]) > 80 else ""),
        )

    console.print(table)
    console.print()
