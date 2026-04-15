"""CLI for agent-debug: analyze traces and scan prompts for risk."""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

app = typer.Typer(
    name="agent-debug",
    help="Diagnose why your AI agent failed. Root cause analysis + fix suggestions.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ─── analyze command ──────────────────────────────────────────────────────────

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


# ─── fix command ──────────────────────────────────────────────────────────────

@app.command()
def fix(
    trace_file: Annotated[Path, typer.Argument(help="Path to trace JSON file")],
    system_prompt: Annotated[
        Optional[Path],
        typer.Option("--system-prompt", "-s", help="System prompt file to patch (.txt/.md)"),
    ] = None,
    tools: Annotated[
        Optional[Path],
        typer.Option("--tools", "-t", help="Tool definitions file to patch (.json)"),
    ] = None,
    code: Annotated[
        Optional[list[Path]],
        typer.Option("--code", "-c", help="Code file(s) to patch (.py). Can be specified multiple times."),
    ] = None,
    test_cmd: Annotated[
        Optional[str],
        typer.Option("--test-cmd", "-T", help="Test command to run after each fix (e.g. 'pytest tests/')"),
    ] = None,
) -> None:
    """Diagnose a failed trace, then interactively apply fix suggestions.

    Shows root cause first, then walks through each fix one by one.
    You confirm each fix before it's applied.

    Example:
        agent-debug fix trace.json --system-prompt prompts/system.txt
        agent-debug fix trace.json --tools tools.json
        agent-debug fix trace.json --system-prompt system.txt --tools tools.json
        agent-debug fix trace.json --code agent.py --test-cmd "pytest tests/"
    """
    if not trace_file.exists():
        err_console.print(f"[red]Error:[/red] File not found: {trace_file}")
        raise typer.Exit(1)

    raw_text = trace_file.read_text()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        err_console.print(f"[red]Error:[/red] Invalid JSON in {trace_file}: {e}")
        raise typer.Exit(1)

    # Determine step count based on whether --test-cmd is provided
    has_test_cmd = bool(test_cmd)
    has_files = bool(code or system_prompt)
    run_preflight = has_test_cmd and has_files
    has_code_fixer = run_preflight and bool(code)
    total_steps = 4 if has_code_fixer else (4 if run_preflight else 3)

    # ── Step 1: Run analysis ──────────────────────────────────────────────
    console.print()
    console.print(f"[bold]Step 1/{total_steps} — Analyzing trace...[/bold]")

    # Collect actual file contents so FixGenerator can produce exact "before" strings
    file_contents: dict[str, str] = {}
    _seen: set[str] = set()
    for _fp in [system_prompt] + list(code or []):
        if _fp and _fp.exists() and str(_fp) not in _seen:
            file_contents[_fp.name] = _fp.read_text()
            _seen.add(str(_fp))

    from agent_debug.adapters import auto_parse
    from agent_debug.pipeline import DiagnosisPipeline
    from agent_debug.agents.code_validator import CodeValidator

    try:
        trace = auto_parse(raw)
    except (ValueError, RuntimeError) as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # ── Step 2 (optional): Pre-flight test run ────────────────────────────
    preflight_failures: str = ""
    if run_preflight:
        console.print()
        console.print(f"[bold]Step 2/{total_steps} — Running tests to identify failures...[/bold]")
        validator = CodeValidator()
        preflight = validator.run(test_cmd)  # type: ignore[arg-type]
        if preflight["passed"]:
            console.print("[green]✓ All tests passing — no code fixes needed[/green]")
            raise typer.Exit(0)
        else:
            # Show last 50 lines of pre-flight output
            output_lines = preflight["output"].splitlines()
            tail = "\n".join(output_lines[-50:]) if output_lines else "(no output)"
            console.print(
                Panel(
                    tail,
                    title="[red]Pre-flight Test Results[/red]",
                    border_style="red",
                )
            )
            # Count failures
            import re as _re
            fail_match = _re.search(r"(\d+) failed", preflight["output"])
            fail_count = fail_match.group(1) if fail_match else "some"
            console.print(f"[red]{fail_count} test(s) failed — generating targeted fixes...[/red]")
            preflight_failures = preflight["output"]

    # ── Step 3 (optional): Fix code files with CodeFixer ─────────────────
    code_fixer_patches: list[tuple] = []  # (file, original_content)
    if has_code_fixer and preflight_failures:
        from agent_debug.agents.code_fixer import CodeFixer
        console.print()
        console.print(f"[bold]Step 3/{total_steps} — Fix code files[/bold]")
        root_cause_explanation_early = ""  # will be filled after pipeline; use empty for now

        for code_file in (code or []):
            if not code_file.exists():
                console.print(f"[yellow]Skipping {code_file} — file not found[/yellow]")
                continue

            console.print()
            console.print(f"[bold]Fixing {code_file.name}...[/bold]")
            try:
                cf = CodeFixer()
                result = cf.fix_file(code_file, preflight_failures)
            except (ValueError, RuntimeError) as _e:
                console.print(f"[red]CodeFixer failed for {code_file.name}: {_e}[/red]")
                continue

            console.print(f"[dim]{result.changes_summary}[/dim]")
            console.print()
            choice = typer.prompt(
                f"Apply these fixes to {code_file.name}?",
                default="n",
                show_choices=True,
                prompt_suffix=" [y/n] ",
            ).strip().lower()

            if choice == "y":
                original_content = code_file.read_text()
                try:
                    code_file.write_text(result.fixed_content)
                    console.print(f"[green]✓ {code_file.name} updated[/green]")
                    code_fixer_patches.append((code_file, original_content))
                    # Refresh file_contents for the pipeline
                    file_contents[code_file.name] = result.fixed_content
                except OSError as _oe:
                    console.print(f"[red]Failed to write {code_file.name}: {_oe}[/red]")
            else:
                console.print(f"[dim]Skipped {code_file.name}.[/dim]")

    # ── Run analysis pipeline ─────────────────────────────────────────────
    try:
        pipeline = DiagnosisPipeline()
        report = pipeline.run_normalized(
            trace,
            file_contents=file_contents or None,
            test_failures=preflight_failures,
        )
    except (ValueError, RuntimeError) as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    classification = report["classification"]
    severity = report["severity"]
    root_cause = report["root_cause"]
    fixes = report["fixes"]
    suggestions = fixes.get("suggestions", [])

    # ── Step 3 or 4: Show failure reason ──────────────────────────────────
    diagnosis_step = 4 if has_code_fixer else (3 if run_preflight else 2)
    console.print()
    console.print(f"[bold]Step {diagnosis_step}/{total_steps} — Apply prompt fixes[/bold]")
    console.print()

    sev_val = severity["severity"]
    sev_color = _severity_color(sev_val)

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

    if not suggestions:
        console.print("[yellow]No fix suggestions generated.[/yellow]")
        raise typer.Exit(0)

    # ── Interactive fix application ───────────────────────────────────────
    apply_step = diagnosis_step
    console.print()
    console.print(f"[bold]Step {apply_step}/{total_steps} — Apply fixes[/bold] ({len(suggestions)} suggestion(s))")
    console.print(f"[dim]Files: system_prompt={system_prompt or '(none)'}  tools={tools or '(none)'}  code={[str(p) for p in (code or [])] or '(none)'}[/dim]")
    console.print(f"[dim]test: {test_cmd or '(none — add --test-cmd to validate)'}[/dim]")
    console.print()

    from agent_debug.agents.auto_fixer import AutoFixer
    fixer = AutoFixer(
        system_prompt_file=system_prompt,
        tools_file=tools,
        code_files=list(code) if code else None,
    )

    applied_count = 0
    skipped_count = 0
    applied_patches: list[tuple] = []  # (file, original_content, fix_num)

    for i, suggestion in enumerate(suggestions, 1):
        conf_color = "green" if suggestion["confidence"] >= 0.7 else "yellow"
        target_file = fixer.file_for(suggestion)

        console.print(
            Panel(
                f"[bold]Target:[/bold] {suggestion['target']}  "
                f"[{conf_color}]confidence {suggestion['confidence']:.0%}[/{conf_color}]\n\n"
                f"[dim]{suggestion['explanation']}[/dim]",
                title=f"[bold green]Fix #{i} of {len(suggestions)}[/bold green]",
                border_style="green",
            )
        )

        # Show diff
        console.print("[bold]Before →[/bold]")
        console.print(Panel(f"[red]{suggestion['before']}[/red]", border_style="red"))
        console.print("[bold]After →[/bold]")
        console.print(Panel(f"[green]{suggestion['after']}[/green]", border_style="green"))

        if target_file:
            console.print(f"[dim]Will modify: {target_file}[/dim]")
        else:
            console.print(
                "[yellow]No target file configured for this fix type. "
                "Pass --system-prompt, --tools, or --code to auto-apply.[/yellow]"
            )

        # Human confirmation
        console.print()
        choice = typer.prompt(
            f"Apply Fix #{i}?",
            default="n",
            show_choices=True,
            prompt_suffix=" [y/n/q(uit)] ",
        ).strip().lower()

        if choice == "q":
            console.print("[dim]Stopped at user request.[/dim]")
            break
        elif choice == "y":
            apply_result = fixer.apply(suggestion)
            if apply_result.status == "applied":
                console.print(f"[green]✓ Fix #{i} applied to {target_file}[/green]")
                applied_count += 1
                # Track for possible revert at end
                applied_patches.append((target_file, apply_result.original_content, i))
            elif apply_result.status == "no_file":
                console.print(
                    f"[yellow]✗ Could not apply Fix #{i} — "
                    "text not found in file or no file specified.[/yellow]"
                )
                _print_manual_instructions(suggestion)
                skipped_count += 1
            elif apply_result.status == "error":
                console.print(f"[red]✗ Failed to write {target_file}[/red]")
                skipped_count += 1
        else:
            console.print(f"[dim]Fix #{i} skipped.[/dim]")
            skipped_count += 1

        console.print()

    # ── Final batch test (run ONCE after all fixes applied) ───────────────
    total_applied = applied_count + len(code_fixer_patches)
    if test_cmd and total_applied > 0:
        console.print()
        console.print(f"[bold]Running final tests on all {total_applied} applied fix(es)...[/bold]")
        final_validation = CodeValidator().run(test_cmd)
        if final_validation["passed"]:
            console.print(
                f"[green]✓ All tests passed ({final_validation['duration_sec']:.1f}s) — fixes verified![/green]"
            )
        else:
            output_lines = final_validation["output"].splitlines()
            tail = "\n".join(output_lines[-30:]) if output_lines else "(no output)"
            console.print(
                Panel(tail, title="[red]Final Test Results[/red]", border_style="red")
            )
            # Count remaining failures
            import re as _re2
            fm = _re2.search(r"(\d+) failed", final_validation["output"])
            remaining = fm.group(1) if fm else "some"
            console.print(f"[yellow]{remaining} test(s) still failing.[/yellow]")
            revert_all = typer.prompt(
                "Revert ALL applied fixes?",
                default="n",
                prompt_suffix=" [y/n] ",
            ).strip().lower()
            if revert_all == "y":
                for patch_file, original, fix_num in applied_patches:
                    if patch_file and original and fixer.revert(patch_file, original):
                        console.print(f"[yellow]↩ Fix #{fix_num} reverted[/yellow]")
                for patch_file, original in code_fixer_patches:
                    if patch_file and original and fixer.revert(patch_file, original):
                        console.print(f"[yellow]↩ CodeFixer patch reverted for {patch_file.name}[/yellow]")
                applied_count = 0
        console.print()

    # ── Summary ───────────────────────────────────────────────────────────
    code_fixed = len(code_fixer_patches)
    total_applied = code_fixed + applied_count
    summary_lines = []
    if code_fixed:
        summary_lines.append(f"[green]Code fixes: {code_fixed} file(s) rewritten[/green]")
    if applied_count:
        summary_lines.append(f"[green]Prompt fixes: {applied_count} applied[/green]")
    if not total_applied:
        summary_lines.append(f"[yellow]Applied: 0[/yellow]")
    summary_lines.append(f"[dim]Skipped: {skipped_count}[/dim]")
    summary_lines.append(f"\n[dim italic]{fixes['disclaimer']}[/dim italic]")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold]Done[/bold]",
            border_style="dim",
        )
    )

    cost = report.get("cost_usd", 0)
    console.print(f"[dim]Analysis cost: ${cost:.4f}[/dim]\n")


def _print_manual_instructions(suggestion: dict) -> None:
    """Print copy-paste instructions when auto-apply fails."""
    console.print(
        Panel(
            f"[bold]Manually apply this change:[/bold]\n\n"
            f"Find this text:\n[red]{suggestion['before'][:300]}[/red]\n\n"
            f"Replace with:\n[green]{suggestion['after'][:300]}[/green]",
            title="[yellow]Manual apply required[/yellow]",
            border_style="yellow",
        )
    )


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
