"""Format DiagnosisReport as GitHub-flavored markdown for PR comments."""

from agent_debug.models.types import DiagnosisReport

_SEVERITY_EMOJI = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🔴"}
_SEVERITY_LABEL = {1: "Minor", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}


def report_to_markdown(report: DiagnosisReport, trace_file: str = "") -> str:
    """Convert a DiagnosisReport to a GitHub PR comment markdown string."""
    c = report["classification"]
    s = report["severity"]
    r = report["root_cause"]
    f = report["fixes"]

    sev = s["severity"]
    sev_emoji = _SEVERITY_EMOJI.get(sev, "🟠")
    sev_label = _SEVERITY_LABEL.get(sev, "Unknown")

    header = f"## 🤖 agent-debug diagnosis"
    if trace_file:
        header += f"\n**File:** `{trace_file}`"

    classification_section = f"""
### {sev_emoji} `{c['subcategory']}` — severity {sev}/5 ({sev_label})

> {s['rationale']}

**Confidence:** {c['confidence']:.0%}
"""

    root_cause_section = f"""
### 🔍 Root Cause

**Failing step:** Step {r['failing_step_index']} (`{r['failing_step_type']}`)

{r['root_cause_explanation']}

<details>
<summary>Evidence from trace</summary>

```
{r['evidence_quote'][:500]}
```

</details>
"""

    suggestions = f.get("suggestions", [])
    if suggestions:
        fix_lines = ["\n### 🔧 Suggested Fixes\n"]
        for i, suggestion in enumerate(suggestions, 1):
            conf_bar = "🟢" if suggestion["confidence"] >= 0.7 else "🟡"
            fix_lines.append(f"**Fix #{i}** — `{suggestion['target']}` {conf_bar} {suggestion['confidence']:.0%} confidence\n")
            fix_lines.append(suggestion["explanation"])
            fix_lines.append("\n<details>")
            fix_lines.append(f"<summary>Show diff</summary>\n")
            fix_lines.append("**Before:**")
            fix_lines.append(f"```\n{suggestion['before']}\n```")
            fix_lines.append("**After:**")
            fix_lines.append(f"```\n{suggestion['after']}\n```")
            fix_lines.append("</details>\n")
        fixes_section = "\n".join(fix_lines)
    else:
        fixes_section = ""

    footer = f"\n---\n*Analysis cost: ${report['cost_usd']:.4f} · [agent-debug](https://github.com/Viktorsdb/agent-debug)*"

    return "\n".join([
        header,
        classification_section,
        root_cause_section,
        fixes_section,
        footer,
    ])


def risk_report_to_markdown(result: dict) -> str:
    """Convert a RiskScorer output to markdown."""
    score = result["overall_score"]
    findings = result.get("findings", [])

    if score <= 3:
        score_emoji = "🟢"
    elif score <= 6:
        score_emoji = "🟡"
    else:
        score_emoji = "🔴"

    lines = [
        "## 🤖 agent-debug pre-deploy scan",
        f"\n**Risk score:** {score_emoji} {score}/10  |  **{len(findings)} issue(s) found**\n",
    ]

    if not findings:
        lines.append("✅ No issues found. Looks good to deploy.")
    else:
        sev_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for f in findings:
            emoji = sev_emoji.get(f["severity"], "🟠")
            lines.append(f"#### {emoji} `{f['check']}` ({f['severity']})")
            lines.append(f"> {f['excerpt'][:150]}")
            lines.append(f"\n💡 {f['suggestion']}\n")

    lines.append(f"\n---\n*[agent-debug](https://github.com/Viktorsdb/agent-debug)*")
    return "\n".join(lines)
