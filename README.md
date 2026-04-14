# agent-debug

**Diagnose why your AI agent failed.** Root cause analysis + concrete fix suggestions, in your terminal.

```
$ agent-debug analyze trace.json

  Failure Classification
  wrong_tool.scope_confusion  severity 3/5  confidence 95%
  The agent used web_search instead of read_file for a local file task.

  Root Cause
  Step 0 (tool_call)
  The agent confused the scope of available tools by incorrectly assuming
  'report.txt' was a web-accessible resource...

  Fix #1  target: system_prompt  confidence 92%
  Before: "You are a helpful assistant with access to tools."
  After:  "...When given a task involving a specific file (e.g., 'read report.txt'),
           always use a local file tool, NOT a web search."
```

---

## Why

Debugging a failed agent trace is painful. You stare at 50 lines of tool calls trying to figure out *why* it went wrong and *what to change*. `agent-debug` runs a multi-agent analysis pipeline on the trace and gives you:

- **What failed** — one of 15 precise subcategories (not just "it broke")
- **Why it failed** — root cause explanation pointing to the exact step
- **How to fix it** — before/after diffs for your system prompt or tool definitions

Think of it as **Sentry for agents**.

---

## Install

```bash
pip install agent-debug
```

Requires Python 3.11+ and an Anthropic API key.

---

## Quick Start

### 1. Capture a trace

Save your agent's execution as a JSON file. Supported formats: **OpenAI**, **Claude SDK**, **LangChain**.

<details>
<summary>OpenAI format</summary>

```json
{
  "task_description": "Read report.txt and summarize it",
  "system_prompt": "You are a helpful assistant.",
  "tool_definitions": [],
  "messages": [],
  "choices": [],
  "succeeded": false
}
```
</details>

<details>
<summary>Claude SDK format</summary>

```json
{
  "task_description": "List Python files and count lines",
  "system_prompt": "You are a code analysis assistant.",
  "tool_definitions": [],
  "messages": [],
  "final_response": { "stop_reason": "end_turn", "content": [], "usage": {} },
  "succeeded": false
}
```
</details>

<details>
<summary>LangChain format</summary>

```json
{
  "task_description": "Get the stock price of AAPL",
  "input": "Get the stock price of AAPL",
  "intermediate_steps": [
    [{"tool": "get_stock_price", "tool_input": "AAPL stock"}, "Error: Invalid ticker"]
  ],
  "output": "The price is $182.50",
  "succeeded": false
}
```
</details>

### 2. Analyze

```bash
export ANTHROPIC_API_KEY=sk-...

agent-debug analyze trace.json
```

Save report to file:

```bash
agent-debug analyze trace.json --output report.json
```

### 3. Pre-deploy risk scan

Catch problems *before* your agent runs:

```bash
agent-debug scan config.json
```

Where `config.json` contains your `system_prompt` and `tool_definitions`.

---

## Failure Taxonomy

`agent-debug` classifies failures into **15 subcategories** across 6 categories:

| Category | Subcategories |
|----------|--------------|
| `wrong_tool` | `similar_name` · `missing_guidance` · `scope_confusion` |
| `hallucination` | `missing_retrieval` · `domain_gap` · `format_pressure` |
| `premature_stop` | `ambiguous_done` · `error_avoidance` · `max_steps_hit` |
| `context_overflow` | `long_conversation` · `large_tool_output` |
| `tool_misinterpretation` | `schema_mismatch` · `error_ignored` · `partial_result` |
| `prompt_ambiguity` | `conflicting_instructions` · `underspecified_scope` |

---

## How It Works

4 Claude agents run in sequence on your trace:

```
trace.json
    │
    ▼
[Adapter]             Normalize OpenAI / Claude / LangChain → common format
    │
    ▼
[PatternClassifier]   Classify into 1 of 15 subcategories
    │
    ▼
[SeverityEstimator]   Rate severity 1–5
    │
    ▼
[RootCauseAnalyst]    Pinpoint the exact failing step + explain why
    │
    ▼
[FixGenerator]        Generate before/after diffs for prompt/tool fixes
    │
    ▼
DiagnosisReport
```

Typical runtime: ~15 seconds. Typical cost: $0.02–$0.05 per trace.

---

## Python API

```python
from agent_debug import DiagnosisPipeline

pipeline = DiagnosisPipeline()
report = pipeline.run(raw_trace_dict)

print(report["classification"]["subcategory"])   # e.g. "wrong_tool.scope_confusion"
print(report["severity"]["severity"])            # e.g. 3
print(report["root_cause"]["root_cause_explanation"])
for fix in report["fixes"]["suggestions"]:
    print(fix["before"], "→", fix["after"])
```

### Custom base URL (third-party Claude API)

```python
import anthropic
from agent_debug import DiagnosisPipeline

client = anthropic.Anthropic(
    api_key="sk-...",
    base_url="https://your-proxy.example.com/claude",
)
pipeline = DiagnosisPipeline(client=client)
report = pipeline.run(trace)
```

Or via environment variables:

```bash
export ANTHROPIC_API_KEY=sk-...
export ANTHROPIC_BASE_URL=https://your-proxy.example.com/claude
agent-debug analyze trace.json
```

---

## Development

```bash
git clone https://github.com/Viktorsdb/agent-debug
cd agent-debug
uv sync
uv run pytest tests/test_adapters/ tests/test_agents/ -v
```

Tests run without an API key (adapters and base agent logic are fully deterministic).

---

## License

MIT
