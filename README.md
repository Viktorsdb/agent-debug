# agent-debug

**Diagnose why your AI agent failed.** Root cause analysis + concrete fix suggestions — in your terminal or automatically on every PR.

```
$ agent-debug analyze trace.json

  Failure Classification
  wrong_tool.scope_confusion  severity 3/5  confidence 95%
  The agent used web_search instead of read_file for a local file task.

  Root Cause  ·  Step 0 (tool_call)
  The agent confused tool scope — it assumed 'report.txt' was a web resource
  instead of using the local read_file tool available to it.

  Fix #1  system_prompt  ·  92% confidence
  Before: "You are a helpful assistant with access to tools."
  After:  "...always use a local file tool for file tasks, NOT web_search."
```

Think of it as **Sentry for agents** — tells you not just *what* broke, but *why* and *what to change*.

---

## Features

- 🔍 **Root cause analysis** — pinpoints the exact step where things went wrong
- 🏷️ **15 failure subcategories** — not just "it failed", but *wrong_tool.scope_confusion* or *hallucination.missing_retrieval*
- 🔧 **Before/after fix diffs** — copy-paste ready changes to your system prompt or tool definitions
- 📸 **Auto-capture** — decorator, patch, or context manager: no manual JSON export needed
- 🤖 **GitHub Action** — auto-comments on PRs when your agent tests fail
- 🔌 **Multi-provider** — bring your own API key: Anthropic, OpenAI, DeepSeek, or Ollama (local/free)
- 📦 **Multi-SDK** — supports OpenAI, Claude SDK, and LangChain trace formats

---

## Install

```bash
pip install agent-debug
```

Requires Python 3.11+.

---

## Auto-capture (no JSON export needed)

Three ways to record traces automatically — pick whichever fits your style.

### Option 1: Decorator

```python
import agent_debug

@agent_debug.trace(output="agent_traces/my_run.json")
def run_my_agent(task: str):
    client = openai.OpenAI()
    # your agent code — nothing else changes
    response = client.chat.completions.create(...)
    return response
```

### Option 2: Patch (zero code changes)

```python
import agent_debug

agent_debug.patch_openai()   # add this one line

# everything below stays exactly the same
client = openai.OpenAI()
response = client.chat.completions.create(...)

agent_debug.save_trace("agent_traces/run.json")
```

### Option 3: Context manager

```python
import agent_debug

# starts recording when entering the block, saves when exiting
with agent_debug.capture("agent_traces/run.json"):
    result = my_agent.run(task)
# trace.json saved automatically
```

All three work with the GitHub Action — just point `traces_dir` at your output folder.

---

## Quickstart (manual trace)

### 1. Set your API key

```bash
# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...
export AGENT_DEBUG_PROVIDER=openai

# DeepSeek (cheapest)
export DEEPSEEK_API_KEY=sk-...
export AGENT_DEBUG_PROVIDER=deepseek

# Ollama (local, free)
export AGENT_DEBUG_PROVIDER=ollama  # no key needed
```

### 2. Save a trace

Export your agent's execution as a JSON file. Three formats supported:

<details>
<summary><b>OpenAI</b> — messages + choices format</summary>

```json
{
  "trace_id": "my-run-001",
  "task_description": "Read report.txt and summarize it",
  "system_prompt": "You are a helpful assistant.",
  "tool_definitions": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a local file by path",
        "parameters": {
          "type": "object",
          "properties": { "path": { "type": "string" } },
          "required": ["path"]
        }
      }
    }
  ],
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Read report.txt and summarize it" },
    {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": { "name": "web_search", "arguments": "{\"query\": \"report.txt\"}" }
      }]
    },
    { "role": "tool", "tool_call_id": "call_1", "content": "No results found" }
  ],
  "choices": [{ "message": { "role": "assistant", "content": "I couldn't find the file." } }],
  "succeeded": false
}
```
</details>

<details>
<summary><b>Claude SDK</b> — content blocks format</summary>

```json
{
  "trace_id": "my-run-002",
  "task_description": "List Python files and count lines in each",
  "system_prompt": "You are a code analysis assistant.",
  "tool_definitions": [{ "name": "bash", "description": "Run a shell command" }],
  "messages": [
    { "role": "user", "content": "List Python files and count lines in each" },
    {
      "role": "assistant",
      "content": [{ "type": "tool_use", "id": "tu_1", "name": "bash", "input": { "command": "find /src -name '*.py'" } }]
    },
    {
      "role": "user",
      "content": [{ "type": "tool_result", "tool_use_id": "tu_1", "content": "/src/main.py\n/src/utils.py" }]
    },
    { "role": "assistant", "content": [{ "type": "text", "text": "Found 2 files." }] }
  ],
  "final_response": { "stop_reason": "end_turn", "content": [], "usage": { "input_tokens": 450, "output_tokens": 35 } },
  "succeeded": false
}
```
</details>

<details>
<summary><b>LangChain</b> — intermediate_steps format</summary>

```json
{
  "trace_id": "my-run-003",
  "task_description": "Get the current stock price of AAPL",
  "input": "Get the current stock price of AAPL",
  "intermediate_steps": [
    [{ "tool": "get_stock_price", "tool_input": "AAPL stock" }, "Error: Invalid ticker format"]
  ],
  "output": "The price is $182.50",
  "succeeded": false
}
```
</details>

### 3. Analyze

```bash
agent-debug analyze trace.json
```

```bash
# Save report as JSON
agent-debug analyze trace.json --output report.json

# Print raw JSON
agent-debug analyze trace.json --json
```

### Pre-deploy risk scan

Catch problems *before* your agent runs:

```bash
agent-debug scan config.json
```

`config.json` needs `system_prompt` and/or `tool_definitions`.

---

## GitHub Action

Auto-diagnose agent failures on every PR — no manual trace export needed.

### Setup

**Step 1.** Save your agent's trace to `agent_traces/` during tests:

```python
# In your test / agent runner:
import json, pathlib

trace = {
    "task_description": "...",
    "messages": [...],
    "choices": [...],
    "succeeded": False,
}
pathlib.Path("agent_traces").mkdir(exist_ok=True)
pathlib.Path("agent_traces/my_agent.trace.json").write_text(json.dumps(trace))
```

**Step 2.** Add to `.github/workflows/agent-test.yml`:

```yaml
name: Agent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run agent tests
        run: pytest tests/
        continue-on-error: true   # let agent-debug run even if tests fail

      - name: Diagnose agent failures
        uses: Viktorsdb/agent-debug@main
        with:
          api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          provider: anthropic        # or: openai, deepseek
          traces_dir: agent_traces
```

**Step 3.** Add your API key as a GitHub secret:
`Settings → Secrets → Actions → New repository secret`

### What you get on every PR

```
🤖 agent-debug diagnosis
File: agent_traces/my_agent.trace.json

🟠 wrong_tool.scope_confusion — severity 3/5 (Medium)
> The agent stopped before completing all parts of the task.
Confidence: 95%

🔍 Root Cause
Failing step: Step 0 (tool_call)
The agent confused tool scope...

🔧 Suggested Fixes
Fix #1 — system_prompt 🟢 92% confidence
Before: "You are a helpful assistant..."
After:  "...always use read_file for local files, not web_search."
```

### Action inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `api_key` | ✅ | — | Your LLM API key |
| `provider` | — | `anthropic` | `anthropic` / `openai` / `deepseek` / `ollama` |
| `model` | — | provider default | Override model name |
| `base_url` | — | — | Custom API endpoint (for proxies) |
| `traces_dir` | — | `agent_traces` | Directory with trace JSON files |
| `fail_on_severity` | — | `3` | Fail CI if severity ≥ this (0 = never fail) |

---

## Multi-Provider Support

`agent-debug` uses Claude / GPT-4o / DeepSeek / Ollama to run the analysis. Bring your own key.

```bash
# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...  agent-debug analyze trace.json

# OpenAI
OPENAI_API_KEY=sk-...  AGENT_DEBUG_PROVIDER=openai  agent-debug analyze trace.json

# DeepSeek — cheapest option (~$0.001/analysis)
DEEPSEEK_API_KEY=sk-...  AGENT_DEBUG_PROVIDER=deepseek  agent-debug analyze trace.json

# Ollama — completely free, runs locally
AGENT_DEBUG_PROVIDER=ollama  agent-debug analyze trace.json

# Third-party relay (any OpenAI-compatible endpoint)
OPENAI_API_KEY=sk-...  OPENAI_BASE_URL=https://my-proxy.com/v1  \
AGENT_DEBUG_PROVIDER=openai  agent-debug analyze trace.json
```

### Python API with custom provider

```python
from agent_debug import DiagnosisPipeline
from agent_debug.providers import get_provider

# Use any provider
pipeline = DiagnosisPipeline(provider=get_provider("deepseek"))
report = pipeline.run(raw_trace_dict)

print(report["classification"]["subcategory"])  # "wrong_tool.scope_confusion"
print(report["severity"]["severity"])           # 3
print(report["root_cause"]["root_cause_explanation"])
for fix in report["fixes"]["suggestions"]:
    print(f"[{fix['target']}]\nBefore: {fix['before']}\nAfter:  {fix['after']}\n")
```

---

## Failure Taxonomy

15 subcategories across 6 categories:

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

```
trace.json
    │
    ▼
[Adapter]              Detect format (OpenAI / Claude / LangChain) → normalize
    │
    ▼
[PatternClassifier]    Classify into 1 of 15 subcategories
    │
    ▼
[SeverityEstimator]    Rate severity 1–5
    │
    ▼
[RootCauseAnalyst]     Pinpoint the exact failing step + explain why
    │
    ▼
[FixGenerator]         Generate before/after diffs for prompt/tool fixes
    │
    ▼
DiagnosisReport  →  terminal / JSON / GitHub PR comment
```

Typical runtime: ~15 seconds. Typical cost: $0.01–$0.05 per trace.

---

## Development

```bash
git clone https://github.com/Viktorsdb/agent-debug
cd agent-debug
uv sync
uv run pytest tests/ -v
```

Tests run without any API key — adapters and base agent logic are fully deterministic.

---

## License

MIT
