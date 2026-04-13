# agent-debug

> Diagnose why your AI agent failed. Get a root cause analysis and a concrete fix suggestion.

**Status:** Early development — private for now.

## What is this?

agent-debug captures your agent's execution trace (tool calls, LLM completions, reasoning steps) and uses Claude to analyze exactly which step failed, why, and what to change to fix it.

Unlike tracing tools that show you *what happened*, agent-debug tells you *why it went wrong* and *what to do about it*.

## Planned features

- Trace collector SDK (Python, TypeScript)
- Failure analyzer — classifies failures into: wrong tool, hallucination, premature stop, context overflow
- Fix suggester — before/after prompt diff, copy-paste ready
- Pre-deploy risk score — static analysis of your system prompt before first run

## Status

Building. Not ready for use yet.

