"""
All TypedDict definitions for agent-debug.

Data flows:
  raw trace (any SDK format)
      └─► NormalizedTrace          (via adapters/)
              └─► ClassifierInput
              └─► ClassifierOutput  ──► AnalystInput
              └─► SeverityOutput    ──► AnalystInput
                                         └─► AnalystOutput ──► FixInput
                                                                  └─► FixOutput
                                                                       └─► DiagnosisReport
"""

from typing import Any, Literal, TypedDict


# ─── Normalized Trace ─────────────────────────────────────────────────────────

class ToolCall(TypedDict):
    index: int            # step position in the trace
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str      # always string; adapters convert complex types
    error: str | None     # None if tool succeeded


class LLMCompletion(TypedDict):
    index: int            # step position
    prompt_summary: str   # first 500 chars of the effective prompt
    response: str
    token_count: int | None


class NormalizedTrace(TypedDict):
    trace_id: str
    sdk_source: Literal["openai", "claude", "langchain", "unknown"]
    task_description: str       # user's original request
    system_prompt: str          # full system prompt
    tool_definitions: list[dict[str, Any]]
    steps: list[ToolCall | LLMCompletion]   # ordered sequence
    final_output: str
    succeeded: bool             # caller labels: did the agent complete the task?
    metadata: dict[str, Any]    # raw SDK-specific extras


# ─── Failure Taxonomy ─────────────────────────────────────────────────────────
#
# Two-level taxonomy: 6 categories × up to 3 subcategories each.
# PatternClassifier outputs a subcategory; UI groups by category.

FailureCategory = Literal[
    "wrong_tool",
    "hallucination",
    "premature_stop",
    "context_overflow",
    "tool_misinterpretation",
    "prompt_ambiguity",
]

FailureSubcategory = Literal[
    # wrong_tool
    "wrong_tool.similar_name",
    "wrong_tool.missing_guidance",
    "wrong_tool.scope_confusion",
    # hallucination
    "hallucination.missing_retrieval",
    "hallucination.domain_gap",
    "hallucination.format_pressure",
    # premature_stop
    "premature_stop.ambiguous_done",
    "premature_stop.error_avoidance",
    "premature_stop.max_steps_hit",
    # context_overflow
    "context_overflow.long_conversation",
    "context_overflow.large_tool_output",
    # tool_misinterpretation
    "tool_misinterpretation.schema_mismatch",
    "tool_misinterpretation.error_ignored",
    "tool_misinterpretation.partial_result",
    # prompt_ambiguity
    "prompt_ambiguity.conflicting_instructions",
    "prompt_ambiguity.underspecified_scope",
]


# ─── Agent IO Types ───────────────────────────────────────────────────────────

class ClassifierInput(TypedDict):
    trace: NormalizedTrace


class ClassifierOutput(TypedDict):
    category: FailureCategory
    subcategory: FailureSubcategory
    evidence_step_index: int       # which step triggered this classification
    confidence: float              # 0.0–1.0


class SeverityInput(TypedDict):
    trace: NormalizedTrace
    category: FailureCategory


class SeverityOutput(TypedDict):
    severity: int                  # 1 (minor) – 5 (data loss / crash)
    rationale: str
    confidence: float


class AnalystInput(TypedDict):
    trace: NormalizedTrace
    classification: ClassifierOutput
    severity: SeverityOutput


class AnalystOutput(TypedDict):
    failing_step_index: int
    failing_step_type: Literal["tool_call", "llm_completion"]
    root_cause_explanation: str    # plain English, 2–4 sentences
    evidence_quote: str            # direct quote from trace proving the cause
    confidence: float


class FixInput(TypedDict):
    trace: NormalizedTrace
    classification: ClassifierOutput
    root_cause: AnalystOutput


class FixSuggestion(TypedDict):
    target: Literal["system_prompt", "tool_definition", "tool_description", "retrieval_setup"]
    before: str
    after: str
    explanation: str               # why this change addresses the root cause
    confidence: float


class FixOutput(TypedDict):
    suggestions: list[FixSuggestion]
    disclaimer: str                # always: "Test before deploying"


# ─── Risk Scorer IO ───────────────────────────────────────────────────────────

class RiskInput(TypedDict):
    system_prompt: str
    tool_definitions: list[dict[str, Any]]


class RiskFinding(TypedDict):
    check: str                     # which check triggered
    severity: Literal["high", "medium", "low"]
    excerpt: str                   # the problematic text
    suggestion: str


class RiskOutput(TypedDict):
    overall_score: int             # 1 (low risk) – 10 (high risk)
    findings: list[RiskFinding]
    confidence: float


# ─── Final Diagnosis Report ───────────────────────────────────────────────────

class DiagnosisReport(TypedDict):
    trace_id: str
    classification: ClassifierOutput
    severity: SeverityOutput
    root_cause: AnalystOutput
    fixes: FixOutput
    cost_usd: float                # estimated API cost for this analysis
