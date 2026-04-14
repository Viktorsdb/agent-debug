"""Sequential analysis pipeline: runs 4 agents in order to produce DiagnosisReport."""

from agent_debug.adapters import auto_parse
from agent_debug.agents.fix_generator import FixGenerator
from agent_debug.agents.pattern_classifier import PatternClassifier
from agent_debug.agents.root_cause_analyst import RootCauseAnalyst
from agent_debug.agents.severity_estimator import SeverityEstimator
from agent_debug.models.types import (
    AnalystInput,
    ClassifierInput,
    DiagnosisReport,
    FixInput,
    NormalizedTrace,
    SeverityInput,
)
from agent_debug.providers.base import LLMProvider


class DiagnosisPipeline:
    """Run all 4 analysis agents sequentially and return a DiagnosisReport.

    Usage:
        # Auto-detect provider from env vars:
        pipeline = DiagnosisPipeline()

        # Explicit provider:
        from agent_debug.providers import get_provider
        pipeline = DiagnosisPipeline(provider=get_provider("openai"))

        report = pipeline.run(raw_trace_dict)
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.classifier = PatternClassifier(provider)
        self.severity = SeverityEstimator(provider)
        self.analyst = RootCauseAnalyst(provider)
        self.fix_gen = FixGenerator(provider)

    def run(self, raw: dict) -> DiagnosisReport:
        """Parse raw trace and run full analysis pipeline.

        Args:
            raw: SDK-specific trace dict (OpenAI, Claude, or LangChain format)

        Returns:
            DiagnosisReport with classification, severity, root cause, and fixes

        Raises:
            ValueError: if trace format is unrecognized or required fields are missing
            RuntimeError: if any agent call times out
        """
        trace = auto_parse(raw)
        return self.run_normalized(trace)

    def run_normalized(self, trace: NormalizedTrace) -> DiagnosisReport:
        """Run pipeline on a pre-normalized trace."""
        total_cost = 0.0

        classification = self.classifier.classify(ClassifierInput(trace=trace))
        total_cost += self.classifier.last_cost_usd

        severity_out = self.severity.estimate(
            SeverityInput(trace=trace, category=classification["category"])
        )
        total_cost += self.severity.last_cost_usd

        root_cause = self.analyst.analyze(
            AnalystInput(
                trace=trace,
                classification=classification,
                severity=severity_out,
            )
        )
        total_cost += self.analyst.last_cost_usd

        fixes = self.fix_gen.generate(
            FixInput(
                trace=trace,
                classification=classification,
                root_cause=root_cause,
            )
        )
        total_cost += self.fix_gen.last_cost_usd

        return DiagnosisReport(
            trace_id=trace["trace_id"],
            classification=classification,
            severity=severity_out,
            root_cause=root_cause,
            fixes=fixes,
            cost_usd=round(total_cost, 6),
        )
