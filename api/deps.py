"""
Module 09D - API Dependencies

Dependency injection for the API.
Provides factories for pipeline and agent instances.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orchestrator.pipeline import Pipeline, PipelineConfig, PoRPackage
from core.schemas.prompts import (
    PromptSpec, PredictionSemantics, DataRequirement,
    SourceTarget, SelectionPolicy,
)
from core.schemas.market import (
    MarketSpec, ResolutionWindow, ResolutionRules, ResolutionRule, DisputePolicy,
)
from core.schemas.transport import ToolPlan
from core.schemas.evidence import (
    EvidenceBundle, EvidenceItem, SourceDescriptor,
    RetrievalReceipt, ProvenanceProof,
)
from core.por.reasoning_trace import ReasoningTrace, ReasoningStep, TracePolicy
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import VerificationResult, CheckResult
from core.por.proof_of_reasoning import verify_por_bundle


class MockPromptEngineer:
    """Mock Prompt Engineer for API testing."""
    
    def run(self, user_input: str, **kwargs):
        market_id = f"mkt_{abs(hash(user_input)) % 100000:05d}"
        market = MarketSpec(
            market_id=market_id,
            question=user_input,
            event_definition=f"Resolution of: {user_input}",
            resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
            resolution_window=ResolutionWindow(
                start=datetime(2026, 12, 31, tzinfo=timezone.utc),
                end=datetime(2027, 1, 1, tzinfo=timezone.utc),
            ),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R_BINARY", description="Binary yes/no", priority=1),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=3600),
        )
        prompt_spec = PromptSpec(
            market=market,
            prediction_semantics=PredictionSemantics(
                target_entity="query_target",
                predicate="satisfies_condition",
            ),
            data_requirements=[
                DataRequirement(
                    requirement_id="req_001",
                    description="Primary data source",
                    source_targets=[
                        SourceTarget(
                            source_id="source_001",
                            uri="https://api.example.com/data",
                            method="GET",
                            expected_content_type="json",
                        ),
                    ],
                    selection_policy=SelectionPolicy(
                        strategy="single_best", min_sources=1, max_sources=1, quorum=1,
                    ),
                ),
            ],
            created_at=None,
            extra={"strict_mode": True},
        )
        tool_plan = ToolPlan(
            plan_id=f"plan_{market_id}",
            requirements=["req_001"],
            sources=["source_001"],
        )
        return prompt_spec, tool_plan


class MockCollector:
    """Mock Collector for API testing."""
    
    def collect(self, prompt_spec, tool_plan, **kwargs):
        return EvidenceBundle(
            bundle_id=f"bundle_{prompt_spec.market.market_id}",
            collector_id="mock_collector_v1",
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    requirement_id="req_001",
                    source=SourceDescriptor(
                        source_id="source_001",
                        uri="https://api.example.com/data",
                        provider="Mock Provider",
                    ),
                    retrieval=RetrievalReceipt(
                        retrieved_at=None,
                        method="http_get",
                        request_fingerprint="0x" + "a1" * 32,
                        response_fingerprint="0x" + "b2" * 32,
                    ),
                    provenance=ProvenanceProof(tier=1, kind="signature"),
                    content_type="json",
                    content={"value": True, "data": "mock_response"},
                    confidence=0.9,
                ),
            ],
        )


class MockAuditor:
    """Mock Auditor for API testing."""
    
    def audit(self, prompt_spec, evidence, **kwargs):
        trace = ReasoningTrace(
            trace_id=f"trace_{prompt_spec.market.market_id}",
            policy=TracePolicy(max_steps=100),
            steps=[
                ReasoningStep(
                    step_id="step_0001",
                    type="extract",
                    action="Extract data from evidence",
                    inputs={"evidence_id": "ev_001"},
                    output={"extracted": True},
                    evidence_ids=["ev_001"],
                ),
                ReasoningStep(
                    step_id="step_0002",
                    type="check",
                    action="Evaluate condition",
                    inputs={"extracted": True},
                    output={"condition_met": True},
                    evidence_ids=["ev_001"],
                    prior_step_ids=["step_0001"],
                ),
            ],
            evidence_refs=["ev_001"],
        )
        verification = VerificationResult(
            ok=True,
            checks=[CheckResult.passed("audit_check", "Audit completed")],
        )
        return trace, verification


class MockJudge:
    """Mock Judge for API testing."""
    
    def judge(self, prompt_spec, evidence, trace, **kwargs):
        verdict = DeterministicVerdict(
            market_id=prompt_spec.market.market_id,
            outcome="YES",
            confidence=0.85,
            resolution_time=None,
            resolution_rule_id="R_BINARY",
        )
        verification = VerificationResult(
            ok=True,
            checks=[CheckResult.passed("judge_check", "Judgment rendered")],
        )
        return verdict, verification


class MockSentinel:
    """Mock Sentinel for API testing."""
    
    def verify(self, package: PoRPackage, *, mode: str = "verify"):
        result = verify_por_bundle(
            package.bundle,
            prompt_spec=package.prompt_spec,
            evidence=package.evidence,
            trace=package.trace,
        )
        challenges = []
        if not result.ok and result.challenge:
            challenges.append({
                "kind": result.challenge.kind,
                "reason": result.challenge.reason,
            })
        return result, challenges


def get_pipeline(
    strict_mode: bool = True,
    enable_sentinel: bool = True,
    enable_replay: bool = False,
) -> Pipeline:
    """
    Create a Pipeline instance with the given configuration.
    
    In production, this would wire up real agent implementations.
    For now, we use mock agents.
    """
    config = PipelineConfig(
        strict_mode=strict_mode,
        enable_sentinel_verify=enable_sentinel,
        enable_replay=enable_replay,
    )
    
    sentinel = MockSentinel() if enable_sentinel else None
    
    return Pipeline(
        config=config,
        prompt_engineer=MockPromptEngineer(),
        collector=MockCollector(),
        auditor=MockAuditor(),
        judge=MockJudge(),
        sentinel=sentinel,
    )


def get_sentinel() -> MockSentinel:
    """Get a Sentinel instance."""
    return MockSentinel()