"""
Module 09A - Pipeline Integration Tests

Tests the pipeline runner with mocked agents to verify:
1. Pipeline returns RunResult with all fields
2. Roots and por_bundle exist
3. Sentinel verification passes when artifacts unchanged
4. Tampered evidence causes sentinel to fail
5. Deterministic timestamp enforcement
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass

# Core imports
from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import PoRRoots, compute_roots, verify_por_bundle
from core.por.reasoning_trace import ReasoningTrace, ReasoningStep, TracePolicy
from core.schemas.evidence import (
    EvidenceBundle, EvidenceItem, SourceDescriptor,
    RetrievalReceipt, ProvenanceProof,
)
from core.schemas.market import (
    MarketSpec, ResolutionWindow, ResolutionRules, ResolutionRule, DisputePolicy,
)
from core.schemas.prompts import (
    PromptSpec, PredictionSemantics, DataRequirement,
    SourceTarget, SelectionPolicy,
)
from core.schemas.transport import ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import VerificationResult, CheckResult

# Orchestrator imports
from orchestrator.pipeline import (
    Pipeline, PipelineConfig, RunResult, PoRPackage,
    create_pipeline,
)
from orchestrator.sop_executor import PipelineState, SOPExecutor, make_step
from orchestrator.market_resolver import resolve_market, MarketResolution


# =============================================================================
# Test Fixtures - Deterministic Test Data
# =============================================================================

def make_market_spec(market_id: str = "mkt_test_001") -> MarketSpec:
    """Create a deterministic MarketSpec for testing."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return MarketSpec(
        market_id=market_id,
        question="Will BTC close above $100,000 on 2026-12-31?",
        event_definition="BTC-USD daily close on 2026-12-31T00:00:00Z > 100000",
        resolution_deadline=datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(rules=[
            ResolutionRule(rule_id="R_BINARY_DECISION", description="Binary yes/no", priority=1),
        ]),
        dispute_policy=DisputePolicy(dispute_window_seconds=3600),
    )


def make_prompt_spec(market_id: str = "mkt_test_001", strict_mode: bool = True) -> PromptSpec:
    """Create a deterministic PromptSpec for testing."""
    return PromptSpec(
        market=make_market_spec(market_id),
        prediction_semantics=PredictionSemantics(
            target_entity="BTC-USD",
            predicate="close_price > threshold",
            threshold="100000",
            timeframe="2026-12-31",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="BTC price from exchange",
                source_targets=[
                    SourceTarget(
                        source_id="exchange",
                        uri="https://api.exchange.example.com/btc-usd",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
        ],
        created_at=None,  # Must be None for deterministic hashing
        extra={"strict_mode": strict_mode, "confidence_policy": {"default_confidence": 0.8}},
    )


def make_tool_plan(market_id: str = "mkt_test_001") -> ToolPlan:
    """Create a deterministic ToolPlan for testing."""
    return ToolPlan(
        plan_id=f"plan_{market_id}",
        requirements=["req_001"],
        sources=["exchange"],
        min_provenance_tier=0,
    )


def make_evidence_bundle(market_id: str = "mkt_test_001") -> EvidenceBundle:
    """Create a deterministic EvidenceBundle for testing."""
    return EvidenceBundle(
        bundle_id=f"bundle_{market_id}",
        collector_id="collector_v1",
        collection_time=None,  # Deterministic
        items=[
            EvidenceItem(
                evidence_id="ev_001",
                requirement_id="req_001",
                source=SourceDescriptor(
                    source_id="exchange",
                    uri="https://api.exchange.example.com/btc-usd",
                    provider="Example Exchange",
                ),
                retrieval=RetrievalReceipt(
                    retrieved_at=None,  # Deterministic
                    method="http_get",
                    request_fingerprint="0x" + "a1" * 32,
                    response_fingerprint="0x" + "b2" * 32,
                ),
                provenance=ProvenanceProof(
                    tier=1,
                    kind="signature",
                ),
                content_type="json",
                content={"price": 105000.0, "timestamp": "2026-12-31T23:59:59Z"},
                normalized={"numeric_value": 105000.0},
                confidence=0.95,
            ),
        ],
    )


def make_reasoning_trace(market_id: str = "mkt_test_001") -> ReasoningTrace:
    """Create a deterministic ReasoningTrace for testing."""
    return ReasoningTrace(
        trace_id=f"trace_{market_id}",
        policy=TracePolicy(max_steps=100),
        steps=[
            ReasoningStep(
                step_id="step_0001",
                type="extract",
                action="Extract price from evidence",
                inputs={"evidence_id": "ev_001"},
                output={"price": 105000.0},
                evidence_ids=["ev_001"],
            ),
            ReasoningStep(
                step_id="step_0002",
                type="check",
                action="Check price against threshold",
                inputs={"price": 105000.0, "threshold": 100000},
                output={"above_threshold": True},
                evidence_ids=["ev_001"],
                prior_step_ids=["step_0001"],
            ),
            ReasoningStep(
                step_id="step_0003",
                type="map",
                action="Map to evaluation variables",
                inputs={"above_threshold": True},
                output={
                    "evaluation_variables": {
                        "event_observed": True,
                        "numeric_value": 105000.0,
                        "conflict_detected": False,
                        "insufficient_evidence": False,
                        "source_summary": ["ev_001"],
                    }
                },
                evidence_ids=["ev_001"],
                prior_step_ids=["step_0002"],
            ),
        ],
        evidence_refs=["ev_001"],
    )


def make_verdict(market_id: str = "mkt_test_001", outcome: str = "YES") -> DeterministicVerdict:
    """Create a deterministic DeterministicVerdict for testing."""
    return DeterministicVerdict(
        market_id=market_id,
        outcome=outcome,
        confidence=0.85,
        resolution_time=None,  # Deterministic
        resolution_rule_id="R_BINARY_DECISION",
    )


# =============================================================================
# Mock Agents
# =============================================================================

class MockPromptEngineer:
    """Mock Prompt Engineer that returns deterministic artifacts."""
    
    def __init__(self, strict_mode: bool = True, created_at: Optional[datetime] = None):
        self.strict_mode = strict_mode
        self.created_at = created_at
        self.call_count = 0
    
    def run(self, user_input: str, *, ctx: Any = None) -> tuple[PromptSpec, ToolPlan]:
        self.call_count += 1
        spec = make_prompt_spec(strict_mode=self.strict_mode)
        if self.created_at is not None:
            # For testing timestamp enforcement - create new spec with timestamp
            spec = PromptSpec(
                market=spec.market,
                prediction_semantics=spec.prediction_semantics,
                data_requirements=spec.data_requirements,
                created_at=self.created_at,
                extra=spec.extra,
            )
        return spec, make_tool_plan()


class MockCollector:
    """Mock Collector that returns deterministic evidence."""
    
    def __init__(self):
        self.call_count = 0
    
    def collect(
        self, prompt_spec: PromptSpec, tool_plan: ToolPlan, *, ctx: Any = None
    ) -> EvidenceBundle:
        self.call_count += 1
        return make_evidence_bundle(prompt_spec.market.market_id)


class MockAuditor:
    """Mock Auditor that returns deterministic trace."""
    
    def __init__(self, verification_ok: bool = True):
        self.verification_ok = verification_ok
        self.call_count = 0
    
    def audit(
        self, prompt_spec: PromptSpec, evidence: EvidenceBundle, *, ctx: Any = None
    ) -> tuple[ReasoningTrace, VerificationResult]:
        self.call_count += 1
        trace = make_reasoning_trace(prompt_spec.market.market_id)
        verification = VerificationResult(
            ok=self.verification_ok,
            checks=[CheckResult.passed("audit_check", "Audit passed")],
        )
        return trace, verification


class MockJudge:
    """Mock Judge that returns deterministic verdict."""
    
    def __init__(self, outcome: str = "YES", verification_ok: bool = True):
        self.outcome = outcome
        self.verification_ok = verification_ok
        self.call_count = 0
    
    def judge(
        self, prompt_spec: PromptSpec, evidence: EvidenceBundle,
        trace: ReasoningTrace, *, ctx: Any = None
    ) -> tuple[DeterministicVerdict, VerificationResult]:
        self.call_count += 1
        verdict = make_verdict(prompt_spec.market.market_id, self.outcome)
        verification = VerificationResult(
            ok=self.verification_ok,
            checks=[CheckResult.passed("judge_check", "Judge passed")],
        )
        return verdict, verification


class MockSentinel:
    """Mock Sentinel that verifies PoR packages."""
    
    def __init__(self, always_pass: bool = False):
        self.always_pass = always_pass
        self.call_count = 0
        self.last_package: Optional[PoRPackage] = None
    
    def verify(
        self, package: PoRPackage, *, mode: str = "verify"
    ) -> tuple[VerificationResult, list[Any]]:
        self.call_count += 1
        self.last_package = package
        
        if self.always_pass:
            return VerificationResult(ok=True, checks=[]), []
        
        # Actually verify the bundle
        result = verify_por_bundle(
            package.bundle,
            prompt_spec=package.prompt_spec,
            evidence=package.evidence,
            trace=package.trace,
        )
        
        challenges = []
        if not result.ok and result.challenge:
            challenges.append({"kind": result.challenge.kind, "reason": result.challenge.reason})
        
        return result, challenges


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipelineConfig:
    """Test PipelineConfig defaults and settings."""
    
    def test_default_config(self):
        config = PipelineConfig()
        assert config.strict_mode is True
        assert config.enable_replay is False
        assert config.enable_sentinel_verify is True
        assert config.deterministic_timestamps is True
    
    def test_custom_config(self):
        config = PipelineConfig(
            strict_mode=False,
            enable_replay=True,
            max_runtime_s=120,
        )
        assert config.strict_mode is False
        assert config.enable_replay is True
        assert config.max_runtime_s == 120


class TestPipelineState:
    """Test PipelineState management."""
    
    def test_initial_state(self):
        state = PipelineState(user_input="test")
        assert state.user_input == "test"
        assert state.ok is True
        assert len(state.checks) == 0
        assert len(state.errors) == 0
    
    def test_add_check_passed(self):
        state = PipelineState(user_input="test")
        state.add_check(CheckResult.passed("test_check", "Test passed"))
        assert len(state.checks) == 1
        assert state.ok is True
    
    def test_add_check_failed(self):
        state = PipelineState(user_input="test")
        state.add_check(CheckResult.failed("test_check", "Test failed"))
        assert len(state.checks) == 1
        assert state.ok is False
    
    def test_add_error(self):
        state = PipelineState(user_input="test")
        state.add_error("Something went wrong")
        assert len(state.errors) == 1
        assert state.ok is False
    
    def test_merge_verification(self):
        state = PipelineState(user_input="test")
        verification = VerificationResult(
            ok=False,
            checks=[CheckResult.failed("v_check", "Verification failed")],
        )
        state.merge_verification(verification)
        assert state.ok is False
        assert len(state.checks) == 1


class TestSOPExecutor:
    """Test SOPExecutor step execution."""
    
    def test_execute_single_step(self):
        executor = SOPExecutor()
        
        def increment_step(state: PipelineState) -> PipelineState:
            state.errors.append("incremented")
            return state
        
        step = make_step("increment", increment_step)
        state = PipelineState(user_input="test")
        result = executor.execute([step], state)
        
        assert "incremented" in result.errors
        assert executor.all_steps_succeeded()
    
    def test_execute_multiple_steps(self):
        executor = SOPExecutor()
        
        steps = [
            make_step("step1", lambda s: s),
            make_step("step2", lambda s: s),
            make_step("step3", lambda s: s),
        ]
        
        state = PipelineState(user_input="test")
        result = executor.execute(steps, state)
        
        assert len(executor.step_results) == 3
        assert executor.all_steps_succeeded()
    
    def test_stop_on_error(self):
        executor = SOPExecutor(stop_on_error=True)
        
        def failing_step(state: PipelineState) -> PipelineState:
            raise ValueError("Step failed")
        
        steps = [
            make_step("step1", lambda s: s),
            make_step("failing", failing_step),
            make_step("step3", lambda s: s),  # Should not execute
        ]
        
        state = PipelineState(user_input="test")
        result = executor.execute(steps, state)
        
        assert len(executor.step_results) == 2
        assert not executor.all_steps_succeeded()
        assert "failing" in executor.get_failed_steps()


class TestPipelineEndToEnd:
    """Test full pipeline execution."""
    
    def test_pipeline_returns_run_result_with_all_fields(self):
        """Verify pipeline returns RunResult with all required fields."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result = pipeline.run("Will BTC close above $100k?")
        
        # Check all fields are populated
        assert result.prompt_spec is not None
        assert result.tool_plan is not None
        assert result.evidence_bundle is not None
        assert result.audit_trace is not None
        assert result.audit_verification is not None
        assert result.verdict is not None
        assert result.judge_verification is not None
        assert result.por_bundle is not None
        assert result.roots is not None
        assert result.ok is True
    
    def test_pipeline_roots_and_por_bundle_exist(self):
        """Verify roots and por_bundle are properly computed."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result = pipeline.run("Test query")
        
        # Verify roots
        assert result.roots is not None
        assert result.roots.prompt_spec_hash.startswith("0x")
        assert result.roots.evidence_root.startswith("0x")
        assert result.roots.reasoning_root.startswith("0x")
        assert result.roots.verdict_hash.startswith("0x")
        assert result.roots.por_root.startswith("0x")
        
        # Verify bundle
        assert result.por_bundle is not None
        assert result.por_bundle.prompt_spec_hash == result.roots.prompt_spec_hash
        assert result.por_bundle.evidence_root == result.roots.evidence_root
        assert result.por_bundle.reasoning_root == result.roots.reasoning_root
        assert result.por_bundle.verdict_hash == result.roots.verdict_hash
        assert result.por_bundle.por_root == result.roots.por_root
    
    def test_pipeline_with_sentinel_verification_passes(self):
        """Verify sentinel verification passes when artifacts unchanged."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=True),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
            sentinel=MockSentinel(),
        )
        
        result = pipeline.run("Test query")
        
        assert result.ok is True
        assert result.sentinel_verification is not None
        assert result.sentinel_verification.ok is True
        assert result.challenges is not None
        assert len(result.challenges) == 0
    
    def test_pipeline_deterministic_roots(self):
        """Verify same inputs produce same roots."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result1 = pipeline.run("Test query")
        result2 = pipeline.run("Test query")
        
        assert result1.roots.por_root == result2.roots.por_root
        assert result1.roots.prompt_spec_hash == result2.roots.prompt_spec_hash
        assert result1.roots.evidence_root == result2.roots.evidence_root
    
    def test_pipeline_verdict_outcome_yes(self):
        """Test pipeline with YES verdict."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(outcome="YES"),
        )
        
        result = pipeline.run("Test")
        
        assert result.verdict.outcome == "YES"
        assert result.outcome == "YES"
        assert result.ok is True
    
    def test_pipeline_verdict_outcome_no(self):
        """Test pipeline with NO verdict."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(outcome="NO"),
        )
        
        result = pipeline.run("Test")
        
        assert result.verdict.outcome == "NO"
        assert result.ok is True
    
    def test_pipeline_verdict_outcome_invalid(self):
        """Test pipeline with INVALID verdict (still ok=True for pipeline)."""
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(outcome="INVALID"),
        )
        
        result = pipeline.run("Test")
        
        assert result.verdict.outcome == "INVALID"
        # Pipeline ok=True because execution succeeded, even if verdict is INVALID
        assert result.ok is True


class TestPipelineStrictMode:
    """Test strict mode enforcement."""
    
    def test_strict_mode_requires_strict_flag(self):
        """Verify strict mode checks PromptSpec.extra['strict_mode']."""
        pipeline = Pipeline(
            config=PipelineConfig(strict_mode=True, enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(strict_mode=False),  # Non-strict
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result = pipeline.run("Test")
        
        # Should have a failed check for strict_mode
        strict_checks = [c for c in result.checks if c.check_id == "strict_mode_check"]
        assert len(strict_checks) == 1
        assert strict_checks[0].ok is False
    
    def test_deterministic_timestamps_enforced(self):
        """Verify timestamps must be None in strict mode."""
        pipeline = Pipeline(
            config=PipelineConfig(
                strict_mode=True,
                deterministic_timestamps=True,
                enable_sentinel_verify=False,
            ),
            prompt_engineer=MockPromptEngineer(
                strict_mode=True,
                created_at=datetime.now(timezone.utc),  # Non-None timestamp
            ),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result = pipeline.run("Test")
        
        # Should have a failed check for timestamp
        timestamp_checks = [c for c in result.checks if "timestamp" in c.check_id]
        assert len(timestamp_checks) == 1
        assert timestamp_checks[0].ok is False


class TestPipelineSentinelIntegration:
    """Test sentinel verification integration."""
    
    def test_sentinel_called_when_enabled(self):
        """Verify sentinel is called when enabled."""
        sentinel = MockSentinel()
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=True),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
            sentinel=sentinel,
        )
        
        result = pipeline.run("Test")
        
        assert sentinel.call_count == 1
        assert sentinel.last_package is not None
    
    def test_sentinel_not_called_when_disabled(self):
        """Verify sentinel is not called when disabled."""
        sentinel = MockSentinel()
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
            sentinel=sentinel,
        )
        
        result = pipeline.run("Test")
        
        assert sentinel.call_count == 0
    
    def test_sentinel_receives_correct_package(self):
        """Verify sentinel receives properly constructed PoRPackage."""
        sentinel = MockSentinel()
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=True),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
            sentinel=sentinel,
        )
        
        result = pipeline.run("Test")
        
        pkg = sentinel.last_package
        assert pkg.bundle is not None
        assert pkg.prompt_spec is not None
        assert pkg.evidence is not None
        assert pkg.trace is not None
        assert pkg.verdict is not None


class TestPipelineTampering:
    """Test that tampering is detected."""
    
    def test_tampered_evidence_detected_by_sentinel(self):
        """Verify tampering with evidence is detected."""
        
        class TamperingSentinel:
            """Sentinel that tampers with evidence before verification."""
            
            def verify(self, package: PoRPackage, *, mode: str = "verify"):
                # Tamper with evidence by modifying an item
                if package.evidence.items:
                    # Create modified evidence
                    tampered_item = EvidenceItem(
                        evidence_id=package.evidence.items[0].evidence_id,
                        requirement_id=package.evidence.items[0].requirement_id,
                        source=package.evidence.items[0].source,
                        retrieval=package.evidence.items[0].retrieval,
                        provenance=package.evidence.items[0].provenance,
                        content_type=package.evidence.items[0].content_type,
                        content={"price": 999999.0},  # TAMPERED!
                        confidence=package.evidence.items[0].confidence,
                    )
                    tampered_evidence = EvidenceBundle(
                        bundle_id=package.evidence.bundle_id,
                        collector_id=package.evidence.collector_id,
                        items=[tampered_item],
                    )
                    
                    # Verify with tampered evidence - should fail
                    result = verify_por_bundle(
                        package.bundle,
                        prompt_spec=package.prompt_spec,
                        evidence=tampered_evidence,  # Use tampered!
                        trace=package.trace,
                    )
                    
                    challenges = []
                    if not result.ok and result.challenge:
                        challenges.append({
                            "kind": result.challenge.kind,
                            "reason": result.challenge.reason,
                        })
                    
                    return result, challenges
                
                return VerificationResult(ok=True, checks=[]), []
        
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=True),
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
            sentinel=TamperingSentinel(),
        )
        
        result = pipeline.run("Test")
        
        # Sentinel should detect the tampering
        assert result.sentinel_verification is not None
        assert result.sentinel_verification.ok is False
        assert len(result.challenges) > 0
        assert result.challenges[0]["kind"] == "evidence_leaf"


class TestMarketResolver:
    """Test market resolver functionality."""
    
    def test_resolve_market_basic(self):
        verdict = make_verdict(outcome="YES")
        resolution = resolve_market(verdict)
        
        assert resolution["market_id"] == "mkt_test_001"
        assert resolution["outcome"] == "YES"
        assert resolution["confidence"] == 0.85
        assert resolution["resolution_rule_id"] == "R_BINARY_DECISION"
    
    def test_market_resolution_dataclass(self):
        from orchestrator.market_resolver import create_resolution
        
        verdict = make_verdict(outcome="NO")
        resolution = create_resolution(verdict)
        
        assert isinstance(resolution, MarketResolution)
        assert resolution.market_id == "mkt_test_001"
        assert resolution.outcome == "NO"


class TestRunResult:
    """Test RunResult properties and methods."""
    
    def test_run_result_to_dict(self):
        result = RunResult(
            verdict=make_verdict(outcome="YES"),
            por_bundle=None,  # Simplified
            ok=True,
        )
        
        d = result.to_dict()
        
        assert d["ok"] is True
        assert d["market_id"] == "mkt_test_001"
        assert d["outcome"] == "YES"
        assert d["confidence"] == 0.85
    
    def test_run_result_properties(self):
        verdict = make_verdict(outcome="NO")
        result = RunResult(verdict=verdict, ok=True)
        
        assert result.market_id == "mkt_test_001"
        assert result.outcome == "NO"


class TestCreatePipelineConvenience:
    """Test create_pipeline convenience function."""
    
    def test_create_pipeline_basic(self):
        pipeline = create_pipeline(
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        assert pipeline.config.strict_mode is True
        assert pipeline.config.enable_sentinel_verify is False  # No sentinel provided
    
    def test_create_pipeline_with_sentinel(self):
        pipeline = create_pipeline(
            prompt_engineer=MockPromptEngineer(),
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
            sentinel=MockSentinel(),
            enable_sentinel=True,
        )
        
        assert pipeline.config.enable_sentinel_verify is True


class TestPipelineMissingAgents:
    """Test pipeline behavior with missing agents."""
    
    def test_missing_prompt_engineer(self):
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            # No prompt_engineer
            collector=MockCollector(),
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result = pipeline.run("Test")
        
        # Should fail at build_por_bundle due to missing artifacts
        assert result.ok is False
        assert len(result.errors) > 0
    
    def test_missing_collector(self):
        pipeline = Pipeline(
            config=PipelineConfig(enable_sentinel_verify=False),
            prompt_engineer=MockPromptEngineer(),
            # No collector
            auditor=MockAuditor(),
            judge=MockJudge(),
        )
        
        result = pipeline.run("Test")
        
        assert result.ok is False


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])