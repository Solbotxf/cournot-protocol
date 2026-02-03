"""
Tests for Sentinel Agent

Tests both strict and basic sentinels.
"""

import pytest
import json
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from agents.sentinel import (
    SentinelStrict,
    SentinelBasic,
    VerificationEngine,
    build_proof_bundle,
    get_sentinel,
    verify_artifacts,
    verify_proof,
)
from core.schemas import (
    DataRequirement,
    DeterministicVerdict,
    DisputePolicy,
    EvidenceBundle,
    EvidenceItem,
    EvidenceRef,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    ProofBundle,
    Provenance,
    ReasoningStep,
    ReasoningTrace,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SentinelReport,
    SourceTarget,
    ToolPlan,
    VerificationResult,
)


def create_test_artifacts():
    """Create a complete set of test artifacts."""
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    market_id = "mk_test123"
    
    # PromptSpec
    prompt_spec = PromptSpec(
        market=MarketSpec(
            market_id=market_id,
            question="Will BTC be above $100k?",
            event_definition="price(BTC_USD) > 100000",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R_THRESHOLD", description="Check threshold", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="bitcoin",
            predicate="price above threshold",
            threshold="100000",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(source_id="coingecko", uri="https://api.coingecko.com/price"),
                ],
                selection_policy=SelectionPolicy(strategy="single_best", min_sources=1, max_sources=1, quorum=1),
            ),
        ],
    )
    
    # ToolPlan
    tool_plan = ToolPlan(
        plan_id="plan_test123",
        requirements=["req_001"],
        sources=["coingecko"],
        min_provenance_tier=0,
    )
    
    # EvidenceBundle
    evidence_bundle = EvidenceBundle(
        bundle_id="bundle_test123",
        market_id=market_id,
        plan_id="plan_test123",
    )
    evidence_bundle.add_item(EvidenceItem(
        evidence_id="ev_001",
        requirement_id="req_001",
        provenance=Provenance(
            source_id="coingecko",
            source_uri="https://api.coingecko.com/price",
            tier=3,
        ),
        success=True,
        extracted_fields={"price_usd": 95000},
    ))
    
    # ReasoningTrace
    reasoning_trace = ReasoningTrace(
        trace_id="trace_test123",
        market_id=market_id,
        bundle_id="bundle_test123",
        steps=[
            ReasoningStep(
                step_id="step_001",
                step_type="threshold_check",
                description="Compare price to threshold",
                evidence_refs=[EvidenceRef(evidence_id="ev_001")],
                conclusion="95000 < 100000",
            ),
            ReasoningStep(
                step_id="step_002",
                step_type="conclusion",
                description="Final conclusion",
                conclusion="Price below threshold",
            ),
        ],
        preliminary_outcome="NO",
        preliminary_confidence=0.75,
        recommended_rule_id="R_THRESHOLD",
    )
    
    # Compute hashes for verdict (must match what VerdictBuilder produces)
    from core.schemas.canonical import dumps_canonical
    import hashlib
    
    spec_dict = prompt_spec.model_dump(exclude={"created_at"})
    prompt_spec_hash = f"0x{hashlib.sha256(dumps_canonical(spec_dict).encode()).hexdigest()}"
    
    evidence_ids = sorted([item.evidence_id for item in evidence_bundle.items])
    evidence_root = f"0x{hashlib.sha256('|'.join(evidence_ids).encode()).hexdigest()}"
    
    step_data = [{"step_id": s.step_id, "step_type": s.step_type, "conclusion": s.conclusion} for s in reasoning_trace.steps]
    reasoning_root = f"0x{hashlib.sha256(dumps_canonical(step_data).encode()).hexdigest()}"
    
    # DeterministicVerdict
    verdict = DeterministicVerdict(
        market_id=market_id,
        outcome="NO",
        confidence=0.75,
        resolution_rule_id="R_THRESHOLD",
        prompt_spec_hash=prompt_spec_hash,
        evidence_root=evidence_root,
        reasoning_root=reasoning_root,
        selected_leaf_refs=["ev_001"],
    )
    
    return prompt_spec, tool_plan, evidence_bundle, reasoning_trace, verdict


def create_test_proof_bundle():
    """Create a complete test ProofBundle."""
    prompt_spec, tool_plan, evidence_bundle, reasoning_trace, verdict = create_test_artifacts()
    
    return ProofBundle(
        bundle_id="proof_test123",
        market_id="mk_test123",
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence_bundle=evidence_bundle,
        reasoning_trace=reasoning_trace,
        verdict=verdict,
    )


class TestVerificationEngine:
    """Tests for VerificationEngine."""
    
    def test_verify_complete_bundle(self):
        """Test verification of a complete valid bundle."""
        ctx = AgentContext.create_minimal()
        engine = VerificationEngine()
        
        bundle = create_test_proof_bundle()
        
        result, report = engine.verify(ctx, bundle)
        
        assert result.ok
        assert report.verified
        assert report.passed_checks > 0
        assert report.failed_checks == 0
    
    def test_completeness_checks(self):
        """Test completeness check detection."""
        ctx = AgentContext.create_minimal()
        engine = VerificationEngine()
        
        bundle = create_test_proof_bundle()
        
        result, report = engine.verify(ctx, bundle)
        
        # Check that completeness checks were performed
        assert len(report.completeness_checks) > 0
        assert all(c["passed"] for c in report.completeness_checks)
    
    def test_consistency_checks(self):
        """Test consistency check detection."""
        ctx = AgentContext.create_minimal()
        engine = VerificationEngine()
        
        bundle = create_test_proof_bundle()
        
        result, report = engine.verify(ctx, bundle)
        
        # Check that consistency checks were performed
        assert len(report.consistency_checks) > 0
    
    def test_detects_market_id_mismatch(self):
        """Test that market ID mismatches are detected."""
        ctx = AgentContext.create_minimal()
        engine = VerificationEngine()
        
        prompt_spec, tool_plan, evidence_bundle, reasoning_trace, verdict = create_test_artifacts()
        
        # Create bundle with mismatched market_id in verdict
        bad_verdict = DeterministicVerdict(
            market_id="mk_wrong",  # Wrong market ID
            outcome="NO",
            confidence=0.75,
            resolution_rule_id="R_THRESHOLD",
        )
        
        bundle = ProofBundle(
            bundle_id="proof_bad",
            market_id="mk_test123",
            prompt_spec=prompt_spec,
            tool_plan=tool_plan,
            evidence_bundle=evidence_bundle,
            reasoning_trace=reasoning_trace,
            verdict=bad_verdict,
        )
        
        result, report = engine.verify(ctx, bundle)
        
        assert not result.ok
        assert not report.verified
        assert report.failed_checks > 0


class TestSentinelStrict:
    """Tests for SentinelStrict agent."""
    
    def test_run_success(self):
        """Test successful verification."""
        ctx = AgentContext.create_minimal()
        
        bundle = create_test_proof_bundle()
        
        sentinel = SentinelStrict()
        result = sentinel.run(ctx, bundle)
        
        assert result.success
        assert result.output is not None
        
        verification_result, report = result.output
        assert verification_result.ok
        assert report.verified
    
    def test_metadata_included(self):
        """Test that metadata is included."""
        ctx = AgentContext.create_minimal()
        
        bundle = create_test_proof_bundle()
        
        sentinel = SentinelStrict()
        result = sentinel.run(ctx, bundle)
        
        assert result.metadata["sentinel"] == "strict"
        assert "verified" in result.metadata
        assert "total_checks" in result.metadata
        assert "pass_rate" in result.metadata
    
    def test_run_from_artifacts(self):
        """Test run_from_artifacts convenience method."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec, tool_plan, evidence_bundle, reasoning_trace, verdict = create_test_artifacts()
        
        sentinel = SentinelStrict()
        result = sentinel.run_from_artifacts(
            ctx, prompt_spec, tool_plan, evidence_bundle,
            reasoning_trace, verdict
        )
        
        assert result.success


class TestSentinelBasic:
    """Tests for SentinelBasic agent."""
    
    def test_run_success(self):
        """Test successful basic verification."""
        ctx = AgentContext.create_minimal()
        
        bundle = create_test_proof_bundle()
        
        sentinel = SentinelBasic()
        result = sentinel.run(ctx, bundle)
        
        assert result.success
        assert result.metadata["sentinel"] == "basic"
    
    def test_faster_than_strict(self):
        """Test that basic is faster (fewer checks)."""
        ctx = AgentContext.create_minimal()
        
        bundle = create_test_proof_bundle()
        
        strict = SentinelStrict()
        basic = SentinelBasic()
        
        strict_result = strict.run(ctx, bundle)
        basic_result = basic.run(ctx, bundle)
        
        _, strict_report = strict_result.output
        _, basic_report = basic_result.output
        
        # Basic should have fewer checks
        assert basic_report.total_checks < strict_report.total_checks


class TestGetSentinel:
    """Tests for get_sentinel function."""
    
    def test_returns_strict_by_default(self):
        """Test that strict sentinel is returned by default."""
        ctx = AgentContext.create_minimal()
        
        sentinel = get_sentinel(ctx)
        
        assert isinstance(sentinel, SentinelStrict)
    
    def test_returns_basic_when_requested(self):
        """Test that basic sentinel is returned when requested."""
        ctx = AgentContext.create_minimal()
        
        sentinel = get_sentinel(ctx, strict=False)
        
        assert isinstance(sentinel, SentinelBasic)


class TestVerifyProof:
    """Tests for verify_proof convenience function."""
    
    def test_verify_valid_bundle(self):
        """Test verifying a valid proof bundle."""
        ctx = AgentContext.create_minimal()
        
        bundle = create_test_proof_bundle()
        
        result = verify_proof(ctx, bundle)
        
        assert result.success
        verification_result, report = result.output
        assert verification_result.ok


class TestVerifyArtifacts:
    """Tests for verify_artifacts convenience function."""
    
    def test_verify_valid_artifacts(self):
        """Test verifying valid artifacts."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec, tool_plan, evidence_bundle, reasoning_trace, verdict = create_test_artifacts()
        
        result = verify_artifacts(
            ctx, prompt_spec, tool_plan, evidence_bundle,
            reasoning_trace, verdict
        )
        
        assert result.success


class TestBuildProofBundle:
    """Tests for build_proof_bundle function."""
    
    def test_builds_complete_bundle(self):
        """Test building a complete proof bundle."""
        prompt_spec, tool_plan, evidence_bundle, reasoning_trace, verdict = create_test_artifacts()
        
        bundle = build_proof_bundle(
            prompt_spec, tool_plan, evidence_bundle,
            reasoning_trace, verdict
        )
        
        assert bundle.is_complete
        assert bundle.market_id == verdict.market_id
        assert bundle.artifact_count == 5  # 5 required + no execution_log


class TestAgentRegistration:
    """Tests for agent registration."""
    
    def test_agents_registered(self):
        """Test that sentinel agents are registered."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.SENTINEL)
        names = [a.name for a in agents]
        
        assert "SentinelStrict" in names
        assert "SentinelBasic" in names
    
    def test_strict_agent_higher_priority(self):
        """Test that strict agent has higher priority."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.SENTINEL)
        
        strict_agent = next(a for a in agents if a.name == "SentinelStrict")
        basic_agent = next(a for a in agents if a.name == "SentinelBasic")
        
        assert strict_agent.priority > basic_agent.priority


class TestSentinelReport:
    """Tests for SentinelReport schema."""
    
    def test_add_check(self):
        """Test adding checks to report."""
        report = SentinelReport(
            report_id="test_report",
            bundle_id="test_bundle",
            market_id="mk_test",
            verified=True,
        )
        
        report.add_check("completeness", "test_check", True, "Test passed")
        
        assert report.total_checks == 1
        assert report.passed_checks == 1
        assert report.failed_checks == 0
    
    def test_add_failed_check(self):
        """Test that failed checks update verified status."""
        report = SentinelReport(
            report_id="test_report",
            bundle_id="test_bundle",
            market_id="mk_test",
            verified=True,
        )
        
        report.add_check("completeness", "test_check", False, "Test failed")
        
        assert not report.verified
        assert report.failed_checks == 1
        assert len(report.errors) == 1
    
    def test_pass_rate(self):
        """Test pass rate calculation."""
        report = SentinelReport(
            report_id="test_report",
            bundle_id="test_bundle",
            market_id="mk_test",
            verified=True,
        )
        
        report.add_check("completeness", "check1", True, "Passed")
        report.add_check("completeness", "check2", True, "Passed")
        report.add_check("completeness", "check3", False, "Failed")
        
        assert report.pass_rate == pytest.approx(2/3)


class TestIntegration:
    """Integration tests for full pipeline through sentinel."""
    
    def test_full_pipeline_verification(self):
        """Test full pipeline from question to verified proof."""
        import json
        from agents.prompt_engineer import compile_prompt
        from agents.collector import CollectorMock
        from agents.auditor import AuditorRuleBased
        from agents.judge import JudgeRuleBased

        mock_response = json.dumps({
            "market_id": "mk_btc_test",
            "question": "Will BTC be above $100k?",
            "event_definition": "price(BTC_USD) > 100000",
            "target_entity": "bitcoin",
            "predicate": "price above threshold",
            "threshold": "100000",
            "resolution_window": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-12-31T23:59:59Z",
            },
            "resolution_deadline": "2025-12-31T23:59:59Z",
            "data_requirements": [{
                "requirement_id": "req_001",
                "description": "Get BTC price",
                "source_targets": [{
                    "source_id": "coingecko",
                    "uri": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                    "method": "GET",
                    "expected_content_type": "json",
                }],
                "selection_policy": {
                    "strategy": "single_best",
                    "min_sources": 1,
                    "max_sources": 1,
                    "quorum": 1,
                },
            }],
            "resolution_rules": [
                {"rule_id": "R_VALIDITY", "description": "Check validity", "priority": 100},
                {"rule_id": "R_THRESHOLD", "description": "Compare to threshold", "priority": 80},
                {"rule_id": "R_INVALID_FALLBACK", "description": "Fallback", "priority": 0},
            ],
            "allowed_sources": [
                {"source_id": "coingecko", "kind": "api", "allow": True},
            ],
        })
        ctx = AgentContext.create_mock(llm_responses=[mock_response])

        # Step 1: Compile prompt
        prompt_result = compile_prompt(ctx, "Will BTC be above $100k?")
        assert prompt_result.success
        prompt_spec, tool_plan = prompt_result.output
        
        # Step 2: Collect evidence (mock)
        collector = CollectorMock(mock_responses={
            "coingecko": {"bitcoin": {"usd": 95000}},
        })
        collect_result = collector.run(ctx, prompt_spec, tool_plan)
        assert collect_result.success
        evidence_bundle, execution_log = collect_result.output
        
        # Step 3: Audit evidence
        auditor = AuditorRuleBased()
        audit_result = auditor.run(ctx, prompt_spec, evidence_bundle)
        assert audit_result.success
        reasoning_trace = audit_result.output
        
        # Step 4: Judge verdict
        judge = JudgeRuleBased()
        judge_result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        assert judge_result.success
        verdict = judge_result.output
        
        # Step 5: Verify with Sentinel
        bundle = build_proof_bundle(
            prompt_spec, tool_plan, evidence_bundle,
            reasoning_trace, verdict, execution_log
        )
        
        sentinel = SentinelStrict()
        sentinel_result = sentinel.run(ctx, bundle)
        
        assert sentinel_result.success
        verification_result, report = sentinel_result.output
        
        # Full pipeline should pass verification
        assert verification_result.ok
        assert report.verified
        assert report.pass_rate > 0.8  # Most checks should pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
