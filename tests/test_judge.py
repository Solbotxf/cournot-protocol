"""
Tests for Judge Agent

Tests both LLM-based and rule-based judges.
"""

import pytest
import json
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from agents.judge import (
    JudgeLLM,
    JudgeRuleBased,
    VerdictBuilder,
    VerdictValidator,
    get_judge,
    judge_verdict,
)
from core.schemas import (
    ConflictRecord,
    DataRequirement,
    DeterministicVerdict,
    DisputePolicy,
    EvidenceBundle,
    EvidenceItem,
    EvidenceRef,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    Provenance,
    ReasoningStep,
    ReasoningTrace,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourceTarget,
)


def create_test_prompt_spec() -> PromptSpec:
    """Create a test PromptSpec."""
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_test123",
            question="Will BTC be above $100k?",
            event_definition="price(BTC_USD) > 100000",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R_VALIDITY", description="Check validity", priority=100),
                ResolutionRule(rule_id="R_THRESHOLD", description="Compare to threshold", priority=80),
                ResolutionRule(rule_id="R_INVALID_FALLBACK", description="Fallback", priority=0),
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
                    SourceTarget(
                        source_id="coingecko",
                        uri="https://api.coingecko.com/api/v3/simple/price",
                    ),
                ],
                selection_policy=SelectionPolicy(strategy="single_best", min_sources=1, max_sources=1, quorum=1),
            ),
        ],
    )


def create_test_evidence_bundle() -> EvidenceBundle:
    """Create a test EvidenceBundle."""
    bundle = EvidenceBundle(
        bundle_id="bundle_test123",
        market_id="mk_test123",
        plan_id="plan_test123",
    )
    
    bundle.add_item(EvidenceItem(
        evidence_id="ev_001",
        requirement_id="req_001",
        provenance=Provenance(
            source_id="coingecko",
            source_uri="https://api.coingecko.com/api/v3/simple/price",
            tier=3,
        ),
        success=True,
        extracted_fields={"price_usd": 95000},
    ))
    
    return bundle


def create_test_reasoning_trace(
    outcome: str = "NO",
    confidence: float = 0.75,
) -> ReasoningTrace:
    """Create a test ReasoningTrace."""
    return ReasoningTrace(
        trace_id="trace_test123",
        market_id="mk_test123",
        bundle_id="bundle_test123",
        steps=[
            ReasoningStep(
                step_id="step_001",
                step_type="validity_check",
                description="Check evidence validity",
                conclusion="Evidence is valid",
            ),
            ReasoningStep(
                step_id="step_002",
                step_type="threshold_check",
                description="Compare price to threshold",
                evidence_refs=[EvidenceRef(evidence_id="ev_001", value_at_reference=95000)],
                rule_id="R_THRESHOLD",
                conclusion="95000 < 100000, threshold not met",
            ),
            ReasoningStep(
                step_id="step_003",
                step_type="conclusion",
                description="Final conclusion",
                conclusion="Price is below threshold",
            ),
        ],
        evidence_summary="Analyzed 1 evidence item from CoinGecko",
        reasoning_summary="Compared BTC price to $100k threshold",
        preliminary_outcome=outcome,
        preliminary_confidence=confidence,
        recommended_rule_id="R_THRESHOLD",
    )


class TestVerdictBuilder:
    """Tests for VerdictBuilder."""
    
    def test_build_basic_verdict(self):
        """Test building a basic verdict."""
        ctx = AgentContext.create_minimal()
        builder = VerdictBuilder()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        verdict = builder.build(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert verdict.market_id == "mk_test123"
        assert verdict.outcome == "NO"
        assert verdict.confidence == 0.75
        assert verdict.resolution_rule_id == "R_THRESHOLD"
        assert verdict.prompt_spec_hash is not None
        assert verdict.evidence_root is not None
        assert verdict.reasoning_root is not None
    
    def test_build_with_override(self):
        """Test building with overrides."""
        ctx = AgentContext.create_minimal()
        builder = VerdictBuilder()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        verdict = builder.build(
            ctx, prompt_spec, evidence_bundle, reasoning_trace,
            override_outcome="YES",
            override_confidence=0.9,
            override_rule_id="R_CUSTOM",
        )
        
        assert verdict.outcome == "YES"
        assert verdict.confidence == 0.9
        assert verdict.resolution_rule_id == "R_CUSTOM"
    
    def test_strict_mode_excludes_timestamp(self):
        """Test that strict mode excludes resolution_time."""
        ctx = AgentContext.create_minimal()
        builder = VerdictBuilder(strict_mode=True)
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        verdict = builder.build(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert verdict.resolution_time is None
    
    def test_uncertain_becomes_invalid(self):
        """Test that UNCERTAIN outcome becomes INVALID."""
        ctx = AgentContext.create_minimal()
        builder = VerdictBuilder()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace(outcome="UNCERTAIN", confidence=0.4)
        
        verdict = builder.build(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert verdict.outcome == "INVALID"


class TestVerdictValidator:
    """Tests for VerdictValidator."""
    
    def test_valid_verdict(self):
        """Test validation of a valid verdict."""
        ctx = AgentContext.create_minimal()
        builder = VerdictBuilder()
        validator = VerdictValidator()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        verdict = builder.build(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        is_valid, errors = validator.validate(verdict, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_market_id(self):
        """Test validation catches market ID mismatch."""
        validator = VerdictValidator()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        # Create verdict with wrong market ID
        verdict = DeterministicVerdict(
            market_id="mk_wrong",
            outcome="NO",
            confidence=0.75,
            resolution_rule_id="R_THRESHOLD",
            prompt_spec_hash="0x123",
            evidence_root="0x456",
            reasoning_root="0x789",
        )
        
        is_valid, errors = validator.validate(verdict, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert not is_valid
        assert any("Market ID mismatch" in e for e in errors)


class TestJudgeRuleBased:
    """Tests for JudgeRuleBased agent."""
    
    def test_run_success(self):
        """Test successful verdict finalization."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        judge = JudgeRuleBased()
        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert result.success
        assert result.output is not None
        assert isinstance(result.output, DeterministicVerdict)
        assert result.output.outcome == "NO"
    
    def test_low_confidence_becomes_invalid(self):
        """Test that low confidence leads to INVALID."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace(outcome="YES", confidence=0.4)
        
        judge = JudgeRuleBased()
        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert result.success
        assert result.output.outcome == "INVALID"
    
    def test_metadata_included(self):
        """Test that metadata is included."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        judge = JudgeRuleBased()
        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert result.metadata["judge"] == "rule_based"
        assert "outcome" in result.metadata
        assert "confidence" in result.metadata
    
    def test_no_evidence_leads_to_invalid(self):
        """Test that no evidence leads to INVALID."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = EvidenceBundle(
            bundle_id="bundle_empty",
            market_id="mk_test123",
            plan_id="plan_test123",
        )
        reasoning_trace = create_test_reasoning_trace(outcome="YES", confidence=0.8)
        
        judge = JudgeRuleBased()
        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert result.success
        assert result.output.outcome == "INVALID"


class TestJudgeLLM:
    """Tests for JudgeLLM agent."""
    
    def test_requires_llm(self):
        """Test that LLM judge requires LLM client."""
        ctx = AgentContext.create_minimal()  # No LLM
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        judge = JudgeLLM()
        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert not result.success
        assert "LLM client required" in result.error
    
    def test_run_with_mock_llm(self):
        """Test run with mocked LLM responses."""
        mock_response = json.dumps({
            "outcome": "NO",
            "confidence": 0.85,
            "resolution_rule_id": "R_THRESHOLD",
            "reasoning_valid": True,
            "reasoning_issues": [],
            "confidence_adjustments": [
                {"reason": "High tier source", "delta": 0.1}
            ],
            "final_justification": "Price 95k is below 100k threshold",
        })
        
        ctx = AgentContext.create_mock(llm_responses=[mock_response])
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        judge = JudgeLLM()
        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert result.success
        assert result.output is not None
        assert result.output.outcome == "NO"
        assert result.output.confidence == 0.85


class TestGetJudge:
    """Tests for get_judge function."""
    
    def test_returns_llm_when_available(self):
        """Test that LLM judge is returned when LLM is available."""
        ctx = AgentContext.create_mock(llm_responses=["{}"])
        
        judge = get_judge(ctx, prefer_llm=True)
        
        assert isinstance(judge, JudgeLLM)
    
    def test_returns_rule_based_when_no_llm(self):
        """Test that rule-based is returned when no LLM."""
        ctx = AgentContext.create_minimal()
        
        judge = get_judge(ctx, prefer_llm=True)
        
        assert isinstance(judge, JudgeRuleBased)
    
    def test_returns_rule_based_when_preferred(self):
        """Test that rule-based is returned when explicitly preferred."""
        ctx = AgentContext.create_mock(llm_responses=["{}"])
        
        judge = get_judge(ctx, prefer_llm=False)
        
        assert isinstance(judge, JudgeRuleBased)


class TestJudgeVerdict:
    """Tests for judge_verdict convenience function."""
    
    def test_judge_with_rule_based(self):
        """Test judge_verdict uses rule-based when no LLM."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        reasoning_trace = create_test_reasoning_trace()
        
        result = judge_verdict(ctx, prompt_spec, evidence_bundle, reasoning_trace)
        
        assert result.success
        assert result.metadata["judge"] == "rule_based"


class TestAgentRegistration:
    """Tests for agent registration."""
    
    def test_agents_registered(self):
        """Test that judge agents are registered."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.JUDGE)
        names = [a.name for a in agents]
        
        assert "JudgeLLM" in names
        assert "JudgeRuleBased" in names
    
    def test_llm_agent_higher_priority(self):
        """Test that LLM agent has higher priority."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.JUDGE)
        
        llm_agent = next(a for a in agents if a.name == "JudgeLLM")
        rule_agent = next(a for a in agents if a.name == "JudgeRuleBased")
        
        assert llm_agent.priority > rule_agent.priority


class TestDeterministicVerdict:
    """Tests for DeterministicVerdict schema."""
    
    def test_is_definitive(self):
        """Test is_definitive property."""
        verdict_yes = DeterministicVerdict(
            market_id="mk_test",
            outcome="YES",
            confidence=0.8,
            resolution_rule_id="R_TEST",
        )
        assert verdict_yes.is_definitive
        
        verdict_no = DeterministicVerdict(
            market_id="mk_test",
            outcome="NO",
            confidence=0.8,
            resolution_rule_id="R_TEST",
        )
        assert verdict_no.is_definitive
        
        verdict_invalid = DeterministicVerdict(
            market_id="mk_test",
            outcome="INVALID",
            confidence=0.5,
            resolution_rule_id="R_TEST",
        )
        assert not verdict_invalid.is_definitive
    
    def test_is_invalid(self):
        """Test is_invalid property."""
        verdict = DeterministicVerdict(
            market_id="mk_test",
            outcome="INVALID",
            confidence=0.5,
            resolution_rule_id="R_TEST",
        )
        assert verdict.is_invalid


class TestIntegration:
    """Integration tests for full pipeline through judge."""
    
    def test_full_pipeline_with_rule_based(self):
        """Test full pipeline from question to verdict."""
        import json
        from agents.prompt_engineer import compile_prompt
        from agents.collector import CollectorMock
        from agents.auditor import AuditorRuleBased

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
        evidence_bundle, _ = collect_result.output
        
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
        
        # With price 95k < 100k threshold → NO
        assert verdict.outcome == "NO"
        assert verdict.confidence >= 0.55
        assert verdict.market_id == prompt_spec.market_id
        assert verdict.prompt_spec_hash is not None
        assert verdict.evidence_root is not None
        assert verdict.reasoning_root is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
