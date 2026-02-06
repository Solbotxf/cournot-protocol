"""
Tests for Auditor Agent

Tests both LLM-based and rule-based auditors.
"""

import pytest
import json
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from agents.auditor import (
    AuditorLLM,
    AuditorRuleBased,
    RuleBasedReasoner,
    audit_evidence,
    get_auditor,
)
from core.schemas import (
    DataRequirement,
    DisputePolicy,
    EvidenceBundle,
    EvidenceItem,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    Provenance,
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


def create_test_evidence_bundle(price: float = 95000) -> EvidenceBundle:
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
        raw_content=json.dumps({"bitcoin": {"usd": price}}),
        content_type="application/json",
        parsed_value={"bitcoin": {"usd": price}},
        success=True,
        status_code=200,
        extracted_fields={"bitcoin_usd": price, "price_usd": price},
    ))
    
    return bundle


class TestRuleBasedReasoner:
    """Tests for RuleBasedReasoner."""
    
    def test_reason_above_threshold(self):
        """Test reasoning when price is above threshold."""
        ctx = AgentContext.create_minimal()
        reasoner = RuleBasedReasoner()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle(price=105000)  # Above 100k
        
        trace = reasoner.reason(ctx, prompt_spec, [evidence_bundle])
        
        assert trace.preliminary_outcome == "YES"
        assert trace.preliminary_confidence > 0.5
        assert trace.step_count >= 3
    
    def test_reason_below_threshold(self):
        """Test reasoning when price is below threshold."""
        ctx = AgentContext.create_minimal()
        reasoner = RuleBasedReasoner()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle(price=95000)  # Below 100k
        
        trace = reasoner.reason(ctx, prompt_spec, [evidence_bundle])
        
        assert trace.preliminary_outcome == "NO"
        assert trace.step_count >= 3
    
    def test_reason_no_evidence(self):
        """Test reasoning with no evidence."""
        ctx = AgentContext.create_minimal()
        reasoner = RuleBasedReasoner()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = EvidenceBundle(
            bundle_id="bundle_empty",
            market_id="mk_test123",
            plan_id="plan_test123",
        )
        
        trace = reasoner.reason(ctx, prompt_spec, [evidence_bundle])
        
        assert trace.preliminary_outcome == "INVALID"
        assert trace.preliminary_confidence < 0.5
    
    def test_conflict_detection(self):
        """Test that conflicts are detected."""
        ctx = AgentContext.create_minimal()
        reasoner = RuleBasedReasoner()
        
        prompt_spec = create_test_prompt_spec()
        
        # Create bundle with conflicting evidence
        bundle = EvidenceBundle(
            bundle_id="bundle_conflict",
            market_id="mk_test123",
            plan_id="plan_test123",
        )
        
        # Two sources with >10% difference
        bundle.add_item(EvidenceItem(
            evidence_id="ev_001",
            requirement_id="req_001",
            provenance=Provenance(source_id="source1", source_uri="https://source1.com", tier=2),
            success=True,
            extracted_fields={"price_usd": 100000},
        ))
        
        bundle.add_item(EvidenceItem(
            evidence_id="ev_002",
            requirement_id="req_001",
            provenance=Provenance(source_id="source2", source_uri="https://source2.com", tier=3),
            success=True,
            extracted_fields={"price_usd": 120000},  # 20% higher
        ))
        
        trace = reasoner.reason(ctx, prompt_spec, [bundle])
        
        assert trace.has_conflicts
        assert len(trace.conflicts) >= 1
        # Higher tier should win
        assert trace.conflicts[0].winning_evidence_id == "ev_002"
    
    def test_strict_mode_excludes_timestamp(self):
        """Test that strict mode excludes created_at."""
        ctx = AgentContext.create_minimal()
        reasoner = RuleBasedReasoner(strict_mode=True)
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        
        trace = reasoner.reason(ctx, prompt_spec, [evidence_bundle])
        
        assert trace.created_at is None


class TestAuditorRuleBased:
    """Tests for AuditorRuleBased agent."""
    
    def test_run_success(self):
        """Test successful audit."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        
        auditor = AuditorRuleBased()
        result = auditor.run(ctx, prompt_spec, evidence_bundle)
        
        assert result.success
        assert result.output is not None
        assert isinstance(result.output, ReasoningTrace)
    
    def test_metadata_included(self):
        """Test that metadata is included."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        
        auditor = AuditorRuleBased()
        result = auditor.run(ctx, prompt_spec, evidence_bundle)
        
        assert result.metadata["auditor"] == "rule_based"
        assert "trace_id" in result.metadata
        assert "step_count" in result.metadata
        assert "preliminary_outcome" in result.metadata


class TestAuditorLLM:
    """Tests for AuditorLLM agent."""
    
    def test_requires_llm(self):
        """Test that LLM auditor requires LLM client."""
        ctx = AgentContext.create_minimal()  # No LLM
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        
        auditor = AuditorLLM()
        result = auditor.run(ctx, prompt_spec, evidence_bundle)
        
        assert not result.success
        assert "LLM client required" in result.error
    
    def test_run_with_mock_llm(self):
        """Test run with mocked LLM responses."""
        mock_response = json.dumps({
            "trace_id": "trace_mock123",
            "evidence_summary": "One evidence item with BTC price",
            "reasoning_summary": "Analyzed price and compared to threshold",
            "steps": [
                {
                    "step_id": "step_001",
                    "step_type": "evidence_analysis",
                    "description": "Analyze BTC price from CoinGecko",
                    "evidence_refs": [{"evidence_id": "ev_001", "value_at_reference": 95000}],
                },
                {
                    "step_id": "step_002",
                    "step_type": "threshold_check",
                    "description": "Compare price to threshold",
                    "rule_id": "R_THRESHOLD",
                    "conclusion": "95000 < 100000",
                },
                {
                    "step_id": "step_003",
                    "step_type": "conclusion",
                    "description": "Draw final conclusion",
                    "conclusion": "Price is below threshold",
                },
            ],
            "conflicts": [],
            "preliminary_outcome": "NO",
            "preliminary_confidence": 0.85,
            "recommended_rule_id": "R_THRESHOLD",
        })
        
        ctx = AgentContext.create_mock(llm_responses=[mock_response])
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        
        auditor = AuditorLLM()
        result = auditor.run(ctx, prompt_spec, evidence_bundle)
        
        assert result.success
        assert result.output is not None
        
        trace = result.output
        assert trace.trace_id == "trace_mock123"
        assert trace.preliminary_outcome == "NO"
        assert len(trace.steps) == 3


class TestGetAuditor:
    """Tests for get_auditor function."""
    
    def test_returns_llm_when_available(self):
        """Test that LLM auditor is returned when LLM is available."""
        ctx = AgentContext.create_mock(llm_responses=["{}"])
        
        auditor = get_auditor(ctx, prefer_llm=True)
        
        assert isinstance(auditor, AuditorLLM)
    
    def test_returns_rule_based_when_no_llm(self):
        """Test that rule-based is returned when no LLM."""
        ctx = AgentContext.create_minimal()
        
        auditor = get_auditor(ctx, prefer_llm=True)
        
        assert isinstance(auditor, AuditorRuleBased)
    
    def test_returns_rule_based_when_preferred(self):
        """Test that rule-based is returned when explicitly preferred."""
        ctx = AgentContext.create_mock(llm_responses=["{}"])
        
        auditor = get_auditor(ctx, prefer_llm=False)
        
        assert isinstance(auditor, AuditorRuleBased)


class TestAuditEvidence:
    """Tests for audit_evidence convenience function."""
    
    def test_audit_with_rule_based(self):
        """Test audit_evidence uses rule-based when no LLM."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        evidence_bundle = create_test_evidence_bundle()
        
        result = audit_evidence(ctx, prompt_spec, evidence_bundle)
        
        assert result.success
        assert result.metadata["auditor"] == "rule_based"


class TestAgentRegistration:
    """Tests for agent registration."""
    
    def test_agents_registered(self):
        """Test that auditor agents are registered."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.AUDITOR)
        names = [a.name for a in agents]
        
        assert "AuditorLLM" in names
        assert "AuditorRuleBased" in names
    
    def test_llm_agent_higher_priority(self):
        """Test that LLM agent has higher priority."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.AUDITOR)
        
        llm_agent = next(a for a in agents if a.name == "AuditorLLM")
        rule_agent = next(a for a in agents if a.name == "AuditorRuleBased")
        
        assert llm_agent.priority > rule_agent.priority


class TestReasoningTrace:
    """Tests for ReasoningTrace schema."""
    
    def test_get_evidence_refs(self):
        """Test getting all evidence references."""
        from core.schemas import ReasoningStep, EvidenceRef
        
        trace = ReasoningTrace(
            trace_id="trace_test",
            market_id="mk_test",
            bundle_id="bundle_test",
            steps=[
                ReasoningStep(
                    step_id="step_001",
                    step_type="evidence_analysis",
                    description="Test step",
                    evidence_refs=[
                        EvidenceRef(evidence_id="ev_001"),
                        EvidenceRef(evidence_id="ev_002"),
                    ],
                ),
                ReasoningStep(
                    step_id="step_002",
                    step_type="conclusion",
                    description="Another step",
                    evidence_refs=[EvidenceRef(evidence_id="ev_001")],
                ),
            ],
        )
        
        refs = trace.get_evidence_refs()
        
        assert len(refs) == 2  # Unique refs
        assert "ev_001" in refs
        assert "ev_002" in refs
    
    def test_get_steps_by_type(self):
        """Test filtering steps by type."""
        from core.schemas import ReasoningStep
        
        trace = ReasoningTrace(
            trace_id="trace_test",
            market_id="mk_test",
            bundle_id="bundle_test",
            steps=[
                ReasoningStep(step_id="s1", step_type="evidence_analysis", description=""),
                ReasoningStep(step_id="s2", step_type="conclusion", description=""),
                ReasoningStep(step_id="s3", step_type="evidence_analysis", description=""),
            ],
        )
        
        analysis_steps = trace.get_steps_by_type("evidence_analysis")
        
        assert len(analysis_steps) == 2


class TestMultipleBundles:
    """Tests for auditor processing multiple evidence bundles."""

    def test_auditor_with_multiple_bundles(self):
        """Test auditor processes multiple evidence bundles from different collectors."""
        ctx = AgentContext.create_minimal()

        prompt_spec = create_test_prompt_spec()

        # Create two bundles with different collector names
        bundle1 = EvidenceBundle(
            bundle_id="bundle_1",
            market_id="mk_test123",
            plan_id="plan_test123",
            collector_name="CollectorLLM",
        )
        bundle1.add_item(EvidenceItem(
            evidence_id="ev_001",
            requirement_id="req_001",
            provenance=Provenance(
                source_id="coingecko",
                source_uri="https://api.coingecko.com/price",
                tier=3,
            ),
            success=True,
            extracted_fields={"price_usd": 105000},
        ))

        bundle2 = EvidenceBundle(
            bundle_id="bundle_2",
            market_id="mk_test123",
            plan_id="plan_test123",
            collector_name="CollectorHyDE",
        )
        bundle2.add_item(EvidenceItem(
            evidence_id="ev_002",
            requirement_id="req_001",
            provenance=Provenance(
                source_id="binance",
                source_uri="https://api.binance.com/price",
                tier=2,
            ),
            success=True,
            extracted_fields={"price_usd": 104500},
        ))

        auditor = AuditorRuleBased()
        result = auditor.run(ctx, prompt_spec, [bundle1, bundle2])

        assert result.success
        trace = result.output

        # With prices above 100k, should be YES
        assert trace.preliminary_outcome == "YES"

        # Verify step count shows processing happened
        assert trace.step_count >= 3

        # Verify metadata is present
        assert result.metadata["auditor"] == "rule_based"
        assert "trace_id" in result.metadata

    def test_reasoner_combines_evidence_from_multiple_bundles(self):
        """Test that reasoner properly combines evidence from all bundles."""
        ctx = AgentContext.create_minimal()
        reasoner = RuleBasedReasoner()

        prompt_spec = create_test_prompt_spec()

        # Bundle 1: Evidence from one source
        bundle1 = EvidenceBundle(
            bundle_id="bundle_1",
            market_id="mk_test123",
            plan_id="plan_test123",
            collector_name="CollectorHTTP",
        )
        bundle1.add_item(EvidenceItem(
            evidence_id="ev_001",
            requirement_id="req_001",
            provenance=Provenance(source_id="source1", source_uri="https://source1.com", tier=3),
            success=True,
            extracted_fields={"price_usd": 110000},
        ))

        # Bundle 2: Evidence from another source
        bundle2 = EvidenceBundle(
            bundle_id="bundle_2",
            market_id="mk_test123",
            plan_id="plan_test123",
            collector_name="CollectorLLM",
        )
        bundle2.add_item(EvidenceItem(
            evidence_id="ev_002",
            requirement_id="req_001",
            provenance=Provenance(source_id="source2", source_uri="https://source2.com", tier=2),
            success=True,
            extracted_fields={"price_usd": 108000},
        ))

        trace = reasoner.reason(ctx, prompt_spec, [bundle1, bundle2])

        # Should use all evidence from both bundles
        assert trace.preliminary_outcome == "YES"  # Both prices above 100k

        # Both pieces of evidence should contribute to the reasoning
        # The trace should reflect analysis of multiple sources
        evidence_refs = trace.get_evidence_refs()
        # At least one evidence should be referenced
        assert len(evidence_refs) >= 1

    def test_multiple_empty_bundles(self):
        """Test handling of multiple empty bundles."""
        ctx = AgentContext.create_minimal()

        prompt_spec = create_test_prompt_spec()

        bundle1 = EvidenceBundle(
            bundle_id="bundle_1",
            market_id="mk_test123",
            plan_id="plan_test123",
            collector_name="CollectorHTTP",
        )
        bundle2 = EvidenceBundle(
            bundle_id="bundle_2",
            market_id="mk_test123",
            plan_id="plan_test123",
            collector_name="CollectorLLM",
        )

        auditor = AuditorRuleBased()
        result = auditor.run(ctx, prompt_spec, [bundle1, bundle2])

        # Should succeed but produce INVALID outcome due to no evidence
        assert result.success
        trace = result.output
        assert trace.preliminary_outcome == "INVALID"


class TestIntegration:
    """Integration tests for full pipeline through auditor."""

    def test_full_pipeline_with_rule_based(self):
        """Test full pipeline from question to reasoning trace."""
        import json
        from agents.prompt_engineer import compile_prompt
        from agents.collector import CollectorMock

        # compile_prompt requires LLM; use mock
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
        trace = audit_result.output

        # With price 95k < 100k threshold, should be NO
        assert trace.preliminary_outcome == "NO"
        assert trace.step_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
