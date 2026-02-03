"""
Tests for Collector Agent

Tests both HTTP-based and mock collectors.
"""

import pytest
import json
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from agents.collector import (
    CollectorHTTP,
    CollectorMock,
    CollectionEngine,
    HttpAdapter,
    MockAdapter,
    collect_evidence,
    get_collector,
    get_adapter,
)
from agents.prompt_engineer import compile_prompt
from core.schemas import (
    DataRequirement,
    EvidenceBundle,
    EvidenceItem,
    MarketSpec,
    PromptSpec,
    Provenance,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourceTarget,
    ToolPlan,
    DisputePolicy,
    PredictionSemantics,
)


def create_test_prompt_spec() -> PromptSpec:
    """Create a test PromptSpec for collector tests."""
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_test123",
            question="Will BTC be above $100k?",
            event_definition="price(BTC_USD) > 100000",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Test rule", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="bitcoin",
            predicate="price above threshold",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price from CoinGecko",
                source_targets=[
                    SourceTarget(
                        source_id="coingecko",
                        uri="https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
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
            DataRequirement(
                requirement_id="req_002",
                description="Get BTC price from Coinbase",
                source_targets=[
                    SourceTarget(
                        source_id="coinbase",
                        uri="https://api.coinbase.com/v2/prices/BTC-USD/spot",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="fallback_chain",
                    min_sources=1,
                    max_sources=2,
                    quorum=1,
                ),
            ),
        ],
    )


def create_test_tool_plan() -> ToolPlan:
    """Create a test ToolPlan."""
    return ToolPlan(
        plan_id="plan_test123",
        requirements=["req_001", "req_002"],
        sources=["coingecko", "coinbase"],
        min_provenance_tier=0,
        allow_fallbacks=True,
    )


class TestMockAdapter:
    """Tests for MockAdapter."""
    
    def test_returns_mock_data(self):
        """Test that mock adapter returns preset data."""
        ctx = AgentContext.create_minimal()
        
        mock_response = {"bitcoin": {"usd": 95000}}
        adapter = MockAdapter(responses={"coingecko": mock_response})
        
        target = SourceTarget(
            source_id="coingecko",
            uri="https://api.coingecko.com/api/v3/simple/price",
        )
        
        evidence = adapter.fetch(ctx, target, "req_001")
        
        assert evidence.success
        assert evidence.parsed_value == mock_response
        assert evidence.provenance.source_id == "coingecko"
    
    def test_default_mock_response(self):
        """Test that mock adapter returns default data if no match."""
        ctx = AgentContext.create_minimal()
        
        adapter = MockAdapter()  # No preset responses
        
        target = SourceTarget(
            source_id="unknown",
            uri="https://example.com/api",
        )
        
        evidence = adapter.fetch(ctx, target, "req_001")
        
        assert evidence.success
        assert evidence.parsed_value["mock"] is True


class TestGetAdapter:
    """Tests for get_adapter function."""
    
    def test_http_adapter(self):
        """Test that http source gets HttpAdapter."""
        adapter = get_adapter("http")
        assert isinstance(adapter, HttpAdapter)
    
    def test_coingecko_adapter(self):
        """Test that coingecko gets CryptoExchangeAdapter."""
        from agents.collector import CryptoExchangeAdapter
        adapter = get_adapter("coingecko")
        assert isinstance(adapter, CryptoExchangeAdapter)
    
    def test_default_to_http(self):
        """Test that unknown source defaults to HttpAdapter."""
        adapter = get_adapter("unknown_source")
        assert isinstance(adapter, HttpAdapter)


class TestCollectorMock:
    """Tests for CollectorMock agent."""
    
    def test_run_success(self):
        """Test successful mock collection."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        collector = CollectorMock(mock_responses={
            "coingecko": {"bitcoin": {"usd": 95000}},
            "coinbase": {"data": {"amount": "95100"}},
        })
        
        result = collector.run(ctx, prompt_spec, tool_plan)
        
        assert result.success
        assert result.output is not None
        
        bundle, execution_log = result.output
        assert isinstance(bundle, EvidenceBundle)
        assert bundle.has_evidence
        assert len(bundle.items) >= 2
    
    def test_metadata_included(self):
        """Test that metadata is included."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        collector = CollectorMock()
        result = collector.run(ctx, prompt_spec, tool_plan)
        
        assert result.metadata["collector"] == "mock"
        assert "bundle_id" in result.metadata


class TestCollectorHTTP:
    """Tests for CollectorHTTP agent."""
    
    def test_requires_http_client(self):
        """Test that HTTP collector requires HTTP client."""
        ctx = AgentContext.create_minimal()  # No HTTP client
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        collector = CollectorHTTP()
        result = collector.run(ctx, prompt_spec, tool_plan)
        
        assert not result.success
        assert "HTTP client not available" in result.error


class TestCollectionEngine:
    """Tests for CollectionEngine."""
    
    def test_execute_with_mock_context(self):
        """Test engine execution with mock context."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        engine = CollectionEngine()
        
        # Engine will fail without HTTP client but should not raise
        bundle, log = engine.execute(ctx, prompt_spec, tool_plan)
        
        # Should have attempted sources
        assert bundle.total_sources_attempted >= 0


class TestGetCollector:
    """Tests for get_collector function."""
    
    def test_returns_mock_when_no_http(self):
        """Test that mock collector is returned when no HTTP client."""
        ctx = AgentContext.create_minimal()
        
        collector = get_collector(ctx)
        
        assert isinstance(collector, CollectorMock)
    
    def test_returns_mock_when_preferred(self):
        """Test that mock is returned when explicitly preferred."""
        ctx = AgentContext.create_minimal()
        
        collector = get_collector(ctx, prefer_http=False)
        
        assert isinstance(collector, CollectorMock)


class TestCollectEvidence:
    """Tests for collect_evidence convenience function."""
    
    def test_collect_with_mock(self):
        """Test collect_evidence uses mock when no HTTP."""
        ctx = AgentContext.create_minimal()
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        result = collect_evidence(ctx, prompt_spec, tool_plan)
        
        assert result.success
        assert result.metadata["collector"] == "mock"


class TestAgentRegistration:
    """Tests for agent registration."""
    
    def test_agents_registered(self):
        """Test that collector agents are registered."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.COLLECTOR)
        names = [a.name for a in agents]
        
        assert "CollectorHTTP" in names
        assert "CollectorMock" in names
    
    def test_http_agent_higher_priority(self):
        """Test that HTTP agent has higher priority."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.COLLECTOR)
        
        http_agent = next(a for a in agents if a.name == "CollectorHTTP")
        mock_agent = next(a for a in agents if a.name == "CollectorMock")
        
        assert http_agent.priority > mock_agent.priority


class TestEvidenceBundle:
    """Tests for EvidenceBundle."""
    
    def test_add_item(self):
        """Test adding items to bundle."""
        bundle = EvidenceBundle(
            bundle_id="test_bundle",
            market_id="mk_test",
            plan_id="plan_test",
        )
        
        item = EvidenceItem(
            evidence_id="ev_001",
            requirement_id="req_001",
            provenance=Provenance(
                source_id="test",
                source_uri="https://test.com",
                tier=2,
            ),
            success=True,
        )
        
        bundle.add_item(item)
        
        assert len(bundle.items) == 1
        assert bundle.total_sources_succeeded == 1
        assert "req_001" in bundle.requirements_fulfilled
    
    def test_get_valid_evidence(self):
        """Test filtering valid evidence."""
        bundle = EvidenceBundle(
            bundle_id="test_bundle",
            market_id="mk_test",
            plan_id="plan_test",
        )
        
        # Add valid item
        bundle.add_item(EvidenceItem(
            evidence_id="ev_001",
            requirement_id="req_001",
            provenance=Provenance(source_id="test", source_uri="https://test.com", tier=2),
            success=True,
        ))
        
        # Add invalid item
        bundle.add_item(EvidenceItem(
            evidence_id="ev_002",
            requirement_id="req_002",
            provenance=Provenance(source_id="test", source_uri="https://test.com", tier=0),
            success=False,
            error="Failed",
        ))
        
        valid = bundle.get_valid_evidence()
        
        assert len(valid) == 1
        assert valid[0].evidence_id == "ev_001"


class TestIntegration:
    """Integration tests for prompt engineer + collector."""
    
    def test_full_pipeline_with_mock(self):
        """Test full pipeline from question to evidence."""
        import json
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
        
        # Step 2: Collect evidence
        collector = CollectorMock(mock_responses={
            "coingecko": {"bitcoin": {"usd": 95000}},
        })
        
        collect_result = collector.run(ctx, prompt_spec, tool_plan)
        assert collect_result.success
        
        bundle, log = collect_result.output
        assert bundle.has_evidence
        assert bundle.market_id == prompt_spec.market_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
