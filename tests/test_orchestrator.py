"""
Tests for Orchestrator Pipeline

Tests registry-based agent selection, fail-closed behavior, and step overrides.
"""

import json
import pytest
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from orchestrator import (
    CapabilityError,
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    PipelineState,
    StepOverrides,
    create_pipeline,
    create_production_pipeline,
    create_test_pipeline,
    resolve_market,
)
from core.schemas import DeterministicVerdict


def _mock_llm_response() -> str:
    """Standard mock LLM response for BTC prompt compilation."""
    return json.dumps({
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


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.mode == ExecutionMode.DEVELOPMENT
        assert config.strict_mode is True
        assert config.deterministic_timestamps is True
        assert config.enable_sentinel_verify is True
        assert not config.is_production
        assert config.allow_fallbacks
    
    def test_production_mode(self):
        """Test production mode configuration."""
        config = PipelineConfig(mode=ExecutionMode.PRODUCTION)
        
        assert config.is_production
        assert not config.allow_fallbacks
    
    def test_step_overrides(self):
        """Test step overrides configuration."""
        overrides = StepOverrides(
            prompt_engineer="CustomPromptEngineer",
            collector="CustomCollector",
        )
        config = PipelineConfig(step_overrides=overrides)
        
        assert config.step_overrides.prompt_engineer == "CustomPromptEngineer"
        assert config.step_overrides.collector == "CustomCollector"
        assert config.step_overrides.auditor is None


class TestPipelineState:
    """Tests for PipelineState."""
    
    def test_add_error(self):
        """Test adding errors marks state as not ok."""
        state = PipelineState()
        
        assert state.ok
        state.add_error("Test error")
        
        assert not state.ok
        assert "Test error" in state.errors
    
    def test_add_check(self):
        """Test adding checks."""
        from core.schemas import CheckResult

        state = PipelineState()

        # Add passing check
        state.add_check(CheckResult.passed("test1", "Passed"))
        assert state.ok
        assert len(state.errors) == 0

        # Add failing check - should NOT set ok=False (checks are informational)
        # but should add message to errors for visibility
        state.add_check(CheckResult.failed("test2", "Failed"))
        assert state.ok  # Check failures don't block execution
        assert len(state.errors) == 1
        assert "test2" in state.errors[0]


class TestPipelineAgentSelection:
    """Tests for registry-based agent selection."""
    
    def test_selects_available_agent(self):
        """Test that pipeline selects available agents."""
        ctx = AgentContext.create_mock(llm_responses=[_mock_llm_response()])
        pipeline = Pipeline(context=ctx)

        agent = pipeline._select_agent(
            AgentStep.PROMPT_ENGINEER,
            None,
            ctx,
        )

        assert agent is not None
        assert agent.name == "PromptEngineerLLM"
    
    def test_respects_override(self):
        """Test that overrides are respected."""
        ctx = AgentContext.create_minimal()
        pipeline = Pipeline(context=ctx)
        
        # Override to use fallback
        agent = pipeline._select_agent(
            AgentStep.PROMPT_ENGINEER,
            "PromptEngineerLLM",
            ctx,
        )
        
        assert agent is not None
        assert agent.name == "PromptEngineerLLM"
    
    def test_production_fails_on_missing_override(self):
        """Test production mode fails when specified override not found."""
        ctx = AgentContext.create_minimal()
        config = PipelineConfig(mode=ExecutionMode.PRODUCTION)
        pipeline = Pipeline(config=config, context=ctx)
        
        with pytest.raises(CapabilityError):
            pipeline._select_agent(
                AgentStep.PROMPT_ENGINEER,
                "NonExistentAgent",
                ctx,
            )
    
    def test_development_uses_fallback_on_missing_override(self):
        """Test development mode falls back when override not found."""
        ctx = AgentContext.create_mock(llm_responses=[_mock_llm_response()])
        config = PipelineConfig(mode=ExecutionMode.DEVELOPMENT)
        pipeline = Pipeline(config=config, context=ctx)

        # Should fall back to available agent
        agent = pipeline._select_agent(
            AgentStep.PROMPT_ENGINEER,
            "NonExistentAgent",
            ctx,
        )

        assert agent is not None


class TestPipelineFailClosed:
    """Tests for fail-closed behavior in production mode."""
    
    def test_production_fails_without_llm_when_required(self):
        """Test production mode fails when LLM required but not available."""
        ctx = AgentContext.create_minimal()  # No LLM
        config = PipelineConfig(
            mode=ExecutionMode.PRODUCTION,
            require_llm=True,
        )
        pipeline = Pipeline(config=config, context=ctx)
        
        result = pipeline.run("Test question")
        
        assert not result.ok
        assert any("LLM capability required" in e for e in result.errors)
    
    def test_production_fails_without_network_when_required(self):
        """Test production mode fails when network required but not available."""
        ctx = AgentContext.create_minimal()  # No HTTP
        config = PipelineConfig(
            mode=ExecutionMode.PRODUCTION,
            require_network=True,
        )
        pipeline = Pipeline(config=config, context=ctx)
        
        result = pipeline.run("Test question")
        
        assert not result.ok
        assert any("NETWORK capability required" in e for e in result.errors)
    
    def test_development_continues_without_capabilities(self):
        """Test development mode continues with fallbacks."""
        ctx = AgentContext.create_minimal()
        config = PipelineConfig(
            mode=ExecutionMode.DEVELOPMENT,
            require_llm=False,
            require_network=False,
        )
        pipeline = Pipeline(config=config, context=ctx)
        
        # Should not fail on capability check
        result = pipeline.run("Will BTC be above $100k?")
        
        # May fail later due to other reasons but not capability check
        cap_errors = [e for e in result.errors if "capability required" in e]
        assert len(cap_errors) == 0


class TestPipelineExecution:
    """Tests for full pipeline execution."""
    
    def test_run_with_mock_context(self):
        """Test running pipeline with mock LLM context."""
        ctx = AgentContext.create_mock(llm_responses=[_mock_llm_response()])

        # Use step overrides to ensure deterministic agents
        overrides = StepOverrides(
            prompt_engineer="PromptEngineerLLM",
            collector="CollectorMock",
            auditor="AuditorRuleBased",
            judge="JudgeRuleBased",
            sentinel="SentinelStrict",
        )
        config = PipelineConfig(
            mode=ExecutionMode.TEST,
            step_overrides=overrides,
            enable_sentinel_verify=True,
        )
        pipeline = Pipeline(config=config, context=ctx)

        result = pipeline.run("Will BTC be above $100k?")

        # Should complete (may have verification issues but should run)
        assert result.prompt_spec is not None
        assert result.tool_plan is not None
    
    def test_result_contains_all_artifacts(self):
        """Test that result contains all pipeline artifacts."""
        pipeline = create_test_pipeline()
        result = pipeline.run("Will ETH be above $5000?")
        
        # Check all artifacts are populated
        assert result.prompt_spec is not None
        assert result.tool_plan is not None
        assert result.evidence_bundle is not None
        assert result.audit_trace is not None
        assert result.verdict is not None


class TestFactoryFunctions:
    """Tests for pipeline factory functions."""
    
    def test_create_pipeline(self):
        """Test create_pipeline function."""
        pipeline = create_pipeline(
            mode=ExecutionMode.DEVELOPMENT,
            strict_mode=True,
        )
        
        assert pipeline.config.mode == ExecutionMode.DEVELOPMENT
        assert pipeline.config.strict_mode is True
    
    def test_create_production_pipeline(self):
        """Test create_production_pipeline function."""
        ctx = AgentContext.create_minimal()
        pipeline = create_production_pipeline(
            ctx,
            require_llm=False,
            require_network=False,
        )
        
        assert pipeline.config.mode == ExecutionMode.PRODUCTION
    
    def test_create_test_pipeline(self):
        """Test create_test_pipeline function."""
        pipeline = create_test_pipeline()
        
        assert pipeline.config.mode == ExecutionMode.TEST
        assert pipeline.config.step_overrides.prompt_engineer == "PromptEngineerLLM"
        assert pipeline.config.step_overrides.collector == "CollectorMock"


class TestMarketResolver:
    """Tests for market resolver functions."""
    
    def test_resolve_market(self):
        """Test resolve_market function."""
        verdict = DeterministicVerdict(
            market_id="mk_test123",
            outcome="YES",
            confidence=0.85,
            resolution_rule_id="R_THRESHOLD",
        )
        
        resolution = resolve_market(verdict)
        
        assert resolution["market_id"] == "mk_test123"
        assert resolution["outcome"] == "YES"
        assert resolution["confidence"] == 0.85
        assert resolution["resolution_rule_id"] == "R_THRESHOLD"


class TestIntegration:
    """Integration tests for orchestrator."""
    
    def test_full_pipeline_deterministic(self):
        """Test that pipeline is deterministic with same inputs."""
        pipeline1 = create_test_pipeline()
        pipeline2 = create_test_pipeline()
        
        result1 = pipeline1.run("Will BTC be above $100k?")
        result2 = pipeline2.run("Will BTC be above $100k?")
        
        # Same input should produce same market_id
        assert result1.market_id == result2.market_id
        
        # Same outcome
        if result1.verdict and result2.verdict:
            assert result1.verdict.outcome == result2.verdict.outcome
    
    def test_pipeline_with_step_overrides(self):
        """Test pipeline with custom step overrides."""
        ctx = AgentContext.create_mock(llm_responses=[_mock_llm_response()])

        overrides = StepOverrides(
            prompt_engineer="PromptEngineerLLM",
            auditor="AuditorRuleBased",
        )

        pipeline = create_pipeline(
            mode=ExecutionMode.DEVELOPMENT,
            context=ctx,
            step_overrides=overrides,
        )

        result = pipeline.run("Will BTC be above $100k?")

        # Should run without errors related to agent selection
        agent_errors = [e for e in result.errors if "agent" in e.lower()]
        # Collector might fail but prompt engineer should work
        assert result.prompt_spec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
