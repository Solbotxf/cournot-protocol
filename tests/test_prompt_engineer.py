"""
Tests for Prompt Engineer Agent

Tests both LLM-based and fallback compilers.
"""

import pytest
import json
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from agents.prompt_engineer import (
    PromptEngineerLLM,
    PromptEngineerFallback,
    LLMPromptCompiler,
    FallbackPromptCompiler,
    compile_prompt,
    get_prompt_engineer,
)
from core.schemas import PromptSpec, ToolPlan


class TestFallbackCompiler:
    """Tests for the fallback pattern-based compiler."""
    
    def test_crypto_price_question(self):
        """Test parsing crypto price question."""
        ctx = AgentContext.create_minimal()
        compiler = FallbackPromptCompiler()
        
        prompt_spec, tool_plan = compiler.compile(
            ctx,
            "Will BTC be above $100,000 by end of 2025?"
        )
        
        # Check market spec
        assert prompt_spec.market_id.startswith("mk_")
        assert "BTC" in prompt_spec.market.question or "btc" in prompt_spec.market.question.lower()
        
        # Check semantics
        assert prompt_spec.prediction_semantics.target_entity == "bitcoin"
        assert prompt_spec.prediction_semantics.threshold == "100000.0"
        
        # Check data requirements
        assert len(prompt_spec.data_requirements) >= 1
        assert any("coingecko" in req.source_targets[0].uri for req in prompt_spec.data_requirements)
        
        # Check tool plan
        assert tool_plan.plan_id.startswith("plan_")
        assert len(tool_plan.requirements) == len(prompt_spec.data_requirements)
    
    def test_ethereum_price_question(self):
        """Test parsing Ethereum price question."""
        ctx = AgentContext.create_minimal()
        compiler = FallbackPromptCompiler()
        
        prompt_spec, tool_plan = compiler.compile(
            ctx,
            "Will ETH exceed $5000?"
        )
        
        assert prompt_spec.prediction_semantics.target_entity == "ethereum"
        assert "5000" in prompt_spec.prediction_semantics.threshold
    
    def test_generic_event_question(self):
        """Test parsing generic event question."""
        ctx = AgentContext.create_minimal()
        compiler = FallbackPromptCompiler()
        
        prompt_spec, tool_plan = compiler.compile(
            ctx,
            "Will the Lakers win the NBA championship?"
        )
        
        assert prompt_spec.market_id.startswith("mk_")
        assert len(prompt_spec.data_requirements) >= 1
        assert prompt_spec.extra.get("question_type") == "event"
    
    def test_deterministic_market_id(self):
        """Test that same question produces same market ID."""
        ctx = AgentContext.create_minimal()
        compiler = FallbackPromptCompiler()
        
        question = "Will BTC reach $150k?"
        
        spec1, _ = compiler.compile(ctx, question)
        spec2, _ = compiler.compile(ctx, question)
        
        assert spec1.market_id == spec2.market_id
    
    def test_resolution_rules_generated(self):
        """Test that resolution rules are generated."""
        ctx = AgentContext.create_minimal()
        compiler = FallbackPromptCompiler()
        
        prompt_spec, _ = compiler.compile(ctx, "Will SOL be above $200?")
        
        rules = prompt_spec.market.resolution_rules.rules
        assert len(rules) >= 3
        
        # Check for expected rules
        rule_ids = {r.rule_id for r in rules}
        assert "R_VALIDITY" in rule_ids
        assert "R_INVALID_FALLBACK" in rule_ids
    
    def test_strict_mode_excludes_timestamp(self):
        """Test that strict mode excludes created_at."""
        ctx = AgentContext.create_minimal()
        compiler = FallbackPromptCompiler(strict_mode=True)
        
        prompt_spec, _ = compiler.compile(ctx, "Test question?")
        
        assert prompt_spec.created_at is None
        assert prompt_spec.extra.get("strict_mode") is True


class TestPromptEngineerFallbackAgent:
    """Tests for the fallback agent."""
    
    def test_run_success(self):
        """Test successful run."""
        ctx = AgentContext.create_minimal()
        agent = PromptEngineerFallback()
        
        result = agent.run(ctx, "Will BTC be above $100k?")
        
        assert result.success
        assert result.output is not None
        
        prompt_spec, tool_plan = result.output
        assert isinstance(prompt_spec, PromptSpec)
        assert isinstance(tool_plan, ToolPlan)
        
        # Check verification
        assert result.verification is not None
        assert result.verification.ok
    
    def test_metadata_included(self):
        """Test that metadata is included."""
        ctx = AgentContext.create_minimal()
        agent = PromptEngineerFallback()
        
        result = agent.run(ctx, "Will ETH exceed $5000?")
        
        assert result.metadata.get("compiler") == "fallback"
        assert "market_id" in result.metadata
        assert "question_type" in result.metadata


class TestPromptEngineerLLMAgent:
    """Tests for the LLM-based agent (with mocked LLM)."""
    
    def test_requires_llm(self):
        """Test that LLM agent requires LLM client."""
        ctx = AgentContext.create_minimal()  # No LLM
        agent = PromptEngineerLLM()
        
        result = agent.run(ctx, "Will BTC be above $100k?")
        
        assert not result.success
        assert "LLM client required" in result.error
    
    def test_run_with_mock_llm(self):
        """Test run with mocked LLM responses."""
        # Create mock response
        mock_response = json.dumps({
            "market_id": "mk_test123",
            "question": "Will BTC be above $100k?",
            "event_definition": "price(BTC_USD) > 100000",
            "target_entity": "bitcoin",
            "predicate": "price above threshold",
            "threshold": "100000",
            "timeframe": "end of 2025",
            "resolution_window": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-12-31T23:59:59Z",
            },
            "resolution_deadline": "2025-12-31T23:59:59Z",
            "data_requirements": [
                {
                    "requirement_id": "req_001",
                    "description": "Get BTC price",
                    "source_targets": [
                        {
                            "source_id": "coingecko",
                            "uri": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                            "method": "GET",
                            "expected_content_type": "json",
                        }
                    ],
                    "selection_policy": {
                        "strategy": "single_best",
                        "min_sources": 1,
                        "max_sources": 1,
                        "quorum": 1,
                    },
                }
            ],
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
        agent = PromptEngineerLLM()
        
        result = agent.run(ctx, "Will BTC be above $100k?")
        
        assert result.success
        assert result.output is not None
        
        prompt_spec, tool_plan = result.output
        assert prompt_spec.market_id == "mk_test123"
        assert len(prompt_spec.data_requirements) == 1
        assert tool_plan.requirements == ["req_001"]


class TestGetPromptEngineer:
    """Tests for get_prompt_engineer function."""
    
    def test_returns_llm_when_available(self):
        """Test that LLM agent is returned when LLM is available."""
        ctx = AgentContext.create_mock(llm_responses=["{}"])
        
        agent = get_prompt_engineer(ctx, prefer_llm=True)
        
        assert isinstance(agent, PromptEngineerLLM)
    
    def test_returns_fallback_when_no_llm(self):
        """Test that fallback is returned when no LLM."""
        ctx = AgentContext.create_minimal()
        
        agent = get_prompt_engineer(ctx, prefer_llm=True)
        
        assert isinstance(agent, PromptEngineerFallback)
    
    def test_returns_fallback_when_preferred(self):
        """Test that fallback is returned when explicitly preferred."""
        ctx = AgentContext.create_mock(llm_responses=["{}"])
        
        agent = get_prompt_engineer(ctx, prefer_llm=False)
        
        assert isinstance(agent, PromptEngineerFallback)


class TestCompilePrompt:
    """Tests for compile_prompt convenience function."""
    
    def test_compile_with_fallback(self):
        """Test compile_prompt uses fallback when no LLM."""
        ctx = AgentContext.create_minimal()
        
        result = compile_prompt(ctx, "Will BTC be above $100k?")
        
        assert result.success
        assert result.metadata.get("compiler") == "fallback"


class TestAgentRegistration:
    """Tests for agent registration."""
    
    def test_agents_registered(self):
        """Test that prompt engineer agents are registered."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.PROMPT_ENGINEER)
        names = [a.name for a in agents]
        
        assert "PromptEngineerLLM" in names
        assert "PromptEngineerFallback" in names
    
    def test_llm_agent_higher_priority(self):
        """Test that LLM agent has higher priority."""
        registry = get_registry()
        
        agents = registry.list_agents(AgentStep.PROMPT_ENGINEER)
        
        llm_agent = next(a for a in agents if a.name == "PromptEngineerLLM")
        fallback_agent = next(a for a in agents if a.name == "PromptEngineerFallback")
        
        assert llm_agent.priority > fallback_agent.priority


class TestLLMCompilerJSONExtraction:
    """Tests for JSON extraction in LLM compiler."""
    
    def test_extract_from_code_block(self):
        """Test extraction from markdown code block."""
        compiler = LLMPromptCompiler()
        
        text = '''Here is the JSON:
```json
{"key": "value"}
```
'''
        result = compiler._extract_json(text)
        assert result == {"key": "value"}
    
    def test_extract_bare_json(self):
        """Test extraction of bare JSON."""
        compiler = LLMPromptCompiler()
        
        text = 'Some text {"key": "value"} more text'
        result = compiler._extract_json(text)
        assert result == {"key": "value"}
    
    def test_extract_pure_json(self):
        """Test extraction of pure JSON."""
        compiler = LLMPromptCompiler()
        
        text = '{"key": "value"}'
        result = compiler._extract_json(text)
        assert result == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
