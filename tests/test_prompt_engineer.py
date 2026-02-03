"""
Tests for Prompt Engineer Agent

Tests the LLM-based compiler.
"""

import pytest
import json
from datetime import datetime, timezone

from agents import AgentContext, AgentStep, get_registry
from agents.prompt_engineer import (
    PromptEngineerLLM,
    LLMPromptCompiler,
    compile_prompt,
    get_prompt_engineer,
)
from core.schemas import PromptSpec, ToolPlan


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

        agent = get_prompt_engineer(ctx)

        assert isinstance(agent, PromptEngineerLLM)

    def test_raises_when_no_llm(self):
        """Test that ValueError is raised when no LLM is configured."""
        ctx = AgentContext.create_minimal()

        with pytest.raises(ValueError, match="LLM client is required"):
            get_prompt_engineer(ctx)


class TestCompilePrompt:
    """Tests for compile_prompt convenience function."""

    def test_compile_requires_llm(self):
        """Test compile_prompt raises when no LLM."""
        ctx = AgentContext.create_minimal()

        with pytest.raises(ValueError, match="LLM client is required"):
            compile_prompt(ctx, "Will BTC be above $100k?")


class TestAgentRegistration:
    """Tests for agent registration."""

    def test_agents_registered(self):
        """Test that prompt engineer agent is registered."""
        registry = get_registry()

        agents = registry.list_agents(AgentStep.PROMPT_ENGINEER)
        names = [a.name for a in agents]

        assert "PromptEngineerLLM" in names


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
