"""
Unit tests for Module 04 - Prompt Engineer.

Tests verify:
1. Deterministic IDs stable across runs
2. LLM compiler JSON extraction
3. Agent requires LLM
"""

import pytest
import json
from datetime import datetime, timezone, timedelta

from agents import AgentContext
from agents.prompt_engineer import (
    compile_prompt,
    PromptEngineerLLM,
    LLMPromptCompiler,
)
from agents.prompt_engineer.models import (
    NormalizedUserRequest,
    generate_deterministic_id,
    generate_requirement_id,
)
from core.schemas.errors import PromptCompilationException


@pytest.mark.skip(reason="InputParser not in current API")
class TestInputParser:
    """Tests for InputParser utilities."""

    def test_extract_threshold_with_dollar_sign(self):
        """Extract threshold with dollar sign."""
        text = "Will BTC close above $100,000?"
        threshold = InputParser.extract_threshold(text)
        assert threshold == "100000"

    def test_extract_threshold_without_dollar(self):
        """Extract threshold without dollar sign."""
        text = "Will the price be above 50000?"
        threshold = InputParser.extract_threshold(text)
        assert threshold == "50000"

    def test_extract_threshold_with_decimals(self):
        """Extract threshold with decimal values."""
        text = "Will ETH be above $3,500.50?"
        threshold = InputParser.extract_threshold(text)
        assert threshold == "3500.50"

    def test_extract_date_iso_format(self):
        """Extract ISO format date."""
        text = "by 2026-12-31?"
        date = InputParser.extract_date(text)
        assert date == "2026-12-31"

    def test_extract_date_text_format(self):
        """Extract text format date."""
        text = "before December 31, 2026?"
        date = InputParser.extract_date(text)
        assert "December 31" in date or "Dec" in date

    def test_extract_timezone_utc(self):
        """Extract UTC timezone."""
        text = "by 2026-12-31 23:59 UTC"
        tz = InputParser.extract_timezone(text)
        assert tz == "UTC"

    def test_extract_timezone_default(self):
        """Default to UTC when no timezone specified."""
        text = "by 2026-12-31 without timezone"
        tz = InputParser.extract_timezone(text)
        assert tz == "UTC"

    def test_extract_sources_coinbase(self):
        """Extract Coinbase source preference."""
        text = "Use Coinbase for the price data"
        sources = InputParser.extract_sources(text)
        assert "exchange" in sources

    def test_extract_sources_multiple(self):
        """Extract multiple source preferences."""
        text = "Use Coinbase for prices. Use Polymarket for market data."
        sources = InputParser.extract_sources(text)
        assert "exchange" in sources
        assert "polymarket" in sources

    def test_extract_urls(self):
        """Extract explicit URLs."""
        text = "Check https://api.example.com/data and https://another.com/endpoint"
        urls = InputParser.extract_urls(text)
        assert len(urls) == 2
        assert "https://api.example.com/data" in urls

    def test_detect_entity_crypto(self):
        """Detect crypto entity."""
        text = "Will BTC reach $100,000?"
        entity = InputParser.detect_entity(text)
        assert entity == "BTC"

    def test_detect_entity_eth(self):
        """Detect ETH entity."""
        text = "Will ETH close above $5000?"
        entity = InputParser.detect_entity(text)
        assert entity == "ETH"

    def test_detect_predicate_close_above(self):
        """Detect close above predicate."""
        text = "Will BTC close above $100,000?"
        predicate = InputParser.detect_predicate(text)
        assert predicate == "price_close_above"


@pytest.mark.skip(reason="SourceTargetBuilder not in current API")
class TestSourceTargetBuilder:
    """Tests for SourceTargetBuilder."""

    def test_build_coinbase_target(self):
        """Build Coinbase API target."""
        target = SourceTargetBuilder.build_coinbase_target("BTC", "2026-12-31")
        assert target.source_id == "exchange"
        assert "coinbase" in target.uri.lower()
        assert target.method == "GET"
        assert target.expected_content_type == "json"

    def test_build_binance_target(self):
        """Build Binance API target."""
        target = SourceTargetBuilder.build_binance_target("ETH", "2026-12-31")
        assert target.source_id == "exchange"
        assert "binance" in target.uri.lower()
        assert target.method == "GET"

    def test_build_polymarket_target(self):
        """Build Polymarket API target."""
        target = SourceTargetBuilder.build_polymarket_target("test-market")
        assert target.source_id == "polymarket"
        assert "polymarket" in target.uri.lower()

    def test_build_generic_http_target(self):
        """Build generic HTTP target."""
        url = "https://api.example.com/data"
        target = SourceTargetBuilder.build_generic_http_target(url)
        assert target.source_id == "http"
        assert target.uri == url
        assert target.method == "GET"


class TestDeterministicIds:
    """Tests for deterministic ID generation."""

    def test_generate_deterministic_id_stable(self):
        """Same inputs produce same ID."""
        id1 = generate_deterministic_id("test", "a", "b", "c")
        id2 = generate_deterministic_id("test", "a", "b", "c")
        assert id1 == id2

    def test_generate_deterministic_id_different_inputs(self):
        """Different inputs produce different IDs."""
        id1 = generate_deterministic_id("test", "a", "b", "c")
        id2 = generate_deterministic_id("test", "a", "b", "d")
        assert id1 != id2

    def test_generate_deterministic_id_prefix(self):
        """ID starts with correct prefix."""
        id1 = generate_deterministic_id("mkt", "data")
        assert id1.startswith("mkt_")

    def test_generate_requirement_id_format(self):
        """Requirement ID has correct format."""
        req_id = generate_requirement_id(1)
        assert req_id == "req_0001"

        req_id2 = generate_requirement_id(10)
        assert req_id2 == "req_0010"


class TestPromptEngineerAgent:
    """Tests for PromptEngineerLLM (requires LLM client)."""

    def test_agent_requires_llm(self):
        """Agent fails gracefully without LLM."""
        ctx = AgentContext.create_minimal()
        agent = PromptEngineerLLM()
        result = agent.run(ctx, "Will BTC close above $100,000?")
        assert not result.success
        assert "LLM client required" in result.error

    def test_agent_has_name_and_version(self):
        """Agent has name and version."""
        agent = PromptEngineerLLM()
        assert agent.name == "PromptEngineerLLM"
        assert agent.version == "v1"

    def test_agent_with_mock_llm(self):
        """Agent compiles with mock LLM."""
        mock_response = json.dumps({
            "market_id": "mk_test",
            "question": "Will BTC close above $100,000?",
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
        agent = PromptEngineerLLM()
        result = agent.run(ctx, "Will BTC close above $100,000?")

        assert result.success
        prompt_spec, tool_plan = result.output
        assert prompt_spec.market_id == "mk_test"
        assert len(prompt_spec.data_requirements) == 1


class TestOrchestratorWrapper:
    """Tests for compile_prompt convenience function."""

    def test_compile_prompt_requires_llm(self):
        """compile_prompt raises when no LLM is configured."""
        ctx = AgentContext.create_minimal()

        with pytest.raises(ValueError, match="LLM client is required"):
            compile_prompt(ctx, "Will SOL close above $200?")


class TestLLMCompilerJSONExtraction:
    """Tests for LLM compiler JSON extraction."""

    def test_extract_from_code_block(self):
        """Test extraction from markdown code block."""
        compiler = LLMPromptCompiler()
        text = '```json\n{"key": "value"}\n```'
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
