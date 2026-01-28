"""
Unit tests for Module 04 - Prompt Engineer.

Tests verify:
1. Every requirement has ≥1 source_targets and valid uri
2. ToolPlan.requirements exactly equals DataRequirement.requirement_id in order
3. OutputSchemaRef points to DeterministicVerdict
4. Confidence policy exists in PromptSpec.extra and is numeric
5. Deterministic IDs stable across runs
6. If user input missing timeframe, assumptions are recorded and default window applied
"""

import pytest
from datetime import datetime, timezone, timedelta

from agents.prompt_engineer import (
    PromptEngineerAgent,
    PromptModule,
    StrictPromptCompilerV1,
    InputParser,
    SourceTargetBuilder,
    NormalizedUserRequest,
    generate_deterministic_id,
    generate_requirement_id,
)
from agents.base_agent import AgentContext
from core.schemas.errors import PromptCompilationException
from orchestrator.prompt_engineer import compile_prompt, validate_prompt_spec


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


class TestStrictPromptCompiler:
    """Tests for StrictPromptCompilerV1."""
    
    def test_compile_btc_price_prediction(self):
        """Compile BTC price prediction with Coinbase."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31 UTC? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        # Verify strict mode
        assert prompt_spec.extra.get("strict_mode") is True
        
        # Verify event definition
        assert "BTC" in prompt_spec.market.event_definition
        assert "100000" in prompt_spec.market.event_definition
        
        # Verify data requirements have source targets
        assert len(prompt_spec.data_requirements) > 0
        for req in prompt_spec.data_requirements:
            assert len(req.source_targets) >= 1
            for target in req.source_targets:
                assert target.uri
                assert target.method
    
    def test_compile_missing_timeframe_records_assumption(self):
        """Missing timeframe records assumption and uses default."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC reach $100,000? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        # Should have recorded assumption about default timeframe
        assumptions = prompt_spec.extra.get("assumptions", [])
        assert any("default" in a.lower() or "window" in a.lower() for a in assumptions)
    
    def test_output_schema_ref_is_deterministic_verdict(self):
        """Output schema ref points to DeterministicVerdict."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        assert prompt_spec.output_schema_ref == "core.schemas.verdict.DeterministicVerdict"
    
    def test_confidence_policy_exists_and_numeric(self):
        """Confidence policy exists and has numeric values."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        confidence_policy = prompt_spec.extra.get("confidence_policy")
        assert confidence_policy is not None
        assert isinstance(confidence_policy.get("min_confidence_for_yesno"), (int, float))
        assert isinstance(confidence_policy.get("default_confidence"), (int, float))
    
    def test_tool_plan_requirements_match_data_requirements(self):
        """ToolPlan.requirements matches DataRequirement.requirement_id in order."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        data_req_ids = [req.requirement_id for req in prompt_spec.data_requirements]
        assert tool_plan.requirements == data_req_ids
    
    def test_deterministic_ids_stable_across_runs(self):
        """Same input produces same IDs across multiple runs."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31 UTC? Use Coinbase."
        
        spec1, plan1 = compiler.compile(user_input)
        spec2, plan2 = compiler.compile(user_input)
        
        assert spec1.market.market_id == spec2.market.market_id
        assert plan1.plan_id == plan2.plan_id
        
        for req1, req2 in zip(spec1.data_requirements, spec2.data_requirements):
            assert req1.requirement_id == req2.requirement_id
    
    def test_empty_input_raises_exception(self):
        """Empty input raises PromptCompilationException."""
        agent = PromptEngineerAgent()
        
        with pytest.raises(PromptCompilationException):
            agent.run("")
    
    def test_created_at_is_none_for_deterministic_hashing(self):
        """PromptSpec.created_at is None for deterministic hashing."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        assert prompt_spec.created_at is None


class TestPromptEngineerAgent:
    """Tests for PromptEngineerAgent."""
    
    def test_agent_basic_compilation(self):
        """Agent compiles basic input."""
        agent = PromptEngineerAgent()
        user_input = "Will ETH close above $5000 on 2026-06-30? Use Binance."
        
        prompt_spec, tool_plan = agent.run(user_input)
        
        assert prompt_spec is not None
        assert tool_plan is not None
        assert prompt_spec.market.market_id
    
    def test_agent_with_custom_context(self):
        """Agent works with custom context."""
        context = AgentContext(
            session_id="test-session",
            config={"test": True},
        )
        agent = PromptEngineerAgent(context=context)
        user_input = "Will BTC reach $100,000 by end of 2026? Use Coinbase."
        
        prompt_spec, tool_plan = agent.run(user_input)
        
        assert prompt_spec is not None
    
    def test_agent_get_module_info(self):
        """Agent returns module info."""
        agent = PromptEngineerAgent()
        info = agent.get_module_info()
        
        assert "module_id" in info
        assert "version" in info
    
    def test_agent_validate_input_empty(self):
        """Agent validates empty input."""
        agent = PromptEngineerAgent()
        assert agent.validate_input("") is False
        assert agent.validate_input("   ") is False
    
    def test_agent_validate_input_valid(self):
        """Agent validates non-empty input."""
        agent = PromptEngineerAgent()
        assert agent.validate_input("Will BTC reach 100k?") is True


class TestOrchestratorWrapper:
    """Tests for orchestrator wrapper functions."""
    
    def test_compile_prompt_function(self):
        """compile_prompt function works."""
        user_input = "Will SOL close above $200 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compile_prompt(user_input)
        
        assert prompt_spec is not None
        assert tool_plan is not None
    
    def test_validate_prompt_spec_valid(self):
        """validate_prompt_spec returns True for valid spec."""
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        prompt_spec, tool_plan = compile_prompt(user_input)
        
        is_valid, errors = validate_prompt_spec(prompt_spec)
        
        assert is_valid is True
        assert len(errors) == 0


class TestSourceTargetValidation:
    """Tests for source target validation."""
    
    def test_every_requirement_has_source_targets(self):
        """Every requirement has at least one source target."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        for req in prompt_spec.data_requirements:
            assert len(req.source_targets) >= 1, f"Requirement {req.requirement_id} has no source targets"
    
    def test_every_source_target_has_valid_uri(self):
        """Every source target has a valid URI."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        for req in prompt_spec.data_requirements:
            for target in req.source_targets:
                assert target.uri, f"Source target in {req.requirement_id} has empty URI"
                # Should start with http:// or https://
                assert target.uri.startswith(("http://", "https://")), \
                    f"Source target URI {target.uri} is not a valid HTTP(S) URL"
    
    def test_selection_policy_present(self):
        """Every requirement has a selection policy."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        for req in prompt_spec.data_requirements:
            assert req.selection_policy is not None
            assert req.selection_policy.strategy in ["single_best", "multi_source_quorum", "fallback_chain"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_explicit_url_in_input(self):
        """Handle explicit URLs in user input."""
        compiler = StrictPromptCompilerV1()
        user_input = "Check https://api.coinbase.com/v2/prices/BTC-USD/spot for BTC price on 2026-12-31"
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        # Should have captured the explicit URL
        all_uris = []
        for req in prompt_spec.data_requirements:
            for target in req.source_targets:
                all_uris.append(target.uri)
        
        # The explicit URL should be in the targets
        assert any("coinbase" in uri.lower() for uri in all_uris)
    
    def test_multiple_sources_creates_fallback_chain(self):
        """Multiple sources from same provider use fallback chain."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase and Binance."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        # Should have exchange requirement with fallback chain
        exchange_reqs = [r for r in prompt_spec.data_requirements if "exchange" in r.preferred_sources]
        if exchange_reqs:
            req = exchange_reqs[0]
            if len(req.source_targets) > 1:
                assert req.selection_policy.strategy == "fallback_chain"
    
    def test_forbidden_behaviors_present(self):
        """Forbidden behaviors are specified."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        assert len(prompt_spec.forbidden_behaviors) > 0
        assert any("hallucinate" in fb.lower() for fb in prompt_spec.forbidden_behaviors)


class TestCompilerInfo:
    """Tests for compiler metadata."""
    
    def test_compiler_info_in_extra(self):
        """Compiler info is recorded in extra."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        compiler_info = prompt_spec.extra.get("compiler")
        assert compiler_info is not None
        assert compiler_info.get("module_id") == "strict_compiler"
        assert compiler_info.get("version") == "1.0.0"
    
    def test_id_scheme_documented(self):
        """ID scheme is documented in extra."""
        compiler = StrictPromptCompilerV1()
        user_input = "Will BTC close above $100,000 on 2026-12-31? Use Coinbase."
        
        prompt_spec, tool_plan = compiler.compile(user_input)
        
        assert prompt_spec.extra.get("id_scheme") == "deterministic_sha256"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])