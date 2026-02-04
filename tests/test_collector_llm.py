"""
Tests for CollectorLLM agent.

Verifies LLM-based evidence collection including JSON parsing,
retry logic, provenance, and registry integration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agents.collector.agent import CollectorLLM, get_collector
from agents.context import AgentContext
from agents.base import AgentCapability, AgentStep
from agents.registry import get_registry
from core.schemas import (
    DataRequirement,
    DisputePolicy,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourceTarget,
    ToolPlan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prompt_spec(
    *,
    requirements: list[DataRequirement] | None = None,
) -> PromptSpec:
    """Create a minimal PromptSpec for testing."""
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    if requirements is None:
        requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(
                        source_id="coingecko",
                        uri="https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
        ]

    return PromptSpec(
        market=MarketSpec(
            market_id="mk_test_llm",
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
            threshold="100000",
        ),
        data_requirements=requirements,
    )


def _make_tool_plan(requirements: list[str] | None = None) -> ToolPlan:
    """Create a minimal ToolPlan for testing."""
    return ToolPlan(
        plan_id="plan_test_llm",
        requirements=requirements or ["req_001"],
        sources=["coingecko"],
    )


def _good_llm_json(**overrides) -> str:
    """Return a valid LLM JSON response string."""
    data = {
        "success": True,
        "extracted_fields": {"price_usd": 105000},
        "parsed_value": 105000,
        "content_summary": "Bitcoin current price is $105,000 USD.",
        "error": None,
    }
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectorLLMRequiresLLM:
    """CollectorLLM must fail gracefully when ctx.llm is None."""

    def test_requires_llm(self):
        ctx = AgentContext.create_mock()
        # Remove LLM from context
        ctx.llm = None
        collector = CollectorLLM()
        result = collector.run(ctx, _make_prompt_spec(), _make_tool_plan())
        assert not result.success
        assert "LLM client not available" in result.error


class TestCollectorLLMSuccess:
    """Successful single-requirement collection."""

    def test_success_single_requirement(self):
        response_json = _good_llm_json()
        ctx = AgentContext.create_mock(llm_responses=[response_json])

        collector = CollectorLLM()
        result = collector.run(ctx, _make_prompt_spec(), _make_tool_plan())

        assert result.success
        bundle, execution_log = result.output
        assert len(bundle.items) == 1
        item = bundle.items[0]
        assert item.success
        assert item.parsed_value == 105000
        assert item.extracted_fields["price_usd"] == 105000
        assert item.requirement_id == "req_001"
        assert "req_001" in bundle.requirements_fulfilled
        assert execution_log.total_calls == 1


class TestCollectorLLMJsonRepair:
    """JSON repair loop: first response invalid, second valid."""

    def test_json_repair_loop(self):
        bad_response = "Here is the data: {invalid json"
        good_response = _good_llm_json()

        ctx = AgentContext.create_mock(llm_responses=[bad_response, good_response])

        collector = CollectorLLM()
        result = collector.run(ctx, _make_prompt_spec(), _make_tool_plan())

        assert result.success
        bundle, _ = result.output
        assert len(bundle.items) == 1
        assert bundle.items[0].success


class TestCollectorLLMRetriesExhausted:
    """All retries exhausted → error evidence."""

    def test_all_retries_exhausted(self):
        bad_response = "not json at all {"
        # Need MAX_RETRIES + 1 = 3 bad responses (initial + 2 retries)
        ctx = AgentContext.create_mock(
            llm_responses=[bad_response, bad_response, bad_response]
        )

        collector = CollectorLLM()
        result = collector.run(ctx, _make_prompt_spec(), _make_tool_plan())

        # Should still return a result (with error evidence)
        assert result.success  # bundle has_evidence is True (error item still added)
        bundle, _ = result.output
        assert len(bundle.items) == 1
        item = bundle.items[0]
        assert not item.success
        assert "JSON parse failed" in item.error


class TestCollectorLLMMultipleRequirements:
    """Iterates tool_plan requirements correctly."""

    def test_multiple_requirements(self):
        requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(source_id="coingecko", uri="https://api.coingecko.com/btc"),
                ],
                selection_policy=SelectionPolicy(strategy="single_best"),
            ),
            DataRequirement(
                requirement_id="req_002",
                description="Get ETH price",
                source_targets=[
                    SourceTarget(source_id="coingecko", uri="https://api.coingecko.com/eth"),
                ],
                selection_policy=SelectionPolicy(strategy="single_best"),
            ),
        ]

        resp1 = _good_llm_json(parsed_value=105000)
        resp2 = _good_llm_json(parsed_value=3500)

        ctx = AgentContext.create_mock(llm_responses=[resp1, resp2])

        prompt_spec = _make_prompt_spec(requirements=requirements)
        tool_plan = _make_tool_plan(requirements=["req_001", "req_002"])

        collector = CollectorLLM()
        result = collector.run(ctx, prompt_spec, tool_plan)

        assert result.success
        bundle, execution_log = result.output
        assert len(bundle.items) == 2
        assert execution_log.total_calls == 2
        assert "req_001" in bundle.requirements_fulfilled
        assert "req_002" in bundle.requirements_fulfilled


class TestCollectorLLMSingleBestStrategy:
    """Stops after first source for single_best strategy."""

    def test_single_best_strategy(self):
        requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(source_id="source_a", uri="https://a.com/price"),
                    SourceTarget(source_id="source_b", uri="https://b.com/price"),
                ],
                selection_policy=SelectionPolicy(strategy="single_best"),
            ),
        ]

        ctx = AgentContext.create_mock(llm_responses=[_good_llm_json()])

        collector = CollectorLLM()
        result = collector.run(
            ctx, _make_prompt_spec(requirements=requirements), _make_tool_plan()
        )

        assert result.success
        bundle, execution_log = result.output
        # Should only have 1 item despite 2 source targets
        assert len(bundle.items) == 1
        assert execution_log.total_calls == 1


class TestCollectorLLMProvenanceTier:
    """Evidence has provenance tier=2 (LLM-interpreted)."""

    def test_provenance_tier(self):
        ctx = AgentContext.create_mock(llm_responses=[_good_llm_json()])

        collector = CollectorLLM()
        result = collector.run(ctx, _make_prompt_spec(), _make_tool_plan())

        bundle, _ = result.output
        item = bundle.items[0]
        assert item.provenance.tier == 2


class TestCollectorLLMRawContentTruncated:
    """raw_content is truncated to <= 100 chars."""

    def test_raw_content_truncated(self):
        long_summary = "x" * 200
        ctx = AgentContext.create_mock(
            llm_responses=[_good_llm_json(content_summary=long_summary)]
        )

        collector = CollectorLLM()
        result = collector.run(ctx, _make_prompt_spec(), _make_tool_plan())

        bundle, _ = result.output
        item = bundle.items[0]
        assert len(item.raw_content) <= 100


class TestCollectorLLMRegistryPriority:
    """CollectorLLM priority=150 > CollectorHTTP priority=100."""

    def test_registry_priority(self):
        registry = get_registry()
        entries = registry._entries.get(AgentStep.COLLECTOR, [])

        llm_entry = next(e for e in entries if e.name == "CollectorLLM")
        http_entry = next(e for e in entries if e.name == "CollectorHTTP")

        assert llm_entry.priority == 150
        assert http_entry.priority == 100
        assert llm_entry.priority > http_entry.priority


class TestCollectorLLMFallbackSelection:
    """Pipeline selects CollectorHTTP when no LLM available."""

    def test_fallback_selection(self):
        ctx = AgentContext.create_mock()
        ctx.llm = None

        # get_collector should fall back to mock (no http either)
        collector = get_collector(ctx)
        assert not isinstance(collector, CollectorLLM)

    def test_selects_llm_when_available(self):
        ctx = AgentContext.create_mock(llm_responses=["{}"])

        collector = get_collector(ctx)
        assert isinstance(collector, CollectorLLM)

    def test_skips_llm_when_prefer_llm_false(self):
        ctx = AgentContext.create_mock(llm_responses=["{}"])

        collector = get_collector(ctx, prefer_llm=False)
        assert not isinstance(collector, CollectorLLM)
