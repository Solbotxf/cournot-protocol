"""
Tests for CollectorGeminiGrounded.

Tests the Gemini-grounded collector with a mock Gemini client to
verify prompt construction, response parsing, grounding extraction,
and EvidenceItem output without requiring a real Google API key.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agents.context import AgentContext
from agents.collector.gemini_grounded_agent import (
    CollectorGeminiGrounded,
    _build_user_prompt,
)
from core.schemas import (
    DataRequirement,
    EvidenceBundle,
    EvidenceItem,
    MarketSpec,
    PromptSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    ToolPlan,
    DisputePolicy,
    PredictionSemantics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def create_test_prompt_spec() -> PromptSpec:
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_gemini_test",
            question="Will Valve launch Premier Season 4 by January 31?",
            event_definition=(
                'This market resolves "Yes" if Valve launches Premier season 4 '
                "by January 31, 2026, 11:59 PM ET."
            ),
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(
                    rule_id="R1",
                    description="Resolve Yes if Premier Season 4 is launched before deadline",
                    priority=100,
                ),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Valve Premier Season 4",
            predicate="launched by deadline",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_gemini_001",
                description="Check if Valve has launched CS2 Premier Season 4",
                source_targets=[],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
                deferred_source_discovery=True,
            ),
        ],
    )


def create_test_tool_plan() -> ToolPlan:
    return ToolPlan(
        plan_id="tp_gemini_test",
        requirements=["req_gemini_001"],
        sources=[],
    )


def _mock_gemini_response(outcome: str = "Yes", reason: str = "Evidence found."):
    """Build a mock Gemini response object with grounding metadata."""
    # Mock the grounding web chunk
    web = MagicMock()
    web.uri = "https://www.counter-strike.net/news/premier-season-4"
    web.title = "CS2 Premier Season 4 Announcement"

    chunk = MagicMock()
    chunk.web = web

    grounding_metadata = MagicMock()
    grounding_metadata.web_search_queries = ["Valve Premier Season 4 launch date 2026"]
    grounding_metadata.grounding_chunks = [chunk]

    # Mock part with text
    part = MagicMock()
    part.text = json.dumps({"outcome": outcome, "reason": reason})

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content
    candidate.grounding_metadata = grounding_metadata

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    """Test prompt construction from PromptSpec."""

    def test_prompt_includes_question(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "Will Valve launch Premier Season 4" in prompt

    def test_prompt_includes_event_definition(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "January 31, 2026" in prompt

    def test_prompt_includes_resolution_rules(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "[R1]" in prompt

    def test_prompt_includes_requirement_description(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "CS2 Premier Season 4" in prompt

    def test_prompt_includes_semantics(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "Valve Premier Season 4" in prompt
        assert "launched by deadline" in prompt


class TestResponseParsing:
    """Test JSON parsing from Gemini response text."""

    def test_parse_clean_json(self):
        text = '{"outcome": "Yes", "reason": "It launched."}'
        parsed = CollectorGeminiGrounded._parse_json(text)
        assert parsed["outcome"] == "Yes"

    def test_parse_markdown_fenced_json(self):
        text = '```json\n{"outcome": "No", "reason": "Not yet."}\n```'
        parsed = CollectorGeminiGrounded._parse_json(text)
        assert parsed["outcome"] == "No"

    def test_parse_json_with_surrounding_text(self):
        text = 'Here is the result: {"outcome": "Yes", "reason": "Done."} end.'
        parsed = CollectorGeminiGrounded._parse_json(text)
        assert parsed["outcome"] == "Yes"

    def test_parse_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            CollectorGeminiGrounded._parse_json("not json at all")


class TestGroundingExtraction:
    """Test extraction of grounding metadata from mock response."""

    def test_extract_sources(self):
        resp = _mock_gemini_response()
        grounding = CollectorGeminiGrounded._extract_grounding(resp)
        assert len(grounding["sources"]) == 1
        assert "counter-strike.net" in grounding["sources"][0]["uri"]

    def test_extract_search_queries(self):
        resp = _mock_gemini_response()
        grounding = CollectorGeminiGrounded._extract_grounding(resp)
        assert len(grounding["search_queries"]) == 1
        assert "Premier Season 4" in grounding["search_queries"][0]

    def test_extract_text(self):
        resp = _mock_gemini_response(outcome="No", reason="Not launched.")
        text = CollectorGeminiGrounded._extract_text(resp)
        parsed = json.loads(text)
        assert parsed["outcome"] == "No"

    def test_empty_grounding_when_missing(self):
        candidate = MagicMock()
        candidate.grounding_metadata = None
        candidate.content = MagicMock()
        candidate.content.parts = [MagicMock(text="{}")]

        response = MagicMock()
        response.candidates = [candidate]

        grounding = CollectorGeminiGrounded._extract_grounding(response)
        assert grounding["sources"] == []
        assert grounding["search_queries"] == []


class TestCollectorGeminiGrounded:
    """Integration tests with mocked Gemini client."""

    def _run_with_mock(self, mock_response):
        """Run the collector with a mocked Gemini client."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        agent = CollectorGeminiGrounded()

        # Patch env for API key, plus _get_client and _call_gemini to avoid importing google.genai
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_call_gemini", return_value=mock_response):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())
        return result, mock_client

    def test_returns_valid_evidence_bundle(self):
        resp = _mock_gemini_response("Yes", "Premier Season 4 launched on Jan 20.")
        result, _ = self._run_with_mock(resp)

        assert result.success
        bundle, exec_log = result.output
        assert isinstance(bundle, EvidenceBundle)
        assert bundle.market_id == "mk_gemini_test"
        assert len(bundle.items) == 1
        assert bundle.items[0].success
        assert bundle.items[0].parsed_value == "Yes"

    def test_evidence_item_has_grounding_sources(self):
        resp = _mock_gemini_response("No", "No announcement found.")
        result, _ = self._run_with_mock(resp)

        item = result.output[0].items[0]
        sources = item.extracted_fields.get("evidence_sources", [])
        assert len(sources) == 1
        assert "counter-strike.net" in sources[0]["url"]

    def test_evidence_item_has_search_queries(self):
        resp = _mock_gemini_response("Yes", "Found it.")
        result, _ = self._run_with_mock(resp)

        item = result.output[0].items[0]
        queries = item.extracted_fields.get("grounding_search_queries", [])
        assert len(queries) >= 1

    def test_execution_log_has_one_call(self):
        resp = _mock_gemini_response("Yes", "Done.")
        result, _ = self._run_with_mock(resp)

        _, exec_log = result.output
        assert len(exec_log.calls) == 1
        assert exec_log.calls[0].tool == "gemini_grounded:search_and_resolve"

    def test_metadata_includes_collector_info(self):
        resp = _mock_gemini_response("Yes", "Done.")
        result, _ = self._run_with_mock(resp)

        assert result.metadata["collector"] == "gemini_grounded"
        assert result.metadata["model"] == "gemini-2.5-flash"

    def test_no_api_key_returns_failure(self):
        agent = CollectorGeminiGrounded()
        ctx = AgentContext.create_minimal()
        # No GOOGLE_API_KEY in env, no config, no explicit key
        with patch.dict("os.environ", {}, clear=True):
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())
        assert not result.success
        assert "API key" in result.error

    def test_gemini_error_returns_failure_evidence(self):
        agent = CollectorGeminiGrounded()

        mock_client = MagicMock()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_call_gemini", side_effect=RuntimeError("API quota exceeded")):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())

        # Should still return a result (not crash), with the error recorded
        bundle, exec_log = result.output
        assert len(bundle.items) == 1
        assert not bundle.items[0].success
        assert "API quota exceeded" in bundle.items[0].error
        assert exec_log.calls[0].error is not None

    def test_custom_model(self):
        resp = _mock_gemini_response("Yes", "Found.")
        mock_client = MagicMock()

        agent = CollectorGeminiGrounded(model="gemini-2.5-pro")
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_call_gemini", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())

        assert result.metadata["model"] == "gemini-2.5-pro"
