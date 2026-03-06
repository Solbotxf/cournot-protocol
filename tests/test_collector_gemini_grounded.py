"""
Tests for CollectorOpenSearch.

Tests the Gemini-grounded collector with a mock Gemini client to
verify prompt construction, response parsing, grounding extraction,
and EvidenceItem output without requiring a real Google API key.
"""

import json
import re
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agents.context import AgentContext
from agents.collector.gemini_grounded_agent import (
    CollectorOpenSearch,
    SOURCE_ANALYSIS_PROMPT,
    _build_user_prompt,
    _build_targeted_source_prompt,
    _extract_required_domains,
    _source_matches_domain,
    _sources_cover_domains,
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
    SourceTarget,
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


def _mock_gemini_response(
    outcome: str = "Yes",
    reason: str = "Evidence found.",
    grounding_supports=None,
):
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
    grounding_metadata.grounding_supports = grounding_supports

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
        parsed = CollectorOpenSearch._parse_json(text)
        assert parsed["outcome"] == "Yes"

    def test_parse_markdown_fenced_json(self):
        text = '```json\n{"outcome": "No", "reason": "Not yet."}\n```'
        parsed = CollectorOpenSearch._parse_json(text)
        assert parsed["outcome"] == "No"

    def test_parse_json_with_surrounding_text(self):
        text = 'Here is the result: {"outcome": "Yes", "reason": "Done."} end.'
        parsed = CollectorOpenSearch._parse_json(text)
        assert parsed["outcome"] == "Yes"

    def test_parse_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            CollectorOpenSearch._parse_json("not json at all")


class TestGroundingExtraction:
    """Test extraction of grounding metadata from mock response."""

    def test_extract_sources(self):
        resp = _mock_gemini_response()
        grounding = CollectorOpenSearch._extract_grounding(resp)
        assert len(grounding["sources"]) == 1
        assert "counter-strike.net" in grounding["sources"][0]["uri"]

    def test_extract_search_queries(self):
        resp = _mock_gemini_response()
        grounding = CollectorOpenSearch._extract_grounding(resp)
        assert len(grounding["search_queries"]) == 1
        assert "Premier Season 4" in grounding["search_queries"][0]

    def test_extract_text(self):
        resp = _mock_gemini_response(outcome="No", reason="Not launched.")
        text = CollectorOpenSearch._extract_text(resp)
        parsed = json.loads(text)
        assert parsed["outcome"] == "No"

    def test_empty_grounding_when_missing(self):
        candidate = MagicMock()
        candidate.grounding_metadata = None
        candidate.content = MagicMock()
        candidate.content.parts = [MagicMock(text="{}")]

        response = MagicMock()
        response.candidates = [candidate]

        grounding = CollectorOpenSearch._extract_grounding(response)
        assert grounding["sources"] == []
        assert grounding["search_queries"] == []


class TestCollectorOpenSearch:
    """Integration tests with mocked Gemini client."""

    def _run_with_mock(self, mock_response):
        """Run the collector with a mocked Gemini client."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        agent = CollectorOpenSearch()

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
        assert exec_log.calls[0].tool == "open_search:search_and_resolve"

    def test_metadata_includes_collector_info(self):
        resp = _mock_gemini_response("Yes", "Done.")
        result, _ = self._run_with_mock(resp)

        assert result.metadata["collector"] == "open_search"
        assert result.metadata["model"] == "gemini-2.5-flash"

    def test_no_api_key_returns_failure(self):
        agent = CollectorOpenSearch()
        ctx = AgentContext.create_minimal()
        # No GOOGLE_API_KEY in env, no config, no explicit key
        with patch.dict("os.environ", {}, clear=True):
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())
        assert not result.success
        assert "API key" in result.error

    def test_gemini_error_returns_failure_evidence(self):
        agent = CollectorOpenSearch()

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

        agent = CollectorOpenSearch(model="gemini-2.5-pro")
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_call_gemini", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())

        assert result.metadata["model"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Data-source-aware search fixtures and helpers
# ---------------------------------------------------------------------------

def _create_spec_with_source_targets() -> PromptSpec:
    """Create a PromptSpec with explicit source_targets (e.g. fotmob.com)."""
    now = datetime(2026, 2, 21, 17, 30, 0, tzinfo=timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_shots_test",
            question="Will Real Madrid make 5+ shots outside box vs Osasuna on Feb 21?",
            event_definition=(
                "Real Madrid records 5 or more shots outside the box "
                "during the La Liga match against Osasuna on February 21, 2026."
            ),
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(
                    rule_id="R1",
                    description="Resolve Yes if shots outside box >= 5",
                    priority=100,
                ),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Real Madrid",
            predicate="records 5 or more shots outside the box",
            threshold="5",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_shots_001",
                description="Real Madrid's shots outside the box count from the match.",
                source_targets=[
                    SourceTarget(
                        source_id="web",
                        uri="https://www.fotmob.com",
                        method="GET",
                        expected_content_type="html",
                        operation="search",
                        search_query="site:fotmob.com Real Madrid vs Osasuna February 21, 2026 shots outside box",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="fallback_chain",
                    min_sources=1,
                    max_sources=3,
                    quorum=1,
                ),
            ),
        ],
    )


def _create_tool_plan_with_sources() -> ToolPlan:
    return ToolPlan(
        plan_id="tp_shots_test",
        requirements=["req_shots_001"],
        sources=["web"],
    )


def _mock_gemini_response_with_source(
    outcome: str = "Yes",
    reason: str = "8 shots found.",
    source_uri: str = "https://www.fotmob.com/matches/real-madrid-vs-osasuna",
    source_title: str = "FotMob - Real Madrid vs Osasuna",
    grounding_supports=None,
):
    """Build a mock Gemini response with a specific source URI."""
    web = MagicMock()
    web.uri = source_uri
    web.title = source_title

    chunk = MagicMock()
    chunk.web = web

    grounding_metadata = MagicMock()
    grounding_metadata.web_search_queries = ["Real Madrid Osasuna shots outside box"]
    grounding_metadata.grounding_chunks = [chunk]
    grounding_metadata.grounding_supports = grounding_supports

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
# Tests: Data-source domain extraction
# ---------------------------------------------------------------------------

class TestExtractRequiredDomains:
    """Test extraction of required domains from DataRequirement."""

    def test_extracts_domain_from_url(self):
        spec = _create_spec_with_source_targets()
        req = spec.data_requirements[0]
        domains = _extract_required_domains(req)
        assert len(domains) == 1
        assert domains[0]["domain"] == "fotmob.com"

    def test_includes_search_query(self):
        spec = _create_spec_with_source_targets()
        req = spec.data_requirements[0]
        domains = _extract_required_domains(req)
        assert "search_query" in domains[0]
        assert "site:fotmob.com" in domains[0]["search_query"]

    def test_empty_for_deferred_discovery(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        domains = _extract_required_domains(req)
        assert domains == []

    def test_strips_www_prefix(self):
        req = DataRequirement(
            requirement_id="req_test",
            description="Test",
            source_targets=[
                SourceTarget(
                    source_id="web",
                    uri="https://www.example.com/data",
                    expected_content_type="json",
                ),
            ],
            selection_policy=SelectionPolicy(
                strategy="single_best", min_sources=1, max_sources=1, quorum=1,
            ),
        )
        domains = _extract_required_domains(req)
        assert domains[0]["domain"] == "example.com"


class TestSourcesCoverDomains:
    """Test domain coverage check logic."""

    def test_matches_exact_domain(self):
        sources = [{"uri": "https://www.fotmob.com/matches/123"}]
        required = [{"domain": "fotmob.com"}]
        assert _sources_cover_domains(sources, required) is True

    def test_no_match(self):
        sources = [{"uri": "https://flashscore.com/match/123"}]
        required = [{"domain": "fotmob.com"}]
        assert _sources_cover_domains(sources, required) is False

    def test_empty_required_is_covered(self):
        sources = [{"uri": "https://example.com"}]
        assert _sources_cover_domains(sources, []) is True

    def test_empty_sources_not_covered(self):
        required = [{"domain": "fotmob.com"}]
        assert _sources_cover_domains([], required) is False

    def test_subdomain_match(self):
        sources = [{"uri": "https://api.fotmob.com/matchstats"}]
        required = [{"domain": "fotmob.com"}]
        assert _sources_cover_domains(sources, required) is True

    def test_vertex_redirect_matched_by_title(self):
        """Vertex AI redirect URLs should match via grounding title."""
        sources = [{
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123",
            "title": "FotMob - Real Madrid vs Osasuna",
        }]
        required = [{"domain": "fotmob.com"}]
        assert _sources_cover_domains(sources, required) is True

    def test_vertex_redirect_no_match_when_title_differs(self):
        sources = [{
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123",
            "title": "ESPN - La Liga Results",
        }]
        required = [{"domain": "fotmob.com"}]
        assert _sources_cover_domains(sources, required) is False


class TestSourceMatchesDomain:
    """Test the per-source domain matching helper."""

    def test_matches_host(self):
        src = {"uri": "https://www.fotmob.com/matches/123", "title": "Some page"}
        assert _source_matches_domain(src, "fotmob.com") is True

    def test_matches_title(self):
        src = {
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/xyz",
            "title": "FotMob - Match Stats",
        }
        assert _source_matches_domain(src, "fotmob.com") is True

    def test_no_match(self):
        src = {
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/xyz",
            "title": "ESPN - Scores",
        }
        assert _source_matches_domain(src, "fotmob.com") is False

    def test_title_match_case_insensitive(self):
        src = {
            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/xyz",
            "title": "FOTMOB Live Scores",
        }
        assert _source_matches_domain(src, "fotmob.com") is True


# ---------------------------------------------------------------------------
# Tests: Prompt includes data-source hints
# ---------------------------------------------------------------------------

class TestBuildUserPromptWithSources:
    """Test that the prompt includes preferred data-source hints."""

    def test_prompt_includes_preferred_source_domain(self):
        spec = _create_spec_with_source_targets()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "fotmob.com" in prompt
        assert "Preferred data sources" in prompt

    def test_prompt_without_sources_has_no_hint(self):
        spec = create_test_prompt_spec()
        req = spec.data_requirements[0]
        prompt = _build_user_prompt(spec, req)
        assert "Preferred data sources" not in prompt


class TestBuildTargetedSourcePrompt:
    """Test the targeted (pass 2) prompt construction."""

    def test_includes_site_search(self):
        spec = _create_spec_with_source_targets()
        req = spec.data_requirements[0]
        domains = _extract_required_domains(req)
        prompt = _build_targeted_source_prompt(spec, req, domains)
        assert "site:fotmob.com" in prompt
        assert "MUST search" in prompt

    def test_includes_first_pass_context(self):
        spec = _create_spec_with_source_targets()
        req = spec.data_requirements[0]
        domains = _extract_required_domains(req)
        prompt = _build_targeted_source_prompt(
            spec, req, domains,
            first_pass_result={"outcome": "Yes", "reason": "8 shots found on flashscore"},
        )
        assert "previous search found" in prompt
        assert "outcome=Yes" in prompt


# ---------------------------------------------------------------------------
# Tests: Two-pass collection with data-source awareness
# ---------------------------------------------------------------------------

class TestTwoPassCollection:
    """Test the two-pass collection logic when source_targets are specified."""

    def test_pass1_covers_domain_no_pass2(self):
        """When pass 1 finds the required domain, no pass 2 should happen."""
        pass1_resp = _mock_gemini_response_with_source(
            outcome="Yes", reason="8 shots on fotmob",
            source_uri="https://www.fotmob.com/matches/123",
        )

        agent = CollectorOpenSearch()
        call_count = 0

        def mock_call(client, prompt):
            nonlocal call_count
            call_count += 1
            return pass1_resp

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        assert result.success
        assert call_count == 1  # Only pass 1
        item = result.output[0].items[0]
        assert item.extracted_fields["pass_used"] == "pass_1"
        assert item.provenance.tier == 1  # tier 1 because domain matched

    def test_pass2_triggered_when_domain_missed(self):
        """When pass 1 misses the required domain, pass 2 should be triggered."""
        pass1_resp = _mock_gemini_response_with_source(
            outcome="Yes", reason="8 shots on flashscore",
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore",
        )
        pass2_resp = _mock_gemini_response_with_source(
            outcome="Yes", reason="8 shots confirmed on fotmob",
            source_uri="https://www.fotmob.com/matches/osasuna-real-madrid",
            source_title="FotMob Match Stats",
        )

        agent = CollectorOpenSearch()
        call_count = 0

        def mock_call(client, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pass1_resp
            return pass2_resp

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        assert result.success
        assert call_count == 2  # Both passes
        item = result.output[0].items[0]
        assert item.extracted_fields["pass_used"] == "pass_2_targeted"
        assert item.provenance.tier == 1  # tier 1 because pass 2 found domain

    def test_pass2_fails_gracefully_uses_pass1(self):
        """When pass 2 fails, pass 1 results should be used."""
        pass1_resp = _mock_gemini_response_with_source(
            outcome="Yes", reason="6 shots found",
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore - Live Scores",
        )

        agent = CollectorOpenSearch()
        call_count = 0

        def mock_call(client, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pass1_resp
            raise RuntimeError("API error on pass 2")

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        assert result.success
        assert call_count == 2
        item = result.output[0].items[0]
        assert item.extracted_fields["pass_used"] == "pass_1"
        assert item.parsed_value == "Yes"

    def test_data_source_evidence_marked_tier1(self):
        """Evidence from required domains should be marked as tier 1."""
        resp = _mock_gemini_response_with_source(
            source_uri="https://www.fotmob.com/matches/stats",
        )

        agent = CollectorOpenSearch()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert len(sources) == 1
        assert sources[0]["credibility_tier"] == 1
        assert sources[0]["is_required_data_source"] is True

    def test_non_required_source_stays_tier2(self):
        """Evidence NOT from required domains should remain tier 2."""
        resp = _mock_gemini_response_with_source(
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore Stats",
        )
        # No pass 2 triggered since deferred_source_discovery spec has no source_targets
        agent = CollectorOpenSearch()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert len(sources) == 1
        assert sources[0]["credibility_tier"] == 2

    def test_no_source_targets_skips_pass2(self):
        """When no source_targets are specified, only pass 1 runs."""
        resp = _mock_gemini_response("Yes", "Found it.")

        agent = CollectorOpenSearch()
        call_count = 0

        def mock_call(client, prompt):
            nonlocal call_count
            call_count += 1
            return resp

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())

        assert call_count == 1
        assert result.success


class TestMergeGrounding:
    """Test grounding metadata merge logic."""

    def test_deduplicates_by_uri(self):
        first = {
            "sources": [
                {"uri": "https://a.com", "title": "A"},
                {"uri": "https://b.com", "title": "B"},
            ],
            "search_queries": ["query 1"],
        }
        second = {
            "sources": [
                {"uri": "https://b.com", "title": "B duplicate"},
                {"uri": "https://c.com", "title": "C"},
            ],
            "search_queries": ["query 2", "query 1"],
        }
        merged = CollectorOpenSearch._merge_grounding(first, second)
        assert len(merged["sources"]) == 3
        uris = [s["uri"] for s in merged["sources"]]
        assert "https://a.com" in uris
        assert "https://b.com" in uris
        assert "https://c.com" in uris
        # Second pass sources come first (higher priority)
        assert merged["sources"][0]["uri"] == "https://b.com"
        # Queries deduplicated, order preserved
        assert merged["search_queries"] == ["query 1", "query 2"]


class TestBuildEvidenceSources:
    """Test evidence source tier assignment."""

    def test_required_domain_gets_tier1(self):
        grounding = {
            "sources": [
                {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob"},
                {"uri": "https://flashscore.com/match/456", "title": "Flashscore"},
            ],
        }
        required = [{"domain": "fotmob.com"}]
        sources = CollectorOpenSearch._build_evidence_sources(grounding, required)
        assert sources[0]["credibility_tier"] == 1
        assert sources[0]["is_required_data_source"] is True
        assert sources[1]["credibility_tier"] == 2
        assert sources[1]["is_required_data_source"] is False

    def test_source_id_is_citation_ref(self):
        """source_id should be '[N]' pattern, not page title."""
        grounding = {
            "sources": [
                {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob Match Stats"},
                {"uri": "https://flashscore.com/match/456", "title": "Flashscore Live"},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert sources[0]["source_id"] == "[1]"
        assert sources[1]["source_id"] == "[2]"

    def test_domain_name_from_title(self):
        """domain_name should carry the grounding title for display."""
        grounding = {
            "sources": [
                {"uri": "https://vertexaisearch.cloud.google.com/redirect/abc", "title": "ESPN - NBA Scores"},
                {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob Match Stats"},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert sources[0]["domain_name"] == "ESPN - NBA Scores"
        assert sources[1]["domain_name"] == "FotMob Match Stats"

    def test_domain_name_falls_back_to_host(self):
        """When title is empty, domain_name should fall back to the host."""
        grounding = {
            "sources": [
                {"uri": "https://www.fotmob.com/matches/123", "title": ""},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert sources[0]["domain_name"] == "fotmob.com"

    def test_vertex_redirect_gets_tier1_via_title(self):
        """Vertex AI redirect URLs should get tier 1 when title matches required domain."""
        grounding = {
            "sources": [
                {
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
                    "title": "FotMob - Real Madrid vs Osasuna",
                },
                {
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/def",
                    "title": "ESPN - La Liga Results",
                },
            ],
        }
        required = [{"domain": "fotmob.com"}]
        sources = CollectorOpenSearch._build_evidence_sources(grounding, required)
        assert sources[0]["credibility_tier"] == 1
        assert sources[0]["is_required_data_source"] is True
        assert sources[1]["credibility_tier"] == 2
        assert sources[1]["is_required_data_source"] is False


# ---------------------------------------------------------------------------
# Tests: Evidence source consistency (key_fact, supports, source_id)
# ---------------------------------------------------------------------------

class TestEvidenceSourceConsistency:
    """OpenSearch evidence_sources should have descriptive key_fact, proper supports, and citation-ref source_id."""

    def _run_with_llm_mock(self, llm_analysis_response: str):
        """Run collector with both Gemini and LLM mocked."""
        gemini_resp = _mock_gemini_response_with_source(
            outcome="Yes",
            reason="Real Madrid recorded 8 shots outside the box against Osasuna.",
            source_uri="https://www.fotmob.com/matches/real-madrid-vs-osasuna",
            source_title="FotMob - Real Madrid vs Osasuna",
        )

        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = llm_analysis_response
        mock_llm.chat.return_value = mock_llm_response

        agent = CollectorOpenSearch()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            ctx.llm = mock_llm
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        return result, mock_llm

    def test_evidence_sources_have_descriptive_key_fact(self):
        """key_fact should contain a descriptive fact, not just the page title."""
        analysis_json = json.dumps({"sources": [
            {
                "source_id": "[1]",
                "key_fact": "FotMob match page shows Real Madrid had 8 shots outside the box.",
                "supports": "YES",
            },
        ]})
        result, _ = self._run_with_llm_mock(analysis_json)

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert len(sources) >= 1
        # key_fact should NOT be just the page title
        assert sources[0]["key_fact"] != "FotMob - Real Madrid vs Osasuna"
        assert "shots" in sources[0]["key_fact"].lower() or "8" in sources[0]["key_fact"]

    def test_evidence_sources_have_proper_supports(self):
        """supports should be 'YES' or 'NO', not always 'N/A'."""
        analysis_json = json.dumps({"sources": [
            {
                "source_id": "[1]",
                "key_fact": "FotMob confirms 8 shots outside box.",
                "supports": "YES",
            },
        ]})
        result, _ = self._run_with_llm_mock(analysis_json)

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert sources[0]["supports"] in ("YES", "NO", "N/A")
        assert sources[0]["supports"] == "YES"

    def test_evidence_sources_have_citation_ref_source_id(self):
        """source_id should be '[N]' pattern, not a page title."""
        analysis_json = json.dumps({"sources": [
            {
                "source_id": "[1]",
                "key_fact": "Match data from FotMob.",
                "supports": "YES",
            },
        ]})
        result, _ = self._run_with_llm_mock(analysis_json)

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert re.match(r"^\[\d+\]$", sources[0]["source_id"])

    def test_credibility_tier_enriched_by_llm(self):
        """credibility_tier should vary based on LLM analysis, not always be 2."""
        analysis_json = json.dumps({"sources": [
            {
                "source_id": "[1]",
                "key_fact": "FotMob match page shows 8 shots.",
                "supports": "YES",
                "credibility_tier": 1,
            },
        ]})
        result, _ = self._run_with_llm_mock(analysis_json)

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert sources[0]["credibility_tier"] == 1

    def test_credibility_tier_invalid_kept_as_default(self):
        """Invalid credibility_tier from LLM should be ignored."""
        analysis_json = json.dumps({"sources": [
            {
                "source_id": "[1]",
                "key_fact": "Some fact.",
                "supports": "YES",
                "credibility_tier": 5,
            },
        ]})
        result, _ = self._run_with_llm_mock(analysis_json)

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        # Should keep the default (1 for required domain match in this fixture)
        assert sources[0]["credibility_tier"] in (1, 2)

    def test_analyze_sources_calls_llm(self):
        """_analyze_sources should call ctx.llm.chat()."""
        analysis_json = json.dumps({"sources": [
            {"source_id": "[1]", "key_fact": "Some fact.", "supports": "YES"},
        ]})
        result, mock_llm = self._run_with_llm_mock(analysis_json)
        mock_llm.chat.assert_called_once()

    def test_analyze_sources_falls_back_on_llm_error(self):
        """If LLM fails, evidence sources should retain title-based key_fact."""
        gemini_resp = _mock_gemini_response_with_source(
            outcome="Yes",
            reason="8 shots found.",
            source_uri="https://www.fotmob.com/matches/123",
            source_title="FotMob Match Stats",
        )

        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("LLM call failed")

        agent = CollectorOpenSearch()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            ctx.llm = mock_llm
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        # Should still succeed with fallback values
        assert result.success
        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert len(sources) == 1
        # Falls back to title-based key_fact
        assert sources[0]["key_fact"] == "FotMob Match Stats"

    def test_no_llm_skips_analysis(self):
        """When ctx.llm is None, evidence sources keep initial values."""
        gemini_resp = _mock_gemini_response_with_source(
            outcome="Yes",
            reason="8 shots found.",
            source_uri="https://www.fotmob.com/matches/123",
            source_title="FotMob Match Stats",
        )

        agent = CollectorOpenSearch()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_call_gemini", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            # ctx.llm is None by default
            result = agent.run(ctx, _create_spec_with_source_targets(), _create_tool_plan_with_sources())

        assert result.success
        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        # Without LLM, keeps title-based key_fact and N/A supports
        assert sources[0]["key_fact"] == "FotMob Match Stats"
        assert sources[0]["supports"] == "N/A"


# ---------------------------------------------------------------------------
# Helpers for grounding_supports mocking
# ---------------------------------------------------------------------------

def _make_mock_support(text: str, chunk_indices: list[int], confidence_scores: list[float] | None = None):
    """Build a mock grounding_support entry."""
    sup = MagicMock()
    seg = MagicMock()
    seg.text = text
    sup.segment = seg
    sup.grounding_chunk_indices = chunk_indices
    sup.confidence_scores = confidence_scores or []
    return sup


# ---------------------------------------------------------------------------
# Tests: Grounding supports extraction
# ---------------------------------------------------------------------------

class TestGroundingSupportExtraction:
    """Test extraction of grounding_supports from Gemini responses."""

    def test_extract_supports_text(self):
        """grounding_supports text segments should be extracted."""
        supports = [
            _make_mock_support(
                "Premier Season 4 launched on January 20, 2026.",
                chunk_indices=[0],
                confidence_scores=[0.95],
            ),
        ]
        resp = _mock_gemini_response(grounding_supports=supports)
        grounding = CollectorOpenSearch._extract_grounding(resp)
        assert len(grounding["supports"]) == 1
        assert grounding["supports"][0]["text"] == "Premier Season 4 launched on January 20, 2026."
        assert grounding["supports"][0]["chunk_indices"] == [0]
        assert grounding["supports"][0]["confidence_scores"] == [0.95]

    def test_extract_multiple_supports(self):
        """Multiple grounding_supports should all be extracted."""
        supports = [
            _make_mock_support("Fact one.", chunk_indices=[0]),
            _make_mock_support("Fact two.", chunk_indices=[0]),
        ]
        resp = _mock_gemini_response(grounding_supports=supports)
        grounding = CollectorOpenSearch._extract_grounding(resp)
        assert len(grounding["supports"]) == 2

    def test_extract_empty_when_no_supports(self):
        """When grounding_supports is None, supports list should be empty."""
        resp = _mock_gemini_response()  # grounding_supports defaults to None
        grounding = CollectorOpenSearch._extract_grounding(resp)
        assert grounding["supports"] == []

    def test_skips_support_with_no_text(self):
        """Supports with no text segment should be skipped."""
        sup = MagicMock()
        sup.segment = None
        sup.grounding_chunk_indices = [0]
        sup.confidence_scores = []
        resp = _mock_gemini_response(grounding_supports=[sup])
        grounding = CollectorOpenSearch._extract_grounding(resp)
        assert grounding["supports"] == []


# ---------------------------------------------------------------------------
# Tests: Real key_fact attribution
# ---------------------------------------------------------------------------

class TestRealKeyFactAttribution:
    """Test that key_fact uses real grounding text when available."""

    def test_key_fact_from_grounding_supports(self):
        """key_fact should use real grounding text when available."""
        grounding = {
            "sources": [
                {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob Match Stats"},
            ],
            "supports": [
                {"text": "Real Madrid had 8 shots outside the box.", "chunk_indices": [0], "confidence_scores": [0.9]},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert sources[0]["key_fact"] == "Real Madrid had 8 shots outside the box."

    def test_key_fact_falls_back_to_title(self):
        """When no grounding_supports, key_fact should use title."""
        grounding = {
            "sources": [
                {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob Match Stats"},
            ],
            "supports": [],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert sources[0]["key_fact"] == "FotMob Match Stats"

    def test_key_fact_concatenates_multiple_texts(self):
        """Multiple text segments for same chunk should be concatenated."""
        grounding = {
            "sources": [
                {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob"},
            ],
            "supports": [
                {"text": "Real Madrid had 8 shots outside the box.", "chunk_indices": [0], "confidence_scores": []},
                {"text": "The match ended 2-1.", "chunk_indices": [0], "confidence_scores": []},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert "8 shots outside the box" in sources[0]["key_fact"]
        assert "match ended 2-1" in sources[0]["key_fact"]

    def test_key_fact_truncated_at_500_chars(self):
        """key_fact from supports should be truncated to 500 characters."""
        grounding = {
            "sources": [
                {"uri": "https://example.com", "title": "Example"},
            ],
            "supports": [
                {"text": "x" * 600, "chunk_indices": [0], "confidence_scores": []},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert len(sources[0]["key_fact"]) == 500

    def test_no_supports_key_defaults_to_empty_supports(self):
        """When grounding dict has no 'supports' key, falls back to title."""
        grounding = {
            "sources": [
                {"uri": "https://example.com", "title": "Example Page"},
            ],
        }
        sources = CollectorOpenSearch._build_evidence_sources(grounding, [])
        assert sources[0]["key_fact"] == "Example Page"


# ---------------------------------------------------------------------------
# Tests: _analyze_sources prompt includes attributed text
# ---------------------------------------------------------------------------

class TestAnalyzeSourcesWithRealText:
    """Test that _analyze_sources prompt includes the real attributed text."""

    def test_prompt_includes_attributed_text(self):
        """_analyze_sources prompt should include the real attributed text."""
        evidence_sources = [{
            "url": "https://www.fotmob.com/matches/123",
            "source_id": "[1]",
            "domain_name": "FotMob Match Stats",
            "credibility_tier": 1,
            "key_fact": "Real Madrid had 8 shots outside the box against Osasuna.",
            "supports": "N/A",
            "date_published": None,
            "is_required_data_source": True,
        }]

        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({"sources": [
            {"source_id": "[1]", "key_fact": "8 shots outside the box.", "supports": "YES", "credibility_tier": 1},
        ]})
        mock_llm.chat.return_value = mock_llm_response

        ctx = AgentContext.create_minimal()
        ctx.llm = mock_llm

        CollectorOpenSearch._analyze_sources(ctx, "Yes", "8 shots found.", evidence_sources)

        # Check the prompt sent to LLM
        call_args = mock_llm.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        assert "Attributed text:" in prompt_content
        assert "8 shots outside the box" in prompt_content

    def test_prompt_no_attributed_text_when_key_fact_equals_domain(self):
        """When key_fact equals domain_name (no supports), no Attributed text line."""
        evidence_sources = [{
            "url": "https://www.fotmob.com/matches/123",
            "source_id": "[1]",
            "domain_name": "FotMob Match Stats",
            "credibility_tier": 2,
            "key_fact": "FotMob Match Stats",
            "supports": "N/A",
            "date_published": None,
            "is_required_data_source": False,
        }]

        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({"sources": [
            {"source_id": "[1]", "key_fact": "Some fact.", "supports": "YES", "credibility_tier": 2},
        ]})
        mock_llm.chat.return_value = mock_llm_response

        ctx = AgentContext.create_minimal()
        ctx.llm = mock_llm

        CollectorOpenSearch._analyze_sources(ctx, "Yes", "Found.", evidence_sources)

        call_args = mock_llm.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        assert "Attributed text:" not in prompt_content
