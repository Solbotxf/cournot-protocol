"""
Tests for CollectorOpenSearch.

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
    CollectorOpenSearch,
    _build_user_prompt,
    _build_targeted_source_prompt,
    _extract_required_domains,
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
