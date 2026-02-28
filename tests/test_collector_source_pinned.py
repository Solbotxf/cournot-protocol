"""
Tests for CollectorSourcePinned.

Tests the strict Gemini-grounded collector that only searches within
required data-source domains specified in source_targets.  Validates
the two-phase approach: Serper URL discovery → Gemini UrlContext+GoogleSearch.
"""

import json
import sys
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

# Ensure google.genai can be imported even when the package isn't installed.
# Several methods import it lazily; tests mock the client so the real SDK
# is never needed, but the import must not raise ModuleNotFoundError.
if "google.genai" not in sys.modules:
    _genai_mock = MagicMock()
    sys.modules.setdefault("google", MagicMock())
    sys.modules["google.genai"] = _genai_mock
    sys.modules["google.genai.types"] = _genai_mock.types

from agents.context import AgentContext
from agents.collector.source_pinned_agent import (
    CollectorSourcePinned,
    _build_strict_prompt,
)
from agents.collector.fotmob import FotMobMatchData, FotMobStat, FotMobShot, FotMobEvent
from core.schemas import (
    DataRequirement,
    EvidenceBundle,
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

def _create_spec_with_sources() -> PromptSpec:
    now = datetime(2026, 2, 22, 16, 30, 0, tzinfo=timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_strict_test",
            question="Will Arsenal have 5 or more corners vs Tottenham on Feb 22?",
            event_definition=(
                "Arsenal is awarded 5 or more corner kicks during the "
                "Premier League match against Tottenham on February 22, 2026."
            ),
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(
                    rule_id="R1",
                    description="Resolve Yes if corners >= 5",
                    priority=100,
                ),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Arsenal",
            predicate="awarded 5 or more corners",
            threshold="5",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_strict_001",
                description="Arsenal's corner count from the match.",
                source_targets=[
                    SourceTarget(
                        source_id="web",
                        uri="https://www.fotmob.com/",
                        method="GET",
                        expected_content_type="html",
                        operation="search",
                        search_query="site:fotmob.com Arsenal vs Tottenham February 22, 2026 corners",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="fallback_chain",
                    min_sources=1,
                    max_sources=3,
                    quorum=1,
                ),
                expected_fields=["Corners"],
            ),
        ],
    )


def _create_spec_no_sources() -> PromptSpec:
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_no_src",
            question="Will something happen?",
            event_definition="Something happens.",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Check it", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Something",
            predicate="happens",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_no_src",
                description="Check something",
                source_targets=[],
                selection_policy=SelectionPolicy(
                    strategy="single_best", min_sources=1, max_sources=1, quorum=1,
                ),
                deferred_source_discovery=True,
            ),
        ],
    )


def _create_tool_plan(req_id: str = "req_strict_001") -> ToolPlan:
    return ToolPlan(
        plan_id="tp_strict_test",
        requirements=[req_id],
        sources=["web"],
    )


def _mock_response(
    outcome: str = "Yes",
    reason: str = "5 corners found.",
    source_uri: str = "https://www.fotmob.com/matches/arsenal-vs-tottenham",
    source_title: str = "FotMob Match Stats",
    url_context_urls: list[dict[str, str]] | None = None,
):
    """Build a mock Gemini response with specific source.

    *url_context_urls*: list of {"url": ..., "status": "success"|"failed"}
    to simulate UrlContext metadata in the response.
    """
    web = MagicMock()
    web.uri = source_uri
    web.title = source_title

    chunk = MagicMock()
    chunk.web = web

    grounding_metadata = MagicMock()
    grounding_metadata.web_search_queries = ["site:fotmob.com Arsenal corners"]
    grounding_metadata.grounding_chunks = [chunk]

    part = MagicMock()
    part.text = json.dumps({"outcome": outcome, "reason": reason})

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content
    candidate.grounding_metadata = grounding_metadata

    # UrlContext metadata
    if url_context_urls:
        url_metadata_list = []
        for entry in url_context_urls:
            um = MagicMock()
            um.retrieved_url = entry["url"]
            status_val = entry.get("status", "success")
            if status_val == "success":
                um.url_retrieval_status = "URL_RETRIEVAL_STATUS_SUCCESS"
            elif status_val == "failed":
                um.url_retrieval_status = "URL_RETRIEVAL_STATUS_FAILED"
            else:
                um.url_retrieval_status = status_val
            url_metadata_list.append(um)
        url_ctx_meta = MagicMock()
        url_ctx_meta.url_metadata = url_metadata_list
        candidate.url_context_metadata = url_ctx_meta
    else:
        candidate.url_context_metadata = None

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _mock_serper_response(urls: list[dict[str, str]]) -> MagicMock:
    """Build a mock HTTP response from Serper API.

    *urls*: list of {"link": ..., "title": ..., "snippet": ...}
    """
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {
        "organic": [
            {
                "link": u["link"],
                "title": u.get("title", ""),
                "snippet": u.get("snippet", ""),
                "position": i + 1,
            }
            for i, u in enumerate(urls)
        ],
    }
    return resp


# ---------------------------------------------------------------------------
# Tests: Strict prompt construction
# ---------------------------------------------------------------------------

class TestBuildStrictPrompt:
    """Test the strict prompt that limits search to required domains."""

    def test_includes_mandatory_domains(self):
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)
        prompt = _build_strict_prompt(spec, req, domains)
        assert "MANDATORY DATA SOURCES" in prompt
        assert "fotmob.com" in prompt
        assert "Do NOT use evidence from any other website" in prompt

    def test_includes_site_search_query(self):
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)
        prompt = _build_strict_prompt(spec, req, domains)
        assert "site:fotmob.com" in prompt

    def test_includes_discovered_urls(self):
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)
        discovered = [
            {"url": "https://www.fotmob.com/matches/arsenal-vs-tottenham/abc123", "title": "Match Page"},
            {"url": "https://www.fotmob.com/news/recap-abc", "title": "Match Recap"},
        ]
        prompt = _build_strict_prompt(spec, req, domains, discovered_urls=discovered)
        assert "DISCOVERED PAGES" in prompt
        assert "fotmob.com/matches/arsenal-vs-tottenham/abc123" in prompt
        assert "fotmob.com/news/recap-abc" in prompt
        assert "Match Page" in prompt

    def test_no_discovered_urls_section_when_empty(self):
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)
        prompt = _build_strict_prompt(spec, req, domains, discovered_urls=None)
        assert "DISCOVERED PAGES" not in prompt

    def test_includes_expected_fields(self):
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)
        prompt = _build_strict_prompt(spec, req, domains)
        assert "'Corners'" in prompt
        assert "Expected data fields" in prompt

    def test_includes_domain_hint_for_fbref(self):
        """When required domain is fbref.com, the prompt should include the domain hint."""
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        # Use fbref.com as the required domain
        fbref_domains = [{"domain": "fbref.com"}]
        prompt = _build_strict_prompt(spec, req, fbref_domains)
        assert "DOMAIN HINT (fbref.com)" in prompt
        assert "FBRef.com match report" in prompt
        assert "Player stats table" in prompt

    def test_no_domain_hint_for_unknown_domain(self):
        """Unknown domains should not have any domain hint in the prompt."""
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)  # fotmob.com
        prompt = _build_strict_prompt(spec, req, domains)
        assert "DOMAIN HINT" not in prompt


# ---------------------------------------------------------------------------
# Tests: Serper URL discovery
# ---------------------------------------------------------------------------

class TestSerperDiscovery:
    """Test the Serper pre-search URL discovery phase."""

    def _make_agent_with_mock_query(self, query="site:fotmob.com Arsenal vs Tottenham"):
        """Create agent with _generate_discovery_query mocked to return a fixed query."""
        agent = CollectorSourcePinned()
        agent._generate_discovery_query = MagicMock(return_value=query)
        return agent

    def test_discovers_urls_from_serper(self):
        agent = self._make_agent_with_mock_query()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)

        serper_resp = _mock_serper_response([
            {"link": "https://www.fotmob.com/matches/arsenal-vs-tottenham/abc123", "title": "Match Page"},
            {"link": "https://www.fotmob.com/news/recap-abc", "title": "Match Recap"},
            {"link": "https://www.fotmob.com/embed/news/ratings", "title": "Player Ratings"},
            {"link": "https://www.fotmob.com/teams/1234/arsenal", "title": "Arsenal Team"},
        ])

        ctx = AgentContext.create_minimal()
        ctx.http = MagicMock()
        ctx.http.post.return_value = serper_resp
        client = MagicMock()

        with patch.dict("os.environ", {"SERPER_API_KEY": "fake-key"}):
            urls = agent._serper_discover_urls(ctx, client, spec, req, domains)

        assert len(urls) == 3  # max 3
        assert urls[0]["url"] == "https://www.fotmob.com/matches/arsenal-vs-tottenham/abc123"
        assert urls[0]["title"] == "Match Page"

    def test_skips_non_domain_urls(self):
        agent = self._make_agent_with_mock_query()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)

        serper_resp = _mock_serper_response([
            {"link": "https://flashscore.com/match/123", "title": "Flashscore"},
            {"link": "https://www.fotmob.com/matches/abc123", "title": "FotMob"},
        ])

        ctx = AgentContext.create_minimal()
        ctx.http = MagicMock()
        ctx.http.post.return_value = serper_resp
        client = MagicMock()

        with patch.dict("os.environ", {"SERPER_API_KEY": "fake-key"}):
            urls = agent._serper_discover_urls(ctx, client, spec, req, domains)

        assert len(urls) == 1
        assert "fotmob.com" in urls[0]["url"]

    def test_returns_empty_without_api_key(self):
        agent = self._make_agent_with_mock_query()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)

        ctx = AgentContext.create_minimal()
        client = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            urls = agent._serper_discover_urls(ctx, client, spec, req, domains)

        assert urls == []

    def test_returns_empty_without_http_client(self):
        agent = self._make_agent_with_mock_query()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)

        ctx = AgentContext.create_minimal()
        ctx.http = None
        client = MagicMock()
        with patch.dict("os.environ", {"SERPER_API_KEY": "fake-key"}):
            urls = agent._serper_discover_urls(ctx, client, spec, req, domains)

        assert urls == []

    def test_handles_serper_error_gracefully(self):
        agent = self._make_agent_with_mock_query()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)

        ctx = AgentContext.create_minimal()
        ctx.http = MagicMock()
        ctx.http.post.side_effect = RuntimeError("Connection failed")
        client = MagicMock()

        with patch.dict("os.environ", {"SERPER_API_KEY": "fake-key"}):
            urls = agent._serper_discover_urls(ctx, client, spec, req, domains)

        assert urls == []

    def test_handles_serper_http_error(self):
        agent = self._make_agent_with_mock_query()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]
        from agents.collector.gemini_grounded_agent import _extract_required_domains
        domains = _extract_required_domains(req)

        resp = MagicMock()
        resp.ok = False
        resp.status_code = 403

        ctx = AgentContext.create_minimal()
        ctx.http = MagicMock()
        ctx.http.post.return_value = resp
        client = MagicMock()

        with patch.dict("os.environ", {"SERPER_API_KEY": "fake-key"}):
            urls = agent._serper_discover_urls(ctx, client, spec, req, domains)

        assert urls == []

    def test_generate_discovery_query_produces_site_scoped_query(self):
        """_generate_discovery_query should produce a site:-scoped query via LLM."""
        agent = CollectorSourcePinned()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]

        # Mock the Gemini client to return a well-formed query
        mock_part = MagicMock()
        mock_part.text = "site:fotmob.com Arsenal vs Tottenham February 22 2026"
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_resp = MagicMock()
        mock_resp.candidates = [mock_candidate]

        client = MagicMock()
        client.models.generate_content.return_value = mock_resp

        query = agent._generate_discovery_query(client, spec, req, "fotmob.com")
        assert "site:fotmob.com" in query
        assert "Arsenal" in query
        assert "Tottenham" in query

    def test_generate_discovery_query_fallback_on_error(self):
        """On LLM error, should fall back to entity + question."""
        agent = CollectorSourcePinned()
        spec = _create_spec_with_sources()
        req = spec.data_requirements[0]

        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("LLM down")

        query = agent._generate_discovery_query(client, spec, req, "fotmob.com")
        assert "site:fotmob.com" in query
        # Fallback uses entity + question
        assert "Arsenal" in query


# ---------------------------------------------------------------------------
# Tests: UrlContext metadata extraction
# ---------------------------------------------------------------------------

class TestExtractUrlContextMetadata:
    """Test extraction of UrlContext metadata from Gemini response."""

    def test_extracts_successful_urls(self):
        resp = _mock_response(
            url_context_urls=[
                {"url": "https://www.fotmob.com/matches/abc123", "status": "success"},
                {"url": "https://www.fotmob.com/news/recap", "status": "success"},
            ],
        )
        meta = CollectorSourcePinned._extract_url_context_metadata(resp)
        assert len(meta) == 2
        assert meta[0]["url"] == "https://www.fotmob.com/matches/abc123"
        assert meta[0]["status"] == "success"

    def test_extracts_failed_urls(self):
        resp = _mock_response(
            url_context_urls=[
                {"url": "https://www.fotmob.com/matches/abc123", "status": "failed"},
            ],
        )
        meta = CollectorSourcePinned._extract_url_context_metadata(resp)
        assert len(meta) == 1
        assert meta[0]["status"] == "failed"

    def test_empty_when_no_url_context(self):
        resp = _mock_response()  # no url_context_urls
        meta = CollectorSourcePinned._extract_url_context_metadata(resp)
        assert meta == []


# ---------------------------------------------------------------------------
# Tests: Collector behavior (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestCollectorSourcePinned:
    """Integration tests for the strict collector."""

    def test_fails_without_source_targets(self):
        """No source_targets → immediate failure."""
        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()):
            ctx = AgentContext.create_minimal()
            result = agent.run(
                ctx,
                _create_spec_no_sources(),
                _create_tool_plan("req_no_src"),
            )

        bundle, _ = result.output
        assert len(bundle.items) == 1
        assert not bundle.items[0].success
        assert "source_targets" in bundle.items[0].error

    def test_success_with_required_domain_found(self):
        """When Gemini returns a source from the required domain → success, tier 1."""
        resp = _mock_response(
            source_uri="https://www.fotmob.com/matches/arsenal-tottenham",
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        assert result.success
        item = result.output[0].items[0]
        assert item.parsed_value == "Yes"
        assert item.provenance.tier == 1
        assert item.extracted_fields["data_source_covered"] is True
        assert item.extracted_fields["attempts_made"] == 1
        sources = item.extracted_fields["evidence_sources"]
        assert any(s["is_required_data_source"] for s in sources)

    def test_retries_until_domain_found(self):
        """When first attempt misses the domain, should retry."""
        miss_resp = _mock_response(
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore",
        )
        hit_resp = _mock_response(
            source_uri="https://www.fotmob.com/matches/arsenal-tottenham",
            source_title="FotMob",
        )

        agent = CollectorSourcePinned()
        call_count = 0

        def mock_call(client, prompt, *, discovered_urls=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return miss_resp
            return hit_resp

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        assert result.success
        assert call_count == 2
        item = result.output[0].items[0]
        assert item.provenance.tier == 1
        assert item.extracted_fields["data_source_covered"] is True
        assert item.extracted_fields["attempts_made"] == 2

    def test_max_attempts_exhausted(self):
        """When all attempts miss the domain, uses best available result."""
        miss_resp = _mock_response(
            outcome="Yes",
            reason="5 corners from flashscore",
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore",
        )

        agent = CollectorSourcePinned(max_attempts=2)
        call_count = 0

        def mock_call(client, prompt, *, discovered_urls=None):
            nonlocal call_count
            call_count += 1
            return miss_resp

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        assert result.success  # still succeeds with outcome
        assert call_count == 2  # tried max_attempts times
        item = result.output[0].items[0]
        assert item.provenance.tier == 2  # tier 2 because domain not found
        assert item.extracted_fields["data_source_covered"] is False
        assert item.extracted_fields["confidence_score"] == 0.5  # reduced confidence

    def test_confidence_higher_when_domain_found(self):
        """Confidence should be 0.9 when domain found, 0.5 when not."""
        hit_resp = _mock_response(
            source_uri="https://www.fotmob.com/matches/123",
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", return_value=hit_resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        item = result.output[0].items[0]
        assert item.extracted_fields["confidence_score"] == 0.9

    def test_non_required_sources_marked_tier3(self):
        """Non-required-domain sources should get credibility_tier 3 (not authoritative)."""
        miss_resp = _mock_response(
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore",
        )

        agent = CollectorSourcePinned(max_attempts=1)
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", return_value=miss_resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        item = result.output[0].items[0]
        sources = item.extracted_fields["evidence_sources"]
        assert len(sources) == 1
        assert sources[0]["credibility_tier"] == 3  # non-authoritative
        assert sources[0]["is_required_data_source"] is False

    def test_api_error_returns_failure(self):
        """API error should return failure evidence."""
        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", side_effect=RuntimeError("API down")):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        bundle, exec_log = result.output
        assert not bundle.items[0].success
        assert "API down" in bundle.items[0].error

    def test_no_api_key_returns_failure(self):
        """Missing API key should fail."""
        agent = CollectorSourcePinned()
        ctx = AgentContext.create_minimal()
        with patch.dict("os.environ", {}, clear=True):
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())
        assert not result.success
        assert "API key" in result.error

    def test_strict_mode_metadata(self):
        """Metadata should indicate strict mode."""
        resp = _mock_response(
            source_uri="https://www.fotmob.com/matches/123",
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=[]), \
             patch.object(agent, "_call_gemini_strict", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        item = result.output[0].items[0]
        assert item.extracted_fields["strict_mode"] is True

    def test_serper_urls_passed_to_gemini(self):
        """Discovered URLs should be passed to _call_gemini_strict."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/abc123", "title": "Match", "snippet": "..."},
        ]
        resp = _mock_response(
            source_uri="https://www.fotmob.com/matches/abc123",
            url_context_urls=[
                {"url": "https://www.fotmob.com/matches/abc123", "status": "success"},
            ],
        )

        agent = CollectorSourcePinned()
        gemini_calls = []

        def mock_call(client, prompt, *, discovered_urls=None):
            gemini_calls.append({"prompt": prompt, "discovered_urls": discovered_urls})
            return resp

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch.object(agent, "_call_gemini_strict", side_effect=mock_call):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        assert result.success
        assert len(gemini_calls) == 1
        # Discovered URLs should be passed to Gemini
        assert gemini_calls[0]["discovered_urls"] == discovered
        # Prompt should mention the discovered URL
        assert "fotmob.com/matches/abc123" in gemini_calls[0]["prompt"]

    def test_url_context_metadata_in_output(self):
        """UrlContext metadata should appear in the evidence output."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/abc123", "title": "Match", "snippet": "..."},
        ]
        resp = _mock_response(
            source_uri="https://www.fotmob.com/matches/abc123",
            url_context_urls=[
                {"url": "https://www.fotmob.com/matches/abc123", "status": "success"},
            ],
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch.object(agent, "_call_gemini_strict", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        item = result.output[0].items[0]
        assert "url_context_statuses" in item.extracted_fields
        statuses = item.extracted_fields["url_context_statuses"]
        assert len(statuses) == 1
        assert statuses[0]["status"] == "success"
        assert "serper_discovered_urls" in item.extracted_fields
        assert item.extracted_fields["serper_discovered_urls"] == ["https://www.fotmob.com/matches/abc123"]

    def test_url_context_sources_count_as_domain_coverage(self):
        """UrlContext success on a required domain should count as coverage,
        even if grounding_chunks has no matching sources."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/abc123", "title": "Match", "snippet": "..."},
        ]
        # Grounding has no fotmob source, but UrlContext fetched it successfully
        resp = _mock_response(
            source_uri="https://flashscore.com/match/123",
            source_title="Flashscore",
            url_context_urls=[
                {"url": "https://www.fotmob.com/matches/abc123", "status": "success"},
            ],
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch.object(agent, "_call_gemini_strict", return_value=resp):
            ctx = AgentContext.create_minimal()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        item = result.output[0].items[0]
        assert item.extracted_fields["data_source_covered"] is True
        assert item.provenance.tier == 1
        assert item.extracted_fields["confidence_score"] == 0.9


class TestFilterToRequiredDomains:
    """Test the domain filtering helper."""

    def test_filters_correctly(self):
        sources = [
            {"uri": "https://www.fotmob.com/matches/123", "title": "FotMob"},
            {"uri": "https://flashscore.com/match/456", "title": "Flashscore"},
            {"uri": "https://api.fotmob.com/stats", "title": "FotMob API"},
        ]
        required = [{"domain": "fotmob.com"}]
        filtered = CollectorSourcePinned._filter_to_required_domains(
            sources, required,
        )
        assert len(filtered) == 2
        assert all("fotmob" in s["uri"] for s in filtered)

    def test_empty_when_no_match(self):
        sources = [
            {"uri": "https://flashscore.com/match/456", "title": "Flashscore"},
        ]
        required = [{"domain": "fotmob.com"}]
        filtered = CollectorSourcePinned._filter_to_required_domains(
            sources, required,
        )
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# FotMob direct extraction helpers & tests
# ---------------------------------------------------------------------------

def _make_fotmob_match_data() -> FotMobMatchData:
    """Build a FotMobMatchData for the Real Madrid shots market."""
    return FotMobMatchData(
        match_id="4837357",
        home_team="Osasuna",
        away_team="Real Madrid",
        url="https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
        home_score=0,
        away_score=4,
        stats_by_category={
            "shots": [
                FotMobStat(title="Total shots", key="total_shots",
                           home_value=13, away_value=15, category="shots"),
                FotMobStat(title="Shots outside box", key="shots_outside_box",
                           home_value=2, away_value=8, category="shots"),
            ],
        },
        shots=[
            FotMobShot(event_type="Goal", team_id=8633, player_name="Vinicius Jr",
                       situation="OpenPlay", shot_type="LeftFoot", minute=12,
                       expected_goals=0.45, is_home=False, on_target=True),
        ],
        events=[
            FotMobEvent(type="Goal", time=12, overload_time=None, is_home=False,
                        player_name="Vinicius Jr", assist_player="Bellingham"),
        ],
        event_types=["Goal"],
    )


def _mock_llm_response(outcome: str = "Yes", reason: str = "8 shots outside box found"):
    """Build a mock Gemini LLM response for Phase 1.5 (no tools)."""
    part = MagicMock()
    part.text = json.dumps({"outcome": outcome, "reason": reason})
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    candidate.url_context_metadata = None
    candidate.grounding_metadata = None
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _create_shots_spec() -> PromptSpec:
    """Market: Will Real Madrid make 5+ shots outside box vs Osasuna."""
    now = datetime(2026, 2, 22, 16, 30, 0, tzinfo=timezone.utc)
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
                        uri="https://www.fotmob.com/",
                        method="GET",
                        expected_content_type="html",
                        operation="search",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="fallback_chain",
                    min_sources=1,
                    max_sources=3,
                    quorum=1,
                ),
                expected_fields=["Shots outside box"],
            ),
        ],
    )


def _mock_fotmob_extractor(match_data=None, error=None):
    """Build a mock SiteExtractor for fotmob that returns given data or raises."""
    from agents.collector.extractors.base import ExtractionError
    from agents.collector.fotmob import summarize_for_llm as fotmob_summarize_for_llm

    ext = MagicMock()
    ext.source_id = "fotmob"
    ext.can_handle.return_value = True

    if error is not None:
        ext.extract_and_summarize.side_effect = ExtractionError(str(error))
    elif match_data is not None:
        summary = fotmob_summarize_for_llm(match_data)
        metadata = {
            "fotmob_url": match_data.url,
            "fotmob_match_id": match_data.match_id,
            "fotmob_home_team": match_data.home_team,
            "fotmob_away_team": match_data.away_team,
            "fotmob_home_score": match_data.home_score,
            "fotmob_away_score": match_data.away_score,
            "fotmob_stats_categories": list(match_data.stats_by_category.keys()),
            "fotmob_shots_count": len(match_data.shots),
            "fotmob_events_count": len(match_data.events),
        }
        ext.extract_and_summarize.return_value = (summary, metadata)

    return ext


class TestFotMobDirectExtraction:
    """Test the fotmob direct extraction phase (Phase 1.5) with LLM resolution."""

    def test_direct_extraction_resolves_yes(self):
        """When fotmob data + LLM resolves Yes, should return direct extraction result."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
             "title": "Match", "snippet": "..."},
        ]

        agent = CollectorSourcePinned()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response(
            outcome="Yes", reason="Real Madrid had 8 shots outside box, threshold is 5."
        )

        mock_ext = _mock_fotmob_extractor(_make_fotmob_match_data())
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor",
                   return_value=mock_ext):
            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        assert item.parsed_value == "Yes"
        assert item.extracted_fields.get("direct_extraction") is True
        assert item.extracted_fields.get("resolution_method") == "llm_with_structured_data"
        assert item.extracted_fields.get("fotmob_home_team") == "Osasuna"
        assert item.extracted_fields.get("fotmob_away_team") == "Real Madrid"
        assert item.provenance.tier == 1
        assert item.extracted_fields.get("confidence_score") == 0.90

    def test_direct_extraction_resolves_no(self):
        """When LLM resolves No, should return No."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
             "title": "Match", "snippet": "..."},
        ]

        agent = CollectorSourcePinned()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response(
            outcome="No", reason="Real Madrid had only 3 shots outside box, below threshold of 5."
        )

        mock_ext = _mock_fotmob_extractor(_make_fotmob_match_data())
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor",
                   return_value=mock_ext):
            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        assert item.parsed_value == "No"
        assert item.extracted_fields.get("direct_extraction") is True

    def test_falls_back_to_gemini_on_extraction_failure(self):
        """When fotmob extraction fails, should fall back to Gemini Phase 2."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
             "title": "Match", "snippet": "..."},
        ]
        gemini_resp = _mock_response(
            outcome="Yes", reason="8 shots found",
            source_uri="https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
        )

        agent = CollectorSourcePinned()

        mock_ext = _mock_fotmob_extractor(error="Page structure changed")
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor",
                   return_value=mock_ext), \
             patch.object(agent, "_call_gemini_strict", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        # Falls back to Gemini — no direct_extraction flag
        assert item.extracted_fields.get("direct_extraction") is not True

    def test_skips_non_match_urls(self):
        """Non-match fotmob URLs (e.g. /teams/) should not trigger extraction."""
        discovered = [
            {"url": "https://www.fotmob.com/teams/8633/overview/real-madrid",
             "title": "Team Page", "snippet": "..."},
        ]
        gemini_resp = _mock_response(
            outcome="Yes", reason="Found",
            source_uri="https://www.fotmob.com/teams/8633",
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch.object(agent, "_call_gemini_strict", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        # Should have used Gemini, not direct extraction
        item = result.output[0].items[0]
        assert item.extracted_fields.get("direct_extraction") is not True

    def test_corner_goal_market(self):
        """Shotmap-based market (corner goal) should resolve via LLM."""
        corner_data = FotMobMatchData(
            match_id="999",
            home_team="Arsenal",
            away_team="Tottenham",
            url="https://www.fotmob.com/matches/arsenal-vs-tottenham/abc",
            home_score=2,
            away_score=0,
            shots=[
                FotMobShot(event_type="Goal", team_id=1, player_name="Saka",
                           situation="FromCorner", shot_type="Head", minute=15,
                           expected_goals=0.35, is_home=True, on_target=True),
            ],
            events=[
                FotMobEvent(type="Goal", time=15, overload_time=None,
                            is_home=True, player_name="Saka",
                            assist_player="Odegaard"),
            ],
            event_types=["Goal"],
        )

        discovered = [
            {"url": "https://www.fotmob.com/matches/arsenal-vs-tottenham/abc",
             "title": "Match", "snippet": "..."},
        ]

        agent = CollectorSourcePinned()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response(
            outcome="Yes",
            reason="Arsenal scored a goal from corner (Saka, 15') per the shotmap data.",
        )

        mock_ext = _mock_fotmob_extractor(corner_data)
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor",
                   return_value=mock_ext):
            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            result = agent.run(ctx, _create_spec_with_sources(), _create_tool_plan())

        assert result.success
        item = result.output[0].items[0]
        assert item.parsed_value == "Yes"
        assert item.extracted_fields.get("direct_extraction") is True
        assert item.extracted_fields.get("resolution_method") == "llm_with_structured_data"
        assert item.extracted_fields.get("fotmob_shots_count") == 1

    def test_llm_failure_returns_none_falls_to_phase2(self):
        """LLM timeout/failure in Phase 1.5 should fall back to Phase 2."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
             "title": "Match", "snippet": "..."},
        ]
        gemini_resp = _mock_response(
            outcome="Yes", reason="8 shots found",
            source_uri="https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
        )

        agent = CollectorSourcePinned()
        mock_client = MagicMock()
        # Phase 1.5 LLM call raises timeout
        mock_client.models.generate_content.side_effect = TimeoutError("LLM timeout")

        mock_ext = _mock_fotmob_extractor(_make_fotmob_match_data())
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor",
                   return_value=mock_ext), \
             patch.object(agent, "_call_gemini_strict", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        # Should have fallen back to Phase 2
        assert item.extracted_fields.get("direct_extraction") is not True


# ---------------------------------------------------------------------------
# Tests: Generic direct extraction dispatch (extractor registry)
# ---------------------------------------------------------------------------

class TestGenericDirectExtraction:
    """Test the generic _try_direct_extraction that dispatches to any extractor."""

    def test_dispatches_to_fotmob_extractor(self):
        """FotMob URL should be handled by the fotmob extractor."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
             "title": "Match", "snippet": "..."},
        ]

        agent = CollectorSourcePinned()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response(
            outcome="Yes", reason="8 shots outside box."
        )

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=mock_client), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor") as mock_find:
            mock_ext = MagicMock()
            mock_ext.source_id = "fotmob"
            mock_ext.can_handle.return_value = True
            mock_ext.extract_and_summarize.return_value = (
                "Match summary text here",
                {"fotmob_match_id": "123", "fotmob_home_team": "Osasuna"},
            )
            mock_find.return_value = mock_ext

            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        assert item.extracted_fields.get("direct_extraction") is True
        assert item.extracted_fields.get("resolution_method") == "llm_with_structured_data"

    def test_falls_through_when_no_extractor_matches(self):
        """Unknown URLs should fall through to Phase 2."""
        discovered = [
            {"url": "https://flashscore.com/match/123", "title": "Match", "snippet": "..."},
        ]
        gemini_resp = _mock_response(
            outcome="Yes", reason="Found",
            source_uri="https://flashscore.com/match/123",
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor", return_value=None), \
             patch.object(agent, "_call_gemini_strict", return_value=gemini_resp):
            ctx = AgentContext.create_minimal()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        assert item.extracted_fields.get("direct_extraction") is not True

    def test_extraction_error_falls_through(self):
        """ExtractionError should fall through to Phase 2."""
        discovered = [
            {"url": "https://www.fotmob.com/matches/foo/bar", "title": "Match", "snippet": "..."},
        ]
        gemini_resp = _mock_response(
            outcome="Yes", reason="Found via search",
            source_uri="https://www.fotmob.com/matches/foo/bar",
        )

        agent = CollectorSourcePinned()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}), \
             patch.object(agent, "_get_client", return_value=MagicMock()), \
             patch.object(agent, "_serper_discover_urls", return_value=discovered), \
             patch("agents.collector.source_pinned_agent.find_extractor") as mock_find, \
             patch.object(agent, "_call_gemini_strict", return_value=gemini_resp):
            from agents.collector.extractors.base import ExtractionError
            mock_ext = MagicMock()
            mock_ext.source_id = "fotmob"
            mock_ext.extract_and_summarize.side_effect = ExtractionError("Cloudflare block")
            mock_find.return_value = mock_ext

            ctx = AgentContext.create_minimal()
            ctx.http = MagicMock()
            spec = _create_shots_spec()
            result = agent.run(ctx, spec, ToolPlan(
                plan_id="tp_shots", requirements=["req_shots_001"], sources=["web"],
            ))

        assert result.success
        item = result.output[0].items[0]
        assert item.extracted_fields.get("direct_extraction") is not True
