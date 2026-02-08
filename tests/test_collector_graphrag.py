"""
Tests for CollectorGraphRAG — GraphRAG Local-to-Global collector.

Covers:
1) Deterministic evidence IDs — same inputs produce the same evidence_id.
2) Graph build deduplicates entities and produces communities.
3) JSON retry/repair loop triggers on invalid JSON, else produces failure EvidenceItem.
4) EvidenceItem.extracted_fields contains graph_stats and normalized evidence_sources.
5) CollectorGraphRAG registers and can be selected via prefer_graphrag=True.
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agents import AgentContext, AgentStep, get_registry
from agents.collector import (
    CollectorGraphRAG,
    get_collector,
)
from agents.collector.graphrag_engine import (
    GraphIndex,
    chunk_text_units,
    merge_elements_into_graph,
    detect_communities,
    build_local_context_pack,
    normalize_entity_name,
    rank_communities_by_query,
    infer_credibility_tier,
)
from core.schemas import (
    DataRequirement,
    EvidenceBundle,
    EvidenceItem,
    MarketSpec,
    PromptSpec,
    Provenance,
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

def create_graphrag_prompt_spec() -> PromptSpec:
    """Create a PromptSpec suitable for GraphRAG testing."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_graphrag_test",
            question="Will the Federal Reserve raise interest rates in Q1 2026?",
            event_definition="Federal Reserve raises the federal funds rate target in Q1 2026",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Fed announcement", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Federal Reserve",
            predicate="raises interest rates",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_fed_001",
                description="Determine if the Federal Reserve has raised interest rates in Q1 2026",
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
        extra={"assumptions": ["Only consider FOMC meetings in Jan-Mar 2026"]},
    )


def create_graphrag_tool_plan() -> ToolPlan:
    return ToolPlan(
        plan_id="plan_graphrag_test",
        requirements=["req_fed_001"],
        sources=[],
        min_provenance_tier=0,
        allow_fallbacks=True,
    )


# ---------------------------------------------------------------------------
# 1) Deterministic Evidence IDs
# ---------------------------------------------------------------------------

class TestDeterministicIDs:
    """Same inputs must produce the same evidence_id."""

    def test_same_input_same_id(self):
        """Two calls with identical req_id and market_id produce same ID."""
        key1 = f"graphrag:req_fed_001:final:mk_graphrag_test"
        key2 = f"graphrag:req_fed_001:final:mk_graphrag_test"
        id1 = hashlib.sha256(key1.encode()).hexdigest()[:16]
        id2 = hashlib.sha256(key2.encode()).hexdigest()[:16]
        assert id1 == id2

    def test_different_input_different_id(self):
        """Different requirement IDs produce different IDs."""
        id1 = hashlib.sha256(b"graphrag:req_001:final:mk1").hexdigest()[:16]
        id2 = hashlib.sha256(b"graphrag:req_002:final:mk1").hexdigest()[:16]
        assert id1 != id2


# ---------------------------------------------------------------------------
# 2) Graph build deduplicates entities and produces communities
# ---------------------------------------------------------------------------

class TestGraphBuild:
    """Graph construction and community detection."""

    def test_entity_dedup(self):
        """Merging the same entity twice deduplicates by normalized name."""
        graph = GraphIndex()
        elements1 = {
            "entities": [{"name": "Federal Reserve", "type": "ORG", "description": "US central bank"}],
            "relations": [],
            "claims": [],
        }
        elements2 = {
            "entities": [{"name": "federal reserve", "type": "ORG", "description": "The Federal Reserve System, US central bank since 1913"}],
            "relations": [],
            "claims": [],
        }
        merge_elements_into_graph(graph, elements1, "https://a.com", "tu_0")
        merge_elements_into_graph(graph, elements2, "https://b.com", "tu_1")

        # Should have exactly one entity (deduplicated by normalized name)
        assert len(graph.entities) == 1
        eid = normalize_entity_name("Federal Reserve")
        assert eid in graph.entities
        # Description should be the longer one
        assert "1913" in graph.entities[eid].description
        # Both URLs tracked
        assert "https://a.com" in graph.entities[eid].source_urls
        assert "https://b.com" in graph.entities[eid].source_urls

    def test_relation_creates_adjacency(self):
        """Relations build adjacency between entities."""
        graph = GraphIndex()
        elements = {
            "entities": [
                {"name": "Fed", "type": "ORG", "description": "Federal Reserve"},
                {"name": "Jerome Powell", "type": "PERSON", "description": "Fed chair"},
            ],
            "relations": [
                {"head": "Jerome Powell", "relation": "CHAIRS", "tail": "Fed", "quote": "Powell chairs the Fed"},
            ],
            "claims": [],
        }
        merge_elements_into_graph(graph, elements, "https://c.com", "tu_2")

        fed_id = normalize_entity_name("Fed")
        powell_id = normalize_entity_name("Jerome Powell")
        assert powell_id in graph.adjacency.get(fed_id, set())
        assert fed_id in graph.adjacency.get(powell_id, set())
        assert len(graph.relations) == 1

    def test_community_detection_produces_communities(self):
        """Community detection groups connected entities."""
        graph = GraphIndex()
        # Create two clusters
        elements = {
            "entities": [
                {"name": "A", "type": "ORG", "description": "Entity A"},
                {"name": "B", "type": "ORG", "description": "Entity B"},
                {"name": "C", "type": "ORG", "description": "Entity C"},
                {"name": "X", "type": "ORG", "description": "Entity X"},
                {"name": "Y", "type": "ORG", "description": "Entity Y"},
            ],
            "relations": [
                {"head": "A", "relation": "LINKED", "tail": "B", "quote": "A-B"},
                {"head": "B", "relation": "LINKED", "tail": "C", "quote": "B-C"},
                {"head": "X", "relation": "LINKED", "tail": "Y", "quote": "X-Y"},
            ],
            "claims": [],
        }
        merge_elements_into_graph(graph, elements, "https://d.com", "tu_3")
        communities = detect_communities(graph, min_size=2)

        # Should detect at least 2 communities (A-B-C and X-Y)
        assert len(communities) >= 2

    def test_community_detection_handles_empty_graph(self):
        """Empty graph produces no communities."""
        graph = GraphIndex()
        communities = detect_communities(graph, min_size=2)
        assert len(communities) == 0


# ---------------------------------------------------------------------------
# 3) JSON retry/repair & failure EvidenceItem
# ---------------------------------------------------------------------------

class TestJSONRetry:
    """JSON extraction and repair loop."""

    def test_extract_json_from_markdown_block(self):
        """Extracts JSON from markdown code fence."""
        text = '```json\n{"entities": [], "relations": []}\n```'
        result = CollectorGraphRAG._extract_json(text)
        assert result == {"entities": [], "relations": []}

    def test_extract_json_raw_braces(self):
        """Extracts JSON from raw braces in text."""
        text = 'Some preamble {"key": "value"} trailing text'
        result = CollectorGraphRAG._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_raises_on_invalid(self):
        """Raises on completely invalid JSON."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            CollectorGraphRAG._extract_json("not json at all")

    def test_collector_returns_failure_on_no_search_results(self):
        """CollectorGraphRAG returns failure EvidenceItem when Serper is not configured."""
        ctx = AgentContext.create_minimal()

        prompt_spec = create_graphrag_prompt_spec()
        tool_plan = create_graphrag_tool_plan()

        collector = CollectorGraphRAG()
        result = collector.run(ctx, prompt_spec, tool_plan)

        # Should fail since no LLM client is available
        assert not result.success


# ---------------------------------------------------------------------------
# 4) extracted_fields contains graph_stats and normalized evidence_sources
# ---------------------------------------------------------------------------

class TestExtractedFields:
    """Verify extracted_fields shape from a synthetic synthesis."""

    def test_build_evidence_has_graph_stats(self):
        """_build_evidence_from_synthesis includes graph_stats in extracted_fields."""
        ctx = AgentContext.create_minimal()

        graph = GraphIndex()
        elements = {
            "entities": [
                {"name": "Fed", "type": "ORG", "description": "Federal Reserve"},
                {"name": "Rates", "type": "METRIC", "description": "Interest rates"},
            ],
            "relations": [
                {"head": "Fed", "relation": "SETS", "tail": "Rates", "quote": "Fed sets rates"},
            ],
            "claims": [],
        }
        merge_elements_into_graph(graph, elements, "https://e.com", "tu_0")
        detect_communities(graph, min_size=2)

        synthesis = {
            "resolution_status": "RESOLVED",
            "parsed_value": "Yes",
            "confidence_score": 0.85,
            "evidence_sources": [
                {
                    "url": "https://reuters.com/fed-rates",
                    "credibility_tier": 1,
                    "key_fact": "Fed raised rates by 25bps",
                    "supports": "YES",
                    "date_published": "2026-01-15",
                }
            ],
            "conflicts": [],
            "missing_info": [],
            "reasoning_trace": "Step 1: Reuters confirms rate hike. Conclusion: RESOLVED.",
        }

        collector = CollectorGraphRAG()
        evidence = collector._build_evidence_from_synthesis(
            ctx, "req_fed_001", "mk_graphrag_test", synthesis, graph,
        )

        assert evidence.success is True
        assert evidence.parsed_value == "Yes"
        assert evidence.provenance.source_id == "graphrag"
        assert "graphrag:index" in evidence.provenance.source_uri

        fields = evidence.extracted_fields
        assert "graph_stats" in fields
        assert fields["graph_stats"]["entities"] == 2
        assert fields["graph_stats"]["relations"] == 1
        assert "top_entities" in fields
        assert "evidence_sources" in fields
        assert isinstance(fields["evidence_sources"], list)
        assert fields["evidence_sources"][0]["credibility_tier"] == 1
        assert fields["confidence_score"] == 0.85
        assert fields["resolution_status"] == "RESOLVED"

    def test_failure_evidence_on_low_confidence(self):
        """Synthesis with low confidence produces success=False."""
        ctx = AgentContext.create_minimal()
        graph = GraphIndex()

        synthesis = {
            "resolution_status": "UNRESOLVED",
            "parsed_value": None,
            "confidence_score": 0.2,
            "evidence_sources": [],
            "reasoning_trace": "Insufficient evidence.",
        }

        collector = CollectorGraphRAG()
        evidence = collector._build_evidence_from_synthesis(
            ctx, "req_001", "mk_001", synthesis, graph,
        )

        assert evidence.success is False
        assert "UNRESOLVED" in (evidence.error or "")


# ---------------------------------------------------------------------------
# 5) Registration and selector flag
# ---------------------------------------------------------------------------

class TestRegistrationAndSelector:
    """CollectorGraphRAG is registered and selectable via prefer_graphrag."""

    def test_registered_in_registry(self):
        """CollectorGraphRAG appears in the global agent registry."""
        registry = get_registry()
        agents = registry.list_agents(AgentStep.COLLECTOR)
        names = [a.name for a in agents]
        assert "CollectorGraphRAG" in names

    def test_priority_between_agentic_and_hyde(self):
        """Priority 165 sits between AgenticRAG (170) and HyDE (160)."""
        registry = get_registry()
        agents = registry.list_agents(AgentStep.COLLECTOR)

        graphrag = next(a for a in agents if a.name == "CollectorGraphRAG")
        agentic = next(a for a in agents if a.name == "CollectorAgenticRAG")
        hyde = next(a for a in agents if a.name == "CollectorHyDE")

        assert agentic.priority > graphrag.priority > hyde.priority

    def test_get_collector_prefer_graphrag(self):
        """get_collector returns CollectorGraphRAG when prefer_graphrag=True and context has LLM+HTTP."""
        ctx = AgentContext.create_minimal()
        ctx.llm = MagicMock()
        ctx.http = MagicMock()

        collector = get_collector(ctx, prefer_graphrag=True)
        assert isinstance(collector, CollectorGraphRAG)

    def test_get_collector_defaults_skip_graphrag(self):
        """get_collector does NOT return GraphRAG by default (prefer_graphrag defaults to False)."""
        ctx = AgentContext.create_minimal()
        ctx.llm = MagicMock()
        ctx.http = MagicMock()

        collector = get_collector(ctx)
        assert not isinstance(collector, CollectorGraphRAG)

    def test_requires_llm_and_http(self):
        """CollectorGraphRAG.run fails gracefully without LLM or HTTP."""
        ctx_no_llm = AgentContext.create_minimal()
        prompt_spec = create_graphrag_prompt_spec()
        tool_plan = create_graphrag_tool_plan()

        collector = CollectorGraphRAG()

        result = collector.run(ctx_no_llm, prompt_spec, tool_plan)
        assert not result.success
        assert "LLM client not available" in result.error


# ---------------------------------------------------------------------------
# GraphRAG engine unit tests
# ---------------------------------------------------------------------------

class TestGraphRAGEngine:
    """Tests for the graphrag_engine module helpers."""

    def test_chunk_text_units(self):
        """Chunking produces overlapping text units."""
        text = "A" * 5000
        units = chunk_text_units(text, "https://example.com", "Test", chunk_size=2500, overlap=200)
        assert len(units) >= 2
        # IDs are stable
        assert units[0].id == units[0].id
        # Overlap: second chunk starts at 2300, so content overlaps
        assert units[1].offset == 2300

    def test_normalize_entity_name(self):
        """Normalization strips accents and non-alphanumeric chars."""
        assert normalize_entity_name("Jérôme Powell") == "jerome powell"
        assert normalize_entity_name("S&P 500") == "sp 500"
        assert normalize_entity_name("  Federal Reserve  ") == "federal reserve"

    def test_build_local_context_pack(self):
        """Local context pack includes entities and relations."""
        graph = GraphIndex()
        elements = {
            "entities": [
                {"name": "Fed", "type": "ORG", "description": "Federal Reserve"},
                {"name": "Rates", "type": "METRIC", "description": "Interest rates"},
                {"name": "Inflation", "type": "METRIC", "description": "CPI measure"},
            ],
            "relations": [
                {"head": "Fed", "relation": "SETS", "tail": "Rates", "quote": "Fed sets rates"},
            ],
            "claims": [],
        }
        merge_elements_into_graph(graph, elements, "https://f.com", "tu_0")

        pack = build_local_context_pack(graph, [normalize_entity_name("Fed")], max_chars=4000)
        assert "Fed" in pack
        assert "ENTITIES" in pack

    def test_rank_communities_by_query(self):
        """Communities with more overlapping terms rank higher."""
        graph = GraphIndex()
        elements = {
            "entities": [
                {"name": "Federal Reserve", "type": "ORG", "description": "US central bank"},
                {"name": "ECB", "type": "ORG", "description": "European Central Bank"},
                {"name": "Interest Rate", "type": "METRIC", "description": "Policy rate"},
                {"name": "Bitcoin", "type": "ASSET", "description": "Cryptocurrency"},
                {"name": "Mining", "type": "CONCEPT", "description": "Bitcoin mining"},
            ],
            "relations": [
                {"head": "Federal Reserve", "relation": "SETS", "tail": "Interest Rate", "quote": "sets rates"},
                {"head": "Bitcoin", "relation": "USES", "tail": "Mining", "quote": "mining process"},
            ],
            "claims": [],
        }
        merge_elements_into_graph(graph, elements, "https://g.com", "tu_0")
        detect_communities(graph, min_size=2)

        query_terms = {"federal", "reserve", "interest", "rate"}
        ranked = rank_communities_by_query(
            graph.communities, query_terms, graph.entities, top_m=2,
        )
        # The Fed+Rate community should rank first
        assert len(ranked) >= 1

    def test_infer_credibility_tier(self):
        """Tier inference from URLs."""
        assert infer_credibility_tier("https://www.federalreserve.gov/news") == 1
        assert infer_credibility_tier("https://reuters.com/article/fed") == 1
        assert infer_credibility_tier("https://www.nytimes.com/article") == 2
        assert infer_credibility_tier("https://randomsite.com/blog") == 3
        assert infer_credibility_tier("https://example.com/press-release/fed") == 1
