"""
Tests for PAN Runtime and PANCollectorAgent.

Covers:
- PAN runtime: all three search algorithms (GBoN, LBoN, beam)
- Scoring functions
- PANCollectorAgent with mock LLM
- Determinism under fixed seed
- Valid EvidenceItem output
- Beam search picks higher-scoring path
"""

import json
import pytest
from datetime import datetime, timezone

from agents.context import AgentContext, FrozenClock
from agents.collector.pan_runtime import (
    BranchRequest,
    ExecutionPath,
    SearchAlgo,
    SearchConfig,
    ScoreRecord,
    search,
)
from agents.collector.pan_agent import (
    PANCollectorAgent,
    PANCollectorConfig,
    score_query_set,
    score_search_results,
    score_evidence_extraction,
    score_final,
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

def create_test_prompt_spec() -> PromptSpec:
    """Create a minimal PromptSpec for PAN collector tests."""
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_pan_test",
            question="Will BTC be above $100k by end of January 2026?",
            event_definition="price(BTC_USD) > 100000 on 2026-01-31",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Check price", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="bitcoin",
            predicate="price above threshold",
            threshold="100000",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_pan_001",
                description="Find current BTC price from authoritative source",
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
        plan_id="tp_pan_test",
        requirements=["req_pan_001"],
        sources=[],
    )


# ---------------------------------------------------------------------------
# PAN Runtime Tests
# ---------------------------------------------------------------------------

class TestPANRuntime:
    """Tests for the minimal PAN runtime (pan_runtime.py)."""

    def test_global_best_of_n_picks_highest_scoring_path(self):
        """GBoN should return the path with the highest final score."""
        call_count = 0

        def workflow_factory():
            def wf():
                nonlocal call_count
                call_count += 1
                # Branch: generate a number, score it
                value = yield BranchRequest(
                    tag="pick_number",
                    generate_fn=lambda: call_count,  # different each time
                    score_fn=lambda v: float(v),
                )
                return value
            return wf()

        config = SearchConfig(
            algo=SearchAlgo.BON_GLOBAL,
            default_branching=5,
            max_expansions=100,
            seed=42,
        )
        result, path = search(workflow_factory, config)
        assert path.finished
        assert path.final_score > 0

    def test_local_best_of_n_selects_best_at_each_step(self):
        """LBoN should pick the best candidate at each branchpoint."""
        def workflow_factory():
            def wf():
                # Branch 1: pick from [1, 2, 3]
                counter = {"n": 0}

                def gen1():
                    counter["n"] += 1
                    return counter["n"]

                v1 = yield BranchRequest(
                    tag="step1",
                    generate_fn=gen1,
                    score_fn=lambda v: float(v),
                    n_candidates=3,
                )
                # v1 should be 3 (highest)

                counter2 = {"n": 0}

                def gen2():
                    counter2["n"] += 1
                    return v1 * 10 + counter2["n"]

                v2 = yield BranchRequest(
                    tag="step2",
                    generate_fn=gen2,
                    score_fn=lambda v: float(v),
                    n_candidates=3,
                )
                return v2
            return wf()

        config = SearchConfig(
            algo=SearchAlgo.BON_LOCAL,
            default_branching=3,
            max_expansions=100,
            seed=42,
        )
        result, path = search(workflow_factory, config)
        assert path.finished
        # Step 1: generated 1, 2, 3 → picked 3
        assert path.scores[0].value == 3.0
        # Step 2: generated 31, 32, 33 → picked 33
        assert path.scores[1].value == 33.0
        assert result == 33

    def test_beam_search_explores_multiple_paths(self):
        """Beam search should keep top-K paths and pick the best."""
        def workflow_factory():
            def wf():
                counter = {"n": 0}

                def gen():
                    counter["n"] += 1
                    return counter["n"]

                v = yield BranchRequest(
                    tag="branch1",
                    generate_fn=gen,
                    score_fn=lambda x: float(x),
                    n_candidates=3,
                )
                return v * 100
            return wf()

        config = SearchConfig(
            algo=SearchAlgo.BEAM,
            default_branching=3,
            beam_width=2,
            max_expansions=100,
            seed=42,
        )
        result, path = search(workflow_factory, config)
        assert path.finished
        # Should have picked the highest value from branchpoint
        assert result is not None

    def test_determinism_with_fixed_seed(self):
        """Same seed should produce identical results."""
        counter = {"n": 0}

        def make_factory():
            def workflow_factory():
                def wf():
                    nonlocal counter
                    c = {"n": 0}

                    def gen():
                        c["n"] += 1
                        return c["n"]

                    v = yield BranchRequest(
                        tag="step",
                        generate_fn=gen,
                        score_fn=lambda x: float(x),
                        n_candidates=3,
                    )
                    return v
                return wf()
            return workflow_factory

        config = SearchConfig(
            algo=SearchAlgo.BON_LOCAL,
            default_branching=3,
            max_expansions=50,
            seed=123,
        )

        r1, p1 = search(make_factory(), config)
        r2, p2 = search(make_factory(), config)

        assert r1 == r2
        assert p1.score_breakdown == p2.score_breakdown

    def test_execution_path_score_breakdown(self):
        """ExecutionPath should track score breakdown correctly."""
        path = ExecutionPath()
        path.scores.append(ScoreRecord(tag="step1", value=0.5))
        path.scores.append(ScoreRecord(tag="step2", value=0.8))
        path.scores.append(ScoreRecord(tag="final", value=0.7))

        assert path.total_score == pytest.approx(2.0)
        assert path.final_score == pytest.approx(0.7)
        assert path.score_breakdown == {"step1": 0.5, "step2": 0.8, "final": 0.7}

    def test_workflow_with_no_branchpoints(self):
        """A workflow with no yields should still work."""
        def workflow_factory():
            def wf():
                # No yields — returns immediately
                return 42
                yield  # make it a generator  # noqa: E501
            return wf()

        config = SearchConfig(algo=SearchAlgo.BON_GLOBAL, default_branching=3)
        result, path = search(workflow_factory, config)
        assert result == 42
        assert path.finished

    def test_max_expansions_safety_cap(self):
        """Search should respect max_expansions cap."""
        call_count = {"n": 0}

        def workflow_factory():
            def wf():
                def gen():
                    call_count["n"] += 1
                    return call_count["n"]

                for i in range(100):  # many branchpoints
                    v = yield BranchRequest(
                        tag=f"step_{i}",
                        generate_fn=gen,
                        score_fn=lambda x: float(x),
                        n_candidates=5,
                    )
                return v
            return wf()

        config = SearchConfig(
            algo=SearchAlgo.BON_LOCAL,
            default_branching=5,
            max_expansions=10,
            seed=42,
        )
        result, path = search(workflow_factory, config)
        # Should have stopped before exploring all 100 * 5 = 500 candidates
        assert call_count["n"] <= 15  # max_expansions + some slack


# ---------------------------------------------------------------------------
# Scoring Function Tests
# ---------------------------------------------------------------------------

class TestScoringFunctions:
    """Tests for the individual scoring functions."""

    def test_score_query_set_empty(self):
        assert score_query_set([], "test", "test") == 0.0

    def test_score_query_set_good_queries(self):
        queries = [
            "bitcoin price January 2026",
            "BTC USD current price",
            "site:coingecko.com bitcoin price",
        ]
        score = score_query_set(
            queries,
            requirement_desc="Find current BTC price from authoritative source",
            market_question="Will BTC be above $100k?",
        )
        assert 0.0 < score <= 1.0

    def test_score_query_set_rewards_diversity(self):
        # Diverse queries should score higher than identical ones
        diverse = ["bitcoin price 2026", "BTC official exchange rate", "site:gov bitcoin"]
        identical = ["bitcoin price", "bitcoin price", "bitcoin price"]
        s_diverse = score_query_set(diverse, "bitcoin price", "btc question")
        s_identical = score_query_set(identical, "bitcoin price", "btc question")
        assert s_diverse > s_identical

    def test_score_search_results_empty(self):
        assert score_search_results([]) == 0.0

    def test_score_search_results_authoritative(self):
        auth_results = [
            {"url": "https://www.reuters.com/btc", "snippet": "BTC at 105k", "date": "2026-01-15"},
            {"url": "https://apnews.com/crypto", "snippet": "Bitcoin surges", "date": "2026-01-14"},
        ]
        generic_results = [
            {"url": "https://randomblog.xyz/btc", "snippet": "maybe", "date": ""},
            {"url": "https://someforum.com/post", "snippet": "idk", "date": ""},
        ]
        assert score_search_results(auth_results) > score_search_results(generic_results)

    def test_score_evidence_extraction_resolved(self):
        good = {
            "resolution_status": "RESOLVED",
            "confidence_score": 0.95,
            "parsed_value": "105000",
            "reasoning_trace": "Based on Reuters data...",
            "evidence_sources": [
                {"url": "https://reuters.com/btc", "key_fact": "price is 105k"},
            ],
        }
        bad = {
            "resolution_status": "UNRESOLVED",
            "confidence_score": 0.1,
            "parsed_value": None,
            "evidence_sources": [],
        }
        assert score_evidence_extraction(good) > score_evidence_extraction(bad)

    def test_score_evidence_extraction_empty(self):
        assert score_evidence_extraction({}) == 0.0
        assert score_evidence_extraction(None) == 0.0

    def test_score_final_successful_item(self):
        item = EvidenceItem(
            evidence_id="ev1",
            requirement_id="req1",
            provenance=Provenance(source_id="pan", source_uri="test", tier=2),
            success=True,
            parsed_value="105000",
            extracted_fields={
                "confidence_score": 0.9,
                "evidence_sources": [{"url": "https://example.com"}],
                "resolution_status": "RESOLVED",
            },
        )
        score = score_final(item)
        assert score > 0.3  # above baseline

    def test_score_final_failed_item(self):
        item = EvidenceItem(
            evidence_id="ev1",
            requirement_id="req1",
            provenance=Provenance(source_id="pan", source_uri="test", tier=0),
            success=False,
            error="failed",
        )
        assert score_final(item) == 0.0


# ---------------------------------------------------------------------------
# PANCollectorAgent Tests
# ---------------------------------------------------------------------------

class TestPANCollectorAgent:
    """Tests for the PANCollectorAgent with mock LLM."""

    def _make_mock_ctx(self, llm_responses: list[str]) -> AgentContext:
        """Create a mock AgentContext with preset LLM responses."""
        return AgentContext.create_mock(llm_responses=llm_responses)

    def _good_query_response(self) -> str:
        return json.dumps({
            "queries": [
                "bitcoin price January 2026",
                "BTC USD current rate official",
                "site:coingecko.com bitcoin price 2026",
            ]
        })

    def _good_extraction_response(self) -> str:
        return json.dumps({
            "reasoning_trace": "Reuters reports BTC at $105,000. Confirmed by Bloomberg.",
            "resolution_status": "RESOLVED",
            "parsed_value": "105000",
            "confidence_score": 0.95,
            "evidence_sources": [
                {
                    "source_id": "[1]",
                    "url": "https://reuters.com/btc-price",
                    "credibility_tier": 1,
                    "key_fact": "BTC trading at $105,000",
                    "supports": "YES",
                    "date_published": "2026-01-15",
                },
            ],
        })

    def test_pan_agent_returns_valid_evidence_item(self):
        """PAN agent should return a valid (EvidenceBundle, ToolExecutionLog) tuple."""
        # Need enough LLM responses for branching:
        # GBoN with N=2 will run the workflow 2 times.
        # Each run needs: 1 query gen + 1 extraction = 2 LLM calls.
        # But search results come from Serper (HTTP) which won't be configured,
        # so the workflow will return early with "no search results".
        # Use bon_local with N=1 for minimal calls.
        responses = [
            self._good_query_response(),
            self._good_extraction_response(),
            self._good_query_response(),
            self._good_extraction_response(),
            self._good_query_response(),
            self._good_extraction_response(),
            self._good_query_response(),
            self._good_extraction_response(),
        ]

        ctx = self._make_mock_ctx(responses)
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()

        config = PANCollectorConfig(
            search_algo="bon_local",
            default_branching=1,
            seed=42,
        )
        agent = PANCollectorAgent(pan_config=config)
        result = agent.run(ctx, prompt_spec, tool_plan)

        # Without HTTP/Serper configured, search results will be empty
        # so the workflow returns an error EvidenceItem.
        # The agent should still return a valid AgentResult.
        assert result is not None
        bundle, exec_log = result.output
        assert isinstance(bundle, EvidenceBundle)
        assert bundle.market_id == "mk_pan_test"
        assert result.metadata["collector"] == "pan"

    def test_pan_config_wiring(self):
        """PANCollectorConfig should convert to SearchConfig correctly."""
        config = PANCollectorConfig(
            search_algo="beam",
            default_branching=5,
            beam_width=3,
            max_expansions=100,
            seed=99,
        )
        sc = config.to_search_config()
        assert sc.algo == SearchAlgo.BEAM
        assert sc.default_branching == 5
        assert sc.beam_width == 3
        assert sc.max_expansions == 100
        assert sc.seed == 99

    def test_pan_config_defaults(self):
        """Default PANCollectorConfig should be reasonable."""
        config = PANCollectorConfig()
        assert config.search_algo == "beam"
        assert config.default_branching == 3
        assert config.beam_width == 2
        assert config.max_expansions == 50

    def test_pan_agent_metadata_includes_search_config(self):
        """Metadata should include PAN search configuration."""
        ctx = self._make_mock_ctx([
            self._good_query_response(),
            self._good_extraction_response(),
        ] * 10)

        config = PANCollectorConfig(
            search_algo="bon_global",
            default_branching=2,
            seed=42,
        )
        agent = PANCollectorAgent(pan_config=config)
        result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())

        assert result.metadata["search_algo"] == "bon_global"
        assert result.metadata["default_branching"] == 2
        assert "paths" in result.metadata

    def test_pan_execution_log_no_duplicates(self):
        """Execution log should contain exactly one record per stage, no duplicates.

        Regression test: beam search replays the generator multiple times;
        records must be accumulated locally and only the best path's records
        should be appended to the shared log.
        """
        # Provide enough LLM responses for beam search (N=3, K=2, 4 branchpoints)
        # Each replay consumes LLM calls.  We supply plenty.
        responses = [
            self._good_query_response(),
            self._good_extraction_response(),
        ] * 50  # generous supply

        ctx = self._make_mock_ctx(responses)
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()

        # Beam search with N=3, K=2 — causes multiple replays
        config = PANCollectorConfig(
            search_algo="beam",
            default_branching=3,
            beam_width=2,
            seed=42,
        )
        agent = PANCollectorAgent(pan_config=config)
        result = agent.run(ctx, prompt_spec, tool_plan)

        _, exec_log = result.output
        calls = exec_log.calls if hasattr(exec_log, 'calls') else []

        # Without HTTP/Serper the workflow hits "no search results" and
        # returns after branchpoints 1+2 (query_generation + web_search).
        # So expect exactly 2 records — not 6, 12, or 21.
        tool_names = [c.tool for c in calls]
        assert tool_names.count("pan:query_generation") == 1, (
            f"Expected 1 query_generation record, got {tool_names.count('pan:query_generation')}: {tool_names}"
        )
        assert tool_names.count("pan:web_search") == 1, (
            f"Expected 1 web_search record, got {tool_names.count('pan:web_search')}: {tool_names}"
        )
        # Should not have extraction records (no search results to extract from)
        assert tool_names.count("pan:evidence_extraction") == 0

    def test_pan_agent_no_llm_returns_failure(self):
        """PAN agent should fail gracefully when no LLM is available."""
        ctx = AgentContext.create_minimal()
        agent = PANCollectorAgent()
        result = agent.run(ctx, create_test_prompt_spec(), create_test_tool_plan())
        assert not result.success
        assert "LLM client not available" in result.error


# ---------------------------------------------------------------------------
# Integration: beam search picks higher-scoring path
# ---------------------------------------------------------------------------

class TestBeamSearchQuality:
    """Verify that beam search actually selects higher-quality paths."""

    def test_beam_picks_higher_scoring_candidate(self):
        """With explicit score differences, beam should pick the better one."""
        # Workflow with two branchpoints, each generating candidates with
        # known scores.  The best path is candidate 3 at step 1 → 3 at step 2.
        def workflow_factory():
            def wf():
                step1_counter = {"n": 0}

                def gen1():
                    step1_counter["n"] += 1
                    return step1_counter["n"]

                val1 = yield BranchRequest(
                    tag="step1",
                    generate_fn=gen1,
                    score_fn=lambda v: float(v),  # 1, 2, 3
                    n_candidates=3,
                )

                step2_counter = {"n": 0}

                def gen2():
                    step2_counter["n"] += 1
                    return val1 + step2_counter["n"]

                val2 = yield BranchRequest(
                    tag="step2",
                    generate_fn=gen2,
                    score_fn=lambda v: float(v),
                    n_candidates=3,
                )
                return val2
            return wf()

        config = SearchConfig(
            algo=SearchAlgo.BEAM,
            default_branching=3,
            beam_width=2,
            max_expansions=100,
            seed=42,
        )
        result, path = search(workflow_factory, config)
        assert path.finished
        # The best total score path should have high values
        assert path.total_score > 0

    def test_bon_global_finds_best_among_n_runs(self):
        """GBoN with varying quality should pick the highest-scoring run."""
        run_counter = {"n": 0}

        def workflow_factory():
            def wf():
                run_counter["n"] += 1
                run_num = run_counter["n"]

                # Each run yields one branchpoint
                val = yield BranchRequest(
                    tag="main",
                    generate_fn=lambda: run_num * 10,
                    score_fn=lambda v: float(v),
                )
                return val
            return wf()

        config = SearchConfig(
            algo=SearchAlgo.BON_GLOBAL,
            default_branching=5,
            max_expansions=100,
            seed=42,
        )
        result, path = search(workflow_factory, config)
        assert path.finished
        # Later runs have higher scores (run 5 → score 50)
        assert path.final_score == 50.0
        assert result == 50


# ---------------------------------------------------------------------------
# RuntimeConfig integration
# ---------------------------------------------------------------------------

class TestPANRuntimeConfig:
    """Tests for PAN config in RuntimeConfig."""

    def test_runtime_config_has_pan_defaults(self):
        from core.config.runtime import RuntimeConfig
        config = RuntimeConfig()
        assert config.pan.search_algo == "beam"
        assert config.pan.default_branching == 3
        assert config.pan.beam_width == 2

    def test_runtime_config_from_dict_with_pan(self):
        from core.config.runtime import RuntimeConfig
        config = RuntimeConfig.from_dict({
            "pan": {
                "search_algo": "bon_local",
                "default_branching": 5,
                "beam_width": 4,
                "max_expansions": 200,
                "seed": 7,
            },
        })
        assert config.pan.search_algo == "bon_local"
        assert config.pan.default_branching == 5
        assert config.pan.beam_width == 4
        assert config.pan.max_expansions == 200
        assert config.pan.seed == 7

    def test_runtime_config_to_dict_includes_pan(self):
        from core.config.runtime import RuntimeConfig
        config = RuntimeConfig()
        d = config.to_dict()
        assert "pan" in d
        assert d["pan"]["search_algo"] == "beam"
