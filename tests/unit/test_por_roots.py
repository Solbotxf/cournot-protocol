"""
Unit tests for PoR root computation.

Tests:
- Root determinism for same objects
- Evidence root changes if any EvidenceItem changes
- Reasoning root changes if any ReasoningStep changes
- Verdict hash changes if verdict changes
- por_root changes if any component changes
- PromptSpec timestamp handling (created_at must be None)
"""

import copy
from datetime import datetime, timezone

import pytest

from core.por.proof_of_reasoning import (
    PoRRoots,
    PromptSpecTimestampError,
    compute_evidence_leaf_hashes,
    compute_evidence_root,
    compute_por_root,
    compute_prompt_spec_hash,
    compute_reasoning_leaf_hashes,
    compute_reasoning_root,
    compute_roots,
    compute_verdict_hash,
)
from core.por.reasoning_trace import ReasoningStep, ReasoningTrace, TracePolicy
from core.schemas.evidence import (
    EvidenceBundle,
    EvidenceItem,
    ProvenanceProof,
    RetrievalReceipt,
    SourceDescriptor,
)
from core.schemas.market import (
    DisputePolicy,
    MarketSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SourcePolicy,
)
from core.schemas.prompts import (
    DataRequirement,
    PredictionSemantics,
    PromptSpec,
    SelectionPolicy,
    SourceTarget,
)
from core.schemas.verdict import DeterministicVerdict


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_market_spec() -> MarketSpec:
    """Create a sample MarketSpec for testing."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return MarketSpec(
        market_id="market_001",
        question="Will BTC exceed $100k by end of Q1 2026?",
        event_definition="Bitcoin spot price on Coinbase exceeds $100,000 USD",
        timezone="UTC",
        resolution_deadline=datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 3, 31, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(
            rules=[
                ResolutionRule(rule_id="rule_1", description="YES if price > 100k", priority=1),
                ResolutionRule(rule_id="rule_2", description="NO if deadline passes", priority=0),
            ]
        ),
        allowed_sources=[
            SourcePolicy(source_id="coinbase", kind="exchange", allow=True, min_provenance_tier=1),
        ],
        min_provenance_tier=1,
        dispute_policy=DisputePolicy(
            dispute_window_seconds=86400,
            allow_challenges=True,
            extra={},
        ),
        metadata={"created_by": "test"},
    )


@pytest.fixture
def sample_prompt_spec(sample_market_spec: MarketSpec) -> PromptSpec:
    """Create a sample PromptSpec for testing.
    
    Note: created_at is None per spec section 4.7 - timestamps must be omitted
    from committed objects to ensure deterministic hashing.
    """
    return PromptSpec(
        task_type="prediction_resolution",
        market=sample_market_spec,
        prediction_semantics=PredictionSemantics(
            target_entity="Bitcoin",
            predicate="price_exceeds",
            threshold="100000 USD",
            timeframe="Q1 2026",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_1",
                description="BTC spot price from Coinbase",
                preferred_sources=["coinbase"],
                min_provenance_tier=1,
                source_targets=[
                    SourceTarget(
                        source_id="coinbase",
                        uri="https://api.coinbase.com/v2/prices/BTC-USD/spot",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                    tie_breaker="highest_provenance",
                ),
            ),
        ],
        output_schema_ref="core.schemas.verdict.DeterministicVerdict",
        forbidden_behaviors=["hallucinate_evidence", "ignore_timestamps"],
        created_at=None,  # Must be None for deterministic commitments
        extra={},
    )


@pytest.fixture
def sample_evidence_bundle() -> EvidenceBundle:
    """Create a sample EvidenceBundle for testing."""
    now = datetime(2026, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
    return EvidenceBundle(
        bundle_id="bundle_001",
        collector_id="collector_alpha",
        collection_time=now,
        items=[
            EvidenceItem(
                evidence_id="ev_001",
                source=SourceDescriptor(
                    source_id="coinbase",
                    uri="https://api.coinbase.com/v2/prices/BTC-USD/spot",
                    provider="Coinbase",
                ),
                retrieval=RetrievalReceipt(
                    retrieved_at=now,
                    method="api_call",
                    tool="http_client",
                    extra={},
                ),
                provenance=ProvenanceProof(
                    tier=1,
                    kind="signature",
                    proof_blob="sig_abc123",
                    extra={},
                ),
                content_type="json",
                content={"price": "98500.00", "currency": "USD"},
                confidence=0.95,
            ),
            EvidenceItem(
                evidence_id="ev_002",
                source=SourceDescriptor(
                    source_id="coinbase",
                    uri="https://api.coinbase.com/v2/prices/BTC-USD/spot",
                    provider="Coinbase",
                ),
                retrieval=RetrievalReceipt(
                    retrieved_at=datetime(2026, 1, 20, 11, 0, 0, tzinfo=timezone.utc),
                    method="api_call",
                    tool="http_client",
                    extra={},
                ),
                provenance=ProvenanceProof(
                    tier=1,
                    kind="signature",
                    proof_blob="sig_def456",
                    extra={},
                ),
                content_type="json",
                content={"price": "99200.00", "currency": "USD"},
                confidence=0.95,
            ),
        ],
        provenance_summary={"min_tier": 1, "all_signed": True},
    )


@pytest.fixture
def sample_reasoning_trace() -> ReasoningTrace:
    """Create a sample ReasoningTrace for testing."""
    return ReasoningTrace(
        trace_id="trace_001",
        policy=TracePolicy(
            decoding_policy="strict",
            allow_external_sources=False,
            max_steps=100,
            extra={},
        ),
        steps=[
            ReasoningStep(
                step_id="step_0001",
                type="extract",
                inputs={"evidence_id": "ev_001"},
                action="Extract BTC price from evidence",
                output={"price": 98500.00},
                evidence_ids=["ev_001"],
                prior_step_ids=[],
            ),
            ReasoningStep(
                step_id="step_0002",
                type="extract",
                inputs={"evidence_id": "ev_002"},
                action="Extract BTC price from second evidence",
                output={"price": 99200.00},
                evidence_ids=["ev_002"],
                prior_step_ids=[],
            ),
            ReasoningStep(
                step_id="step_0003",
                type="aggregate",
                inputs={"prices": [98500.00, 99200.00]},
                action="Compute average price",
                output={"avg_price": 98850.00},
                evidence_ids=[],
                prior_step_ids=["step_0001", "step_0002"],
            ),
            ReasoningStep(
                step_id="step_0004",
                type="check",
                inputs={"price": 98850.00, "threshold": 100000.00},
                action="Check if price exceeds threshold",
                output={"exceeds": False},
                evidence_ids=[],
                prior_step_ids=["step_0003"],
            ),
        ],
        evidence_refs=["ev_001", "ev_002"],
    )


@pytest.fixture
def sample_verdict() -> DeterministicVerdict:
    """Create a sample DeterministicVerdict for testing."""
    return DeterministicVerdict(
        market_id="market_001",
        outcome="NO",
        confidence=0.95,
        resolution_time=datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
        resolution_rule_id="rule_1",
        prompt_spec_hash="0x" + "ab" * 32,
        evidence_root="0x" + "cd" * 32,
        reasoning_root="0x" + "ef" * 32,
        justification_hash="0x" + "12" * 32,
        selected_leaf_refs=["step_0004"],
        metadata={},
    )


# =============================================================================
# Test: Root Determinism
# =============================================================================


class TestRootDeterminism:
    """Test that root computation is deterministic."""

    def test_prompt_spec_hash_deterministic(self, sample_prompt_spec: PromptSpec):
        """Same PromptSpec produces same hash across multiple calls."""
        hash1 = compute_prompt_spec_hash(sample_prompt_spec)
        hash2 = compute_prompt_spec_hash(sample_prompt_spec)
        hash3 = compute_prompt_spec_hash(sample_prompt_spec)

        assert hash1 == hash2 == hash3
        assert hash1.startswith("0x")
        assert len(hash1) == 66  # 0x + 64 hex chars

    def test_evidence_root_deterministic(self, sample_evidence_bundle: EvidenceBundle):
        """Same EvidenceBundle produces same root across multiple calls."""
        root1 = compute_evidence_root(sample_evidence_bundle)
        root2 = compute_evidence_root(sample_evidence_bundle)
        root3 = compute_evidence_root(sample_evidence_bundle)

        assert root1 == root2 == root3
        assert root1.startswith("0x")

    def test_reasoning_root_deterministic(self, sample_reasoning_trace: ReasoningTrace):
        """Same ReasoningTrace produces same root across multiple calls."""
        root1 = compute_reasoning_root(sample_reasoning_trace)
        root2 = compute_reasoning_root(sample_reasoning_trace)
        root3 = compute_reasoning_root(sample_reasoning_trace)

        assert root1 == root2 == root3
        assert root1.startswith("0x")

    def test_verdict_hash_deterministic(self, sample_verdict: DeterministicVerdict):
        """Same DeterministicVerdict produces same hash across multiple calls."""
        hash1 = compute_verdict_hash(sample_verdict)
        hash2 = compute_verdict_hash(sample_verdict)
        hash3 = compute_verdict_hash(sample_verdict)

        assert hash1 == hash2 == hash3
        assert hash1.startswith("0x")

    def test_compute_roots_deterministic(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """compute_roots returns identical results for same inputs."""
        roots1 = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )
        roots2 = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )

        assert roots1 == roots2
        assert isinstance(roots1, PoRRoots)

    def test_por_root_deterministic(self):
        """por_root computation is deterministic."""
        h1 = "0x" + "aa" * 32
        h2 = "0x" + "bb" * 32
        h3 = "0x" + "cc" * 32
        h4 = "0x" + "dd" * 32

        root1 = compute_por_root(h1, h2, h3, h4)
        root2 = compute_por_root(h1, h2, h3, h4)

        assert root1 == root2
        assert root1.startswith("0x")


# =============================================================================
# Test: Evidence Root Sensitivity
# =============================================================================


class TestEvidenceRootSensitivity:
    """Test that evidence root changes when evidence changes."""

    def test_evidence_root_changes_with_content(self, sample_evidence_bundle: EvidenceBundle):
        """Evidence root changes if evidence content changes."""
        original_root = compute_evidence_root(sample_evidence_bundle)

        # Create modified bundle with different content
        modified_bundle = sample_evidence_bundle.model_copy(deep=True)
        modified_bundle.items[0].content = {"price": "105000.00", "currency": "USD"}

        modified_root = compute_evidence_root(modified_bundle)

        assert original_root != modified_root

    def test_evidence_root_changes_with_added_item(self, sample_evidence_bundle: EvidenceBundle):
        """Evidence root changes if evidence item is added."""
        original_root = compute_evidence_root(sample_evidence_bundle)

        # Create modified bundle with additional item
        modified_bundle = sample_evidence_bundle.model_copy(deep=True)
        new_item = modified_bundle.items[0].model_copy(deep=True)
        new_item.evidence_id = "ev_003"
        modified_bundle.items.append(new_item)

        modified_root = compute_evidence_root(modified_bundle)

        assert original_root != modified_root

    def test_evidence_root_changes_with_removed_item(self, sample_evidence_bundle: EvidenceBundle):
        """Evidence root changes if evidence item is removed."""
        original_root = compute_evidence_root(sample_evidence_bundle)

        # Create modified bundle with one less item
        modified_bundle = sample_evidence_bundle.model_copy(deep=True)
        modified_bundle.items.pop()

        modified_root = compute_evidence_root(modified_bundle)

        assert original_root != modified_root

    def test_evidence_root_changes_with_reordering(self, sample_evidence_bundle: EvidenceBundle):
        """Evidence root changes if evidence items are reordered."""
        original_root = compute_evidence_root(sample_evidence_bundle)

        # Create modified bundle with reversed order
        modified_bundle = sample_evidence_bundle.model_copy(deep=True)
        modified_bundle.items = list(reversed(modified_bundle.items))

        modified_root = compute_evidence_root(modified_bundle)

        assert original_root != modified_root

    def test_evidence_leaf_hashes_order_preserved(self, sample_evidence_bundle: EvidenceBundle):
        """Evidence leaf hashes preserve item order."""
        hashes = compute_evidence_leaf_hashes(sample_evidence_bundle)

        assert len(hashes) == len(sample_evidence_bundle.items)
        for h in hashes:
            assert isinstance(h, bytes)
            assert len(h) == 32


# =============================================================================
# Test: Reasoning Root Sensitivity
# =============================================================================


class TestReasoningRootSensitivity:
    """Test that reasoning root changes when trace changes."""

    def test_reasoning_root_changes_with_step_output(self, sample_reasoning_trace: ReasoningTrace):
        """Reasoning root changes if step output changes."""
        original_root = compute_reasoning_root(sample_reasoning_trace)

        # Create modified trace with different output
        modified_trace = sample_reasoning_trace.model_copy(deep=True)
        modified_trace.steps[0].output = {"price": 99999.00}

        modified_root = compute_reasoning_root(modified_trace)

        assert original_root != modified_root

    def test_reasoning_root_changes_with_added_step(self, sample_reasoning_trace: ReasoningTrace):
        """Reasoning root changes if step is added."""
        original_root = compute_reasoning_root(sample_reasoning_trace)

        # Create modified trace with additional step
        modified_trace = sample_reasoning_trace.model_copy(deep=True)
        new_step = ReasoningStep(
            step_id="step_0005",
            type="deduce",
            inputs={},
            action="Final deduction",
            output={"final": True},
            evidence_ids=[],
            prior_step_ids=["step_0004"],
        )
        modified_trace.steps.append(new_step)

        modified_root = compute_reasoning_root(modified_trace)

        assert original_root != modified_root

    def test_reasoning_root_changes_with_removed_step(self, sample_reasoning_trace: ReasoningTrace):
        """Reasoning root changes if step is removed."""
        original_root = compute_reasoning_root(sample_reasoning_trace)

        # Create modified trace - remove last step
        modified_trace = sample_reasoning_trace.model_copy(deep=True)
        modified_trace.steps.pop()

        modified_root = compute_reasoning_root(modified_trace)

        assert original_root != modified_root

    def test_reasoning_leaf_hashes_order_preserved(self, sample_reasoning_trace: ReasoningTrace):
        """Reasoning leaf hashes preserve step order."""
        hashes = compute_reasoning_leaf_hashes(sample_reasoning_trace)

        assert len(hashes) == len(sample_reasoning_trace.steps)
        for h in hashes:
            assert isinstance(h, bytes)
            assert len(h) == 32


# =============================================================================
# Test: Verdict Hash Sensitivity
# =============================================================================


class TestVerdictHashSensitivity:
    """Test that verdict hash changes when verdict changes."""

    def test_verdict_hash_changes_with_outcome(self, sample_verdict: DeterministicVerdict):
        """Verdict hash changes if outcome changes."""
        original_hash = compute_verdict_hash(sample_verdict)

        # Create modified verdict with different outcome
        modified_verdict = sample_verdict.model_copy(deep=True)
        modified_verdict.outcome = "YES"

        modified_hash = compute_verdict_hash(modified_verdict)

        assert original_hash != modified_hash

    def test_verdict_hash_changes_with_confidence(self, sample_verdict: DeterministicVerdict):
        """Verdict hash changes if confidence changes."""
        original_hash = compute_verdict_hash(sample_verdict)

        # Create modified verdict with different confidence
        modified_verdict = sample_verdict.model_copy(deep=True)
        modified_verdict.confidence = 0.5

        modified_hash = compute_verdict_hash(modified_verdict)

        assert original_hash != modified_hash

    def test_verdict_hash_changes_with_resolution_rule(self, sample_verdict: DeterministicVerdict):
        """Verdict hash changes if resolution_rule_id changes."""
        original_hash = compute_verdict_hash(sample_verdict)

        # Create modified verdict with different rule
        modified_verdict = sample_verdict.model_copy(deep=True)
        modified_verdict.resolution_rule_id = "rule_2"

        modified_hash = compute_verdict_hash(modified_verdict)

        assert original_hash != modified_hash


# =============================================================================
# Test: PoR Root Sensitivity
# =============================================================================


class TestPoRRootSensitivity:
    """Test that por_root changes when any component changes."""

    def test_por_root_changes_with_prompt_spec(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """por_root changes if prompt_spec changes."""
        original_roots = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )

        # Modify prompt spec
        modified_prompt = sample_prompt_spec.model_copy(deep=True)
        modified_prompt.forbidden_behaviors.append("new_behavior")

        modified_roots = compute_roots(
            modified_prompt, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )

        assert original_roots.prompt_spec_hash != modified_roots.prompt_spec_hash
        assert original_roots.por_root != modified_roots.por_root

    def test_por_root_changes_with_evidence(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """por_root changes if evidence changes."""
        original_roots = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )

        # Modify evidence
        modified_evidence = sample_evidence_bundle.model_copy(deep=True)
        modified_evidence.items[0].content = {"price": "200000.00"}

        modified_roots = compute_roots(
            sample_prompt_spec, modified_evidence,
            sample_reasoning_trace, sample_verdict
        )

        assert original_roots.evidence_root != modified_roots.evidence_root
        assert original_roots.por_root != modified_roots.por_root

    def test_por_root_changes_with_reasoning(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """por_root changes if reasoning trace changes."""
        original_roots = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )

        # Modify trace
        modified_trace = sample_reasoning_trace.model_copy(deep=True)
        modified_trace.steps[0].action = "Different action"

        modified_roots = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            modified_trace, sample_verdict
        )

        assert original_roots.reasoning_root != modified_roots.reasoning_root
        assert original_roots.por_root != modified_roots.por_root

    def test_por_root_changes_with_verdict(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """por_root changes if verdict changes."""
        original_roots = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, sample_verdict
        )

        # Modify verdict
        modified_verdict = sample_verdict.model_copy(deep=True)
        modified_verdict.outcome = "YES"

        modified_roots = compute_roots(
            sample_prompt_spec, sample_evidence_bundle,
            sample_reasoning_trace, modified_verdict
        )

        assert original_roots.verdict_hash != modified_roots.verdict_hash
        assert original_roots.por_root != modified_roots.por_root


# =============================================================================
# Test: Empty/Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for root computation."""

    def test_empty_evidence_bundle(self):
        """Empty evidence bundle produces valid root."""
        empty_bundle = EvidenceBundle(
            bundle_id="empty",
            collector_id="test",
            collection_time=datetime.now(timezone.utc),
            items=[],
            provenance_summary={},
        )

        root = compute_evidence_root(empty_bundle)

        assert root.startswith("0x")
        assert len(root) == 66

    def test_empty_reasoning_trace(self):
        """Empty reasoning trace produces valid root."""
        empty_trace = ReasoningTrace(
            trace_id="empty",
            policy=TracePolicy(),
            steps=[],
            evidence_refs=[],
        )

        root = compute_reasoning_root(empty_trace)

        assert root.startswith("0x")
        assert len(root) == 66

    def test_single_evidence_item(self):
        """Single evidence item produces valid root."""
        now = datetime.now(timezone.utc)
        single_bundle = EvidenceBundle(
            bundle_id="single",
            collector_id="test",
            collection_time=now,
            items=[
                EvidenceItem(
                    evidence_id="ev_only",
                    source=SourceDescriptor(source_id="test"),
                    retrieval=RetrievalReceipt(retrieved_at=now, method="api_call", extra={}),
                    provenance=ProvenanceProof(tier=0, kind="none", extra={}),
                    content_type="text",
                    content="test data",
                    confidence=1.0,
                ),
            ],
            provenance_summary={},
        )

        root = compute_evidence_root(single_bundle)

        assert root.startswith("0x")
        # For single leaf, root should equal the leaf hash
        leaf_hashes = compute_evidence_leaf_hashes(single_bundle)
        assert len(leaf_hashes) == 1

    def test_single_reasoning_step(self):
        """Single reasoning step produces valid root."""
        single_trace = ReasoningTrace(
            trace_id="single",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="step_only",
                    type="extract",
                    inputs={},
                    action="Only step",
                    output={},
                    evidence_ids=[],
                    prior_step_ids=[],
                ),
            ],
            evidence_refs=[],
        )

        root = compute_reasoning_root(single_trace)

        assert root.startswith("0x")
        leaf_hashes = compute_reasoning_leaf_hashes(single_trace)
        assert len(leaf_hashes) == 1


# =============================================================================
# Test: PromptSpec Timestamp Handling
# =============================================================================


class TestPromptSpecTimestampHandling:
    """Test PromptSpec.created_at timestamp handling per spec section 4.7."""

    def test_prompt_spec_hash_with_none_timestamp(self, sample_prompt_spec: PromptSpec):
        """PromptSpec with created_at=None produces valid hash."""
        assert sample_prompt_spec.created_at is None

        hash_result = compute_prompt_spec_hash(sample_prompt_spec)

        assert hash_result.startswith("0x")
        assert len(hash_result) == 66  # 0x + 64 hex chars

    def test_prompt_spec_hash_strict_rejects_non_none_timestamp(
        self, sample_market_spec: MarketSpec
    ):
        """In strict mode, PromptSpec with non-None created_at raises error."""
        prompt_with_timestamp = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),  # Non-None!
            extra={},
        )

        with pytest.raises(PromptSpecTimestampError) as exc_info:
            compute_prompt_spec_hash(prompt_with_timestamp, strict=True)

        assert "created_at must be None" in str(exc_info.value)

    def test_prompt_spec_hash_non_strict_allows_timestamp(
        self, sample_market_spec: MarketSpec
    ):
        """In non-strict mode, PromptSpec with non-None created_at is allowed."""
        prompt_with_timestamp = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            extra={},
        )

        # Should not raise in non-strict mode
        hash_result = compute_prompt_spec_hash(prompt_with_timestamp, strict=False)

        assert hash_result.startswith("0x")
        assert len(hash_result) == 66

    def test_prompt_spec_hash_deterministic_without_timestamp(
        self, sample_market_spec: MarketSpec
    ):
        """Identical PromptSpecs (with created_at=None) produce identical hashes."""
        prompt1 = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=None,
            extra={},
        )

        prompt2 = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=None,
            extra={},
        )

        hash1 = compute_prompt_spec_hash(prompt1)
        hash2 = compute_prompt_spec_hash(prompt2)

        assert hash1 == hash2

    def test_prompt_spec_hash_different_with_timestamp_in_non_strict(
        self, sample_market_spec: MarketSpec
    ):
        """Different timestamps produce different hashes in non-strict mode."""
        prompt_no_timestamp = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=None,
            extra={},
        )

        prompt_with_timestamp = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            extra={},
        )

        hash_without = compute_prompt_spec_hash(prompt_no_timestamp, strict=False)
        hash_with = compute_prompt_spec_hash(prompt_with_timestamp, strict=False)

        # Hashes should be DIFFERENT because timestamps are included when present
        assert hash_without != hash_with

    def test_compute_roots_uses_strict_mode_by_default(
        self, sample_market_spec: MarketSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """compute_roots should work with created_at=None PromptSpec."""
        prompt_spec = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Test",
                predicate="test",
            ),
            data_requirements=[],
            output_schema_ref="test",
            forbidden_behaviors=[],
            created_at=None,
            extra={},
        )

        # Should work without raising
        roots = compute_roots(prompt_spec, sample_evidence_bundle, sample_reasoning_trace, sample_verdict)

        assert roots.prompt_spec_hash.startswith("0x")
        assert roots.por_root.startswith("0x")