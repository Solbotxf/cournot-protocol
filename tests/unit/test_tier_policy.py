"""
Module 05 - Tier Policy Tests

Tests for provenance tier classification and policy enforcement.
"""

import pytest
from datetime import datetime, timezone

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

from agents.collector.verification import (
    DEFAULT_TIER_MAPPING,
    SelectionPolicyEnforcer,
    TierPolicy,
)
from agents.collector.data_sources import FetchedArtifact


def create_test_evidence_item(
    evidence_id: str = "ev_001",
    tier: int = 0,
    kind: str = "none",
    requirement_id: str = "req_001"
) -> EvidenceItem:
    """Create a test evidence item."""
    return EvidenceItem(
        evidence_id=evidence_id,
        requirement_id=requirement_id,
        source=SourceDescriptor(
            source_id="test",
            uri="https://example.com/data",
        ),
        retrieval=RetrievalReceipt(
            retrieved_at=datetime.now(timezone.utc),
            method="http_get",
            request_fingerprint="0x1234",
            response_fingerprint="0x5678",
        ),
        provenance=ProvenanceProof(
            tier=tier,
            kind=kind,
        ),
        content_type="json",
        content={"test": "data"},
    )


def create_test_evidence_bundle(
    items: list[EvidenceItem] | None = None
) -> EvidenceBundle:
    """Create a test evidence bundle."""
    if items is None:
        items = [create_test_evidence_item()]

    return EvidenceBundle(
        collection_time=datetime(2026, 1, 20, 10, 0, 0, tzinfo=timezone.utc),
        bundle_id="bundle_001",
        collector_id="test_collector",
        items=items,
    )


def create_test_prompt_spec(
    market_min_tier: int = 0,
    requirement_min_tier: int = 0,
    allowed_sources: list[SourcePolicy] | None = None
) -> PromptSpec:
    """Create a test prompt spec."""
    return PromptSpec(
        market=MarketSpec(
            market_id="test_market",
            question="Test?",
            event_definition="Test event",
            resolution_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
            resolution_window=ResolutionWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 12, 31, tzinfo=timezone.utc),
            ),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="rule_001", description="Test rule")
            ]),
            allowed_sources=allowed_sources or [
                SourcePolicy(source_id="test", kind="api", allow=True)
            ],
            min_provenance_tier=market_min_tier,
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Test",
            predicate="will happen",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Test requirement",
                source_targets=[
                    SourceTarget(
                        source_id="test",
                        uri="https://example.com/data",
                        method="GET",
                        expected_content_type="json",
                    )
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
                min_provenance_tier=requirement_min_tier,
            )
        ],
    )


class TestTierMapping:
    """Tests for provenance tier mapping."""

    def test_default_tier_mapping(self):
        """Default tier mapping should be correctly defined."""
        assert DEFAULT_TIER_MAPPING["none"] == 0
        assert DEFAULT_TIER_MAPPING["hashlog"] == 1
        assert DEFAULT_TIER_MAPPING["signature"] == 2
        assert DEFAULT_TIER_MAPPING["notary"] == 2
        assert DEFAULT_TIER_MAPPING["zktls"] == 3

    def test_custom_tier_mapping(self):
        """Custom tier mapping should override defaults."""
        custom_mapping = {
            "none": 0,
            "signature": 5,
            "zktls": 10,
        }

        policy = TierPolicy(tier_mapping=custom_mapping)

        assert policy.get_tier_for_kind("none") == 0
        assert policy.get_tier_for_kind("signature") == 5
        assert policy.get_tier_for_kind("zktls") == 10


class TestRequiredTier:
    """Tests for required tier computation."""

    def test_market_tier_only(self):
        """Should use market tier when requirement tier is 0."""
        prompt_spec = create_test_prompt_spec(
            market_min_tier=2, requirement_min_tier=0)
        policy = TierPolicy()

        required = policy.required_tier(
            prompt_spec,
            prompt_spec.data_requirements[0]
        )

        assert required == 2

    def test_requirement_tier_only(self):
        """Should use requirement tier when market tier is 0."""
        prompt_spec = create_test_prompt_spec(
            market_min_tier=0, requirement_min_tier=3)
        policy = TierPolicy()

        required = policy.required_tier(
            prompt_spec,
            prompt_spec.data_requirements[0]
        )

        assert required == 3

    def test_max_of_both_tiers(self):
        """Should use maximum of market and requirement tiers."""
        prompt_spec = create_test_prompt_spec(
            market_min_tier=1, requirement_min_tier=2)
        policy = TierPolicy()

        required = policy.required_tier(
            prompt_spec,
            prompt_spec.data_requirements[0]
        )

        assert required == 2

        # Reverse case
        prompt_spec2 = create_test_prompt_spec(
            market_min_tier=3, requirement_min_tier=1)
        required2 = policy.required_tier(
            prompt_spec2,
            prompt_spec2.data_requirements[0]
        )

        assert required2 == 3


class TestProvenanceClassification:
    """Tests for provenance classification."""

    def test_classify_plain_http(self):
        """Plain HTTP response should be tier 0."""
        policy = TierPolicy()

        target = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data",
            method="GET",
            expected_content_type="json",
        )

        artifact = FetchedArtifact(
            raw_bytes=b'{"data": 1}',
            content_type="json",
            parsed={"data": 1},
            status_code=200,
        )

        provenance = policy.classify_provenance(target, artifact)

        assert provenance.tier == 0
        assert provenance.kind == "none"

    def test_classify_with_signature_header(self):
        """Response with signature header should be tier 2."""
        policy = TierPolicy()

        target = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data",
            method="GET",
            expected_content_type="json",
        )

        artifact = FetchedArtifact(
            raw_bytes=b'{"data": 1}',
            content_type="json",
            parsed={"data": 1},
            status_code=200,
            response_headers={"x-signature": "abc123signature"},
        )

        provenance = policy.classify_provenance(target, artifact)

        assert provenance.tier == 2
        assert provenance.kind == "signature"

    def test_classify_with_explicit_kind(self):
        """Explicit kind should override inference."""
        policy = TierPolicy()

        target = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data",
            method="GET",
            expected_content_type="json",
        )

        artifact = FetchedArtifact(
            raw_bytes=b'{"data": 1}',
            content_type="json",
            parsed={"data": 1},
            status_code=200,
        )

        provenance = policy.classify_provenance(
            target,
            artifact,
            proof_kind="zktls",
            proof_blob="zkproof123"
        )

        assert provenance.tier == 3
        assert provenance.kind == "zktls"
        assert provenance.proof_blob == "zkproof123"


class TestItemTierCheck:
    """Tests for individual item tier checking."""

    def test_tier_met(self):
        """Check should pass when tier is met."""
        policy = TierPolicy()

        item = create_test_evidence_item(tier=2, kind="signature")
        check = policy.check_item_tier(item, required_tier=1)

        assert check.ok is True
        assert check.severity == "info"

    def test_tier_exact_match(self):
        """Check should pass when tier exactly matches."""
        policy = TierPolicy()

        item = create_test_evidence_item(tier=2, kind="signature")
        check = policy.check_item_tier(item, required_tier=2)

        assert check.ok is True

    def test_tier_not_met_strict(self):
        """Check should fail in strict mode when tier not met."""
        policy = TierPolicy(strict_mode=True)

        item = create_test_evidence_item(tier=0, kind="none")
        check = policy.check_item_tier(item, required_tier=2)

        assert check.ok is False
        assert check.severity == "error"

    def test_tier_not_met_lenient(self):
        """Check should warn in lenient mode when tier not met."""
        policy = TierPolicy(strict_mode=False)

        item = create_test_evidence_item(tier=0, kind="none")
        check = policy.check_item_tier(item, required_tier=2)

        assert check.ok is False
        assert check.severity == "warn"


class TestBundleEnforcement:
    """Tests for bundle-level tier policy enforcement."""

    def test_all_items_meet_tier(self):
        """Should pass when all items meet tier requirements."""
        policy = TierPolicy()

        items = [
            create_test_evidence_item("ev_001", tier=2, kind="signature"),
            create_test_evidence_item("ev_002", tier=3, kind="zktls"),
        ]
        bundle = create_test_evidence_bundle(items)
        prompt_spec = create_test_prompt_spec(market_min_tier=1)

        result = policy.enforce(bundle, prompt_spec)

        assert result.ok is True
        assert len(result.checks) == 2

    def test_some_items_fail_tier_strict(self):
        """Should fail in strict mode when any item fails tier."""
        policy = TierPolicy(strict_mode=True)

        items = [
            create_test_evidence_item("ev_001", tier=2, kind="signature"),
            create_test_evidence_item("ev_002", tier=0, kind="none"),
        ]
        bundle = create_test_evidence_bundle(items)
        prompt_spec = create_test_prompt_spec(market_min_tier=1)

        result = policy.enforce(bundle, prompt_spec)

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "provenance_tier"

    def test_empty_bundle(self):
        """Should handle empty bundle."""
        policy = TierPolicy()

        bundle = create_test_evidence_bundle(items=[])
        prompt_spec = create_test_prompt_spec()

        result = policy.enforce(bundle, prompt_spec)

        assert result.ok is True
        assert len(result.checks) == 0


class TestSelectionPolicyEnforcement:
    """Tests for selection policy enforcement."""

    def test_single_best_met(self):
        """Single best policy should pass with one qualifying item."""
        enforcer = SelectionPolicyEnforcer()

        items = [create_test_evidence_item(tier=1)]
        requirement = DataRequirement(
            requirement_id="req_001",
            description="Test",
            source_targets=[
                SourceTarget(
                    source_id="test",
                    uri="https://example.com",
                    method="GET",
                    expected_content_type="json",
                )
            ],
            selection_policy=SelectionPolicy(
                strategy="single_best",
                min_sources=1,
                max_sources=1,
                quorum=1,
            ),
        )

        result = enforcer.enforce_quorum(items, requirement, required_tier=1)

        assert result.ok is True

    def test_quorum_met(self):
        """Quorum policy should pass when quorum is met."""
        enforcer = SelectionPolicyEnforcer()

        items = [
            create_test_evidence_item("ev_001", tier=2),
            create_test_evidence_item("ev_002", tier=2),
            create_test_evidence_item("ev_003", tier=2),
        ]
        requirement = DataRequirement(
            requirement_id="req_001",
            description="Test",
            source_targets=[
                SourceTarget(
                    source_id="test",
                    uri="https://example.com",
                    method="GET",
                    expected_content_type="json",
                )
            ],
            selection_policy=SelectionPolicy(
                strategy="multi_source_quorum",
                min_sources=2,
                max_sources=3,
                quorum=2,
            ),
        )

        result = enforcer.enforce_quorum(items, requirement, required_tier=1)

        assert result.ok is True

    def test_quorum_not_met_strict(self):
        """Quorum policy should fail when quorum not met in strict mode."""
        enforcer = SelectionPolicyEnforcer(strict_mode=True)

        items = [
            create_test_evidence_item("ev_001", tier=2),
        ]
        requirement = DataRequirement(
            requirement_id="req_001",
            description="Test",
            source_targets=[
                SourceTarget(
                    source_id="test",
                    uri="https://example.com",
                    method="GET",
                    expected_content_type="json",
                )
            ],
            selection_policy=SelectionPolicy(
                strategy="multi_source_quorum",
                min_sources=2,
                max_sources=3,
                quorum=2,
            ),
        )

        result = enforcer.enforce_quorum(items, requirement, required_tier=1)

        assert result.ok is False
        assert result.challenge is not None

    def test_quorum_items_below_tier_not_counted(self):
        """Items below required tier should not count toward quorum."""
        enforcer = SelectionPolicyEnforcer(strict_mode=True)

        items = [
            create_test_evidence_item("ev_001", tier=2),
            create_test_evidence_item("ev_002", tier=0),  # Below tier
            create_test_evidence_item("ev_003", tier=0),  # Below tier
        ]
        requirement = DataRequirement(
            requirement_id="req_001",
            description="Test",
            source_targets=[
                SourceTarget(
                    source_id="test",
                    uri="https://example.com",
                    method="GET",
                    expected_content_type="json",
                )
            ],
            selection_policy=SelectionPolicy(
                strategy="multi_source_quorum",
                min_sources=2,
                max_sources=3,
                quorum=2,
            ),
        )

        result = enforcer.enforce_quorum(items, requirement, required_tier=1)

        # Only 1 item meets tier, need 2 for quorum
        assert result.ok is False

    def test_fallback_chain_met(self):
        """Fallback chain policy should pass with one qualifying item."""
        enforcer = SelectionPolicyEnforcer()

        items = [create_test_evidence_item(tier=1)]
        requirement = DataRequirement(
            requirement_id="req_001",
            description="Test",
            source_targets=[
                SourceTarget(
                    source_id="test",
                    uri="https://example.com",
                    method="GET",
                    expected_content_type="json",
                )
            ],
            selection_policy=SelectionPolicy(
                strategy="fallback_chain",
                min_sources=1,
                max_sources=3,
                quorum=1,
            ),
        )

        result = enforcer.enforce_quorum(items, requirement, required_tier=1)

        assert result.ok is True
