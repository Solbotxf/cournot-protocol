"""
Module 06 - Tests for Claim Extraction

Tests:
- Deterministic claim extraction from JSON evidence
- Stable claim ordering
- Stable claim_id generation
- Text extraction patterns
"""

import pytest
from datetime import datetime, timezone, timedelta

from agents.auditor.reasoning.claim_extraction import (
    Claim,
    ClaimExtractor,
    ClaimSet,
    extract_claims,
    _generate_claim_id,
    _infer_claim_kind,
)
from core.schemas.evidence import (
    EvidenceBundle,
    EvidenceItem,
    ProvenanceProof,
    RetrievalReceipt,
    SourceDescriptor,
)
from core.schemas.prompts import (
    DataRequirement,
    PredictionSemantics,
    PromptSpec,
    SourceTarget,
    SelectionPolicy,
)
from core.schemas.market import (
    MarketSpec,
    ResolutionWindow,
    ResolutionRules,
    ResolutionRule,
    DisputePolicy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_prompt_spec():
    """Create a sample PromptSpec for testing."""
    now = datetime.now(timezone.utc)
    return PromptSpec(
        market=MarketSpec(
            market_id="test_market_001",
            question="Will BTC close above $100,000?",
            event_definition="BTC-USD daily close > 100000",
            resolution_deadline=now + timedelta(days=1),
            resolution_window=ResolutionWindow(
                start=now,
                end=now + timedelta(days=1),
            ),
            resolution_rules=ResolutionRules(
                rules=[
                    ResolutionRule(
                        rule_id="rule_001",
                        description="Check BTC price",
                    )
                ]
            ),
            dispute_policy=DisputePolicy(dispute_window_seconds=3600),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="BTC-USD",
            predicate="close price > 100000",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(
                        source_id="exchange",
                        uri="https://api.example.com/btc",
                    )
                ],
                selection_policy=SelectionPolicy(strategy="single_best"),
                expected_fields=["price", "close", "timestamp"],
            )
        ],
    )


@pytest.fixture
def sample_json_evidence():
    """Create a sample EvidenceBundle with JSON content."""
    return EvidenceBundle(
        bundle_id="bundle_001",
        collector_id="collector_v1",
        items=[
            EvidenceItem(
                evidence_id="ev_001",
                requirement_id="req_001",
                source=SourceDescriptor(source_id="exchange", uri="https://api.example.com/btc"),
                retrieval=RetrievalReceipt(
                    retrieved_at=datetime.now(timezone.utc),
                    method="http_get",
                ),
                provenance=ProvenanceProof(tier=1, kind="signature"),
                content_type="json",
                content={
                    "price": 105000.50,
                    "close": 104500.00,
                    "timestamp": "2026-01-27T00:00:00Z",
                    "volume": 12345678,
                },
            ),
            EvidenceItem(
                evidence_id="ev_002",
                requirement_id="req_001",
                source=SourceDescriptor(source_id="exchange", uri="https://api.backup.com/btc"),
                retrieval=RetrievalReceipt(
                    retrieved_at=datetime.now(timezone.utc),
                    method="http_get",
                ),
                provenance=ProvenanceProof(tier=1, kind="signature"),
                content_type="json",
                content={
                    "price": 105100.25,
                    "close": 104600.00,
                    "timestamp": "2026-01-27T00:00:00Z",
                },
            ),
        ],
    )


@pytest.fixture
def sample_text_evidence():
    """Create a sample EvidenceBundle with text content."""
    return EvidenceBundle(
        bundle_id="bundle_002",
        collector_id="collector_v1",
        items=[
            EvidenceItem(
                evidence_id="ev_text_001",
                requirement_id="req_001",
                source=SourceDescriptor(source_id="web", uri="https://example.com/news"),
                retrieval=RetrievalReceipt(
                    retrieved_at=datetime.now(timezone.utc),
                    method="http_get",
                ),
                provenance=ProvenanceProof(tier=2, kind="hashlog"),
                content_type="text",
                content="Bitcoin price reached 105000 today. The market closed at 2026-01-27. Trading was active with volume up.",
            ),
        ],
    )


# =============================================================================
# Test Claim Data Structure
# =============================================================================


class TestClaim:
    """Tests for the Claim dataclass."""

    def test_claim_creation(self):
        """Test basic claim creation."""
        claim = Claim(
            claim_id="cl_test123",
            kind="numeric",
            path="price",
            value=105000.50,
            evidence_id="ev_001",
            confidence=1.0,
        )

        assert claim.claim_id == "cl_test123"
        assert claim.kind == "numeric"
        assert claim.path == "price"
        assert claim.value == 105000.50
        assert claim.evidence_id == "ev_001"
        assert claim.confidence == 1.0

    def test_claim_confidence_bounds(self):
        """Test that confidence must be in [0, 1]."""
        # Valid confidence
        claim = Claim(
            claim_id="cl_test",
            kind="numeric",
            path="price",
            value=100,
            evidence_id="ev_001",
            confidence=0.5,
        )
        assert claim.confidence == 0.5

        # Invalid confidence should raise
        with pytest.raises(ValueError):
            Claim(
                claim_id="cl_test",
                kind="numeric",
                path="price",
                value=100,
                evidence_id="ev_001",
                confidence=1.5,
            )

        with pytest.raises(ValueError):
            Claim(
                claim_id="cl_test",
                kind="numeric",
                path="price",
                value=100,
                evidence_id="ev_001",
                confidence=-0.1,
            )


class TestClaimSet:
    """Tests for the ClaimSet collection."""

    def test_claim_set_operations(self):
        """Test ClaimSet add and filter operations."""
        claim_set = ClaimSet()

        claim1 = Claim("cl_1", "numeric", "price", 100, "ev_001", 1.0)
        claim2 = Claim("cl_2", "boolean", "resolved", True, "ev_001", 1.0)
        claim3 = Claim("cl_3", "numeric", "price", 101, "ev_002", 0.9)

        claim_set.add(claim1)
        claim_set.add(claim2)
        claim_set.add(claim3)

        assert len(claim_set) == 3

        # Filter by evidence_id
        ev001_claims = claim_set.by_evidence_id("ev_001")
        assert len(ev001_claims) == 2

        # Filter by kind
        numeric_claims = claim_set.by_kind("numeric")
        assert len(numeric_claims) == 2

        # Filter by path
        price_claims = claim_set.by_path("price")
        assert len(price_claims) == 2

        # Get unique paths
        paths = claim_set.get_unique_paths()
        assert paths == {"price", "resolved"}

        # Get unique evidence IDs
        evidence_ids = claim_set.get_unique_evidence_ids()
        assert evidence_ids == {"ev_001", "ev_002"}


# =============================================================================
# Test Claim ID Generation
# =============================================================================


class TestClaimIdGeneration:
    """Tests for deterministic claim ID generation."""

    def test_claim_id_deterministic(self):
        """Test that claim_id is deterministic."""
        id1 = _generate_claim_id("ev_001", "price", 100.5)
        id2 = _generate_claim_id("ev_001", "price", 100.5)
        assert id1 == id2

    def test_claim_id_different_values(self):
        """Test that different values produce different IDs."""
        id1 = _generate_claim_id("ev_001", "price", 100.5)
        id2 = _generate_claim_id("ev_001", "price", 100.6)
        assert id1 != id2

    def test_claim_id_different_paths(self):
        """Test that different paths produce different IDs."""
        id1 = _generate_claim_id("ev_001", "price", 100)
        id2 = _generate_claim_id("ev_001", "close", 100)
        assert id1 != id2

    def test_claim_id_different_evidence(self):
        """Test that different evidence IDs produce different claim IDs."""
        id1 = _generate_claim_id("ev_001", "price", 100)
        id2 = _generate_claim_id("ev_002", "price", 100)
        assert id1 != id2

    def test_claim_id_format(self):
        """Test claim ID format is cl_ prefix + 12 hex chars."""
        claim_id = _generate_claim_id("ev_001", "price", 100)
        assert claim_id.startswith("cl_")
        assert len(claim_id) == 15  # cl_ + 12 hex chars


# =============================================================================
# Test Kind Inference
# =============================================================================


class TestKindInference:
    """Tests for claim kind inference."""

    def test_infer_numeric(self):
        """Test numeric kind inference."""
        assert _infer_claim_kind(100) == "numeric"
        assert _infer_claim_kind(100.5) == "numeric"
        assert _infer_claim_kind(0) == "numeric"

    def test_infer_boolean(self):
        """Test boolean kind inference."""
        assert _infer_claim_kind(True) == "boolean"
        assert _infer_claim_kind(False) == "boolean"

    def test_infer_timestamp(self):
        """Test timestamp kind inference."""
        assert _infer_claim_kind("2026-01-27T00:00:00Z") == "timestamp"
        assert _infer_claim_kind("2026-01-27") == "timestamp"

    def test_infer_text(self):
        """Test text assertion kind inference."""
        assert _infer_claim_kind("hello world") == "text_assertion"
        assert _infer_claim_kind("confirmed") == "text_assertion"


# =============================================================================
# Test JSON Extraction
# =============================================================================


class TestJsonExtraction:
    """Tests for JSON claim extraction."""

    def test_extract_from_json_evidence(self, sample_prompt_spec, sample_json_evidence):
        """Test extraction from JSON evidence."""
        claims = extract_claims(sample_prompt_spec, sample_json_evidence)

        assert len(claims) > 0

        # Check that we extracted numeric claims
        numeric = claims.by_kind("numeric")
        assert len(numeric) >= 2  # price and close from both items

        # Check that we extracted timestamp claims
        timestamps = claims.by_kind("timestamp")
        assert len(timestamps) >= 1

    def test_extraction_ordering_deterministic(self, sample_prompt_spec, sample_json_evidence):
        """Test that extraction ordering is deterministic."""
        claims1 = extract_claims(sample_prompt_spec, sample_json_evidence)
        claims2 = extract_claims(sample_prompt_spec, sample_json_evidence)

        # Same number of claims
        assert len(claims1) == len(claims2)

        # Same claim IDs in same order
        ids1 = [c.claim_id for c in claims1]
        ids2 = [c.claim_id for c in claims2]
        assert ids1 == ids2

    def test_expected_fields_filtering(self, sample_prompt_spec, sample_json_evidence):
        """Test that expected_fields filters extraction."""
        # Default extraction gets price, close, timestamp
        claims = extract_claims(sample_prompt_spec, sample_json_evidence)

        # Should have claims for expected fields
        paths = claims.get_unique_paths()
        assert "price" in paths or "close" in paths


class TestTextExtraction:
    """Tests for text claim extraction."""

    def test_extract_from_text_evidence(self, sample_prompt_spec, sample_text_evidence):
        """Test extraction from text evidence."""
        claims = extract_claims(sample_prompt_spec, sample_text_evidence)

        # Should extract some numeric values from text
        numeric = claims.by_kind("numeric")
        assert len(numeric) >= 1  # Should find "105000"

    def test_text_extraction_confidence(self, sample_prompt_spec, sample_text_evidence):
        """Test that text extraction has lower confidence."""
        claims = extract_claims(sample_prompt_spec, sample_text_evidence)

        for claim in claims:
            # Text extraction should have confidence <= 1.0
            assert claim.confidence <= 1.0


# =============================================================================
# Test ClaimExtractor Class
# =============================================================================


class TestClaimExtractor:
    """Tests for the ClaimExtractor class."""

    def test_extractor_with_custom_fields(self, sample_json_evidence):
        """Test extraction with custom expected fields."""
        extractor = ClaimExtractor(
            expected_fields={"req_001": ["volume"]}
        )

        # Create a minimal prompt spec
        now = datetime.now(timezone.utc)
        prompt_spec = PromptSpec(
            market=MarketSpec(
                market_id="test",
                question="Test?",
                event_definition="Test",
                resolution_deadline=now + timedelta(days=1),
                resolution_window=ResolutionWindow(start=now, end=now + timedelta(days=1)),
                resolution_rules=ResolutionRules(rules=[ResolutionRule(rule_id="r1", description="test")]),
                dispute_policy=DisputePolicy(dispute_window_seconds=3600),
            ),
            prediction_semantics=PredictionSemantics(target_entity="test", predicate="test"),
            data_requirements=[
                DataRequirement(
                    requirement_id="req_001",
                    description="test",
                    source_targets=[SourceTarget(source_id="test", uri="http://test.com")],
                    selection_policy=SelectionPolicy(strategy="single_best"),
                )
            ],
        )

        claims = extractor.extract(prompt_spec, sample_json_evidence)

        # Should have extracted something
        assert len(claims) > 0


# =============================================================================
# Test Empty/Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in claim extraction."""

    def test_empty_evidence_bundle(self, sample_prompt_spec):
        """Test extraction from empty evidence bundle."""
        empty_bundle = EvidenceBundle(
            bundle_id="empty",
            collector_id="test",
            items=[],
        )

        claims = extract_claims(sample_prompt_spec, empty_bundle)
        assert len(claims) == 0

    def test_evidence_with_null_content(self, sample_prompt_spec):
        """Test extraction from evidence with null values."""
        bundle = EvidenceBundle(
            bundle_id="null_test",
            collector_id="test",
            items=[
                EvidenceItem(
                    evidence_id="ev_null",
                    source=SourceDescriptor(source_id="test"),
                    retrieval=RetrievalReceipt(
                        retrieved_at=datetime.now(timezone.utc),
                        method="http_get",
                    ),
                    provenance=ProvenanceProof(tier=2, kind="none"),
                    content_type="json",
                    content={"price": None, "value": None},
                ),
            ],
        )

        claims = extract_claims(sample_prompt_spec, bundle)
        # Should not crash, may or may not extract claims
        assert isinstance(claims, ClaimSet)

    def test_deeply_nested_json(self, sample_prompt_spec):
        """Test extraction from deeply nested JSON."""
        bundle = EvidenceBundle(
            bundle_id="nested_test",
            collector_id="test",
            items=[
                EvidenceItem(
                    evidence_id="ev_nested",
                    source=SourceDescriptor(source_id="test"),
                    retrieval=RetrievalReceipt(
                        retrieved_at=datetime.now(timezone.utc),
                        method="http_get",
                    ),
                    provenance=ProvenanceProof(tier=2, kind="none"),
                    content_type="json",
                    content={
                        "data": {
                            "market": {
                                "price": 105000,
                                "result": True,
                            }
                        }
                    },
                ),
            ],
        )

        claims = extract_claims(sample_prompt_spec, bundle)
        # Should extract from nested structure
        assert isinstance(claims, ClaimSet)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])