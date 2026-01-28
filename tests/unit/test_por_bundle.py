"""
Unit tests for PoR bundle building and verification.

Tests:
- Bundle builds with consistent commitments
- verify_por_bundle(bundle, prompt_spec, evidence, trace) returns ok
- Tamper bundle.evidence_root → verify fails with challenge kind evidence/por
- Tamper embedded verdict.outcome but not verdict_hash → verify fails on verdict_hash mismatch
- Hash format validation (non-0x should fail)
- PromptSpec timestamp validation (created_at must be None)
"""

from datetime import datetime, timezone

import pytest

from core.por.por_bundle import PoRBundle, TEEAttestation
from core.por.proof_of_reasoning import (
    PromptSpecTimestampError,
    build_por_bundle,
    compute_evidence_root,
    compute_por_root,
    compute_prompt_spec_hash,
    compute_reasoning_root,
    compute_roots,
    compute_verdict_hash,
    verify_por_bundle,
    verify_por_bundle_structure,
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
    return MarketSpec(
        market_id="market_test_001",
        question="Will event X occur?",
        event_definition="Event X is defined as...",
        timezone="UTC",
        resolution_deadline=datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(
            rules=[ResolutionRule(rule_id="r1", description="Primary rule", priority=1)]
        ),
        allowed_sources=[SourcePolicy(source_id="src1", kind="api", allow=True)],
        min_provenance_tier=0,
        dispute_policy=DisputePolicy(dispute_window_seconds=3600, allow_challenges=True, extra={}),
        metadata={},
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
            target_entity="Entity X",
            predicate="occurs",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="dr1",
                description="Data requirement 1",
                source_targets=[
                    SourceTarget(
                        source_id="src1",
                        uri="https://api.example.com/data",
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
        forbidden_behaviors=["hallucinate"],
        created_at=None,  # Must be None for deterministic commitments
        extra={},
    )


@pytest.fixture
def sample_evidence_bundle() -> EvidenceBundle:
    """Create a sample EvidenceBundle for testing."""
    now = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return EvidenceBundle(
        bundle_id="bundle_test_001",
        collector_id="collector_test",
        collection_time=now,
        items=[
            EvidenceItem(
                evidence_id="ev_test_001",
                source=SourceDescriptor(source_id="src1", uri="https://example.com/data"),
                retrieval=RetrievalReceipt(retrieved_at=now, method="http_get", extra={}),
                provenance=ProvenanceProof(tier=1, kind="signature", extra={}),
                content_type="json",
                content={"value": 42},
                confidence=0.9,
            ),
            EvidenceItem(
                evidence_id="ev_test_002",
                source=SourceDescriptor(source_id="src1", uri="https://example.com/data2"),
                retrieval=RetrievalReceipt(retrieved_at=now, method="http_get", extra={}),
                provenance=ProvenanceProof(tier=1, kind="signature", extra={}),
                content_type="json",
                content={"value": 100},
                confidence=0.85,
            ),
        ],
        provenance_summary={"min_tier": 1},
    )


@pytest.fixture
def sample_reasoning_trace() -> ReasoningTrace:
    """Create a sample ReasoningTrace for testing."""
    return ReasoningTrace(
        trace_id="trace_test_001",
        policy=TracePolicy(decoding_policy="strict", max_steps=50, extra={}),
        steps=[
            ReasoningStep(
                step_id="step_001",
                type="extract",
                inputs={"ev": "ev_test_001"},
                action="Extract value",
                output={"extracted": 42},
                evidence_ids=["ev_test_001"],
                prior_step_ids=[],
            ),
            ReasoningStep(
                step_id="step_002",
                type="check",
                inputs={"value": 42, "threshold": 50},
                action="Compare values",
                output={"result": False},
                evidence_ids=[],
                prior_step_ids=["step_001"],
            ),
        ],
        evidence_refs=["ev_test_001", "ev_test_002"],
    )


@pytest.fixture
def sample_verdict() -> DeterministicVerdict:
    """Create a sample DeterministicVerdict for testing."""
    return DeterministicVerdict(
        market_id="market_test_001",
        outcome="NO",
        confidence=0.9,
        resolution_time=datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
        resolution_rule_id="r1",
        metadata={},
    )


@pytest.fixture
def sample_por_bundle(
    sample_prompt_spec: PromptSpec,
    sample_evidence_bundle: EvidenceBundle,
    sample_reasoning_trace: ReasoningTrace,
    sample_verdict: DeterministicVerdict,
) -> PoRBundle:
    """Create a valid PoR bundle using build_por_bundle."""
    return build_por_bundle(
        sample_prompt_spec,
        sample_evidence_bundle,
        sample_reasoning_trace,
        sample_verdict,
    )


# =============================================================================
# Test: Bundle Building
# =============================================================================


class TestBundleBuilding:
    """Test PoR bundle construction."""

    def test_build_por_bundle_creates_valid_bundle(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle creates a bundle with all required fields."""
        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
        )

        assert isinstance(bundle, PoRBundle)
        assert bundle.market_id == sample_verdict.market_id
        assert bundle.verdict == sample_verdict
        assert bundle.prompt_spec_hash.startswith("0x")
        assert bundle.evidence_root.startswith("0x")
        assert bundle.reasoning_root.startswith("0x")
        assert bundle.verdict_hash.startswith("0x")
        assert bundle.por_root is not None
        assert bundle.por_root.startswith("0x")

    def test_build_por_bundle_commitments_are_correct(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle computes correct commitments."""
        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
        )

        # Verify each commitment matches independent computation
        assert bundle.prompt_spec_hash == compute_prompt_spec_hash(sample_prompt_spec)
        assert bundle.evidence_root == compute_evidence_root(sample_evidence_bundle)
        assert bundle.reasoning_root == compute_reasoning_root(sample_reasoning_trace)
        assert bundle.verdict_hash == compute_verdict_hash(sample_verdict)

        expected_por_root = compute_por_root(
            bundle.prompt_spec_hash,
            bundle.evidence_root,
            bundle.reasoning_root,
            bundle.verdict_hash,
        )
        assert bundle.por_root == expected_por_root

    def test_build_por_bundle_without_por_root(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle can exclude por_root."""
        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
            include_por_root=False,
        )

        assert bundle.por_root is None
        # Other commitments should still be present
        assert bundle.prompt_spec_hash is not None
        assert bundle.evidence_root is not None

    def test_build_por_bundle_with_tee_attestation(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle can include TEE attestation."""
        attestation = TEEAttestation(
            attestation_type="sgx",
            quote="base64_quote_data",
            enclave_measurement="0x" + "aa" * 32,
            timestamp=datetime.now(timezone.utc),
            extra={},
        )

        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
            tee_attestation=attestation,
        )

        assert bundle.tee_attestation == attestation

    def test_build_por_bundle_with_signatures(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle can include signatures."""
        signatures = {
            "collector": "sig_collector_xyz",
            "auditor": "sig_auditor_abc",
        }

        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
            signatures=signatures,
        )

        assert bundle.signatures == signatures

    def test_build_por_bundle_with_metadata(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle can include metadata."""
        metadata = {"version": "1.0", "pipeline_id": "pipe_123"}

        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
            metadata=metadata,
        )

        assert bundle.metadata == metadata


# =============================================================================
# Test: Bundle Verification - Valid Cases
# =============================================================================


class TestBundleVerificationValid:
    """Test bundle verification for valid bundles."""

    def test_verify_valid_bundle_structure(self, sample_por_bundle: PoRBundle):
        """verify_por_bundle_structure passes for valid bundle."""
        result = verify_por_bundle_structure(sample_por_bundle)

        assert result.ok is True
        assert result.challenge is None
        assert len(result.checks) > 0
        assert all(c.ok for c in result.checks)

    def test_verify_valid_bundle_full(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """verify_por_bundle passes with all artifacts provided."""
        result = verify_por_bundle(
            sample_por_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is True
        assert result.challenge is None
        # Should have checks for structure + all commitment verifications
        check_ids = {c.check_id for c in result.checks}
        assert "prompt_spec_hash_match" in check_ids
        assert "evidence_root_match" in check_ids
        assert "reasoning_root_match" in check_ids
        assert "verdict_hash_match" in check_ids

    def test_verify_valid_bundle_partial(self, sample_por_bundle: PoRBundle):
        """verify_por_bundle passes with no artifacts (structure only + verdict hash)."""
        result = verify_por_bundle(sample_por_bundle)

        assert result.ok is True
        assert result.challenge is None

    def test_verify_valid_bundle_with_evidence_only(
        self,
        sample_por_bundle: PoRBundle,
        sample_evidence_bundle: EvidenceBundle,
    ):
        """verify_por_bundle passes with only evidence provided."""
        result = verify_por_bundle(
            sample_por_bundle,
            evidence=sample_evidence_bundle,
        )

        assert result.ok is True
        check_ids = {c.check_id for c in result.checks}
        assert "evidence_root_match" in check_ids


# =============================================================================
# Test: Bundle Verification - Tamper Detection
# =============================================================================


class TestBundleVerificationTamperDetection:
    """Test bundle verification detects tampering."""

    def test_tampered_evidence_root_detected(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Tampered evidence_root is detected."""
        # Tamper the evidence root
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.evidence_root = "0x" + "ff" * 32

        result = verify_por_bundle(
            tampered_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "evidence_leaf"

    def test_tampered_reasoning_root_detected(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Tampered reasoning_root is detected."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.reasoning_root = "0x" + "ee" * 32

        result = verify_por_bundle(
            tampered_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "reasoning_leaf"

    def test_tampered_prompt_spec_hash_detected(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Tampered prompt_spec_hash is detected."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.prompt_spec_hash = "0x" + "dd" * 32

        result = verify_por_bundle(
            tampered_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "por_bundle"

    def test_tampered_verdict_outcome_detected(self, sample_por_bundle: PoRBundle):
        """Tampered verdict.outcome (without updating verdict_hash) is detected."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        # Change the verdict outcome but keep the old verdict_hash
        tampered_bundle.verdict.outcome = "YES"  # Was "NO"

        result = verify_por_bundle(tampered_bundle)

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "verdict_hash"

    def test_tampered_verdict_confidence_detected(self, sample_por_bundle: PoRBundle):
        """Tampered verdict.confidence (without updating verdict_hash) is detected."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.verdict.confidence = 0.1  # Was 0.9

        result = verify_por_bundle(tampered_bundle)

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "verdict_hash"

    def test_tampered_por_root_detected(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Tampered por_root is detected."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.por_root = "0x" + "cc" * 32

        result = verify_por_bundle(
            tampered_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "por_bundle"

    def test_internally_inconsistent_por_root_detected(self, sample_por_bundle: PoRBundle):
        """Internally inconsistent por_root (doesn't match other commitments) is detected."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.por_root = "0x" + "bb" * 32

        # Verify without full artifacts - should check internal consistency
        result = verify_por_bundle(tampered_bundle)

        assert result.ok is False
        assert result.challenge is not None
        assert result.challenge.kind == "por_bundle"


# =============================================================================
# Test: Hash Format Validation
# =============================================================================


class TestHashFormatValidation:
    """Test hash format validation in bundles."""

    def test_invalid_hash_no_prefix_rejected(self, sample_verdict: DeterministicVerdict):
        """Hash without 0x prefix is rejected."""
        with pytest.raises(ValueError):
            PoRBundle(
                market_id="market_test_001",
                prompt_spec_hash="aa" * 32,  # Missing 0x
                evidence_root="0x" + "bb" * 32,
                reasoning_root="0x" + "cc" * 32,
                verdict_hash="0x" + "dd" * 32,
                verdict=sample_verdict,
                created_at=datetime.now(timezone.utc),
            )

    def test_invalid_hash_wrong_length_rejected(self, sample_verdict: DeterministicVerdict):
        """Hash with wrong length is rejected."""
        with pytest.raises(ValueError):
            PoRBundle(
                market_id="market_test_001",
                prompt_spec_hash="0x" + "aa" * 16,  # Only 16 bytes
                evidence_root="0x" + "bb" * 32,
                reasoning_root="0x" + "cc" * 32,
                verdict_hash="0x" + "dd" * 32,
                verdict=sample_verdict,
                created_at=datetime.now(timezone.utc),
            )

    def test_invalid_hash_non_hex_rejected(self, sample_verdict: DeterministicVerdict):
        """Hash with non-hex characters is rejected."""
        with pytest.raises(ValueError):
            PoRBundle(
                market_id="market_test_001",
                prompt_spec_hash="0x" + "gg" * 32,  # Invalid hex
                evidence_root="0x" + "bb" * 32,
                reasoning_root="0x" + "cc" * 32,
                verdict_hash="0x" + "dd" * 32,
                verdict=sample_verdict,
                created_at=datetime.now(timezone.utc),
            )

    def test_hash_format_check_in_verification(self):
        """verify_por_bundle_structure checks hash formats."""
        # Create a bundle with valid hashes first, then manually corrupt
        verdict = DeterministicVerdict(
            market_id="test",
            outcome="YES",
            confidence=0.9,
            resolution_time=datetime.now(timezone.utc),
            resolution_rule_id="r1",
            metadata={},
        )

        # This should work - but we want to test the verification catches bad formats
        # We need to bypass Pydantic validation to test the verification layer
        bundle = PoRBundle(
            market_id="test",
            prompt_spec_hash="0x" + "aa" * 32,
            evidence_root="0x" + "bb" * 32,
            reasoning_root="0x" + "cc" * 32,
            verdict_hash="0x" + "dd" * 32,
            verdict=verdict,
            created_at=datetime.now(timezone.utc),
        )

        result = verify_por_bundle_structure(bundle)

        # Should pass since all hashes are valid format
        assert result.ok is True
        # Check that format checks were performed
        check_ids = {c.check_id for c in result.checks}
        assert "hash_format_prompt_spec_hash" in check_ids
        assert "hash_format_evidence_root" in check_ids
        assert "hash_format_reasoning_root" in check_ids
        assert "hash_format_verdict_hash" in check_ids


# =============================================================================
# Test: Market ID Consistency
# =============================================================================


class TestMarketIdConsistency:
    """Test market_id consistency validation."""

    def test_mismatched_market_id_rejected(self):
        """Bundle with mismatched market_id is rejected at construction."""
        verdict = DeterministicVerdict(
            market_id="market_A",
            outcome="YES",
            confidence=0.9,
            resolution_time=datetime.now(timezone.utc),
            resolution_rule_id="r1",
            metadata={},
        )

        with pytest.raises(ValueError, match="market_id"):
            PoRBundle(
                market_id="market_B",  # Mismatch!
                prompt_spec_hash="0x" + "aa" * 32,
                evidence_root="0x" + "bb" * 32,
                reasoning_root="0x" + "cc" * 32,
                verdict_hash="0x" + "dd" * 32,
                verdict=verdict,
                created_at=datetime.now(timezone.utc),
            )

    def test_matching_market_id_accepted(self):
        """Bundle with matching market_id is accepted."""
        verdict = DeterministicVerdict(
            market_id="market_same",
            outcome="YES",
            confidence=0.9,
            resolution_time=datetime.now(timezone.utc),
            resolution_rule_id="r1",
            metadata={},
        )

        bundle = PoRBundle(
            market_id="market_same",
            prompt_spec_hash="0x" + "aa" * 32,
            evidence_root="0x" + "bb" * 32,
            reasoning_root="0x" + "cc" * 32,
            verdict_hash="0x" + "dd" * 32,
            verdict=verdict,
            created_at=datetime.now(timezone.utc),
        )

        assert bundle.market_id == bundle.verdict.market_id


# =============================================================================
# Test: Verification Result Structure
# =============================================================================


class TestVerificationResultStructure:
    """Test structure of verification results."""

    def test_verification_result_has_all_checks(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Verification result includes all expected checks."""
        result = verify_por_bundle(
            sample_por_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        check_ids = {c.check_id for c in result.checks}

        # Structure checks
        assert "hash_format_prompt_spec_hash" in check_ids
        assert "hash_format_evidence_root" in check_ids
        assert "hash_format_reasoning_root" in check_ids
        assert "hash_format_verdict_hash" in check_ids
        assert "market_id_match" in check_ids

        # Commitment checks
        assert "prompt_spec_hash_match" in check_ids
        assert "evidence_root_match" in check_ids
        assert "reasoning_root_match" in check_ids
        assert "verdict_hash_match" in check_ids

    def test_failed_check_has_error_severity(self, sample_por_bundle: PoRBundle):
        """Failed checks have 'error' severity."""
        tampered_bundle = sample_por_bundle.model_copy(deep=True)
        tampered_bundle.verdict.outcome = "YES"  # Tamper

        result = verify_por_bundle(tampered_bundle)

        failed_checks = [c for c in result.checks if not c.ok]
        assert len(failed_checks) > 0
        for check in failed_checks:
            assert check.severity == "error"

    def test_passed_check_has_info_severity(self, sample_por_bundle: PoRBundle):
        """Passed checks have 'info' severity."""
        result = verify_por_bundle(sample_por_bundle)

        passed_checks = [c for c in result.checks if c.ok]
        assert len(passed_checks) > 0
        for check in passed_checks:
            assert check.severity == "info"

    def test_check_details_contain_expected_and_computed(
        self,
        sample_por_bundle: PoRBundle,
        sample_evidence_bundle: EvidenceBundle,
    ):
        """Commitment check details include expected and computed values."""
        result = verify_por_bundle(
            sample_por_bundle,
            evidence=sample_evidence_bundle,
        )

        evidence_check = next(c for c in result.checks if c.check_id == "evidence_root_match")
        assert "expected" in evidence_check.details
        assert "computed" in evidence_check.details


# =============================================================================
# Test: PromptSpec Timestamp Validation in Verification
# =============================================================================


class TestPromptSpecTimestampValidation:
    """Test PromptSpec.created_at timestamp validation in bundle verification."""

    def test_verify_bundle_with_valid_prompt_spec_timestamp(
        self,
        sample_por_bundle: PoRBundle,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Verification passes when PromptSpec.created_at is None."""
        assert sample_prompt_spec.created_at is None

        result = verify_por_bundle(
            sample_por_bundle,
            prompt_spec=sample_prompt_spec,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is True
        # Check that timestamp check was performed and passed
        check_ids = {c.check_id for c in result.checks}
        assert "prompt_spec_timestamp_check" in check_ids
        timestamp_check = next(c for c in result.checks if c.check_id == "prompt_spec_timestamp_check")
        assert timestamp_check.ok is True

    def test_verify_bundle_with_invalid_prompt_spec_timestamp(
        self,
        sample_por_bundle: PoRBundle,
        sample_market_spec: MarketSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
    ):
        """Verification fails when PromptSpec.created_at is not None."""
        # Create a PromptSpec with a timestamp (violates spec)
        prompt_with_timestamp = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Entity X",
                predicate="occurs",
            ),
            data_requirements=[
                DataRequirement(
                    requirement_id="dr1",
                    description="Data requirement 1",
                    source_targets=[
                        SourceTarget(
                            source_id="src1",
                            uri="https://api.example.com/data",
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
            forbidden_behaviors=["hallucinate"],
            created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # Non-None!
            extra={},
        )

        result = verify_por_bundle(
            sample_por_bundle,
            prompt_spec=prompt_with_timestamp,
            evidence=sample_evidence_bundle,
            trace=sample_reasoning_trace,
        )

        assert result.ok is False
        # Check that timestamp check failed
        timestamp_check = next(c for c in result.checks if c.check_id == "prompt_spec_timestamp_check")
        assert timestamp_check.ok is False
        assert "created_at" in timestamp_check.details
        # Challenge should reference the timestamp violation
        assert result.challenge is not None
        assert "timestamp" in result.challenge.reason.lower() or "created_at" in result.challenge.reason.lower()

    def test_build_por_bundle_with_valid_prompt_spec(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle works with PromptSpec.created_at=None."""
        assert sample_prompt_spec.created_at is None

        # Should not raise
        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
        )

        assert bundle is not None
        assert bundle.prompt_spec_hash.startswith("0x")

    def test_build_por_bundle_with_invalid_prompt_spec_raises(
        self,
        sample_market_spec: MarketSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """build_por_bundle raises when PromptSpec.created_at is not None."""
        prompt_with_timestamp = PromptSpec(
            task_type="prediction_resolution",
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Entity X",
                predicate="occurs",
            ),
            data_requirements=[
                DataRequirement(
                    requirement_id="dr1",
                    description="Data requirement 1",
                    source_targets=[
                        SourceTarget(
                            source_id="src1",
                            uri="https://api.example.com/data",
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
            forbidden_behaviors=["hallucinate"],
            created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # Non-None!
            extra={},
        )

        with pytest.raises(PromptSpecTimestampError):
            build_por_bundle(
                prompt_with_timestamp,
                sample_evidence_bundle,
                sample_reasoning_trace,
                sample_verdict,
            )

    def test_timestamp_in_bundle_metadata_is_allowed(
        self,
        sample_prompt_spec: PromptSpec,
        sample_evidence_bundle: EvidenceBundle,
        sample_reasoning_trace: ReasoningTrace,
        sample_verdict: DeterministicVerdict,
    ):
        """Timestamps in PoRBundle.metadata are allowed (not committed)."""
        bundle = build_por_bundle(
            sample_prompt_spec,
            sample_evidence_bundle,
            sample_reasoning_trace,
            sample_verdict,
            metadata={
                "operational_timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_start": "2026-01-15T10:00:00Z",
            },
        )

        assert bundle.metadata is not None
        assert "operational_timestamp" in bundle.metadata
        # Bundle should still be valid
        result = verify_por_bundle(bundle)
        assert result.ok is True