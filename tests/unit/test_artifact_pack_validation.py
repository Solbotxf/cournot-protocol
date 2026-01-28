"""
Module 09B - Artifact Pack Validation Tests

Tests for semantic validation:
1. Mismatched market_id across files → validation fails
2. Changed verdict without updating por_bundle → validation fails
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.crypto.hashing import hash_canonical, to_hex
from core.por.por_bundle import PoRBundle
from core.por.reasoning_trace import ReasoningTrace, ReasoningStep, TracePolicy
from core.por.proof_of_reasoning import compute_roots, compute_verdict_hash
from core.schemas.evidence import (
    EvidenceBundle, EvidenceItem, SourceDescriptor,
    RetrievalReceipt, ProvenanceProof,
)
from core.schemas.market import (
    MarketSpec, ResolutionWindow, ResolutionRules, ResolutionRule, DisputePolicy,
)
from core.schemas.prompts import (
    PromptSpec, PredictionSemantics, DataRequirement,
    SourceTarget, SelectionPolicy,
)
from core.schemas.transport import ToolPlan
from core.schemas.verdict import DeterministicVerdict

from orchestrator.pipeline import PoRPackage
from orchestrator.artifacts.pack import (
    validate_pack,
    validate_market_id_consistency,
    validate_verdict_hash,
    validate_prompt_spec_hash,
    validate_evidence_root,
    validate_reasoning_root,
    validate_por_root,
    validate_embedded_verdict,
    validate_evidence_references,
    validate_manifest_consistency,
)
from orchestrator.artifacts.manifest import PackManifest


# =============================================================================
# Test Fixtures
# =============================================================================

def make_market_spec(market_id: str = "mkt_test_val") -> MarketSpec:
    return MarketSpec(
        market_id=market_id,
        question="Will BTC close above $100,000?",
        event_definition="BTC-USD daily close > 100000",
        resolution_deadline=datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(rules=[
            ResolutionRule(rule_id="R_BINARY", description="Binary decision", priority=1),
        ]),
        dispute_policy=DisputePolicy(dispute_window_seconds=3600),
    )


def make_prompt_spec(market_id: str = "mkt_test_val") -> PromptSpec:
    return PromptSpec(
        market=make_market_spec(market_id),
        prediction_semantics=PredictionSemantics(
            target_entity="BTC-USD",
            predicate="close_price > threshold",
            threshold="100000",
            timeframe="2026-12-31",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="BTC price",
                source_targets=[
                    SourceTarget(
                        source_id="exchange",
                        uri="https://api.exchange.example.com/btc-usd",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best", min_sources=1, max_sources=1, quorum=1,
                ),
            ),
        ],
        created_at=None,
        extra={"strict_mode": True},
    )


def make_tool_plan(market_id: str = "mkt_test_val") -> ToolPlan:
    return ToolPlan(
        plan_id=f"plan_{market_id}",
        requirements=["req_001"],
        sources=["exchange"],
        min_provenance_tier=0,
    )


def make_evidence_bundle(market_id: str = "mkt_test_val") -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=f"bundle_{market_id}",
        collector_id="collector_v1",
        collection_time=None,
        items=[
            EvidenceItem(
                evidence_id="ev_001",
                requirement_id="req_001",
                source=SourceDescriptor(
                    source_id="exchange",
                    uri="https://api.exchange.example.com/btc-usd",
                    provider="Example Exchange",
                ),
                retrieval=RetrievalReceipt(
                    retrieved_at=None,
                    method="http_get",
                    request_fingerprint="0x" + "a1" * 32,
                    response_fingerprint="0x" + "b2" * 32,
                ),
                provenance=ProvenanceProof(tier=1, kind="signature"),
                content_type="json",
                content={"price": 105000.0},
                confidence=0.95,
            ),
        ],
    )


def make_reasoning_trace(market_id: str = "mkt_test_val") -> ReasoningTrace:
    return ReasoningTrace(
        trace_id=f"trace_{market_id}",
        policy=TracePolicy(max_steps=100),
        steps=[
            ReasoningStep(
                step_id="step_0001",
                type="extract",
                action="Extract price",
                inputs={"evidence_id": "ev_001"},
                output={"price": 105000.0},
                evidence_ids=["ev_001"],
            ),
        ],
        evidence_refs=["ev_001"],
    )


def make_verdict(market_id: str = "mkt_test_val", outcome: str = "YES") -> DeterministicVerdict:
    return DeterministicVerdict(
        market_id=market_id,
        outcome=outcome,
        confidence=0.85,
        resolution_time=None,
        resolution_rule_id="R_BINARY",
    )


def make_valid_package(market_id: str = "mkt_test_val") -> PoRPackage:
    """Create a valid, consistent PoRPackage."""
    prompt_spec = make_prompt_spec(market_id)
    tool_plan = make_tool_plan(market_id)
    evidence = make_evidence_bundle(market_id)
    trace = make_reasoning_trace(market_id)
    verdict = make_verdict(market_id)
    
    roots = compute_roots(prompt_spec, evidence, trace, verdict)
    
    bundle = PoRBundle(
        market_id=market_id,
        prompt_spec_hash=roots.prompt_spec_hash,
        evidence_root=roots.evidence_root,
        reasoning_root=roots.reasoning_root,
        verdict_hash=roots.verdict_hash,
        por_root=roots.por_root,
        verdict=verdict,
        created_at=datetime.now(timezone.utc),
    )
    
    return PoRPackage(
        bundle=bundle,
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence=evidence,
        trace=trace,
        verdict=verdict,
    )


# =============================================================================
# Valid Package Tests
# =============================================================================

class TestValidPackage:
    """Test validation of a valid package."""
    
    def test_valid_package_passes_all_checks(self):
        """A properly constructed package passes all validation checks."""
        package = make_valid_package()
        result = validate_pack(package)
        
        assert result.ok is True
        assert all(c.ok for c in result.checks)
        assert result.challenge is None
    
    def test_valid_package_market_id_consistency(self):
        """Valid package passes market_id consistency check."""
        package = make_valid_package()
        checks = validate_market_id_consistency(package)
        
        assert all(c.ok for c in checks)
    
    def test_valid_package_verdict_hash(self):
        """Valid package passes verdict hash check."""
        package = make_valid_package()
        checks = validate_verdict_hash(package)
        
        assert all(c.ok for c in checks)
    
    def test_valid_package_evidence_references(self):
        """Valid package passes evidence reference check."""
        package = make_valid_package()
        checks = validate_evidence_references(package)
        
        assert all(c.ok for c in checks)


# =============================================================================
# Market ID Mismatch Tests
# =============================================================================

class TestMarketIdMismatch:
    """Test detection of market_id mismatches."""
    
    def test_verdict_market_id_mismatch(self):
        """Detect mismatch between prompt_spec and verdict market_id."""
        package = make_valid_package("mkt_original")
        
        # Create verdict with different market_id
        wrong_verdict = make_verdict("mkt_different")
        
        # Recompute roots with wrong verdict
        roots = compute_roots(
            package.prompt_spec, package.evidence, package.trace, wrong_verdict
        )
        
        # Create bundle with wrong market_id
        wrong_bundle = PoRBundle(
            market_id="mkt_different",
            prompt_spec_hash=roots.prompt_spec_hash,
            evidence_root=roots.evidence_root,
            reasoning_root=roots.reasoning_root,
            verdict_hash=roots.verdict_hash,
            por_root=roots.por_root,
            verdict=wrong_verdict,
            created_at=datetime.now(timezone.utc),
        )
        
        bad_package = PoRPackage(
            bundle=wrong_bundle,
            prompt_spec=package.prompt_spec,  # Original market_id
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=wrong_verdict,  # Different market_id
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("market_id" in c.check_id for c in failed)
    
    def test_bundle_market_id_mismatch(self):
        """Detect mismatch between prompt_spec and bundle market_id.
        
        Note: PoRBundle enforces bundle.market_id == bundle.verdict.market_id via
        Pydantic validator, so we test prompt_spec mismatch instead.
        """
        # Create package with different market_id in prompt_spec vs verdict/bundle
        prompt_spec = make_prompt_spec("mkt_prompt")  # Different market_id
        tool_plan = make_tool_plan("mkt_prompt")
        evidence = make_evidence_bundle("mkt_bundle")
        trace = make_reasoning_trace("mkt_bundle")
        verdict = make_verdict("mkt_bundle")  # Different market_id
        
        roots = compute_roots(prompt_spec, evidence, trace, verdict)
        
        bundle = PoRBundle(
            market_id="mkt_bundle",  # Matches verdict
            prompt_spec_hash=roots.prompt_spec_hash,
            evidence_root=roots.evidence_root,
            reasoning_root=roots.reasoning_root,
            verdict_hash=roots.verdict_hash,
            por_root=roots.por_root,
            verdict=verdict,
            created_at=datetime.now(timezone.utc),
        )
        
        bad_package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,  # Has "mkt_prompt" market_id
            tool_plan=tool_plan,
            evidence=evidence,
            trace=trace,
            verdict=verdict,  # Has "mkt_bundle" market_id
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("market_id" in c.check_id for c in failed)


# =============================================================================
# Commitment Mismatch Tests
# =============================================================================

class TestCommitmentMismatch:
    """Test detection of commitment/hash mismatches."""
    
    def test_verdict_hash_mismatch(self):
        """Detect mismatch when verdict changed but bundle.verdict_hash not updated."""
        package = make_valid_package()
        
        # Change verdict outcome without updating bundle
        modified_verdict = DeterministicVerdict(
            market_id=package.verdict.market_id,
            outcome="NO",  # Changed from YES
            confidence=package.verdict.confidence,
            resolution_time=None,
            resolution_rule_id=package.verdict.resolution_rule_id,
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,  # Still has old verdict_hash
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=modified_verdict,  # Changed verdict
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("verdict_hash" in c.check_id for c in failed)
    
    def test_evidence_root_mismatch(self):
        """Detect mismatch when evidence changed but bundle.evidence_root not updated."""
        package = make_valid_package()
        
        # Create modified evidence bundle
        modified_evidence = EvidenceBundle(
            bundle_id=package.evidence.bundle_id,
            collector_id=package.evidence.collector_id,
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    requirement_id="req_001",
                    source=package.evidence.items[0].source,
                    retrieval=package.evidence.items[0].retrieval,
                    provenance=package.evidence.items[0].provenance,
                    content_type="json",
                    content={"price": 999999.0},  # Different content!
                    confidence=0.95,
                ),
            ],
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,  # Still has old evidence_root
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=modified_evidence,  # Changed evidence
            trace=package.trace,
            verdict=package.verdict,
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("evidence_root" in c.check_id for c in failed)
    
    def test_reasoning_root_mismatch(self):
        """Detect mismatch when trace changed but bundle.reasoning_root not updated."""
        package = make_valid_package()
        
        # Create modified trace
        modified_trace = ReasoningTrace(
            trace_id=package.trace.trace_id,
            policy=package.trace.policy,
            steps=[
                ReasoningStep(
                    step_id="step_0001",
                    type="extract",
                    action="Different action!",  # Changed
                    inputs={"evidence_id": "ev_001"},
                    output={"price": 999999.0},  # Changed
                    evidence_ids=["ev_001"],
                ),
            ],
            evidence_refs=["ev_001"],
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=modified_trace,  # Changed trace
            verdict=package.verdict,
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("reasoning_root" in c.check_id for c in failed)
    
    def test_prompt_spec_hash_mismatch(self):
        """Detect mismatch when prompt_spec changed but hash not updated."""
        package = make_valid_package()
        
        # Create modified prompt_spec
        modified_prompt = PromptSpec(
            market=package.prompt_spec.market,
            prediction_semantics=PredictionSemantics(
                target_entity="ETH-USD",  # Changed!
                predicate="close_price > threshold",
                threshold="5000",  # Changed!
                timeframe="2026-12-31",
            ),
            data_requirements=package.prompt_spec.data_requirements,
            created_at=None,
            extra={"strict_mode": True},
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,  # Still has old prompt_spec_hash
            prompt_spec=modified_prompt,  # Changed
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=package.verdict,
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("prompt_spec_hash" in c.check_id for c in failed)
    
    def test_por_root_mismatch(self):
        """Detect mismatch when por_root doesn't match computed root."""
        package = make_valid_package()
        
        # Create bundle with wrong por_root
        wrong_bundle = PoRBundle(
            market_id=package.bundle.market_id,
            prompt_spec_hash=package.bundle.prompt_spec_hash,
            evidence_root=package.bundle.evidence_root,
            reasoning_root=package.bundle.reasoning_root,
            verdict_hash=package.bundle.verdict_hash,
            por_root="0x" + "ff" * 32,  # Wrong!
            verdict=package.bundle.verdict,
            created_at=datetime.now(timezone.utc),
        )
        
        bad_package = PoRPackage(
            bundle=wrong_bundle,
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=package.verdict,
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("por_root" in c.check_id for c in failed)


# =============================================================================
# Evidence Reference Tests
# =============================================================================

class TestEvidenceReferences:
    """Test validation of evidence references in trace."""
    
    def test_invalid_evidence_reference_detected(self):
        """Detect when trace references non-existent evidence."""
        package = make_valid_package()
        
        # Create trace with invalid evidence reference
        bad_trace = ReasoningTrace(
            trace_id=package.trace.trace_id,
            policy=package.trace.policy,
            steps=[
                ReasoningStep(
                    step_id="step_0001",
                    type="extract",
                    action="Extract price",
                    inputs={"evidence_id": "ev_nonexistent"},  # Bad reference!
                    output={"price": 105000.0},
                    evidence_ids=["ev_nonexistent"],  # Bad reference!
                ),
            ],
            evidence_refs=["ev_nonexistent"],  # Bad reference!
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=bad_trace,
            verdict=package.verdict,
        )
        
        checks = validate_evidence_references(bad_package)
        
        assert not all(c.ok for c in checks)
        failed = [c for c in checks if not c.ok]
        assert any("evidence_references" in c.check_id for c in failed)


# =============================================================================
# Embedded Verdict Tests
# =============================================================================

class TestEmbeddedVerdict:
    """Test validation of embedded verdict in bundle."""
    
    def test_embedded_verdict_mismatch(self):
        """Detect when bundle.verdict differs from standalone verdict."""
        package = make_valid_package()
        
        # Create a different standalone verdict
        different_verdict = DeterministicVerdict(
            market_id=package.verdict.market_id,
            outcome="NO",  # Different!
            confidence=0.5,  # Different!
            resolution_time=None,
            resolution_rule_id=package.verdict.resolution_rule_id,
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,  # Has original verdict embedded
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=different_verdict,  # Different from embedded
        )
        
        checks = validate_embedded_verdict(bad_package)
        
        assert not all(c.ok for c in checks)


# =============================================================================
# Manifest Consistency Tests
# =============================================================================

class TestManifestConsistency:
    """Test validation of manifest against package."""
    
    def test_manifest_market_id_mismatch(self):
        """Detect when manifest.market_id differs from package."""
        package = make_valid_package("mkt_original")
        
        manifest = PackManifest(
            market_id="mkt_different",  # Wrong!
            por_root=package.bundle.por_root or "",
        )
        
        result = validate_manifest_consistency(manifest, package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("market_id" in c.check_id for c in failed)
    
    def test_manifest_por_root_mismatch(self):
        """Detect when manifest.por_root differs from bundle."""
        package = make_valid_package()
        
        manifest = PackManifest(
            market_id=package.bundle.market_id,
            por_root="0x" + "ff" * 32,  # Wrong!
        )
        
        result = validate_manifest_consistency(manifest, package)
        
        assert result.ok is False
        failed = [c for c in result.checks if not c.ok]
        assert any("por_root" in c.check_id for c in failed)


# =============================================================================
# Timestamp Tests
# =============================================================================

class TestTimestampValidation:
    """Test validation of timestamp requirements."""
    
    def test_prompt_spec_with_timestamp_fails(self):
        """Detect when prompt_spec.created_at is not None."""
        package = make_valid_package()
        
        # Create prompt_spec with timestamp
        bad_prompt = PromptSpec(
            market=package.prompt_spec.market,
            prediction_semantics=package.prompt_spec.prediction_semantics,
            data_requirements=package.prompt_spec.data_requirements,
            created_at=datetime.now(timezone.utc),  # Not None!
            extra={"strict_mode": True},
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,
            prompt_spec=bad_prompt,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=package.verdict,
        )
        
        checks = validate_prompt_spec_hash(bad_package)
        
        # Should have failed timestamp check
        timestamp_checks = [c for c in checks if "timestamp" in c.check_id]
        assert len(timestamp_checks) > 0
        assert not timestamp_checks[0].ok


# =============================================================================
# Challenge Generation Tests
# =============================================================================

class TestChallengeGeneration:
    """Test that validation generates appropriate challenges."""
    
    def test_evidence_mismatch_generates_evidence_challenge(self):
        """Evidence mismatch should generate evidence_leaf challenge."""
        package = make_valid_package()
        
        modified_evidence = EvidenceBundle(
            bundle_id=package.evidence.bundle_id,
            collector_id=package.evidence.collector_id,
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    requirement_id="req_001",
                    source=package.evidence.items[0].source,
                    retrieval=package.evidence.items[0].retrieval,
                    provenance=package.evidence.items[0].provenance,
                    content_type="json",
                    content={"price": 0.0},  # Changed
                    confidence=0.95,
                ),
            ],
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=modified_evidence,
            trace=package.trace,
            verdict=package.verdict,
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        assert result.challenge is not None
        # First failure is evidence-related
        assert result.challenge.kind == "evidence_leaf"
    
    def test_verdict_mismatch_generates_verdict_challenge(self):
        """Verdict mismatch should generate verdict_hash challenge."""
        package = make_valid_package()
        
        modified_verdict = DeterministicVerdict(
            market_id=package.verdict.market_id,
            outcome="NO",
            confidence=package.verdict.confidence,
            resolution_time=None,
            resolution_rule_id=package.verdict.resolution_rule_id,
        )
        
        bad_package = PoRPackage(
            bundle=package.bundle,
            prompt_spec=package.prompt_spec,
            tool_plan=package.tool_plan,
            evidence=package.evidence,
            trace=package.trace,
            verdict=modified_verdict,
        )
        
        result = validate_pack(bad_package)
        
        assert result.ok is False
        assert result.challenge is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])