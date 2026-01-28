"""
Module 06 - Tests for Trace Policy Verification

Tests:
- Trace referencing unknown evidence_id fails
- Non-monotonic prior_step_ids fails
- Missing final evaluation_variables fails
- Valid traces pass
"""

import pytest
from datetime import datetime, timezone, timedelta

from agents.auditor.verification.trace_policy import (
    TracePolicyVerifier,
    verify_trace_policy,
)
from core.por.reasoning_trace import ReasoningStep, ReasoningTrace, TracePolicy
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
            )
        ],
    )


@pytest.fixture
def sample_evidence():
    """Create a sample EvidenceBundle for testing."""
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
                content={"price": 105000},
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
                content={"price": 105100},
            ),
        ],
    )


@pytest.fixture
def valid_trace(sample_evidence):
    """Create a valid ReasoningTrace for testing."""
    return ReasoningTrace(
        trace_id="trace_valid_001",
        policy=TracePolicy(max_steps=100),
        steps=[
            ReasoningStep(
                step_id="step_0000",
                type="extract",
                inputs={"evidence_id": "ev_001"},
                action="Extract claims from ev_001",
                output={"claims": []},
                evidence_ids=["ev_001"],
                prior_step_ids=[],
            ),
            ReasoningStep(
                step_id="step_0001",
                type="extract",
                inputs={"evidence_id": "ev_002"},
                action="Extract claims from ev_002",
                output={"claims": []},
                evidence_ids=["ev_002"],
                prior_step_ids=[],
            ),
            ReasoningStep(
                step_id="step_0002",
                type="check",
                inputs={},
                action="Check for contradictions",
                output={"has_contradictions": False},
                evidence_ids=["ev_001", "ev_002"],
                prior_step_ids=["step_0000", "step_0001"],
            ),
            ReasoningStep(
                step_id="step_0003",
                type="map",
                inputs={},
                action="Map to evaluation variables",
                output={
                    "evaluation_variables": {
                        "event_observed": True,
                        "conflict_detected": False,
                        "insufficient_evidence": False,
                    }
                },
                evidence_ids=["ev_001", "ev_002"],
                prior_step_ids=["step_0002"],
            ),
        ],
        evidence_refs=["ev_001", "ev_002"],
    )


# =============================================================================
# Test Valid Traces
# =============================================================================


class TestValidTraces:
    """Tests for valid trace verification."""

    def test_valid_trace_passes(self, sample_prompt_spec, sample_evidence, valid_trace):
        """Test that a valid trace passes verification."""
        result = verify_trace_policy(valid_trace, sample_prompt_spec, sample_evidence)

        assert result.ok
        assert result.challenge is None

    def test_valid_trace_with_verifier(self, sample_prompt_spec, sample_evidence, valid_trace):
        """Test verification using TracePolicyVerifier class."""
        verifier = TracePolicyVerifier()
        result = verifier.verify(valid_trace, sample_prompt_spec, sample_evidence)

        assert result.ok
        assert len(result.checks) > 0


# =============================================================================
# Test Unknown Evidence References
# =============================================================================


class TestUnknownEvidenceReferences:
    """Tests for unknown evidence ID detection."""

    def test_unknown_evidence_id_fails(self, sample_prompt_spec, sample_evidence):
        """Test that referencing unknown evidence ID fails."""
        trace = ReasoningTrace(
            trace_id="trace_bad_evidence",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="extract",
                    inputs={},
                    action="Extract",
                    output={},
                    evidence_ids=["ev_UNKNOWN"],  # Unknown evidence ID
                    prior_step_ids=[],
                ),
                ReasoningStep(
                    step_id="step_0001",
                    type="map",
                    inputs={},
                    action="Map",
                    output={"evaluation_variables": {"event_observed": None}},
                    evidence_ids=[],
                    prior_step_ids=["step_0000"],
                ),
            ],
            evidence_refs=["ev_UNKNOWN"],  # Also references unknown
        )

        result = verify_trace_policy(trace, sample_prompt_spec, sample_evidence)

        assert not result.ok
        # Should have a failed evidence_references check
        failed_checks = [c for c in result.checks if not c.ok]
        assert any("evidence" in c.check_id for c in failed_checks)

    def test_evidence_refs_must_exist(self, sample_prompt_spec, sample_evidence):
        """Test that evidence_refs must exist in bundle."""
        trace = ReasoningTrace(
            trace_id="trace_bad_refs",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="map",
                    inputs={},
                    action="Map",
                    output={"evaluation_variables": {"event_observed": None}},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[],
                ),
            ],
            evidence_refs=["ev_001", "ev_MISSING"],  # ev_MISSING doesn't exist
        )

        result = verify_trace_policy(trace, sample_prompt_spec, sample_evidence)

        assert not result.ok


# =============================================================================
# Test Prior Step References
# =============================================================================


class TestPriorStepReferences:
    """Tests for prior step reference validation."""

    def test_non_existent_prior_step_fails(self, sample_prompt_spec, sample_evidence):
        """Test that referencing non-existent prior step fails during construction.
        
        The ReasoningTrace model validator catches this error during object creation.
        """
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            ReasoningTrace(
                trace_id="trace_bad_prior",
                policy=TracePolicy(),
                steps=[
                    ReasoningStep(
                        step_id="step_0000",
                        type="extract",
                        inputs={},
                        action="Extract",
                        output={},
                        evidence_ids=["ev_001"],
                        prior_step_ids=["step_9999"],  # Doesn't exist
                    ),
                    ReasoningStep(
                        step_id="step_0001",
                        type="map",
                        inputs={},
                        action="Map",
                        output={"evaluation_variables": {"event_observed": None}},
                        evidence_ids=[],
                        prior_step_ids=["step_0000"],
                    ),
                ],
                evidence_refs=["ev_001"],
            )
        
        # Verify the error message mentions prior_step_id
        assert "prior_step_id" in str(exc_info.value)

    def test_forward_reference_fails(self, sample_prompt_spec, sample_evidence):
        """Test that forward references (to later steps) fail during construction.
        
        The ReasoningTrace model validator catches this error during object creation.
        """
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            ReasoningTrace(
                trace_id="trace_forward_ref",
                policy=TracePolicy(),
                steps=[
                    ReasoningStep(
                        step_id="step_0000",
                        type="extract",
                        inputs={},
                        action="Extract",
                        output={},
                        evidence_ids=["ev_001"],
                        prior_step_ids=["step_0001"],  # Forward reference!
                    ),
                    ReasoningStep(
                        step_id="step_0001",
                        type="map",
                        inputs={},
                        action="Map",
                        output={"evaluation_variables": {"event_observed": None}},
                        evidence_ids=[],
                        prior_step_ids=[],
                    ),
                ],
                evidence_refs=["ev_001"],
            )
        
        # Verify the error message mentions the issue
        assert "prior_step_id" in str(exc_info.value) or "earlier step" in str(exc_info.value)


# =============================================================================
# Test Evaluation Variables
# =============================================================================


class TestEvaluationVariables:
    """Tests for evaluation_variables requirement."""

    def test_missing_evaluation_variables_fails(self, sample_prompt_spec, sample_evidence):
        """Test that missing evaluation_variables fails."""
        trace = ReasoningTrace(
            trace_id="trace_no_eval",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="extract",
                    inputs={},
                    action="Extract",
                    output={},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[],
                ),
                ReasoningStep(
                    step_id="step_0001",
                    type="map",
                    inputs={},
                    action="Map",
                    output={},  # No evaluation_variables!
                    evidence_ids=[],
                    prior_step_ids=["step_0000"],
                ),
            ],
            evidence_refs=["ev_001"],
        )

        verifier = TracePolicyVerifier(require_evaluation_variables=True)
        result = verifier.verify(trace, sample_prompt_spec, sample_evidence)

        assert not result.ok
        failed_checks = [c for c in result.checks if not c.ok]
        assert any("evaluation" in c.check_id for c in failed_checks)

    def test_evaluation_variables_optional_when_disabled(
        self, sample_prompt_spec, sample_evidence
    ):
        """Test that evaluation_variables check can be disabled."""
        trace = ReasoningTrace(
            trace_id="trace_no_eval_ok",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="extract",
                    inputs={},
                    action="Extract",
                    output={},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[],
                ),
            ],
            evidence_refs=["ev_001"],
        )

        verifier = TracePolicyVerifier(require_evaluation_variables=False)
        result = verifier.verify(trace, sample_prompt_spec, sample_evidence)

        # Should pass even without evaluation_variables
        eval_checks = [c for c in result.checks if "evaluation" in c.check_id]
        assert all(c.ok for c in eval_checks)


# =============================================================================
# Test Step ID Validation
# =============================================================================


class TestStepIdValidation:
    """Tests for step ID validation."""

    def test_duplicate_step_id_fails(self, sample_prompt_spec, sample_evidence):
        """Test that duplicate step IDs fail during construction.
        
        The ReasoningTrace model validator catches this error during object creation.
        """
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            ReasoningTrace(
                trace_id="trace_dup_ids",
                policy=TracePolicy(),
                steps=[
                    ReasoningStep(
                        step_id="step_0000",
                        type="extract",
                        inputs={},
                        action="Extract 1",
                        output={},
                        evidence_ids=["ev_001"],
                        prior_step_ids=[],
                    ),
                    ReasoningStep(
                        step_id="step_0000",  # Duplicate!
                        type="extract",
                        inputs={},
                        action="Extract 2",
                        output={},
                        evidence_ids=["ev_002"],
                        prior_step_ids=[],
                    ),
                ],
                evidence_refs=["ev_001", "ev_002"],
            )
        
        # Verify the error mentions duplicate step_id
        assert "Duplicate step_id" in str(exc_info.value)

    def test_step_id_format_warning(self, sample_prompt_spec, sample_evidence):
        """Test that non-standard step ID format produces warning."""
        trace = ReasoningTrace(
            trace_id="trace_bad_format",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="my_custom_step",  # Non-standard format
                    type="extract",
                    inputs={},
                    action="Extract",
                    output={},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[],
                ),
                ReasoningStep(
                    step_id="another_step",  # Non-standard format
                    type="map",
                    inputs={},
                    action="Map",
                    output={"evaluation_variables": {"event_observed": None}},
                    evidence_ids=[],
                    prior_step_ids=["my_custom_step"],
                ),
            ],
            evidence_refs=["ev_001"],
        )

        verifier = TracePolicyVerifier(strict_step_id_format=True)
        result = verifier.verify(trace, sample_prompt_spec, sample_evidence)

        # Should have format warnings but might still be ok
        format_checks = [c for c in result.checks if "format" in c.check_id]
        assert len(format_checks) > 0


# =============================================================================
# Test Max Steps
# =============================================================================


class TestMaxSteps:
    """Tests for max_steps limit."""

    def test_exceeds_max_steps_fails(self, sample_prompt_spec, sample_evidence):
        """Test that exceeding max_steps fails during construction.
        
        The ReasoningTrace model validator catches this error during object creation.
        """
        from pydantic import ValidationError
        
        # Create a trace with many steps
        steps = []
        for i in range(50):
            steps.append(
                ReasoningStep(
                    step_id=f"step_{i:04d}",
                    type="extract",
                    inputs={},
                    action=f"Step {i}",
                    output={},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[f"step_{i-1:04d}"] if i > 0 else [],
                )
            )

        # Add final map step
        steps.append(
            ReasoningStep(
                step_id=f"step_{50:04d}",
                type="map",
                inputs={},
                action="Map",
                output={"evaluation_variables": {"event_observed": None}},
                evidence_ids=[],
                prior_step_ids=[f"step_{49:04d}"],
            )
        )

        with pytest.raises(ValidationError) as exc_info:
            ReasoningTrace(
                trace_id="trace_many_steps",
                policy=TracePolicy(max_steps=10),  # Only allow 10 steps
                steps=steps,  # But we have 51
                evidence_refs=["ev_001"],
            )
        
        # Verify the error mentions max_steps
        assert "max_steps" in str(exc_info.value)


# =============================================================================
# Test External Sources
# =============================================================================


class TestExternalSources:
    """Tests for external source detection."""

    def test_external_url_when_forbidden_fails(self, sample_prompt_spec, sample_evidence):
        """Test that external_url in inputs fails when policy forbids."""
        trace = ReasoningTrace(
            trace_id="trace_external",
            policy=TracePolicy(allow_external_sources=False),
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="extract",
                    inputs={"external_url": "https://malicious.com"},  # External!
                    action="Fetch from external source",
                    output={},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[],
                ),
                ReasoningStep(
                    step_id="step_0001",
                    type="map",
                    inputs={},
                    action="Map",
                    output={"evaluation_variables": {"event_observed": None}},
                    evidence_ids=[],
                    prior_step_ids=["step_0000"],
                ),
            ],
            evidence_refs=["ev_001"],
        )

        result = verify_trace_policy(trace, sample_prompt_spec, sample_evidence)

        assert not result.ok
        failed_checks = [c for c in result.checks if not c.ok]
        assert any("external" in c.check_id for c in failed_checks)

    def test_external_url_allowed_when_policy_permits(
        self, sample_prompt_spec, sample_evidence
    ):
        """Test that external_url is allowed when policy permits."""
        trace = ReasoningTrace(
            trace_id="trace_external_ok",
            policy=TracePolicy(allow_external_sources=True),  # Allow external
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="extract",
                    inputs={"external_url": "https://external.com"},
                    action="Fetch",
                    output={},
                    evidence_ids=["ev_001"],
                    prior_step_ids=[],
                ),
                ReasoningStep(
                    step_id="step_0001",
                    type="map",
                    inputs={},
                    action="Map",
                    output={"evaluation_variables": {"event_observed": None}},
                    evidence_ids=[],
                    prior_step_ids=["step_0000"],
                ),
            ],
            evidence_refs=["ev_001"],
        )

        result = verify_trace_policy(trace, sample_prompt_spec, sample_evidence)

        # External sources check should pass
        external_checks = [c for c in result.checks if "external" in c.check_id]
        assert all(c.ok for c in external_checks)


# =============================================================================
# Test Challenge References
# =============================================================================


class TestChallengeReferences:
    """Tests for challenge reference generation on failures."""

    def test_challenge_ref_on_failure(self, sample_prompt_spec, sample_evidence):
        """Test that failing verification produces a ChallengeRef."""
        trace = ReasoningTrace(
            trace_id="trace_fail",
            policy=TracePolicy(),
            steps=[
                ReasoningStep(
                    step_id="step_0000",
                    type="extract",
                    inputs={},
                    action="Extract",
                    output={},
                    evidence_ids=["ev_UNKNOWN"],  # Unknown evidence
                    prior_step_ids=[],
                ),
            ],
            evidence_refs=["ev_UNKNOWN"],
        )

        result = verify_trace_policy(trace, sample_prompt_spec, sample_evidence)

        assert not result.ok
        assert result.challenge is not None
        assert result.challenge.kind == "reasoning_leaf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])