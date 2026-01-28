"""
Module 08 - Validator/Sentinel Tests: Sentinel Verification

Tests for SentinelAgent verify-only mode:
- Valid package passes verification
- Tampered evidence_root produces evidence_leaf challenge
- Tampered reasoning_root produces reasoning_leaf challenge  
- Tampered verdict produces verdict_hash challenge
- Tampered por_root produces por_root challenge

Owner: Protocol Verification Engineer
Module ID: M08
"""

import pytest
from typing import Any

from core.por.proof_of_reasoning import (
    compute_evidence_root,
    compute_reasoning_root,
    compute_verdict_hash,
    compute_por_root,
)
from core.crypto.hashing import hash_canonical, to_hex

from agents.validator import (
    SentinelAgent,
    PoRPackage,
    Challenge,
)

from fixtures import (
    make_valid_por_package,
    make_tampered_bundle,
    make_prompt_spec,
    make_evidence_bundle,
    make_evidence_item,
    make_reasoning_trace,
    make_reasoning_step,
    make_verdict,
)


class TestSentinelVerifyValid:
    """Tests for valid package verification."""

    def test_valid_package_passes_verification(self):
        """A valid PoR package should pass all verification checks."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=tool_plan,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package, mode="verify")

        assert result.ok is True, f"Expected verification to pass, but got errors: {result.get_error_messages()}"
        assert len(challenges) == 0, f"Expected no challenges, but got: {[c.kind for c in challenges]}"

    def test_valid_package_all_checks_pass(self):
        """All individual checks should pass for a valid package."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        # Check that we have expected checks
        check_ids = [c.check_id for c in result.checks]
        assert "strict_mode_flag" in check_ids
        assert "output_schema_ref" in check_ids
        assert "evaluation_variables" in check_ids
        assert "prompt_spec_hash_match" in check_ids
        assert "evidence_root_match" in check_ids
        assert "reasoning_root_match" in check_ids
        assert "verdict_hash_match" in check_ids

        # All should pass
        for check in result.checks:
            assert check.ok is True, f"Check '{check.check_id}' failed: {check.message}"

    def test_non_strict_mode_allows_missing_strict_flag(self):
        """When strict_mode=False, missing strict_mode flag should pass."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        # Remove strict_mode from extra
        prompt_spec.extra.pop("strict_mode", None)

        # Rebuild bundle with updated prompt_spec
        from core.por.proof_of_reasoning import build_por_bundle
        bundle = build_por_bundle(prompt_spec, evidence, trace, verdict)

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        # Should pass with strict_mode=False
        sentinel = SentinelAgent(strict_mode=False)
        result, challenges = sentinel.verify(package)

        assert result.ok is True


class TestSentinelVerifyEvidenceMismatch:
    """Tests for evidence root mismatch detection."""

    def test_tampered_evidence_root_fails(self):
        """Tampered evidence_root should fail verification."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        # Tamper with evidence_root
        tampered_root = "0x" + "ff" * 32
        tampered_bundle = make_tampered_bundle(bundle, "evidence_root", tampered_root)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False
        assert len(challenges) >= 1

        # Should have evidence_leaf challenge
        evidence_challenges = [c for c in challenges if c.kind == "evidence_leaf"]
        assert len(evidence_challenges) >= 1

        challenge = evidence_challenges[0]
        assert challenge.market_id == bundle.market_id
        assert "expected_root" in challenge.details or "computed_root" in challenge.details

    def test_evidence_mismatch_challenge_has_sample_proofs(self):
        """Evidence mismatch challenge should include sample Merkle proofs."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_root = "0x" + "aa" * 32
        tampered_bundle = make_tampered_bundle(bundle, "evidence_root", tampered_root)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True, max_sample_proofs=3)
        result, challenges = sentinel.verify(package)

        evidence_challenges = [c for c in challenges if c.kind == "evidence_leaf"]
        assert len(evidence_challenges) >= 1

        challenge = evidence_challenges[0]
        # Should have sample proofs in details
        assert "sample_proofs" in challenge.details or challenge.details.get("expected_root")


class TestSentinelVerifyReasoningMismatch:
    """Tests for reasoning root mismatch detection."""

    def test_tampered_reasoning_root_fails(self):
        """Tampered reasoning_root should fail verification."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_root = "0x" + "bb" * 32
        tampered_bundle = make_tampered_bundle(bundle, "reasoning_root", tampered_root)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        reasoning_challenges = [c for c in challenges if c.kind == "reasoning_leaf"]
        assert len(reasoning_challenges) >= 1

        challenge = reasoning_challenges[0]
        assert challenge.market_id == bundle.market_id

    def test_modified_trace_step_causes_reasoning_mismatch(self):
        """Modifying a trace step should cause reasoning root mismatch."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        # Modify the trace by changing a step's output
        modified_trace = make_reasoning_trace(
            trace_id=trace.trace_id,
            steps=[
                make_reasoning_step(
                    step_id="step_0001",
                    step_type="extract",
                    evidence_ids=["ev_001"],
                    output={"extracted_price": 999999.0},  # Different value
                ),
                make_reasoning_step(
                    step_id="step_0002",
                    step_type="check",
                    prior_step_ids=["step_0001"],
                    output={"price_above_threshold": True},
                ),
                make_reasoning_step(
                    step_id="step_0003",
                    step_type="map",
                    prior_step_ids=["step_0002"],
                    output={
                        "evaluation_variables": {
                            "event_observed": True,
                            "numeric_value": 999999.0,
                            "conflict_detected": False,
                            "insufficient_evidence": False,
                            "source_summary": ["ev_001"],
                        }
                    },
                ),
            ],
        )

        # Original bundle still has old reasoning_root
        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=modified_trace,  # Modified trace
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        # Should detect reasoning_root mismatch
        reasoning_challenges = [c for c in challenges if c.kind == "reasoning_leaf"]
        assert len(reasoning_challenges) >= 1


class TestSentinelVerifyVerdictMismatch:
    """Tests for verdict hash mismatch detection."""

    def test_tampered_verdict_hash_fails(self):
        """Tampered verdict_hash should fail verification."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_hash = "0x" + "cc" * 32
        tampered_bundle = make_tampered_bundle(bundle, "verdict_hash", tampered_hash)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        verdict_challenges = [c for c in challenges if c.kind == "verdict_hash"]
        assert len(verdict_challenges) >= 1

    def test_modified_verdict_outcome_causes_mismatch(self):
        """Changing verdict outcome should cause verdict hash mismatch."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        # Create verdict with different outcome
        modified_verdict = make_verdict(
            market_id=verdict.market_id,
            outcome="NO",  # Different from original
            confidence=verdict.confidence,
        )

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=modified_verdict,  # Modified verdict
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        # Should have verdict_hash challenges
        verdict_challenges = [c for c in challenges if c.kind == "verdict_hash"]
        assert len(verdict_challenges) >= 1


class TestSentinelVerifyPoRRootMismatch:
    """Tests for por_root mismatch detection."""

    def test_tampered_por_root_fails(self):
        """Tampered por_root should fail verification."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_root = "0x" + "dd" * 32
        tampered_bundle = make_tampered_bundle(bundle, "por_root", tampered_root)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        por_challenges = [c for c in challenges if c.kind == "por_root"]
        assert len(por_challenges) >= 1

        challenge = por_challenges[0]
        assert "component_hashes" in challenge.details


class TestSentinelVerifyPromptMismatch:
    """Tests for prompt spec hash mismatch detection."""

    def test_tampered_prompt_spec_hash_fails(self):
        """Tampered prompt_spec_hash should fail verification."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_hash = "0x" + "ee" * 32
        tampered_bundle = make_tampered_bundle(bundle, "prompt_spec_hash", tampered_hash)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        prompt_challenges = [c for c in challenges if c.kind == "prompt_hash"]
        assert len(prompt_challenges) >= 1


class TestSentinelChallengeStructure:
    """Tests for challenge structure and determinism."""

    def test_challenge_has_deterministic_id(self):
        """Challenge IDs should be deterministic."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_root = "0x" + "ff" * 32
        tampered_bundle = make_tampered_bundle(bundle, "evidence_root", tampered_root)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)

        # Run twice
        _, challenges1 = sentinel.verify(package)
        _, challenges2 = sentinel.verify(package)

        assert len(challenges1) > 0
        assert len(challenges2) > 0

        # Challenge IDs should be the same
        ids1 = [c.challenge_id for c in challenges1]
        ids2 = [c.challenge_id for c in challenges2]
        assert ids1 == ids2

    def test_challenge_can_be_serialized(self):
        """Challenges should be serializable to dict."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        tampered_root = "0x" + "ff" * 32
        tampered_bundle = make_tampered_bundle(bundle, "evidence_root", tampered_root)

        package = PoRPackage(
            bundle=tampered_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        _, challenges = sentinel.verify(package)

        assert len(challenges) > 0

        for challenge in challenges:
            serialized = challenge.to_dict()
            assert isinstance(serialized, dict)
            assert "challenge_id" in serialized
            assert "kind" in serialized
            assert "market_id" in serialized
            assert "bundle_root" in serialized


class TestSentinelStrictModeChecks:
    """Tests for strict mode validation."""

    def test_missing_evaluation_variables_fails_strict(self):
        """Missing evaluation_variables in final step should fail strict mode."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        # Create trace without evaluation_variables
        bad_trace = make_reasoning_trace(
            steps=[
                make_reasoning_step(
                    step_id="step_0001",
                    step_type="extract",
                    evidence_ids=["ev_001"],
                    output={"extracted_price": 105000.0},
                ),
                make_reasoning_step(
                    step_id="step_0002",
                    step_type="map",
                    prior_step_ids=["step_0001"],
                    output={},  # Missing evaluation_variables
                ),
            ],
        )

        # Rebuild bundle with bad trace
        from core.por.proof_of_reasoning import build_por_bundle
        new_bundle = build_por_bundle(prompt_spec, evidence, bad_trace, verdict)

        package = PoRPackage(
            bundle=new_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=bad_trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        # Should have evaluation_variables check failure
        eval_checks = [c for c in result.checks if c.check_id == "evaluation_variables"]
        assert len(eval_checks) == 1
        assert eval_checks[0].ok is False

    def test_wrong_output_schema_ref_fails_strict(self):
        """Wrong output_schema_ref should fail strict mode."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        # Change output_schema_ref
        prompt_spec.output_schema_ref = "wrong.schema.ref"

        # Rebuild bundle
        from core.por.proof_of_reasoning import build_por_bundle
        new_bundle = build_por_bundle(prompt_spec, evidence, trace, verdict)

        package = PoRPackage(
            bundle=new_bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )

        sentinel = SentinelAgent(strict_mode=True)
        result, challenges = sentinel.verify(package)

        assert result.ok is False

        schema_checks = [c for c in result.checks if c.check_id == "output_schema_ref"]
        assert len(schema_checks) == 1
        assert schema_checks[0].ok is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])