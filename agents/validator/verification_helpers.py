"""
Module 08 - Validator/Sentinel: Verification Helpers

Helper functions for verification checks used by SentinelAgent.

Owner: Protocol Verification Engineer
Module ID: M08
"""

from __future__ import annotations

from typing import Any, Optional

from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import compute_roots, compute_verdict_hash
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult

from .challenges import (
    Challenge,
    make_por_root_challenge,
    make_prompt_hash_challenge,
    make_verdict_hash_challenge,
)
from .pinpoint import (
    create_evidence_mismatch_challenge,
    create_reasoning_mismatch_challenge,
)


def check_strict_mode_flag(prompt_spec: PromptSpec, required: bool) -> CheckResult:
    """Check if strict_mode flag is properly set."""
    strict_mode_set = prompt_spec.extra.get("strict_mode", False)
    return CheckResult(
        check_id="strict_mode_flag",
        ok=strict_mode_set or not required,
        severity="info" if strict_mode_set else ("error" if required else "warn"),
        message="strict_mode is set" if strict_mode_set else "strict_mode not set in PromptSpec",
        details={"strict_mode": strict_mode_set, "required": required},
    )


def check_output_schema_ref(prompt_spec: PromptSpec) -> CheckResult:
    """Check if output_schema_ref is correct."""
    expected_schema = "core.schemas.verdict.DeterministicVerdict"
    actual_schema = prompt_spec.output_schema_ref
    schema_ok = actual_schema == expected_schema

    return CheckResult(
        check_id="output_schema_ref",
        ok=schema_ok,
        severity="info" if schema_ok else "error",
        message="output_schema_ref is correct" if schema_ok else f"Unexpected output_schema_ref: {actual_schema}",
        details={"expected": expected_schema, "actual": actual_schema},
    )


def check_evaluation_variables(trace: ReasoningTrace) -> CheckResult:
    """Check that trace has final evaluation_variables."""
    if not trace.steps:
        return CheckResult(
            check_id="evaluation_variables",
            ok=False,
            severity="error",
            message="Trace has no steps",
            details={},
        )

    # Look for final map or aggregate step with evaluation_variables
    final_step = None
    for step in reversed(trace.steps):
        if step.type in ("map", "aggregate"):
            final_step = step
            break

    if final_step is None:
        final_step = trace.steps[-1]

    eval_vars = final_step.output.get("evaluation_variables")
    has_eval_vars = eval_vars is not None

    return CheckResult(
        check_id="evaluation_variables",
        ok=has_eval_vars,
        severity="info" if has_eval_vars else "error",
        message="Trace has evaluation_variables" if has_eval_vars else "Missing evaluation_variables in final step",
        details={
            "final_step_id": final_step.step_id,
            "final_step_type": final_step.type,
            "has_evaluation_variables": has_eval_vars,
        },
    )


def verify_commitments(
    bundle: PoRBundle,
    prompt_spec: PromptSpec,
    evidence: EvidenceBundle,
    trace: ReasoningTrace,
    verdict: DeterministicVerdict,
    max_sample_proofs: int = 3,
) -> tuple[list[CheckResult], list[Challenge]]:
    """
    Verify all commitment hashes match.

    Args:
        bundle: The PoR bundle with expected commitments.
        prompt_spec: The prompt specification.
        evidence: The evidence bundle.
        trace: The reasoning trace.
        verdict: The deterministic verdict.
        max_sample_proofs: Max sample proofs for mismatch challenges.

    Returns:
        Tuple of (list of CheckResults, list of Challenges).
    """
    checks: list[CheckResult] = []
    challenges: list[Challenge] = []

    # Compute all roots
    try:
        computed_roots = compute_roots(prompt_spec, evidence, trace, verdict)
    except Exception as e:
        checks.append(CheckResult(
            check_id="compute_roots",
            ok=False,
            severity="error",
            message=f"Failed to compute roots: {e}",
            details={"error": str(e)},
        ))
        return checks, challenges

    # Check prompt_spec_hash
    prompt_match = bundle.prompt_spec_hash == computed_roots.prompt_spec_hash
    checks.append(CheckResult(
        check_id="prompt_spec_hash_match",
        ok=prompt_match,
        severity="info" if prompt_match else "error",
        message="prompt_spec_hash matches" if prompt_match else "prompt_spec_hash mismatch",
        details={"expected": bundle.prompt_spec_hash, "computed": computed_roots.prompt_spec_hash},
    ))

    if not prompt_match:
        challenges.append(make_prompt_hash_challenge(
            market_id=bundle.market_id,
            bundle_root=bundle.por_root or bundle.verdict_hash,
            expected_hash=bundle.prompt_spec_hash,
            computed_hash=computed_roots.prompt_spec_hash,
        ))

    # Check evidence_root
    evidence_match = bundle.evidence_root == computed_roots.evidence_root
    checks.append(CheckResult(
        check_id="evidence_root_match",
        ok=evidence_match,
        severity="info" if evidence_match else "error",
        message="evidence_root matches" if evidence_match else "evidence_root mismatch",
        details={"expected": bundle.evidence_root, "computed": computed_roots.evidence_root},
    ))

    if not evidence_match:
        challenges.append(create_evidence_mismatch_challenge(
            bundle, evidence, computed_roots.evidence_root, max_sample_proofs
        ))

    # Check reasoning_root
    reasoning_match = bundle.reasoning_root == computed_roots.reasoning_root
    checks.append(CheckResult(
        check_id="reasoning_root_match",
        ok=reasoning_match,
        severity="info" if reasoning_match else "error",
        message="reasoning_root matches" if reasoning_match else "reasoning_root mismatch",
        details={"expected": bundle.reasoning_root, "computed": computed_roots.reasoning_root},
    ))

    if not reasoning_match:
        challenges.append(create_reasoning_mismatch_challenge(
            bundle, trace, computed_roots.reasoning_root, max_sample_proofs
        ))

    # Check verdict_hash
    verdict_match = bundle.verdict_hash == computed_roots.verdict_hash
    checks.append(CheckResult(
        check_id="verdict_hash_match",
        ok=verdict_match,
        severity="info" if verdict_match else "error",
        message="verdict_hash matches" if verdict_match else "verdict_hash mismatch",
        details={"expected": bundle.verdict_hash, "computed": computed_roots.verdict_hash},
    ))

    if not verdict_match:
        challenges.append(make_verdict_hash_challenge(
            market_id=bundle.market_id,
            bundle_root=bundle.por_root or bundle.verdict_hash,
            expected_hash=bundle.verdict_hash,
            computed_hash=computed_roots.verdict_hash,
        ))

    # Check por_root if present
    if bundle.por_root is not None:
        por_match = bundle.por_root == computed_roots.por_root
        checks.append(CheckResult(
            check_id="por_root_match",
            ok=por_match,
            severity="info" if por_match else "error",
            message="por_root matches" if por_match else "por_root mismatch",
            details={"expected": bundle.por_root, "computed": computed_roots.por_root},
        ))

        if not por_match:
            challenges.append(make_por_root_challenge(
                market_id=bundle.market_id,
                expected_root=bundle.por_root,
                computed_root=computed_roots.por_root,
                component_hashes={
                    "prompt_spec_hash": computed_roots.prompt_spec_hash,
                    "evidence_root": computed_roots.evidence_root,
                    "reasoning_root": computed_roots.reasoning_root,
                    "verdict_hash": computed_roots.verdict_hash,
                },
            ))

    return checks, challenges


def verify_verdict_integrity(
    bundle: PoRBundle,
    verdict: DeterministicVerdict,
) -> tuple[list[CheckResult], list[Challenge]]:
    """
    Verify verdict hash matches and verdict is valid.

    Args:
        bundle: The PoR bundle with expected verdict_hash.
        verdict: The deterministic verdict to verify.

    Returns:
        Tuple of (list of CheckResults, list of Challenges).
    """
    checks: list[CheckResult] = []
    challenges: list[Challenge] = []

    # Recompute verdict hash
    computed_hash = compute_verdict_hash(verdict)
    expected_hash = bundle.verdict_hash

    match = computed_hash == expected_hash
    checks.append(CheckResult(
        check_id="verdict_hash_recompute",
        ok=match,
        severity="info" if match else "error",
        message="Verdict hash recomputation matches" if match else "Verdict hash recomputation mismatch",
        details={"expected": expected_hash, "computed": computed_hash},
    ))

    if not match:
        challenges.append(make_verdict_hash_challenge(
            market_id=bundle.market_id,
            bundle_root=bundle.por_root or expected_hash,
            expected_hash=expected_hash,
            computed_hash=computed_hash,
        ))

    # Verify verdict outcome is valid
    valid_outcomes = {"YES", "NO", "INVALID"}
    outcome_valid = verdict.outcome in valid_outcomes
    checks.append(CheckResult(
        check_id="verdict_outcome_valid",
        ok=outcome_valid,
        severity="info" if outcome_valid else "error",
        message=f"Verdict outcome '{verdict.outcome}' is valid" if outcome_valid
                else f"Invalid verdict outcome: {verdict.outcome}",
        details={"outcome": verdict.outcome, "valid_outcomes": list(valid_outcomes)},
    ))

    # Verify confidence bounds
    confidence_valid = 0.0 <= verdict.confidence <= 1.0
    checks.append(CheckResult(
        check_id="verdict_confidence_bounds",
        ok=confidence_valid,
        severity="info" if confidence_valid else "error",
        message="Verdict confidence within bounds" if confidence_valid
                else f"Verdict confidence out of bounds: {verdict.confidence}",
        details={"confidence": verdict.confidence},
    ))

    return checks, challenges


__all__ = [
    "check_strict_mode_flag",
    "check_output_schema_ref",
    "check_evaluation_variables",
    "verify_commitments",
    "verify_verdict_integrity",
]