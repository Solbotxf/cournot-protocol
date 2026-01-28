"""
Module 09B - Artifact Packaging & IO
File: pack.py

Purpose: Validate semantic correctness of artifact packs.
"""

from __future__ import annotations

from typing import Any

from core.crypto.hashing import hash_canonical, to_hex
from core.schemas.verification import CheckResult, VerificationResult, ChallengeRef
from core.por.proof_of_reasoning import (
    compute_prompt_spec_hash,
    compute_evidence_root,
    compute_reasoning_root,
    compute_verdict_hash,
    compute_por_root,
)

from orchestrator.pipeline import PoRPackage
from orchestrator.artifacts.manifest import PackManifest


def _make_check(check_id: str, ok: bool, message: str, details: dict[str, Any] | None = None) -> CheckResult:
    """Create a CheckResult."""
    return CheckResult(
        check_id=check_id,
        ok=ok,
        severity="info" if ok else "error",
        message=message,
        details=details or {},
    )


def validate_market_id_consistency(package: PoRPackage) -> list[CheckResult]:
    """Validate market_id is consistent across all artifacts."""
    checks: list[CheckResult] = []
    
    prompt_market_id = package.prompt_spec.market.market_id
    verdict_market_id = package.verdict.market_id
    bundle_market_id = package.bundle.market_id
    
    # Check prompt_spec.market.market_id == verdict.market_id
    if prompt_market_id == verdict_market_id:
        checks.append(_make_check(
            "market_id_prompt_verdict",
            True,
            "PromptSpec and Verdict market_id match",
        ))
    else:
        checks.append(_make_check(
            "market_id_prompt_verdict",
            False,
            "PromptSpec and Verdict market_id mismatch",
            {"prompt_spec": prompt_market_id, "verdict": verdict_market_id},
        ))
    
    # Check verdict.market_id == bundle.market_id
    if verdict_market_id == bundle_market_id:
        checks.append(_make_check(
            "market_id_verdict_bundle",
            True,
            "Verdict and Bundle market_id match",
        ))
    else:
        checks.append(_make_check(
            "market_id_verdict_bundle",
            False,
            "Verdict and Bundle market_id mismatch",
            {"verdict": verdict_market_id, "bundle": bundle_market_id},
        ))
    
    return checks


def validate_verdict_hash(package: PoRPackage) -> list[CheckResult]:
    """Validate bundle.verdict_hash matches hash of verdict."""
    checks: list[CheckResult] = []
    
    computed_hash = compute_verdict_hash(package.verdict)
    bundle_hash = package.bundle.verdict_hash
    
    if computed_hash == bundle_hash:
        checks.append(_make_check(
            "verdict_hash",
            True,
            "Verdict hash matches bundle.verdict_hash",
        ))
    else:
        checks.append(_make_check(
            "verdict_hash",
            False,
            "Verdict hash mismatch",
            {"computed": computed_hash, "bundle": bundle_hash},
        ))
    
    return checks


def validate_prompt_spec_hash(package: PoRPackage) -> list[CheckResult]:
    """Validate bundle.prompt_spec_hash matches hash of prompt_spec."""
    checks: list[CheckResult] = []
    
    # Check if prompt_spec.created_at is None (required for deterministic hashing)
    if package.prompt_spec.created_at is not None:
        checks.append(_make_check(
            "prompt_spec_timestamp",
            False,
            "PromptSpec.created_at must be None for deterministic hashing",
        ))
        # Still compute hash for comparison but note the issue
        computed_hash = to_hex(hash_canonical(package.prompt_spec))
    else:
        checks.append(_make_check(
            "prompt_spec_timestamp",
            True,
            "PromptSpec.created_at is None",
        ))
        computed_hash = compute_prompt_spec_hash(package.prompt_spec, strict=False)
    
    bundle_hash = package.bundle.prompt_spec_hash
    
    if computed_hash == bundle_hash:
        checks.append(_make_check(
            "prompt_spec_hash",
            True,
            "PromptSpec hash matches bundle.prompt_spec_hash",
        ))
    else:
        checks.append(_make_check(
            "prompt_spec_hash",
            False,
            "PromptSpec hash mismatch",
            {"computed": computed_hash, "bundle": bundle_hash},
        ))
    
    return checks


def validate_evidence_root(package: PoRPackage) -> list[CheckResult]:
    """Validate bundle.evidence_root matches computed evidence merkle root."""
    checks: list[CheckResult] = []
    
    computed_root = compute_evidence_root(package.evidence)
    bundle_root = package.bundle.evidence_root
    
    if computed_root == bundle_root:
        checks.append(_make_check(
            "evidence_root",
            True,
            "Evidence root matches bundle.evidence_root",
        ))
    else:
        checks.append(_make_check(
            "evidence_root",
            False,
            "Evidence root mismatch",
            {"computed": computed_root, "bundle": bundle_root},
        ))
    
    return checks


def validate_reasoning_root(package: PoRPackage) -> list[CheckResult]:
    """Validate bundle.reasoning_root matches computed reasoning merkle root."""
    checks: list[CheckResult] = []
    
    computed_root = compute_reasoning_root(package.trace)
    bundle_root = package.bundle.reasoning_root
    
    if computed_root == bundle_root:
        checks.append(_make_check(
            "reasoning_root",
            True,
            "Reasoning root matches bundle.reasoning_root",
        ))
    else:
        checks.append(_make_check(
            "reasoning_root",
            False,
            "Reasoning root mismatch",
            {"computed": computed_root, "bundle": bundle_root},
        ))
    
    return checks


def validate_por_root(package: PoRPackage) -> list[CheckResult]:
    """Validate bundle.por_root matches computed PoR root."""
    checks: list[CheckResult] = []
    
    if package.bundle.por_root is None:
        checks.append(_make_check(
            "por_root",
            True,
            "PoR root not present in bundle (optional)",
        ))
        return checks
    
    # Compute expected por_root from the four leaves
    prompt_hash = compute_prompt_spec_hash(package.prompt_spec, strict=False)
    evidence_root = compute_evidence_root(package.evidence)
    reasoning_root = compute_reasoning_root(package.trace)
    verdict_hash = compute_verdict_hash(package.verdict)
    
    computed_por_root = compute_por_root(prompt_hash, evidence_root, reasoning_root, verdict_hash)
    
    if computed_por_root == package.bundle.por_root:
        checks.append(_make_check(
            "por_root",
            True,
            "PoR root matches bundle.por_root",
        ))
    else:
        checks.append(_make_check(
            "por_root",
            False,
            "PoR root mismatch",
            {"computed": computed_por_root, "bundle": package.bundle.por_root},
        ))
    
    return checks


def validate_embedded_verdict(package: PoRPackage) -> list[CheckResult]:
    """Validate bundle.verdict matches standalone verdict."""
    checks: list[CheckResult] = []
    
    # Compare key fields
    bundle_verdict = package.bundle.verdict
    standalone_verdict = package.verdict
    
    matches = (
        bundle_verdict.market_id == standalone_verdict.market_id
        and bundle_verdict.outcome == standalone_verdict.outcome
        and bundle_verdict.confidence == standalone_verdict.confidence
        and bundle_verdict.resolution_rule_id == standalone_verdict.resolution_rule_id
    )
    
    if matches:
        checks.append(_make_check(
            "embedded_verdict",
            True,
            "Bundle embedded verdict matches standalone verdict",
        ))
    else:
        checks.append(_make_check(
            "embedded_verdict",
            False,
            "Bundle embedded verdict differs from standalone verdict",
            {
                "bundle_outcome": bundle_verdict.outcome,
                "standalone_outcome": standalone_verdict.outcome,
            },
        ))
    
    return checks


def validate_evidence_references(package: PoRPackage) -> list[CheckResult]:
    """Validate that trace evidence references exist in evidence bundle."""
    checks: list[CheckResult] = []
    
    bundle_evidence_ids = set(package.evidence.evidence_ids)
    trace_evidence_ids = package.trace.get_all_evidence_ids()
    trace_evidence_ids.update(package.trace.evidence_refs)
    
    invalid_refs = trace_evidence_ids - bundle_evidence_ids
    
    if not invalid_refs:
        checks.append(_make_check(
            "evidence_references",
            True,
            "All trace evidence references exist in bundle",
        ))
    else:
        checks.append(_make_check(
            "evidence_references",
            False,
            f"Invalid evidence references in trace: {sorted(invalid_refs)}",
            {"invalid_refs": sorted(invalid_refs)},
        ))
    
    return checks


def validate_pack(package: PoRPackage, *, strict: bool = True) -> VerificationResult:
    """
    Validate semantic correctness of a PoRPackage.
    
    Performs the following checks:
    1. Market ID consistency across prompt_spec, verdict, and bundle
    2. Verdict hash matches bundle.verdict_hash
    3. PromptSpec hash matches bundle.prompt_spec_hash
    4. Evidence root matches bundle.evidence_root
    5. Reasoning root matches bundle.reasoning_root
    6. PoR root matches bundle.por_root (if present)
    7. Embedded verdict matches standalone verdict
    8. Evidence references in trace are valid
    
    Args:
        package: The PoRPackage to validate
        strict: If True, any failure results in ok=False
    
    Returns:
        VerificationResult with all checks
    """
    all_checks: list[CheckResult] = []
    
    # Run all validations
    all_checks.extend(validate_market_id_consistency(package))
    all_checks.extend(validate_verdict_hash(package))
    all_checks.extend(validate_prompt_spec_hash(package))
    all_checks.extend(validate_evidence_root(package))
    all_checks.extend(validate_reasoning_root(package))
    all_checks.extend(validate_por_root(package))
    all_checks.extend(validate_embedded_verdict(package))
    all_checks.extend(validate_evidence_references(package))
    
    # Determine overall result
    all_ok = all(c.ok for c in all_checks)
    
    # Build challenge if not ok
    challenge = None
    if not all_ok:
        failed_checks = [c for c in all_checks if not c.ok]
        if failed_checks:
            first_failure = failed_checks[0]
            # Determine challenge kind based on check_id
            if "evidence" in first_failure.check_id:
                kind = "evidence_leaf"
            elif "reasoning" in first_failure.check_id:
                kind = "reasoning_leaf"
            elif "verdict" in first_failure.check_id:
                kind = "verdict_hash"
            else:
                kind = "por_bundle"
            challenge = ChallengeRef(kind=kind, reason=first_failure.message)
    
    return VerificationResult(ok=all_ok, checks=all_checks, challenge=challenge)


def validate_manifest_consistency(manifest: PackManifest, package: PoRPackage) -> VerificationResult:
    """
    Validate manifest metadata matches package contents.
    
    Checks:
    1. manifest.market_id matches package market_id
    2. manifest.por_root matches bundle.por_root
    """
    checks: list[CheckResult] = []
    
    # Check market_id
    if manifest.market_id == package.bundle.market_id:
        checks.append(_make_check(
            "manifest_market_id",
            True,
            "Manifest market_id matches package",
        ))
    else:
        checks.append(_make_check(
            "manifest_market_id",
            False,
            "Manifest market_id mismatch",
            {"manifest": manifest.market_id, "package": package.bundle.market_id},
        ))
    
    # Check por_root
    bundle_por_root = package.bundle.por_root or ""
    if manifest.por_root == bundle_por_root:
        checks.append(_make_check(
            "manifest_por_root",
            True,
            "Manifest por_root matches package",
        ))
    else:
        checks.append(_make_check(
            "manifest_por_root",
            False,
            "Manifest por_root mismatch",
            {"manifest": manifest.por_root, "package": bundle_por_root},
        ))
    
    all_ok = all(c.ok for c in checks)
    return VerificationResult(ok=all_ok, checks=checks)


__all__ = [
    "validate_pack",
    "validate_market_id_consistency",
    "validate_verdict_hash",
    "validate_prompt_spec_hash",
    "validate_evidence_root",
    "validate_reasoning_root",
    "validate_por_root",
    "validate_embedded_verdict",
    "validate_evidence_references",
    "validate_manifest_consistency",
]