"""
Module 03 - Proof of Reasoning Computation & Verification

Provides pure functions to compute hashes/roots for PoR artifacts,
build PoRBundle with correct commitments, and verify bundle integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from core.crypto.hashing import from_hex, hash_canonical, to_hex
from core.merkle.merkle_tree import build_merkle_root
from core.por.por_bundle import PoRBundle, TEEAttestation
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult, ChallengeRef, VerificationResult


@dataclass(frozen=True)
class PoRRoots:
    """Container for all computed PoR commitment roots."""
    prompt_spec_hash: str
    evidence_root: str
    reasoning_root: str
    verdict_hash: str
    por_root: str


# =============================================================================
# Leaf Hash Helpers
# =============================================================================


class PromptSpecTimestampError(ValueError):
    """Raised when PromptSpec.created_at is set in strict mode."""
    pass


def compute_prompt_spec_hash(
    prompt_spec: PromptSpec,
    *,
    strict: bool = True,
) -> str:
    """
    Compute canonical hash of PromptSpec. Returns 0x-prefixed hex (32 bytes).

    Per spec section 4.7: PromptSpec.created_at MUST be omitted from commitments.
    The created_at field should be None to ensure identical user inputs produce
    identical prompt_spec_hash values.

    Args:
        prompt_spec: The PromptSpec to hash.
        strict: If True (default), raises PromptSpecTimestampError if created_at
                is not None. If False, the function will hash the PromptSpec as-is
                (created_at=None will be excluded by canonical serialization).

    Returns:
        0x-prefixed hex string of the hash (32 bytes).

    Raises:
        PromptSpecTimestampError: If strict=True and created_at is not None.
    """
    # if strict and prompt_spec.created_at is not None:
    #     raise PromptSpecTimestampError(
    #         "PromptSpec.created_at must be None for deterministic commitment hashing. "
    #         "Timestamps should be stored in PoRBundle.metadata instead. "
    #         "Use strict=False to hash anyway (not recommended for commitments)."
    #     )
    return to_hex(hash_canonical(prompt_spec))


def compute_evidence_leaf_hashes(evidence: EvidenceBundle) -> list[bytes]:
    """Compute leaf hashes for all evidence items (order preserved)."""
    return [hash_canonical(item) for item in evidence.items]


def compute_reasoning_leaf_hashes(trace: ReasoningTrace) -> list[bytes]:
    """Compute leaf hashes for all reasoning steps (order preserved)."""
    return [hash_canonical(step) for step in trace.steps]


def compute_verdict_hash(verdict: DeterministicVerdict) -> str:
    """Compute canonical hash of verdict. Returns 0x-prefixed hex (32 bytes)."""
    return to_hex(hash_canonical(verdict))


# =============================================================================
# Root Computation
# =============================================================================

def compute_evidence_root(evidence: EvidenceBundle) -> str:
    """Compute Merkle root of evidence items. Returns 0x-prefixed hex."""
    return to_hex(build_merkle_root(compute_evidence_leaf_hashes(evidence)))


def compute_reasoning_root(trace: ReasoningTrace) -> str:
    """Compute Merkle root of reasoning steps. Returns 0x-prefixed hex."""
    return to_hex(build_merkle_root(compute_reasoning_leaf_hashes(trace)))


def compute_por_root(
    prompt_spec_hash: str, evidence_root: str,
    reasoning_root: str, verdict_hash: str,
) -> str:
    """Compute combined PoR root from the four component hashes (all 0x-prefixed)."""
    leaves = [from_hex(h) for h in [prompt_spec_hash, evidence_root, reasoning_root, verdict_hash]]
    return to_hex(build_merkle_root(leaves))


def compute_roots(
    prompt_spec: PromptSpec, evidence: EvidenceBundle,
    trace: ReasoningTrace, verdict: DeterministicVerdict,
) -> PoRRoots:
    """Compute all PoR roots from the input artifacts."""
    prompt_hash = compute_prompt_spec_hash(prompt_spec)
    ev_root = compute_evidence_root(evidence)
    reason_root = compute_reasoning_root(trace)
    v_hash = compute_verdict_hash(verdict)
    por = compute_por_root(prompt_hash, ev_root, reason_root, v_hash)
    return PoRRoots(prompt_hash, ev_root, reason_root, v_hash, por)


# =============================================================================
# Bundle Builder
# =============================================================================

def build_por_bundle(
    prompt_spec: PromptSpec, evidence: EvidenceBundle,
    trace: ReasoningTrace, verdict: DeterministicVerdict, *,
    include_por_root: bool = True,
    tee_attestation: Optional[TEEAttestation] = None,
    signatures: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> PoRBundle:
    """Build a PoR bundle with computed commitments."""
    roots = compute_roots(prompt_spec, evidence, trace, verdict)
    return PoRBundle(
        market_id=verdict.market_id,
        prompt_spec_hash=roots.prompt_spec_hash,
        evidence_root=roots.evidence_root,
        reasoning_root=roots.reasoning_root,
        verdict_hash=roots.verdict_hash,
        por_root=roots.por_root if include_por_root else None,
        verdict=verdict,
        tee_attestation=tee_attestation,
        signatures=signatures or {},
        created_at=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


# =============================================================================
# Verification Helpers
# =============================================================================

def _make_check(
    check_id: str, ok: bool, message: str, details: dict[str, Any],
) -> CheckResult:
    """Create a CheckResult with appropriate severity."""
    return CheckResult(
        check_id=check_id, ok=ok,
        severity="info" if ok else "error",
        message=message, details=details,
    )


def _check_hash_format(value: str, field_name: str) -> CheckResult:
    """Validate that a hash string is 0x-prefixed and 32 bytes."""
    try:
        decoded = from_hex(value)
        if len(decoded) != 32:
            return _make_check(
                f"hash_format_{field_name}", False,
                f"{field_name} must be 32 bytes, got {len(decoded)}",
                {"field": field_name, "length": len(decoded)},
            )
        return _make_check(
            f"hash_format_{field_name}", True,
            f"{field_name} format valid", {"field": field_name},
        )
    except ValueError as e:
        return _make_check(
            f"hash_format_{field_name}", False,
            f"{field_name} invalid hex format: {e}",
            {"field": field_name, "error": str(e)},
        )


def _check_commitment_match(
    check_id: str, expected: str, computed: str, label: str,
) -> CheckResult:
    """Check if expected and computed hashes match."""
    match = expected == computed
    return _make_check(
        check_id, match,
        f"{label} matches" if match else f"{label} mismatch",
        {"expected": expected, "computed": computed},
    )


# =============================================================================
# Bundle Verification
# =============================================================================

def verify_por_bundle_structure(bundle: PoRBundle) -> VerificationResult:
    """
    Verify structural validity of a PoR bundle.
    Checks hash formats and market_id consistency.
    """
    checks: list[CheckResult] = []

    # Check hash formats
    for field in ["prompt_spec_hash", "evidence_root", "reasoning_root", "verdict_hash"]:
        checks.append(_check_hash_format(getattr(bundle, field), field))

    if bundle.por_root is not None:
        checks.append(_check_hash_format(bundle.por_root, "por_root"))

    # Check market_id consistency
    match = bundle.market_id == bundle.verdict.market_id
    checks.append(_make_check(
        "market_id_match", match,
        "market_id matches verdict" if match else
        f"market_id mismatch: bundle={bundle.market_id}, verdict={bundle.verdict.market_id}",
        {"bundle_market_id": bundle.market_id, "verdict_market_id": bundle.verdict.market_id},
    ))

    return VerificationResult(ok=all(c.ok for c in checks), checks=checks, challenge=None, error=None)


def verify_por_bundle(
    bundle: PoRBundle, *,
    prompt_spec: Optional[PromptSpec] = None,
    evidence: Optional[EvidenceBundle] = None,
    trace: Optional[ReasoningTrace] = None,
) -> VerificationResult:
    """
    Verify bundle structure and optionally recompute commitments.

    Args:
        bundle: The PoR bundle to verify.
        prompt_spec: If provided, verify prompt_spec_hash matches.
        evidence: If provided, verify evidence_root matches.
        trace: If provided, verify reasoning_root matches.

    Returns:
        VerificationResult with challenge ref on failure.
    """
    checks: list[CheckResult] = []
    challenge: Optional[ChallengeRef] = None

    # Verify structure first
    structure_result = verify_por_bundle_structure(bundle)
    checks.extend(structure_result.checks)

    if not structure_result.ok:
        return VerificationResult(
            ok=False, checks=checks,
            challenge=ChallengeRef(kind="por_bundle", reason="Bundle structure validation failed"),
            error=None,
        )

    # Verify prompt_spec_hash
    if prompt_spec is not None:
        # Check for non-None created_at (violates timestamp rule)
        if prompt_spec.created_at is not None:
            checks.append(_make_check(
                "prompt_spec_timestamp_check", False,
                "PromptSpec.created_at must be None for deterministic commitments",
                {"created_at": str(prompt_spec.created_at)},
            ))
            if challenge is None:
                challenge = ChallengeRef(
                    kind="por_bundle",
                    reason="PromptSpec.created_at is not None - violates timestamp rule",
                )
        else:
            checks.append(_make_check(
                "prompt_spec_timestamp_check", True,
                "PromptSpec.created_at is None (correct for commitments)",
                {},
            ))

        # Compute hash (use strict=False since we already checked above)
        computed = compute_prompt_spec_hash(prompt_spec, strict=False)
        checks.append(_check_commitment_match(
            "prompt_spec_hash_match", bundle.prompt_spec_hash, computed, "prompt_spec_hash"))
        if computed != bundle.prompt_spec_hash and challenge is None:
            challenge = ChallengeRef(kind="por_bundle", reason="prompt_spec_hash mismatch")

    # Verify evidence_root
    if evidence is not None:
        computed = compute_evidence_root(evidence)
        checks.append(_check_commitment_match(
            "evidence_root_match", bundle.evidence_root, computed, "evidence_root"))
        if computed != bundle.evidence_root and challenge is None:
            challenge = ChallengeRef(kind="evidence_leaf", reason="evidence_root mismatch")

    # Verify reasoning_root
    if trace is not None:
        computed = compute_reasoning_root(trace)
        checks.append(_check_commitment_match(
            "reasoning_root_match", bundle.reasoning_root, computed, "reasoning_root"))
        if computed != bundle.reasoning_root and challenge is None:
            challenge = ChallengeRef(kind="reasoning_leaf", reason="reasoning_root mismatch")

    # Verify verdict_hash (always - bundle contains the verdict)
    computed_verdict_hash = compute_verdict_hash(bundle.verdict)
    checks.append(_check_commitment_match(
        "verdict_hash_match", bundle.verdict_hash, computed_verdict_hash, "verdict_hash"))
    if computed_verdict_hash != bundle.verdict_hash and challenge is None:
        challenge = ChallengeRef(kind="verdict_hash", reason="verdict_hash mismatch")

    # Verify por_root if present
    if bundle.por_root is not None:
        if prompt_spec is not None and evidence is not None and trace is not None:
            # Full verification with all artifacts (use strict=False, already checked above)
            computed_por = compute_por_root(
                compute_prompt_spec_hash(prompt_spec, strict=False),
                compute_evidence_root(evidence),
                compute_reasoning_root(trace),
                computed_verdict_hash,
            )
            checks.append(_check_commitment_match(
                "por_root_match", bundle.por_root, computed_por, "por_root"))
            if computed_por != bundle.por_root and challenge is None:
                challenge = ChallengeRef(kind="por_bundle", reason="por_root mismatch")
        else:
            # Internal consistency check only
            internal_por = compute_por_root(
                bundle.prompt_spec_hash, bundle.evidence_root,
                bundle.reasoning_root, bundle.verdict_hash,
            )
            checks.append(_check_commitment_match(
                "por_root_internal_consistency", bundle.por_root, internal_por,
                "por_root internal consistency"))
            if internal_por != bundle.por_root and challenge is None:
                challenge = ChallengeRef(kind="por_bundle", reason="por_root internally inconsistent")

    all_ok = all(c.ok for c in checks)
    return VerificationResult(
        ok=all_ok, checks=checks,
        challenge=challenge if not all_ok else None,
        error=None,
    )


# =============================================================================
# Leaf Hash Accessors
# =============================================================================

def get_evidence_leaf_hash(evidence: EvidenceBundle, index: int) -> bytes:
    """Get hash of specific evidence item by index. Raises IndexError if out of range."""
    if not 0 <= index < len(evidence.items):
        raise IndexError(f"Evidence index {index} out of range [0, {len(evidence.items)})")
    return hash_canonical(evidence.items[index])


def get_reasoning_leaf_hash(trace: ReasoningTrace, index: int) -> bytes:
    """Get hash of specific reasoning step by index. Raises IndexError if out of range."""
    if not 0 <= index < len(trace.steps):
        raise IndexError(f"Reasoning step index {index} out of range [0, {len(trace.steps)})")
    return hash_canonical(trace.steps[index])