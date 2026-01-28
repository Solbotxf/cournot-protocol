"""
Module 08 - Validator/Sentinel: Sentinel Agent

Provides an independent validator (sentinel) that can:
1. Verify a provided PoR package end-to-end
2. Optionally replay evidence collection deterministically
3. Detect mismatches and policy failures
4. Generate pinpoint challenges with Merkle proofs

Owner: Protocol Verification Engineer
Module ID: M08
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import (
    compute_evidence_root,
    verify_por_bundle_structure,
)
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult, ChallengeRef, VerificationResult

from .challenges import Challenge, make_replay_divergence_challenge
from .replay_executor import CollectorProtocol, ReplayExecutor
from .verification_helpers import (
    check_evaluation_variables,
    check_output_schema_ref,
    check_strict_mode_flag,
    verify_commitments,
    verify_verdict_integrity,
)


# Verification mode types
VerifyMode = Literal["verify", "replay"]


@dataclass
class PoRPackage:
    """
    Internal container for all PoR artifacts needed for verification.

    This packages together:
    - The PoR bundle with commitments
    - The original artifacts (prompt_spec, evidence, trace, verdict)
    - Optional tool_plan for replay mode
    """
    bundle: PoRBundle
    prompt_spec: PromptSpec
    evidence: EvidenceBundle
    trace: ReasoningTrace
    verdict: DeterministicVerdict
    tool_plan: Optional[ToolPlan] = None

    def __post_init__(self) -> None:
        """Validate package consistency."""
        if self.bundle.market_id != self.verdict.market_id:
            raise ValueError(
                f"Bundle market_id '{self.bundle.market_id}' does not match "
                f"verdict market_id '{self.verdict.market_id}'"
            )
        if self.prompt_spec.market.market_id != self.bundle.market_id:
            raise ValueError(
                f"PromptSpec market_id '{self.prompt_spec.market.market_id}' does not match "
                f"bundle market_id '{self.bundle.market_id}'"
            )


@dataclass
class SentinelVerificationResult:
    """Complete verification result from the Sentinel."""
    verification: VerificationResult
    challenges: list[Challenge] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Overall verification success."""
        return self.verification.ok

    @property
    def has_challenges(self) -> bool:
        """Check if any challenges were generated."""
        return len(self.challenges) > 0


class SentinelAgent:
    """
    Independent validator (sentinel) for PoR packages.

    Supports two modes:
    - verify: Verify-only mode (default) - no network calls
    - replay: Replay mode - re-fetches evidence for comparison
    """

    role: str = "validator"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        collector: Optional[CollectorProtocol] = None,
        strict_mode: bool = True,
        max_sample_proofs: int = 3,
    ) -> None:
        """
        Initialize the Sentinel agent.

        Args:
            name: Optional agent name.
            collector: Optional collector for replay mode.
            strict_mode: If True, require strict_mode in PromptSpec.
            max_sample_proofs: Max sample proofs in mismatch challenges.
        """
        self.name = name or "sentinel"
        self._replay_executor = ReplayExecutor(collector)
        self._strict_mode = strict_mode
        self._max_sample_proofs = max_sample_proofs

    def verify(
        self,
        package: PoRPackage,
        *,
        mode: VerifyMode = "verify",
    ) -> tuple[VerificationResult, list[Challenge]]:
        """
        Verify a PoR package end-to-end.

        Args:
            package: The PoR package containing bundle and artifacts.
            mode: "verify" for verify-only, "replay" for evidence replay.

        Returns:
            Tuple of (VerificationResult, list[Challenge]).
        """
        checks: list[CheckResult] = []
        challenges: list[Challenge] = []

        # Step 1: Schema lock / strict checks
        strict_checks, strict_ok = self._check_strict_mode(package)
        checks.extend(strict_checks)

        if not strict_ok and self._strict_mode:
            return VerificationResult(
                ok=False,
                checks=checks,
                challenge=ChallengeRef(kind="por_bundle", reason="Strict mode validation failed"),
            ), challenges

        # Step 2: Bundle structural verification (hash formats, market_id match)
        bundle_result = verify_por_bundle_structure(package.bundle)
        checks.extend(bundle_result.checks)

        if not bundle_result.ok:
            challenges.append(self._make_structure_challenge(package, bundle_result))
            return VerificationResult(
                ok=False,
                checks=checks,
                challenge=bundle_result.challenge,
            ), challenges

        # Step 3: Recompute and verify commitments
        commitment_checks, commitment_challenges = verify_commitments(
            package.bundle,
            package.prompt_spec,
            package.evidence,
            package.trace,
            package.verdict,
            self._max_sample_proofs,
        )
        checks.extend(commitment_checks)
        challenges.extend(commitment_challenges)

        # Step 4: Cross-check verdict
        verdict_checks, verdict_challenges = verify_verdict_integrity(
            package.bundle, package.verdict
        )
        checks.extend(verdict_checks)
        challenges.extend(verdict_challenges)

        # Step 5: Replay mode (optional)
        if mode == "replay":
            replay_checks, replay_challenges = self._verify_replay(package)
            checks.extend(replay_checks)
            challenges.extend(replay_challenges)

        # Compile final result
        all_ok = all(c.ok for c in checks)
        challenge_ref = self._find_challenge_ref(checks) if not all_ok else None

        return VerificationResult(
            ok=all_ok,
            checks=checks,
            challenge=challenge_ref,
        ), challenges

    def _check_strict_mode(
        self,
        package: PoRPackage,
    ) -> tuple[list[CheckResult], bool]:
        """Check strict mode requirements."""
        checks: list[CheckResult] = [
            check_strict_mode_flag(package.prompt_spec, self._strict_mode),
            check_output_schema_ref(package.prompt_spec),
            check_evaluation_variables(package.trace),
        ]
        return checks, all(c.ok for c in checks)

    def _verify_replay(
        self,
        package: PoRPackage,
    ) -> tuple[list[CheckResult], list[Challenge]]:
        """Verify via evidence replay (optional mode)."""
        checks: list[CheckResult] = []
        challenges: list[Challenge] = []

        if not self._replay_executor.is_enabled:
            checks.append(CheckResult(
                check_id="replay_mode",
                ok=True,
                severity="warn",
                message="Replay mode requested but collector not available",
                details={},
            ))
            return checks, challenges

        if package.tool_plan is None:
            checks.append(CheckResult(
                check_id="replay_tool_plan",
                ok=True,
                severity="warn",
                message="Replay mode requested but tool_plan not available",
                details={},
            ))
            return checks, challenges

        # Execute replay
        replay_result = self._replay_executor.replay_and_compare(
            package.prompt_spec,
            package.tool_plan,
            package.evidence,
        )
        checks.extend(replay_result.verification.checks)

        # Generate replay divergence challenge if needed
        if replay_result.divergent_items and replay_result.replayed_bundle:
            original_root = compute_evidence_root(package.evidence)
            replayed_root = compute_evidence_root(replay_result.replayed_bundle)

            challenges.append(make_replay_divergence_challenge(
                market_id=package.bundle.market_id,
                bundle_root=package.bundle.por_root or package.bundle.verdict_hash,
                original_evidence_root=original_root,
                replayed_evidence_root=replayed_root,
                divergent_items=replay_result.divergent_items,
            ))

        return checks, challenges

    def _find_challenge_ref(self, checks: list[CheckResult]) -> Optional[ChallengeRef]:
        """Find the first error check and create appropriate ChallengeRef."""
        for check in checks:
            if not check.ok:
                kind = "por_bundle"
                if "evidence" in check.check_id:
                    kind = "evidence_leaf"
                elif "reasoning" in check.check_id:
                    kind = "reasoning_leaf"
                elif "verdict" in check.check_id:
                    kind = "verdict_hash"
                return ChallengeRef(kind=kind, reason=check.message)
        return None

    def _make_structure_challenge(
        self,
        package: PoRPackage,
        result: VerificationResult,
    ) -> Challenge:
        """Create a challenge from structural verification failure."""
        return Challenge(
            challenge_id=f"ch_struct_{package.bundle.market_id[:8]}",
            kind="por_root",
            market_id=package.bundle.market_id,
            bundle_root=package.bundle.por_root or package.bundle.verdict_hash,
            details={
                "reason": "structural_validation_failed",
                "failed_checks": [c.check_id for c in result.checks if not c.ok],
            },
        )


__all__ = [
    "VerifyMode",
    "PoRPackage",
    "SentinelVerificationResult",
    "SentinelAgent",
]