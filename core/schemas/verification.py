"""
Module 01 - Schemas & Canonicalization
File: verification.py

Purpose: Standard result format for verification steps.
These schemas are used by Collector/Auditor/Judge/Sentinel to communicate
verification outcomes.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import CournotError


# Severity levels for checks
CheckSeverity = Literal["info", "warn", "error"]

# Types of challengeable artifacts
ChallengeKind = Literal["evidence_leaf", "reasoning_leaf", "verdict_hash", "por_bundle", "provenance_tier"]


class CheckResult(BaseModel):
    """
    Result of a single verification check.

    Checks are atomic verification steps that can pass or fail.
    """

    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(
        ...,
        description="Unique identifier for this check",
        min_length=1,
    )
    ok: bool = Field(
        ...,
        description="Whether the check passed",
    )
    severity: CheckSeverity = Field(
        ...,
        description="Severity level of this check",
    )
    message: str = Field(
        ...,
        description="Human-readable message describing the result",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the check",
    )

    @property
    def is_error(self) -> bool:
        """Check if this is an error-level failure."""
        return not self.ok and self.severity == "error"

    @property
    def is_warning(self) -> bool:
        """Check if this is a warning."""
        return self.severity == "warn"

    @classmethod
    def passed(
        cls,
        check_id: str,
        message: str = "Check passed",
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Create a passed check result."""
        return cls(
            check_id=check_id,
            ok=True,
            severity="info",
            message=message,
            details=details or {},
        )

    @classmethod
    def warning(
        cls,
        check_id: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Create a warning check result."""
        return cls(
            check_id=check_id,
            ok=True,  # Warnings don't fail the check
            severity="warn",
            message=message,
            details=details or {},
        )

    @classmethod
    def failed(
        cls,
        check_id: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "CheckResult":
        """Create a failed check result."""
        return cls(
            check_id=check_id,
            ok=False,
            severity="error",
            message=message,
            details=details or {},
        )


class ChallengeRef(BaseModel):
    """
    Reference to a challengeable artifact.

    Used when verification fails to identify what can be disputed.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ChallengeKind = Field(
        ...,
        description="Type of artifact being challenged",
    )
    leaf_index: int | None = Field(
        default=None,
        description="Index of the leaf in the Merkle tree (if applicable)",
    )
    evidence_id: str | None = Field(
        default=None,
        description="ID of the evidence item (for evidence_leaf challenges)",
    )
    step_id: str | None = Field(
        default=None,
        description="ID of the reasoning step (for reasoning_leaf challenges)",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for the challenge",
    )

    @classmethod
    def evidence_challenge(
        cls,
        evidence_id: str,
        leaf_index: int | None = None,
        reason: str | None = None,
    ) -> "ChallengeRef":
        """Create a challenge reference for an evidence leaf."""
        return cls(
            kind="evidence_leaf",
            evidence_id=evidence_id,
            leaf_index=leaf_index,
            reason=reason,
        )

    @classmethod
    def reasoning_challenge(
        cls,
        step_id: str,
        leaf_index: int | None = None,
        reason: str | None = None,
    ) -> "ChallengeRef":
        """Create a challenge reference for a reasoning leaf."""
        return cls(
            kind="reasoning_leaf",
            step_id=step_id,
            leaf_index=leaf_index,
            reason=reason,
        )

    @classmethod
    def verdict_challenge(
        cls,
        reason: str | None = None,
    ) -> "ChallengeRef":
        """Create a challenge reference for the verdict hash."""
        return cls(
            kind="verdict_hash",
            reason=reason,
        )

    @classmethod
    def bundle_challenge(
        cls,
        reason: str | None = None,
    ) -> "ChallengeRef":
        """Create a challenge reference for the PoR bundle."""
        return cls(
            kind="por_bundle",
            reason=reason,
        )


class VerificationResult(BaseModel):
    """
    Complete result of a verification process.

    This is the standard format for communicating verification outcomes
    between modules without using exceptions.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool = Field(
        ...,
        description="Overall verification success",
    )
    checks: list[CheckResult] = Field(
        default_factory=list,
        description="Individual check results",
    )
    challenge: ChallengeRef | None = Field(
        default=None,
        description="Reference to challengeable artifact if verification failed",
    )
    error: CournotError | None = Field(
        default=None,
        description="Error details if verification encountered an exception",
    )

    @property
    def has_errors(self) -> bool:
        """Check if any checks failed with error severity."""
        return any(check.is_error for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        """Check if any checks produced warnings."""
        return any(check.is_warning for check in self.checks)

    @property
    def error_count(self) -> int:
        """Count of error-level failures."""
        return sum(1 for check in self.checks if check.is_error)

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return sum(1 for check in self.checks if check.is_warning)

    @property
    def passed_count(self) -> int:
        """Count of passed checks."""
        return sum(1 for check in self.checks if check.ok)

    def get_failed_checks(self) -> list[CheckResult]:
        """Get all failed checks."""
        return [check for check in self.checks if not check.ok]

    def get_error_messages(self) -> list[str]:
        """Get all error messages."""
        return [check.message for check in self.checks if check.is_error]

    @classmethod
    def success(cls, checks: list[CheckResult] | None = None) -> "VerificationResult":
        """Create a successful verification result."""
        return cls(ok=True, checks=checks or [])

    @classmethod
    def failure(
        cls,
        checks: list[CheckResult],
        challenge: ChallengeRef | None = None,
        error: CournotError | None = None,
    ) -> "VerificationResult":
        """Create a failed verification result."""
        return cls(
            ok=False,
            checks=checks,
            challenge=challenge,
            error=error,
        )

    @classmethod
    def from_error(cls, error: CournotError) -> "VerificationResult":
        """Create a verification result from an error."""
        return cls(ok=False, checks=[], error=error)

    def add_check(self, check: CheckResult) -> None:
        """Add a check result."""
        self.checks.append(check)
        # Update ok status if we added a failed check
        if not check.ok:
            self.ok = False

    def merge(self, other: "VerificationResult") -> "VerificationResult":
        """Merge another verification result into this one."""
        return VerificationResult(
            ok=self.ok and other.ok,
            checks=self.checks + other.checks,
            challenge=self.challenge or other.challenge,
            error=self.error or other.error,
        )


class ProvenanceCheck(BaseModel):
    """
    Result of checking evidence provenance.

    Used specifically for Collector verification.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(
        ...,
        description="ID of the evidence item checked",
    )
    tier_verified: bool = Field(
        ...,
        description="Whether the claimed tier was verified",
    )
    actual_tier: int = Field(
        ...,
        description="The actual provenance tier determined",
    )
    proof_valid: bool = Field(
        ...,
        description="Whether the provenance proof is valid",
    )
    message: str = Field(
        default="",
        description="Additional information about the check",
    )


class TraceConsistencyCheck(BaseModel):
    """
    Result of checking reasoning trace consistency.

    Used specifically for Auditor verification.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(
        ...,
        description="ID of the reasoning trace checked",
    )
    all_evidence_referenced: bool = Field(
        ...,
        description="Whether all evidence references are valid",
    )
    no_circular_dependencies: bool = Field(
        ...,
        description="Whether there are no circular step dependencies",
    )
    no_contradictions: bool = Field(
        ...,
        description="Whether there are no logical contradictions",
    )
    invalid_references: list[str] = Field(
        default_factory=list,
        description="List of invalid evidence/step references",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="List of detected contradictions",
    )