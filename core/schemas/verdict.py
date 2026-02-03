"""
Module 01 - Schemas & Canonicalization
File: verdict.py

Purpose: Deterministic settlement output schemas.
These models define the verdict structure produced by the Judge agent.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .versioning import SCHEMA_VERSION


# The canonical outcome type for predictions.
# For binary markets: "YES", "NO", or "INVALID".
# For multi-choice markets: one of the enumerated outcomes or "INVALID".
Outcome = str

# Kept for type hints in binary-only contexts
BinaryOutcome = Literal["YES", "NO", "INVALID"]


class DeterministicVerdict(BaseModel):
    """
    The deterministic verdict for a prediction market resolution.

    This is the final output of the Judge agent and must be:
    - Schema-locked (exact fields, no deviation)
    - Deterministically serializable
    - Suitable for on-chain settlement
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for this model",
    )
    market_id: str = Field(
        ...,
        description="ID of the market being resolved",
        min_length=1,
    )
    outcome: Outcome = Field(
        ...,
        description="The resolution outcome. For binary: YES, NO, or INVALID. "
        "For multi-choice: one of the enumerated outcomes or INVALID.",
    )
    confidence: float = Field(
        ...,
        description="Confidence in the verdict (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    resolution_time: datetime | None = Field(
        default=None,
        description="Timestamp when the resolution was determined. "
        "OPTIONAL - defaults to None for deterministic hashing. "
        "When None, excluded from canonical serialization.",
    )
    resolution_rule_id: str = Field(
        ...,
        description="ID of the rule that determined this verdict",
        min_length=1,
    )
    prompt_spec_hash: str | None = Field(
        default=None,
        description="Hash of the PromptSpec used for resolution",
    )
    evidence_root: str | None = Field(
        default=None,
        description="Merkle root of the evidence bundle",
    )
    reasoning_root: str | None = Field(
        default=None,
        description="Merkle root of the reasoning trace",
    )
    justification_hash: str | None = Field(
        default=None,
        description="Hash of the full justification/reasoning",
    )
    selected_leaf_refs: list[str] = Field(
        default_factory=list,
        description="References to selected evidence/reasoning leaves",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional verdict metadata",
    )

    @field_validator("resolution_rule_id")
    @classmethod
    def validate_rule_id_not_empty(cls, v: str) -> str:
        """Ensure resolution_rule_id is not just whitespace."""
        if not v.strip():
            raise ValueError("resolution_rule_id must not be empty or whitespace")
        return v

    @property
    def is_definitive(self) -> bool:
        """Check if the verdict is definitive (not INVALID)."""
        return self.outcome != "INVALID"

    @property
    def is_invalid(self) -> bool:
        """Check if the verdict is INVALID."""
        return self.outcome == "INVALID"

    @property
    def has_merkle_roots(self) -> bool:
        """Check if Merkle roots are present."""
        return self.evidence_root is not None and self.reasoning_root is not None


class VerdictJustification(BaseModel):
    """
    Detailed justification for a verdict.

    This provides human-readable explanation and references
    supporting the DeterministicVerdict.
    """

    model_config = ConfigDict(extra="forbid")

    verdict_id: str = Field(
        ...,
        description="Reference to the verdict this justifies",
    )
    summary: str = Field(
        ...,
        description="Summary of the justification",
    )
    key_evidence_refs: list[str] = Field(
        default_factory=list,
        description="References to key evidence items",
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Key reasoning step references",
    )
    rule_application: str = Field(
        ...,
        description="Explanation of how the resolution rule was applied",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Any caveats or limitations",
    )


class VerdictValidationResult(BaseModel):
    """
    Result of validating a verdict against schema and policy.

    Used by the Judge verifier to confirm verdict validity.
    """

    model_config = ConfigDict(extra="forbid")

    valid: bool = Field(
        ...,
        description="Whether the verdict is valid",
    )
    schema_compliant: bool = Field(
        ...,
        description="Whether the verdict matches the expected schema",
    )
    confidence_in_bounds: bool = Field(
        ...,
        description="Whether confidence is within valid bounds",
    )
    rule_id_valid: bool = Field(
        ...,
        description="Whether the resolution rule ID is valid",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of validation errors",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of validation warnings",
    )

    @classmethod
    def success(cls) -> "VerdictValidationResult":
        """Create a successful validation result."""
        return cls(
            valid=True,
            schema_compliant=True,
            confidence_in_bounds=True,
            rule_id_valid=True,
        )

    @classmethod
    def failure(cls, errors: list[str]) -> "VerdictValidationResult":
        """Create a failed validation result."""
        return cls(
            valid=False,
            schema_compliant=False,
            confidence_in_bounds=True,  # May be overridden
            rule_id_valid=True,  # May be overridden
            errors=errors,
        )