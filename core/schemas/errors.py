"""
Module 01 - Schemas & Canonicalization
File: errors.py

Purpose: Standard error taxonomy across the Cournot pipeline.
Defines both Pydantic models for structured error communication
and Python exceptions for control flow.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Error Codes (Machine-Readable Constants)
# =============================================================================

class ErrorCodes:
    """Stable machine-readable error codes used across the pipeline."""

    # Schema & Validation Errors
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    CANONICALIZATION_ERROR = "CANONICALIZATION_ERROR"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    
    # Prompt Errors
    PROMPT_COMPILATION_ERROR = "PROMPT_COMPILATION_ERROR"

    # Evidence & Provenance Errors
    EVIDENCE_POLICY_VIOLATION = "EVIDENCE_POLICY_VIOLATION"
    PROVENANCE_VERIFICATION_FAILED = "PROVENANCE_VERIFICATION_FAILED"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    EVIDENCE_STALE = "EVIDENCE_STALE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    
    TIER_POLICY_VIOLATION = "TIER_POLICY_VIOLATION"
    SELECTION_POLICY_VIOLATION = "SELECTION_POLICY_VIOLATION"
    QUORUM_NOT_MET = "QUORUM_NOT_MET"

    # Reasoning & Trace Errors
    TRACE_POLICY_VIOLATION = "TRACE_POLICY_VIOLATION"
    REASONING_CONTRADICTION = "REASONING_CONTRADICTION"
    EVIDENCE_REFERENCE_INVALID = "EVIDENCE_REFERENCE_INVALID"
    STEP_DEPENDENCY_MISSING = "STEP_DEPENDENCY_MISSING"

    # Verdict & Determinism Errors
    DETERMINISM_VIOLATION = "DETERMINISM_VIOLATION"
    VERDICT_SCHEMA_MISMATCH = "VERDICT_SCHEMA_MISMATCH"
    CONFIDENCE_OUT_OF_BOUNDS = "CONFIDENCE_OUT_OF_BOUNDS"

    # Time & Window Errors
    TIME_WINDOW_VIOLATION = "TIME_WINDOW_VIOLATION"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    TIMESTAMP_INVALID = "TIMESTAMP_INVALID"

    # Merkle & Commitment Errors
    MERKLE_PROOF_INVALID = "MERKLE_PROOF_INVALID"
    ROOT_MISMATCH = "ROOT_MISMATCH"
    LEAF_HASH_MISMATCH = "LEAF_HASH_MISMATCH"

    # Pipeline & Execution Errors
    SOP_EXECUTION_ERROR = "SOP_EXECUTION_ERROR"
    STAGE_TIMEOUT = "STAGE_TIMEOUT"
    AGENT_FAILURE = "AGENT_FAILURE"

    # Challenge & Verification Errors
    CHALLENGE_INVALID = "CHALLENGE_INVALID"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"


# =============================================================================
# Pydantic Error Models (Structured Communication)
# =============================================================================

class CournotError(BaseModel):
    """
    Base error model for structured error communication across the pipeline.

    This model is used for passing errors between modules without exceptions,
    enabling structured error handling and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
    )

    code: str = Field(
        ...,
        description="Stable machine-readable error code",
        examples=[ErrorCodes.SCHEMA_VALIDATION_ERROR],
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured details about the error",
    )
    retryable: bool = Field(
        default=False,
        description="Whether the operation can be retried",
    )

    def to_exception(self) -> "CournotException":
        """Convert this error model to a raised exception."""
        return CournotException(
            code=self.code,
            message=self.message,
            details=self.details,
            retryable=self.retryable,
        )


class ValidationError(CournotError):
    """Error model for schema/data validation failures."""

    code: str = Field(default=ErrorCodes.SCHEMA_VALIDATION_ERROR)
    field_path: str | None = Field(
        default=None,
        description="Path to the field that failed validation",
    )
    expected: str | None = Field(
        default=None,
        description="Expected value or type",
    )
    actual: str | None = Field(
        default=None,
        description="Actual value received",
    )


class ProvenanceError(CournotError):
    """Error model for provenance verification failures."""

    code: str = Field(default=ErrorCodes.PROVENANCE_VERIFICATION_FAILED)
    evidence_id: str | None = Field(
        default=None,
        description="ID of the evidence item that failed verification",
    )
    tier: int | None = Field(
        default=None,
        description="Provenance tier that was expected/required",
    )


class TimeWindowError(CournotError):
    """Error model for time window violations."""

    code: str = Field(default=ErrorCodes.TIME_WINDOW_VIOLATION)
    expected_start: str | None = Field(default=None)
    expected_end: str | None = Field(default=None)
    actual_time: str | None = Field(default=None)


# =============================================================================
# Python Exceptions (Control Flow)
# =============================================================================

class CournotException(Exception):
    """
    Base exception for all Cournot protocol errors.

    This exception carries structured error information and can be
    converted to/from CournotError models.
    """

    def __init__(
        self,
        message: str,
        code: str = "COURNOT_ERROR",
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_error_model(self) -> CournotError:
        """Convert this exception to a CournotError model."""
        return CournotError(
            code=self.code,
            message=self.message,
            details=self.details,
            retryable=self.retryable,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class CanonicalizationException(CournotException):
    """Exception raised when canonical serialization fails."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ErrorCodes.CANONICALIZATION_ERROR,
            details=details,
            retryable=False,
        )


class SchemaValidationException(CournotException):
    """Exception raised when schema validation fails."""

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_details = details or {}
        if field_path:
            full_details["field_path"] = field_path
        super().__init__(
            message=message,
            code=ErrorCodes.SCHEMA_VALIDATION_ERROR,
            details=full_details,
            retryable=False,
        )


class ProvenanceException(CournotException):
    """Exception raised when provenance verification fails."""

    def __init__(
        self,
        message: str,
        evidence_id: str | None = None,
        tier: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_details = details or {}
        if evidence_id:
            full_details["evidence_id"] = evidence_id
        if tier is not None:
            full_details["tier"] = tier
        super().__init__(
            message=message,
            code=ErrorCodes.PROVENANCE_VERIFICATION_FAILED,
            details=full_details,
            retryable=False,
        )


class TimeWindowException(CournotException):
    """Exception raised when time window constraints are violated."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ErrorCodes.TIME_WINDOW_VIOLATION,
            details=details,
            retryable=False,
        )


class DeterminismException(CournotException):
    """Exception raised when determinism constraints are violated."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ErrorCodes.DETERMINISM_VIOLATION,
            details=details,
            retryable=False,
        )


class TracePolicyException(CournotException):
    """Exception raised when trace policy is violated."""

    def __init__(
        self,
        message: str,
        step_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_details = details or {}
        if step_id:
            full_details["step_id"] = step_id
        super().__init__(
            message=message,
            code=ErrorCodes.TRACE_POLICY_VIOLATION,
            details=full_details,
            retryable=False,
        )


class MerkleVerificationException(CournotException):
    """Exception raised when Merkle proof verification fails."""

    def __init__(
        self,
        message: str,
        leaf_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_details = details or {}
        if leaf_index is not None:
            full_details["leaf_index"] = leaf_index
        super().__init__(
            message=message,
            code=ErrorCodes.MERKLE_PROOF_INVALID,
            details=full_details,
            retryable=False,
        )


class SOPExecutionException(CournotException):
    """Exception raised when SOP execution fails."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        full_details = details or {}
        if stage:
            full_details["stage"] = stage
        super().__init__(
            message=message,
            code=ErrorCodes.SOP_EXECUTION_ERROR,
            details=full_details,
            retryable=retryable,
        )

class PromptCompilationException(CournotException):
    """Exception raised when prompt compilation fails."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=ErrorCodes.PROMPT_COMPILATION_ERROR,
            details=details,
            retryable=False,
        )