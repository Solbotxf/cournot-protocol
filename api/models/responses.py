"""
Module 09D - API Response Models

Pydantic models for API response serialization.
"""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /health endpoint."""
    
    ok: bool = True
    service: str = "cournot-protocol-api"
    version: str = "v1"


class RunSummary(BaseModel):
    """Summary of pipeline execution."""
    
    market_id: str = Field(..., description="Market identifier")
    outcome: str = Field(..., description="Resolution outcome: YES, NO, or INVALID")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    por_root: str = Field(..., description="Proof of Reasoning root hash")
    prompt_spec_hash: str | None = Field(default=None)
    evidence_root: str | None = Field(default=None)
    reasoning_root: str | None = Field(default=None)
    execution_mode: str = Field(default="development", description="Execution mode used")


class VerificationInfo(BaseModel):
    """Verification results from sentinel."""
    
    sentinel_ok: bool = Field(..., description="Whether sentinel verification passed")
    total_checks: int = Field(default=0, description="Total number of checks performed")
    passed_checks: int = Field(default=0, description="Number of passed checks")
    failed_checks: int = Field(default=0, description="Number of failed checks")
    checks: list[dict[str, Any]] = Field(default_factory=list)
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    """Response for POST /run endpoint (JSON format)."""
    
    ok: bool = Field(..., description="Whether the pipeline executed successfully")
    summary: RunSummary = Field(..., description="Execution summary")
    artifacts: dict[str, Any] | None = Field(
        default=None,
        description="Full artifacts (prompt_spec, tool_plan, evidence_bundle, etc.)",
    )
    verification: VerificationInfo | None = Field(
        default=None,
        description="Verification results",
    )
    errors: list[str] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    """Response for POST /verify endpoint."""
    
    ok: bool = Field(..., description="Overall verification status")
    hashes_ok: bool = Field(..., description="Whether file hashes match manifest")
    semantic_ok: bool = Field(..., description="Whether semantic validation passed")
    sentinel_ok: bool | None = Field(
        default=None,
        description="Whether sentinel verification passed (if enabled)",
    )
    por_root: str = Field(..., description="PoR root from the pack")
    market_id: str = Field(..., description="Market ID from the pack")
    outcome: str = Field(..., description="Verdict outcome from the pack")
    checks: list[dict[str, Any]] = Field(default_factory=list)
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DivergenceInfo(BaseModel):
    """Information about evidence divergence during replay."""
    
    expected_evidence_root: str = Field(..., description="Original evidence root")
    replayed_evidence_root: str = Field(..., description="Replayed evidence root")
    divergences: list[dict[str, Any]] = Field(default_factory=list)


class ReplayResponse(BaseModel):
    """Response for POST /replay endpoint."""
    
    ok: bool = Field(..., description="Overall replay verification status")
    replay_ok: bool = Field(..., description="Whether replay matched original evidence")
    sentinel_ok: bool = Field(..., description="Whether sentinel verification passed")
    por_root: str = Field(..., description="PoR root from the pack")
    market_id: str = Field(..., description="Market ID from the pack")
    divergence: DivergenceInfo | None = Field(
        default=None,
        description="Divergence details if replay didn't match",
    )
    checks: list[dict[str, Any]] = Field(default_factory=list)
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    ok: bool = False
    error: ErrorDetail = Field(..., description="Error details")
