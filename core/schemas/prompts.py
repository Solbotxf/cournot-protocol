"""
Module 01 - Schemas & Canonicalization
File: prompts.py

Purpose: Prompt Engineer output contracts.
These schemas define the structured output from the Prompt Engineer agent.

Key additions (v1.1):
- SourceTarget: Explicit data fetching instructions
- SelectionPolicy: Strategy for selecting from multiple sources
- DataRequirement: Extended with source_targets and selection_policy
- PromptSpec.created_at: Now OPTIONAL (defaults to None) for deterministic hashing
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .market import MarketSpec
from .versioning import SCHEMA_VERSION

# Use forward reference to avoid circular imports
if TYPE_CHECKING:
    from .transport import ToolPlan


# Type aliases for SourceTarget and SelectionPolicy
HttpMethod = Literal["GET", "POST", "RPC", "WS", "OTHER"]
ContentTypeHint = Literal["json", "text", "html", "bytes"]
SelectionStrategy = Literal["single_best", "multi_source_quorum", "fallback_chain"]
TieBreakerStrategy = Literal["highest_provenance", "source_priority", "most_recent"]


class SourceTarget(BaseModel):
    """
    Explicit data fetching instruction for a single source.

    The Prompt Engineer MUST express executable data fetching instructions
    explicitly. This schema captures a fully specified target endpoint.

    IMPORTANT:
    - uri MUST be a fully specified absolute URI when applicable
    - uri is treated as an opaque string for commitments (no normalization)
    - Order of SourceTargets in a list is semantically meaningful (priority/fallback)
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(
        ...,
        description="Logical source adapter ID (e.g., 'http', 'web', 'polymarket', 'exchange')",
        min_length=1,
    )
    uri: str = Field(
        ...,
        description="Explicit URL/endpoint. MUST be a fully specified absolute URI when applicable. "
        "Treated as opaque string - no normalization permitted.",
        min_length=1,
    )
    method: HttpMethod = Field(
        default="GET",
        description="HTTP method or protocol action",
    )
    expected_content_type: ContentTypeHint = Field(
        default="json",
        description="Expected content type of the response",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers to include in the request",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameters or request parameters",
    )
    body: dict[str, Any] | str | None = Field(
        default=None,
        description="Request body for POST/RPC methods",
    )
    auth_ref: str | None = Field(
        default=None,
        description="Reference to authentication config. Secrets are NEVER stored in schemas.",
    )
    cache_ttl_seconds: int | None = Field(
        default=None,
        description="Cache TTL hint in seconds. None means no caching preference.",
        ge=0,
    )
    notes: str | None = Field(
        default=None,
        description="Additional notes about this source target",
    )


class SelectionPolicy(BaseModel):
    """
    Policy for selecting evidence from multiple source targets.

    Defines how the Collector should handle multiple sources for a single
    data requirement.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: SelectionStrategy = Field(
        ...,
        description="Strategy for source selection: "
        "'single_best' = use highest priority source that succeeds, "
        "'multi_source_quorum' = require quorum of sources to agree, "
        "'fallback_chain' = try sources in order until success",
    )
    min_sources: int = Field(
        default=1,
        description="Minimum number of sources required",
        ge=1,
    )
    max_sources: int = Field(
        default=3,
        description="Maximum number of sources to query",
        ge=1,
    )
    quorum: int = Field(
        default=1,
        description="Number of sources that must agree (for multi_source_quorum strategy)",
        ge=1,
    )
    tie_breaker: TieBreakerStrategy = Field(
        default="highest_provenance",
        description="How to break ties when sources disagree: "
        "'highest_provenance' = prefer higher provenance tier, "
        "'source_priority' = prefer earlier source in list, "
        "'most_recent' = prefer most recent data",
    )

    @model_validator(mode="after")
    def validate_quorum_bounds(self) -> "SelectionPolicy":
        """Ensure quorum is within valid bounds."""
        if self.quorum > self.max_sources:
            raise ValueError(
                f"quorum ({self.quorum}) cannot exceed max_sources ({self.max_sources})"
            )
        if self.min_sources > self.max_sources:
            raise ValueError(
                f"min_sources ({self.min_sources}) cannot exceed max_sources ({self.max_sources})"
            )
        return self


class PredictionSemantics(BaseModel):
    """
    Semantic breakdown of what is being predicted.

    This captures the structured meaning of the prediction question.
    """

    model_config = ConfigDict(extra="forbid")

    target_entity: str = Field(
        ...,
        description="The entity being predicted about (e.g., company, person, event)",
        min_length=1,
    )
    predicate: str = Field(
        ...,
        description="The predicate or condition being evaluated",
        min_length=1,
    )
    threshold: str | None = Field(
        default=None,
        description="Threshold or boundary condition if applicable",
    )
    timeframe: str | None = Field(
        default=None,
        description="Time frame for the prediction",
    )


class DataRequirement(BaseModel):
    """
    A single data requirement for resolving the prediction.

    Specifies what data is needed and from where it should be sourced.

    IMPORTANT:
    - source_targets is REQUIRED and MUST contain at least 1 entry
    - source_targets order is semantically meaningful (priority/fallback order)
    - source_targets order MUST be preserved (not sorted) during serialization
    - selection_policy is REQUIRED
    """

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(
        ...,
        description="Unique identifier for this requirement",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="Human-readable description of what data is needed",
        min_length=1,
    )
    source_targets: list[SourceTarget] = Field(
        ...,
        description="Ordered list of source targets. Order is meaningful (priority/fallback). "
        "MUST contain at least 1 entry.",
        min_length=1,
    )
    selection_policy: SelectionPolicy = Field(
        ...,
        description="Policy for selecting from multiple sources",
    )
    min_provenance_tier: int = Field(
        default=0,
        description="Minimum provenance tier required for this data",
        ge=0,
    )
    
    expected_fields: list[str] | None = Field(
        default=None,
        description="Optional list of expected JSON keys; None means no filtering.",
    )

    # Legacy field kept for backward compatibility
    preferred_sources: list[str] = Field(
        default_factory=list,
        description="[DEPRECATED] Use source_targets instead. List of preferred source IDs.",
    )

    @model_validator(mode="after")
    def validate_source_targets(self) -> "DataRequirement":
        """Ensure source_targets has at least one entry."""
        if not self.source_targets:
            raise ValueError("source_targets MUST contain at least 1 entry")
        return self

    def get_source_target_by_id(self, source_id: str) -> SourceTarget | None:
        """Find a source target by its source_id."""
        for target in self.source_targets:
            if target.source_id == source_id:
                return target
        return None

    def get_source_uris(self) -> list[str]:
        """Get all URIs from source targets in order."""
        return [target.uri for target in self.source_targets]


class PromptSpec(BaseModel):
    """
    Complete specification output from the Prompt Engineer.

    This is the structured prompt that drives the rest of the pipeline.
    It must be unambiguous and contain all information needed for resolution.

    IMPORTANT - Timestamp Determinism:
    - created_at is OPTIONAL and defaults to None
    - When created_at is None, it is excluded from canonical serialization
    - This ensures identical user inputs produce identical prompt_spec_hash
    - If timestamps are needed, store them outside the committed object
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for this model",
    )
    task_type: str = Field(
        default="prediction_resolution",
        description="Type of task being performed",
    )
    market: MarketSpec = Field(
        ...,
        description="The market specification being resolved",
    )
    prediction_semantics: PredictionSemantics = Field(
        ...,
        description="Semantic breakdown of the prediction",
    )
    data_requirements: list[DataRequirement] = Field(
        default_factory=list,
        description="List of data requirements for resolution",
    )
    output_schema_ref: str = Field(
        default="core.schemas.verdict.DeterministicVerdict",
        description="Reference to the output schema for the verdict",
    )
    forbidden_behaviors: list[str] = Field(
        default_factory=list,
        description="List of behaviors that must not occur during resolution",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp when this spec was created. "
        "OPTIONAL - defaults to None for deterministic hashing. "
        "When None, excluded from canonical serialization.",
    )
    tool_plan: Any | None = Field(
        default=None,
        description="Optional tool execution plan (ToolPlan type)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional specification parameters",
    )

    def get_requirement_by_id(self, requirement_id: str) -> DataRequirement | None:
        """Find a data requirement by its ID."""
        for req in self.data_requirements:
            if req.requirement_id == requirement_id:
                return req
        return None

    def has_forbidden_behavior(self, behavior: str) -> bool:
        """Check if a behavior is in the forbidden list."""
        return behavior.lower() in [b.lower() for b in self.forbidden_behaviors]

    @property
    def market_id(self) -> str:
        """Convenience accessor for the market ID."""
        return self.market.market_id

    def get_all_source_targets(self) -> list[SourceTarget]:
        """Get all source targets from all data requirements."""
        targets = []
        for req in self.data_requirements:
            targets.extend(req.source_targets)
        return targets


class PromptValidationResult(BaseModel):
    """
    Result of validating a prompt specification.

    Used to report any issues with a PromptSpec before execution.
    """

    model_config = ConfigDict(extra="forbid")

    valid: bool = Field(
        ...,
        description="Whether the prompt spec is valid",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of validation errors",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of validation warnings",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="List of required fields that are missing or invalid",
    )

    @classmethod
    def success(cls) -> "PromptValidationResult":
        """Create a successful validation result."""
        return cls(valid=True)

    @classmethod
    def failure(
        cls,
        errors: list[str],
        warnings: list[str] | None = None,
        missing_fields: list[str] | None = None,
    ) -> "PromptValidationResult":
        """Create a failed validation result."""
        return cls(
            valid=False,
            errors=errors,
            warnings=warnings or [],
            missing_fields=missing_fields or [],
        )