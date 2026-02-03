"""
Module 01 - Schemas & Canonicalization
File: market.py

Purpose: Market specification and resolution rules.
These schemas define the contract for prediction market configuration
and resolution policy.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .versioning import SCHEMA_VERSION


class ResolutionWindow(BaseModel):
    """
    Time window during which market resolution can occur.

    The resolution must happen between start and end times.
    """

    model_config = ConfigDict(extra="forbid")

    start: datetime = Field(
        ...,
        description="Start of the resolution window (UTC)",
    )
    end: datetime = Field(
        ...,
        description="End of the resolution window (UTC)",
    )


class SourcePolicy(BaseModel):
    """
    Policy for a data source used in resolution.

    Defines what sources are allowed and their minimum provenance requirements.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(
        ...,
        description="Unique identifier for the source",
        min_length=1,
    )
    kind: str = Field(
        ...,
        description="Type of source (e.g., 'api', 'web', 'chain', 'manual')",
    )
    allow: bool = Field(
        default=True,
        description="Whether this source is allowed",
    )
    min_provenance_tier: int = Field(
        default=0,
        description="Minimum provenance tier required from this source",
        ge=0,
    )
    notes: str | None = Field(
        default=None,
        description="Additional notes about the source policy",
    )


class ResolutionRule(BaseModel):
    """
    A single resolution rule that maps evidence to verdict.

    Rules are evaluated in priority order (higher priority first).
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(
        ...,
        description="Unique identifier for this rule",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="Human-readable description of the rule",
    )
    priority: int = Field(
        default=0,
        description="Rule priority (higher values = higher priority)",
    )


class ResolutionRules(BaseModel):
    """
    Collection of resolution rules for a market.

    Rules define how evidence is mapped to YES/NO/INVALID verdicts.
    """

    model_config = ConfigDict(extra="forbid")

    rules: list[ResolutionRule] = Field(
        default_factory=list,
        description="List of resolution rules",
    )

    def get_sorted_rules(self) -> list[ResolutionRule]:
        """Return rules sorted by priority (highest first)."""
        return sorted(self.rules, key=lambda r: r.priority, reverse=True)

    def get_rule_by_id(self, rule_id: str) -> ResolutionRule | None:
        """Find a rule by its ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None


class DisputePolicy(BaseModel):
    """
    Policy for handling disputes after initial resolution.

    Defines the window and conditions for challenging a verdict.
    """

    model_config = ConfigDict(extra="forbid")

    dispute_window_seconds: int = Field(
        ...,
        description="Duration of the dispute window in seconds",
        gt=0,
    )
    allow_challenges: bool = Field(
        default=True,
        description="Whether challenges are allowed",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional dispute policy parameters",
    )


class MarketSpec(BaseModel):
    """
    Complete specification for a prediction market.

    This is the primary contract for market configuration and must be
    unambiguous and deterministic.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for this model",
    )
    market_id: str = Field(
        ...,
        description="Unique identifier for the market",
        min_length=1,
    )
    question: str = Field(
        ...,
        description="The prediction question being asked",
        min_length=1,
    )
    event_definition: str = Field(
        ...,
        description="Unambiguous definition of the event being predicted. "
        "MUST be fully explicit and leave no room for interpretation.",
        min_length=1,
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for interpreting time-based conditions",
    )
    resolution_deadline: datetime = Field(
        ...,
        description="Deadline by which the market must be resolved",
    )
    resolution_window: ResolutionWindow = Field(
        ...,
        description="Time window during which resolution can occur",
    )
    resolution_rules: ResolutionRules = Field(
        ...,
        description="Rules for mapping evidence to verdict",
    )
    allowed_sources: list[SourcePolicy] = Field(
        default_factory=list,
        description="List of allowed data sources for resolution",
    )
    min_provenance_tier: int = Field(
        default=0,
        description="Global minimum provenance tier required",
        ge=0,
    )
    dispute_policy: DisputePolicy = Field(
        ...,
        description="Policy for handling disputes",
    )
    market_type: Literal["binary", "multi_choice"] = Field(
        default="binary",
        description="Type of market: 'binary' for YES/NO, 'multi_choice' for >2 discrete outcomes",
    )
    possible_outcomes: list[str] = Field(
        default_factory=lambda: ["YES", "NO"],
        description="List of possible outcomes. For binary: ['YES', 'NO']. For multi_choice: 2+ custom outcomes.",
        min_length=2,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional market metadata",
    )

    @model_validator(mode="after")
    def validate_market_type_outcomes(self) -> "MarketSpec":
        """Validate that market_type and possible_outcomes are consistent."""
        if self.market_type == "binary":
            if self.possible_outcomes != ["YES", "NO"]:
                raise ValueError(
                    "Binary markets must have possible_outcomes=['YES', 'NO']"
                )
        else:  # multi_choice
            if len(self.possible_outcomes) < 2:
                raise ValueError(
                    "Multi-choice markets must have at least 2 possible outcomes"
                )
            if "INVALID" in self.possible_outcomes:
                raise ValueError(
                    "'INVALID' must not be in possible_outcomes (it is always implicitly allowed)"
                )
        return self

    def get_valid_outcomes(self) -> list[str]:
        """Return all valid outcomes including INVALID."""
        return self.possible_outcomes + ["INVALID"]

    def is_valid_outcome(self, outcome: str) -> bool:
        """Check if an outcome is valid for this market."""
        return outcome in self.get_valid_outcomes()

    @field_validator("resolution_rules", mode="after")
    @classmethod
    def validate_has_rules(cls, v: ResolutionRules) -> ResolutionRules:
        """Ensure at least one resolution rule is defined."""
        if not v.rules:
            raise ValueError("At least one resolution rule must be defined")
        return v

    def get_allowed_source_ids(self) -> set[str]:
        """Get the set of allowed source IDs."""
        return {sp.source_id for sp in self.allowed_sources if sp.allow}

    def is_source_allowed(self, source_id: str) -> bool:
        """Check if a source is allowed for this market."""
        for policy in self.allowed_sources:
            if policy.source_id == source_id:
                return policy.allow
        # If not explicitly listed, default to not allowed
        return False

    def get_source_min_tier(self, source_id: str) -> int:
        """Get the minimum provenance tier for a specific source."""
        for policy in self.allowed_sources:
            if policy.source_id == source_id:
                return max(policy.min_provenance_tier, self.min_provenance_tier)
        return self.min_provenance_tier