"""
Module 01 - Schemas & Canonicalization
File: evidence.py

Purpose: Evidence bundle and provenance schemas.
These models define the evidence structure used by the Collector agent.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .versioning import SCHEMA_VERSION

# Provenance tier definitions
# Tier 0: Self-reported / unverified
# Tier 1: Single third-party source
# Tier 2: Multiple independent sources
# Tier 3: Official/authoritative source
# Tier 4: Cryptographically verifiable (on-chain)
ProvenanceTier = Literal[0, 1, 2, 3, 4]


class EvidenceSource(BaseModel):
    """
    A single source cited in collected evidence.

    Standardized across all collectors so frontends can parse
    evidence_sources consistently.
    """

    model_config = ConfigDict(extra="allow")

    url: str = Field(..., description="URL of the source")
    source_id: str | None = Field(default=None, description="Short identifier (e.g. '[1]')")
    credibility_tier: int = Field(
        default=3,
        description="Credibility tier: 1=authoritative, 2=reputable, 3=low-confidence",
        ge=1,
        le=3,
    )
    key_fact: str = Field(default="", description="Key fact or quote extracted from this source")
    supports: str = Field(
        default="N/A",
        description="Whether this source supports resolution: YES, NO, or N/A",
    )
    date_published: str | None = Field(default=None, description="Publication date (YYYY-MM-DD)")


class Provenance(BaseModel):
    """
    Provenance metadata for a piece of evidence.
    
    Tracks where data came from and how trustworthy it is.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    source_id: str = Field(..., description="Identifier of the data source", min_length=1)
    source_uri: str = Field(..., description="URI from which data was fetched", min_length=1)
    tier: ProvenanceTier = Field(default=1, description="Provenance tier (0-4)")
    fetched_at: datetime | None = Field(
        default=None,
        description="When the data was fetched (OPTIONAL for determinism)",
    )
    receipt_id: str | None = Field(default=None, description="Reference to HTTP receipt")
    content_hash: str | None = Field(default=None, description="Hash of raw response content")
    cache_hit: bool = Field(default=False, description="Whether this was served from cache")
    ttl_seconds: int | None = Field(default=None, description="Cache TTL if applicable")
    
    @property
    def is_authoritative(self) -> bool:
        """Check if this is from an authoritative source (tier 3+)."""
        return self.tier >= 3
    
    @property
    def is_verifiable(self) -> bool:
        """Check if this is cryptographically verifiable (tier 4)."""
        return self.tier >= 4


class EvidenceItem(BaseModel):
    """
    A single piece of evidence collected from a source.
    
    Contains the raw data, parsed value, and provenance.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    evidence_id: str = Field(..., description="Unique identifier for this evidence", min_length=1)
    requirement_id: str = Field(..., description="ID of the DataRequirement this fulfills", min_length=1)
    provenance: Provenance = Field(..., description="Source and trust metadata")
    
    # Content
    raw_content: str | None = Field(default=None, description="Raw response content (truncated if large)")
    content_type: str = Field(default="application/json", description="MIME type of content")
    parsed_value: Any = Field(default=None, description="Parsed/extracted value")
    
    # Status
    success: bool = Field(default=True, description="Whether fetch was successful")
    error: str | None = Field(default=None, description="Error message if failed")
    status_code: int | None = Field(default=None, description="HTTP status code")
    
    # Extraction
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Specific fields extracted from response",
    )
    
    @property
    def is_valid(self) -> bool:
        """Check if this evidence is valid and usable."""
        return self.success and self.error is None
    
    @property
    def meets_tier(self) -> int:
        """Get the provenance tier."""
        return self.provenance.tier
    
    def get_field(self, field_name: str, default: Any = None) -> Any:
        """Get an extracted field value."""
        return self.extracted_fields.get(field_name, default)


class EvidenceBundle(BaseModel):
    """
    Collection of evidence items for a market resolution.
    
    This is the primary output of the Collector agent.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    schema_version: str = Field(default=SCHEMA_VERSION)
    bundle_id: str = Field(..., description="Unique identifier for this bundle", min_length=1)
    market_id: str = Field(..., description="ID of the market this evidence is for", min_length=1)
    plan_id: str = Field(..., description="ID of the ToolPlan that was executed", min_length=1)
    collector_name: str | None = Field(
        default=None,
        description="Name of the collector agent that produced this bundle",
    )
    weight: float = Field(
        default=1.0,
        description="Weight for evidence aggregation (1.0 = equal weight)",
        ge=0.0,
        le=10.0,
    )

    # Evidence items
    items: list[EvidenceItem] = Field(default_factory=list)
    
    # Execution metadata
    collected_at: datetime | None = Field(
        default=None,
        description="When collection completed (OPTIONAL for determinism)",
    )
    execution_time_ms: float | None = Field(default=None, description="Total execution time")
    
    # Summary
    total_sources_attempted: int = Field(default=0)
    total_sources_succeeded: int = Field(default=0)
    requirements_fulfilled: list[str] = Field(default_factory=list)
    requirements_unfulfilled: list[str] = Field(default_factory=list)
    
    @property
    def has_evidence(self) -> bool:
        """Check if any evidence was collected."""
        return len(self.items) > 0
    
    @property
    def all_requirements_met(self) -> bool:
        """Check if all requirements were fulfilled."""
        return len(self.requirements_unfulfilled) == 0
    
    @property
    def success_rate(self) -> float:
        """Get the success rate of source fetches."""
        if self.total_sources_attempted == 0:
            return 0.0
        return self.total_sources_succeeded / self.total_sources_attempted
    
    def get_evidence_for_requirement(self, requirement_id: str) -> list[EvidenceItem]:
        """Get all evidence items for a specific requirement."""
        return [item for item in self.items if item.requirement_id == requirement_id]
    
    def get_valid_evidence(self) -> list[EvidenceItem]:
        """Get all valid evidence items."""
        return [item for item in self.items if item.is_valid]
    
    def get_highest_tier_evidence(self) -> EvidenceItem | None:
        """Get the evidence with highest provenance tier."""
        valid = self.get_valid_evidence()
        if not valid:
            return None
        return max(valid, key=lambda e: e.provenance.tier)
    
    def add_item(self, item: EvidenceItem) -> None:
        """Add an evidence item to the bundle."""
        self.items.append(item)
        if item.success:
            self.total_sources_succeeded += 1
            if item.requirement_id not in self.requirements_fulfilled:
                self.requirements_fulfilled.append(item.requirement_id)
                # Remove from unfulfilled if present
                if item.requirement_id in self.requirements_unfulfilled:
                    self.requirements_unfulfilled.remove(item.requirement_id)
        self.total_sources_attempted += 1


class CollectionResult(BaseModel):
    """
    Result of a collection operation.
    
    Wraps EvidenceBundle with execution status.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    success: bool = Field(...)
    bundle: EvidenceBundle | None = Field(default=None)
    error: str | None = Field(default=None)
    warnings: list[str] = Field(default_factory=list)
    
    @classmethod
    def from_bundle(cls, bundle: EvidenceBundle) -> "CollectionResult":
        """Create a successful result from a bundle."""
        return cls(
            success=bundle.has_evidence,
            bundle=bundle,
            error=None if bundle.has_evidence else "No evidence collected",
        )
    
    @classmethod
    def failure(cls, error: str) -> "CollectionResult":
        """Create a failure result."""
        return cls(success=False, error=error)
