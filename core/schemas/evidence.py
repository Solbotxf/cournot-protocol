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


# Type aliases for evidence-related enums
RetrievalMethod = Literal["http_get", "api_call", "chain_rpc", "manual_import", "other"]
ProvenanceKind = Literal["zktls", "signature", "notary", "hashlog", "none"]
ContentType = Literal["text", "json", "html", "bytes"]


class SourceDescriptor(BaseModel):
    """
    Descriptor for a data source.

    Identifies where evidence came from.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(
        ...,
        description="Unique identifier for the source",
        min_length=1,
    )
    uri: str | None = Field(
        default=None,
        description="URI/URL of the source",
    )
    provider: str | None = Field(
        default=None,
        description="Name of the data provider",
    )
    notes: str | None = Field(
        default=None,
        description="Additional notes about the source",
    )


class RetrievalReceipt(BaseModel):
    """
    Receipt documenting how evidence was retrieved.

    Provides audit trail for data collection.
    """

    model_config = ConfigDict(extra="forbid")

    retrieved_at: datetime | None = Field(
        default=None,
        description="Timestamp when the data was retrieved",
    )
    method: RetrievalMethod = Field(
        ...,
        description="Method used to retrieve the data",
    )
    tool: str | None = Field(
        default=None,
        description="Tool/agent that performed the retrieval",
    )
    request_fingerprint: str | None = Field(
        default=None,
        description="Hash/fingerprint of the request made",
    )
    response_fingerprint: str | None = Field(
        default=None,
        description="Hash/fingerprint of the response received",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional retrieval metadata",
    )
    status_code: int | None = Field(
        default=None,
        description="Status code",
    )


class ProvenanceProof(BaseModel):
    """
    Cryptographic proof of data provenance.

    Provides verifiable evidence of data authenticity.

    Provenance Tiers:
        - Tier 0: zkTLS proof (highest assurance)
        - Tier 1: Server signature / API signed response
        - Tier 2: Notarized snapshot (hash in public log)
        - Tier 3: Plain HTTP fetch (lowest assurance)
    """

    model_config = ConfigDict(extra="forbid")

    tier: int = Field(
        ...,
        description="Provenance tier (0=highest, 3=lowest)",
        ge=0,
    )
    kind: ProvenanceKind = Field(
        ...,
        description="Type of provenance proof",
    )
    proof_blob: str | None = Field(
        default=None,
        description="The actual proof data (encoded)",
    )
    verifier: str | None = Field(
        default=None,
        description="Identifier of the verifier/attestation service",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional proof metadata",
    )

    @field_validator("tier")
    @classmethod
    def validate_tier_range(cls, v: int) -> int:
        """Validate tier is within expected range."""
        if v > 3:
            # Allow higher tiers but warn (future extensibility)
            pass
        return v

    @property
    def is_cryptographic(self) -> bool:
        """Check if this proof is cryptographically verifiable."""
        return self.kind in ("zktls", "signature", "notary")

    @property
    def is_high_assurance(self) -> bool:
        """Check if this is a high-assurance proof (tier 0-1)."""
        return self.tier <= 1


class EvidenceItem(BaseModel):
    """
    A single piece of evidence with provenance.

    This is the atomic unit of evidence in the system.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for this model",
    )
    evidence_id: str = Field(
        ...,
        description="Unique identifier for this evidence item",
        min_length=1,
    )
    requirement_id: str = Field(
        default=None,
        description="Requirement id",
        min_length=1,
    )
    source: SourceDescriptor = Field(
        ...,
        description="Source of the evidence",
    )
    retrieval: RetrievalReceipt = Field(
        ...,
        description="Receipt documenting retrieval",
    )
    provenance: ProvenanceProof = Field(
        ...,
        description="Proof of data provenance",
    )
    content_type: ContentType = Field(
        ...,
        description="Type of the content",
    )
    content: Any = Field(
        ...,
        description="The actual evidence content",
    )
    normalized: Any | None = Field(
        default=None,
        description="Normalized/processed version of the content",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for this evidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    @property
    def provenance_tier(self) -> int:
        """Convenience accessor for provenance tier."""
        return self.provenance.tier

    @property
    def is_high_provenance(self) -> bool:
        """Check if this evidence has high provenance assurance."""
        return self.provenance.is_high_assurance

    def meets_tier_requirement(self, min_tier: int) -> bool:
        """Check if this evidence meets a minimum tier requirement."""
        return self.provenance.tier <= min_tier  # Lower tier = higher assurance


class EvidenceBundle(BaseModel):
    """
    Collection of evidence items gathered for a resolution.

    This is the output of the Collector agent.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for this model",
    )
    bundle_id: str = Field(
        ...,
        description="Unique identifier for this bundle",
        min_length=1,
    )
    collector_id: str = Field(
        ...,
        description="Identifier of the collector that created this bundle",
        min_length=1,
    )
    collection_time: datetime | None = Field(
        default=None,
        description="Timestamp when collection completed (optional for determinism)",
    )
    items: list[EvidenceItem] = Field(
        default_factory=list,
        description="List of evidence items in the bundle",
    )
    provenance_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of provenance across all items",
    )

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "EvidenceBundle":
        """Ensure all evidence IDs are unique within the bundle."""
        ids = [item.evidence_id for item in self.items]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate evidence IDs in bundle: {set(duplicates)}")
        return self

    @property
    def item_count(self) -> int:
        """Number of evidence items in the bundle."""
        return len(self.items)

    @property
    def evidence_ids(self) -> list[str]:
        """List of all evidence IDs in the bundle."""
        return [item.evidence_id for item in self.items]

    def get_item_by_id(self, evidence_id: str) -> EvidenceItem | None:
        """Find an evidence item by its ID."""
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def get_items_by_source(self, source_id: str) -> list[EvidenceItem]:
        """Get all items from a specific source."""
        return [item for item in self.items if item.source.source_id == source_id]

    def get_items_by_tier(self, max_tier: int) -> list[EvidenceItem]:
        """Get all items meeting a maximum tier requirement."""
        return [item for item in self.items if item.provenance.tier <= max_tier]

    def compute_provenance_summary(self) -> dict[str, Any]:
        """Compute a summary of provenance across all items."""
        if not self.items:
            return {"total_items": 0, "by_tier": {}, "by_kind": {}}

        by_tier: dict[int, int] = {}
        by_kind: dict[str, int] = {}

        for item in self.items:
            tier = item.provenance.tier
            kind = item.provenance.kind

            by_tier[tier] = by_tier.get(tier, 0) + 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

        return {
            "total_items": len(self.items),
            "by_tier": by_tier,
            "by_kind": by_kind,
            "min_tier": min(by_tier.keys()),
            "max_tier": max(by_tier.keys()),
        }