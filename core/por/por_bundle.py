"""
Module 03 - PoR Bundle Model

Defines the verifiable Proof of Reasoning package that can be anchored and disputed.
Contains commitments (hashes/roots) to all artifacts and the embedded verdict.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.schemas.verdict import DeterministicVerdict
from core.schemas.versioning import PROTOCOL_VERSION, SCHEMA_VERSION


# Regex pattern for validating hex strings (0x followed by 64 hex chars = 32 bytes)
HEX_HASH_PATTERN = re.compile(r"^0x[0-9a-fA-F]{64}$")


def validate_hex_hash(value: str, field_name: str) -> str:
    """Validate that a value is a valid 32-byte hex hash with 0x prefix."""
    if not HEX_HASH_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a valid 32-byte hex string with 0x prefix "
            f"(64 hex chars), got: {value[:20]}..." if len(value) > 20 else value
        )
    return value.lower()  # Normalize to lowercase


class TEEAttestation(BaseModel):
    """
    TEE attestation for fast-path anchoring.
    Minimal structure - can be extended based on specific TEE requirements.
    """

    model_config = ConfigDict(extra="forbid")

    attestation_type: str = Field(
        ...,
        description="Type of TEE attestation (e.g., 'sgx', 'nitro', 'tdx')",
    )
    quote: str = Field(
        ...,
        description="The attestation quote/report (base64 or hex encoded)",
    )
    enclave_measurement: Optional[str] = Field(
        default=None,
        description="Enclave measurement hash (MRENCLAVE for SGX)",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Attestation timestamp",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class PoRBundle(BaseModel):
    """
    Proof of Reasoning Bundle.

    This is the verifiable package that contains commitments to:
    - The prompt specification (what was asked)
    - The evidence bundle (what data was collected)
    - The reasoning trace (how conclusions were reached)
    - The verdict (the final decision)

    Plus the embedded verdict for convenience and optional anchoring artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    # Versioning
    schema_version: str = Field(default=SCHEMA_VERSION)
    protocol_version: str = Field(default=PROTOCOL_VERSION)

    # Market identification
    market_id: str = Field(
        ...,
        description="Market identifier - must match verdict.market_id",
        min_length=1,
    )

    # Commitments (all are 0x-prefixed hex hashes)
    prompt_spec_hash: str = Field(
        ...,
        description="Hash of the canonical PromptSpec (0x-prefixed, 32 bytes)",
    )
    evidence_root: str = Field(
        ...,
        description="Merkle root of evidence items (0x-prefixed, 32 bytes)",
    )
    reasoning_root: str = Field(
        ...,
        description="Merkle root of reasoning steps (0x-prefixed, 32 bytes)",
    )
    verdict_hash: str = Field(
        ...,
        description="Hash of the canonical DeterministicVerdict (0x-prefixed, 32 bytes)",
    )
    por_root: Optional[str] = Field(
        default=None,
        description="Combined PoR root hash (0x-prefixed, 32 bytes)",
    )

    # Embedded payload
    verdict: DeterministicVerdict = Field(
        ...,
        description="The full verdict object (for convenience, can be verified against verdict_hash)",
    )

    # Anchoring / signatures
    tee_attestation: Optional[TEEAttestation] = Field(
        default=None,
        description="Optional TEE attestation for fast-path anchoring",
    )
    signatures: dict[str, Any] = Field(
        default_factory=dict,
        description="Signatures from various parties (collector, auditor, judge, etc.)",
    )

    # Metadata
    created_at: datetime = Field(
        ...,
        description="When this bundle was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("prompt_spec_hash")
    @classmethod
    def validate_prompt_spec_hash(cls, v: str) -> str:
        return validate_hex_hash(v, "prompt_spec_hash")

    @field_validator("evidence_root")
    @classmethod
    def validate_evidence_root(cls, v: str) -> str:
        return validate_hex_hash(v, "evidence_root")

    @field_validator("reasoning_root")
    @classmethod
    def validate_reasoning_root(cls, v: str) -> str:
        return validate_hex_hash(v, "reasoning_root")

    @field_validator("verdict_hash")
    @classmethod
    def validate_verdict_hash(cls, v: str) -> str:
        return validate_hex_hash(v, "verdict_hash")

    @field_validator("por_root")
    @classmethod
    def validate_por_root(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_hex_hash(v, "por_root")
        return v

    @model_validator(mode="after")
    def validate_market_id_match(self) -> "PoRBundle":
        """Ensure market_id matches the embedded verdict's market_id."""
        if self.market_id != self.verdict.market_id:
            raise ValueError(
                f"Bundle market_id '{self.market_id}' does not match "
                f"verdict.market_id '{self.verdict.market_id}'"
            )
        return self