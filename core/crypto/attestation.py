from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class TEEAttestation(BaseModel):
    """Represents a fast-path TEE attestation statement."""
    schema_version: str = "v1"
    attester: str
    report: Dict[str, Any]
    signature: Optional[str] = None


class AttestationProvider:
    """Interface for producing/verifying TEE attestations."""

    def attest(self, payload: bytes) -> TEEAttestation:
        raise NotImplementedError

    def verify(self, attestation: TEEAttestation, payload: bytes) -> bool:
        raise NotImplementedError
