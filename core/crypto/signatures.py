from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Signature:
    signer_id: str
    signature_hex: str
    scheme: str = "ed25519"


def sign(payload: bytes, private_key: bytes, signer_id: str) -> Signature:
    """Stub: Sign payload. Replace with real crypto."""
    raise NotImplementedError


def verify(payload: bytes, signature: Signature, public_key: bytes) -> bool:
    """Stub: Verify signature. Replace with real crypto."""
    raise NotImplementedError
