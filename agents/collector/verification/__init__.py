"""
Collector Verification Package

Provenance verification modules including tier policy enforcement
and proof verification (signatures, zkTLS).
"""

from agents.collector.verification.signature_verifier import (
    SignatureVerifier,
)
from agents.collector.verification.tier_policy import (
    DEFAULT_TIER_MAPPING,
    SelectionPolicyEnforcer,
    TierPolicy,
)
from agents.collector.verification.zktls_verifier import (
    ZkTLSProofMetadata,
    ZkTLSVerifier,
)

__all__ = [
    # Tier Policy
    "TierPolicy",
    "SelectionPolicyEnforcer",
    "DEFAULT_TIER_MAPPING",
    # Signature Verification
    "SignatureVerifier",
    # zkTLS Verification
    "ZkTLSVerifier",
    "ZkTLSProofMetadata",
]