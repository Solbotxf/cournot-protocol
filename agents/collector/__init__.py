"""
Collector Agent Package

Data acquisition and provenance verification for the Cournot protocol.

The Collector agent:
1. Executes explicit SourceTarget requests (URLs/endpoints)
2. Captures retrieval receipts (timestamps, fingerprints)
3. Attaches provenance proofs (tiered)
4. Enforces minimum provenance tier + selection policy
5. Outputs EvidenceBundle with stable evidence_id and ordering
"""

from agents.collector.collector_agent import (
    CollectorAgent,
    CollectorConfig,
)
from agents.collector.data_sources import (
    APISource,
    BaseSource,
    ExchangeSource,
    FetchedArtifact,
    HTTPSource,
    PolymarketSource,
    SourceProtocol,
    WebSource,
    get_source_for_target,
)
from agents.collector.verification import (
    DEFAULT_TIER_MAPPING,
    SelectionPolicyEnforcer,
    SignatureVerifier,
    TierPolicy,
    ZkTLSProofMetadata,
    ZkTLSVerifier,
)
from agents.collector.collector_utils import (
    compute_bundle_id,
    compute_evidence_id,
    compute_request_fingerprint,
    compute_response_fingerprint,
    compute_tier_distribution,
    map_method,
    normalize_content,
)

__all__ = [
    # Main Agent
    "CollectorAgent",
    "CollectorConfig",
    # Data Sources
    "BaseSource",
    "FetchedArtifact",
    "SourceProtocol",
    "HTTPSource",
    "APISource",
    "WebSource",
    "PolymarketSource",
    "ExchangeSource",
    "get_source_for_target",
    # Verification
    "TierPolicy",
    "SelectionPolicyEnforcer",
    "DEFAULT_TIER_MAPPING",
    "SignatureVerifier",
    "ZkTLSVerifier",
    "ZkTLSProofMetadata",
    # Utilities
    "compute_bundle_id",
    "compute_evidence_id",
    "compute_request_fingerprint",
    "compute_response_fingerprint",
    "compute_tier_distribution",
    "map_method",
    "normalize_content",
]