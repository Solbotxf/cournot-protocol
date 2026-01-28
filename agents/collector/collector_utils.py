"""
Module 05 - Collector Utilities

Helper functions for the collector agent, including fingerprint
computation, evidence ID generation, and content normalization.
"""

from typing import Any, Optional

from core.crypto.hashing import hash_canonical, sha256, to_hex
from core.schemas.evidence import EvidenceItem
from core.schemas.prompts import DataRequirement, SourceTarget

from agents.collector.data_sources.base_source import FetchedArtifact


def compute_request_fingerprint(target: SourceTarget) -> str:
    """
    Compute deterministic fingerprint of request.
    
    Args:
        target: Source target
        
    Returns:
        Hex fingerprint string with 0x prefix
    """
    request_data = {
        "uri": target.uri,
        "method": target.method,
        "params": target.params or {},
        "body": target.body,
    }
    return to_hex(hash_canonical(request_data))


def compute_response_fingerprint(artifact: FetchedArtifact) -> str:
    """
    Compute deterministic fingerprint of response.
    
    Args:
        artifact: Fetched artifact
        
    Returns:
        Hex fingerprint string with 0x prefix
    """
    return to_hex(sha256(artifact.raw_bytes))


def compute_evidence_id(
    requirement_id: str,
    request_fingerprint: str,
    response_fingerprint: str
) -> str:
    """
    Compute deterministic evidence ID.
    
    Args:
        requirement_id: Requirement identifier
        request_fingerprint: Request fingerprint
        response_fingerprint: Response fingerprint
        
    Returns:
        Evidence ID string prefixed with "ev_"
    """
    id_data = {
        "requirement_id": requirement_id,
        "request_fingerprint": request_fingerprint,
        "response_fingerprint": response_fingerprint,
    }
    hash_bytes = hash_canonical(id_data)
    return "ev_" + to_hex(hash_bytes)[2:18]  # Take first 16 hex chars


def compute_bundle_id(market_id: str, plan_id: str) -> str:
    """
    Compute deterministic bundle ID.
    
    Args:
        market_id: Market identifier
        plan_id: Plan identifier
        
    Returns:
        Bundle ID string prefixed with "bundle_"
    """
    bundle_id_data = {
        "market_id": market_id,
        "plan_id": plan_id,
    }
    return "bundle_" + to_hex(hash_canonical(bundle_id_data))[2:18]


def map_method(method: str) -> str:
    """
    Map SourceTarget method to RetrievalReceipt method.
    
    Args:
        method: Source target method (GET, POST, etc.)
        
    Returns:
        Retrieval receipt method string
    """
    method_map = {
        "GET": "http_get",
        "POST": "http_post",
        "RPC": "chain_rpc",
        "WS": "other",
        "OTHER": "other",
    }
    return method_map.get(method, "other")


def normalize_content(
    artifact: FetchedArtifact,
    requirement: DataRequirement
) -> Optional[dict[str, Any]]:
    """
    Extract normalized fields from artifact.
    
    Args:
        artifact: Fetched artifact
        requirement: Data requirement with expected fields
        
    Returns:
        Normalized content dict or None
    """
    if not artifact.is_json or not requirement.expected_fields:
        return None
    
    data = artifact.parsed
    if not isinstance(data, dict):
        return None
    
    normalized = {}
    for field in requirement.expected_fields:
        if field in data:
            normalized[field] = data[field]
    
    return normalized if normalized else None


def compute_tier_distribution(items: list[EvidenceItem]) -> dict[int, int]:
    """
    Compute distribution of provenance tiers.
    
    Args:
        items: List of evidence items
        
    Returns:
        Dict mapping tier level to count
    """
    distribution: dict[int, int] = {}
    for item in items:
        tier = item.provenance.tier
        distribution[tier] = distribution.get(tier, 0) + 1
    return distribution