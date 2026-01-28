"""
Module 08 - Validator/Sentinel: Challenge Model

Defines serializable "challenge packets" that can be used for dispute resolution.
Challenges identify specific mismatches in PoR commitments with Merkle proofs
where applicable.

Owner: Protocol Verification Engineer
Module ID: M08
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.crypto.hashing import hash_canonical, to_hex
from core.merkle import build_merkle_proof, MerkleProof


# Challenge kinds that can be disputed
ChallengeKindType = Literal[
    "prompt_hash",
    "evidence_leaf",
    "reasoning_leaf",
    "verdict_hash",
    "por_root",
    "replay_divergence",
]


@dataclass(frozen=True)
class MerkleProofData:
    """
    Serializable Merkle proof data for inclusion in challenges.
    """
    siblings: list[str]  # hex-encoded sibling hashes
    index: int
    root: str  # hex-encoded root

    @classmethod
    def from_merkle_proof(cls, proof: MerkleProof) -> "MerkleProofData":
        """Create from a MerkleProof object."""
        return cls(
            siblings=[to_hex(s) for s in proof.siblings],
            index=proof.index,
            root=to_hex(proof.root),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "siblings": self.siblings,
            "index": self.index,
            "root": self.root,
        }


@dataclass
class Challenge:
    """
    A challenge packet identifying a disputed component in a PoR bundle.

    This can be used for on-chain dispute submission or peer-to-peer
    verification challenges.
    """
    challenge_id: str
    kind: ChallengeKindType
    market_id: str
    bundle_root: str  # por_root from the bundle
    component_root: Optional[str] = None  # evidence_root/reasoning_root if relevant
    leaf_index: Optional[int] = None
    leaf_hash: Optional[str] = None
    merkle_proof: Optional[MerkleProofData] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert challenge to dictionary for serialization."""
        result: dict[str, Any] = {
            "challenge_id": self.challenge_id,
            "kind": self.kind,
            "market_id": self.market_id,
            "bundle_root": self.bundle_root,
        }

        if self.component_root is not None:
            result["component_root"] = self.component_root
        if self.leaf_index is not None:
            result["leaf_index"] = self.leaf_index
        if self.leaf_hash is not None:
            result["leaf_hash"] = self.leaf_hash
        if self.merkle_proof is not None:
            result["merkle_proof"] = self.merkle_proof.to_dict()
        if self.details:
            result["details"] = self.details

        return result


def _compute_challenge_id(
    kind: ChallengeKindType,
    market_id: str,
    bundle_root: str,
    leaf_index: Optional[int],
    leaf_hash: Optional[str],
) -> str:
    """
    Compute deterministic challenge ID.

    challenge_id = "ch_" + sha256(canonical(kind|market_id|bundle_root|leaf_index|leaf_hash))[:12]
    """
    # Build canonical input string
    parts = [kind, market_id, bundle_root]
    if leaf_index is not None:
        parts.append(str(leaf_index))
    if leaf_hash is not None:
        parts.append(leaf_hash)

    canonical_input = "|".join(parts)
    hash_bytes = hash_canonical(canonical_input)
    return "ch_" + to_hex(hash_bytes)[2:14]  # Skip 0x, take first 12 chars


def make_prompt_hash_challenge(
    market_id: str,
    bundle_root: str,
    expected_hash: str,
    computed_hash: str,
    details: Optional[dict[str, Any]] = None,
) -> Challenge:
    """
    Create a challenge for prompt_spec_hash mismatch.

    Args:
        market_id: Market identifier.
        bundle_root: The por_root from the disputed bundle.
        expected_hash: Hash from the bundle (claimed).
        computed_hash: Hash computed by verifier.
        details: Additional diagnostic information.

    Returns:
        Challenge object for prompt hash dispute.
    """
    challenge_id = _compute_challenge_id(
        "prompt_hash", market_id, bundle_root, None, expected_hash
    )

    return Challenge(
        challenge_id=challenge_id,
        kind="prompt_hash",
        market_id=market_id,
        bundle_root=bundle_root,
        leaf_hash=expected_hash,
        details={
            "expected": expected_hash,
            "computed": computed_hash,
            **(details or {}),
        },
    )


def make_verdict_hash_challenge(
    market_id: str,
    bundle_root: str,
    expected_hash: str,
    computed_hash: str,
    details: Optional[dict[str, Any]] = None,
) -> Challenge:
    """
    Create a challenge for verdict_hash mismatch.

    Args:
        market_id: Market identifier.
        bundle_root: The por_root from the disputed bundle.
        expected_hash: Hash from the bundle (claimed).
        computed_hash: Hash computed by verifier.
        details: Additional diagnostic information.

    Returns:
        Challenge object for verdict hash dispute.
    """
    challenge_id = _compute_challenge_id(
        "verdict_hash", market_id, bundle_root, None, expected_hash
    )

    return Challenge(
        challenge_id=challenge_id,
        kind="verdict_hash",
        market_id=market_id,
        bundle_root=bundle_root,
        leaf_hash=expected_hash,
        details={
            "expected": expected_hash,
            "computed": computed_hash,
            **(details or {}),
        },
    )


def make_merkle_leaf_challenge(
    kind: Literal["evidence_leaf", "reasoning_leaf"],
    market_id: str,
    bundle_root: str,
    component_root: str,
    leaf_hash: str,
    leaf_index: int,
    proof: MerkleProof,
    details: Optional[dict[str, Any]] = None,
) -> Challenge:
    """
    Create a challenge for a Merkle leaf mismatch with proof.

    Args:
        kind: Either "evidence_leaf" or "reasoning_leaf".
        market_id: Market identifier.
        bundle_root: The por_root from the disputed bundle.
        component_root: The evidence_root or reasoning_root.
        leaf_hash: Hash of the disputed leaf.
        leaf_index: Index of the leaf in the tree.
        proof: MerkleProof for the leaf.
        details: Additional diagnostic information.

    Returns:
        Challenge object with Merkle proof for pinpoint dispute.
    """
    challenge_id = _compute_challenge_id(
        kind, market_id, bundle_root, leaf_index, leaf_hash
    )

    return Challenge(
        challenge_id=challenge_id,
        kind=kind,
        market_id=market_id,
        bundle_root=bundle_root,
        component_root=component_root,
        leaf_index=leaf_index,
        leaf_hash=leaf_hash,
        merkle_proof=MerkleProofData.from_merkle_proof(proof),
        details=details or {},
    )


def make_root_mismatch_challenge(
    kind: Literal["evidence_leaf", "reasoning_leaf"],
    market_id: str,
    bundle_root: str,
    expected_root: str,
    computed_root: str,
    sample_leaf_proofs: Optional[list[tuple[int, str, MerkleProof]]] = None,
    details: Optional[dict[str, Any]] = None,
) -> Challenge:
    """
    Create a challenge for root mismatch (when exact leaf cannot be pinpointed).

    This is used when the entire evidence_root or reasoning_root differs but
    we cannot identify which specific leaf is wrong.

    Args:
        kind: Either "evidence_leaf" or "reasoning_leaf".
        market_id: Market identifier.
        bundle_root: The por_root from the disputed bundle.
        expected_root: Root from the bundle (claimed).
        computed_root: Root computed by verifier.
        sample_leaf_proofs: Optional list of (index, leaf_hash, proof) tuples
                          for sample leaves from the computed tree.
        details: Additional diagnostic information.

    Returns:
        Challenge object for root mismatch dispute.
    """
    challenge_id = _compute_challenge_id(
        kind, market_id, bundle_root, None, expected_root
    )

    challenge_details: dict[str, Any] = {
        "expected_root": expected_root,
        "computed_root": computed_root,
        **(details or {}),
    }

    # Include sample proofs if provided
    if sample_leaf_proofs:
        challenge_details["sample_proofs"] = [
            {
                "leaf_index": idx,
                "leaf_hash": lh,
                "proof": MerkleProofData.from_merkle_proof(p).to_dict(),
            }
            for idx, lh, p in sample_leaf_proofs
        ]

    return Challenge(
        challenge_id=challenge_id,
        kind=kind,
        market_id=market_id,
        bundle_root=bundle_root,
        component_root=expected_root,
        details=challenge_details,
    )


def make_por_root_challenge(
    market_id: str,
    expected_root: str,
    computed_root: str,
    component_hashes: dict[str, str],
    details: Optional[dict[str, Any]] = None,
) -> Challenge:
    """
    Create a challenge for por_root mismatch.

    Args:
        market_id: Market identifier.
        expected_root: por_root from the bundle (claimed).
        computed_root: por_root computed by verifier.
        component_hashes: Dict of component hashes used in computation.
        details: Additional diagnostic information.

    Returns:
        Challenge object for por_root dispute.
    """
    challenge_id = _compute_challenge_id(
        "por_root", market_id, expected_root, None, None
    )

    return Challenge(
        challenge_id=challenge_id,
        kind="por_root",
        market_id=market_id,
        bundle_root=expected_root,
        details={
            "expected": expected_root,
            "computed": computed_root,
            "component_hashes": component_hashes,
            **(details or {}),
        },
    )


def make_replay_divergence_challenge(
    market_id: str,
    bundle_root: str,
    original_evidence_root: str,
    replayed_evidence_root: str,
    divergent_items: Optional[list[dict[str, Any]]] = None,
    details: Optional[dict[str, Any]] = None,
) -> Challenge:
    """
    Create a challenge for replay divergence.

    This is emitted when replaying evidence collection produces different
    results than the original bundle.

    Args:
        market_id: Market identifier.
        bundle_root: The por_root from the bundle.
        original_evidence_root: evidence_root from the original bundle.
        replayed_evidence_root: evidence_root from replayed collection.
        divergent_items: List of items that differ between original and replay.
        details: Additional diagnostic information.

    Returns:
        Challenge object for replay divergence dispute.
    """
    challenge_id = _compute_challenge_id(
        "replay_divergence", market_id, bundle_root, None, original_evidence_root
    )

    challenge_details: dict[str, Any] = {
        "original_evidence_root": original_evidence_root,
        "replayed_evidence_root": replayed_evidence_root,
        **(details or {}),
    }

    if divergent_items:
        challenge_details["divergent_items"] = divergent_items

    return Challenge(
        challenge_id=challenge_id,
        kind="replay_divergence",
        market_id=market_id,
        bundle_root=bundle_root,
        component_root=original_evidence_root,
        details=challenge_details,
    )


__all__ = [
    "ChallengeKindType",
    "MerkleProofData",
    "Challenge",
    "make_prompt_hash_challenge",
    "make_verdict_hash_challenge",
    "make_merkle_leaf_challenge",
    "make_root_mismatch_challenge",
    "make_por_root_challenge",
    "make_replay_divergence_challenge",
]