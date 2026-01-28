"""
Module 08 - Validator/Sentinel: Pinpoint Challenge Creation

Helper functions for creating pinpoint challenges when mismatches are detected.
These helpers generate Merkle proofs for disputed leaves.

Owner: Protocol Verification Engineer
Module ID: M08
"""

from __future__ import annotations

from typing import Optional

from core.crypto.hashing import to_hex
from core.merkle import build_merkle_proof, MerkleProof
from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import (
    compute_evidence_leaf_hashes,
    compute_reasoning_leaf_hashes,
)
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle

from .challenges import (
    Challenge,
    make_root_mismatch_challenge,
)


def create_evidence_mismatch_challenge(
    bundle: PoRBundle,
    evidence: EvidenceBundle,
    computed_root: str,
    max_sample_proofs: int = 3,
) -> Challenge:
    """
    Create a challenge for evidence root mismatch.

    Generates sample leaf proofs for diagnostic purposes.

    Args:
        bundle: The PoR bundle with expected commitments.
        evidence: The evidence bundle to compute proofs from.
        computed_root: The computed evidence root that differs.
        max_sample_proofs: Maximum number of sample proofs to include.

    Returns:
        Challenge object with root mismatch details.
    """
    # Compute leaf hashes for sample proofs
    leaf_hashes = compute_evidence_leaf_hashes(evidence)

    if not leaf_hashes:
        # No evidence items - create root mismatch challenge without samples
        return make_root_mismatch_challenge(
            kind="evidence_leaf",
            market_id=bundle.market_id,
            bundle_root=bundle.por_root or bundle.verdict_hash,
            expected_root=bundle.evidence_root,
            computed_root=computed_root,
        )

    # Generate sample proofs (up to max_sample_proofs)
    sample_proofs: list[tuple[int, str, MerkleProof]] = []
    for i in range(min(len(leaf_hashes), max_sample_proofs)):
        proof = build_merkle_proof(leaf_hashes, i)
        sample_proofs.append((i, to_hex(leaf_hashes[i]), proof))

    return make_root_mismatch_challenge(
        kind="evidence_leaf",
        market_id=bundle.market_id,
        bundle_root=bundle.por_root or bundle.verdict_hash,
        expected_root=bundle.evidence_root,
        computed_root=computed_root,
        sample_leaf_proofs=sample_proofs,
    )


def create_reasoning_mismatch_challenge(
    bundle: PoRBundle,
    trace: ReasoningTrace,
    computed_root: str,
    max_sample_proofs: int = 3,
) -> Challenge:
    """
    Create a challenge for reasoning root mismatch.

    Generates sample leaf proofs for diagnostic purposes.

    Args:
        bundle: The PoR bundle with expected commitments.
        trace: The reasoning trace to compute proofs from.
        computed_root: The computed reasoning root that differs.
        max_sample_proofs: Maximum number of sample proofs to include.

    Returns:
        Challenge object with root mismatch details and step IDs.
    """
    # Compute leaf hashes for sample proofs
    leaf_hashes = compute_reasoning_leaf_hashes(trace)

    if not leaf_hashes:
        # No reasoning steps - create root mismatch challenge without samples
        return make_root_mismatch_challenge(
            kind="reasoning_leaf",
            market_id=bundle.market_id,
            bundle_root=bundle.por_root or bundle.verdict_hash,
            expected_root=bundle.reasoning_root,
            computed_root=computed_root,
        )

    # Generate sample proofs (up to max_sample_proofs)
    sample_proofs: list[tuple[int, str, MerkleProof]] = []
    for i in range(min(len(leaf_hashes), max_sample_proofs)):
        proof = build_merkle_proof(leaf_hashes, i)
        sample_proofs.append((i, to_hex(leaf_hashes[i]), proof))

    return make_root_mismatch_challenge(
        kind="reasoning_leaf",
        market_id=bundle.market_id,
        bundle_root=bundle.por_root or bundle.verdict_hash,
        expected_root=bundle.reasoning_root,
        computed_root=computed_root,
        sample_leaf_proofs=sample_proofs,
        details={
            "step_ids": [step.step_id for step in trace.steps[:max_sample_proofs]],
        },
    )


__all__ = [
    "create_evidence_mismatch_challenge",
    "create_reasoning_mismatch_challenge",
]