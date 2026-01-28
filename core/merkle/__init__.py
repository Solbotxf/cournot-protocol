"""
Module 02 - Merkle Tree and Commitments
Deterministic Merkle tree construction + proof generation/verification.

Owner: Protocol/Crypto Engineer
Module ID: M02

This module provides:
- MerkleProof: Dataclass representing a Merkle inclusion proof
- build_merkle_root: Compute root from leaf hashes
- build_merkle_proof: Generate proof for a specific leaf
- verify_merkle_proof: Verify a proof against its claimed root

Canonical Commitment Rules:
1. Leaf hashing: sha256(dumps_canonical(obj).encode("utf-8"))
2. Parent hashing: sha256(left + right)
3. Padding: Duplicate last node if odd number at any level
4. Empty tree: sha256(b"")
5. Single leaf: root = leaf

Usage:
    from core.merkle import build_merkle_root, build_merkle_proof, verify_merkle_proof
    from core.crypto import hash_canonical
    
    # Create leaves from objects
    leaves = [hash_canonical(obj) for obj in objects]
    
    # Compute root
    root = build_merkle_root(leaves)
    
    # Generate proof for leaf at index 2
    proof = build_merkle_proof(leaves, index=2)
    
    # Verify proof
    assert verify_merkle_proof(proof)
"""
from .merkle_tree import (
    EMPTY_TREE_ROOT,
    MerkleProof,
    merkle_parent,
    build_merkle_root,
    build_merkle_proof,
    verify_merkle_proof,
    compute_tree_depth,
)

from .merkle_proofs import (
    MerkleProver,
    MerkleVerifier,
)


__all__ = [
    # Core types
    "MerkleProof",
    "EMPTY_TREE_ROOT",
    # Core functions
    "merkle_parent",
    "build_merkle_root",
    "build_merkle_proof",
    "verify_merkle_proof",
    "compute_tree_depth",
    # Convenience classes
    "MerkleProver",
    "MerkleVerifier",
]