"""
Module 02 - Merkle Proofs Convenience Wrappers
Thin wrappers around core Merkle tree functions for cleaner API.

Owner: Protocol/Crypto Engineer
Module ID: M02

This module provides class-based interfaces:
- MerkleProver: Generate proofs for leaves
- MerkleVerifier: Verify proofs

These are convenience wrappers around the functions in merkle_tree.py.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.crypto.hashing import hash_canonical
from core.merkle.merkle_tree import (
    MerkleProof,
    build_merkle_proof,
    build_merkle_root,
    verify_merkle_proof,
)


class MerkleProver:
    """
    Convenience class for generating Merkle proofs.
    
    Provides static methods for proof generation from:
    - Pre-hashed leaves (bytes)
    - Raw objects (will be canonically hashed)
    
    Example:
        >>> leaves = [sha256(b"a"), sha256(b"b"), sha256(b"c")]
        >>> proof = MerkleProver.prove(leaves, index=1)
        >>> proof.leaf == leaves[1]
        True
    """
    
    @staticmethod
    def prove(leaves: Sequence[bytes], index: int) -> MerkleProof:
        """
        Generate a Merkle proof for the leaf at the given index.
        
        Args:
            leaves: Sequence of pre-hashed leaf values
            index: 0-based index of the leaf to prove
            
        Returns:
            MerkleProof for the specified leaf
            
        Raises:
            IndexError: If index is out of range
            ValueError: If leaves is empty
        """
        return build_merkle_proof(leaves, index)
    
    @staticmethod
    def prove_object(objects: Sequence[Any], index: int) -> MerkleProof:
        """
        Generate a Merkle proof for an object at the given index.
        
        Objects are first converted to leaf hashes via canonical hashing.
        
        Args:
            objects: Sequence of objects to hash canonically
            index: 0-based index of the object to prove
            
        Returns:
            MerkleProof for the specified object
            
        Raises:
            IndexError: If index is out of range
            ValueError: If objects is empty
        """
        leaves = [hash_canonical(obj) for obj in objects]
        return build_merkle_proof(leaves, index)
    
    @staticmethod
    def compute_root(leaves: Sequence[bytes]) -> bytes:
        """
        Compute the Merkle root for a sequence of leaves.
        
        Args:
            leaves: Sequence of pre-hashed leaf values
            
        Returns:
            32-byte Merkle root
        """
        return build_merkle_root(leaves)
    
    @staticmethod
    def compute_root_from_objects(objects: Sequence[Any]) -> bytes:
        """
        Compute the Merkle root for a sequence of objects.
        
        Objects are first converted to leaf hashes via canonical hashing.
        
        Args:
            objects: Sequence of objects to hash canonically
            
        Returns:
            32-byte Merkle root
        """
        leaves = [hash_canonical(obj) for obj in objects]
        return build_merkle_root(leaves)


class MerkleVerifier:
    """
    Convenience class for verifying Merkle proofs.
    
    Provides static methods for proof verification.
    
    Example:
        >>> proof = MerkleProver.prove(leaves, index=1)
        >>> MerkleVerifier.verify(proof)
        True
    """
    
    @staticmethod
    def verify(proof: MerkleProof) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            proof: MerkleProof to verify
            
        Returns:
            True if the proof is valid, False otherwise
        """
        return verify_merkle_proof(proof)
    
    @staticmethod
    def verify_leaf_in_root(
        leaf: bytes,
        index: int,
        siblings: list[bytes],
        root: bytes,
    ) -> bool:
        """
        Verify a leaf is included in a Merkle root using raw components.
        
        This is a convenience method that constructs a MerkleProof
        and verifies it.
        
        Args:
            leaf: The leaf hash to verify
            index: The claimed index of the leaf
            siblings: List of sibling hashes (bottom-up)
            root: The claimed Merkle root
            
        Returns:
            True if the proof is valid, False otherwise
        """
        proof = MerkleProof(
            leaf=leaf,
            index=index,
            siblings=siblings,
            root=root,
        )
        return verify_merkle_proof(proof)
    
    @staticmethod
    def verify_object_in_root(
        obj: Any,
        index: int,
        siblings: list[bytes],
        root: bytes,
    ) -> bool:
        """
        Verify an object is included in a Merkle root.
        
        The object is canonically hashed to produce the leaf hash.
        
        Args:
            obj: The object to verify (will be canonically hashed)
            index: The claimed index of the object
            siblings: List of sibling hashes (bottom-up)
            root: The claimed Merkle root
            
        Returns:
            True if the proof is valid, False otherwise
        """
        leaf = hash_canonical(obj)
        return MerkleVerifier.verify_leaf_in_root(leaf, index, siblings, root)


__all__ = [
    "MerkleProver",
    "MerkleVerifier",
]