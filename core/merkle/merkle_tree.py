"""
Module 02 - Merkle Tree Implementation
Deterministic Merkle tree construction, proof generation, and verification.

Owner: Protocol/Crypto Engineer
Module ID: M02

This module provides:
- Deterministic Merkle root computation
- Merkle proof generation for any leaf index
- Merkle proof verification
- Standard padding rule for odd number of leaves

Canonical Commitment Rules (Hard Contracts):
1. Leaf hashing: leaf = sha256(dumps_canonical(obj).encode("utf-8"))
   - Implemented via core.crypto.hashing.hash_canonical()
2. Parent hashing: parent = sha256(left + right)
3. Padding rule: Duplicate last node if odd number at any level
4. Empty leaves: build_merkle_root([]) returns sha256(b"")
5. Single leaf: root = leaf (the leaf hash itself)

Determinism Notes:
- No randomness or non-deterministic ordering
- Leaf ordering is defined by upstream (evidence items order, reasoning steps order)
- This module never sorts leaves - it trusts input order
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.crypto.hashing import sha256


# Empty tree sentinel: sha256 of empty bytes
EMPTY_TREE_ROOT: bytes = sha256(b"")


@dataclass(frozen=True)
class MerkleProof:
    """
    A Merkle proof for a single leaf in a Merkle tree.
    
    The proof allows verification that a leaf is included in a tree
    with a known root, without revealing the entire tree.
    
    Attributes:
        leaf: The leaf hash being proven (32 bytes typically)
        index: The 0-based index of the leaf in the original leaf list
        siblings: List of sibling hashes from bottom to top of tree
        root: The Merkle root this proof is against
    """
    leaf: bytes
    index: int
    siblings: list[bytes]
    root: bytes
    
    def __post_init__(self) -> None:
        """Validate proof structure."""
        if self.index < 0:
            raise ValueError(f"Leaf index must be non-negative, got {self.index}")


def merkle_parent(left: bytes, right: bytes) -> bytes:
    """
    Compute the parent hash of two child nodes.
    
    Parent hash is deterministic: sha256(left + right)
    
    Args:
        left: Left child hash
        right: Right child hash
        
    Returns:
        Parent hash (32 bytes)
    """
    return sha256(left + right)


def build_merkle_root(leaves: Sequence[bytes]) -> bytes:
    """
    Build a Merkle root from a sequence of leaf hashes.
    
    Algorithm:
    1. If empty: return sha256(b"")
    2. If single leaf: return the leaf itself
    3. Otherwise, iteratively build levels:
       - If odd number of nodes, duplicate the last node
       - Pair adjacent nodes and compute parent hashes
       - Repeat until single root remains
    
    Padding Rule: Duplicate last node at each level if odd.
    Example: [a, b, c] -> [a, b, c, c] -> [parent(a,b), parent(c,c)]
    
    Args:
        leaves: Sequence of leaf hashes (typically 32 bytes each)
                Order matters and is preserved.
        
    Returns:
        32-byte Merkle root
        
    Example:
        >>> leaves = [sha256(b"a"), sha256(b"b"), sha256(b"c")]
        >>> root = build_merkle_root(leaves)
        >>> len(root)
        32
    """
    # Handle empty case
    if len(leaves) == 0:
        return EMPTY_TREE_ROOT
    
    # Handle single leaf case - root is the leaf itself
    if len(leaves) == 1:
        return leaves[0]
    
    # Start with leaf level
    current_level: list[bytes] = list(leaves)
    
    # Build tree level by level
    while len(current_level) > 1:
        # Pad with duplicate of last node if odd
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])
        
        # Build next level
        next_level: list[bytes] = []
        for i in range(0, len(current_level), 2):
            parent = merkle_parent(current_level[i], current_level[i + 1])
            next_level.append(parent)
        
        current_level = next_level
    
    return current_level[0]


def build_merkle_proof(leaves: Sequence[bytes], index: int) -> MerkleProof:
    """
    Generate a Merkle proof for the leaf at the given index.
    
    The proof consists of the sibling hashes needed to recompute
    the path from the leaf to the root.
    
    Algorithm:
    1. Start at the target leaf index
    2. At each level:
       - If odd number of nodes, pad with duplicate of last
       - Record the sibling hash (index XOR 1)
       - Move up: index = index // 2
    3. Continue until root level
    
    Args:
        leaves: Sequence of leaf hashes
        index: 0-based index of the leaf to prove
        
    Returns:
        MerkleProof with leaf, index, siblings (bottom-up), and root
        
    Raises:
        IndexError: If index is out of range
        ValueError: If leaves is empty
    """
    if len(leaves) == 0:
        raise ValueError("Cannot generate proof for empty leaf list")
    
    if index < 0 or index >= len(leaves):
        raise IndexError(
            f"Leaf index {index} out of range for {len(leaves)} leaves"
        )
    
    # Special case: single leaf (no siblings needed)
    if len(leaves) == 1:
        return MerkleProof(
            leaf=leaves[0],
            index=0,
            siblings=[],
            root=leaves[0],
        )
    
    # Collect siblings as we traverse up the tree
    siblings: list[bytes] = []
    current_level: list[bytes] = list(leaves)
    current_index = index
    
    while len(current_level) > 1:
        # Pad with duplicate of last node if odd
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])
        
        # Get sibling index (XOR with 1 flips last bit)
        sibling_index = current_index ^ 1
        siblings.append(current_level[sibling_index])
        
        # Build next level
        next_level: list[bytes] = []
        for i in range(0, len(current_level), 2):
            parent = merkle_parent(current_level[i], current_level[i + 1])
            next_level.append(parent)
        
        # Move index up
        current_index = current_index // 2
        current_level = next_level
    
    return MerkleProof(
        leaf=leaves[index],
        index=index,
        siblings=siblings,
        root=current_level[0],
    )


def verify_merkle_proof(proof: MerkleProof) -> bool:
    """
    Verify a Merkle proof.
    
    Recomputes the root from the leaf and siblings, checking
    against the claimed root in the proof.
    
    Algorithm:
    1. Start with the leaf hash
    2. For each sibling (bottom-up):
       - If current index is even: hash = parent(hash, sibling)
       - If current index is odd: hash = parent(sibling, hash)
       - Move up: index = index // 2
    3. Check computed root equals claimed root
    
    Args:
        proof: MerkleProof to verify
        
    Returns:
        True if proof is valid, False otherwise
    """
    # Start with the leaf
    current_hash = proof.leaf
    current_index = proof.index
    
    # Traverse up using siblings
    for sibling in proof.siblings:
        if current_index % 2 == 0:
            # Current node is left child
            current_hash = merkle_parent(current_hash, sibling)
        else:
            # Current node is right child
            current_hash = merkle_parent(sibling, current_hash)
        
        # Move up to parent level
        current_index = current_index // 2
    
    # Verify against claimed root
    return current_hash == proof.root


def compute_tree_depth(num_leaves: int) -> int:
    """
    Compute the depth of a Merkle tree with given number of leaves.
    
    Depth is the number of levels from leaves to root (inclusive).
    A single leaf has depth 1, two leaves have depth 2, etc.
    
    Args:
        num_leaves: Number of leaves in the tree
        
    Returns:
        Tree depth (0 for empty tree)
    """
    if num_leaves == 0:
        return 0
    if num_leaves == 1:
        return 1
    
    depth = 1
    n = num_leaves
    while n > 1:
        # Account for padding
        if n % 2 == 1:
            n += 1
        n = n // 2
        depth += 1
    
    return depth


__all__ = [
    "EMPTY_TREE_ROOT",
    "MerkleProof",
    "merkle_parent",
    "build_merkle_root",
    "build_merkle_proof",
    "verify_merkle_proof",
    "compute_tree_depth",
]