"""
Module 02 - Merkle Tree Unit Tests
Tests for core/merkle/merkle_tree.py

Required tests per Module 02 spec:
1. Root determinism - same leaves → same root across runs
2. Padding correctness - odd leaf count uses "duplicate last" rule
3. Proof verification - generate proof for each index, verify passes
4. Tamper detection - tampered sibling/leaf/root fails verification
5. Empty leaves - build_merkle_root([]) returns sha256(b"")
6. Single leaf - root equals leaf
"""
import pytest

from core.crypto.hashing import sha256, hash_canonical
from core.merkle.merkle_tree import (
    EMPTY_TREE_ROOT,
    MerkleProof,
    merkle_parent,
    build_merkle_root,
    build_merkle_proof,
    verify_merkle_proof,
    compute_tree_depth,
)
from core.merkle.merkle_proofs import MerkleProver, MerkleVerifier


class TestEmptyTree:
    """Tests for empty tree behavior."""
    
    def test_empty_leaves_returns_sha256_empty(self):
        """build_merkle_root([]) returns sha256(b"")."""
        result = build_merkle_root([])
        expected = sha256(b"")
        
        assert result == expected
        assert result == EMPTY_TREE_ROOT
    
    def test_empty_tree_root_constant(self):
        """EMPTY_TREE_ROOT matches sha256(b"")."""
        assert EMPTY_TREE_ROOT == sha256(b"")
    
    def test_build_proof_empty_raises(self):
        """Cannot generate proof for empty tree."""
        with pytest.raises(ValueError, match="empty"):
            build_merkle_proof([], 0)


class TestSingleLeaf:
    """Tests for single leaf tree behavior."""
    
    def test_single_leaf_root_equals_leaf(self):
        """Root of single-leaf tree equals the leaf itself."""
        leaf = sha256(b"single leaf")
        root = build_merkle_root([leaf])
        
        assert root == leaf
    
    def test_single_leaf_proof_no_siblings(self):
        """Proof for single leaf has no siblings."""
        leaf = sha256(b"only one")
        proof = build_merkle_proof([leaf], 0)
        
        assert proof.leaf == leaf
        assert proof.index == 0
        assert proof.siblings == []
        assert proof.root == leaf
    
    def test_single_leaf_proof_verifies(self):
        """Proof for single leaf verifies correctly."""
        leaf = sha256(b"single")
        proof = build_merkle_proof([leaf], 0)
        
        assert verify_merkle_proof(proof)


class TestRootDeterminism:
    """Tests for deterministic root computation."""
    
    def test_same_leaves_same_root(self):
        """Same leaves produce same root across multiple calls."""
        leaves = [sha256(b"a"), sha256(b"b"), sha256(b"c")]
        
        roots = [build_merkle_root(leaves) for _ in range(10)]
        
        assert all(r == roots[0] for r in roots)
    
    def test_root_deterministic_different_runs(self):
        """Root is deterministic (simulated by recreating leaves)."""
        # Recreate leaves from scratch
        def make_leaves():
            return [sha256(f"leaf{i}".encode()) for i in range(5)]
        
        root1 = build_merkle_root(make_leaves())
        root2 = build_merkle_root(make_leaves())
        
        assert root1 == root2
    
    def test_different_leaves_different_roots(self):
        """Different leaves produce different roots."""
        leaves1 = [sha256(b"a"), sha256(b"b")]
        leaves2 = [sha256(b"x"), sha256(b"y")]
        
        assert build_merkle_root(leaves1) != build_merkle_root(leaves2)
    
    def test_leaf_order_matters(self):
        """Different leaf ordering produces different roots."""
        leaves1 = [sha256(b"a"), sha256(b"b"), sha256(b"c")]
        leaves2 = [sha256(b"c"), sha256(b"b"), sha256(b"a")]
        
        assert build_merkle_root(leaves1) != build_merkle_root(leaves2)


class TestPaddingCorrectness:
    """Tests for odd-number padding behavior."""
    
    def test_padding_rule_three_leaves(self):
        """Three leaves use duplicate-last padding."""
        a = sha256(b"a")
        b = sha256(b"b")
        c = sha256(b"c")
        
        # Manual calculation:
        # Level 0: [a, b, c, c]  (c duplicated)
        # Level 1: [parent(a,b), parent(c,c)]
        # Level 2: [parent(parent(a,b), parent(c,c))]
        
        ab = merkle_parent(a, b)
        cc = merkle_parent(c, c)
        expected_root = merkle_parent(ab, cc)
        
        actual_root = build_merkle_root([a, b, c])
        
        assert actual_root == expected_root
    
    def test_padding_rule_five_leaves(self):
        """Five leaves use duplicate-last padding at multiple levels."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(5)]
        a, b, c, d, e = leaves
        
        # Manual calculation:
        # Level 0: [a, b, c, d, e, e]  (e duplicated)
        # Level 1: [ab, cd, ee]
        # Level 2: [ab, cd, ee, ee]  (ee duplicated)
        # Level 3: [abcd, eeee]
        # Level 4: [root]
        
        ab = merkle_parent(a, b)
        cd = merkle_parent(c, d)
        ee = merkle_parent(e, e)
        
        abcd = merkle_parent(ab, cd)
        eeee = merkle_parent(ee, ee)
        
        expected_root = merkle_parent(abcd, eeee)
        actual_root = build_merkle_root(leaves)
        
        assert actual_root == expected_root
    
    def test_even_leaves_no_padding_needed(self):
        """Even number of leaves needs no padding at leaf level."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        a, b, c, d = leaves
        
        # Manual calculation:
        # Level 0: [a, b, c, d]
        # Level 1: [ab, cd]
        # Level 2: [root]
        
        ab = merkle_parent(a, b)
        cd = merkle_parent(c, d)
        expected_root = merkle_parent(ab, cd)
        
        assert build_merkle_root(leaves) == expected_root


class TestProofVerification:
    """Tests for proof generation and verification."""
    
    def test_proof_verifies_for_each_index(self):
        """Generate and verify proof for every leaf index."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(7)]
        
        for i in range(len(leaves)):
            proof = build_merkle_proof(leaves, i)
            assert verify_merkle_proof(proof), f"Proof failed for index {i}"
    
    def test_proof_contains_correct_leaf(self):
        """Proof contains the correct leaf hash."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(5)]
        
        for i, expected_leaf in enumerate(leaves):
            proof = build_merkle_proof(leaves, i)
            assert proof.leaf == expected_leaf
            assert proof.index == i
    
    def test_proof_root_matches_tree_root(self):
        """Proof root matches independently computed tree root."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(6)]
        expected_root = build_merkle_root(leaves)
        
        for i in range(len(leaves)):
            proof = build_merkle_proof(leaves, i)
            assert proof.root == expected_root
    
    def test_proof_index_out_of_range_raises(self):
        """Out-of-range index raises IndexError."""
        leaves = [sha256(b"a"), sha256(b"b"), sha256(b"c")]
        
        with pytest.raises(IndexError):
            build_merkle_proof(leaves, 3)
        
        with pytest.raises(IndexError):
            build_merkle_proof(leaves, -1)
        
        with pytest.raises(IndexError):
            build_merkle_proof(leaves, 100)


class TestTamperDetection:
    """Tests for tamper detection (invalid proofs fail)."""
    
    def test_tampered_sibling_fails(self):
        """Proof with tampered sibling fails verification."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        proof = build_merkle_proof(leaves, 1)
        
        # Tamper with a sibling
        tampered_siblings = list(proof.siblings)
        tampered_siblings[0] = sha256(b"tampered")
        
        tampered_proof = MerkleProof(
            leaf=proof.leaf,
            index=proof.index,
            siblings=tampered_siblings,
            root=proof.root,
        )
        
        assert not verify_merkle_proof(tampered_proof)
    
    def test_tampered_leaf_fails(self):
        """Proof with tampered leaf fails verification."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        proof = build_merkle_proof(leaves, 2)
        
        tampered_proof = MerkleProof(
            leaf=sha256(b"wrong leaf"),
            index=proof.index,
            siblings=proof.siblings,
            root=proof.root,
        )
        
        assert not verify_merkle_proof(tampered_proof)
    
    def test_tampered_root_fails(self):
        """Proof with tampered root fails verification."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        proof = build_merkle_proof(leaves, 0)
        
        tampered_proof = MerkleProof(
            leaf=proof.leaf,
            index=proof.index,
            siblings=proof.siblings,
            root=sha256(b"wrong root"),
        )
        
        assert not verify_merkle_proof(tampered_proof)
    
    def test_wrong_index_fails(self):
        """Proof with wrong index fails verification."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        proof = build_merkle_proof(leaves, 1)
        
        # Use wrong index
        tampered_proof = MerkleProof(
            leaf=proof.leaf,
            index=2,  # Wrong index
            siblings=proof.siblings,
            root=proof.root,
        )
        
        assert not verify_merkle_proof(tampered_proof)
    
    def test_missing_sibling_fails(self):
        """Proof with missing sibling fails verification."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(8)]
        proof = build_merkle_proof(leaves, 3)
        
        # Remove a sibling
        truncated_siblings = proof.siblings[:-1]
        
        tampered_proof = MerkleProof(
            leaf=proof.leaf,
            index=proof.index,
            siblings=truncated_siblings,
            root=proof.root,
        )
        
        assert not verify_merkle_proof(tampered_proof)


class TestMerkleParent:
    """Tests for merkle_parent() function."""
    
    def test_merkle_parent_deterministic(self):
        """merkle_parent is deterministic."""
        left = sha256(b"left")
        right = sha256(b"right")
        
        results = [merkle_parent(left, right) for _ in range(5)]
        
        assert all(r == results[0] for r in results)
    
    def test_merkle_parent_order_matters(self):
        """merkle_parent(a, b) != merkle_parent(b, a)."""
        a = sha256(b"a")
        b = sha256(b"b")
        
        assert merkle_parent(a, b) != merkle_parent(b, a)
    
    def test_merkle_parent_equals_sha256_concat(self):
        """merkle_parent equals sha256 of concatenation."""
        left = sha256(b"left")
        right = sha256(b"right")
        
        expected = sha256(left + right)
        actual = merkle_parent(left, right)
        
        assert actual == expected


class TestComputeTreeDepth:
    """Tests for compute_tree_depth() function."""
    
    def test_depth_empty(self):
        """Empty tree has depth 0."""
        assert compute_tree_depth(0) == 0
    
    def test_depth_single_leaf(self):
        """Single leaf has depth 1."""
        assert compute_tree_depth(1) == 1
    
    def test_depth_two_leaves(self):
        """Two leaves have depth 2."""
        assert compute_tree_depth(2) == 2
    
    def test_depth_power_of_two(self):
        """Power of two leaves."""
        assert compute_tree_depth(4) == 3  # leaf level + 2 parent levels
        assert compute_tree_depth(8) == 4
        assert compute_tree_depth(16) == 5
    
    def test_depth_non_power_of_two(self):
        """Non-power-of-two requires extra depth due to padding."""
        assert compute_tree_depth(3) == 3  # 3->4 leaves, depth 3
        assert compute_tree_depth(5) == 4  # 5->6->8 leaves, depth 4
        assert compute_tree_depth(7) == 4  # 7->8 leaves, depth 4


class TestConvenienceClasses:
    """Tests for MerkleProver and MerkleVerifier classes."""
    
    def test_prover_prove(self):
        """MerkleProver.prove works correctly."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        
        proof = MerkleProver.prove(leaves, 2)
        
        assert proof.leaf == leaves[2]
        assert MerkleVerifier.verify(proof)
    
    def test_prover_compute_root(self):
        """MerkleProver.compute_root matches build_merkle_root."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(5)]
        
        expected = build_merkle_root(leaves)
        actual = MerkleProver.compute_root(leaves)
        
        assert actual == expected
    
    def test_prover_prove_object(self):
        """MerkleProver.prove_object works with canonical hashing."""
        objects = [
            {"id": 1, "data": "first"},
            {"id": 2, "data": "second"},
            {"id": 3, "data": "third"},
        ]
        
        proof = MerkleProver.prove_object(objects, 1)
        
        # Verify leaf is hash of canonicalized object
        expected_leaf = hash_canonical(objects[1])
        assert proof.leaf == expected_leaf
        
        assert MerkleVerifier.verify(proof)
    
    def test_verifier_verify_leaf_in_root(self):
        """MerkleVerifier.verify_leaf_in_root works."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(4)]
        proof = build_merkle_proof(leaves, 1)
        
        result = MerkleVerifier.verify_leaf_in_root(
            leaf=proof.leaf,
            index=proof.index,
            siblings=proof.siblings,
            root=proof.root,
        )
        
        assert result
    
    def test_verifier_verify_object_in_root(self):
        """MerkleVerifier.verify_object_in_root works."""
        objects = [{"key": f"value{i}"} for i in range(4)]
        leaves = [hash_canonical(obj) for obj in objects]
        root = build_merkle_root(leaves)
        proof = build_merkle_proof(leaves, 2)
        
        result = MerkleVerifier.verify_object_in_root(
            obj=objects[2],
            index=2,
            siblings=proof.siblings,
            root=root,
        )
        
        assert result


class TestLargeTree:
    """Tests with larger trees."""
    
    def test_large_tree_all_proofs_valid(self):
        """All proofs verify in a larger tree."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(100)]
        
        for i in range(len(leaves)):
            proof = build_merkle_proof(leaves, i)
            assert verify_merkle_proof(proof), f"Failed at index {i}"
    
    def test_large_tree_root_deterministic(self):
        """Large tree root is deterministic."""
        leaves = [sha256(f"leaf{i}".encode()) for i in range(256)]
        
        root1 = build_merkle_root(leaves)
        root2 = build_merkle_root(leaves)
        
        assert root1 == root2


class TestMerkleProofDataclass:
    """Tests for MerkleProof dataclass."""
    
    def test_merkle_proof_immutable(self):
        """MerkleProof is frozen (immutable)."""
        proof = MerkleProof(
            leaf=sha256(b"test"),
            index=0,
            siblings=[],
            root=sha256(b"test"),
        )
        
        with pytest.raises(AttributeError):
            proof.index = 1
    
    def test_merkle_proof_negative_index_raises(self):
        """MerkleProof rejects negative index."""
        with pytest.raises(ValueError, match="non-negative"):
            MerkleProof(
                leaf=sha256(b"test"),
                index=-1,
                siblings=[],
                root=sha256(b"test"),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])