Module 02 — Merkle & Commitments

Module ID: M02
Owner Role: Protocol/Crypto Engineer (hashing, Merkle proofs, determinism)
Goal: Provide deterministic Merkle tree construction + proof generation/verification for PoR artifacts (Evidence leaves, Reasoning leaves, Verdict leaf, etc.). This module underpins pinpoint verification and is used by core/por/*.

⸻

1) Owned Files (Implement These)

Primary owned
	•	core/merkle/merkle_tree.py
	•	core/merkle/merkle_proofs.py
	•	core/merkle/__init__.py

Required dependency to implement (minimal surface for leaves)
	•	core/crypto/hashing.py (only the hashing helpers; signatures/attestation are not part of M02)

Note: core/schemas/canonical.py from M01 must exist because leaves should be hashed from canonical JSON.

⸻

2) Dependencies & Inputs

2.1 Depends on
	•	core/schemas/canonical.dumps_canonical (deterministic serialization)
	•	hashlib (sha256)
	•	Standard libs: dataclasses, typing

2.2 Used by
	•	core/por/proof_of_reasoning.py (leaf hashes + roots)
	•	agents/validator/* (pinpoint challenge proofs)
	•	orchestrator/* (bundle assembly)

⸻

3) Scope

In scope
	•	SHA-256 hashing utilities for canonical data
	•	Merkle root computation for a list of leaves
	•	Merkle proof generation for any leaf index
	•	Merkle proof verification
	•	Deterministic padding rule for odd number of leaves
	•	Clear, stable leaf ordering assumptions

Out of scope
	•	Any signature schemes (that’s core/crypto/signatures.py)
	•	TEE attestation (that’s core/crypto/attestation.py)
	•	PoR bundle schema (that’s M03)
	•	Network/storage (no disk IO)

⸻

4) Canonical Commitment Rules (Hard Contracts)

This module defines how commitments are computed:

4.1 Leaf hashing

A “leaf” is a bytes32 produced by hashing canonical JSON.

Rule:
	•	For a model/dict obj:
	•	leaf = sha256( dumps_canonical(obj).encode("utf-8") )

This is implemented via hashing.hash_canonical(obj).

4.2 Merkle parent hashing

Parent hash MUST be deterministic:
	•	parent = sha256(left + right) where left and right are raw bytes of the child hashes.

4.3 Padding rule for odd nodes (must pick one)

Pick one and document it; all modules must use the same rule.

Recommended (standard and simple): Duplicate last hash
	•	If a level has odd number of nodes, duplicate the last node to make it even.

Example:
	•	nodes: [a, b, c] → pad → [a, b, c, c]

This ensures root is deterministic.

4.4 Empty leaves behavior

Define and freeze behavior:
	•	If leaves=[], build_merkle_root([]) returns sha256(b"") (recommended)
OR return b"\x00"*32 (less standard)

Recommended: sha256(b"")

⸻

5) Public API (Functions/Types Other Modules Depend On)

5.1 core/crypto/hashing.py

Must expose:
	•	sha256(data: bytes) -> bytes
	•	hash_bytes(data: bytes) -> bytes (alias ok)
	•	hash_canonical(obj: Any) -> bytes
	•	to_hex(b: bytes) -> str  (e.g., 0x...)
	•	from_hex(s: str) -> bytes (validate 0x prefix)

5.2 core/merkle/merkle_tree.py

Must expose:
	•	MerkleProof dataclass (or Pydantic, but dataclass is fine)
	•	leaf: bytes
	•	index: int
	•	siblings: list[bytes] (bottom-up)
	•	root: bytes
	•	merkle_parent(left: bytes, right: bytes) -> bytes
	•	build_merkle_root(leaves: Sequence[bytes]) -> bytes
	•	build_merkle_proof(leaves: Sequence[bytes], index: int) -> MerkleProof
	•	verify_merkle_proof(proof: MerkleProof) -> bool

5.3 core/merkle/merkle_proofs.py

Must expose convenience wrappers:
	•	MerkleProver.prove(leaves, index) -> MerkleProof
	•	MerkleVerifier.verify(proof) -> bool

⸻

6) File-by-File Implementation Specs

6.1 core/crypto/hashing.py

Purpose: Basic hashing and canonical hashing utilities.

Implementation requirements
	•	sha256() using hashlib.sha256
	•	hash_canonical() uses core.schemas.canonical.dumps_canonical
	•	to_hex() uses 0x prefix
	•	from_hex() validates:
	•	must start with 0x
	•	must have even length
	•	must decode to bytes

Security/determinism
	•	Always hash raw bytes exactly as specified.
	•	Don’t auto-strip whitespace beyond canonical JSON.

⸻

6.2 core/merkle/merkle_tree.py

Purpose: Deterministic Merkle tree functions.

Padding behavior
	•	Duplicate last node at each level if odd.

Implementation approach
	•	Implement iterative level-building:
	•	start level = list(leaves)
	•	if empty → return sha256(b"")
	•	while len(level) > 1:
	•	if odd: append last element
	•	build next_level by pairing
	•	root is single element

Proof generation
	•	Store sibling for each level for the target index
	•	Each time:
	•	if odd: pad
	•	sibling index = i ^ 1
	•	record sibling hash
	•	update i = i // 2

Verification
	•	Recompute from leaf and siblings using index parity:
	•	if index even: hash = parent(hash, sibling)
	•	if index odd: hash = parent(sibling, hash)
	•	index = index // 2
	•	check equals root

Edge cases
	•	index out of range → raise IndexError
	•	leaves not bytes or wrong length → either accept any length bytes or enforce 32 bytes
	•	Recommended: accept any bytes but document typical 32 bytes.

⸻

6.3 core/merkle/merkle_proofs.py

Purpose: thin wrappers around tree functions.

Keep it tiny:
	•	MerkleProver calls build_merkle_proof
	•	MerkleVerifier calls verify_merkle_proof

⸻

6.4 core/merkle/__init__.py

Export:
	•	MerkleProof, build_merkle_root, build_merkle_proof, verify_merkle_proof

⸻

7) Required Tests (Add in tests/unit/)

Create new test file(s):
	•	tests/unit/test_merkle_tree.py
	•	tests/unit/test_hashing.py

7.1 test_hashing.py
	•	hash_bytes stable
	•	hash_canonical stable for dict key ordering differences
	•	to_hex/from_hex round trip

7.2 test_merkle_tree.py

Must include:
	1.	Root determinism

	•	same leaves → same root across runs

	2.	Padding correctness

	•	for odd leaf count, ensure behavior matches “duplicate last”
	•	verify root equals expected (compute manually for small case)

	3.	Proof verification

	•	generate proof for each index, verify passes
	•	tamper sibling, verify fails
	•	tamper leaf, verify fails
	•	tamper root, verify fails

	4.	Empty leaves

	•	build_merkle_root([]) returns sha256(b"")

	5.	Single leaf

	•	root equals leaf (or equals sha256(leaf)?)
	•	Important: define rule:
	•	Recommended: root = leaf when only one leaf hash exists (standard Merkle tree behavior).
	•	tests should confirm this.

⸻

8) Determinism & Interop Notes
	•	Do not include any randomness or non-deterministic ordering.
	•	Leaf ordering must be defined by upstream:
	•	evidence leaves ordered by EvidenceBundle.items order
	•	reasoning leaves ordered by ReasoningTrace.steps order
	•	This module never sorts leaves—it trusts input order.

⸻

9) Acceptance Checklist (“Done When”)
	•	hashing.py implements canonical hashing using dumps_canonical
	•	build_merkle_root is deterministic, correct padding behavior documented
	•	build_merkle_proof + verify_merkle_proof work for all leaf indices
	•	Unit tests cover determinism, padding, empty tree, and tamper detection
	•	All files remain ≤ 500 LOC each
	•	Public exports exist as specified in __init__.py

⸻
