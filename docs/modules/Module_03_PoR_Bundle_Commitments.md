Module 03 — Proof of Reasoning (PoR) Bundle & Commitments

Module ID: M03
Owner Role: Protocol Engineer (commitments, bundle format, verification entrypoints)
Goal: Implement PoR artifacts and root computation so the pipeline can produce a verifiable, challengeable “Proof of Reasoning” package for each market resolution.

This module binds together:
	•	Canonical hashing (M01)
	•	Merkle roots/proofs (M02)
	•	Core schemas (M01)
…and produces a PoRBundle that downstream orchestrator and sentinel verifier can consume.

⸻

1) Owned Files

Under core/por/ implement:
	1.	reasoning_trace.py  (trace schema exists already in M01? In your tree it is here, so implement here as protocol trace types; it depends on core/schemas only)
	2.	por_bundle.py
	3.	proof_of_reasoning.py
	4.	__init__.py

Your core/schemas/ already covers Market/Prompt/Evidence/Verdict.
Here in core/por/ we define the trace structure (ReasoningTrace) and bundle assembly/verification.

⸻

2) Dependencies & Imports

Depends on
	•	core.schemas.canonical.dumps_canonical (for deterministic hashing via hash_canonical)
	•	core.crypto.hashing.hash_canonical, to_hex, from_hex
	•	core.merkle.merkle_tree.build_merkle_root, build_merkle_proof, verify_merkle_proof
	•	core.schemas.evidence.EvidenceBundle
	•	core.schemas.prompts.PromptSpec
	•	core.schemas.verdict.DeterministicVerdict
	•	core.schemas.verification.VerificationResult, CheckResult, ChallengeRef
	•	(optional) core.crypto.signatures.Signature if you want typed signatures in bundle
	•	(optional) core.crypto.attestation.TEEAttestation for fast-path anchoring

Used by
	•	orchestrator/sop_executor.py and orchestrator/pipeline.py
	•	agents/validator/sentinel_agent.py
	•	agents/judge/* (to fill verdict roots)
	•	agents/auditor/* (to build reasoning trace)

⸻

3) Scope

In scope
	•	Define ReasoningTrace and ReasoningStep structure and constraints
	•	Compute:
	•	prompt_spec_hash
	•	evidence_root
	•	reasoning_root
	•	verdict_hash
	•	(optional) combined por_root
	•	Build PoRBundle with these commitments
	•	Provide verification helpers:
	•	verify_por_bundle_structure(bundle)
	•	verify_commitments(bundle, prompt_spec, evidence_bundle, trace)
	•	prove_leaf(bundle, kind, index) (optional convenience)

Out of scope
	•	LLM prompting / determinism decoding policies (that’s core/llm/* + agents)
	•	Evidence provenance proof verification (collector verification module)
	•	Dispute protocol / on-chain settlement (later modules)

⸻

4) Commitments & Determinism Contracts

These rules must be identical across all modules.

4.1 PromptSpec hash
	•	prompt_spec_hash = to_hex(hash_canonical(prompt_spec))

4.2 Evidence leaves and root
	•	Evidence leaf i: leaf_i = hash_canonical(EvidenceItem) (bytes)
	•	Evidence root: to_hex(build_merkle_root([leaf_i...]))
	•	Leaf ordering: preserve EvidenceBundle.items order (no sorting)

4.3 Reasoning leaves and root
	•	Reasoning leaf j: leaf_j = hash_canonical(ReasoningStep) (bytes)
	•	Reasoning root: to_hex(build_merkle_root([leaf_j...]))
	•	Leaf ordering: preserve ReasoningTrace.steps order

4.4 Verdict hash
	•	verdict_hash = to_hex(hash_canonical(verdict))

4.5 Optional PoR root

4.6 PromptSpec Commitment Scope (Strict)
  prompt_spec_hash commits to the entire PromptSpec object as canonically serialized, including:
	•	market (question, event_definition, resolution_window, resolution_rules, allowed_sources, etc.)
	•	prediction_semantics
	•	data_requirements, including source_targets and selection_policy
	•	any other fields that are not None (since exclude_none=True)

  Therefore, any change to a requirement’s SourceTarget.uri, HTTP method, expected content type, or selection policy MUST change prompt_spec_hash and thus the PoR commitments. This is intended behavior.

4.7 PromptSpec Timestamp Rule (Hash Stability)
  To ensure stable commitments for identical user inputs:
	•	PromptSpec.created_at MUST be omitted from commitments by default.
	•	The schema contract requires PromptSpec.created_at to be OPTIONAL and default to None.
	•	The canonical serializer MUST exclude None fields.
	•	PoR commitment functions MUST compute prompt_spec_hash = hash_canonical(prompt_spec); as a result, created_at=None will not affect the hash.

  If timestamps are needed for operational logging, they MUST be placed in non-committed metadata fields (e.g., PoRBundle.created_at, PoRBundle.metadata) rather than in PromptSpec.
Recommended combined root:
	•	por_root = to_hex(build_merkle_root([from_hex(prompt_spec_hash), from_hex(evidence_root), from_hex(reasoning_root), from_hex(verdict_hash)]))

This gives you a single anchor hash.

⸻

5) Reasoning Trace Model (core/por/reasoning_trace.py)

Purpose

Defines the canonical trace format that the Auditor emits. This is what gets Merkle-committed so a challenger can pinpoint a broken step.

Models to implement (Pydantic v2 preferred)
	•	TracePolicy
	•	schema_version = SCHEMA_VERSION
	•	decoding_policy: str = "strict"
	•	allow_external_sources: bool = False
	•	max_steps: int = 200
	•	extra: dict
	•	ReasoningStep
	•	schema_version = SCHEMA_VERSION
	•	step_id: str (deterministic id recommended: step_0001 etc.)
	•	type: Literal["search","extract","check","deduce","aggregate","map"]
	•	inputs: dict
	•	action: str
	•	output: dict
	•	evidence_ids: list[str]
	•	prior_step_ids: list[str]
	•	hash_commitment: Optional[str] = None (optional: can be filled later; do not rely on it)
	•	ReasoningTrace
	•	schema_version = SCHEMA_VERSION
	•	trace_id: str
	•	policy: TracePolicy
	•	steps: list[ReasoningStep]
	•	evidence_refs: list[str]  (must be subset of EvidenceBundle evidence_ids)

Validation rules
	•	len(steps) <= policy.max_steps
	•	Every step_id unique
	•	Every prior_step_id must refer to an earlier step
	•	evidence_refs unique
	•	If policy.allow_external_sources is False:
	•	steps should not contain inputs["external_url"] (optional enforcement later)
(Full semantic enforcement can happen in Auditor verification module, but basic structural rules belong here.)

⸻

6) PoR Bundle Model (core/por/por_bundle.py)

Purpose

Defines the verifiable package that can be anchored and disputed.

Model: PoRBundle (Pydantic)

Fields:
	•	schema_version = SCHEMA_VERSION
	•	protocol_version = PROTOCOL_VERSION (import from schemas/versioning.py)
	•	market_id: str

Commitments:
	•	prompt_spec_hash: str  (0x…)
	•	evidence_root: str     (0x…)
	•	reasoning_root: str    (0x…)
	•	verdict_hash: str      (0x…)
	•	por_root: Optional[str] = None

Payload:
	•	verdict: DeterministicVerdict  (embed full verdict for convenience; can be on-chain hashed)

Anchoring / signatures (keep minimal, extensible):
	•	tee_attestation: Optional[TEEAttestation] = None
	•	signatures: dict = {}  (e.g., collector/auditor/judge/anchor signatures; store opaque for now)

Metadata:
	•	created_at: datetime
	•	metadata: dict

Validation rules
	•	All hash strings must be hex 0x prefixed and decode to 32 bytes (use from_hex)
	•	market_id must equal verdict.market_id
	•	If por_root provided, it must be consistent with other commitments (can verify in proof_of_reasoning.py)

⸻

7) Root Computation & Bundle Builder (core/por/proof_of_reasoning.py)

Purpose

Provide pure functions to compute hashes/roots and assemble/verify bundles.

Required types
	•	PoRRoots dataclass (or BaseModel) containing:
	•	prompt_spec_hash: str
	•	evidence_root: str
	•	reasoning_root: str
	•	verdict_hash: str
	•	por_root: str (recommended always computed)

Required functions

A) Leaf hash helpers
	•	compute_prompt_spec_hash(prompt_spec: PromptSpec) -> str
	•	compute_evidence_leaf_hashes(evidence: EvidenceBundle) -> list[bytes]
	•	compute_reasoning_leaf_hashes(trace: ReasoningTrace) -> list[bytes]
	•	compute_verdict_hash(verdict: DeterministicVerdict) -> str

B) Root computation
	•	compute_roots(prompt_spec, evidence, trace, verdict) -> PoRRoots
	•	uses M02 build_merkle_root
	•	uses M02 padding rule implicitly

C) Bundle builder
	•	build_por_bundle(prompt_spec, evidence, trace, verdict, *, include_por_root=True, tee_attestation=None, signatures=None, metadata=None) -> PoRBundle
	•	Computes commitments
	•	Embeds verdict
	•	Sets timestamps/metadata

D) Bundle verifier (structure + commitments)
	•	verify_por_bundle(bundle: PoRBundle, *, prompt_spec=None, evidence=None, trace=None) -> VerificationResult
	•	Always checks structural validity (hash format, market_id match)
	•	If prompt/evidence/trace provided, recompute roots and ensure equal
	•	If por_root present, recompute and ensure equal
	•	Return VerificationResult(ok=..., checks=[...], challenge=...)

ChallengeRef strategy
When mismatch found, return challenge identifying which component failed:
	•	If evidence root mismatch: ChallengeRef(kind="evidence_leaf", leaf_index=?) if you can isolate; otherwise just kind without index
	•	If reasoning root mismatch: kind="reasoning_leaf"
	•	If verdict mismatch: kind="verdict_hash"
	•	If por_root mismatch: kind="por_bundle"

Pinpointing exact leaf index is optional in M03; can be implemented later (sentinel module may handle pinpointing).

Bundle Metadata vs Committed Content
  •	PoRBundle.created_at and PoRBundle.metadata may include operational timestamps and runtime identifiers because they are not used to derive prompt_spec_hash, evidence_root, reasoning_root, or verdict_hash unless explicitly committed. The only committed values are the hashes/roots defined in this doc.

compute_prompt_spec_hash() MUST explicitly set/require prompt_spec.created_at is None (or ignore it) before hashing, and return a verification error if a non-None created_at is present in strict mode.
⸻

8) core/por/__init__.py Exports

Export minimum public API:
	•	ReasoningTrace, ReasoningStep, TracePolicy
	•	PoRBundle
	•	PoRRoots, compute_roots, build_por_bundle, verify_por_bundle

⸻

9) Tests Required (Add Under tests/unit)

Create:
	•	tests/unit/test_por_roots.py
	•	tests/unit/test_por_bundle.py

9.1 test_por_roots.py
	•	root determinism for same objects
	•	evidence root changes if any EvidenceItem changes
	•	reasoning root changes if any ReasoningStep changes
	•	verdict hash changes if verdict changes
	•	por_root changes if any component changes

9.2 test_por_bundle.py
	•	bundle builds with consistent commitments
	•	verify_por_bundle(bundle, prompt_spec, evidence, trace) returns ok
	•	tamper bundle.evidence_root → verify fails with challenge kind evidence/por
	•	tamper embedded verdict.outcome but not verdict_hash → verify fails on verdict_hash mismatch
	•	hash format validation (non-0x should fail)

⸻

10) Implementation Notes / Pitfalls
	1.	Never sort evidence items or trace steps for hashing—order must be preserved.
	2.	Use from_hex() to validate hash strings are 32 bytes.
	3.	Keep verify_por_bundle() side-effect free (pure).
	4.	Do not embed huge evidence/trace inside bundle yet (keep minimal for now).
	•	Bundle includes commitments + verdict; the evidence/trace are “off-chain artifacts” referenced by hashes.
	5.	Keep each file < 500 LOC by avoiding too many helper classes—prefer small pure functions.

⸻

11) Acceptance Checklist (“Done When”)
	•	ReasoningTrace models implemented with basic structural validation
	•	PoRBundle model implemented with hash-format validation
	•	compute_roots() computes prompt/evidence/reasoning/verdict hashes and combined por_root deterministically
	•	build_por_bundle() assembles a bundle with correct commitments
	•	verify_por_bundle() validates structure and optionally recomputes commitments
	•	Unit tests cover determinism and tamper detection
	•	All implemented files remain ≤ 500 LOC each
