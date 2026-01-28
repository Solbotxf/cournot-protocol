Module 08 — Validator/Sentinel (Replay Executor & Challenges)

Module ID: M08
Owner Role: Protocol Verification Engineer
Goal: Provide an independent validator ("sentinel") that can:
	1.	verify a provided PoR package (bundle + artifacts) end-to-end,
	2.	optionally replay steps deterministically (especially evidence fetches),
	3.	detect mismatches / policy failures,
	4.	generate pinpoint challenges (Merkle proofs + disputed leaf references) suitable for dispute resolution.

This module is your "external verifier node" implementation.

⸻

1) Owned Files

Under agents/validator/ implement:
	•	sentinel_agent.py
	•	replay_executor.py
	•	challenges.py
	•	__init__.py

⸻

2) Dependencies

Protocol & schemas
	•	core.por.por_bundle.PoRBundle
	•	core.por.proof_of_reasoning.verify_por_bundle, compute_roots (recommended)
	•	core.por.reasoning_trace.ReasoningTrace
	•	core.schemas.prompts.PromptSpec
	•	core.schemas.transport.ToolPlan
	•	core.schemas.evidence.EvidenceBundle
	•	core.schemas.verdict.DeterministicVerdict
	•	core.schemas.verification.VerificationResult, CheckResult, ChallengeRef
	•	core.crypto.hashing.from_hex, to_hex, hash_canonical
	•	core.merkle.build_merkle_proof, verify_merkle_proof

Collector replay (for optional evidence replay)
	•	agents.collector.collector_agent.CollectorAgent
	•	agents.collector.data_sources.* (or a replay-safe subset)

⸻

3) Scope

In scope
	•	Verify commitments: prompt hash, evidence root, reasoning root, verdict hash, por_root
	•	Verify artifact integrity: schemas, strict mode expectations
	•	Generate challenges:
	•	"evidence leaf mismatch" proof
	•	"reasoning leaf mismatch" proof
	•	"verdict hash mismatch" proof
	•	"prompt spec mismatch" hash proof (just hash mismatch)
	•	Optional replay of evidence using stored SourceTargets:
	•	compare newly fetched response fingerprints to recorded ones
	•	highlight divergence

Out of scope
	•	Actual on-chain dispute submission (later module)
	•	Producing new verdicts beyond verifying (but can re-run Judge to compare)

⸻

4) Validator Operating Modes

Sentinel supports 2 modes:

Mode A — Verify-only (default)
	•	No network calls
	•	Uses provided PromptSpec, EvidenceBundle, ReasoningTrace, Verdict
	•	Recomputes roots and compares to PoRBundle commitments
	•	Generates challenges for mismatches

Mode B — Replay (optional)
	•	Uses PromptSpec.data_requirements.source_targets + ToolPlan to re-fetch evidence
	•	Builds a "replayed EvidenceBundle" deterministically
	•	Compares replayed evidence roots / evidence fingerprints with provided bundle
	•	Emits "replay divergence" checks and challenges

Replay can be disabled in strict environments or when offline.

⸻

5) Inputs / Outputs

Primary entrypoint

SentinelAgent.verify(package: PoRPackage, *, mode="verify"|"replay") -> tuple[VerificationResult, list[Challenge]]

Where PoRPackage is an internal container that includes:
	•	bundle: PoRBundle
	•	prompt_spec: PromptSpec
	•	tool_plan: ToolPlan | None
	•	evidence: EvidenceBundle
	•	trace: ReasoningTrace
	•	verdict: DeterministicVerdict

You can define PoRPackage in sentinel_agent.py as a dataclass (internal only).

Output
	•	VerificationResult includes:
	•	ok: bool
	•	checks: list[CheckResult]
	•	optional challenge: ChallengeRef (top-level hint)
	•	Challenge[] (detailed objects defined in challenges.py)

⸻

6) Challenge Model (agents/validator/challenges.py)

Define a serializable "challenge packet" that later modules can send on-chain or to peers.

Challenge fields
	•	challenge_id: str (deterministic from inputs)
	•	kind: Literal["prompt_hash","evidence_leaf","reasoning_leaf","verdict_hash","por_root","replay_divergence"]
	•	market_id: str
	•	bundle_root: str (por_root)
	•	component_root: str | None (evidence_root/reasoning_root if relevant)
	•	leaf_index: int | None
	•	leaf_hash: str | None
	•	merkle_proof: dict | None
	•	siblings: list[str] (hex)
	•	index: int
	•	root: str
	•	details: dict (freeform diagnostics)

Deterministic challenge_id
	•	challenge_id = "ch_" + sha256(canonical(kind|market_id|bundle_root|leaf_index|leaf_hash))[:12]

⸻

7) Verification Steps (sentinel_agent.py)

SentinelAgent interface

class SentinelAgent(BaseAgent):
    role = "validator"

    def verify(self, package: PoRPackage, *, mode: str = "verify") -> tuple[VerificationResult, list[Challenge]]:
        ...

Verify-only algorithm
	1.	Schema lock / strict checks
	•	ensure prompt_spec.extra["strict_mode"] == True (if strict pipeline)
	•	ensure prompt_spec.output_schema_ref is DeterministicVerdict
	2.	Bundle structural verification
	•	call verify_por_bundle(bundle, prompt_spec=..., evidence=..., trace=...)
	3.	Recompute commitments
	•	roots = compute_roots(prompt_spec, evidence, trace, verdict)
	•	compare with bundle fields
	4.	Cross-check verdict
	•	recompute hash_canonical(verdict) and compare to bundle.verdict_hash
	5.	If mismatch found
	•	attempt to create pinpoint challenge:
	•	if evidence_root mismatch → locate mismatching leaf (see Section 8)
	•	if reasoning_root mismatch → locate mismatching step leaf
	•	if verdict mismatch → create verdict_hash challenge
	•	if prompt mismatch → prompt_hash challenge
	6.	Return VerificationResult + challenges list

⸻

8) Pinpointing Mismatched Leaves (replay_executor.py or sentinel_agent.py)

Pinpointing is optional but strongly recommended.

Evidence leaf mismatch pinpointing

Given:
	•	expected evidence_root from bundle
	•	computed evidence_root from evidence bundle

Procedure:
	1.	Compute leaf hashes list for evidence bundle: L = [hash_canonical(item_i)]
	2.	Build Merkle proof for each leaf index from computed leaves against computed root
	3.	Compare computed root vs expected root:
	•	If roots mismatch, you still can generate a challenge referencing a specific leaf by:
	•	presenting leaf hash + proof against computed root (proves what you have),
	•	and claiming it differs from the bundle commitment
	4.	Better: try to identify which leaf causes mismatch:
	•	If you can access the "expected leaves" list (rare), compare directly.
	•	Otherwise:
	•	Generate challenge that disputes entire evidence_root (kind="evidence_leaf", leaf_index=None) OR
	•	Provide first leaf proof as a representative mismatch

Recommended approach (practical and simple):
	•	Always output a root mismatch challenge (kind="evidence_leaf", leaf_index=None)
	•	Additionally output N (e.g., 1–3) sample leaf proofs from your computed bundle for dispute sampling

Reasoning leaf mismatch pinpointing

Same as evidence:
	•	leaf_j = hash_canonical(ReasoningStep)
	•	build proof and include step_id in details

⸻

9) Replay Mode (replay_executor.py)

Purpose

Rebuild evidence deterministically from PromptSpec + ToolPlan and compare.

Implement:

class ReplayExecutor:
    def __init__(self, collector: CollectorAgent):
        ...

    def replay_evidence(self, prompt_spec: PromptSpec, tool_plan: ToolPlan) -> tuple[EvidenceBundle, VerificationResult]:
        ...

Rules:
	•	Must use the same selection policy and SourceTarget ordering
	•	Must avoid non-deterministic fields (timestamps default None)
	•	Must compare:
	•	request_fingerprint and response_fingerprint if available
	•	evidence_root differences

Output checks:
	•	replay_divergence if fingerprints differ
	•	replay_unavailable if a source fails to fetch

Challenge emission in replay mode:
	•	If replayed evidence_root differs from provided bundle evidence_root:
	•	emit Challenge(kind="replay_divergence") with both roots and details

⸻

10) Challenge Construction Helpers (challenges.py)

Implement helper functions:
	•	make_prompt_hash_challenge(...)
	•	make_verdict_hash_challenge(...)
	•	make_merkle_leaf_challenge(kind, market_id, component_root, leaf_hash, proof, ...)

For Merkle proofs:
	•	Use build_merkle_proof(leaves, idx) from Module 02
	•	Serialize proof siblings as hex strings

⸻

11) Tests Required

Add unit tests:

tests/unit/test_sentinel_verify.py
	•	valid package passes
	•	tamper bundle.evidence_root fails and produces challenge kind evidence_leaf
	•	tamper trace step changes reasoning_root mismatch challenge
	•	tamper verdict outcome produces verdict_hash challenge

tests/unit/test_replay_executor.py
	•	mock collector to return deterministic evidence bundle
	•	replay divergence produces replay_divergence challenge
	•	replay disabled or missing tool_plan handles gracefully

Mock everything; no network calls.

⸻

12) Acceptance Checklist
	•	Sentinel verifies PoR commitments end-to-end in verify-only mode
	•	When mismatch occurs, outputs structured challenges with Merkle proofs (where applicable)
	•	Replay mode can rebuild evidence (mocked) and detect divergence deterministically
	•	Unit tests cover mismatch detection + challenge creation
	•	Each file ≤ 500 LOC