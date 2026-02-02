Module 05 — Collector & Provenance

Module ID: M05
Owner Role: Data/Infra Engineer (retrieval, adapters, provenance, determinism)
Goal: Execute ToolPlan + PromptSpec.data_requirements to produce a deterministic, policy-compliant EvidenceBundle that downstream Auditor/Judge can rely on and that can be Merkle-committed.

Collector is responsible for:
	1.	executing explicit SourceTarget requests (URLs/endpoints)
	2.	capturing retrieval receipts (timestamps, fingerprints)
	3.	attaching provenance proofs (tiered)
	4.	enforcing minimum provenance tier + selection policy
	5.	outputting EvidenceBundle with stable evidence_id and stable item ordering

⸻

1) Owned Files

Under agents/collector/ implement:

Agent
	•	collector_agent.py
	•	__init__.py

Data sources (adapters)
	•	data_sources/base_source.py
	•	data_sources/http_source.py
	•	data_sources/web_source.py
	•	data_sources/polymarket_source.py
	•	data_sources/__init__.py

Verification/policy
	•	verification/tier_policy.py
	•	verification/signature_verifier.py
	•	verification/zktls_verifier.py
	•	verification/__init__.py

⸻

2) Dependencies

Schemas
	•	core.schemas.prompts.PromptSpec, DataRequirement
	•	core.schemas.transport.ToolPlan, SourceTarget, SelectionPolicy  (per updated Doc 01/04)
	•	core.schemas.evidence.EvidenceBundle, EvidenceItem, SourceDescriptor, RetrievalReceipt, ProvenanceProof
	•	core.schemas.verification.VerificationResult, CheckResult, ChallengeRef
	•	core.schemas.errors.CournotError

Protocol utilities
	•	core.schemas.canonical.dumps_canonical
	•	core.crypto.hashing.hash_canonical, to_hex

Optional libs (if you include)
	•	requests or httpx for HTTP (pick one; keep minimal)
	•	For web source, you may stub (no browser automation required yet)

⸻

3) Scope

In scope
	•	Deterministic evidence ID generation
	•	Executing explicit source targets (SourceTarget.uri, method, params)
	•	Capturing content + normalized form
	•	Producing receipts and provenance proof shells
	•	Policy enforcement:
	•	min provenance tier
	•	fallback/quorum selection policy
	•	Return EvidenceBundle + per-item verification results (embedded or returned separately)

Out of scope
	•	Auditor reasoning
	•	Judge mapping
	•	PoR commitments (computed later)
	•	Real zkTLS proof generation (may be stubbed with placeholder proof_blob)

⸻

4) Collector Input / Output Contracts

Input

CollectorAgent.collect(prompt_spec: PromptSpec, tool_plan: ToolPlan) -> EvidenceBundle

Collector MUST use:
	•	prompt_spec.data_requirements[*].source_targets
	•	prompt_spec.data_requirements[*].selection_policy
	•	prompt_spec.market.allowed_sources (whitelist)
	•	prompt_spec.market.min_provenance_tier and requirement-level min_provenance_tier

Output

EvidenceBundle must be:
	•	deterministic ordering
	•	contains one or more EvidenceItem per requirement depending on selection policy
	•	includes provenance tier info per item
	•	includes retrieval receipt fields sufficient for replay attempts

⸻

5) Determinism Rules

5.1 Evidence ID generation (must be stable)

Define evidence_id as:
	•	evidence_id = "ev_" + sha256( canonical( requirement_id + "|" + source_target.uri + "|" + source_target.method + "|" + canonical(request_payload) + "|" + content_fingerprint ) )[:16]

However, content_fingerprint depends on response. For strict determinism across runs, you need a stable scheme:

Recommended split:
	•	request_id (deterministic, pre-fetch) computed from requirement + target request fields
	•	evidence_id computed from request_id + response_fingerprint after fetch

Store request_id inside RetrievalReceipt.request_fingerprint.

Rules
	•	request_fingerprint: deterministic from SourceTarget fields (uri/method/params/body/headers subset, and operation/search_query when present)
	•	response_fingerprint: deterministic hash of raw response bytes (or canonical JSON if JSON)
	•	evidence_id: deterministic from (requirement_id, source_id, uri, operation or "fetch", search_query or ""), so same URI with different operation or search_query yields different evidence_id
	•	evidence_id: deterministic from (requirement_id, request_fingerprint, response_fingerprint)

5.2 Ordering

EvidenceBundle.items MUST be ordered deterministically:
	1.	by DataRequirement order in PromptSpec.data_requirements
	2.	within a requirement:
	•	preserve source_targets order for fallback_chain
	•	for multi_source_quorum, order by source_targets order (not by completion time)

5.3 Timestamps
	•	RetrievalReceipt.retrieved_at is allowed to be “now” (not commitment-stable), because evidence commitments will hash the whole EvidenceItem.
But you already commit EvidenceItems in Module 03, so this matters.

Therefore to preserve stable commitments across replays:
	•	retrieved_at MUST be optional and default None (exclude from canonical) OR
	•	you must store timestamps outside committed EvidenceItem.

Recommended consistent approach (matching PromptSpec rule):
	•	Make RetrievalReceipt.retrieved_at: Optional[datetime] = None and omit by default.
	•	Put operational timestamps in EvidenceBundle.provenance_summary["collected_at"] or non-committed logs.

If your current schema has required retrieved_at, update schema to optional to keep PoR stable across replays.

⸻

6) Provenance Tier Model (Policy)

Collector must attach:
	•	ProvenanceProof.tier (int)
	•	ProvenanceProof.kind in {"zktls", "signature", "notary", "hashlog", "none"}

Default tier mapping (suggested)
	•	tier 0: "none" (plain http)
	•	tier 1: "signature" (server-signed payload or your own signing)
	•	tier 2: "zktls" (zkTLS proof)
	•	tier 3+: future

Collector may not actually generate zkTLS yet; can store placeholder proof_blob and fail policy if tier required > achieved.

⸻

7) Policy Enforcement Rules

7.1 Allowed sources

For each SourceTarget.source_id, Collector must verify it exists in:
	•	prompt_spec.market.allowed_sources[*].source_id with allow=True
If not allowed: skip and record check failure.

7.2 Minimum provenance tier

For each requirement:
	•	required_tier = max(prompt_spec.market.min_provenance_tier, requirement.min_provenance_tier)
Collector must ensure final selected evidence meets required_tier, else:
	•	produce evidence items but mark provenance tier insufficient (and allow downstream INVALID)
	•	OR return VerificationResult with ok=False (preferred if strict)

Recommended behavior:
	•	still output bundle, but include provenance_summary["policy_failures"] and set per-item flags in EvidenceItem.normalized or EvidenceBundle.provenance_summary.

(Strict pipelines often prefer hard fail; choose one and document in code.)

7.3 Selection Policy handling
	•	single_best:
	•	try first source_target only (or pick by allowed_sources priority)
	•	fallback_chain:
	•	attempt targets in order until one succeeds AND meets tier
	•	multi_source_quorum:
	•	attempt up to max_sources
	•	require at least quorum successes meeting tier
	•	if quorum not met → requirement unresolved

⸻

8) BaseSource Interface (agents/collector/data_sources/base_source.py)

Implement an adapter protocol:

class BaseSource(Protocol):
    source_id: str

    def fetch(self, target: SourceTarget, *, timeout_s: int = 20) -> "FetchedArtifact":
        ...

Define FetchedArtifact (dataclass) with:
	•	raw_bytes: bytes
	•	content_type: str (json/text/html/bytes)
	•	parsed: Any (dict/list/str or None)
	•	response_headers: dict
	•	status_code: int | None
	•	final_url: str | None
	•	error: str | None

⸻

9) Concrete Sources

9.1 http_source.py
	•	supports GET/POST
	•	uses requests or httpx
	•	respects target.headers, target.params, target.body
	•	returns FetchedArtifact

9.2 web_source.py

For now:
	•	stub implementation that fetches via http (or raises NotImplemented with clear message)
Later can integrate playwright.

9.3 polymarket_source.py
	•	if SourceTarget.uri provided, treat it as explicit API URL
	•	parse JSON response
	•	avoid hardcoding endpoints inside adapter (Prompt Engineer should provide explicit uri)

⸻

10) Evidence Construction (core behavior in collector_agent.py)

Steps per requirement
	1.	Validate source_targets not empty
	2.	For each target in selection order:
	•	choose adapter by source_id
	•	fetch
	•	compute fingerprints:
	•	request_fingerprint = sha256(canonical(request_fields))
	•	response_fingerprint = sha256(raw_bytes)
	•	create EvidenceItem:
	•	source: includes uri, provider, source_id
	•	retrieval: includes method, request_fingerprint, response_fingerprint (timestamps optional)
	•	provenance: fill tier/kind (tier policy determines)
	•	content: parsed if JSON else text
	•	normalized: optional normalized fields (e.g., extracted numeric value)
	3.	Apply provenance tier verification (zktls/signature stubs)
	4.	SelectionPolicy decides which EvidenceItems are included in bundle for that requirement
	5.	Record summary

Output bundle fields
	•	bundle_id: deterministic from market_id + plan_id (hash)
	•	collector_id: fixed string (e.g., "collector_v1")
	•	collection_time: OPTIONAL and default None (exclude from canonical if you want stable commitments)
	•	items: ordered as per determinism rules
	•	provenance_summary: include:
	•	policy decisions
	•	failures
	•	adapters used
	•	optional operational timestamps (non-committed log)

⸻

11) Verification Modules

11.1 verification/tier_policy.py

Implement:
	•	TierPolicy.required_tier(prompt_spec, requirement) -> int
	•	TierPolicy.classify_provenance(target, fetched_artifact) -> ProvenanceProof
	•	TierPolicy.enforce(bundle, prompt_spec) -> VerificationResult

11.2 verification/signature_verifier.py

For now:
	•	stub verifier that checks presence of expected signature fields if ProvenanceProof.kind=="signature"
	•	returns VerificationResult

11.3 verification/zktls_verifier.py

For now:
	•	stub verifier that checks proof_blob exists and has expected shape if kind=="zktls"
	•	return VerificationResult

Real cryptographic verification can come later; structure should be in place.

⸻

12) Tests Required

Add:
	•	tests/unit/test_collector_determinism.py
	•	tests/unit/test_tier_policy.py

12.1 Determinism tests
	•	identical PromptSpec+ToolPlan with mocked fetch responses → identical EvidenceItem canonical dumps (assuming timestamps omitted)
	•	stable ordering: requirement order then source order
	•	evidence_id stable given same responses

12.2 Policy tests
	•	min tier unmet → enforcement fails or marks unresolved (depending on your policy choice)
	•	quorum policy requires quorum successes

Mock adapters rather than making network calls.

⸻

13) Acceptance Checklist
	•	Collector runs purely with explicit SourceTargets (no hidden endpoint discovery)
	•	Evidence IDs + ordering deterministic under mocked responses
	•	TierPolicy enforces min_provenance_tier and selection policy
	•	Signature/ZkTLS verifiers exist (stubs ok) and integrate into enforcement flow
	•	EvidenceBundle produced conforms to schemas and supports PoR commitment stability (timestamps optional)
	•	All files ≤ 500 LOC each; unit tests pass
