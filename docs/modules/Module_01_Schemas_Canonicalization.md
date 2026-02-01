Module 01 — Schemas & Canonicalization

Module ID: M01
Owner Role: Schema Engineer (Pydantic / API-contract specialist)
Primary Goal: Define the typed contracts that every other module uses and ensure deterministic canonical serialization for hashing and Merkle commitments.

This module is the single source of truth for cross-module data structures:
	•	Prompt Engineer outputs
	•	Collector evidence bundle
	•	Auditor reasoning trace references (schemas used by trace)
	•	Judge deterministic verdict
	•	Verification results and errors across pipeline

Everything downstream depends on these models being stable, strict, and versioned.

⸻

1) Owned Files (Implement These Only)

Under core/schemas/:
	1.	versioning.py
	2.	errors.py
	3.	canonical.py
	4.	market.py
	5.	prompts.py
	6.	transport.py
	7.	evidence.py
	8.	verdict.py
	9.	verification.py
	10.	__init__.py

Test owned by this module:
	•	tests/unit/test_canonical_json.py

⸻

2) Dependencies & Constraints

2.1 Dependencies
	•	Python 3.10+
	•	pydantic (recommended v2)
	•	Standard libs: datetime, json, hashlib not required here but OK

2.2 Hard Constraints
	•	Canonical JSON must be deterministic across runs
	•	All schemas must include schema_version where applicable (or be wrapped by a versioning strategy)
	•	Models must be strict: reject unexpected fields by default
	•	All datetimes must serialize deterministically (UTC + Z)
	•	Each file must stay under 500 LOC

⸻

3) "Public Contracts" (What Other Modules Rely On)

These models/functions must exist and remain stable:

Canonical serialization API
	•	canonical.to_canonical_json_dict(obj) -> dict
	•	canonical.dumps_canonical(obj) -> str
	•	canonical.normalize_datetime(dt) -> datetime
	•	canonical.ensure_utc(dt) -> datetime

Core Schemas
	•	MarketSpec, ResolutionWindow, ResolutionRule, ResolutionRules, SourcePolicy, DisputePolicy
	•	PromptSpec, PredictionSemantics, DataRequirement
	•	ToolPlan, ToolCallRecord
	•	EvidenceBundle, EvidenceItem, SourceDescriptor, RetrievalReceipt, ProvenanceProof
	•	DeterministicVerdict (Outcome enum)
	•	VerificationResult, CheckResult, ChallengeRef (in verification.py)
	•	CournotError, plus specific errors (in errors.py)

3.1) SourceTarget and SelectionPolicy (Strict Source Targeting)
The Prompt Engineer MUST express executable data fetching instructions explicitly. To support this, the schema layer defines SourceTarget and SelectionPolicy, and extends DataRequirement accordingly.

New schemas
	•	SourceTarget
    •	source_id: str — logical source adapter id (e.g. "http", "web", "polymarket", "exchange").
    •	uri: str — explicit URL/endpoint. MUST be a fully specified absolute URI when applicable.
    •	method: Literal["GET","POST","RPC","WS","OTHER"]
    •	expected_content_type: Literal["json","text","html","bytes"]
    •	headers: dict = {}
    •	params: dict = {}
    •	body: dict | str | None = None
    •	auth_ref: str | None = None — reference only; secrets never stored in schemas.
    •	cache_ttl_seconds: int | None = None
    •	operation: Literal["fetch","search"] | None = None — how to obtain data: None or "fetch" = direct request to uri; "search" = execute search (e.g. via search_query or site/uri) then fetch. Excluded from canonical when None for backward-compatible prompt_spec_hash.
    •	search_query: str | None = None — when operation is "search", the search query string (e.g. site:example.com "exact phrase"). Optional; query may also be in params/uri.
    •	notes: str | None = None
  •	SelectionPolicy
    •	strategy: Literal["single_best","multi_source_quorum","fallback_chain"]
    •	min_sources: int = 1
    •	max_sources: int = 3
    •	quorum: int = 1 — used when strategy="multi_source_quorum".
    •	tie_breaker: Literal["highest_provenance","source_priority","most_recent"]

DataRequirement extension
	•	DataRequirement.source_targets: list[SourceTarget] is REQUIRED and MUST contain at least 1 entry.
	•	DataRequirement.selection_policy: SelectionPolicy is REQUIRED.

Validation requirements
	•	source_targets MUST preserve order. The order is semantically meaningful (priority/fallback order) and MUST NOT be sorted.
	•	Each SourceTarget.uri MUST be treated as an opaque string for commitments; no normalization (e.g., stripping trailing slashes, reordering query params) is permitted within the schema layer.
	•	operation and search_query participate in canonical serialization when present; when None they are excluded (exclude_none), so existing prompt_specs without these fields retain the same prompt_spec_hash.

⸻

4) Canonicalization Rules (Non-Negotiable)

Canonical JSON is used for hashing, Merkle leaves, and deterministic verification.

4.1 JSON output rules

dumps_canonical(obj) MUST:
	•	use json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)
	•	exclude None fields (unless explicitly needed)
	•	output datetimes as ISO-8601 with Z suffix, e.g. "2026-01-27T21:35:00Z"
	•	represent enums as their string values
	•	avoid non-deterministic fields (random IDs) unless they are part of contract
	•	reject NaN/Infinity floats (raise)

4.2 Datetime normalization

All datetimes in models should be interpreted as:
	•	If naive (no tzinfo): treat as UTC by default (or explicitly document)
	•	Internally store as timezone-aware UTC
	•	Serialize as UTC Z

4.3 Field strictness

All Pydantic models should use:
	•	model_config = ConfigDict(extra="forbid", frozen=False) (or frozen selectively)
	•	validate_assignment=True if helpful (optional)

4.4 Canonicalization of Source Targets
  •	Canonical JSON serialization MUST preserve source_targets ordering exactly as provided. The schema/canonical layer MUST NOT transform URIs (including query strings). Canonicalization treats uri as a verbatim string.

4.5 Timestamp Determinism

Commitment-Safe Timestamps
Any field that is included in objects that will be committed (hashed/Merkle-leafed) MUST be deterministic for identical inputs. To avoid non-deterministic PoR commitments, the following rule applies:
	•	PromptSpec.created_at MUST be OPTIONAL and defaults to None.
	•	When created_at is None, canonical serialization MUST exclude it (via exclude_none=True), ensuring identical user inputs produce identical prompt_spec_hash.
	•	If a deployment requires timestamps, they MUST be stored outside the committed object (e.g., in bundle metadata) OR must be explicitly provided as deterministic input and documented.

Default policy: PromptSpec.created_at = None to keep hashes stable across identical compilations.
⸻

5) File-by-File Implementation Specs

5.1 versioning.py

Purpose: Centralize protocol/schema version constants.

Must include:
	•	SCHEMA_VERSION = "v1"
	•	PROTOCOL_VERSION = "v1"
	•	type alias: SchemaVersion = Literal["v1"] (future-proof)
	•	helper: assert_supported_schema_version(v: str) -> None

Notes:
	•	Keep tiny (<80 LOC)
	•	No imports from other schema files to avoid cycles

⸻

5.2 errors.py

Purpose: Standard error taxonomy across pipeline.

Define base error model(s):
	•	class CournotError(BaseModel):
	•	code: str (stable machine-readable string)
	•	message: str (human message)
	•	details: dict = {}
	•	retryable: bool = False

Recommended error codes (examples):
	•	SCHEMA_VALIDATION_ERROR
	•	CANONICALIZATION_ERROR
	•	EVIDENCE_POLICY_VIOLATION
	•	PROVENANCE_VERIFICATION_FAILED
	•	TRACE_POLICY_VIOLATION
	•	DETERMINISM_VIOLATION
	•	TIME_WINDOW_VIOLATION

Also add lightweight exceptions (optional, but useful):
	•	class CournotException(Exception): ...
	•	class CanonicalizationException(CournotException): ...

⸻

5.3 canonical.py

Purpose: Deterministic serialization utilities.

Must implement:
	•	ensure_utc(dt: datetime) -> datetime
	•	normalize_datetime(dt: datetime) -> datetime (calls ensure_utc)
	•	to_canonical_json_dict(obj: Any) -> dict
	•	Supports Pydantic models and plain dicts/lists/primitives
	•	dumps_canonical(obj: Any) -> str
	•	canonicalize_value(v: Any) -> Any
	•	Recursively handles datetime, enum, BaseModel, dict/list
	•	CANONICAL_JSON_SEPARATORS = (",", ":")

Behaviors:
	•	If object is Pydantic model: use model_dump(mode="json", by_alias=True, exclude_none=True)
	•	Enforce deterministic datetime formatting with Z
	•	Reject NaN/Infinity (use math.isfinite for floats)

⸻

5.4 market.py

Purpose: Market specification and resolution rules.

Models:
	•	ResolutionWindow { start: datetime, end: datetime }
	•	SourcePolicy { source_id, kind, allow, min_provenance_tier, notes? }
	•	ResolutionRule { rule_id, description, priority:int=0 }
	•	ResolutionRules { rules: list[ResolutionRule] }
	•	DisputePolicy { dispute_window_seconds:int, allow_challenges:bool, extra:dict }
	•	MarketSpec:
	•	schema_version: str = SCHEMA_VERSION
	•	market_id: str
	•	question: str
	•	event_definition: str (MUST be unambiguous)
	•	timezone: str = "UTC"
	•	resolution_deadline: datetime
	•	resolution_window: ResolutionWindow
	•	resolution_rules: ResolutionRules
	•	allowed_sources: list[SourcePolicy]
	•	min_provenance_tier: int = 0
	•	dispute_policy: DisputePolicy
	•	metadata: dict

Validation rules (recommended):
	•	resolution_window.start <= resolution_window.end
	•	resolution_deadline within window end (or equals)
	•	min_provenance_tier >= 0

⸻

5.5 prompts.py

Purpose: Prompt Engineer output contracts.

Models:
	•	PredictionSemantics:
	•	target_entity: str
	•	predicate: str
	•	threshold?: str
	•	timeframe?: str
	•	DataRequirement:
	•	requirement_id: str
	•	description: str
	•	preferred_sources: list[str] = []
	•	min_provenance_tier: int = 0
	•	PromptSpec:
	•	schema_version = SCHEMA_VERSION
	•	task_type = "prediction_resolution"
	•	market: MarketSpec
	•	prediction_semantics: PredictionSemantics
	•	data_requirements: list[DataRequirement]
	•	output_schema_ref: str = "core.schemas.verdict.DeterministicVerdict"
	•	forbidden_behaviors: list[str]
	•	created_at: datetime
	•	tool_plan: Optional[ToolPlan] = None (import from transport carefully)
	•	extra: dict

Import note: Avoid circular imports.
	•	If ToolPlan causes cycles, type it as "ToolPlan" (string forward ref) and import inside TYPE_CHECKING.

⸻

5.6 transport.py

Purpose: Tool call logs and tool planning.

Models:
	•	ToolCallRecord:
	•	tool: str
	•	input: dict
	•	output?: dict
	•	started_at?: str (ISO string for simplicity)
	•	ended_at?: str
	•	error?: str
	•	ToolPlan:
	•	plan_id: str
	•	requirements: list[str]
	•	sources: list[str]
	•	min_provenance_tier: int = 0
	•	allow_fallbacks: bool = True
	•	extra: dict

Keep this minimal; tool execution specifics live elsewhere.

⸻

5.7 evidence.py

Purpose: Evidence bundle + provenance.

Models:
	•	SourceDescriptor:
	•	source_id: str
	•	uri?: str
	•	provider?: str
	•	notes?: str
	•	RetrievalReceipt:
	•	retrieved_at: datetime
	•	method: Literal["http_get","api_call","chain_rpc","manual_import","other"]
	•	tool?: str
	•	request_fingerprint?: str
	•	response_fingerprint?: str
	•	extra: dict
	•	ProvenanceProof:
	•	tier: int
	•	kind: Literal["zktls","signature","notary","hashlog","none"]
	•	proof_blob?: str
	•	verifier?: str
	•	extra: dict
	•	EvidenceItem:
	•	schema_version = SCHEMA_VERSION
	•	evidence_id: str
	•	source: SourceDescriptor
	•	retrieval: RetrievalReceipt
	•	provenance: ProvenanceProof
	•	content_type: Literal["text","json","html","bytes"]
	•	content: Any
	•	normalized?: Any
	•	confidence: float = 1.0 (0..1)
	•	EvidenceBundle:
	•	schema_version = SCHEMA_VERSION
	•	bundle_id: str
	•	collector_id: str
	•	collection_time: datetime
	•	items: list[EvidenceItem]
	•	provenance_summary: dict

Validation rules:
	•	Provenance tier >= 0
	•	Confidence in [0, 1]
	•	EvidenceBundle.items unique evidence_id (warn/validate)

⸻

5.8 verdict.py

Purpose: Deterministic settlement output.

Types/Models:
	•	Outcome = Literal["YES","NO","INVALID"]
	•	DeterministicVerdict:
	•	schema_version = SCHEMA_VERSION
	•	market_id: str
	•	outcome: Outcome
	•	confidence: float (0..1)
	•	resolution_time: datetime
	•	resolution_rule_id: str
	•	prompt_spec_hash?: str
	•	evidence_root?: str
	•	reasoning_root?: str
	•	justification_hash?: str
	•	selected_leaf_refs: list[str] = []
	•	metadata: dict = {}

Rules:
	•	confidence bounds enforced
	•	resolution_rule_id must not be empty

⸻

5.9 verification.py

Purpose: Standard result format for verification steps (Collector/Auditor/Judge/Sentinel).

Models:
	•	CheckResult:
	•	check_id: str
	•	ok: bool
	•	severity: Literal["info","warn","error"]
	•	message: str
	•	details: dict
	•	ChallengeRef:
	•	kind: Literal["evidence_leaf","reasoning_leaf","verdict_hash","por_bundle"]
	•	leaf_index?: int
	•	evidence_id?: str
	•	step_id?: str
	•	reason?: str
	•	VerificationResult:
	•	ok: bool
	•	checks: list[CheckResult]
	•	challenge?: ChallengeRef
	•	error?: CournotError

This schema is how modules communicate "what failed" without exceptions.

⸻

5.10 __init__.py

Export the public API.

Must export (minimum):
	•	canonical functions: dumps_canonical, to_canonical_json_dict
	•	main schemas: MarketSpec, PromptSpec, EvidenceBundle, DeterministicVerdict, VerificationResult, CournotError
  •	core.schemas MUST export SourceTarget and SelectionPolicy (from their chosen module) and the extended DataRequirement type.

Keep it clean to avoid star-import ambiguity.

⸻

6) Testing Requirements

Implement at least these tests in tests/unit/test_canonical_json.py:

Test A: Deterministic ordering
	•	Create a model with dict fields inserted in random order
	•	Ensure dumps_canonical() output is identical across runs

Test B: Datetime normalization
	•	Provide naive datetime and aware datetime
	•	Ensure both serialize to the same UTC Z format

Test C: Exclude None
	•	Ensure optional None fields do not appear

Test D: Float safety
	•	NaN/Infinity triggers exception

Test E: Round trip stability (optional)
	•	json.loads(dumps_canonical(model)) returns dict with expected keys

⸻

7) Implementation Notes (To Avoid Traps)
	1.	Avoid circular imports
	•	Use forward references + TYPE_CHECKING in prompts.py when referring to ToolPlan.
	2.	Do not compute hashes here
	•	Hashing belongs to core/crypto/hashing.py, which will depend on dumps_canonical.
	3.	Be strict early
	•	Set extra="forbid" so schema drift is caught.
	4.	Keep models small
	•	Don’t add business logic into schema files.

⸻

8) "Done When" Checklist

This module is complete when:
	•	All schema files implemented with strict Pydantic models
	•	dumps_canonical() meets determinism rules
	•	Unit tests pass for canonical serialization requirements
	•	No circular import errors when importing core.schemas from other modules
	•	Every primary model contains schema_version = "v1" (or uses versioning strategy)

⸻

If you want, I can generate Module Doc #02 (Merkle & Commitments) next, or I can generate the actual code stubs for these schema files following the constraints (≤ 500 LOC per file) so your AI can start filling in logic immediately.