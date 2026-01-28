Module 06 — Auditor (Reasoning Trace & Sanity Verification)

Module ID: M06
Owner Role: AI/Reasoning Engineer + Verification Engineer
Goal: Given PromptSpec and EvidenceBundle, produce a deterministic, checkable ReasoningTrace that:
	•	extracts relevant claims from evidence,
	•	maps evidence to the market event_definition,
	•	runs contradiction and logic sanity checks,
	•	produces a trace whose steps reference evidence IDs only (no ungrounded facts),
	•	passes trace policy verification or returns structured failures.

Auditor does not finalize verdict; it produces the "audit trail" for Judge + Sentinel.

⸻

1) Owned Files

Under agents/auditor/ implement:

Agent
	•	auditor_agent.py
	•	__init__.py

Reasoning helpers
	•	reasoning/claim_extraction.py
	•	reasoning/contradiction_checks.py
	•	reasoning/dspy_programs.py (optional; can be thin wrapper / stub)
	•	reasoning/__init__.py

Verification
	•	verification/trace_policy.py
	•	verification/logic_sanity.py
	•	verification/__init__.py

⸻

2) Dependencies

Schemas / protocol
	•	core.schemas.prompts.PromptSpec, DataRequirement
	•	core.schemas.evidence.EvidenceBundle, EvidenceItem
	•	core.por.reasoning_trace.ReasoningTrace, ReasoningStep, TracePolicy
	•	core.schemas.verification.VerificationResult, CheckResult, ChallengeRef
	•	core.schemas.errors.CournotError
	•	core.schemas.canonical.dumps_canonical
	•	core.crypto.hashing.hash_canonical, to_hex

Optional LLM usage
	•	core.llm.llm.LLMClient
	•	core.llm.determinism (temperature=0, schema-locked if possible)

⸻

3) Scope

In scope
	•	Deterministic extraction of "claims" from evidence
	•	Building ReasoningTrace steps
	•	Ensuring all steps cite evidence_ids
	•	Contradiction detection across claims
	•	Logic sanity checks against event_definition and resolution_window
	•	Trace policy enforcement (structure, step ordering, evidence references)

Out of scope
	•	Data retrieval (Collector)
	•	Final outcome mapping (Judge)
	•	Merkle/PoR bundle assembly (Module 03)
	•	Cryptographic provenance verification (Collector verification)

⸻

4) Auditor Input/Output Contract

Input

AuditorAgent.audit(prompt_spec: PromptSpec, evidence: EvidenceBundle) -> ReasoningTrace

Output

A ReasoningTrace where:
	•	steps are ordered deterministically
	•	each step references evidence via evidence_ids
	•	no step introduces facts not backed by evidence
	•	includes explicit checks and extracted intermediate values
	•	(optional) includes step-level hash_commitment or step_hash (not required)

Auditor may also return or attach a VerificationResult—recommended as a second return value:
	•	audit(...) -> tuple[ReasoningTrace, VerificationResult]
But if you prefer single return, store summary checks in ReasoningTrace.policy.extra.

⸻

5) Determinism Rules (Hard)

5.1 No random ordering
	•	evidence processing order must follow EvidenceBundle.items order
	•	claim ordering within an evidence item must be stable (e.g., lexical ordering by extracted key, or stable extraction procedure)

5.2 LLM determinism (if used)
	•	temperature=0
	•	fixed seed if your LLM client supports it
	•	schema-locked JSON outputs
	•	decoding policy recorded in TracePolicy.decoding_policy

5.3 Step IDs
	•	MUST be deterministic: step_0001, step_0002, …
	•	Steps count must be bounded by TracePolicy.max_steps

⸻

6) Trace Semantics: What Auditor Must Produce

Auditor must produce a trace that is useful for later deterministic judging:

Required trace content

At minimum, trace must include:
	1.	Extract: extract structured signals from evidence relevant to event_definition
	2.	Check: check evidence meets requirements (presence, tier markers if available)
	3.	Aggregate: reconcile multiple sources per selection policy (quorum/fallback)
	4.	Deduce: produce a derived boolean/value needed for Judge mapping (without deciding YES/NO)
	5.	Map: map derived result into "evaluation variables" that Judge will later use

Auditor should not decide verdict; it should output a derived "evaluation state".

Recommended derived output

In final step output, include:
	•	evaluation_variables: dict containing:
	•	event_observed: bool | None  (None if insufficient/invalid)
	•	numeric_value?: float
	•	timestamp?: str
	•	source_summary: list of evidence_ids used
	•	conflict_detected: bool
	•	insufficient_evidence: bool

⸻

7) Implementation Requirements

7.1 auditor_agent.py

Implement:

class AuditorAgent(BaseAgent):
    role = "auditor"

    def __init__(self, llm: Optional[LLMClient] = None, *, name: Optional[str] = None):
        ...

    def audit(self, prompt_spec: PromptSpec, evidence: EvidenceBundle, *, ctx: Optional[AgentContext] = None) -> tuple[ReasoningTrace, VerificationResult]:
        ...

Audit steps (high-level algorithm)
	1.	Create TracePolicy (strict, bounded)
	2.	Validate evidence bundle non-empty
	3.	Call ClaimExtractor.extract(prompt_spec, evidence) → ClaimSet
	4.	Run LogicSanityChecker.check(prompt_spec, evidence, claims) → checks
	5.	Run ContradictionChecker.check(claims) → checks
	6.	Build deterministic ReasoningTrace.steps:
	•	step: search/extract per evidence item
	•	step: aggregate per requirement
	•	step: consistency checks
	•	step: produce evaluation_variables
	7.	Run TracePolicyVerifier.verify(trace, prompt_spec, evidence) (structure + evidence refs)
	8.	Return (trace, verification_result)

⸻

8) Claim Extraction (reasoning/claim_extraction.py)

Purpose

Extract structured claims from evidence items that can be reasoned over deterministically.

Data structures (internal)

Define simple internal structures (dataclasses are fine):
	•	Claim
	•	claim_id: str (deterministic)
	•	kind: str (e.g., "numeric", "boolean", "text_assertion")
	•	path: str (json path or extraction rule)
	•	value: Any
	•	evidence_id: str
	•	confidence: float
	•	ClaimSet
	•	claims: list[Claim]
	•	helpers: by_evidence_id, by_kind

Extraction strategy (must be deterministic)
	•	If EvidenceItem.content_type=="json":
	•	Extract using explicit keys declared by PromptSpec (recommended via DataRequirement.expected_fields in extra)
	•	If not available, extract minimal common fields: result, value, price, timestamp when present
	•	If text/html:
	•	Use simple regex patterns declared in PromptSpec.extra or requirement.extra
	•	Avoid LLM if possible for determinism; if LLM used, must be schema-locked JSON.

Deterministic claim_id
	•	claim_id = "cl_" + sha256(evidence_id + "|" + path + "|" + canonical(value))[:12]

⸻

9) Contradiction Checks (reasoning/contradiction_checks.py)

Purpose

Detect conflicting claims across sources and flag them for Judge/Invalid handling.

Implement:
	•	ContradictionChecker.check(claims: ClaimSet) -> list[CheckResult]

Rules (minimal):
	•	For numeric claims of same "semantic key" (e.g., same metric and timestamp):
	•	if values differ beyond tolerance → conflict
	•	For boolean claims of same key:
	•	if both True and False exist → conflict
	•	If conflict exists:
	•	emit CheckResult(severity="warn" or "error") with details:
	•	involved claim_ids
	•	evidence_ids
	•	values

You can keep semantic-key grouping simple:
	•	key = (kind, path) initially

⸻

10) Logic Sanity (verification/logic_sanity.py)

Purpose

Validate the reasoning aligns with event_definition and time window constraints.

Implement:
	•	LogicSanityChecker.check(prompt_spec, evidence, claims) -> list[CheckResult]

Minimum checks:
	1.	Time window sanity

	•	If evidence includes timestamps, ensure they fall within MarketSpec.resolution_window or acceptable slack

	2.	Requirement coverage

	•	Ensure each DataRequirement has at least one EvidenceItem in the bundle (or mark insufficient)

	3.	Event definition variables

	•	Ensure you can compute required variables for event_definition (e.g., numeric threshold exists)

	4.	Confidence sanity

	•	Evidence confidence in [0,1]; claim confidence in [0,1]

⸻

11) Trace Policy Verification (verification/trace_policy.py)

Purpose

Hard validation of trace structure and grounding.

Implement:
	•	TracePolicyVerifier.verify(trace: ReasoningTrace, prompt_spec: PromptSpec, evidence: EvidenceBundle) -> VerificationResult

Rules:
	•	step_id unique, monotonically increasing
	•	each step’s prior_step_ids refer to earlier steps only
	•	every evidence_id referenced exists in EvidenceBundle
	•	number of steps <= policy.max_steps
	•	disallow "external facts" fields in step inputs when policy forbids (optional)
	•	require at least one final step of type "map" or "aggregate" that outputs evaluation_variables

When failure occurs:
	•	VerificationResult.ok=False
	•	include ChallengeRef(kind="reasoning_leaf", step_id=...) where possible

⸻

12) Optional DSPy Programs (reasoning/dspy_programs.py)

If you don’t use DSPy now:
	•	Provide stub scaffolding with clear docstrings
	•	Keep it under 100 LOC
	•	Ensure AuditorAgent can operate without it

⸻

13) Tests Required

Add unit tests:

tests/unit/test_claim_extraction.py
	•	deterministic claim extraction from JSON evidence
	•	stable claim ordering
	•	claim_id stable

tests/unit/test_contradictions.py
	•	numeric conflict detection triggers expected CheckResult
	•	boolean conflict detection triggers expected CheckResult

tests/unit/test_trace_policy.py
	•	trace referencing unknown evidence_id fails
	•	non-monotonic prior_step_ids fails
	•	missing final evaluation_variables fails

Mock PromptSpec/EvidenceBundle objects; no network; no LLM required.

⸻

14) Acceptance Checklist ("Done When")
	•	AuditorAgent produces ReasoningTrace + VerificationResult deterministically
	•	All steps are grounded via evidence_ids; no unreferenced facts
	•	Contradiction checks and logic sanity checks produce structured CheckResults
	•	TracePolicyVerifier enforces structure and grounding
	•	Unit tests pass; each file ≤ 500 LOC
