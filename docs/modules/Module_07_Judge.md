Module 07 — Judge (Deterministic Verdict: YES/NO/INVALID + Confidence)

Module ID: M07
Owner Role: Protocol/Judging Engineer (deterministic evaluation, confidence, rule execution)
Goal: Produce a DeterministicVerdict from (PromptSpec, EvidenceBundle, ReasoningTrace) by executing strictly-defined resolution rules and confidence policy. The Judge must be reproducible (same inputs → same verdict) and must generate verification artifacts for later disputes.

Judge is responsible for:
	1.	interpreting MarketSpec.event_definition and resolution_rules
	2.	consuming Auditor’s evaluation_variables (and/or claims) deterministically
	3.	mapping to Outcome ∈ {YES, NO, INVALID}
	4.	computing confidence ∈ [0,1] from defined policy
	5.	emitting a DeterministicVerdict including references to commitments (hashes/roots) when available

⸻

1) Owned Files

Under agents/judge/ implement:

Agent
	•	judge_agent.py
	•	__init__.py

Verification helpers
	•	verification/deterministic_mapping.py
	•	verification/schema_lock.py
	•	verification/__init__.py

⸻

2) Dependencies

Schemas / protocol
	•	core.schemas.prompts.PromptSpec
	•	core.schemas.market.MarketSpec
	•	core.schemas.evidence.EvidenceBundle
	•	core.schemas.verdict.DeterministicVerdict, Outcome
	•	core.schemas.verification.VerificationResult, CheckResult, ChallengeRef
	•	core.schemas.errors.CournotError
	•	core.por.reasoning_trace.ReasoningTrace, ReasoningStep
	•	core.crypto.hashing.hash_canonical, to_hex
	•	core.merkle.build_merkle_root (optional if Judge computes evidence_root/reasoning_root itself; otherwise orchestrator does)

Optional
	•	core.por.proof_of_reasoning.compute_roots (if already implemented; recommended)

⸻

3) Inputs / Outputs

Input

JudgeAgent.judge(prompt_spec: PromptSpec, evidence: EvidenceBundle, trace: ReasoningTrace, *, ctx=None) -> tuple[DeterministicVerdict, VerificationResult]

Output
	•	DeterministicVerdict must set:
	•	market_id
	•	outcome: YES/NO/INVALID
	•	confidence: float
	•	resolution_time (recommend None or deterministic; see timestamp note below)
	•	resolution_rule_id (which rule decided)
	•	references: prompt_spec_hash, evidence_root, reasoning_root if available
	•	VerificationResult describing checks and any challenge hint if invalid/failure

⸻

4) Hard Determinism Rules

4.1 No stochastic behavior

Judge logic must be pure deterministic—no LLM calls needed. If an LLM is used for parsing event_definition (not recommended), it must be schema-locked + deterministic.

4.2 Timestamp determinism

To keep PoR stable across replays, apply the same rule as PromptSpec/Evidence:
	•	DeterministicVerdict.resolution_time SHOULD be OPTIONAL and default to None (excluded from canonical), OR be passed deterministically by orchestrator.
Recommended: set resolution_time=None in strict mode and put operational time in bundle metadata.

4.3 Stable tie-breakers

When multiple evidence items exist, use deterministic ordering:
	•	EvidenceBundle.items order is authoritative
	•	SelectionPolicy tie_breaker must be respected
	•	Never depend on “first completed request time”

⸻

5) Judge Decision Model

Judge must use two sources of truth, in priority order:
	1.	trace final step output evaluation_variables produced by Auditor
	2.	fallback to direct evidence inspection if evaluation_variables missing (should generally produce INVALID)

Required structure: evaluation_variables

Auditor should produce a final "map" (or "aggregate") step with:
	•	evaluation_variables.event_observed: bool | None
	•	evaluation_variables.numeric_value?: float
	•	evaluation_variables.timestamp?: str
	•	evaluation_variables.conflict_detected: bool
	•	evaluation_variables.insufficient_evidence: bool
	•	evaluation_variables.source_summary: list[str] (evidence_ids used)

Judge should treat missing variables as failure and return INVALID with low confidence.

⸻

6) Rule Execution (ResolutionRules)

MarketSpec.resolution_rules.rules are treated as ordered rules with IDs and priorities. For now, implement a standard engine:

Standard rule set (required support)

Judge must support at least these logical rule IDs (by convention, produced by Prompt Engineer):
	1.	R_VALIDITY
	2.	R_CONFLICT
	3.	R_BINARY_DECISION
	4.	R_CONFIDENCE
	5.	R_INVALID_FALLBACK

Even if Prompt Engineer uses different IDs, Judge must still operate by parsing rules list; but supporting these canonical IDs gives the strict pipeline.

Rule evaluation order:
	•	Always run validity → conflict → binary decision → confidence
	•	If any stage fails, outcome INVALID

⸻

7) Deterministic Mapping Engine (verification/deterministic_mapping.py)

Purpose

Implement the pure function that maps inputs to verdict.

Must expose:
	•	map_to_verdict(prompt_spec, evidence, trace) -> tuple[DeterministicVerdict, list[CheckResult]]

Algorithm (strict):

Step 1: Extract evaluation_variables
	•	locate final step:
	•	last step of type "map" preferred, else last "aggregate"
	•	extract evaluation = step.output.get("evaluation_variables", {})

If missing → INVALID (check severity error).

Step 2: Validity decision
If any of these true → INVALID:
	•	evaluation.insufficient_evidence is True
	•	required variables missing for event_definition (e.g., numeric threshold needed)
	•	required provenance tier not met (if auditor recorded it; else ignore)

Emit check:
	•	check_id="validity" ok=False severity=“error”

Step 3: Conflict handling
If evaluation.conflict_detected is True:
	•	consult PromptSpec / MarketSpec policy hints:
	•	from PromptSpec.extra["confidence_policy"] and/or DataRequirement.selection_policy
	•	if conflict policy indicates quorum not met or tie unresolved → INVALID
	•	else continue but reduce confidence

Emit check:
	•	check_id="conflict" ok=True severity=“warn”

Step 4: Binary decision
Evaluate event_definition:
	•	In v1, Judge does not parse arbitrary natural language. It expects event_definition to be compiled into a machine-checkable predicate using evaluation_variables.

Required constraint for strict mode:
Prompt Engineer MUST encode event_definition such that Judge can evaluate by:
	•	event_observed boolean OR
	•	numeric compare: numeric_value vs threshold stored in PromptSpec.prediction_semantics.threshold

Decision:
	•	if event_observed is True → YES
	•	if event_observed is False → NO
	•	if event_observed is None → INVALID

Emit check:
	•	check_id="binary_decision" ok=…

Step 5: Confidence assignment
Compute confidence deterministically using policy:

Base inputs:
	•	base = PromptSpec.extra.confidence_policy.default_confidence (default 0.7 if missing)
	•	apply reductions:
	•	conflict_detected → *0.8
	•	fallback_used (if auditor indicates) → *0.9
	•	quorum_strength (if available) → * factor
	•	clamp to [0,1]

If outcome INVALID:
	•	confidence must be <= PromptSpec.extra.confidence_policy.min_confidence_for_yesno (recommended 0.55)
	•	recommend set confidence to 0.0 or 0.3 depending on failure type

Emit check:
	•	check_id="confidence" ok=True

Finally set resolution_rule_id to which stage decided (e.g., R_BINARY_DECISION or R_VALIDITY).

⸻

8) Schema Lock (verification/schema_lock.py)

Purpose

Ensure Judge only relies on schema-locked, deterministic structures.

Implement:
	•	SchemaLock.verify_prompt_spec(prompt_spec) -> list[CheckResult]
	•	SchemaLock.verify_trace(trace) -> list[CheckResult]

Checks:
	•	PromptSpec.output_schema_ref must equal core.schemas.verdict.DeterministicVerdict
	•	PromptSpec.extra["strict_mode"] is True (for strict pipeline)
	•	trace has final evaluation_variables
	•	Outcome must be one of YES/NO/INVALID (enforced by schema, but still check)

This prevents “flexible” prompt modules from breaking Judge determinism accidentally.

⸻

9) JudgeAgent (judge_agent.py)

Implement:

class JudgeAgent(BaseAgent):
    role = "judge"

    def __init__(self, *, name: str | None = None):
        ...

    def judge(self, prompt_spec: PromptSpec, evidence: EvidenceBundle, trace: ReasoningTrace, *, ctx=None) -> tuple[DeterministicVerdict, VerificationResult]:
        ...

Steps
	1.	schema lock checks
	2.	call map_to_verdict()
	3.	compute references:
	•	prompt_spec_hash (optional, use hash_canonical(prompt_spec))
	•	evidence_root/reasoning_root (optional; better if orchestrator provides)
	4.	return verdict + VerificationResult populated with checks

ChallengeRef behavior
	•	If INVALID because missing evaluation_variables: ChallengeRef(kind="reasoning_leaf", step_id=...)
	•	If INVALID due to evidence insufficiency: ChallengeRef(kind="evidence_leaf")
	•	If mismatch in schema lock: ChallengeRef(kind="por_bundle")

⸻

10) Tests Required

Add:
	•	tests/unit/test_judge_mapping.py
	•	tests/unit/test_schema_lock.py

10.1 Mapping tests
	1.	event_observed=True → YES, confidence >= min threshold
	2.	event_observed=False → NO
	3.	event_observed=None or insufficient_evidence True → INVALID with low confidence
	4.	conflict_detected True reduces confidence
	5.	missing confidence policy uses defaults deterministically

10.2 Schema lock tests
	•	wrong output_schema_ref fails
	•	missing strict_mode fails (if strict pipeline enabled)
	•	missing evaluation_variables fails

Use synthetic PromptSpec/EvidenceBundle/ReasoningTrace objects; no network/LLM.

⸻

11) Acceptance Checklist
	•	Judge produces deterministic YES/NO/INVALID + confidence from evaluation_variables
	•	Validity/conflict/decision/confidence checks are returned as structured CheckResults
	•	SchemaLock prevents non-strict prompt modules from breaking determinism
	•	Verdict contains correct market_id and resolution_rule_id
	•	Unit tests pass; each file ≤ 500 LOC
