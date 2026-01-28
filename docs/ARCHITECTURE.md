# Cournot Protocol — AI Oracle Verification Layer (Implementation Design Doc)

## Project Summary

### Goal
Build an oracle-grade, multi-agent resolution engine for long-tail prediction markets using Proof of Reasoning (PoR):
- Collector: fetches evidence + provenance proof (zkTLS or tiered proof)
- Auditor: performs transparent reasoning over evidence and produces a Merkle-ized reasoning trace
- Judge: enforces deterministic, schema-locked output suitable for on-chain settlement
Then provide a Fast Path execution (TEE Anchor) and Slow Path verification (Sentinel challengers) model.  ￼

Your current repo already contains:

-	core/merkle_tree.py, core/proof_of_reasoning.py, core/schemas.py, core/llm.py
-	role agents in agents/collector, agents/auditor, agents/judge
-	orchestrator/pipeline.py, orchestrator/prompt_engineer.py, orchestrator/market_resolver.py
-	validation/polymarket_validator.py

This doc proposes a structure that (a) matches the whitepaper SOP, (b) cleanly separates “AI role code” from “protocol verification code”, and (c) is modular enough for multiple AIs to implement components independently.

## Core Design Principles
1.	Role purity: each agent role (Prompt Engineer, Collector, Auditor, Judge, Validator) has a narrow, testable contract.
2.	Typed, deterministic IO: all cross-module messages use strict schemas (Pydantic) and stable serialization.
3.	Evidence-first: the Auditor is not allowed to reason unless evidence objects are present and validated.
4.	Merkle anchoring: every SOP step emits a leaf commitment; the pipeline commits a root that can be “pinpoint challenged”.
5.	Pluggable verification tiers: zkTLS when possible, otherwise allow tiered proofs (signed API receipts, timestamped snapshots, etc.) with explicit confidence.
6. Separation of concerns:
  - AI orchestration (prompts, LLM calls, tools) ≠ verification (Merkle, signatures, re-execution)
  - market resolution policy (rules, time windows) ≠ data collection (sources)
7.	Production readiness: observability, reproducibility, and adversarial testing are first-class.


## Proposed Repository Structure
cournot-protocol
├── ARCHITECTURE.md
├── core
│   ├── crypto
│   │   ├── __init__.py
│   │   ├── signatures.py
│   │   ├── attestation.py
│   │   └── hashing.py
│   ├── llm
│   │   ├── determinism.py
│   │   ├── __init__.py
│   │   └── llm.py
│   ├── merkle
│   │   ├── __init__.py
│   │   ├── merkle_tree.py
│   │   └── merkle_proofs.py
│   ├── schemas
│   │   ├── verification.py
│   │   ├── transport.py
│   │   ├── market.py
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── canonical.py
│   │   ├── versioning.py
│   │   ├── errors.py
│   │   ├── evidence.py
│   │   └── verdict.py
│   ├── por
│   │   ├── por_bundle.py
│   │   ├── reasoning_trace.py
│   │   ├── proof_of_reasoning.py
│   │   └── __init__.py
│   └── interfaces
│       ├── attester.py
│       ├── llm_provider.py
│       └── source.py
├── requirements.txt
├── config
│   └── prompts
│       ├── auditor_prompts.yaml
│       ├── judge_prompts.yaml
│       ├── orchestrator_prompts.yaml
│       └── collector_prompts.yaml
├── tests
│   ├── unit
│   │   └── test_canonical_json.py
│   └── README.md
├── agents
│   ├── validator
│   │   ├── __init__.py
│   │   ├── replay_executor.py
│   │   ├── sentinel_agent.py
│   │   └── challenges.py
│   ├── base_agent.py
│   ├── collector
│   │   ├── collector_agent.py
│   │   ├── __init__.py
│   │   ├── verification
│   │   │   ├── signature_verifier.py
│   │   │   ├── __init__.py
│   │   │   ├── zktls_verifier.py
│   │   │   └── tier_policy.py
│   │   └── data_sources
│   │       ├── polymarket_source.py
│   │       ├── __init__.py
│   │       ├── web_source.py
│   │       ├── http_source.py
│   │       └── base_source.py
│   ├── __init__.py
│   ├── judge
│   │   ├── judge_agent.py
│   │   ├── __init__.py
│   │   └── verification
│   │       ├── __init__.py
│   │       ├── deterministic_mapping.py
│   │       └── schema_lock.py
│   ├── prompt_engineer
│   │   ├── __init__.py
│   │   └── prompt_agent.py
│   └── auditor
│       ├── __init__.py
│       ├── reasoning
│       │   ├── dspy_programs.py
│       │   ├── contradiction_checks.py
│       │   ├── __init__.py
│       │   └── claim_extraction.py
│       ├── auditor_agent.py
│       └── verification
│           ├── trace_policy.py
│           ├── __init__.py
│           └── logic_sanity.py
├── docs
│   └── modules
│       └── Schemas_Canonicalization.md
├── README.md
└── orchestrator
    ├── market_resolver.py
    ├── prompt_engineer.py
    ├── __init__.py
    ├── sop_executor.py
    └── pipeline.py


## System Roles & Responsibilities (SOP)

This is the canonical SOP you should implement (and enforce in sop_executor.py).

4.1 Prompt Engineer (Pre-SOP Gate)

Purpose: convert user free-text into a structured PromptSpec and MarketSpec that is unambiguous and executable.

Outputs
	•	MarketSpec: event definition, timezone, deadline, resolution source policy
	•	PromptSpec: final “structured prompt” given to downstream agents
	•	ToolPlan: what evidence is required, from which sources, with which verification tier
	•	VerdictRules: how to map evidence → verdict (YES/NO/INVALID)

Failure Modes
	•	ambiguous event semantics
	•	missing time window
	•	missing objective resolution criteria
	•	non-verifiable sources requested

⸻

4.2 Collector (Stage 1: Proof of Authenticity)

Purpose: fetch evidence and attach provenance proofs.

Outputs
	•	EvidenceBundle with multiple EvidenceItem
	•	Each EvidenceItem includes:
	•	content (raw or normalized)
	•	source_descriptor (URL/API/server identity)
	•	provenance_proof (zkTLS proof / signed receipt / hash+timestamp)
	•	retrieval_receipt (time, method, tool logs)

Tiered Verification Policy
	•	Tier 0: zkTLS proof (best)
	•	Tier 1: server signature / API signed response
	•	Tier 2: notarized snapshot (hash in public log)
	•	Tier 3: plain HTTP fetch (allowed only if policy permits; low confidence)

Collector must declare confidence and tier per item; downstream logic can require minimum tiers.

⸻

4.3 Auditor (Stage 2: Proof of Logic)

Purpose: interpret evidence, extract claims, resolve contradictions, and produce a ReasoningTrace that can be Merkle-committed.

Outputs
	•	ReasoningTrace = ordered steps with:
	•	inputs referenced by evidence IDs
	•	intermediate claims
	•	checks performed (freshness, contradiction, source tier constraints)
	•	final analysis artifact (still not the on-chain verdict)
	•	LogicReport: summary + list of assumptions (should be minimized)

Constraints
	•	Must be deterministic or “replayable” under a fixed decoding policy.
	•	Must never introduce evidence not in EvidenceBundle.

⸻

4.4 Judge (Stage 3: Proof of Determinism)

Purpose: map analysis → DeterministicVerdict with schema-locking.

Outputs
	•	DeterministicVerdict:
	•	outcome ∈ {YES, NO, INVALID}
	•	confidence (bounded numeric)
	•	justification_hash (points to reasoning root or selected leaf set)
	•	market_id, resolution_time
	•	resolution_rule_id (which rule triggered)

Constraints
	•	schema locked decoding (Outlines/logit-locking style)
	•	reject ambiguous outputs
	•	ensure single canonical serialization

⸻

4.5 Sentinel / Validator (Slow Path)

Purpose: verify the PoR bundle:
	•	authenticity proofs are valid
	•	reasoning trace is consistent (no “leaf contradiction”)
	•	verdict schema compliance
	•	optionally re-execute deterministic reasoning module (opML-like replay)

Outputs
	•	VerificationResult (pass/fail + challenged leaf index)
	•	ChallengeBundle for on-chain dispute (future)

⸻

## Canonical Data Model (AI-Friendly Schemas)

Implement all of these as Pydantic models in core/schemas/*. These are the “API contracts” between modules.

5.1 MarketSpec

Fields:
	•	market_id: str
	•	question: str
	•	event_definition: str (must be fully explicit)
	•	timezone: str
	•	resolution_deadline: datetime
	•	resolution_window: {start, end}
	•	resolution_rules: ResolutionRules
	•	allowed_sources: list[SourcePolicy]
	•	min_provenance_tier: int
	•	dispute_policy: DisputePolicy

5.2 PromptSpec (Structured Prompt)

Fields:
	•	task_type: "prediction_resolution" | ...
	•	market: MarketSpec
	•	prediction_semantics: {target_entity, predicate, threshold, timeframe}
	•	data_requirements: list[DataRequirement]
	•	output_schema: str (reference to DeterministicVerdict schema)
	•	forbidden_behaviors: list[str] (no hallucinated evidence, etc.)

5.3 EvidenceBundle
	•	bundle_id
	•	items: list[EvidenceItem]
	•	collection_time
	•	collector_id
	•	provenance_summary

Each EvidenceItem:
	•	evidence_id
	•	source: SourceDescriptor
	•	retrieval: RetrievalReceipt
	•	provenance: ProvenanceProof
	•	content_type
	•	content (raw text/json)
	•	normalized (optional)

5.4 ReasoningTrace
	•	trace_id
	•	steps: list[ReasoningStep]
	•	policy: TracePolicy (determinism + constraints)
	•	evidence_refs: list[evidence_id]

ReasoningStep includes:
	•	step_id
	•	type: "search" | "extract" | "check" | "deduce" | "aggregate" | ...
	•	inputs: {evidence_ids, prior_step_ids}
	•	action
	•	output
	•	hash_commitment

5.5 PoRBundle (Protocol Artifact)
	•	market_id
	•	prompt_spec_hash
	•	evidence_root
	•	reasoning_root
	•	verdict: DeterministicVerdict
	•	tee_attestation (fast path, optional)
	•	signatures: {collector_sig, auditor_sig, judge_sig, anchor_sig}


## Merkleization & Commit Rules

6.1 Leaf Strategy

Each SOP stage emits leaves:
	•	Leaf[0..n] for Collector evidence receipts (or evidence commitments)
	•	Leaf[...] for Auditor reasoning steps
	•	Leaf[...] for Judge verdict mapping

Leaf hash should be computed from canonical JSON:
	•	stable key ordering
	•	stable numeric encoding
	•	explicit version field

6.2 Roots
	•	evidence_root = MerkleRoot(hash(EvidenceItem_i))
	•	reasoning_root = MerkleRoot(hash(ReasoningStep_j))
	•	optionally por_root = MerkleRoot([evidence_root, reasoning_root, hash(verdict)])

Pinpoint Verification
	•	allow proving any step with a Merkle branch:
	•	“Step k contradicts Evidence i”
	•	“Timestamp policy violated”
	•	“Judge produced invalid schema mapping”

## Orchestrator Pipeline (End-to-End Call Graph)

7.1 Fast Path (TEE Anchor Execution)

orchestrator/pipeline.py should implement:
	1.	PromptEngineerAgent.run(user_input) -> PromptSpec + MarketSpec + ToolPlan
	2.	CollectorAgent.collect(tool_plan) -> EvidenceBundle
	3.	CollectorVerifier.verify(evidence_bundle) -> EvidenceVerification
	4.	AuditorAgent.reason(prompt_spec, evidence_bundle) -> ReasoningTrace
	5.	AuditorVerifier.verify(trace, evidence_bundle) -> LogicVerification
	6.	JudgeAgent.judge(prompt_spec, trace) -> DeterministicVerdict
	7.	JudgeVerifier.verify(verdict) -> SchemaVerification
	8.	PoRBuilder.build(...) -> PoRBundle
	9.	AnchorAttestation.sign(por_bundle) -> attested PoRBundle
	10.	MarketResolver.finalize(market_id, verdict, por_root) -> settlement payload

7.2 Slow Path (Sentinel Verification)

agents/validator/sentinel_agent.py implements:
	•	verify_authenticity(evidence_bundle)
	•	verify_logic(reasoning_trace, evidence_bundle)
	•	verify_determinism(verdict)
	•	challenge(leaf_ref) if failed

⸻

## Module-by-Module Implementation Specs (Hand-off Friendly)

Each section below is written so a separate AI engineer can implement it independently.

⸻

8.1 core/schemas/* (Contracts Layer)

Owner AI: “Schema Engineer”
Purpose: define all Pydantic models + canonical serialization rules.

Deliverables
	•	Pydantic models listed in Section 5
	•	to_canonical_json() helper for every model
	•	versioning strategy: schema_version: "v1"

Tests
	•	round-trip serialize/deserialize
	•	canonical JSON stable ordering snapshot tests

⸻

8.2 core/por/* (PoR Bundle + Trace Rules)

Owner AI: “Protocol Engineer”
Purpose: build PoRBundle, compute roots, validate leaf structure.

Key APIs
	•	compute_leaf_hash(obj) -> bytes32
	•	build_evidence_root(bundle) -> bytes32
	•	build_reasoning_root(trace) -> bytes32
	•	build_por_bundle(...) -> PoRBundle
	•	verify_por_bundle(bundle) -> VerificationResult

Tests
	•	deterministic root computation
	•	tampering tests: modify one step → root mismatch


8.3 agents/prompt_engineer/prompt_agent.py

Owner AI: “Prompt Architect”
Purpose: convert user input to PromptSpec + ToolPlan.

Inputs
	•	raw user text
	•	optional market template (Polymarket-like, or your own)

Outputs
	•	PromptSpec
	•	ToolPlan with explicit sources + constraints

Policies
	•	enforce: timeframe, timezone, disambiguation, objective resolution rule
	•	produce a “prompt bundle” referencing config YAML prompts

Tests
	•	ambiguous input → returns INVALID with missing_fields list
	•	consistent structured prompt for repeated runs

⸻

8.4 agents/collector/collector_agent.py

Owner AI: “Data Acquisition Engineer”
Purpose: evidence fetching from multiple sources with receipts.

Pluggable Sources
	•	BaseSource.fetch(requirement) -> EvidenceItem
	•	PolymarketSource, WebSource, HttpSource

Verification
	•	attach ProvenanceProof from the best available tier
	•	ensure retrieval_receipt includes timestamps and method

Tests
	•	mocked source responses
	•	provenance tier policy enforcement

⸻

8.5 agents/collector/verification/*

Owner AI: “Evidence Verification Engineer”
Purpose: verify provenance proofs.

APIs
	•	verify(item: EvidenceItem) -> ProvenanceCheck
	•	tier_policy.enforce(bundle, min_tier)

Note
	•	zkTLS verification can be stubbed with interfaces until integrated.

⸻

8.6 agents/auditor/auditor_agent.py + reasoning/*

Owner AI: “Reasoning Engineer”
Purpose: deterministic, evidence-grounded reasoning + trace emission.

Core Steps
	1.	normalize evidence
	2.	extract claims
	3.	run contradiction detection
	4.	apply freshness/time-window checks
	5.	produce structured reasoning steps

Trace Requirements
	•	every step references evidence IDs
	•	no free-form “new facts”
	•	emit ReasoningStep objects with stable fields

Tests
	•	contradiction case: evidence says A and not-A → must mark INVALID or reduce confidence
	•	stale evidence case: timestamp out of window → reject

⸻

8.7 agents/judge/judge_agent.py + verification/*

Owner AI: “Determinism Engineer”
Purpose: schema-locked verdict mapping.

Rules
	•	output must be exactly DeterministicVerdict
	•	map to enum {YES, NO, INVALID}
	•	include rule_id + confidence bounds
	•	optionally produce selected_leaf_refs for compact justification

Tests
	•	invalid JSON -> rejected
	•	ambiguous language -> INVALID

⸻

8.8 orchestrator/sop_executor.py

Owner AI: “Workflow Engineer”
Purpose: enforce SOP order, timeouts, leaf collection, error handling.

Responsibilities
	•	stage runner: run_stage(stage_name, agent_call) -> stage_output
	•	leaf registry: collects leaves from each stage
	•	uniform error type: SOPExecutionError

Tests
	•	stage failure bubbles with structured error
	•	leaf collection yields consistent leaf ordering

⸻

8.9 orchestrator/market_resolver.py

Owner AI: “Market Engineer”
Purpose: apply settlement policy and produce final resolution payload.

Inputs
	•	MarketSpec
	•	DeterministicVerdict
	•	PoRBundle roots

Outputs
	•	ResolutionPayload for API / on-chain client
	•	for Polymarket integration: adapter style

Tests
	•	rules mapping correctness
	•	deadline enforcement

⸻

8.10 agents/validator/* (Sentinel)

Owner AI: “Adversarial Verification Engineer”
Purpose: Slow path verifier and challenge creation.

Key Features
	•	re-verify provenance
	•	re-check trace consistency
	•	deterministic replay hooks
	•	generate ChallengeBundle referencing a leaf index + Merkle branch

Tests
	•	tampered leaf detection
	•	replay mismatch detection

⸻

## Prompt & Config Layout (YAML)

Keep your config/prompts/*.yaml, but standardize them with:
	•	role
	•	purpose
	•	input_schema_ref
	•	output_schema_ref
	•	constraints
	•	few_shots (optional)
	•	determinism_policy

Example (conceptual):
	•	orchestrator_prompts.yaml: “how to call tools and enforce SOP”
	•	collector_prompts.yaml: “how to choose sources + emit receipts”
	•	auditor_prompts.yaml: “how to emit ReasoningStep objects”
	•	judge_prompts.yaml: “strict JSON verdict only”

⸻

## End-to-End Example Scenario (Prediction Market)

User input (free text):
“Will Company X announce an acquisition of Company Y before March 31, 2026?”

10.1 Prompt Engineer output (structured)
	•	Event: “Official press release OR SEC filing confirming acquisition intent”
	•	Time window: now → 2026-03-31 23:59:59 (timezone fixed)
	•	Sources: SEC EDGAR, company newsroom, trusted wire
	•	Rule: YES if primary source confirms; NO if deadline passes without confirmation; INVALID if sources unavailable.

10.2 Collector
	•	fetch EDGAR filings, press releases
	•	attach provenance tier proofs (best possible)
	•	produce EvidenceBundle

10.3 Auditor
	•	extract claims: “acquisition announced” vs “rumor”
	•	reject rumors unless policy allows Tier-3 sources
	•	produce reasoning trace and root

10.4 Judge
	•	map to YES/NO/INVALID deterministically
	•	produce final verdict payload

10.5 Validator
	•	checks evidence proofs + trace consistency
	•	if Auditor ignored an EDGAR timestamp → challenge that leaf

⸻
## Testing Strategy
	1.	Unit tests: schemas, merkle roots, leaf hashing, signature verification
	2.	Integration tests: pipeline with mocked sources and deterministic LLM
	3.	Adversarial tests:
	•	contradictory evidence
	•	stale evidence
	•	prompt injection attempts in evidence content
	•	collector returning wrong tier metadata
	•	auditor hallucinating evidence IDs

⸻

## Implementation Milestones (Suggested Order)
	1.	Finalize core/schemas + canonical serialization
	2.	Implement core/por bundle + Merkle rules
	3.	Prompt Engineer: produce PromptSpec + ToolPlan reliably
	4.	Collector: pluggable sources + receipts + tier policy
	5.	Auditor: reasoning trace emission + contradiction checks
	6.	Judge: schema lock + deterministic verdict mapping
	7.	SOP executor: leaf registry + consistent PoRBundle
	8.	Sentinel validator: verification + pinpoint challenge generation
	9.	API routes + example Polymarket adapter

⸻

## What to Hand to Different AIs (Clear Work Packages)
	•	AI #1 (Schema Engineer): implement all Pydantic models + canonical JSON + tests
	•	AI #2 (Protocol Engineer): implement PoRBundle builder/verifier + Merkle proofs + tests
	•	AI #3 (Prompt Architect): implement Prompt Engineer agent + ToolPlan compiler + YAML prompt contract
	•	AI #4 (Collector Engineer): implement data sources + EvidenceBundle creation + receipts
	•	AI #5 (Reasoning Engineer): implement deterministic reasoning trace + contradiction & freshness logic
	•	AI #6 (Determinism Engineer): implement Judge agent + schema locking + output constraints
	•	AI #7 (Workflow Engineer): implement SOP executor + orchestration + error taxonomy
	•	AI #8 (Sentinel Engineer): implement verification & challenge bundles

Each package only depends on schemas and a small set of stable interfaces.
