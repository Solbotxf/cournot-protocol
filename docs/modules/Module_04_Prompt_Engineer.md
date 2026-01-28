Module 04 — Prompt Engineer (Strict & Pluggable)

Module ID: M04
Owner Role: Prompt Compiler Engineer
Goal: Compile raw user text into a deterministic, strictly validated PromptSpec + ToolPlan where:
	•	the prediction is unambiguous,
	•	evidence sources are explicit (URLs/endpoints),
	•	resolution rules are deterministic,
	•	downstream output is constrained to YES/NO/INVALID with confidence.

This module is also pluggable: you can replace the prompt compiler with alternate modules without changing orchestrator/collector contracts.

⸻

0) Required Schema Updates (Minimal Additions)

To support strictness, the Prompt Engineer requires these schema capabilities:

A) Add/extend core/schemas/prompts.py

Add explicit source targets to DataRequirement:
	•	source_targets: list[SourceTarget]
	•	selection_policy: SelectionPolicy

B) Add SourceTarget and SelectionPolicy (new in core/schemas/transport.py OR prompts.py)

Recommended location: core/schemas/transport.py since it governs “how tools run”.

SourceTarget must allow explicit URL/endpoints:
	•	source_id: str (e.g., “http”, “web”, “polymarket”, “exchange”)
	•	uri: str (explicit URL or endpoint)
	•	method: Literal["GET","POST","RPC","WS","OTHER"]
	•	headers?: dict
	•	params?: dict
	•	body?: dict|str
	•	auth_ref?: str (string reference only; no secrets)
	•	expected_content_type: Literal["json","text","html","bytes"]
	•	cache_ttl_seconds?: int
	•	notes?: str

SelectionPolicy
	•	strategy: Literal["single_best","multi_source_quorum","fallback_chain"]
	•	min_sources: int = 1
	•	max_sources: int = 3
	•	quorum: int = 1  (for multi_source_quorum)
	•	tie_breaker: Literal["highest_provenance","source_priority","most_recent"]

If you don’t want to modify schemas yet, you can store these in DataRequirement.extra, but the strict version should make them first-class.

C) Verdict output already constrained

core/schemas/verdict.py already has Outcome = YES/NO/INVALID and confidence. Prompt Engineer must set PromptSpec.output_schema_ref = core.schemas.verdict.DeterministicVerdict and must define rules for outcome/confidence.

⸻

1) Owned Files
	•	agents/prompt_engineer/prompt_agent.py
	•	agents/prompt_engineer/__init__.py
	•	(optional) orchestrator/prompt_engineer.py as wrapper

⸻

2) High-Level Contract (What This Module Must Output)

PromptEngineerAgent.run(user_input) -> (PromptSpec, ToolPlan) where:

PromptSpec MUST include:
	•	market: MarketSpec with explicit:
	  •	event_definition (machine-resolvable)
	  •	resolution_window + resolution_deadline
	  •	resolution_rules (deterministic mapping to YES/NO/INVALID)
	  •	allowed_sources aligned with explicit source targets
	•	prediction_semantics: PredictionSemantics normalized
	•	data_requirements: list[DataRequirement] where each requirement includes explicit URLs/endpoints (SourceTargets)
	•	output_schema_ref = "core.schemas.verdict.DeterministicVerdict"
	•	extra includes:
	  •	assumptions (if any)
	  •	id_scheme
	  •	strict_mode: true

ToolPlan MUST include:
	•	explicit ordered requirements list
	•	source IDs referenced by requirements
	•	min_provenance_tier (global floor)
	•	execution mode & selection policy hints in extra

⸻

3) “Structured Input” (Normalized User Request Schema)

Even if the user provides raw text, your module should internally normalize to:

NormalizedUserRequest (internal-only, not a schema file required)

Fields to derive:
	•	question: str (user-facing)
	•	time: { timezone, window_start, window_end }
	•	event_definition: str (machine statement)
	•	outcome_type: Literal["binary"] (only binary for now)
	•	threshold?: str
	•	source_preferences: list[str] (optional)
	•	source_targets: list[SourceTarget] (explicit)
	•	confidence_policy: { min_confidence_for_yesno: float, default_confidence: float }
	•	resolution_policy: { invalid_conditions: list[str], conflict_policy: ... }

If fields are missing, fill defaults but must record assumptions.

⸻

4) Strict Outcome Rules (YES/NO/INVALID + Confidence)

Prompt Engineer must encode deterministic resolution guidance such that Judge can map trace → verdict.

4.1 Outcome constraints
	•	Outcome ∈ {YES, NO, INVALID}
	•	Confidence ∈ [0,1]

4.2 Confidence policy (minimum)

PromptSpec.extra should define:
	•	confidence_policy.min_confidence_for_yesno (recommended default 0.55)
	•	if evidence is insufficient OR conflicting without quorum → outcome must be INVALID (not YES/NO)

4.3 Deterministic mapping rules must be explicit

MarketSpec.resolution_rules should include deterministic rules, e.g.:
	1.	Validity Rule
	•	If required evidence (all required requirements) is missing OR below provenance tier → INVALID
	2.	Conflict Rule
	•	If conflicting evidence:
	  •	prefer higher provenance tier
	  •	if same tier: apply source_priority or quorum rules
	3.	Binary Decision Rule
	•	If event_definition evaluates True → YES
	•	Else → NO
	4.	Confidence Assignment Rule
	•	If single-source: confidence = clamp(source_confidence)
	•	If quorum: confidence = min(1.0, quorum_count / max_sources) * provenance_weight
	•	If fallback used: reduce confidence

You’re not implementing Judge here, but Prompt Engineer must supply the policy text + machine hints.

⸻

5) DataRequirement: Make Sources Explicit

Update DataRequirement usage so each requirement is fully fetchable.

Each DataRequirement MUST include:
	•	requirement_id
	•	description
	•	source_targets: list[SourceTarget] (>=1)
	•	selection_policy: SelectionPolicy
	•	min_provenance_tier
	•	expected_fields (optional, can be in extra): what JSON keys or text patterns are needed

Example for “official CPI number”:
	•	target1: uri="https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0" method POST expected json
	•	target2: uri="https://www.bls.gov/cpi/" method GET expected html (fallback)
	•	policy: fallback_chain, min_sources=1, max_sources=2

Example for “Polymarket resolution”:
	•	target: uri="https://gamma-api.polymarket.com/markets/<id>" GET json
	•	policy: single_best

⸻

6) Pluggability: Define a Prompt Module Interface

In agents/prompt_engineer/prompt_agent.py implement a clean separation:

Interface (internal)

class PromptModule(Protocol):
    module_id: str
    version: str

    def compile(self, user_input: str, *, ctx: Optional[AgentContext] = None) -> tuple[PromptSpec, ToolPlan]:
        ...

Default implementation
	•	StrictPromptCompilerV1(PromptModule) inside same file (or separate file later)
	•	PromptEngineerAgent becomes a thin wrapper that delegates to a module:

class PromptEngineerAgent(BaseAgent):
    def __init__(self, module: Optional[PromptModule] = None, ...):
        self.module = module or StrictPromptCompilerV1(...)

Benefit: You can later swap in a different compiler (e.g., LLM-first vs DSL-first) without changing orchestrator.

⸻

7) Implementation Requirements (prompt_agent.py)

Required behavior

run() must:
	1.	parse → normalized request
	2.	enforce strictness:
	•	reject impossible window
	•	ensure event_definition non-empty
	•	ensure at least 1 DataRequirement
	•	ensure every requirement has explicit source_targets
	3.	create MarketSpec + rules
	4.	produce PromptSpec + ToolPlan
	5.	record assumptions, provenance tier requirements, and confidence policy

Deterministic IDs

Same as before, but ensure stable ordering:
	•	deterministic market_id
	•	stable requirement_id sequence
	•	deterministic plan_id

“Strict Mode”

Always set:
	•	PromptSpec.extra["strict_mode"] = True
	•	PromptSpec.extra["compiler"] = {"module_id": ..., "version": ...}

⸻

8) Testing Requirements (tests/unit/test_prompt_engineer.py)

Add tests verifying strict structure:
	1.	Every requirement has ≥1 source_targets and valid uri
	2.	ToolPlan.requirements exactly equals DataRequirement.requirement_id in order
	3.	OutputSchemaRef points to DeterministicVerdict
	4.	Confidence policy exists in PromptSpec.extra and is numeric
	5.	Deterministic IDs stable across runs
	6.	If user input missing timeframe, assumptions are recorded and default window applied

⸻

9) Acceptance Checklist
	•	PromptSpec produced is fully resolvable (explicit event_definition + window + rules)
	•	DataRequirements contain explicit URLs/endpoints (source_targets)
	•	Output schema is locked to DeterministicVerdict (YES/NO/INVALID + confidence)
	•	Compiler is swappable via PromptModule interface
	•	Strict mode recorded in PromptSpec.extra
	•	All tests pass; file remains ≤ 500 LOC

⸻

10) Example Output (What “Strict” Looks Like)

User input:

“Will BTC close above $100,000 on 2026-12-31 UTC? Use Coinbase.”

Prompt Engineer outputs:
	•	MarketSpec.event_definition:
	  •	"BTC-USD daily close on 2026-12-31T00:00:00Z (Coinbase) > 100000"
	•	DataRequirement req_0001:
	  •	source_targets:
	    •	{"source_id":"exchange","uri":"https://api.exchange.coinbase.com/products/BTC-USD/candles?...","method":"GET","expected_content_type":"json"}
	    •	fallback http URL if specified
	  •	selection_policy: fallback_chain
	•	PromptSpec.extra.confidence_policy:
	  •	{ "min_confidence_for_yesno": 0.55, "default_confidence": 0.7 }
