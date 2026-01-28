Module 09A — Pipeline Integration (In-Process Runtime Wiring)

Module ID: M09A
Owner Role: Systems Integrator / Runtime Engineer
Goal: Provide a deterministic, testable, in-process pipeline runner that composes Modules 04–08 into an executable flow.

Key Output: a single entrypoint that takes user_input and returns a RunResult containing all artifacts:
	•	PromptSpec + ToolPlan
	•	EvidenceBundle
	•	ReasoningTrace (+ audit checks)
	•	DeterministicVerdict (+ judge checks)
	•	PoRBundle (+ roots)
	•	Optional Sentinel verification result + challenges

⸻

1) Owned Files

In orchestrator/ implement or update:
	•	pipeline.py  ✅ (main runner)
	•	sop_executor.py ✅ (step executor / glue)
	•	market_resolver.py ✅ (maps to “market resolution scenario”; can be thin)
	•	__init__.py
	•	(optional) prompt_engineer.py (thin wrapper; if you already have it, keep it tiny)

Do NOT implement API routes or persistence in 09A.

⸻

2) Dependencies

Agents
	•	agents.prompt_engineer.PromptEngineerAgent
	•	agents.collector.CollectorAgent
	•	agents.auditor.AuditorAgent
	•	agents.judge.JudgeAgent
	•	agents.validator.SentinelAgent (optional)

Core protocol
	•	core.por.proof_of_reasoning.compute_roots
	•	core.por.por_bundle.PoRBundle, build_por_bundle
	•	core.schemas.verification.VerificationResult, CheckResult
	•	core.schemas.market.MarketSpec / PromptSpec / EvidenceBundle / ReasoningTrace / DeterministicVerdict

⸻

3) Non-Goals (Keep 09A Tight)
	•	No web server (FastAPI) yet
	•	No on-chain interactions
	•	No DB or filesystem persistence
	•	No message queue / async orchestration
	•	No distributed execution

⸻

4) Required Public API

4.1 PipelineConfig

Define in orchestrator/pipeline.py:

Fields (suggested):
	•	strict_mode: bool = True
	•	enable_replay: bool = False (sentinel replay)
	•	enable_sentinel_verify: bool = True
	•	max_runtime_s: int = 60
	•	deterministic_timestamps: bool = True (ensures created_at / retrieved_at / resolution_time = None)
	•	debug: bool = False

4.2 RunResult

Define in orchestrator/pipeline.py (Pydantic or dataclass):
	•	prompt_spec
	•	tool_plan
	•	evidence_bundle
	•	audit_trace
	•	audit_verification: VerificationResult
	•	verdict
	•	judge_verification: VerificationResult
	•	por_bundle
	•	roots (optional PoRRoots)
	•	sentinel_verification: VerificationResult | None
	•	challenges: list | None
	•	ok: bool
	•	checks: list[CheckResult] (aggregated)

4.3 Pipeline.run(user_input: str) -> RunResult

This is the main entrypoint.

⸻

5) Orchestrator Control Flow

Step 1 — Prompt compilation

Call PromptEngineerAgent.run(user_input) → (PromptSpec, ToolPlan)
	•	If config.strict_mode: assert prompt_spec.extra["strict_mode"] == True
	•	If config.deterministic_timestamps: ensure prompt_spec.created_at is None (or equivalent)

Step 2 — Evidence collection

Call CollectorAgent.collect(prompt_spec, tool_plan) → EvidenceBundle
	•	If deterministic timestamps: enforce retrieved_at=None (if schema supports it)
	•	Collector policy failures should be recorded in EvidenceBundle summary; orchestrator should still proceed.

Step 3 — Audit / reasoning trace

Call AuditorAgent.audit(prompt_spec, evidence_bundle) → (ReasoningTrace, VerificationResult)
	•	If audit_verification.ok is False:
	•	orchestrator still proceeds to Judge but will likely lead to INVALID

Step 4 — Judge verdict

Call JudgeAgent.judge(prompt_spec, evidence_bundle, trace) → (DeterministicVerdict, VerificationResult)

Step 5 — Compute roots and build PoR bundle

Compute roots = compute_roots(prompt_spec, evidence_bundle, trace, verdict)
Build PoRBundle = build_por_bundle(...) with:
	•	commitments from roots
	•	verdict embedded
	•	created_at allowed in PoRBundle (non-committed) or set None; be consistent

Step 6 — Optional Sentinel verify

If enabled:
	•	assemble PoRPackage and call SentinelAgent.verify(...)
	•	if replay enabled and tool_plan present: sentinel can replay

Step 7 — Aggregate checks and final ok
	•	ok = all of:
	•	prompt compilation succeeded
	•	judge_verification ok
	•	(optional) sentinel ok
	•	If verdict outcome INVALID, ok may still be True (it’s a valid resolution). Use:
	•	ok = pipeline executed successfully
	•	store verdict.outcome separately for business logic

⸻

6) SOP Executor (orchestrator/sop_executor.py)

Purpose: keep pipeline steps composable and testable; minimal.

Implement a small “step runner” abstraction:
	•	SOPStep protocol:
	•	name
	•	run(state) -> state
	•	PipelineState dataclass:
	•	holds artifacts incrementally
	•	SOPExecutor.execute(steps, state) -> state

This helps you later add 09B (API) and 09C (persistence) without rewriting flow.

Keep it small; do NOT add DAG scheduler yet.

⸻

7) market_resolver.py (Thin)

Purpose: translate DeterministicVerdict into “market resolution record”.

Implement:
	•	resolve_market(verdict: DeterministicVerdict) -> dict
Return:
	•	{market_id, outcome, confidence, resolution_rule_id}

No persistence.

⸻

8) Deterministic Timestamp Enforcement (Important)

In strict mode, orchestrator must strip/avoid non-deterministic fields in committed objects:
	•	PromptSpec.created_at must be None
	•	RetrievalReceipt.retrieved_at should be None
	•	Verdict.resolution_time should be None

Implement helper in pipeline.py:
	•	_enforce_commitment_safe_timestamps(prompt_spec, evidence_bundle, verdict)

If your schemas already default these to None, this becomes a no-op.

⸻

9) Tests Required

Add:
	•	tests/unit/test_pipeline_integration.py

Strategy:
	•	Use mocked Collector/Auditor/Judge that return deterministic artifacts OR use real implementations with stubbed data sources.
	•	Ensure:
	•	pipeline returns RunResult with all fields
	•	roots and por_bundle exist
	•	sentinel verification passes when artifacts unchanged
	•	tamper an evidence item and sentinel fails

⸻

10) Acceptance Checklist
	•	Pipeline.run() executes end-to-end locally
	•	Returns full artifact pack (PromptSpec/ToolPlan/Evidence/Trace/Verdict/PoRBundle)
	•	Deterministic timestamp stripping enforced in strict mode
	•	Optional sentinel verification integrated
	•	Unit test validates integration
	•	Each file ≤ 500 LOC

⸻

Next splits after 09A (so you can parallelize)
	•	Module 09B — Packaging & Artifact IO
Define PoRPackage serialization (JSON), load/save, zip bundles, etc.
	•	Module 09C — API Surface (FastAPI routes)
Minimal endpoint: /run, /verify, /challenge.
	•	Module 09D — CLI
python -m cournot run "<user_input>" --out artifacts.json