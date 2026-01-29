# Change Doc — Productionizing Agents (LLM + Network) & Making Them Modular/Swappable

**Status:** Proposed refactor plan  
**Scope:** Modules 04–08 primarily (Agents), plus orchestrator wiring (Module 09A) and artifact/verification adjustments (Module 09B/03).  
**Goal:** Replace current local/pattern-based “agents” with **production-capable** implementations that can use **external LLMs and real network data**, while keeping the system **deterministic in its committed artifacts** and making each agent **pluggable/swappable**.  
**Non-goal:** Rewriting core commitment primitives; we preserve canonicalization + commitments and extend receipts/provenance where needed.

---

## 0) Key outcomes required

1. **Agents can call external LLM providers** (OpenAI/Gemini/Anthropic/local models) via a unified interface.
2. **Agents can fetch network data** (HTTP, Polymarket APIs, RSS, blockchain RPC, etc.) via a unified “DataSource” layer.
3. **Every external call is recorded as a Receipt** (request + response + metadata + hashing) so outputs are reproducible/auditable.
4. **Agents are modular and swappable**:
   - standard Agent interface
   - registry + selection policy
   - config-driven selection by step
5. The **pipeline can dynamically choose different agent implementations** per step based on: question type, data availability, cost, latency, “strictness tier”, etc.
6. **Committed artifacts remain deterministic**: non-deterministic operational metadata is either normalized or placed in non-committed receipts/check files.

---

## 1) New cross-cutting abstractions (add these first)

These are foundational changes used by all agent modules.

### 1.1 Agent Interface + Capability Model

Create a shared interface in e.g. `agents/base.py`:

```python
class Agent(Protocol):
    name: str
    version: str
    capabilities: set[str]  # e.g. {"llm", "network", "polymarket"}

    async def run(self, ctx: "AgentContext", **kwargs) -> "AgentResult":
        ...
```

- `version` is required and must change when agent behavior changes (so packs are traceable).
- `capabilities` allows the pipeline to route/choose agents.

### 1.2 AgentContext (dependency injection)

Create `AgentContext` containing *only* abstract service handles:

- `llm: LLMClient`
- `http: HttpClient`
- `clock: Clock` (can be deterministic/frozen)
- `tracer: ReceiptRecorder`
- `logger`
- `config: RuntimeConfig`
- `cache: Cache` (optional)
- `secrets: SecretsProvider` (only used in runtime, never committed)

This prevents agents from directly importing concrete network/LLM libs, and makes agents swappable/testable.

### 1.3 LLMClient (provider-agnostic)

Define an abstract `LLMClient` and concrete adapters:

- `OpenAIClient`
- `AnthropicClient`
- `GeminiClient`
- `LocalHFClient` (optional)

Required features:
- standard `chat()` API
- response must include `model`, `usage`, and **raw text/json**
- support tool/function calling later (optional)

### 1.4 HttpClient + DataSource Plugins

Define `HttpClient` and a `DataSource` plugin interface:

- `HttpClient.request(method, url, headers, body, timeout)`
- `DataSource.fetch(requirement, ctx) -> EvidenceItem(s)`

`DataSource` examples:
- `PolymarketSource`
- `GenericWebSource`
- `ChainRPCSource`

### 1.5 ReceiptRecorder (critical)

Every external interaction must be recorded as a receipt, stored as **non-committed** operational logs *or* committed as part of EvidenceBundle if your tier requires.

Receipt schema (recommended):
- `receipt_id`
- `kind: "llm" | "http" | "rpc"`
- `request: {…}`
- `response: {…}`
- `timing: {started_at, ended_at}` (normalized or excluded from commitment)
- `hashes: {request_hash, response_hash}`

**Important:** For determinism, the *committed* artifacts should contain stable hashes and canonical content; timestamps can be excluded or normalized.

### 1.6 Agent Registry + Selection Policy

Add `agents/registry.py`:
- registry maps `(step, agent_name)` -> factory
- supports multiple agents per step

Add `orchestrator/selector.py`:
- selects agent for each step via config + heuristic rules
- allows dynamic selection based on PromptSpec/Market type/strictness tier

Example config:
```yaml
agents:
  prompt_engineer:
    default: "PromptEngineerLLM:v1"
    options:
      - "PromptEngineerLLM:v1"
      - "PromptEngineerRegexFallback:v1"
  collector:
    default: "CollectorNetwork:v1"
  auditor:
    default: "AuditorLLM:v1"
  judge:
    default: "JudgeLLMStructured:v1"
```

---

## 2) Module-by-module change requirements

Below is the work broken down by modules. Each module section includes:
- **Problems today**
- **Target design**
- **Interfaces**
- **Implementation steps**
- **Acceptance criteria**

---

## Module 04 — Prompt Engineer Agent (START HERE)

### 4.1 Problems today
- PromptSpec generation is local/pattern-based, brittle, not generalizable.
- No real parsing of user intent into a strict market resolution spec.
- Not modular; hard to swap prompt strategies or models.

### 4.2 Target design
Create a swappable prompt engineer architecture with:
- `PromptEngineerLLM` (primary, production)
- `PromptEngineerFallback` (local deterministic fallback)
- strict JSON output validated by schema
- explicit ToolPlan generation (optional but recommended)

### 4.3 Required interfaces
`PromptEngineerAgent.run(ctx, user_input: str) -> PromptEngineerResult`
- `prompt_spec: PromptSpec`
- `tool_plan: ToolPlan | None`
- `receipts: list[ReceiptRef]`
- `checks: VerificationResult`

### 4.4 LLM prompt design requirements
The LLM must output **only JSON** matching PromptSpec schema:
- event definition, resolution criteria, time window, sources
- required evidence list with explicit URL targets
- explicit “resolution sources” and their hierarchy
- stable IDs (deterministic ID generation strategy)
- confidence rubric (used later by judge)

Determinism strategy:
- stable IDs derived from hashes of normalized fields (e.g., `sha256(title|question|source_url)`)

### 4.5 Implementation steps
1) Add `PromptEngineerLLM` using `LLMClient.chat()`.
2) Implement schema validation + repair loop (max 2 retries):
   - if JSON invalid, send back error message and request corrected JSON.
3) Add `PromptEngineerFallback` using current pattern logic as fallback.
4) Add selection policy:
   - if `ctx.llm` available -> use LLM agent
   - else fallback
5) Attach LLM receipts to the run result (receipt refs committed optionally or stored under `checks/` in pack).

### 4.6 Acceptance criteria
- For diverse questions, PromptSpec always validates.
- PromptSpec includes at least one explicit source target per requirement.
- IDs are stable across runs given identical input.
- Agent can be swapped via config without code changes.

---

## Module 05 — Evidence Collector Agent

### 5.1 Problems today
- Collector does not actually fetch network evidence.
- Evidence is simulated or derived from local patterns.
- No provenance receipts tied to real network calls.

### 5.2 Target design
- Collector uses DataSource plugins to fetch data described in PromptSpec/ToolPlan.
- Every fetch produces:
  - EvidenceItem (normalized content)
  - RetrievalReceipt (request/response hash + minimal metadata)
- Support tiered verification:
  - Tier 0: basic HTTP receipt
  - Tier 1+: signed provenance / zkTLS / notary (future)

### 5.3 Required interfaces
`CollectorAgent.collect(ctx, prompt_spec, tool_plan) -> EvidenceBundle + VerificationResult`

EvidenceBundle fields to ensure:
- list of `EvidenceItem {id, source, url, content, extracted_facts, hashes}`
- `retrieval_receipts: list[ReceiptRef or Receipt]`
- stable evidence ordering (sort by URL + requirement index)

### 5.4 Implementation steps
1) Implement `HttpClient` (requests/aiohttp).
2) Implement `PolymarketSource` first (since you have historical CSV and event URLs):
   - fetch event info from Polymarket endpoints if available
   - if CSV provided, create a `CSVSource` that loads from a configured path
3) Implement `GenericWebSource` for resolution sources (AP/Fox/NBC etc.)
4) Normalize and canonicalize evidence content:
   - store raw response separately (optional)
   - commit normalized extracted facts + stable hashes
5) Add retries + rate limiting + timeout controls.
6) Attach receipts.

### 5.5 Acceptance criteria
- EvidenceBundle contains real fetched content or configured offline dataset (CSV).
- Each evidence item has a receipt and hashes.
- EvidenceRoot stable across reruns when inputs and fetched content identical.

---

## Module 06 — Auditor Agent

### 6.1 Problems today
- Reasoning trace derived locally; doesn’t leverage LLM to interpret evidence.
- No robust contradiction checks or citation linking to evidence items.

### 6.2 Target design
Auditor produces a structured ReasoningTrace using LLM:
- Extract claims
- Map each claim to evidence IDs
- Flag contradictions / missing evidence
- Produce audit checks

### 6.3 Required interfaces
`AuditorAgent.audit(ctx, prompt_spec, evidence_bundle) -> ReasoningTrace + VerificationResult`

Trace must include:
- `claims[] {claim_id, statement, support_evidence_ids[], confidence}`
- `gaps[]`
- `contradictions[]`
- deterministic ordering

### 6.4 Implementation steps
1) Implement `AuditorLLM`:
   - input: PromptSpec + summarized Evidence (with evidence IDs)
   - output: strict JSON for ReasoningTrace
2) Add optional local heuristic checks (cheap):
   - missing required sources
   - evidence outdated vs time window
3) Receipts stored like other LLM calls.

### 6.5 Acceptance criteria
- ReasoningTrace validates against schema.
- Every claim references evidence IDs or is flagged as unsupported.
- Contradictions detected for obvious conflicts.

---

## Module 07 — Judge Agent

### 7.1 Problems today
- Verdict is produced by local rules; not robust.
- No calibrated confidence; no structured justification trace.

### 7.2 Target design
Judge uses LLM but must produce deterministic output:
- `outcome`: YES/NO/INVALID
- `confidence`: float 0–1 with rubric
- `key_evidence_ids`: list
- `why_invalid`: required if INVALID

### 7.3 Required interfaces
`JudgeAgent.judge(ctx, prompt_spec, evidence_bundle, trace) -> Verdict + VerificationResult`

### 7.4 Implementation steps
1) Implement `JudgeLLMStructured`:
   - prompt includes explicit rubric
   - requires JSON-only output
2) Add deterministic post-processing:
   - clamp confidence to [0,1]
   - sort evidence IDs
3) Add local sanity checks:
   - if trace shows missing critical evidence -> force INVALID
4) Record LLM receipts.

### 7.5 Acceptance criteria
- Verdict always schema-valid.
- Verdict references evidence IDs.
- INVALID includes an explicit reason.

---

## Module 08 — Sentinel / Validator Agent

### 8.1 Problems today
- Verification does not consider external receipts or replay capability.
- No modular replay executor.

### 8.2 Target design
Sentinel verifies a PoRPackage by:
- recomputing commitments
- verifying por_root
- optionally replaying data fetch steps via ToolPlan and comparing hashes

### 8.3 Implementation steps
1) Extend PoRPackage schema to include receipt refs or embedded receipts (tiered).
2) Implement `ReplayExecutor`:
   - reads ToolPlan
   - calls DataSources
   - compares resulting response hashes (or extracted facts hashes)
3) Provide “offline verify mode”:
   - verify commitments only, without network
4) Output challenge objects for mismatches.

### 8.4 Acceptance criteria
- Can validate pack without network.
- Can replay when enabled and detect mismatches.

---

## Module 09A — Orchestrator Integration

### 9.1 Problems today
- Pipeline wired to fixed agents with no external capability.
- No agent selection strategy; no dependency injection.

### 9.2 Target design
- Orchestrator builds AgentContext (llm/http/tracer/config) once per run.
- Uses Selector to pick agents per step.
- Produces RunResult with pack + checks + receipts.

### 9.3 Implementation steps
1) Implement `RuntimeConfig` parsing (env + yaml).
2) Add `AgentRegistry` and `AgentSelector`.
3) Update Pipeline.run():
   - select PromptEngineer
   - select Collector
   - select Auditor
   - select Judge
   - build PoR bundle and pack
4) Ensure receipts are stored in a consistent pack location (e.g. `checks/receipts/`).

---

## Module 09B — Artifact Packaging

### 9.4 Changes needed
- Add a place for receipts:
  - `checks/receipts/*.json`
- Add manifest entries for receipt files (hash them)
- Decide commitment policy:
  - tier 0: receipts not part of commitments
  - tier 1+: receipt hashes included

### 9.5 Acceptance criteria
- Packs remain portable and verifiable.
- Receipt files are discoverable and hashed in manifest.

---

## Module 03 — Commitment / Bundle Adjustments (minimal)

### 3.1 Changes needed
- Define how receipts affect commitments (tier-based).
- If receipts included in evidence commitment, normalize timestamps.
- Maintain backwards compatibility by making receipt inclusion optional.

---

## 3) Modularity & Swappability Strategy (important)

### 3.1 Agent Factories
Each agent is created via a factory function that accepts `RuntimeConfig` and returns an Agent instance.

### 3.2 Step contracts
Keep step I/O schemas stable:
- PromptEngineer: -> PromptSpec (+ ToolPlan)
- Collector: -> EvidenceBundle
- Auditor: -> ReasoningTrace
- Judge: -> Verdict

### 3.3 Dynamic routing (future)
Add `StepRouter` policy rules:
- if question is “simple” -> cheaper model
- if strict tier high -> use premium model + replay enabled
- if sources restricted -> choose specialized collector

---

## 4) Backward compatibility plan

1) Keep current pattern-based agents as `*Fallback`.
2) Default selection uses LLM/network when configured, otherwise fallback.
3) Add `schema_version` increments only when you change committed formats.

---

## 5) Testing plan (must-have)

### Unit tests
- schema validation for each agent output
- determinism: IDs stable, ordering stable
- receipt recording

### Integration tests
- run pipeline with mocked LLM + mocked HTTP
- run pipeline with replay verify
- verify por_root recomputation matches

### Golden tests
- fixed input, fixed mocked responses -> exact artifact hashes match expected

---

## 6) Immediate implementation order (recommended)

1) **Add AgentContext + LLMClient + HttpClient + ReceiptRecorder**
2) **Module 04: PromptEngineerLLM** (with fallback)
3) **Module 05: CollectorNetwork + PolymarketSource + CSVSource**
4) **Module 06: AuditorLLM**
5) **Module 07: JudgeLLMStructured**
6) Orchestrator selector + config
7) Packaging receipts + manifest hashing
8) Sentinel replay (optional but high value)

---

## 7) Definition of Done (DoD)

- A run using real network + LLM produces:
  - PromptSpec, EvidenceBundle, ReasoningTrace, Verdict
  - PoRBundle and por_root
  - receipts saved in pack and hashed in manifest
- The pack is verifiable offline (commitments match).
- Agents are selectable by config and can be swapped without code changes.
