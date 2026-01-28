# Cournot Protocol — Architecture

This document is the **single source of truth** for how this repository is structured and how the core “Proof of Reasoning” (PoR) pipeline works end-to-end.

The goal is that **any AI (or human)** can pick up *any module* in this repo and implement/modify code safely without guessing hidden contracts.

---

## 1. What this project is

Cournot is a deterministic, verifiable pipeline that turns a user’s market question into:

1) a **structured, unambiguous PromptSpec**,  
2) an **EvidenceBundle** with provenance,  
3) an **Audited ReasoningTrace**,  
4) a **DeterministicVerdict (YES/NO/INVALID + confidence)**,  
5) a **PoRBundle** with cryptographic commitments (hashes / Merkle roots) that bind the above artifacts,  
6) an **Artifact Pack** (directory or zip) that is portable and independently verifiable.

“Deterministic” here means:
- the committed artifacts serialize identically via canonical JSON, and
- non-deterministic fields (timestamps, random IDs) are **excluded from commitments** or normalized to `None` in strict mode.

---

## 2. Repository layout

High-level folders and the responsibilities they own:

- `core/`
  - **Pure, deterministic primitives**: canonical JSON, hashing, Merkle commitments, schemas/models, PoR bundle/roots computation.
  - No network calls, no LLM calls.
- `agents/`
  - “Role” implementations: prompt engineer, collector, auditor, judge, validator/sentinel.
  - This is where policy/LLM logic lives.
- `orchestrator/`
  - Composes agents into a deterministic pipeline runner.
  - Responsible for SOP sequencing, state passing, aggregation, building PoR bundles, and producing artifact packs.
- `orchestrator/artifacts/`
  - Portable pack formats (directory/zip), manifest + hashes, load/save/validate utilities.
- `api/`
  - Minimal FastAPI surface for `/run`, `/verify`, `/replay` and request/response models.
- `docs/modules/`
  - Module design contracts (what each module must do, public interfaces, acceptance tests).
- `tests/`
  - Unit + integration tests that enforce determinism, schema stability, and verification behavior.

---

## 3. System artifacts (the “contracts”)

These are the objects you should treat as **public interfaces**. They should be versioned, backward compatible where possible, and always serializable via canonical JSON.

### 3.1 PromptSpec (+ MarketSpec)

**Produced by:** Prompt Engineer (Module 04)  
**Consumed by:** Collector, Auditor, Judge, Pipeline/Orchestrator

Purpose: make the user’s intent executable and unambiguous:
- explicit event definition and time window
- explicit resolution criteria and allowed resolution sources
- explicit required data sources/endpoints (“source_targets”)
- output schema locked to `DeterministicVerdict`

### 3.2 ToolPlan (optional but recommended)

**Produced by:** Prompt Engineer  
Purpose: the execution plan for evidence collection and replay (when applicable).

### 3.3 EvidenceBundle

**Produced by:** Collector (Module 05)  
Purpose: a list of EvidenceItems + provenance proofs and retrieval receipts.

The EvidenceBundle must be able to commit to a stable **evidence_root** for PoR.

### 3.4 ReasoningTrace

**Produced by:** Auditor (Module 06)  
Purpose: a structured trace of how conclusions were derived from evidence, with policy checks.

The trace must be able to commit to a stable **reasoning_root** for PoR.

### 3.5 DeterministicVerdict

**Produced by:** Judge (Module 07)  
Purpose: the final outcome and confidence; *not* free-form text.

Required outcomes:
- `YES`
- `NO`
- `INVALID` (means the system executed successfully but the market cannot be resolved deterministically)

### 3.6 PoRRoots + PoRBundle

**Produced by:** Orchestrator (Module 09A) using core commitments (Modules 02–03)  
Purpose:
- Bind `PromptSpec`, `EvidenceBundle`, `ReasoningTrace`, and `DeterministicVerdict` into a single cryptographic commitment record.
- Provide `por_root` and component hashes/roots.

### 3.7 PoRPackage (portable bundle)

**Produced by:** Artifact Packaging (Module 09B)  
Purpose: a single container holding the bundle plus all referenced artifacts, ready to save/load/verify.

---

## 4. Pipeline (SOP) — end-to-end control flow

The orchestrator runs a strict sequence of steps (“SOP”). The **primary entrypoint** is:

- `Pipeline.run(user_input: str) -> RunResult`

### 4.1 Step-by-step flow

1) **Prompt compilation**
   - PromptEngineerAgent.run(user_input) → `(PromptSpec, ToolPlan)`
   - strict mode must record `PromptSpec.extra["strict_mode"] = True`
   - deterministic timestamps (if enabled) must strip timestamps from committed artifacts

2) **Evidence collection**
   - CollectorAgent.collect(prompt_spec, tool_plan) → `EvidenceBundle`
   - provenance proofs/receipts must be attached (tier-dependent)
   - failures should be recorded but the pipeline may continue

3) **Audit / reasoning trace**
   - AuditorAgent.audit(prompt_spec, evidence_bundle) → `(ReasoningTrace, VerificationResult)`
   - if audit verification fails, pipeline still continues, but likely ends in INVALID

4) **Judge verdict**
   - JudgeAgent.judge(prompt_spec, evidence_bundle, trace) → `(DeterministicVerdict, VerificationResult)`

5) **Compute roots + build PoR bundle**
   - compute roots from (PromptSpec, EvidenceBundle, ReasoningTrace, Verdict)
   - build PoRBundle with the verdict embedded

6) **Optional Sentinel verify**
   - assemble a PoRPackage and run SentinelAgent.verify(...)
   - optional replay (if ToolPlan exists and replay is enabled)

7) **Aggregate checks**
   - `ok` indicates whether the pipeline executed successfully (even if the verdict is INVALID)

### 4.2 Determinism rules enforced by orchestrator

In strict/deterministic mode, **committed objects must not include non-deterministic fields**:
- PromptSpec timestamps must be `None`
- RetrievalReceipt timestamps should be `None`
- Verdict resolution timestamps should be `None`

If a schema includes such fields, they must be excluded from the canonical commitment or normalized before hashing.

---

## 5. Module map (what each module owns)

> Naming follows the design docs in `docs/modules/`. The concrete Python layout may differ slightly; the responsibilities below are the contracts to preserve.

### Module 01 — Schemas & Canonicalization (core)

**Owns:**
- Canonical JSON serialization (`dumps_canonical`)
- Schema versioning (protocol/schema version constants)
- Pydantic models for PromptSpec, ToolPlan, EvidenceBundle, ReasoningTrace, DeterministicVerdict, VerificationResult, etc.

**Public API (typical):**
- `dumps_canonical(obj) -> str|bytes`
- `model.model_dump(...)` must be commitment-safe (exclude unstable fields)

**Key constraints:**
- Canonical output must be stable across platforms and runs.
- Schema evolution must be explicit (versions), and new fields should be optional with sane defaults.

### Module 02 — Merkle Commitments (core)

**Owns:**
- Merkle root computation from leaf hashes
- Merkle proof format and verification

**Public API (typical):**
- `build_merkle_root(leaf_hashes: list[bytes]) -> bytes`
- `build_merkle_proof(leaf_hashes, index) -> MerkleProof`
- `verify_merkle_proof(root, leaf_hash, proof) -> bool`

**Key constraints:**
- Must define deterministic padding behavior (e.g., duplicate last leaf when odd).

### Module 03 — PoR Bundle & Commitments (core)

**Owns:**
- Commitment computation (prompt hash, evidence root, reasoning root, verdict hash)
- PoRRoots structure and por_root derivation
- PoRBundle schema and verify helpers

**Public API (typical):**
- `compute_roots(prompt_spec, evidence_bundle, trace, verdict) -> PoRRoots`
- `build_por_bundle(roots, verdict, ...) -> PoRBundle`
- `verify_por_bundle(bundle, prompt_spec, evidence_bundle, trace, verdict) -> VerificationResult`

### Module 04 — Prompt Engineer (agent)

**Owns:**
- Conversion from user free-text → strict PromptSpec (+ ToolPlan)
- Deterministic IDs and strict mode metadata
- Validation that the PromptSpec is executable and unambiguous

**Inputs:** `user_input: str`  
**Outputs:** `(PromptSpec, ToolPlan)`

**Acceptance requirements (examples):**
- every DataRequirement has ≥1 explicit URL/source_target
- ToolPlan requirements match PromptSpec requirement ordering
- output schema locked to DeterministicVerdict (YES/NO/INVALID + confidence)
- strict mode recorded in `PromptSpec.extra`

### Module 05 — Evidence Collector (agent)

**Owns:**
- Fetching evidence from sources described in PromptSpec/ToolPlan
- Generating RetrievalReceipts and provenance proofs (tier-based)
- Normalizing/structuring evidence into EvidenceBundle

**Inputs:** `(PromptSpec, ToolPlan)`  
**Outputs:** `EvidenceBundle`

**Key sub-areas:**
- `agents/collector/data_sources/` (polymarket, web/http, etc.)
- `agents/collector/verification/` (signature verification, zkTLS verifier, tier policy)

### Module 06 — Auditor (agent)

**Owns:**
- Producing a structured ReasoningTrace from evidence and prompt spec
- Running contradiction checks, claim extraction, and trace-policy verification
- Outputting a VerificationResult with granular CheckResults

**Inputs:** `(PromptSpec, EvidenceBundle)`  
**Outputs:** `(ReasoningTrace, VerificationResult)`

### Module 07 — Judge (agent)

**Owns:**
- Mapping evidence + reasoning trace to a DeterministicVerdict
- Schema locking and deterministic mapping checks
- Outputting a VerificationResult (judge checks)

**Inputs:** `(PromptSpec, EvidenceBundle, ReasoningTrace)`  
**Outputs:** `(DeterministicVerdict, VerificationResult)`

### Module 08 — Validator / Sentinel (agent)

**Owns:**
- Independent re-verification of a PoRPackage / artifact pack
- Recomputing commitments and verifying por_root
- Optional replay (if ToolPlan exists) using `replay_executor`
- Producing challenges / failures (`agents/validator/challenges.py`)

**Inputs:** `PoRPackage` (or pack path/bytes)  
**Outputs:** `VerificationResult` (+ optional challenges)

### Module 09A — Pipeline Integration (orchestrator)

**Owns:**
- The in-process deterministic pipeline runner composing Modules 04–08
- SOP step execution (SOPExecutor), PipelineState
- Aggregating VerificationResults
- Producing RunResult containing all artifacts + roots + bundle

**Key types:**
- `PipelineConfig` (strict mode, replay enabling, deterministic timestamp mode)
- `RunResult` (full output container)
- `resolve_market(verdict)` helper (thin translation layer)

### Module 09B — Artifact Packaging & IO (orchestrator)

**Owns:**
- PoRPackage container type
- Pack layout, manifest, and file hashing
- Save/load/validate functions

**Pack formats:**
- Directory format (inspectable)
- Zip format (portable)

**Proposed layout:**
```
pack/
  manifest.json
  prompt_spec.json
  tool_plan.json          (optional)
  evidence_bundle.json
  reasoning_trace.json
  verdict.json
  por_bundle.json
  checks/                 (optional)
    pipeline_checks.json
    audit_checks.json
    judge_checks.json
    sentinel_checks.json
  blobs/                  (optional future)
```

### Module 09C — CLI (tooling)

**Owns:**
- A command-line interface to run and verify pipeline outputs.

**Typical commands:**
- `cournot run "<question>" --out <path> [--strict]`
- `cournot verify <pack_path> [--por-root <expected>]`
- `cournot replay <pack_path> [...]`

The CLI should be a thin wrapper over orchestrator and artifact APIs.

### Module 09D — Minimal API (service)

**Owns:**
- FastAPI surface for running and verifying PoR.

**Endpoints (typical):**
- `POST /run` → execute pipeline and return verdict + por_root + pack path/bytes
- `POST /verify` → verify an existing pack
- `POST /replay` → replay steps (if ToolPlan supports it)
- `GET /healthz` → liveness

The API should call orchestrator/artifacts primitives; no business logic should be embedded here.

---

## 6. “How to implement safely” (rules for contributors & AIs)

1) **Never bypass core schemas**
   - Agents and orchestrator must exchange Pydantic schema objects (or their canonical JSON equivalents).

2) **Canonical JSON is the truth**
   - Any commitment hash must be over canonical JSON bytes.
   - If you change a schema, you must consider versioning and back-compat.

3) **Determinism is a feature**
   - Do not commit runtime timestamps, random IDs, or environment-specific data.
   - If you need them operationally, store them in non-committed fields or the pack’s optional `checks/`.

4) **Verification is composable**
   - Every module should return `VerificationResult` with granular `CheckResult`s.
   - Sentinel should be able to run without network access when verifying an existing pack (unless replay is requested).

5) **Pluggability**
   - Add new data sources under `agents/collector/data_sources/`.
   - Add new provenance verifiers under `agents/collector/verification/`.
   - Add new prompt compilers behind the Prompt Engineer interface.

---

## 7. Quick developer mental model

If you only remember one thing:

> The pipeline produces a portable pack that can be validated by recomputing commitments and comparing the final `por_root`.

The practical consequences:
- every field in committed artifacts must be stable and canonicalizable,
- pack layout and manifest hashing must be consistent,
- validators should never “trust” agent output without recomputing what can be recomputed.

---

## 8. Appendix — recommended “entrypoints” when navigating code

- **Run pipeline (programmatic):** `orchestrator.Pipeline.run(...)`
- **Pack IO:** `orchestrator.artifacts.save_pack(...)`, `load_pack(...)`, `validate_pack(...)`
- **Collector:** `agents.collector.CollectorAgent.collect(...)`
- **Auditor:** `agents.auditor.AuditorAgent.audit(...)`
- **Judge:** `agents.judge.JudgeAgent.judge(...)`
- **Sentinel:** `agents.validator.SentinelAgent.verify(...)`
- **API:** `api/app.py`

