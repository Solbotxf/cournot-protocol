Module 09B — Artifact Packaging & IO

Module ID: M09B
Owner Role: Systems / Tooling Engineer
Goal: Define a stable on-disk (and transferable) format for a complete proof package (“artifact pack”), with deterministic serialization and convenient load/save APIs.

Key Outcome:
	•	save_pack(run_result) -> .zip or directory
	•	load_pack(path) -> PoRPackage
	•	validate_pack(package) -> VerificationResult

⸻

1) Owned Files / Directories

Create a new directory (recommended):

docs/modules/            (docs already exists; do not add code)
orchestrator/
  artifacts/
    __init__.py
    pack.py
    io.py
    manifest.py

If you prefer fewer files, you can combine pack.py + io.py, but keep each file ≤500 LOC.

Do not mix this into agents/validator/*. Packaging belongs to orchestrator tooling.

⸻

2) Dependencies

Core schemas
	•	core.schemas.prompts.PromptSpec
	•	core.schemas.transport.ToolPlan
	•	core.schemas.evidence.EvidenceBundle
	•	core.por.reasoning_trace.ReasoningTrace
	•	core.schemas.verdict.DeterministicVerdict
	•	core.por.por_bundle.PoRBundle
	•	core.schemas.verification.VerificationResult, CheckResult
	•	core.schemas.versioning.PROTOCOL_VERSION / schema versions (if present)

Utilities
	•	core.schemas.canonical.dumps_canonical (for deterministic json dumps)
	•	core.crypto.hashing.hash_canonical, to_hex
	•	Standard lib: json, zipfile, pathlib, hashlib, datetime, typing

⸻

3) Core Concepts

3.1 PoRPackage (portable bundle)

An internal container type used across 09A/09B/08:

Contains:
	•	bundle: PoRBundle  (commitments + embedded verdict)
	•	prompt_spec: PromptSpec
	•	tool_plan: ToolPlan | None
	•	evidence: EvidenceBundle
	•	trace: ReasoningTrace
	•	verdict: DeterministicVerdict (may equal bundle.verdict; store anyway for convenience)

3.2 Artifact Pack Layout (on disk)

Support two formats:
	1.	Directory format: easy to inspect
	2.	Zip format: easy to transmit

Proposed file layout (directory root):

pack/
  manifest.json
  prompt_spec.json
  tool_plan.json          (optional)
  evidence_bundle.json
  reasoning_trace.json
  verdict.json
  por_bundle.json
  checks/
    pipeline_checks.json  (optional)
    audit_checks.json     (optional)
    judge_checks.json     (optional)
    sentinel_checks.json  (optional)
  blobs/                  (optional future: raw html, screenshots, etc.)

v1 stores only JSON artifacts, no large binary blobs. Add blobs/ later.

⸻

4) Deterministic Serialization Rules

4.1 JSON formatting

All saved JSON MUST be:
	•	canonical (stable key ordering, no whitespace variability)
	•	UTF-8
	•	newline at end optional

Implement a helper:
	•	dump_json(obj) -> str using dumps_canonical() where possible, or json.dumps(..., sort_keys=True, separators=(",",":")) for plain dict.

4.2 Content hashes

Manifest must include SHA-256 for each artifact file content:
	•	sha256_hex = hashlib.sha256(file_bytes).hexdigest()

This allows tamper detection independent of PoR commitments.

⸻

5) Manifest Specification (orchestrator/artifacts/manifest.py)

Purpose

A small metadata file that describes the pack and provides file hashes + pointers.

PackManifest fields (Pydantic or dataclass)
	•	format_version: str = "pack.v1"
	•	protocol_version: str (from your core versioning)
	•	created_at: str | None (optional; informational only)
	•	market_id: str
	•	por_root: str (bundle.por_root if present else computed root in bundle metadata)
	•	files: dict[str, ManifestFileEntry]
	•	notes: dict = {}

Where ManifestFileEntry:
	•	path: str (relative path)
	•	sha256: str
	•	bytes: int

Required file keys:
	•	prompt_spec
	•	evidence_bundle
	•	reasoning_trace
	•	verdict
	•	por_bundle
Optional:
	•	tool_plan

⸻

6) Saving Packs (orchestrator/artifacts/io.py)

API functions
	•	save_pack_dir(package: PoRPackage, out_dir: str | Path, *, include_checks: dict | None = None) -> Path
	•	save_pack_zip(package: PoRPackage, out_zip: str | Path, *, include_checks: dict | None = None) -> Path

Rules
	•	Always write JSON files first
	•	Compute SHA-256 for each file and build manifest
	•	Write manifest last
	•	For zip: create zip from directory or stream-write each file

Minimum required behavior
	•	Always include: prompt_spec, evidence, trace, verdict, por_bundle, manifest
	•	tool_plan may be absent; but if present in package, include it
	•	Ensure filenames exactly match manifest

⸻

7) Loading Packs (orchestrator/artifacts/io.py)

API functions
	•	load_pack_dir(dir_path) -> PoRPackage
	•	load_pack_zip(zip_path) -> PoRPackage
	•	load_pack(path) -> PoRPackage (auto-detect dir/zip)

Hash verification
	•	Read manifest
	•	Recompute sha256 for each listed file
	•	If mismatch:
	•	raise CournotError OR return VerificationResult(ok=False) depending on style
Recommended: provide both:
	•	validate_pack_files(path) -> VerificationResult
	•	load_pack(..., verify_hashes=True) optionally

⸻

8) Pack Validation (orchestrator/artifacts/pack.py)

Purpose

Validate semantic correctness beyond file hashes:
	•	schema versions
	•	market_id consistency
	•	PoR commitments consistency

Implement:
	•	validate_pack(package: PoRPackage) -> VerificationResult

Validation checks (minimum)
	1.	package.prompt_spec.market.market_id == package.verdict.market_id == package.bundle.market_id
	2.	bundle.verdict_hash == hash_canonical(verdict) (if bundle embeds verdict separately, ensure equal)
	3.	If bundle.prompt_spec_hash exists: compare with hash_canonical(prompt_spec)
	4.	If you have compute_roots() available:
	•	recompute evidence_root/reasoning_root/verdict_hash and compare to bundle
	5.	If manifest.protocol_version mismatches prompt_spec.protocol_version: warn/error depending on strict mode

Return VerificationResult with CheckResult entries.

This is separate from Sentinel verify; it’s a packaging-level validation.

⸻

9) Interop Considerations

9.1 Forward compatibility

Manifest has format_version. When loading:
	•	reject unknown major version, warn for newer minor versions
	•	preserve unknown notes keys

9.2 Optional blobs

Do not store raw responses yet. Later you can add:
	•	blobs/ev_<id>.raw
	•	update manifest files list accordingly

⸻

10) Tests Required

Add:
	•	tests/unit/test_artifact_pack_io.py
	•	tests/unit/test_artifact_pack_validation.py

test_artifact_pack_io.py
	•	build a synthetic PoRPackage with small objects
	•	save as dir, load back, objects round-trip
	•	save as zip, load back, objects round-trip
	•	tamper a JSON file → hash validation fails

test_artifact_pack_validation.py
	•	make mismatched market_id across files → validation fails
	•	change verdict without updating por_bundle → validation fails

⸻

11) Acceptance Checklist
	•	Can save/load pack as directory and zip
	•	Manifest includes sha256 for all files
	•	Tampering triggers hash validation failure
	•	validate_pack checks semantic consistency (market_id + commitments)
	•	Unit tests pass; each file ≤ 500 LOC

⸻

Suggested Next Split (09C)
	•	Module 09C — CLI: cournot run, cournot pack, cournot verify
	•	Module 09D — Minimal API: /run, /pack, /verify, /challenge
