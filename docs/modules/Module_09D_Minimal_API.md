Module 09D — Minimal API (FastAPI)

Module ID: M09D
Owner Role: Backend Engineer
Goal: Expose a minimal HTTP API that can:
	1.	run the full pipeline and return artifacts (or return a pack)
	2.	verify an uploaded pack (offline verify-only)
	3.	optionally replay/verify (online)
	4.	return challenges when verification fails

No persistence, no background jobs, no websocket streaming.

⸻

1) Owned Files / Directories

Use your existing api/ tree:

api/
  __init__.py
  app.py
  models/
    __init__.py
    requests.py
    responses.py
  routes/
    __init__.py
    run.py
    verify.py
    replay.py
    health.py

Optional:
	•	api/deps.py (shared dependency wiring)
	•	api/errors.py (HTTP error mapping)

⸻

2) Dependencies
	•	fastapi, uvicorn (add to requirements if not already)
	•	pydantic
	•	Standard lib: io, json, zipfile, tempfile, pathlib

Internal modules:
	•	orchestrator.pipeline.Pipeline, PipelineConfig
	•	orchestrator.artifacts.io.save_pack_zip/load_pack_zip/load_pack_dir
	•	orchestrator.artifacts.pack.validate_pack
	•	agents.validator.SentinelAgent

⸻

3) API Principles (Strict v1)
	•	Stateless: each request is self-contained
	•	Deterministic: strict mode defaults on
	•	Artifact format: JSON objects match your schemas, plus optional pack.zip
	•	Error handling: always return structured error JSON

⸻

4) Endpoints

4.1 GET /health

Returns liveness.

Response

{ "ok": true, "service": "cournot-protocol-api", "version": "v1" }


⸻

4.2 POST /run

Runs the pipeline.

Request JSON
	•	user_input: str (required)
	•	strict_mode: bool = true
	•	enable_sentinel_verify: bool = true
	•	enable_replay: bool = false
	•	return_format: "json" | "pack_zip" = "json"
	•	include_checks: bool = false (include checks/challenges inline)

Response (return_format=“json”)
	•	run_result: minimal summary + optionally full artifacts
Recommended: include full artifacts under artifacts so clients can save packs themselves.

Response shape

{
  "ok": true,
  "summary": { "market_id": "...", "outcome": "YES", "confidence": 0.72, "por_root": "0x..." },
  "artifacts": {
    "prompt_spec": {...},
    "tool_plan": {...},
    "evidence_bundle": {...},
    "reasoning_trace": {...},
    "verdict": {...},
    "por_bundle": {...}
  },
  "verification": {
    "sentinel_ok": true,
    "checks": [],
    "challenges": []
  }
}

Response (return_format=“pack_zip”)
Return a zip file (binary) with Content-Type: application/zip containing the 09B artifact pack directory structure.
	•	For simplicity: return bytes directly using StreamingResponse.

Headers:
	•	X-Market-Id
	•	X-Por-Root
	•	X-Outcome
	•	X-Confidence

⸻

4.3 POST /verify

Verify an uploaded pack (zip) offline.

Request
multipart/form-data:
	•	file: pack zip (pack.zip)
	•	include_checks: bool (optional, default false)
	•	enable_sentinel: bool (default true)

Response JSON

{
  "ok": true,
  "hashes_ok": true,
  "semantic_ok": true,
  "sentinel_ok": true,
  "por_root": "0x...",
  "challenges": []
}

If fail, ok=false and challenges included (if sentinel enabled).

⸻

4.4 POST /replay

Replay evidence and verify (online).

Request
multipart/form-data:
	•	file: pack zip
	•	timeout_s: int = 30
	•	include_checks: bool = false

Response

{
  "ok": false,
  "replay_ok": false,
  "sentinel_ok": false,
  "por_root": "0x...",
  "divergence": { "expected_evidence_root": "...", "replayed_evidence_root": "..." },
  "challenges": [ ... ]
}


⸻

5) Request/Response Models (api/models)

requests.py

Define Pydantic models:
	•	RunRequest
	•	user_input: str
	•	strict_mode: bool = True
	•	enable_sentinel_verify: bool = True
	•	enable_replay: bool = False
	•	return_format: Literal["json","pack_zip"] = "json"
	•	include_checks: bool = False

responses.py
	•	RunSummary
	•	RunResponse
	•	VerifyResponse
	•	ReplayResponse
	•	ErrorResponse

Keep models thin; artifacts can be typed as dict if you don’t want Pydantic to re-validate everything.

⸻

6) Dependency Wiring (api/app.py + api/deps.py)

api/app.py
	•	create FastAPI app
	•	include routers: health, run, verify, replay

api/deps.py

Provide singleton-style factories:
	•	get_pipeline() -> Pipeline (configurable per request)
	•	get_sentinel() -> SentinelAgent

But do NOT keep global mutable state.

⸻

7) Implementation Notes

7.1 Run endpoint logic
	1.	Parse RunRequest
	2.	Build PipelineConfig from request flags
	3.	Run pipeline
	4.	If return JSON: serialize artifacts via .model_dump() or dict conversion
	5.	If pack_zip:
	•	create temp dir
	•	build PoRPackage
	•	save_pack_zip(...) to temp file
	•	return StreamingResponse of zip bytes

7.2 Verify/replay pack upload handling
	•	Save uploaded zip into temp dir
	•	load_pack_zip(..., verify_hashes=True)
	•	validate_pack()
	•	If sentinel enabled: SentinelAgent.verify(package, mode="verify")
	•	For replay: mode="replay" (or separate ReplayExecutor)

⸻

8) Error Handling Contract

Use consistent errors:

HTTP 400:
	•	invalid input
	•	missing file
	•	unsupported format

HTTP 500:
	•	internal runtime error

Error response:

{
  "ok": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "...",
    "details": {}
  }
}


⸻

9) Security / Limits (v1 minimal)

Add basic safeguards:
	•	max request size for upload (FastAPI/Uvicorn config)
	•	max user_input length (e.g., 8k chars)
	•	replay timeout
	•	disable replay by default

No authentication yet; keep it out of scope.

⸻

10) Tests Required

Add:
	•	tests/unit/test_api_smoke.py

Use FastAPI TestClient:
	1.	GET /health returns ok
	2.	POST /run returns JSON summary
	3.	POST /run return_format=pack_zip returns zip and headers
	4.	POST /verify with a known-good pack zip returns ok
	5.	Tampered pack returns ok=false and meaningful checks

If you don’t have a fixture pack yet, generate one in the test using Pipeline with stubbed sources.

⸻

11) Acceptance Checklist
	•	uvicorn api.app:app --reload starts
	•	/health works
	•	/run works and returns artifacts or pack.zip
	•	/verify checks hashes + commitments and returns challenges on failure
	•	/replay detects divergence (when possible)
	•	Unit tests pass; each file ≤ 500 LOC