Module 09C — CLI

Module ID: M09C
Owner Role: Tooling/DevEx Engineer
Goal: Provide a developer-friendly command line interface to:
	1.	run the full pipeline on a user query
	2.	save artifact packs (dir/zip)
	3.	verify packs (offline)
	4.	optionally replay evidence + verify (online)

⸻

1) Owned Files

Add a small CLI package:

cournot_cli/
  __init__.py
  main.py
  commands/
    __init__.py
    run.py
    pack.py
    verify.py
    replay.py

And add an entrypoint in pyproject.toml later (optional). If you’re not using pyproject, keep runnable via:
	•	python -m cournot_cli ...

If you don’t want a new top-level package, you can put CLI under orchestrator/cli/, but keep it isolated from core logic.

⸻

2) Dependencies
	•	Standard library only preferred: argparse, json, pathlib, sys, textwrap
	•	Use your internal modules:
	•	orchestrator.pipeline.Pipeline, PipelineConfig
	•	orchestrator.artifacts.io.save_pack_dir/save_pack_zip/load_pack
	•	orchestrator.artifacts.pack.validate_pack
	•	agents.validator.sentinel_agent.SentinelAgent
	•	agents.validator.replay_executor.ReplayExecutor (optional replay)
	•	Optional: rich for pretty output (avoid for v1; keep simple)

⸻

3) CLI Commands (Public UX Contract)

The CLI tool name in examples: cournot. If you don’t install a console script, users can run python -m cournot_cli.

3.1 run

Runs full pipeline and optionally writes an artifact pack.

Command

cournot run "<user_input>" --out pack.zip

Flags
	•	--out PATH optional: if provided, save pack (zip if .zip, otherwise directory)
	•	--no-sentinel default false: disable sentinel verify
	•	--replay default false: sentinel replay mode (network)
	•	--strict/--no-strict default strict
	•	--json print machine-readable summary JSON to stdout
	•	--debug print expanded checks

Output
	•	prints: market_id, outcome, confidence, por_root, saved path (if any)
	•	exit code:
	•	0 = pipeline executed
	•	2 = verification failed (sentinel/validation), but run executed
	•	1 = runtime error

Note: outcome INVALID is still exit code 0 (it’s a valid resolution), unless verification fails.

⸻

3.2 pack

Create a pack from an existing set of artifact JSON files (advanced).

Command

cournot pack --from-dir ./artifacts --out pack.zip

Flags
	•	--from-dir PATH contains required JSON files
	•	--out PATH zip or dir

This is optional; if you don’t need it, you can omit pack command entirely and rely on run --out.

⸻

3.3 verify

Offline verify a pack:
	•	verify file hashes (manifest)
	•	validate semantic consistency (09B validate_pack)
	•	sentinel verify-only mode (08)

Command

cournot verify pack.zip

Flags
	•	--no-sentinel only do pack-level validation
	•	--json output verification report JSON
	•	--debug include detailed checks and challenge objects

Exit codes
	•	0 all checks ok
	•	2 verification failed (invalid hashes / invalid commitments)
	•	1 runtime error

⸻

3.4 replay

Replay evidence and compare (online). Uses sentinel replay + collector.

Command

cournot replay pack.zip

Flags
	•	--timeout 30
	•	--json
	•	--debug

Exit codes
	•	0 replay matches provided evidence (or no divergence detected)
	•	2 divergence detected / replay verification failed
	•	1 runtime error

⸻

4) Data Models for CLI Output

4.1 RunSummary (dict or dataclass)
	•	market_id
	•	outcome
	•	confidence
	•	por_root
	•	prompt_spec_hash (optional)
	•	evidence_root (optional)
	•	reasoning_root (optional)
	•	saved_to (optional)
	•	ok (pipeline ok)
	•	verification_ok (sentinel ok)
	•	checks (optional list when –debug)

4.2 VerifySummary
	•	pack_path
	•	hashes_ok
	•	semantic_ok
	•	sentinel_ok (if enabled)
	•	por_root
	•	challenges (optional when –debug)

⸻

5) Implementation Requirements

5.1 cournot_cli/main.py
	•	parse argv
	•	dispatch subcommands
	•	unify exit codes
	•	handle exceptions cleanly (print concise error to stderr; use non-zero exit)

5.2 Command modules

commands/run.py
Implement:
	•	run_cmd(args) -> int
Process:

	1.	instantiate PipelineConfig from flags
	2.	Pipeline(config).run(user_input)
	3.	if --out, save pack using 09B IO
	4.	print summary (human or JSON)

commands/verify.py
Implement:
	•	verify_cmd(args) -> int
Process:

	1.	load pack (verify hashes)
	2.	validate_pack (semantic)
	3.	sentinel verify-only unless --no-sentinel
	4.	print summary, including challenges if debug

commands/replay.py
Implement:
	•	replay_cmd(args) -> int
Process:

	1.	load pack (verify hashes)
	2.	sentinel verify replay mode:
	•	create collector + replay executor
	3.	print divergence details, challenges if any

commands/pack.py (optional)
	•	create PoRPackage from JSON files in a directory
	•	then save to out path and produce manifest

⸻

6) Wiring to Existing Modules

Pipeline integration (09A)

run should call:
	•	Pipeline.run(user_input) and receive RunResult
If RunResult already includes por_bundle and artifacts, build PoRPackage easily.

Pack IO (09B)
	•	if out endswith .zip → save_pack_zip(...) else save_pack_dir(...)
	•	verify loads with load_pack(..., verify_hashes=True)

Sentinel (08)
	•	verify uses SentinelAgent.verify(package, mode="verify")
	•	replay uses SentinelAgent.verify(package, mode="replay") OR manually:
	•	ReplayExecutor.replay_evidence(...) then compare roots

Prefer routing through Sentinel for consistency.

⸻

7) Exit Code Policy (Strict)

Use consistent exit codes:
	•	0: command succeeded; if “run”, pipeline executed; verification (if performed) passed
	•	2: command succeeded but verification failed (hash mismatch / commitment mismatch / replay divergence)
	•	1: runtime error / invalid CLI usage

⸻

8) Minimal Printing Format (Human)

run prints

market_id: mk_xxxxx
outcome: YES
confidence: 0.72
por_root: 0x...
saved: ./pack.zip
sentinel_ok: true

verify prints

pack: ./pack.zip
hashes_ok: true
semantic_ok: true
sentinel_ok: true
por_root: 0x...

With --debug print checks count and first N check messages.

⸻

9) Tests Required

Add:
	•	tests/unit/test_cli_smoke.py

Use subprocess to run:
	•	python -m cournot_cli run "..." --json
	•	python -m cournot_cli verify ... --json

If subprocess tests are annoying, test command functions directly by passing an argparse-like object.

At minimum:
	•	verify command returns 0 for a known-good sample pack (you can create one in test fixtures)
	•	verify returns 2 if manifest hash mismatch (tamper file in temp dir)

⸻

10) Acceptance Checklist
	•	python -m cournot_cli run "<query>" --out pack.zip works
	•	python -m cournot_cli verify pack.zip works offline
	•	python -m cournot_cli replay pack.zip detects divergence (with mocks or known endpoints)
	•	Exit codes follow policy
	•	Each file ≤ 500 LOC
