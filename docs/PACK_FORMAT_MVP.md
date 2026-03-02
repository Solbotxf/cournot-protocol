# Cournot Artifact Pack Format (MVP)

This document defines the **minimum stable pack format** used by the off-chain dispute verifier (`cournot dispute verify`).

> Source of truth in code: `orchestrator/artifacts/io.py` and `orchestrator/artifacts/manifest.py`.

## 1) Supported container types

A pack can be provided as:

- a **directory** (e.g. `./pack_dir/`)
- a **zip file** (e.g. `./pack.zip`)

The zip file uses the **same internal relative paths** as the directory format.

## 2) Required files (MVP)

The pack must include these JSON files at the top level:

- `manifest.json`
- `prompt_spec.json`
- `evidence_bundle.json`
- `reasoning_trace.json`
- `verdict.json`
- `por_bundle.json`

Optional:

- `tool_plan.json`
- `checks/*_checks.json`

If any required file is missing, the dispute verifier reports `kind=pack_missing`.

## 3) manifest.json

`manifest.json` is used for:

- format version compatibility checks
- required-file presence checks
- SHA-256 integrity checks for each artifact file

If the manifest cannot be loaded/parsed, the dispute verifier reports `kind=manifest_invalid`.

## 4) PoR commitments and `por_root`

The dispute verifier does **not** trust hashes/roots blindly.

It loads the artifacts, then recomputes commitments using Module 03 helpers:

- `core/por/proof_of_reasoning.py::compute_prompt_spec_hash`
- `core/por/proof_of_reasoning.py::compute_evidence_root`
- `core/por/proof_of_reasoning.py::compute_reasoning_root`
- `core/por/proof_of_reasoning.py::compute_verdict_hash`
- `core/por/proof_of_reasoning.py::compute_por_root`
- and uses `core/por/proof_of_reasoning.py::verify_por_bundle` for consistency checks

Canonicalization is enforced by:

- `core/crypto/hashing.py::hash_canonical` (which uses the canonical JSON rules from Module 01)

Merkle ordering is **preserved**:

- Evidence leaf ordering: the order of `EvidenceBundle.items` (no sorting)
- Reasoning leaf ordering: the order of `ReasoningTrace.steps` (no sorting)

## 5) Future chain integration fields

The dispute verifier output includes fields intended for future on-chain integration:

- `market_id`
- `por_root`
- `pack_uri`
- `sentinel_version`
- `timestamp`

In the MVP, `pack_uri` is either the provided `--pack-uri` value or the user-supplied local path string.
