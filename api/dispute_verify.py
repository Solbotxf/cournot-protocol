"""Off-chain dispute verification (API-first).

This module provides a stable, reusable entrypoint for verifying an artifact pack
for dispute/challenge purposes.

Primary consumer: other Python code and the HTTP API layer in `api/`.
The CLI (`cournot dispute verify`) must be a thin wrapper around this module.

Output contract (stable JSON-compatible dict):
- ok
- por_root
- expected_por_root (optional)
- pack_uri
- sentinel_version
- timestamp
- challenge_ref { kind, leaf_index?, expected?, got?, reason? } (optional)
- errors[]
- checks[] (optional; best-effort)

ChallengeRef.kind (MVP enum):
- pack_missing
- manifest_invalid
- evidence_leaf
- reasoning_leaf
- verdict_hash
- por_bundle

leaf_index is best-effort; may be null with single-pack verification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.por.proof_of_reasoning import compute_roots, compute_verdict_hash, verify_por_bundle
from orchestrator.artifacts.io import (
    PackHashMismatchError,
    PackIOError,
    PackMissingFileError,
    load_pack,
    validate_pack_files,
)


ChallengeKind = str


@dataclass
class ChallengeRefOut:
    kind: ChallengeKind
    leaf_index: int | None = None
    expected: str | None = None
    got: str | None = None
    reason: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sentinel_version() -> str:
    # Keep stable and human-readable.
    return "SentinelStrict:v1"


def _best_effort_leaf_index_from_baseline(*, kind: str, pack: Any, baseline: Any) -> int | None:
    """Try to locate leaf_index by diffing against a baseline pack.

    This does NOT prove which leaf changed cryptographically; it's a usability
    enhancement for debugging.
    """
    try:
        if kind == "evidence_leaf":
            a = pack.evidence.items
            b = baseline.evidence.items
            if len(a) != len(b):
                return None
            for i, (ai, bi) in enumerate(zip(a, b)):
                if ai.model_dump(mode="json") != bi.model_dump(mode="json"):
                    return i
            return None

        if kind == "reasoning_leaf":
            a = pack.trace.steps
            b = baseline.trace.steps
            if len(a) != len(b):
                return None
            for i, (ai, bi) in enumerate(zip(a, b)):
                if ai.model_dump(mode="json") != bi.model_dump(mode="json"):
                    return i
            return None
    except Exception:
        return None

    return None


def verify_pack(
    pack_path_or_zip: str | Path,
    expected_por_root: str | None = None,
    pack_uri: str | None = None,
    *,
    baseline_pack: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an artifact pack for dispute purposes.

    Args:
        pack_path_or_zip: local path to pack dir or zip (MVP)
        expected_por_root: optional expected por_root to compare
        pack_uri: optional override for output pack_uri (e.g. ipfs://CID)
        baseline_pack: optional baseline pack for best-effort leaf_index localization

    Returns:
        JSON-compatible dict (stable output contract)
    """

    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    pack_input = str(pack_path_or_zip)
    uri = pack_uri or pack_input

    now = _utc_now_iso()

    pack_path = Path(pack_path_or_zip)
    if not pack_path.exists():
        return {
            "ok": False,
            "por_root": None,
            "expected_por_root": expected_por_root,
            "pack_uri": uri,
            "sentinel_version": _sentinel_version(),
            "timestamp": now,
            "challenge_ref": asdict(
                ChallengeRefOut(kind="pack_missing", reason=f"Pack not found: {pack_input}")
            ),
            "errors": [f"pack_missing: {pack_input}"],
            "checks": [],
        }

    # 1) Validate manifest + hashes (best-effort)
    hash_result = validate_pack_files(pack_path)
    checks.extend(
        [
            {
                "source": "manifest",
                "check_id": c.check_id,
                "ok": c.ok,
                "message": c.message,
                "details": c.details,
            }
            for c in hash_result.checks
        ]
    )
    if not hash_result.ok:
        errors.extend([c.message for c in hash_result.checks if not c.ok])

    # 2) Load pack
    try:
        pack = load_pack(pack_path, verify_hashes=False)
    except PackMissingFileError as e:
        return {
            "ok": False,
            "por_root": None,
            "expected_por_root": expected_por_root,
            "pack_uri": uri,
            "sentinel_version": _sentinel_version(),
            "timestamp": now,
            "challenge_ref": asdict(ChallengeRefOut(kind="pack_missing", reason=str(e))),
            "errors": errors + [str(e)],
            "checks": checks,
        }
    except (PackHashMismatchError, PackIOError) as e:
        return {
            "ok": False,
            "por_root": None,
            "expected_por_root": expected_por_root,
            "pack_uri": uri,
            "sentinel_version": _sentinel_version(),
            "timestamp": now,
            "challenge_ref": asdict(ChallengeRefOut(kind="manifest_invalid", reason=str(e))),
            "errors": errors + [str(e)],
            "checks": checks,
        }

    # 3) Recompute roots and verify PoR bundle
    roots = compute_roots(pack.prompt_spec, pack.evidence, pack.trace, pack.verdict)
    por_root = pack.bundle.por_root or roots.por_root

    verify_res = verify_por_bundle(
        pack.bundle,
        prompt_spec=pack.prompt_spec,
        evidence=pack.evidence,
        trace=pack.trace,
    )
    checks.extend(
        [
            {
                "source": "por",
                "check_id": c.check_id,
                "ok": c.ok,
                "message": c.message,
                "details": c.details,
            }
            for c in verify_res.checks
        ]
    )

    # Required extra check: verdict.json must match bundle.verdict_hash
    external_verdict_hash = compute_verdict_hash(pack.verdict)
    verdict_artifact_ok = external_verdict_hash == pack.bundle.verdict_hash
    if not verdict_artifact_ok:
        errors.append("verdict.json hash mismatch vs por_bundle.verdict_hash")

    challenge: ChallengeRefOut | None = None

    if not verdict_artifact_ok:
        challenge = ChallengeRefOut(
            kind="verdict_hash",
            expected=pack.bundle.verdict_hash,
            got=external_verdict_hash,
            reason="verdict.json mismatch with por_bundle.verdict_hash",
        )

    if (not verify_res.ok) and challenge is None:
        kind = verify_res.challenge.kind if verify_res.challenge is not None else "por_bundle"

        leaf_index = None
        if baseline_pack is not None:
            bp = Path(baseline_pack)
            if bp.exists():
                try:
                    baseline = load_pack(bp, verify_hashes=False)
                    leaf_index = _best_effort_leaf_index_from_baseline(
                        kind=kind, pack=pack, baseline=baseline
                    )
                except Exception:
                    leaf_index = None

        if leaf_index is None and kind in ("evidence_leaf", "reasoning_leaf"):
            errors.append(
                "leaf_index unavailable with single-pack verification; provide baseline_pack to improve localization"
            )

        challenge = ChallengeRefOut(
            kind=kind,
            leaf_index=leaf_index,
            reason=verify_res.challenge.reason if verify_res.challenge else "verification failed",
        )

    if expected_por_root is not None and por_root != expected_por_root:
        errors.append("expected_por_root mismatch")
        if challenge is None:
            challenge = ChallengeRefOut(
                kind="por_bundle",
                expected=expected_por_root,
                got=por_root,
                reason="expected_por_root mismatch",
            )
        else:
            challenge.expected = challenge.expected or expected_por_root
            challenge.got = challenge.got or por_root

    ok = (
        challenge is None
        and hash_result.ok
        and verify_res.ok
        and verdict_artifact_ok
        and (expected_por_root is None or por_root == expected_por_root)
    )

    out: dict[str, Any] = {
        "ok": ok,
        "market_id": pack.bundle.market_id,
        "por_root": por_root,
        "expected_por_root": expected_por_root,
        "pack_uri": uri,
        "sentinel_version": _sentinel_version(),
        "timestamp": now,
        "errors": errors,
        "checks": checks,
    }
    if challenge is not None:
        out["challenge_ref"] = asdict(challenge)

    return out
