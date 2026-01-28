"""
Module 09C - CLI Verify Command (Production)

Verify an artifact pack offline:
- Verify file hashes (manifest)
- Validate semantic consistency
- Optionally run sentinel verification

Usage:
    cournot verify pack.zip [--no-sentinel] [--json] [--debug]
"""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from orchestrator.artifacts.io import load_pack, validate_pack_files, PackIOError
from orchestrator.artifacts.pack import validate_pack
from orchestrator.pipeline import PoRPackage
from core.por.proof_of_reasoning import verify_por_bundle
from core.schemas.verification import VerificationResult


logger = logging.getLogger(__name__)


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VERIFICATION_FAILED = 2


@dataclass
class VerifySummary:
    """Summary of pack verification for CLI output."""
    pack_path: str = ""
    por_root: str = ""
    hashes_ok: bool = False
    semantic_ok: bool = False
    sentinel_ok: bool | None = None
    market_id: str = ""
    outcome: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    challenges: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.sentinel_ok is None:
            del d["sentinel_ok"]
        if not d["checks"]:
            del d["checks"]
        if not d["challenges"]:
            del d["challenges"]
        if not d["errors"]:
            del d["errors"]
        return d
    
    @property
    def all_ok(self) -> bool:
        """Check if all verifications passed."""
        if not self.hashes_ok or not self.semantic_ok:
            return False
        if self.sentinel_ok is not None and not self.sentinel_ok:
            return False
        return True


def verify_hashes(pack_path: Path) -> tuple[bool, VerificationResult]:
    """Verify file hashes in the pack."""
    logger.info(f"Verifying file hashes for: {pack_path}")
    result = validate_pack_files(pack_path)
    return result.ok, result


def verify_semantic(package: PoRPackage) -> tuple[bool, VerificationResult]:
    """Verify semantic consistency of the pack."""
    logger.info("Verifying semantic consistency...")
    result = validate_pack(package)
    return result.ok, result


def verify_sentinel(package: PoRPackage) -> tuple[bool, VerificationResult, list[dict]]:
    """Run sentinel verification on the pack."""
    logger.info("Running sentinel verification...")
    result = verify_por_bundle(
        package.bundle,
        prompt_spec=package.prompt_spec,
        evidence=package.evidence,
        trace=package.trace,
    )
    
    challenges = []
    if not result.ok and result.challenge:
        challenges.append({
            "kind": result.challenge.kind,
            "reason": result.challenge.reason,
            "evidence_id": result.challenge.evidence_id,
            "step_id": result.challenge.step_id,
        })
    
    return result.ok, result, challenges


def build_summary(
    pack_path: str,
    package: PoRPackage,
    hash_result: VerificationResult,
    semantic_result: VerificationResult,
    sentinel_result: VerificationResult | None = None,
    sentinel_challenges: list[dict] | None = None,
    debug: bool = False,
) -> VerifySummary:
    """Build a VerifySummary from verification results."""
    summary = VerifySummary(
        pack_path=pack_path,
        por_root=package.bundle.por_root or "",
        hashes_ok=hash_result.ok,
        semantic_ok=semantic_result.ok,
        market_id=package.bundle.market_id,
        outcome=package.verdict.outcome if package.verdict else "",
    )
    
    if sentinel_result is not None:
        summary.sentinel_ok = sentinel_result.ok
    
    if sentinel_challenges:
        summary.challenges = sentinel_challenges
    
    # Collect errors
    if not hash_result.ok:
        for check in hash_result.checks:
            if not check.ok:
                summary.errors.append(f"Hash: {check.message}")
    
    if not semantic_result.ok:
        for check in semantic_result.checks:
            if not check.ok:
                summary.errors.append(f"Semantic: {check.message}")
    
    if sentinel_result and not sentinel_result.ok:
        for check in sentinel_result.checks:
            if not check.ok:
                summary.errors.append(f"Sentinel: {check.message}")
    
    # Include detailed checks if debug
    if debug:
        all_checks = []
        for check in hash_result.checks:
            all_checks.append({
                "source": "hash",
                "check_id": check.check_id,
                "ok": check.ok,
                "message": check.message,
            })
        for check in semantic_result.checks:
            all_checks.append({
                "source": "semantic",
                "check_id": check.check_id,
                "ok": check.ok,
                "message": check.message,
            })
        if sentinel_result:
            for check in sentinel_result.checks:
                all_checks.append({
                    "source": "sentinel",
                    "check_id": check.check_id,
                    "ok": check.ok,
                    "message": check.message,
                })
        summary.checks = all_checks
    
    return summary


def print_summary_human(summary: VerifySummary) -> None:
    """Print summary in human-readable format."""
    print(f"pack: {summary.pack_path}")
    print(f"market_id: {summary.market_id}")
    print(f"outcome: {summary.outcome}")
    print(f"por_root: {summary.por_root}")
    print(f"hashes_ok: {str(summary.hashes_ok).lower()}")
    print(f"semantic_ok: {str(summary.semantic_ok).lower()}")
    if summary.sentinel_ok is not None:
        print(f"sentinel_ok: {str(summary.sentinel_ok).lower()}")
    
    if summary.errors:
        print(f"\nerrors ({len(summary.errors)}):")
        for err in summary.errors[:10]:
            print(f"  ✗ {err}")
    
    if summary.challenges:
        print(f"\nchallenges ({len(summary.challenges)}):")
        for ch in summary.challenges:
            print(f"  - {ch.get('kind', 'unknown')}: {ch.get('reason', '')}")
    
    if summary.checks:
        passed = sum(1 for c in summary.checks if c["ok"])
        failed = len(summary.checks) - passed
        print(f"\nchecks: {passed} passed, {failed} failed")
        for check in summary.checks[:20]:
            status = "✓" if check["ok"] else "✗"
            print(f"  {status} [{check['source']}] {check['check_id']}")


def print_summary_json(summary: VerifySummary) -> None:
    """Print summary as JSON."""
    print(json.dumps(summary.to_dict(), indent=2))


def verify_cmd(args: Namespace) -> int:
    """
    Execute the verify command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code
    """
    pack_path = Path(args.pack_path)
    no_sentinel = args.no_sentinel
    output_json = args.json
    debug = args.debug
    
    # Check pack exists
    if not pack_path.exists():
        print(f"Error: Pack not found: {pack_path}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Step 1: Verify file hashes
    try:
        hashes_ok, hash_result = verify_hashes(pack_path)
    except Exception as e:
        if debug:
            raise
        print(f"Error verifying hashes: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Step 2: Load pack (will fail if hashes don't match with verify_hashes=True)
    try:
        # Load without hash verification since we already did it
        package = load_pack(pack_path, verify_hashes=False)
    except PackIOError as e:
        print(f"Error loading pack: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if debug:
            raise
        print(f"Error loading pack: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Step 3: Verify semantic consistency
    try:
        semantic_ok, semantic_result = verify_semantic(package)
    except Exception as e:
        if debug:
            raise
        print(f"Error verifying semantics: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Step 4: Sentinel verification (unless disabled)
    sentinel_result = None
    sentinel_challenges = None
    if not no_sentinel:
        try:
            sentinel_ok, sentinel_result, sentinel_challenges = verify_sentinel(package)
        except Exception as e:
            if debug:
                raise
            print(f"Error in sentinel verification: {e}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
    
    # Build and print summary
    summary = build_summary(
        pack_path=str(pack_path),
        package=package,
        hash_result=hash_result,
        semantic_result=semantic_result,
        sentinel_result=sentinel_result,
        sentinel_challenges=sentinel_challenges,
        debug=debug,
    )
    
    if output_json:
        print_summary_json(summary)
    else:
        print_summary_human(summary)
    
    # Log completion
    if summary.all_ok:
        logger.info("Verification passed")
    else:
        logger.warning("Verification failed")
    
    # Determine exit code
    if summary.all_ok:
        return EXIT_SUCCESS
    else:
        return EXIT_VERIFICATION_FAILED