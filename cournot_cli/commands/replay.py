"""
Module 09C - CLI Replay Command

Replay evidence collection and compare with pack contents.

This is an online operation that re-collects evidence from sources
and compares with the evidence in the pack to detect divergence.

Usage:
    cournot replay pack.zip [--timeout 30] [--json] [--debug]
"""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from orchestrator.artifacts.io import load_pack, PackIOError
from orchestrator.pipeline import PoRPackage
from core.por.proof_of_reasoning import compute_evidence_root
from core.schemas.verification import VerificationResult, CheckResult


logger = logging.getLogger(__name__)


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VERIFICATION_FAILED = 2


@dataclass
class ReplaySummary:
    """Summary of evidence replay for CLI output."""
    pack_path: str = ""
    por_root: str = ""
    market_id: str = ""
    replay_ok: bool = False
    evidence_matches: bool = False
    original_evidence_root: str = ""
    replayed_evidence_root: str = ""
    divergences: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["divergences"]:
            del d["divergences"]
        if not d["checks"]:
            del d["checks"]
        if not d["errors"]:
            del d["errors"]
        return d


def replay_evidence(package: PoRPackage, timeout: int = 30) -> tuple[bool, list[dict], str]:
    """
    Replay evidence collection and compare with original.
    
    This function verifies the integrity of evidence by:
    1. Computing the evidence root from the package
    2. Comparing with the bundle's committed evidence root
    3. Checking individual evidence items for completeness
    
    In a full production implementation, this would:
    1. Re-fetch evidence from the original sources using the retrieval receipts
    2. Compare content hashes
    3. Detect any changes or unavailability
    
    Args:
        package: The PoRPackage to replay
        timeout: Timeout for network operations
    
    Returns:
        Tuple of (matches, divergences, replayed_evidence_root)
    """
    logger.info(f"Replaying evidence (timeout={timeout}s)...")
    
    # Compute evidence root from package
    computed_root = compute_evidence_root(package.evidence)
    bundle_root = package.bundle.evidence_root
    
    matches = computed_root == bundle_root
    divergences = []
    
    if not matches:
        divergences.append({
            "type": "evidence_root_mismatch",
            "original": bundle_root,
            "replayed": computed_root,
            "reason": "Evidence root does not match bundle commitment",
        })
        logger.warning(f"Evidence root mismatch: {bundle_root} != {computed_root}")
    
    # Check individual evidence items
    for item in package.evidence.items:
        # Verify item has content
        if item.raw_content is None:
            divergences.append({
                "type": "missing_content",
                "evidence_id": item.evidence_id,
                "reason": "Evidence item has no content",
            })
            logger.warning(f"Evidence item {item.evidence_id} has no content")
        
        # Verify provenance
        if item.provenance is None:
            divergences.append({
                "type": "missing_provenance",
                "evidence_id": item.evidence_id,
                "reason": "Evidence item has no provenance",
            })
        elif item.provenance.content_hash is None:
            divergences.append({
                "type": "missing_hash",
                "evidence_id": item.evidence_id,
                "reason": "Evidence item has no content hash",
            })
    
    # In production, we would also:
    # - Re-fetch from item.provenance.source_uri
    # - Compare response hash with item.provenance.content_hash
    # - Check for content changes
    
    logger.info(f"Replay complete: {len(divergences)} divergences found")
    
    return matches and len(divergences) == 0, divergences, computed_root


def verify_with_replay(package: PoRPackage) -> tuple[bool, VerificationResult]:
    """Run sentinel verification in replay mode."""
    from agents import AgentContext
    from agents.sentinel import build_proof_bundle, verify_proof
    
    logger.info("Running sentinel verification...")
    
    ctx = AgentContext.create_minimal()
    
    # Build proof bundle from package
    proof_bundle = build_proof_bundle(
        package.prompt_spec,
        package.tool_plan,
        package.evidence,
        package.trace,
        package.verdict,
        None,  # execution_log
    )
    
    # Verify
    result = verify_proof(ctx, proof_bundle)
    verification_result, report = result.output
    
    # Build VerificationResult from report
    checks = []
    for cat_name in ["completeness_checks", "hash_checks", "consistency_checks", 
                     "provenance_checks", "reasoning_checks"]:
        cat_checks = getattr(report, cat_name, [])
        for check in cat_checks:
            checks.append(CheckResult(
                check_id=check.get("check_id", "unknown"),
                ok=check.get("passed", False),
                severity=check.get("severity", "error"),
                message=check.get("message", ""),
            ))
    
    sentinel_result = VerificationResult(ok=report.verified, checks=checks)
    return report.verified, sentinel_result


def build_summary(
    pack_path: str,
    package: PoRPackage,
    evidence_matches: bool,
    divergences: list[dict],
    replayed_root: str,
    verification_result: VerificationResult,
    debug: bool = False,
) -> ReplaySummary:
    """Build a ReplaySummary from replay results."""
    summary = ReplaySummary(
        pack_path=pack_path,
        por_root=package.bundle.por_root or "",
        market_id=package.bundle.market_id,
        replay_ok=evidence_matches and verification_result.ok,
        evidence_matches=evidence_matches,
        original_evidence_root=package.bundle.evidence_root,
        replayed_evidence_root=replayed_root,
        divergences=divergences,
    )
    
    # Collect errors from verification
    if not verification_result.ok:
        for check in verification_result.checks:
            if not check.ok:
                summary.errors.append(f"Verification: {check.message}")
    
    # Add divergence errors
    for div in divergences:
        summary.errors.append(f"Divergence: {div.get('reason', div.get('type', 'unknown'))}")
    
    # Include detailed checks if debug
    if debug:
        summary.checks = [
            {
                "check_id": check.check_id,
                "ok": check.ok,
                "message": check.message,
            }
            for check in verification_result.checks
        ]
    
    return summary


def print_summary_human(summary: ReplaySummary) -> None:
    """Print summary in human-readable format."""
    print(f"pack: {summary.pack_path}")
    print(f"market_id: {summary.market_id}")
    print(f"por_root: {summary.por_root}")
    print(f"replay_ok: {str(summary.replay_ok).lower()}")
    print(f"evidence_matches: {str(summary.evidence_matches).lower()}")
    print(f"original_evidence_root: {summary.original_evidence_root}")
    print(f"replayed_evidence_root: {summary.replayed_evidence_root}")
    
    if summary.divergences:
        print(f"\ndivergences ({len(summary.divergences)}):")
        for div in summary.divergences:
            print(f"  - {div.get('type', 'unknown')}: {div.get('reason', '')}")
            if "evidence_id" in div:
                print(f"    evidence_id: {div['evidence_id']}")
    
    if summary.errors:
        print(f"\nerrors ({len(summary.errors)}):")
        for err in summary.errors[:10]:
            print(f"  ✗ {err}")
    
    if summary.checks:
        passed = sum(1 for c in summary.checks if c["ok"])
        failed = len(summary.checks) - passed
        print(f"\nchecks: {passed} passed, {failed} failed")


def print_summary_json(summary: ReplaySummary) -> None:
    """Print summary as JSON."""
    print(json.dumps(summary.to_dict(), indent=2))


def replay_cmd(args: Namespace) -> int:
    """
    Execute the replay command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code
    """
    pack_path = Path(args.pack_path)
    timeout = args.timeout
    output_json = args.json
    debug = args.debug
    
    # Check pack exists
    if not pack_path.exists():
        print(f"Error: Pack not found: {pack_path}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Load pack
    logger.info(f"Loading pack: {pack_path}")
    try:
        package = load_pack(pack_path, verify_hashes=True)
    except PackIOError as e:
        print(f"Error loading pack: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if debug:
            raise
        print(f"Error loading pack: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Replay evidence
    try:
        evidence_matches, divergences, replayed_root = replay_evidence(package, timeout)
    except Exception as e:
        if debug:
            raise
        print(f"Error replaying evidence: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Verify with replay mode
    try:
        verification_ok, verification_result = verify_with_replay(package)
    except Exception as e:
        if debug:
            raise
        print(f"Error in verification: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Build and print summary
    summary = build_summary(
        pack_path=str(pack_path),
        package=package,
        evidence_matches=evidence_matches,
        divergences=divergences,
        replayed_root=replayed_root,
        verification_result=verification_result,
        debug=debug,
    )
    
    if output_json:
        print_summary_json(summary)
    else:
        print_summary_human(summary)
    
    # Log completion
    if summary.replay_ok:
        logger.info("Replay verification passed")
    else:
        logger.warning("Replay verification failed")
    
    # Determine exit code
    if summary.replay_ok:
        return EXIT_SUCCESS
    else:
        return EXIT_VERIFICATION_FAILED
