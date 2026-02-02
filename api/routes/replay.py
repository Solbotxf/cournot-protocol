"""
Module 09D - Replay Route

Replay evidence collection and verify against original pack.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, Form

from api.models.responses import ReplayResponse, DivergenceInfo
from api.deps import get_verification_context
from api.errors import MissingFileError, InvalidPackError, InternalError

from orchestrator.artifacts.io import load_pack, PackIOError
from orchestrator.pipeline import PoRPackage
from core.por.proof_of_reasoning import compute_evidence_root
from agents.sentinel import verify_artifacts


logger = logging.getLogger(__name__)

router = APIRouter(tags=["replay"])


def replay_evidence(package: PoRPackage, timeout_s: int) -> tuple[bool, str, list[dict[str, Any]]]:
    """
    Replay evidence collection and compare with original.
    
    In production, this would:
    1. Re-fetch evidence from original sources using retrieval receipts
    2. Compare with evidence in the package
    3. Report divergences
    
    For now, we verify the evidence structure and hashes match.
    
    Returns:
        Tuple of (matches, replayed_root, divergences)
    """
    logger.info(f"Replaying evidence (timeout={timeout_s}s)...")
    
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
    
    # Check individual evidence items for completeness
    for item in package.evidence.items:
        # Check for missing content
        if item.raw_content is None and item.success:
            divergences.append({
                "type": "missing_content",
                "evidence_id": item.evidence_id,
                "reason": "Evidence item marked successful but has no content",
            })
        
        # Check for missing provenance
        if item.provenance is None:
            divergences.append({
                "type": "missing_provenance",
                "evidence_id": item.evidence_id,
                "reason": "Evidence item has no provenance record",
            })
        elif item.provenance.content_hash is None and item.success:
            divergences.append({
                "type": "missing_hash",
                "evidence_id": item.evidence_id,
                "reason": "Evidence item has no content hash",
            })
    
    logger.info(f"Replay complete: {len(divergences)} divergences found")
    
    return len(divergences) == 0, computed_root, divergences


def run_sentinel_verification(
    package: PoRPackage,
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Run sentinel verification in replay mode.
    
    Returns:
        Tuple of (ok, checks, challenges, errors)
    """
    ctx = get_verification_context()
    
    result = verify_artifacts(
        ctx,
        prompt_spec=package.prompt_spec,
        tool_plan=package.tool_plan,
        evidence_bundle=package.evidence,
        reasoning_trace=package.trace,
        verdict=package.verdict,
        execution_log=None,
        strict=True,
    )
    
    if not result.success:
        return False, [], [], [result.error or "Verification failed"]
    
    verification_result, report = result.output
    
    # Build checks from report
    checks = []
    for cat_name in ["completeness_checks", "hash_checks", "consistency_checks", 
                     "provenance_checks", "reasoning_checks"]:
        cat_checks = getattr(report, cat_name, [])
        for check in cat_checks:
            checks.append({
                "check_id": check.get("check_id", "unknown"),
                "ok": check.get("passed", False),
                "message": check.get("message", ""),
                "source": "sentinel",
            })
    
    # Build challenges from errors
    challenges = []
    if not report.verified and report.errors:
        for error in report.errors:
            challenges.append({
                "kind": "verification_error",
                "reason": error,
            })
    
    return report.verified, checks, challenges, report.errors or []


@router.post("/replay", response_model=ReplayResponse)
async def replay_pack(
    file: UploadFile = File(..., description="Artifact pack zip file"),
    timeout_s: int = Form(default=30, ge=1, le=300, description="Timeout in seconds"),
    include_checks: bool = Form(default=False, description="Include detailed checks"),
) -> ReplayResponse:
    """
    Replay evidence and verify against original pack.
    
    Re-collects evidence from sources and compares with the evidence
    stored in the pack to detect any divergence.
    """
    if file is None or file.filename == "":
        raise MissingFileError("No file uploaded")
    
    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise InternalError(f"Failed to save uploaded file: {str(e)}")
    
    try:
        # Load pack with hash verification
        logger.info(f"Loading pack: {file.filename}")
        try:
            package = load_pack(tmp_path, verify_hashes=True)
        except PackIOError as e:
            raise InvalidPackError(f"Failed to load pack: {str(e)}")
        
        # Replay evidence
        replay_ok, replayed_root, divergences = replay_evidence(package, timeout_s)
        
        # Sentinel verification
        logger.info("Running sentinel verification...")
        sentinel_ok, sentinel_checks, challenges, errors = run_sentinel_verification(package)
        
        # Build divergence info if there are divergences
        divergence_info = None
        if not replay_ok or divergences:
            divergence_info = DivergenceInfo(
                expected_evidence_root=package.bundle.evidence_root,
                replayed_evidence_root=replayed_root,
                divergences=divergences,
            )
        
        # Combine checks
        all_checks = []
        if include_checks:
            all_checks = sentinel_checks
        
        # Overall status
        overall_ok = replay_ok and sentinel_ok
        
        return ReplayResponse(
            ok=overall_ok,
            replay_ok=replay_ok,
            sentinel_ok=sentinel_ok,
            por_root=package.bundle.por_root or "",
            market_id=package.bundle.market_id,
            divergence=divergence_info,
            checks=all_checks,
            challenges=challenges,
            errors=errors,
        )
    
    except InvalidPackError:
        raise
    except Exception as e:
        logger.exception("Replay failed")
        raise InternalError(f"Replay failed: {str(e)}")
    
    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)
