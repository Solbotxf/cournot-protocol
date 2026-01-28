"""
Module 09D - Verify Route

Verify an uploaded artifact pack.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, Form

from api.models.responses import VerifyResponse
from api.errors import MissingFileError, InvalidPackError, InternalError

from orchestrator.artifacts.io import load_pack, validate_pack_files, PackIOError
from orchestrator.artifacts.pack import validate_pack
from orchestrator.pipeline import PoRPackage
from core.por.proof_of_reasoning import verify_por_bundle


router = APIRouter(tags=["verification"])


def run_hash_verification(pack_path: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Run file hash verification."""
    result = validate_pack_files(pack_path)
    checks = [
        {"check_id": c.check_id, "ok": c.ok, "message": c.message, "source": "hash"}
        for c in result.checks
    ]
    return result.ok, checks


def run_semantic_verification(package: PoRPackage) -> tuple[bool, list[dict[str, Any]]]:
    """Run semantic validation."""
    result = validate_pack(package)
    checks = [
        {"check_id": c.check_id, "ok": c.ok, "message": c.message, "source": "semantic"}
        for c in result.checks
    ]
    return result.ok, checks


def run_sentinel_verification(package: PoRPackage) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run sentinel verification."""
    result = verify_por_bundle(
        package.bundle,
        prompt_spec=package.prompt_spec,
        evidence=package.evidence,
        trace=package.trace,
    )
    
    checks = [
        {"check_id": c.check_id, "ok": c.ok, "message": c.message, "source": "sentinel"}
        for c in result.checks
    ]
    
    challenges = []
    if not result.ok and result.challenge:
        challenges.append({
            "kind": result.challenge.kind,
            "reason": result.challenge.reason,
            "evidence_id": result.challenge.evidence_id,
            "step_id": result.challenge.step_id,
        })
    
    return result.ok, checks, challenges


@router.post("/verify", response_model=VerifyResponse)
async def verify_pack(
    file: UploadFile = File(..., description="Artifact pack zip file"),
    include_checks: bool = Form(default=False, description="Include detailed checks"),
    enable_sentinel: bool = Form(default=True, description="Enable sentinel verification"),
) -> VerifyResponse:
    """
    Verify an uploaded artifact pack.
    
    Performs:
    1. File hash verification (manifest)
    2. Semantic validation (market_id consistency, commitments)
    3. Sentinel verification (optional)
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
        # Step 1: Hash verification
        hashes_ok, hash_checks = run_hash_verification(tmp_path)
        
        # Step 2: Load pack
        try:
            package = load_pack(tmp_path, verify_hashes=False)
        except PackIOError as e:
            raise InvalidPackError(f"Failed to load pack: {str(e)}")
        
        # Step 3: Semantic verification
        semantic_ok, semantic_checks = run_semantic_verification(package)
        
        # Step 4: Sentinel verification (optional)
        sentinel_ok = None
        sentinel_checks = []
        challenges = []
        
        if enable_sentinel:
            sentinel_ok, sentinel_checks, challenges = run_sentinel_verification(package)
        
        # Combine all checks
        all_checks = []
        if include_checks:
            all_checks = hash_checks + semantic_checks + sentinel_checks
        
        # Determine overall status
        overall_ok = hashes_ok and semantic_ok
        if sentinel_ok is not None:
            overall_ok = overall_ok and sentinel_ok
        
        return VerifyResponse(
            ok=overall_ok,
            hashes_ok=hashes_ok,
            semantic_ok=semantic_ok,
            sentinel_ok=sentinel_ok,
            por_root=package.bundle.por_root or "",
            market_id=package.bundle.market_id,
            outcome=package.verdict.outcome if package.verdict else "",
            checks=all_checks,
            challenges=challenges,
        )
    
    except InvalidPackError:
        raise
    except Exception as e:
        raise InternalError(f"Verification failed: {str(e)}")
    
    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)