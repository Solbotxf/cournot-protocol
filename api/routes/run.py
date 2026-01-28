"""
Module 09D - Run Route

Execute the pipeline and return results.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models.requests import RunRequest
from api.models.responses import RunResponse, RunSummary, VerificationInfo
from api.deps import get_pipeline
from api.errors import InternalError

from orchestrator.pipeline import PoRPackage, RunResult
from orchestrator.artifacts.io import save_pack_zip


router = APIRouter(tags=["pipeline"])


def build_summary(result: RunResult) -> RunSummary:
    """Build RunSummary from pipeline result."""
    return RunSummary(
        market_id=result.market_id or "",
        outcome=result.outcome or "",
        confidence=result.verdict.confidence if result.verdict else 0.0,
        por_root=result.por_bundle.por_root if result.por_bundle else "",
        prompt_spec_hash=result.roots.prompt_spec_hash if result.roots else None,
        evidence_root=result.roots.evidence_root if result.roots else None,
        reasoning_root=result.roots.reasoning_root if result.roots else None,
    )


def build_artifacts(result: RunResult) -> dict[str, Any]:
    """Build artifacts dict from pipeline result."""
    artifacts = {}
    
    if result.prompt_spec:
        artifacts["prompt_spec"] = result.prompt_spec.model_dump(mode="json")
    if result.tool_plan:
        artifacts["tool_plan"] = result.tool_plan.model_dump(mode="json")
    if result.evidence_bundle:
        artifacts["evidence_bundle"] = result.evidence_bundle.model_dump(mode="json")
    if result.audit_trace:
        artifacts["reasoning_trace"] = result.audit_trace.model_dump(mode="json")
    if result.verdict:
        artifacts["verdict"] = result.verdict.model_dump(mode="json")
    if result.por_bundle:
        artifacts["por_bundle"] = result.por_bundle.model_dump(mode="json")
    
    return artifacts


def build_verification(result: RunResult, include_checks: bool) -> VerificationInfo | None:
    """Build verification info from pipeline result."""
    if result.sentinel_verification is None:
        return None
    
    checks = []
    if include_checks and result.checks:
        checks = [
            {"check_id": c.check_id, "ok": c.ok, "message": c.message}
            for c in result.checks
        ]
    
    challenges = []
    if result.challenges:
        challenges = result.challenges
    
    return VerificationInfo(
        sentinel_ok=result.sentinel_verification.ok,
        checks=checks,
        challenges=challenges,
    )


def build_package(result: RunResult) -> PoRPackage | None:
    """Build PoRPackage from pipeline result."""
    if result.por_bundle is None:
        return None
    
    return PoRPackage(
        bundle=result.por_bundle,
        prompt_spec=result.prompt_spec,
        tool_plan=result.tool_plan,
        evidence=result.evidence_bundle,
        trace=result.audit_trace,
        verdict=result.verdict,
    )


@router.post("/run")
async def run_pipeline(request: RunRequest):
    """
    Execute the Cournot pipeline.
    
    Runs the full pipeline on the provided user input and returns either:
    - JSON response with artifacts (return_format="json")
    - ZIP file with artifact pack (return_format="pack_zip")
    """
    try:
        # Create and run pipeline
        pipeline = get_pipeline(
            strict_mode=request.strict_mode,
            enable_sentinel=request.enable_sentinel_verify,
            enable_replay=request.enable_replay,
        )
        
        result = pipeline.run(request.user_input)
        
        # Handle pack_zip format
        if request.return_format == "pack_zip":
            package = build_package(result)
            if package is None:
                raise InternalError("Failed to build artifact package")
            
            # Create temp file for zip
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            try:
                save_pack_zip(package, tmp_path)
                zip_bytes = tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)
            
            # Build headers
            headers = {
                "X-Market-Id": result.market_id or "",
                "X-Por-Root": result.por_bundle.por_root if result.por_bundle else "",
                "X-Outcome": result.outcome or "",
                "X-Confidence": str(result.verdict.confidence if result.verdict else 0.0),
                "Content-Disposition": f"attachment; filename=pack_{result.market_id or 'unknown'}.zip",
            }
            
            return StreamingResponse(
                io.BytesIO(zip_bytes),
                media_type="application/zip",
                headers=headers,
            )
        
        # Handle JSON format
        summary = build_summary(result)
        artifacts = build_artifacts(result)
        verification = build_verification(result, request.include_checks)
        
        return RunResponse(
            ok=result.ok,
            summary=summary,
            artifacts=artifacts,
            verification=verification,
            errors=list(result.errors) if result.errors else [],
        )
    
    except Exception as e:
        raise InternalError(f"Pipeline execution failed: {str(e)}")