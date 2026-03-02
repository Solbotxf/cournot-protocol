"""Module 09D - Dispute Route

POST /dispute

Dispute v0 (reasoning_only) reruns audit/judge using request-carried artifacts.

Hard constraints (v0.3):
- True stateless: do NOT read or look up any historical runs/cases/artifacts.
- reasoning_only: reject any patch/*_patch fields; evidence_bundle is passed through (no merge).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from api.deps import get_agent_context
from api.errors import APIError, InvalidRequestError


logger = logging.getLogger(__name__)

router = APIRouter(tags=["dispute"])


# -----------------------------
# Schemas
# -----------------------------

ReasonCode = Literal[
    "REASONING_ERROR",
    "LOGIC_GAP",
    "EVIDENCE_MISREAD",
    "EVIDENCE_MISSING",
]


class DisputeTarget(BaseModel):
    artifact: str = Field(..., min_length=1)
    leaf_path: str | None = None


class DisputeRequest(BaseModel):
    mode: Literal["reasoning_only"] = Field(default="reasoning_only")

    # Optional, for dashboard correlation only (stateless: no lookup)
    case_id: str | None = Field(default=None)

    reason_code: ReasonCode
    message: str = Field(..., min_length=1, max_length=8000)

    target: DisputeTarget | None = None

    # Caller MUST provide full artifacts; we validate by parsing into core schemas.
    prompt_spec: dict[str, Any]
    evidence_bundle: dict[str, Any]

    # Optional: allow caller to supply previous reasoning_trace to enable judge-only rerun.
    reasoning_trace: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def forbid_patch_fields(cls, data: Any):
        if not isinstance(data, dict):
            return data
        # v0.3: reject if *any* patch/_patch key appears at top-level, even if null
        for k in list(data.keys()):
            if k == "patch" or k.endswith("_patch"):
                raise ValueError("patch fields are not allowed in reasoning_only")
        return data


class DisputeErrorResponse(BaseModel):
    ok: Literal[False] = False
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class DisputeResponse(BaseModel):
    ok: Literal[True] = True
    case_id: str | None = None
    rerun_plan: list[str]
    artifacts: dict[str, Any]
    diff: None = None


# -----------------------------
# Helpers
# -----------------------------

_RERUN_PLAN: dict[str, list[str]] = {
    "REASONING_ERROR": ["judge"],
    "LOGIC_GAP": ["judge"],
    "EVIDENCE_MISREAD": ["audit", "judge"],
    "EVIDENCE_MISSING": ["collect", "audit", "judge"],
}


def _dispute_not_allowed(message: str, *, path: str | None = None, reason: str | None = None):
    details: list[dict[str, Any]] = []
    if path or reason:
        details.append({"path": path or "", "reason": reason or ""})
    # v0.3: Use 422 + DISPUTE_NOT_ALLOWED
    raise APIError(
        code="DISPUTE_NOT_ALLOWED",
        message=message,
        status_code=422,
        details={"details": details},
    )


# -----------------------------
# Route
# -----------------------------

@router.post("/dispute", response_model=DisputeResponse)
async def dispute(request: DisputeRequest) -> DisputeResponse:
    """Rerun audit/judge based on a dispute, returning new artifacts."""

    # v0: mode is fixed by schema; still keep a guard
    if request.mode != "reasoning_only":
        _dispute_not_allowed("Only reasoning_only is supported in v0")

    # v0: EVIDENCE_MISSING implies collect; not supported without tool_plan/collectors
    if request.reason_code == "EVIDENCE_MISSING":
        _dispute_not_allowed(
            "reason_code EVIDENCE_MISSING requires collect, which is not supported in v0",
            path="reason_code",
            reason="collect_not_supported_in_v0",
        )

    rerun_plan = _RERUN_PLAN[request.reason_code]

    # Parse artifacts into core schemas (strict validation)
    try:
        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle
        from core.schemas.reasoning import ReasoningTrace

        prompt_spec = PromptSpec(**request.prompt_spec)
        evidence_bundle = EvidenceBundle(**request.evidence_bundle)
        reasoning_trace = ReasoningTrace(**request.reasoning_trace) if request.reasoning_trace else None
    except Exception as e:
        # 400 - invalid request payload
        raise InvalidRequestError(f"Invalid artifacts in request: {e}")

    # Build ctx (development-friendly) + inject dispute_context
    ctx = get_agent_context(with_llm=True, with_http=True, llm_override=None)
    ctx.extra["dispute_context"] = {
        "reason_code": request.reason_code,
        "message": request.message,
        "target": request.target.model_dump(mode="json") if request.target else None,
    }

    # Rerun auditor if needed
    from agents.auditor import get_auditor
    from agents.judge import get_judge

    evidence_bundles = [evidence_bundle]

    executed_plan: list[str] = []

    if "audit" in rerun_plan or reasoning_trace is None:
        auditor = get_auditor(ctx)
        audit_result = await asyncio.to_thread(auditor.run, ctx, prompt_spec, evidence_bundles)
        if not audit_result.success:
            raise InvalidRequestError(f"Audit failed: {audit_result.error}")
        reasoning_trace = audit_result.output
        executed_plan.append("audit")

    # Judge
    judge = get_judge(ctx)
    judge_result = await asyncio.to_thread(judge.run, ctx, prompt_spec, evidence_bundles, reasoning_trace)
    if not judge_result.success:
        raise InvalidRequestError(f"Judge failed: {judge_result.error}")
    verdict = judge_result.output
    executed_plan.append("judge")

    artifacts = {
        "prompt_spec": prompt_spec.model_dump(mode="json"),
        "evidence_bundle": evidence_bundle.model_dump(mode="json"),
        "reasoning_trace": reasoning_trace.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
    }

    # Return executed plan (may include audit even when rerun_plan was judge-only)
    return DisputeResponse(
        case_id=request.case_id,
        rerun_plan=executed_plan,
        artifacts=artifacts,
        diff=None,
    )
