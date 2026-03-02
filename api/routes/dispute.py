"""
Dispute Route

POST /dispute — Submit a dispute against a pipeline step,
rerun downstream steps with dispute context, return new artifacts + diff.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.errors import InvalidRequestError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dispute"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReasonCode(str, Enum):
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    EVIDENCE_STALE = "EVIDENCE_STALE"
    SOURCE_UNRELIABLE = "SOURCE_UNRELIABLE"
    REASONING_STEP_INCORRECT = "REASONING_STEP_INCORRECT"
    REASONING_MISSING_EVIDENCE = "REASONING_MISSING_EVIDENCE"
    VERDICT_WRONG_MAPPING = "VERDICT_WRONG_MAPPING"
    VERDICT_CONFIDENCE_TOO_HIGH = "VERDICT_CONFIDENCE_TOO_HIGH"
    SPEC_AMBIGUOUS = "SPEC_AMBIGUOUS"
    OTHER = "OTHER"


class DisputeTarget(BaseModel):
    step: Literal["prompt_spec", "collect", "audit", "judge"]
    leaf_path: str | None = None


class DisputeContext(BaseModel):
    prompt_spec: dict[str, Any]
    evidence_bundle: list[dict[str, Any]]
    reasoning_trace: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None


class DisputeRequest(BaseModel):
    mode: Literal["reasoning_only", "full_rerun"] = "reasoning_only"
    target: DisputeTarget
    reason_code: ReasonCode
    message: str = Field(..., min_length=1, max_length=2000)
    patch: dict[str, Any] | None = None
    context: DisputeContext


class DiffSummary(BaseModel):
    steps_rerun: list[str]
    verdict_changed: bool
    outcome_before: str | None = None
    outcome_after: str | None = None
    confidence_before: float | None = None
    confidence_after: float | None = None


class DisputeResponse(BaseModel):
    ok: bool
    dispute_id: str
    steps_rerun: list[str]
    new_reasoning_trace: dict[str, Any] | None = None
    new_verdict: dict[str, Any] | None = None
    diff: DiffSummary | None = None
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Patch whitelist: (mode, target_step) -> set of allowed patch keys
# ---------------------------------------------------------------------------

PATCH_WHITELIST: dict[tuple[str, str], set[str]] = {
    ("reasoning_only", "audit"): {"confidence_override"},
    ("reasoning_only", "judge"): {"confidence_override"},
    ("full_rerun", "collect"): {"source_targets", "collector_names"},
    ("full_rerun", "audit"): {"confidence_override"},
    ("full_rerun", "judge"): {"confidence_override"},
    ("full_rerun", "prompt_spec"): {"assumptions", "event_definition"},
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dispute_request(req: DisputeRequest) -> list[str]:
    """Validate dispute request and return list of error strings (empty = valid)."""
    errors: list[str] = []

    # reasoning_only mode cannot target collect or prompt_spec
    if req.mode == "reasoning_only" and req.target.step in ("collect", "prompt_spec"):
        errors.append(
            f"reasoning_only mode cannot target '{req.target.step}'; "
            f"use full_rerun mode for upstream step disputes"
        )

    # judge target requires reasoning_trace in context
    if req.target.step == "judge" and req.context.reasoning_trace is None:
        errors.append(
            "reasoning_trace is required in context when targeting 'judge'"
        )

    # Validate patch keys against whitelist
    if req.patch:
        allowed = PATCH_WHITELIST.get((req.mode, req.target.step), set())
        disallowed = set(req.patch.keys()) - allowed
        if disallowed:
            errors.append(
                f"Patch keys {sorted(disallowed)} are not allowed for "
                f"mode={req.mode}, target={req.target.step}. "
                f"Allowed: {sorted(allowed) if allowed else 'none'}"
            )

    return errors


# ---------------------------------------------------------------------------
# Rerun plan
# ---------------------------------------------------------------------------

_PIPELINE_ORDER = ["collect", "audit", "judge"]


def compute_rerun_plan(target_step: str) -> list[str]:
    """Compute the list of steps to rerun from target_step onward."""
    if target_step == "prompt_spec":
        return list(_PIPELINE_ORDER)
    idx = _PIPELINE_ORDER.index(target_step)
    return _PIPELINE_ORDER[idx:]


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------

def compute_diff_summary(
    steps_rerun: list[str],
    old_verdict: Any | None,
    new_verdict: Any | None,
) -> DiffSummary:
    """Compare old and new verdicts to produce a diff summary."""
    if old_verdict is None or new_verdict is None:
        return DiffSummary(
            steps_rerun=steps_rerun,
            verdict_changed=old_verdict != new_verdict,
        )

    outcome_before = getattr(old_verdict, "outcome", None)
    outcome_after = getattr(new_verdict, "outcome", None)
    confidence_before = getattr(old_verdict, "confidence", None)
    confidence_after = getattr(new_verdict, "confidence", None)

    return DiffSummary(
        steps_rerun=steps_rerun,
        verdict_changed=outcome_before != outcome_after,
        outcome_before=outcome_before,
        outcome_after=outcome_after,
        confidence_before=confidence_before,
        confidence_after=confidence_after,
    )


# ---------------------------------------------------------------------------
# Endpoint (stub — implemented in Task 4)
# ---------------------------------------------------------------------------

@router.post("/dispute", response_model=DisputeResponse)
async def submit_dispute(request: DisputeRequest) -> DisputeResponse:
    """Submit a dispute against a specific pipeline step."""
    # Validate
    errors = validate_dispute_request(request)
    if errors:
        raise InvalidRequestError("; ".join(errors))

    return DisputeResponse(
        ok=False,
        dispute_id="not_implemented",
        steps_rerun=[],
        errors=["Endpoint not yet implemented"],
    )
