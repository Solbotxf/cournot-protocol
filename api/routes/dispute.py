"""
Dispute Route

POST /dispute — Submit a dispute against a pipeline step,
rerun downstream steps with dispute context, return new artifacts + diff.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.errors import InternalError, InvalidRequestError

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
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/dispute", response_model=DisputeResponse)
async def submit_dispute(request: DisputeRequest) -> DisputeResponse:
    """Submit a dispute against a specific pipeline step.

    Validates the request, reruns the minimal set of downstream steps with
    dispute context injected into LLM prompts, and returns new artifacts
    with a diff summary.
    """
    # Validate
    errors = validate_dispute_request(request)
    if errors:
        raise InvalidRequestError("; ".join(errors))

    try:
        # Lazy imports (same pattern as steps.py)
        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle
        from core.schemas.reasoning import ReasoningTrace
        from core.schemas.verdict import DeterministicVerdict
        from agents.auditor import get_auditor
        from agents.judge import get_judge
        from api.deps import get_agent_context, build_llm_override

        dispute_id = f"disp_{uuid.uuid4().hex[:12]}"
        rerun_plan = compute_rerun_plan(request.target.step)
        step_errors: list[str] = []

        # --- Parse context artifacts ---
        try:
            prompt_spec = PromptSpec(**request.context.prompt_spec)
        except Exception as e:
            return DisputeResponse(
                ok=False, dispute_id=dispute_id,
                steps_rerun=rerun_plan,
                errors=[f"Invalid prompt_spec: {e}"],
            )

        evidence_bundles: list[EvidenceBundle] = []
        for i, eb_dict in enumerate(request.context.evidence_bundle):
            try:
                evidence_bundles.append(EvidenceBundle(**eb_dict))
            except Exception as e:
                return DisputeResponse(
                    ok=False, dispute_id=dispute_id,
                    steps_rerun=rerun_plan,
                    errors=[f"Invalid evidence_bundle[{i}]: {e}"],
                )

        if not evidence_bundles:
            return DisputeResponse(
                ok=False, dispute_id=dispute_id,
                steps_rerun=rerun_plan,
                errors=["No evidence bundles provided"],
            )

        reasoning_trace: ReasoningTrace | None = None
        if request.context.reasoning_trace is not None:
            try:
                reasoning_trace = ReasoningTrace(**request.context.reasoning_trace)
            except Exception as e:
                return DisputeResponse(
                    ok=False, dispute_id=dispute_id,
                    steps_rerun=rerun_plan,
                    errors=[f"Invalid reasoning_trace: {e}"],
                )

        old_verdict: DeterministicVerdict | None = None
        if request.context.verdict is not None:
            try:
                old_verdict = DeterministicVerdict(**request.context.verdict)
            except Exception as e:
                step_errors.append(f"Could not parse original verdict: {e}")

        # Build dispute context payload for agent injection
        dispute_ctx_payload = {
            "reason_code": request.reason_code.value,
            "message": request.message,
            "leaf_path": request.target.leaf_path,
            "target_step": request.target.step,
            "patch": request.patch,
        }

        new_reasoning_trace: ReasoningTrace | None = None
        new_verdict: DeterministicVerdict | None = None

        # --- Rerun: collect ---
        if "collect" in rerun_plan:
            # Collect rerun is only available in full_rerun mode (validated above).
            # For now, skip actual collection and use provided evidence bundles.
            # Full collection rerun would require tool_plan and collector orchestration.
            logger.info(f"[{dispute_id}] Collect rerun requested — using provided evidence bundles")

        # --- Rerun: audit ---
        if "audit" in rerun_plan:
            logger.info(f"[{dispute_id}] Running auditor with dispute context")
            try:
                override = build_llm_override(None, None, agent_name="auditor")
                ctx = get_agent_context(with_llm=True, llm_override=override)
                ctx.extra["dispute_context"] = dispute_ctx_payload
                auditor = get_auditor(ctx)
                result = await asyncio.to_thread(
                    auditor.run, ctx, prompt_spec, evidence_bundles,
                )
                if not result.success:
                    step_errors.append(f"Audit failed: {result.error}")
                else:
                    reasoning_trace = result.output
                    new_reasoning_trace = reasoning_trace
            except Exception as e:
                step_errors.append(f"Audit error: {e}")

        # --- Rerun: judge ---
        if "judge" in rerun_plan:
            if reasoning_trace is None:
                step_errors.append("Cannot run judge: no reasoning trace available")
            else:
                logger.info(f"[{dispute_id}] Running judge with dispute context")
                try:
                    override = build_llm_override(None, None, agent_name="judge")
                    ctx = get_agent_context(with_llm=True, llm_override=override)
                    ctx.extra["dispute_context"] = dispute_ctx_payload
                    judge = get_judge(ctx)
                    result = await asyncio.to_thread(
                        judge.run, ctx, prompt_spec, evidence_bundles, reasoning_trace,
                    )
                    if not result.success:
                        step_errors.append(f"Judge failed: {result.error}")
                    else:
                        new_verdict = result.output
                except Exception as e:
                    step_errors.append(f"Judge error: {e}")

        # --- Diff summary ---
        diff = compute_diff_summary(rerun_plan, old_verdict, new_verdict)

        return DisputeResponse(
            ok=len(step_errors) == 0,
            dispute_id=dispute_id,
            steps_rerun=rerun_plan,
            new_reasoning_trace=(
                new_reasoning_trace.model_dump(mode="json")
                if new_reasoning_trace else None
            ),
            new_verdict=(
                new_verdict.model_dump(mode="json")
                if new_verdict else None
            ),
            diff=diff,
            errors=step_errors,
        )

    except InvalidRequestError:
        raise
    except Exception as e:
        logger.exception("Dispute step failed")
        raise InternalError(f"Dispute step failed: {str(e)}")
