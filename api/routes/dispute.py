"""Dispute Route

POST /dispute

Goal: allow *stateless* dispute-driven reruns of downstream pipeline steps.

Design notes (dashboard-driven MVP):
- Frontend must provide all necessary context (prompt_spec, evidence_bundle, and
  optionally reasoning_trace) so the protocol does not look up historical runs.
- Dispute context is injected into Auditor/Judge LLM prompts via ctx.extra["dispute_context"].
- MVP supports disputing judge reasoning and evidence interpretation.

This endpoint is intentionally conservative:
- It does NOT persist disputes.
- It does NOT fetch or look up artifacts from storage.

"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import get_agent_context
from api.errors import InvalidRequestError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dispute"])


# -----------------------------
# Schemas
# -----------------------------

ReasonCode = Literal[
    # Judge-focused
    "REASONING_ERROR",
    "LOGIC_GAP",
    # Evidence-focused
    "EVIDENCE_MISREAD",
    "EVIDENCE_INSUFFICIENT",
    # Generic
    "OTHER",
]


class DisputeTarget(BaseModel):
    """UI hint for what is being disputed.

    The backend does not hard-depend on this, but it is included in the LLM
    context injection to make the rerun dispute-aware.
    """

    artifact: Literal["evidence_bundle", "reasoning_trace", "verdict", "prompt_spec"] = Field(
        ..., description="Which artifact is disputed"
    )
    leaf_path: str | None = Field(
        default=None, description="Optional JSONPath-like pointer for the disputed leaf"
    )


class DisputePatch(BaseModel):
    """Optional patch/override operations.

    MVP supports:

    - evidence_items_append: list of EvidenceItem JSON objects to append to the
      provided evidence_bundle.items before rerun.

    - prompt_spec_override: partial PromptSpec JSON to deep-merge into the
      provided prompt_spec before rerun (stateless).

    Merge strategy:
    - dicts: deep-merge
    - lists: replace wholesale (frontend should include any items it wants to keep)
    """

    evidence_items_append: list[dict[str, Any]] | None = None
    prompt_spec_override: dict[str, Any] | None = None


class DisputeRequest(BaseModel):
    mode: Literal["reasoning_only", "full_rerun"] = Field(
        default="reasoning_only",
        description="reasoning_only reruns downstream steps using provided evidence. full_rerun re-collects evidence then reruns.",
    )

    # Optional dashboard correlation only (stateless: no lookup)
    case_id: str | None = Field(default=None)

    reason_code: ReasonCode
    message: str = Field(..., min_length=1, max_length=8000)

    target: DisputeTarget | None = None

    # Full context artifacts (stateless contract)
    prompt_spec: dict[str, Any]
    evidence_bundle: dict[str, Any] | None = None

    # Optional: if present, allow judge-only rerun
    reasoning_trace: dict[str, Any] | None = None

    # Optional: required for full_rerun (stateless re-collect)
    tool_plan: dict[str, Any] | None = None
    collectors: list[str] | None = None

    # Optional patch operations (e.g., append evidence items, prompt_spec_override)
    patch: DisputePatch | dict[str, Any] | None = None


class DisputeResponse(BaseModel):
    ok: Literal[True] = True
    case_id: str | None = None
    rerun_plan: list[str]
    artifacts: dict[str, Any]
    diff: dict[str, Any] | None = None


# -----------------------------
# Planning
# -----------------------------

_RERUN_PLAN: dict[ReasonCode, list[str]] = {
    "REASONING_ERROR": ["judge"],
    "LOGIC_GAP": ["judge"],
    "EVIDENCE_MISREAD": ["audit", "judge"],
    "EVIDENCE_INSUFFICIENT": ["audit", "judge"],
    "OTHER": ["judge"],
}


def _normalize_patch(patch: DisputePatch | dict[str, Any] | None) -> DisputePatch | None:
    if patch is None:
        return None
    if isinstance(patch, DisputePatch):
        return patch
    if isinstance(patch, dict):
        try:
            return DisputePatch(**patch)
        except Exception:
            # Treat as no-op patch if schema doesn't match; keep raw dict for LLM context injection.
            return None
    return None


def _deep_merge(base: Any, override: Any) -> Any:
    """Deep merge override onto base.

    - dict: recurse
    - list: replace (return override)
    - scalar/other: replace

    This keeps behavior predictable and makes overrides explicit.
    """
    if override is None:
        return base

    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(out.get(k), v)
        return out

    if isinstance(override, list):
        return override

    return override


# -----------------------------
# Route
# -----------------------------


@router.post("/dispute", response_model=DisputeResponse)
async def dispute(request: DisputeRequest) -> DisputeResponse:
    """Rerun audit/judge based on a dispute, returning new artifacts."""

    if request.mode not in ("reasoning_only", "full_rerun"):
        raise InvalidRequestError("Invalid mode")

    rerun_plan = _RERUN_PLAN.get(request.reason_code, ["judge"])

    # Parse artifacts into core schemas (strict validation)
    try:
        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle, EvidenceItem
        from core.schemas.reasoning import ReasoningTrace

        # Apply patch operations (prompt_spec override + evidence append)
        normalized_patch = _normalize_patch(request.patch)

        merged_prompt_spec_dict = request.prompt_spec
        if normalized_patch and normalized_patch.prompt_spec_override:
            merged_prompt_spec_dict = _deep_merge(request.prompt_spec, normalized_patch.prompt_spec_override)

        prompt_spec = PromptSpec(**merged_prompt_spec_dict)

        evidence_bundle = EvidenceBundle(**request.evidence_bundle) if request.evidence_bundle is not None else None
        reasoning_trace = (
            ReasoningTrace(**request.reasoning_trace) if request.reasoning_trace else None
        )

        if evidence_bundle is not None and normalized_patch and normalized_patch.evidence_items_append:
            for raw_item in normalized_patch.evidence_items_append:
                evidence_bundle.items.append(EvidenceItem(**raw_item))

    except Exception as e:
        raise InvalidRequestError(f"Invalid artifacts in request: {e}")

    # Build ctx + inject dispute_context for LLM-aware reruns
    ctx = get_agent_context(with_llm=True, with_http=True, llm_override=None)
    ctx.extra["dispute_context"] = {
        "reason_code": request.reason_code,
        "message": request.message,
        "target": request.target.model_dump(mode="json") if request.target else None,
        "patch": request.patch if isinstance(request.patch, dict) else (request.patch.model_dump(mode="json") if request.patch else None),
    }

    from agents.auditor import get_auditor
    from agents.judge import get_judge

    # Evidence bundles for rerun
    evidence_bundles = [evidence_bundle] if evidence_bundle is not None else []

    executed_plan: list[str] = []

    # Optional: full rerun includes evidence collection
    if request.mode == "full_rerun":
        if request.tool_plan is None:
            raise InvalidRequestError("tool_plan is required for mode=full_rerun")
        if not request.collectors:
            raise InvalidRequestError("collectors is required for mode=full_rerun")

        try:
            from core.schemas.transport import ToolPlan
            from core.schemas.evidence import EvidenceBundle
            from agents.registry import get_registry
            from api.deps import build_llm_override

            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            raise InvalidRequestError(f"Invalid tool_plan: {e}")

        registry = get_registry()

        collector_override = build_llm_override(None, None, agent_name="collector")
        collector_ctx = get_agent_context(with_llm=True, with_http=True, llm_override=collector_override)

        tasks = []
        task_names = []
        for collector_name in request.collectors:
            try:
                # Match /step/resolve collector instantiation patterns
                if collector_name == "CollectorPAN":
                    from agents.collector.pan_agent import PANCollectorAgent
                    collector = PANCollectorAgent()
                elif collector_name == "CollectorOpenSearch":
                    from agents.collector.gemini_grounded_agent import CollectorOpenSearch
                    collector = CollectorOpenSearch()
                elif collector_name == "CollectorSitePinned":
                    from agents.collector.source_pinned_agent import CollectorSitePinned
                    collector = CollectorSitePinned()
                elif collector_name == "CollectorCRP":
                    from agents.collector.crp_agent import CollectorCRP
                    collector = CollectorCRP()
                else:
                    collector = registry.get_agent_by_name(collector_name, collector_ctx)
            except Exception as e:
                raise InvalidRequestError(f"Collector not found: {collector_name} ({e})")

            tasks.append(asyncio.to_thread(collector.run, collector_ctx, prompt_spec, tool_plan))
            task_names.append(collector.name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        evidence_bundles = []
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Collector {name} failed: {result}")
                continue
            if not getattr(result, "success", False):
                logger.warning(f"Collector {name} failed: {getattr(result, 'error', None)}")
                continue
            bundle, _ = result.output
            bundle.collector_name = name
            evidence_bundles.append(bundle)

        if not evidence_bundles:
            raise InvalidRequestError("No evidence collected in full_rerun")

        executed_plan.append("collect")

    # Audit (if requested or if reasoning_trace missing)
    if "audit" in rerun_plan or reasoning_trace is None:
        if not evidence_bundles:
            raise InvalidRequestError("evidence_bundle is required for reasoning_only, or enable full_rerun")
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

    primary_bundle = evidence_bundles[0] if evidence_bundles else None

    artifacts = {
        "prompt_spec": prompt_spec.model_dump(mode="json"),
        # Back-compat: include a single primary bundle
        "evidence_bundle": primary_bundle.model_dump(mode="json") if primary_bundle else None,
        # Preferred: include all bundles
        "evidence_bundles": [b.model_dump(mode="json") for b in evidence_bundles],
        "reasoning_trace": reasoning_trace.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
    }

    # Lightweight diff summary (frontend can compute richer diff)
    diff: dict[str, Any] = {
        "steps_rerun": executed_plan,
        "verdict_changed": None,
    }

    return DisputeResponse(
        case_id=request.case_id,
        rerun_plan=executed_plan,
        artifacts=artifacts,
        diff=diff,
    )
