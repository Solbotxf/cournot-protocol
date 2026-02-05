"""
Module 09D - Steps Route

Execute individual pipeline steps.

Endpoints:
- POST /step/prompt   - Compile query into PromptSpec + ToolPlan
- POST /step/collect  - Collect evidence from sources
- POST /step/audit    - Generate reasoning trace
- POST /step/judge    - Produce final verdict
- POST /step/bundle   - Build PoR bundle
- POST /step/resolve  - Run all steps (collect → audit → judge → bundle)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import get_agent_context, get_pipeline
from api.errors import InternalError, PipelineError

from agents import AgentContext


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/step", tags=["steps"])


# ---------------------------------------------------------------------------
# Prompt Engineer Step
# ---------------------------------------------------------------------------

class PromptEngineerRequest(BaseModel):
    """Request for prompt engineer step."""

    user_input: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The prediction market query to compile",
    )
    strict_mode: bool = Field(
        default=True,
        description="Enable strict mode for deterministic hashing",
    )


class PromptEngineerResponse(BaseModel):
    """Response from prompt engineer step."""

    ok: bool = Field(..., description="Whether compilation succeeded")
    market_id: str | None = Field(default=None, description="Generated market ID")
    prompt_spec: dict[str, Any] | None = Field(default=None, description="Compiled prompt specification")
    tool_plan: dict[str, Any] | None = Field(default=None, description="Tool execution plan")
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, description="Error message if failed")


# ---------------------------------------------------------------------------
# Resolve Step
# ---------------------------------------------------------------------------

class ResolveRequest(BaseModel):
    """Request for resolve step — runs collect → audit → judge → PoR bundle."""

    prompt_spec: dict[str, Any] = Field(
        ..., description="Compiled prompt specification (from /step/prompt output)"
    )
    tool_plan: dict[str, Any] = Field(
        ..., description="Tool execution plan (from /step/prompt output)"
    )
    execution_mode: Literal["production", "development", "test"] = Field(
        default="development",
        description="Execution mode for agent selection",
    )
    include_raw_content: bool = Field(
        default=False,
        description="Include raw_content in evidence_bundle items (omitted by default to reduce size)",
    )


class ResolveResponse(BaseModel):
    """Response from resolve step."""

    ok: bool = Field(..., description="Whether resolution succeeded")
    market_id: str | None = Field(default=None, description="Market identifier")
    outcome: str | None = Field(default=None, description="Resolution outcome: YES, NO, or INVALID")
    confidence: float | None = Field(default=None, description="Confidence score (0.0-1.0)")
    por_root: str | None = Field(default=None, description="Proof of Reasoning root hash")
    artifacts: dict[str, Any] | None = Field(
        default=None,
        description="Intermediate artifacts (evidence_bundle, reasoning_trace, verdict, por_bundle)",
    )
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Collect Step
# ---------------------------------------------------------------------------

class CollectRequest(BaseModel):
    """Request for evidence collection step."""

    prompt_spec: dict[str, Any] = Field(
        ..., description="Compiled prompt specification (from /step/prompt)"
    )
    tool_plan: dict[str, Any] = Field(
        ..., description="Tool execution plan (from /step/prompt)"
    )
    execution_mode: Literal["production", "development", "test"] = Field(
        default="development",
        description="Execution mode for agent selection",
    )
    include_raw_content: bool = Field(
        default=False,
        description="Include raw_content in evidence items",
    )


class CollectResponse(BaseModel):
    """Response from evidence collection step."""

    ok: bool = Field(..., description="Whether collection succeeded")
    evidence_bundle: dict[str, Any] | None = Field(
        default=None, description="Collected evidence bundle"
    )
    execution_log: dict[str, Any] | None = Field(
        default=None, description="Tool execution log"
    )
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Audit Step
# ---------------------------------------------------------------------------

class AuditRequest(BaseModel):
    """Request for audit/reasoning step."""

    prompt_spec: dict[str, Any] = Field(
        ..., description="Compiled prompt specification"
    )
    evidence_bundle: dict[str, Any] = Field(
        ..., description="Evidence bundle (from /step/collect)"
    )
    execution_mode: Literal["production", "development", "test"] = Field(
        default="development",
        description="Execution mode for agent selection",
    )


class AuditResponse(BaseModel):
    """Response from audit step."""

    ok: bool = Field(..., description="Whether audit succeeded")
    reasoning_trace: dict[str, Any] | None = Field(
        default=None, description="Generated reasoning trace"
    )
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Judge Step
# ---------------------------------------------------------------------------

class JudgeRequest(BaseModel):
    """Request for judge/verdict step."""

    prompt_spec: dict[str, Any] = Field(
        ..., description="Compiled prompt specification"
    )
    evidence_bundle: dict[str, Any] = Field(
        ..., description="Evidence bundle (from /step/collect)"
    )
    reasoning_trace: dict[str, Any] = Field(
        ..., description="Reasoning trace (from /step/audit)"
    )
    execution_mode: Literal["production", "development", "test"] = Field(
        default="development",
        description="Execution mode for agent selection",
    )


class JudgeResponse(BaseModel):
    """Response from judge step."""

    ok: bool = Field(..., description="Whether verdict succeeded")
    verdict: dict[str, Any] | None = Field(
        default=None, description="Final verdict"
    )
    outcome: str | None = Field(default=None, description="Resolution outcome")
    confidence: float | None = Field(default=None, description="Confidence score")
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bundle Step
# ---------------------------------------------------------------------------

class BundleRequest(BaseModel):
    """Request for PoR bundle step."""

    prompt_spec: dict[str, Any] = Field(
        ..., description="Compiled prompt specification"
    )
    evidence_bundle: dict[str, Any] = Field(
        ..., description="Evidence bundle (from /step/collect)"
    )
    reasoning_trace: dict[str, Any] = Field(
        ..., description="Reasoning trace (from /step/audit)"
    )
    verdict: dict[str, Any] = Field(
        ..., description="Verdict (from /step/judge)"
    )


class BundleResponse(BaseModel):
    """Response from PoR bundle step."""

    ok: bool = Field(..., description="Whether bundle build succeeded")
    por_bundle: dict[str, Any] | None = Field(
        default=None, description="Proof of Reasoning bundle"
    )
    por_root: str | None = Field(default=None, description="PoR root hash")
    roots: dict[str, Any] | None = Field(
        default=None, description="All computed roots"
    )
    errors: list[str] = Field(default_factory=list)


# ===========================================================================
# Endpoint Implementations
# ===========================================================================

@router.post("/prompt", response_model=PromptEngineerResponse)
async def run_prompt_engineer(request: PromptEngineerRequest) -> PromptEngineerResponse:
    """
    Run only the Prompt Engineer step.
    
    Compiles a natural language query into a structured PromptSpec and ToolPlan.
    """
    try:
        logger.info(f"Compiling prompt: {request.user_input[:50]}...")
        
        ctx = get_agent_context(with_llm=True)
        
        # Use the LLM agent for prompt compilation
        from agents.prompt_engineer import PromptEngineerLLM

        agent = PromptEngineerLLM(strict_mode=request.strict_mode)
        result = agent.run(ctx, request.user_input)
        
        if not result.success:
            return PromptEngineerResponse(
                ok=False,
                error=result.error,
                metadata=result.metadata or {},
            )
        
        prompt_spec, tool_plan = result.output
        
        return PromptEngineerResponse(
            ok=True,
            market_id=prompt_spec.market.market_id,
            prompt_spec=prompt_spec.model_dump(mode="json"),
            tool_plan=tool_plan.model_dump(mode="json"),
            metadata=result.metadata or {},
        )
    
    except Exception as e:
        logger.exception("Prompt engineer failed")
        raise InternalError(f"Prompt engineer failed: {str(e)}")


@router.post("/resolve", response_model=ResolveResponse)
async def run_resolve(request: ResolveRequest) -> ResolveResponse:
    """
    Run the resolve step: collect evidence, audit, judge, and build PoR bundle.

    Accepts prompt_spec and tool_plan from the /step/prompt output and runs
    all remaining pipeline steps to produce a final verdict.
    """
    try:
        logger.info("Running resolve step...")

        # Parse prompt_spec and tool_plan from request dicts
        from core.schemas.prompts import PromptSpec
        from core.schemas.transport import ToolPlan

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return ResolveResponse(
                ok=False,
                errors=[f"Invalid prompt_spec: {e}"],
            )

        try:
            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            return ResolveResponse(
                ok=False,
                errors=[f"Invalid tool_plan: {e}"],
            )

        # Create pipeline with LLM + HTTP enabled
        pipeline = get_pipeline(
            mode=request.execution_mode,
            with_llm=True,
            with_http=True,
            enable_sentinel=False,
        )

        result = pipeline.run_from_prompt(prompt_spec, tool_plan)

        # Build artifacts dict; omit raw_content unless requested
        artifacts: dict[str, Any] = {}
        if result.evidence_bundle:
            eb = result.evidence_bundle.model_dump(mode="json")
            if not request.include_raw_content:
                for item in eb.get("items", []):
                    item["raw_content"] = None
                    item["parsed_value"] = None
            artifacts["evidence_bundle"] = eb
        if result.audit_trace:
            artifacts["reasoning_trace"] = result.audit_trace.model_dump(mode="json")
        if result.verdict:
            artifacts["verdict"] = result.verdict.model_dump(mode="json")
        if result.por_bundle:
            artifacts["por_bundle"] = result.por_bundle.model_dump(mode="json")

        return ResolveResponse(
            ok=result.ok,
            market_id=result.verdict.market_id if result.verdict else None,
            outcome=result.verdict.outcome if result.verdict else None,
            confidence=result.verdict.confidence if result.verdict else None,
            por_root=result.por_bundle.por_root if result.por_bundle else None,
            artifacts=artifacts or None,
            errors=list(result.errors) if result.errors else [],
        )

    except Exception as e:
        logger.exception("Resolve step failed")
        raise InternalError(f"Resolve step failed: {str(e)}")


# ---------------------------------------------------------------------------
# Individual Step Endpoints (for interactive frontend)
# ---------------------------------------------------------------------------

@router.post("/collect", response_model=CollectResponse)
async def run_collect(request: CollectRequest) -> CollectResponse:
    """
    Run evidence collection step.

    Collects evidence from sources specified in the tool_plan.
    This is step 1 of the resolution pipeline.
    """
    try:
        logger.info("Running collect step...")

        from core.schemas.prompts import PromptSpec
        from core.schemas.transport import ToolPlan

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return CollectResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            return CollectResponse(ok=False, errors=[f"Invalid tool_plan: {e}"])

        # Get context with LLM + HTTP for collector
        ctx = get_agent_context(with_llm=True, with_http=True)

        # Select and run collector agent
        from agents.collector import get_collector

        collector = get_collector(ctx)
        logger.info(f"Using collector: {collector.name}")

        result = collector.run(ctx, prompt_spec, tool_plan)

        if not result.success:
            return CollectResponse(ok=False, errors=[result.error or "Collection failed"])

        evidence_bundle, execution_log = result.output

        # Optionally strip raw_content
        eb_dict = evidence_bundle.model_dump(mode="json")
        if not request.include_raw_content:
            for item in eb_dict.get("items", []):
                item["raw_content"] = None
                item["parsed_value"] = None

        return CollectResponse(
            ok=True,
            evidence_bundle=eb_dict,
            execution_log=execution_log.model_dump(mode="json") if execution_log else None,
        )

    except Exception as e:
        logger.exception("Collect step failed")
        raise InternalError(f"Collect step failed: {str(e)}")


@router.post("/audit", response_model=AuditResponse)
async def run_audit(request: AuditRequest) -> AuditResponse:
    """
    Run audit/reasoning step.

    Analyzes evidence and generates a reasoning trace.
    This is step 2 of the resolution pipeline.
    """
    try:
        logger.info("Running audit step...")

        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return AuditResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            evidence_bundle = EvidenceBundle(**request.evidence_bundle)
        except Exception as e:
            return AuditResponse(ok=False, errors=[f"Invalid evidence_bundle: {e}"])

        # Get context with LLM for auditor
        ctx = get_agent_context(with_llm=True)

        # Select and run auditor agent
        from agents.auditor import get_auditor

        auditor = get_auditor(ctx)
        logger.info(f"Using auditor: {auditor.name}")

        result = auditor.run(ctx, prompt_spec, evidence_bundle)

        if not result.success:
            return AuditResponse(ok=False, errors=[result.error or "Audit failed"])

        trace = result.output

        return AuditResponse(
            ok=True,
            reasoning_trace=trace.model_dump(mode="json"),
        )

    except Exception as e:
        logger.exception("Audit step failed")
        raise InternalError(f"Audit step failed: {str(e)}")


@router.post("/judge", response_model=JudgeResponse)
async def run_judge(request: JudgeRequest) -> JudgeResponse:
    """
    Run judge/verdict step.

    Reviews reasoning and produces a final verdict.
    This is step 3 of the resolution pipeline.
    """
    try:
        logger.info("Running judge step...")

        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle
        from core.schemas.reasoning import ReasoningTrace

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return JudgeResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            evidence_bundle = EvidenceBundle(**request.evidence_bundle)
        except Exception as e:
            return JudgeResponse(ok=False, errors=[f"Invalid evidence_bundle: {e}"])

        try:
            reasoning_trace = ReasoningTrace(**request.reasoning_trace)
        except Exception as e:
            return JudgeResponse(ok=False, errors=[f"Invalid reasoning_trace: {e}"])

        # Get context with LLM for judge
        ctx = get_agent_context(with_llm=True)

        # Select and run judge agent
        from agents.judge import get_judge

        judge = get_judge(ctx)
        logger.info(f"Using judge: {judge.name}")

        result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)

        if not result.success:
            return JudgeResponse(ok=False, errors=[result.error or "Judge failed"])

        verdict = result.output

        return JudgeResponse(
            ok=True,
            verdict=verdict.model_dump(mode="json"),
            outcome=verdict.outcome,
            confidence=verdict.confidence,
        )

    except Exception as e:
        logger.exception("Judge step failed")
        raise InternalError(f"Judge step failed: {str(e)}")


@router.post("/bundle", response_model=BundleResponse)
async def run_bundle(request: BundleRequest) -> BundleResponse:
    """
    Build Proof of Reasoning bundle.

    Combines all artifacts into a cryptographically verifiable bundle.
    This is step 4 of the resolution pipeline.
    """
    try:
        logger.info("Running bundle step...")

        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle
        from core.schemas.reasoning import ReasoningTrace
        from core.schemas.verdict import DeterministicVerdict
        from core.por.proof_of_reasoning import build_por_bundle, compute_roots

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return BundleResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            evidence_bundle = EvidenceBundle(**request.evidence_bundle)
        except Exception as e:
            return BundleResponse(ok=False, errors=[f"Invalid evidence_bundle: {e}"])

        try:
            reasoning_trace = ReasoningTrace(**request.reasoning_trace)
        except Exception as e:
            return BundleResponse(ok=False, errors=[f"Invalid reasoning_trace: {e}"])

        try:
            verdict = DeterministicVerdict(**request.verdict)
        except Exception as e:
            return BundleResponse(ok=False, errors=[f"Invalid verdict: {e}"])

        # Compute roots
        roots = compute_roots(prompt_spec, evidence_bundle, reasoning_trace, verdict)

        # Build PoR bundle
        por_bundle = build_por_bundle(
            prompt_spec,
            evidence_bundle,
            reasoning_trace,
            verdict,
            include_por_root=True,
            metadata={"pipeline_version": "09A", "mode": "api"},
        )

        return BundleResponse(
            ok=True,
            por_bundle=por_bundle.model_dump(mode="json"),
            por_root=por_bundle.por_root,
            roots={
                "prompt_spec_hash": roots.prompt_spec_hash,
                "evidence_root": roots.evidence_root,
                "reasoning_root": roots.reasoning_root,
                "por_root": roots.por_root,
            },
        )

    except Exception as e:
        logger.exception("Bundle step failed")
        raise InternalError(f"Bundle step failed: {str(e)}")