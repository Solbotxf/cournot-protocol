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
    collectors: list[Literal["CollectorLLM", "CollectorHyDE", "CollectorHTTP", "CollectorMock"]] = Field(
        default=["CollectorLLM"],
        description="Which collector agents to use (runs all in sequence)",
        min_length=1,
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
        description="Intermediate artifacts (evidence_bundles, reasoning_trace, verdict, por_bundle)",
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
    collectors: list[Literal["CollectorLLM", "CollectorHyDE", "CollectorHTTP", "CollectorMock"]] = Field(
        default=["CollectorLLM"],
        description="Which collector agents to use (runs all in sequence)",
        min_length=1,
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
    collectors_used: list[str] = Field(default_factory=list, description="Names of collectors used")
    evidence_bundles: list[dict[str, Any]] = Field(
        default_factory=list, description="Evidence bundles from each collector"
    )
    execution_logs: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution logs from each collector"
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
    evidence_bundles: list[dict[str, Any]] = Field(
        ..., description="Evidence bundles (from /step/collect)"
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
    evidence_bundles: list[dict[str, Any]] = Field(
        ..., description="Evidence bundles (from /step/collect)"
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
    evidence_bundles: list[dict[str, Any]] = Field(
        ..., description="Evidence bundles (from /step/collect)"
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

    Supports multiple collectors - runs each in sequence and combines evidence.
    """
    try:
        logger.info(f"Running resolve step with collectors: {request.collectors}")

        from core.schemas.prompts import PromptSpec
        from core.schemas.transport import ToolPlan
        from core.schemas.evidence import EvidenceBundle
        from agents.registry import get_registry
        from agents.auditor import get_auditor
        from agents.judge import get_judge
        from core.por.proof_of_reasoning import build_por_bundle, compute_roots

        errors: list[str] = []

        # Parse inputs
        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return ResolveResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            return ResolveResponse(ok=False, errors=[f"Invalid tool_plan: {e}"])

        # Get context with LLM + HTTP
        ctx = get_agent_context(with_llm=True, with_http=True)
        registry = get_registry()

        # Step 1: Run multiple collectors
        evidence_bundles: list[EvidenceBundle] = []
        collectors_used: list[str] = []

        for collector_name in request.collectors:
            try:
                collector = registry.get_agent_by_name(collector_name, ctx)
            except ValueError:
                errors.append(f"Collector not found: {collector_name}")
                continue

            logger.info(f"Running collector: {collector.name}")
            result = collector.run(ctx, prompt_spec, tool_plan)

            if not result.success:
                errors.append(f"{collector_name}: {result.error or 'Collection failed'}")
                continue

            bundle, _ = result.output
            bundle.collector_name = collector.name
            evidence_bundles.append(bundle)
            collectors_used.append(collector.name)

        if not evidence_bundles:
            return ResolveResponse(
                ok=False,
                errors=errors + ["No evidence collected from any collector"],
            )

        # Step 2: Run auditor
        logger.info(f"Running auditor with {len(evidence_bundles)} bundles")
        auditor = get_auditor(ctx)
        audit_result = auditor.run(ctx, prompt_spec, evidence_bundles)

        if not audit_result.success:
            return ResolveResponse(
                ok=False,
                errors=errors + [f"Audit failed: {audit_result.error}"],
            )

        reasoning_trace = audit_result.output

        # Step 3: Run judge
        logger.info("Running judge")
        judge = get_judge(ctx)
        judge_result = judge.run(ctx, prompt_spec, evidence_bundles, reasoning_trace)

        if not judge_result.success:
            return ResolveResponse(
                ok=False,
                errors=errors + [f"Judge failed: {judge_result.error}"],
            )

        verdict = judge_result.output

        # Step 4: Build PoR bundle (use first evidence bundle for hashing)
        logger.info("Building PoR bundle")
        primary_bundle = evidence_bundles[0]
        roots = compute_roots(prompt_spec, primary_bundle, reasoning_trace, verdict)
        por_bundle = build_por_bundle(
            prompt_spec,
            primary_bundle,
            reasoning_trace,
            verdict,
            include_por_root=True,
            metadata={
                "pipeline_version": "09A",
                "mode": "api",
                "collectors_used": collectors_used,
            },
        )

        # Build artifacts dict
        artifacts: dict[str, Any] = {}

        evidence_bundles_data = []
        for eb in evidence_bundles:
            eb_dict = eb.model_dump(mode="json")
            if not request.include_raw_content:
                for item in eb_dict.get("items", []):
                    item["raw_content"] = None
                    item["parsed_value"] = None
            evidence_bundles_data.append(eb_dict)
        artifacts["evidence_bundles"] = evidence_bundles_data
        artifacts["collectors_used"] = collectors_used
        artifacts["reasoning_trace"] = reasoning_trace.model_dump(mode="json")
        artifacts["verdict"] = verdict.model_dump(mode="json")
        artifacts["por_bundle"] = por_bundle.model_dump(mode="json")

        return ResolveResponse(
            ok=True,
            market_id=verdict.market_id,
            outcome=verdict.outcome,
            confidence=verdict.confidence,
            por_root=por_bundle.por_root,
            artifacts=artifacts,
            errors=errors,  # May have non-fatal collector errors
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
    Run evidence collection step with one or more collectors.
    """
    try:
        logger.info(f"Running collect step with collectors: {request.collectors}")

        from core.schemas.prompts import PromptSpec
        from core.schemas.transport import ToolPlan
        from agents.registry import get_registry

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return CollectResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            return CollectResponse(ok=False, errors=[f"Invalid tool_plan: {e}"])

        ctx = get_agent_context(with_llm=True, with_http=True)
        registry = get_registry()

        bundles = []
        logs = []
        collectors_used = []
        errors = []

        for collector_name in request.collectors:
            try:
                collector = registry.get_agent_by_name(collector_name, ctx)
            except ValueError:
                errors.append(f"Collector not found: {collector_name}")
                continue

            logger.info(f"Running collector: {collector.name}")
            result = collector.run(ctx, prompt_spec, tool_plan)

            if not result.success:
                errors.append(f"{collector_name}: {result.error or 'Collection failed'}")
                continue

            evidence_bundle, execution_log = result.output
            evidence_bundle.collector_name = collector.name

            eb_dict = evidence_bundle.model_dump(mode="json")
            if not request.include_raw_content:
                for item in eb_dict.get("items", []):
                    item["raw_content"] = None
                    item["parsed_value"] = None

            bundles.append(eb_dict)
            logs.append(execution_log.model_dump(mode="json") if execution_log else {})
            collectors_used.append(collector.name)

        return CollectResponse(
            ok=len(bundles) > 0,
            collectors_used=collectors_used,
            evidence_bundles=bundles,
            execution_logs=logs,
            errors=errors,
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

        evidence_bundles = []
        for i, eb_dict in enumerate(request.evidence_bundles):
            try:
                evidence_bundles.append(EvidenceBundle(**eb_dict))
            except Exception as e:
                return AuditResponse(ok=False, errors=[f"Invalid evidence_bundle[{i}]: {e}"])

        if not evidence_bundles:
            return AuditResponse(ok=False, errors=["No evidence bundles provided"])

        ctx = get_agent_context(with_llm=True)

        from agents.auditor import get_auditor
        auditor = get_auditor(ctx)
        logger.info(f"Using auditor: {auditor.name}")

        result = auditor.run(ctx, prompt_spec, evidence_bundles)

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

        evidence_bundles = []
        for i, eb_dict in enumerate(request.evidence_bundles):
            try:
                evidence_bundles.append(EvidenceBundle(**eb_dict))
            except Exception as e:
                return JudgeResponse(ok=False, errors=[f"Invalid evidence_bundle[{i}]: {e}"])

        if not evidence_bundles:
            return JudgeResponse(ok=False, errors=["No evidence bundles provided"])

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

        result = judge.run(ctx, prompt_spec, evidence_bundles, reasoning_trace)

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

        evidence_bundles = []
        for i, eb_dict in enumerate(request.evidence_bundles):
            try:
                evidence_bundles.append(EvidenceBundle(**eb_dict))
            except Exception as e:
                return BundleResponse(ok=False, errors=[f"Invalid evidence_bundle[{i}]: {e}"])

        if not evidence_bundles:
            return BundleResponse(ok=False, errors=["No evidence bundles provided"])

        # Use first bundle for PoR computation
        evidence_bundle = evidence_bundles[0]

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