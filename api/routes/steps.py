"""
Module 09D - Steps Route

Execute individual pipeline steps.
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