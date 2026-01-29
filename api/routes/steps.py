"""
Module 09D - Steps Route

Execute individual pipeline steps.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import get_agent_context
from api.errors import InternalError, PipelineError

from agents import AgentContext


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/step", tags=["steps"])


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


@router.post("/prompt", response_model=PromptEngineerResponse)
async def run_prompt_engineer(request: PromptEngineerRequest) -> PromptEngineerResponse:
    """
    Run only the Prompt Engineer step.
    
    Compiles a natural language query into a structured PromptSpec and ToolPlan.
    """
    try:
        logger.info(f"Compiling prompt: {request.user_input[:50]}...")
        
        ctx = get_agent_context()
        
        # Use the agent directly for strict_mode control
        from agents.prompt_engineer import PromptEngineerFallback
        
        agent = PromptEngineerFallback(strict_mode=request.strict_mode)
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