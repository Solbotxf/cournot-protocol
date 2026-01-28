"""
Orchestrator wrapper for Prompt Engineer.

Provides a simplified interface for the orchestrator to invoke
the Prompt Engineer agent without directly managing the agent lifecycle.
"""

from typing import Any, Optional

from agents.base_agent import AgentContext
from agents.prompt_engineer import (
    PromptEngineerAgent,
    PromptModule,
    StrictPromptCompilerV1,
)
from core.schemas import PromptSpec, ToolPlan
from core.schemas.errors import PromptCompilationException


def compile_prompt(
    user_input: str,
    *,
    module: Optional[PromptModule] = None,
    context: Optional[AgentContext] = None,
    config: dict[str, Any] | None = None
) -> tuple[PromptSpec, ToolPlan]:
    """
    Compile user input into PromptSpec and ToolPlan.
    
    This is the primary entry point for the orchestrator to invoke
    the Prompt Engineer.
    
    Args:
        user_input: Raw user text describing the prediction
        module: Optional custom PromptModule implementation
        context: Optional agent execution context
        config: Optional configuration for the compiler
        
    Returns:
        Tuple of (PromptSpec, ToolPlan)
        
    Raises:
        PromptCompilationException: If compilation fails
    """
    agent = PromptEngineerAgent(module=module, context=context, config=config)
    return agent.run(user_input)


def validate_prompt_spec(prompt_spec: PromptSpec) -> tuple[bool, list[str]]:
    """
    Validate a PromptSpec for completeness and strictness.
    
    Args:
        prompt_spec: The PromptSpec to validate
        
    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []
    
    # Check strict mode
    if not prompt_spec.extra.get("strict_mode"):
        errors.append("PromptSpec not in strict mode")
    
    # Check event definition
    if not prompt_spec.market.event_definition:
        errors.append("Missing event definition")
    
    # Check data requirements have source targets
    for req in prompt_spec.data_requirements:
        if not req.source_targets:
            errors.append(f"Requirement {req.requirement_id} has no source targets")
    
    # Check output schema reference
    if prompt_spec.output_schema_ref != "core.schemas.verdict.DeterministicVerdict":
        errors.append("Output schema must be DeterministicVerdict")
    
    # Check confidence policy exists
    if not prompt_spec.extra.get("confidence_policy"):
        errors.append("Missing confidence policy")
    
    # Check tool plan exists and matches requirements
    if prompt_spec.tool_plan:
        req_ids = {req.requirement_id for req in prompt_spec.data_requirements}
        plan_req_ids = set(prompt_spec.tool_plan.requirements)
        if req_ids != plan_req_ids:
            errors.append("ToolPlan requirements do not match DataRequirements")
    else:
        errors.append("Missing tool plan")
    
    return len(errors) == 0, errors


def create_strict_compiler(config: dict[str, Any] | None = None) -> StrictPromptCompilerV1:
    """
    Create a new strict prompt compiler with optional configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured StrictPromptCompilerV1 instance
    """
    return StrictPromptCompilerV1(config)


__all__ = [
    "compile_prompt",
    "validate_prompt_spec",
    "create_strict_compiler",
]