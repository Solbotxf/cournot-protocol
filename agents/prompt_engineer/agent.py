"""
Prompt Engineer Agent

Converts user free-text questions into structured PromptSpec and ToolPlan.

Uses LLM for compilation. Requires an LLM client in the agent context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, AgentStep, BaseAgent
from agents.registry import register_agent
from core.schemas import PromptSpec, ToolPlan, VerificationResult, CheckResult

from .llm_compiler import LLMPromptCompiler

if TYPE_CHECKING:
    from agents.context import AgentContext


class PromptEngineerLLM(BaseAgent):
    """
    LLM-based Prompt Engineer.
    
    Uses LLM to convert user questions into structured specifications.
    Primary production implementation.
    
    Features:
    - Full semantic understanding of questions
    - Intelligent source selection
    - JSON validation with repair loop
    - Receipt recording for LLM calls
    """
    
    _name = "PromptEngineerLLM"
    _version = "v1"
    _capabilities = {AgentCapability.LLM}
    
    def __init__(self, *, strict_mode: bool = True, **kwargs) -> None:
        """
        Initialize LLM-based prompt engineer.
        
        Args:
            strict_mode: If True, exclude timestamps from hashing
        """
        super().__init__(**kwargs)
        self.compiler = LLMPromptCompiler(strict_mode=strict_mode)
        self.strict_mode = strict_mode
    
    def run(
        self,
        ctx: "AgentContext",
        user_input: str,
    ) -> AgentResult:
        """
        Compile user input to PromptSpec and ToolPlan.
        
        Args:
            ctx: Agent context with LLM client
            user_input: User's prediction question
        
        Returns:
            AgentResult with (PromptSpec, ToolPlan) as output
        """
        ctx.info(f"PromptEngineerLLM compiling: {user_input[:50]}...")
        
        try:
            # Compile using LLM
            prompt_spec, tool_plan = self.compiler.compile(ctx, user_input)
            
            # Validate the output
            verification = self._validate_output(prompt_spec, tool_plan)
            
            if not verification.ok:
                return AgentResult.failure(
                    error=f"Validation failed: {[c.message for c in verification.get_failed_checks()]}",
                    output=(prompt_spec, tool_plan),
                    verification=verification,
                )
            
            ctx.info(f"Successfully compiled to market_id={prompt_spec.market_id}")
            
            return AgentResult(
                output=(prompt_spec, tool_plan),
                verification=verification,
                receipts=ctx.get_receipt_refs(),
                metadata={
                    "compiler": "llm",
                    "strict_mode": self.strict_mode,
                    "market_id": prompt_spec.market_id,
                    "num_requirements": len(prompt_spec.data_requirements),
                },
            )
        
        except Exception as e:
            ctx.error(f"LLM compilation failed: {e}")
            return AgentResult.failure(
                error=str(e),
            )
    
    def _validate_output(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        """Validate the compiled output."""
        checks: list[CheckResult] = []
        
        # Check 1: Has data requirements
        if not prompt_spec.data_requirements:
            checks.append(CheckResult.failed(
                check_id="has_requirements",
                message="PromptSpec must have at least one data requirement",
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="has_requirements",
                message=f"Has {len(prompt_spec.data_requirements)} data requirements",
            ))
        
        # Check 2: All requirements have source targets
        for req in prompt_spec.data_requirements:
            if not req.source_targets:
                checks.append(CheckResult.failed(
                    check_id=f"req_{req.requirement_id}_has_sources",
                    message=f"Requirement {req.requirement_id} has no source targets",
                ))
            else:
                for target in req.source_targets:
                    if not target.uri:
                        checks.append(CheckResult.failed(
                            check_id=f"req_{req.requirement_id}_has_uri",
                            message=f"Source target {target.source_id} has no URI",
                        ))
        
        # Check 3: ToolPlan matches requirements
        req_ids = {req.requirement_id for req in prompt_spec.data_requirements}
        plan_ids = set(tool_plan.requirements)
        
        if req_ids != plan_ids:
            missing = req_ids - plan_ids
            extra = plan_ids - req_ids
            checks.append(CheckResult.failed(
                check_id="plan_matches_spec",
                message=f"ToolPlan mismatch: missing={missing}, extra={extra}",
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="plan_matches_spec",
                message="ToolPlan requirements match PromptSpec",
            ))
        
        # Check 4: Resolution rules exist
        if not prompt_spec.market.resolution_rules.rules:
            checks.append(CheckResult.failed(
                check_id="has_rules",
                message="Market must have resolution rules",
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="has_rules",
                message=f"Has {len(prompt_spec.market.resolution_rules.rules)} resolution rules",
            ))
        
        # Determine overall status
        ok = all(c.ok for c in checks)
        
        return VerificationResult(ok=ok, checks=checks)


def get_prompt_engineer(ctx: "AgentContext") -> BaseAgent:
    """
    Get the prompt engineer agent.

    Requires an LLM client in the agent context.

    Args:
        ctx: Agent context (must have ctx.llm set)

    Returns:
        PromptEngineerLLM

    Raises:
        ValueError: If no LLM client is configured
    """
    if ctx.llm is None:
        raise ValueError(
            "LLM client is required for prompt compilation. "
            "Configure an LLM provider in the agent context."
        )
    return PromptEngineerLLM()


def compile_prompt(
    ctx: "AgentContext",
    user_input: str,
) -> AgentResult:
    """
    Convenience function to compile a prompt.

    Args:
        ctx: Agent context (must have ctx.llm set)
        user_input: User's prediction question

    Returns:
        AgentResult with (PromptSpec, ToolPlan) as output
    """
    agent = get_prompt_engineer(ctx)
    return agent.run(ctx, user_input)


# Register agents with the global registry
def _register_agents() -> None:
    """Register prompt engineer agents."""
    register_agent(
        step=AgentStep.PROMPT_ENGINEER,
        name="PromptEngineerLLM",
        factory=lambda ctx: PromptEngineerLLM(),
        capabilities={AgentCapability.LLM},
        priority=100,
    )


# Auto-register on import
_register_agents()
