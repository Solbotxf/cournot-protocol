"""
Prompt Engineer Agent Module.

Exports the Prompt Engineer agent and related components for
converting user input into structured PromptSpec and ToolPlan.
"""

from .prompt_agent import (
    CompilationResult,
    InputParser,
    NormalizedUserRequest,
    PromptEngineerAgent,
    PromptModule,
    SourceTargetBuilder,
    StrictPromptCompilerV1,
    generate_deterministic_id,
    generate_requirement_id,
)

__all__ = [
    # Main agent
    "PromptEngineerAgent",
    # Protocol
    "PromptModule",
    # Default implementation
    "StrictPromptCompilerV1",
    # Utilities
    "InputParser",
    "SourceTargetBuilder",
    "NormalizedUserRequest",
    "CompilationResult",
    "generate_deterministic_id",
    "generate_requirement_id",
]