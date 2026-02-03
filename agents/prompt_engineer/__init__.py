"""
Prompt Engineer Agent Module

Converts user free-text questions into structured PromptSpec and ToolPlan.

Usage:
    from agents.prompt_engineer import compile_prompt, PromptEngineerLLM

    # Using convenience function
    result = compile_prompt(ctx, "Will BTC be above $100k by end of 2025?")
    prompt_spec, tool_plan = result.output

    # Using specific agent
    agent = PromptEngineerLLM()
    result = agent.run(ctx, "Will BTC be above $100k by end of 2025?")
"""

from .agent import (
    PromptEngineerLLM,
    get_prompt_engineer,
    compile_prompt,
)

from .llm_compiler import LLMPromptCompiler

__all__ = [
    # Agents
    "PromptEngineerLLM",
    # Functions
    "get_prompt_engineer",
    "compile_prompt",
    # Compilers
    "LLMPromptCompiler",
]