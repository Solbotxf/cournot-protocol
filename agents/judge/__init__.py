"""
Judge Agent Module

Finalizes verdicts from reasoning traces.

Usage:
    from agents.judge import judge_verdict, JudgeLLM, JudgeRuleBased
    
    # Using convenience function (auto-selects based on context)
    result = judge_verdict(ctx, prompt_spec, evidence_bundle, reasoning_trace)
    verdict = result.output
    
    # Using specific agent
    judge = JudgeRuleBased()
    result = judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)
"""

from .agent import (
    JudgeLLM,
    JudgeRuleBased,
    get_judge,
    judge_verdict,
)

from .verdict_builder import VerdictBuilder, VerdictValidator

__all__ = [
    # Agents
    "JudgeLLM",
    "JudgeRuleBased",
    # Functions
    "get_judge",
    "judge_verdict",
    # Builders
    "VerdictBuilder",
    "VerdictValidator",
]
