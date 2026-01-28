"""
Module 07 - Judge Agent
Produces deterministic verdicts from (PromptSpec, EvidenceBundle, ReasoningTrace).

Owner: Protocol/Judging Engineer
Module ID: M07

This module provides:
- JudgeAgent: The main agent that produces DeterministicVerdict
- Schema lock verification for deterministic judging
- Deterministic mapping engine for YES/NO/INVALID decisions
- Confidence computation based on policy

Public API:
- JudgeAgent: Main entry point
- AgentContext: Context for agent operations
- SchemaLock: Schema validation utilities
- EvaluationVariables: Structured evaluation state
- ConfidencePolicy: Confidence computation policy

Usage:
    from agents.judge import JudgeAgent
    
    judge = JudgeAgent()
    verdict, result = judge.judge(prompt_spec, evidence, trace)
    
    if result.ok:
        print(f"Verdict: {verdict.outcome}")
    else:
        print(f"Failed: {result.get_error_messages()}")
"""

from .judge_agent import (
    JudgeAgent,
    AgentContext,
    BaseAgent,
)

from .verification import (
    # Deterministic mapping
    EvaluationVariables,
    ConfidencePolicy,
    extract_evaluation_variables,
    check_validity,
    check_conflicts,
    determine_binary_outcome,
    compute_confidence,
    map_to_verdict,
    # Schema lock
    SchemaLock,
    EXPECTED_OUTPUT_SCHEMA_REF,
    VALID_OUTCOMES,
)


__all__ = [
    # Agent
    "JudgeAgent",
    "AgentContext",
    "BaseAgent",
    # Verification - deterministic mapping
    "EvaluationVariables",
    "ConfidencePolicy",
    "extract_evaluation_variables",
    "check_validity",
    "check_conflicts",
    "determine_binary_outcome",
    "compute_confidence",
    "map_to_verdict",
    # Verification - schema lock
    "SchemaLock",
    "EXPECTED_OUTPUT_SCHEMA_REF",
    "VALID_OUTCOMES",
]