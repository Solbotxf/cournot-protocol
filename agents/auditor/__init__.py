"""
Auditor Agent Module

Generates reasoning traces from evidence bundles.

Usage:
    from agents.auditor import audit_evidence, AuditorLLM, AuditorRuleBased
    
    # Using convenience function (auto-selects based on context)
    result = audit_evidence(ctx, prompt_spec, evidence_bundle)
    reasoning_trace = result.output
    
    # Using specific agent
    auditor = AuditorRuleBased()
    result = auditor.run(ctx, prompt_spec, evidence_bundle)
"""

from .agent import (
    AuditorLLM,
    AuditorRuleBased,
    audit_evidence,
    get_auditor,
)

from .reasoner import RuleBasedReasoner
from .llm_reasoner import LLMReasoner

__all__ = [
    # Agents
    "AuditorLLM",
    "AuditorRuleBased",
    # Functions
    "audit_evidence",
    "get_auditor",
    # Reasoners
    "RuleBasedReasoner",
    "LLMReasoner",
]
