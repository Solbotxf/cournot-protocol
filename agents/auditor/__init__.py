"""
Module 06 - Auditor (Reasoning Trace & Sanity Verification)

Owner: AI/Reasoning Engineer + Verification Engineer
Module ID: M06

Purpose: Given PromptSpec and EvidenceBundle, produce a deterministic,
checkable ReasoningTrace that:
- Extracts relevant claims from evidence
- Maps evidence to the market event_definition
- Runs contradiction and logic sanity checks
- Produces a trace whose steps reference evidence IDs only
- Passes trace policy verification or returns structured failures

The Auditor does not finalize the verdict; it produces the "audit trail"
for Judge + Sentinel verification.

Public API:
- AuditorAgent: Main agent class with audit() method
- audit: Convenience function for simple auditing
- AgentContext: Execution context dataclass
- Claim extraction: Claim, ClaimSet, ClaimExtractor, extract_claims
- Contradiction detection: Contradiction, ContradictionChecker, check_contradictions
- Verification: TracePolicyVerifier, LogicSanityChecker

Example:
    from agents.auditor import AuditorAgent, audit

    agent = AuditorAgent()
    trace, verification = agent.audit(prompt_spec, evidence_bundle)

    # Or use convenience function
    trace, verification = audit(prompt_spec, evidence_bundle)
"""

from .auditor_agent import (
    AgentContext,
    AuditorAgent,
    BaseAgent,
    audit,
)
from .reasoning import (
    Claim,
    ClaimExtractor,
    ClaimSet,
    Contradiction,
    ContradictionChecker,
    check_contradictions,
    extract_claims,
)
from .verification import (
    LogicSanityChecker,
    TracePolicyVerifier,
    check_logic_sanity,
    verify_trace_policy,
)

__all__ = [
    # Main agent
    "AuditorAgent",
    "BaseAgent",
    "AgentContext",
    "audit",
    # Claim extraction
    "Claim",
    "ClaimSet",
    "ClaimExtractor",
    "extract_claims",
    # Contradiction detection
    "Contradiction",
    "ContradictionChecker",
    "check_contradictions",
    # Verification
    "TracePolicyVerifier",
    "LogicSanityChecker",
    "verify_trace_policy",
    "check_logic_sanity",
]