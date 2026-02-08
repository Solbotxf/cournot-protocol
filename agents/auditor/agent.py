"""
Auditor Agent

Generates reasoning traces from evidence bundles.

Two implementations:
- AuditorLLM: Uses LLM for deep reasoning (production)
- AuditorRuleBased: Deterministic rule-based reasoning (fallback)

Selection logic:
- If ctx.llm is available → use LLM auditor
- Otherwise → use rule-based auditor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, AgentStep, BaseAgent
from agents.registry import register_agent
from core.schemas import (
    CheckResult,
    EvidenceBundle,
    PromptSpec,
    ReasoningTrace,
    VerificationResult,
)

from .llm_reasoner import LLMReasoner
from .reasoner import RuleBasedReasoner

if TYPE_CHECKING:
    from agents.context import AgentContext


class AuditorLLM(BaseAgent):
    """
    LLM-based Auditor Agent.
    
    Uses LLM to generate detailed reasoning traces with deep analysis.
    Primary production implementation.
    
    Features:
    - Full semantic understanding of evidence
    - Complex conflict resolution
    - Nuanced confidence assessment
    - Natural language reasoning summaries
    """
    
    _name = "AuditorLLM"
    _version = "v1"
    _capabilities = {AgentCapability.LLM}
    
    def __init__(self, *, strict_mode: bool = True, **kwargs) -> None:
        """
        Initialize LLM auditor.
        
        Args:
            strict_mode: If True, exclude timestamps from hashing
        """
        super().__init__(**kwargs)
        self.reasoner = LLMReasoner(strict_mode=strict_mode)
        self.strict_mode = strict_mode
    
    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundles: EvidenceBundle | list[EvidenceBundle],
    ) -> AgentResult:
        """
        Generate reasoning trace from evidence.

        Args:
            ctx: Agent context with LLM client
            prompt_spec: The prompt specification
            evidence_bundles: Collected evidence (single bundle or list)

        Returns:
            AgentResult with ReasoningTrace as output
        """
        # Normalize to list
        if isinstance(evidence_bundles, EvidenceBundle):
            evidence_bundles = [evidence_bundles]

        total_items = sum(len(b.items) for b in evidence_bundles)
        ctx.info(f"AuditorLLM analyzing {total_items} evidence items from {len(evidence_bundles)} bundles")

        # Check LLM client
        if not ctx.llm:
            return AgentResult.failure(
                error="LLM client required for LLM auditor",
            )

        try:
            # Generate reasoning trace
            trace = self.reasoner.reason(ctx, prompt_spec, evidence_bundles)

            # Validate output
            verification = self._validate_trace(trace, evidence_bundles)
            
            ctx.info(
                f"Reasoning complete: {trace.step_count} steps, "
                f"preliminary outcome: {trace.preliminary_outcome}"
            )
            
            return AgentResult(
                output=trace,
                verification=verification,
                receipts=ctx.get_receipt_refs(),
                metadata={
                    "auditor": "llm",
                    "strict_mode": self.strict_mode,
                    "trace_id": trace.trace_id,
                    "step_count": trace.step_count,
                    "conflict_count": len(trace.conflicts),
                    "preliminary_outcome": trace.preliminary_outcome,
                    "preliminary_confidence": trace.preliminary_confidence,
                },
            )
            
        except Exception as e:
            ctx.error(f"LLM reasoning failed: {e}")
            return AgentResult.failure(error=str(e))
    
    def _validate_trace(
        self,
        trace: ReasoningTrace,
        evidence_bundles: list[EvidenceBundle],
    ) -> VerificationResult:
        """Validate the reasoning trace."""
        checks: list[CheckResult] = []

        # Check 1: Has steps
        if trace.steps:
            checks.append(CheckResult.passed(
                check_id="has_steps",
                message=f"Trace has {trace.step_count} reasoning steps",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_steps",
                message="Trace has no reasoning steps",
            ))

        # Check 2: Has preliminary outcome
        if trace.preliminary_outcome:
            checks.append(CheckResult.passed(
                check_id="has_outcome",
                message=f"Preliminary outcome: {trace.preliminary_outcome}",
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="has_outcome",
                message="No preliminary outcome determined",
            ))

        # Check 3: References evidence
        ref_ids = set(trace.get_evidence_refs())
        # Collect evidence IDs from ALL bundles
        evidence_ids: set[str] = set()
        for bundle in evidence_bundles:
            evidence_ids.update(e.evidence_id for e in bundle.items)
        
        if ref_ids:
            invalid_refs = ref_ids - evidence_ids
            if invalid_refs:
                checks.append(CheckResult.warning(
                    check_id="evidence_refs",
                    message=f"References unknown evidence: {invalid_refs}",
                ))
            else:
                checks.append(CheckResult.passed(
                    check_id="evidence_refs",
                    message=f"References {len(ref_ids)} evidence items",
                ))
        else:
            checks.append(CheckResult.warning(
                check_id="evidence_refs",
                message="No evidence references in reasoning",
            ))
        
        # Check 4: Confidence in bounds
        if trace.preliminary_confidence is not None:
            if 0.0 <= trace.preliminary_confidence <= 1.0:
                checks.append(CheckResult.passed(
                    check_id="confidence_bounds",
                    message=f"Confidence: {trace.preliminary_confidence:.2f}",
                ))
            else:
                checks.append(CheckResult.failed(
                    check_id="confidence_bounds",
                    message=f"Confidence out of bounds: {trace.preliminary_confidence}",
                ))
        
        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)


class AuditorRuleBased(BaseAgent):
    """
    Rule-based Auditor Agent.
    
    Uses deterministic rule-based reasoning.
    Fully deterministic - same input always produces same output.
    
    Used when:
    - LLM is unavailable
    - Deterministic behavior is required
    - Testing and development
    """
    
    _name = "AuditorRuleBased"
    _version = "v1"
    _capabilities = {AgentCapability.DETERMINISTIC, AgentCapability.REPLAY}
    
    def __init__(self, *, strict_mode: bool = True, **kwargs) -> None:
        """
        Initialize rule-based auditor.
        
        Args:
            strict_mode: If True, exclude timestamps from hashing
        """
        super().__init__(**kwargs)
        self.reasoner = RuleBasedReasoner(strict_mode=strict_mode)
        self.strict_mode = strict_mode
    
    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundles: EvidenceBundle | list[EvidenceBundle],
    ) -> AgentResult:
        """
        Generate reasoning trace from evidence.

        Args:
            ctx: Agent context
            prompt_spec: The prompt specification
            evidence_bundles: Collected evidence (single bundle or list)

        Returns:
            AgentResult with ReasoningTrace as output
        """
        # Normalize to list
        if isinstance(evidence_bundles, EvidenceBundle):
            evidence_bundles = [evidence_bundles]

        total_items = sum(len(b.items) for b in evidence_bundles)
        ctx.info(f"AuditorRuleBased analyzing {total_items} evidence items from {len(evidence_bundles)} bundles")

        try:
            # Generate reasoning trace
            trace = self.reasoner.reason(ctx, prompt_spec, evidence_bundles)

            # Validate output
            verification = self._validate_trace(trace, evidence_bundles)
            
            ctx.info(
                f"Reasoning complete: {trace.step_count} steps, "
                f"preliminary outcome: {trace.preliminary_outcome}"
            )
            
            return AgentResult(
                output=trace,
                verification=verification,
                receipts=ctx.get_receipt_refs(),
                metadata={
                    "auditor": "rule_based",
                    "strict_mode": self.strict_mode,
                    "trace_id": trace.trace_id,
                    "step_count": trace.step_count,
                    "conflict_count": len(trace.conflicts),
                    "preliminary_outcome": trace.preliminary_outcome,
                    "preliminary_confidence": trace.preliminary_confidence,
                },
            )
            
        except Exception as e:
            ctx.error(f"Rule-based reasoning failed: {e}")
            return AgentResult.failure(error=str(e))
    
    def _validate_trace(
        self,
        trace: ReasoningTrace,
        evidence_bundles: list[EvidenceBundle],
    ) -> VerificationResult:
        """Validate the reasoning trace."""
        checks: list[CheckResult] = []

        # Check 1: Has steps
        if trace.steps:
            checks.append(CheckResult.passed(
                check_id="has_steps",
                message=f"Trace has {trace.step_count} reasoning steps",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_steps",
                message="Trace has no reasoning steps",
            ))

        # Check 2: Has preliminary outcome
        if trace.preliminary_outcome:
            checks.append(CheckResult.passed(
                check_id="has_outcome",
                message=f"Preliminary outcome: {trace.preliminary_outcome}",
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="has_outcome",
                message="No preliminary outcome determined",
            ))

        # Check 3: Trace ID format
        if trace.trace_id.startswith("trace_"):
            checks.append(CheckResult.passed(
                check_id="trace_id_format",
                message="Trace ID follows format",
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="trace_id_format",
                message=f"Unexpected trace ID format: {trace.trace_id}",
            ))
        
        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)


def get_auditor(ctx: "AgentContext", *, prefer_llm: bool = True) -> BaseAgent:
    """
    Get the appropriate auditor based on context.
    
    Args:
        ctx: Agent context
        prefer_llm: If True and LLM is available, use LLM auditor
    
    Returns:
        AuditorLLM or AuditorRuleBased
    """
    if prefer_llm and ctx.llm is not None:
        return AuditorLLM()
    return AuditorRuleBased()


def audit_evidence(
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: EvidenceBundle | list[EvidenceBundle],
    *,
    prefer_llm: bool = True,
) -> AgentResult:
    """
    Convenience function to audit evidence.

    Args:
        ctx: Agent context
        prompt_spec: The prompt specification
        evidence_bundles: Collected evidence (single bundle or list)
        prefer_llm: If True and LLM is available, use LLM auditor

    Returns:
        AgentResult with ReasoningTrace as output
    """
    auditor = get_auditor(ctx, prefer_llm=prefer_llm)
    return auditor.run(ctx, prompt_spec, evidence_bundles)


# Register agents with the global registry
def _register_agents() -> None:
    """Register auditor agents."""
    register_agent(
        step=AgentStep.AUDITOR,
        name="AuditorLLM",
        factory=lambda ctx: AuditorLLM(),
        capabilities={AgentCapability.LLM},
        priority=100,  # Primary
        metadata={"description": "LLM-powered auditor that performs deep semantic reasoning over evidence. Analyzes source credibility, identifies conflicts, and produces a structured reasoning trace."},
    )

    register_agent(
        step=AgentStep.AUDITOR,
        name="AuditorRuleBased",
        factory=lambda ctx: AuditorRuleBased(),
        capabilities={AgentCapability.DETERMINISTIC, AgentCapability.REPLAY},
        priority=50,  # Fallback
        is_fallback=True,
        metadata={"description": "Deterministic rule-based auditor. Applies predefined rules to evidence without LLM. Fully reproducible and suitable for replay verification."},
    )


# Auto-register on import
_register_agents()
