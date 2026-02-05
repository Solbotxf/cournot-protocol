"""
Judge Agent

Finalizes verdicts from reasoning traces.

Two implementations:
- JudgeLLM: Uses LLM to review and finalize (production)
- JudgeRuleBased: Deterministic rule-based finalization (fallback)

Selection logic:
- If ctx.llm is available → use LLM judge
- Otherwise → use rule-based judge
"""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, AgentStep, BaseAgent
from agents.registry import register_agent
from core.schemas import (
    CheckResult,
    DeterministicVerdict,
    EvidenceBundle,
    PromptSpec,
    ReasoningTrace,
    VerificationResult,
)

from .verdict_builder import VerdictBuilder, VerdictValidator
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from agents.context import AgentContext


class JudgeLLM(BaseAgent):
    """
    LLM-based Judge Agent.
    
    Uses LLM to review reasoning traces and finalize verdicts.
    Primary production implementation.
    
    Features:
    - Reviews auditor reasoning for soundness
    - Can adjust confidence based on review
    - Provides detailed justification
    - Catches reasoning errors
    """
    
    _name = "JudgeLLM"
    _version = "v1"
    _capabilities = {AgentCapability.LLM}
    
    MAX_RETRIES = 2
    
    def __init__(self, *, strict_mode: bool = True, **kwargs) -> None:
        """
        Initialize LLM judge.
        
        Args:
            strict_mode: If True, exclude timestamps from hashing
        """
        super().__init__(**kwargs)
        self.builder = VerdictBuilder(strict_mode=strict_mode)
        self.validator = VerdictValidator()
        self.strict_mode = strict_mode
    
    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        reasoning_trace: ReasoningTrace,
    ) -> AgentResult:
        """
        Finalize verdict from reasoning trace.
        
        Args:
            ctx: Agent context with LLM client
            prompt_spec: The prompt specification
            evidence_bundle: Collected evidence
            reasoning_trace: The reasoning trace from auditor
        
        Returns:
            AgentResult with DeterministicVerdict as output
        """
        ctx.info(f"JudgeLLM reviewing trace {reasoning_trace.trace_id}")
        
        # Check LLM client
        if not ctx.llm:
            return AgentResult.failure(
                error="LLM client required for LLM judge",
            )
        
        try:
            # Get LLM review
            review = self._get_llm_review(ctx, prompt_spec, evidence_bundle, reasoning_trace)
            
            # Extract verdict parameters from review
            outcome = self._extract_outcome(review, reasoning_trace, prompt_spec)
            confidence = self._extract_confidence(review, reasoning_trace)
            rule_id = self._extract_rule_id(review, reasoning_trace)
            
            # Build final verdict
            verdict = self.builder.build(
                ctx,
                prompt_spec,
                evidence_bundle,
                reasoning_trace,
                override_outcome=outcome,
                override_confidence=confidence,
                override_rule_id=rule_id,
            )
            
            # Add LLM review to metadata
            verdict.metadata["llm_review"] = review
            
            # Validate verdict
            verification = self._validate_verdict(verdict, prompt_spec, evidence_bundle, reasoning_trace)
            
            ctx.info(f"Verdict finalized: {verdict.outcome} with {verdict.confidence:.0%} confidence")
            
            return AgentResult(
                output=verdict,
                verification=verification,
                receipts=ctx.get_receipt_refs(),
                metadata={
                    "judge": "llm",
                    "strict_mode": self.strict_mode,
                    "market_id": verdict.market_id,
                    "outcome": verdict.outcome,
                    "confidence": verdict.confidence,
                    "rule_id": verdict.resolution_rule_id,
                    "reasoning_valid": review.get("reasoning_valid", True),
                },
            )
            
        except Exception as e:
            ctx.error(f"LLM judge failed: {e}")
            return AgentResult.failure(error=str(e))
    
    def _get_llm_review(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        trace: ReasoningTrace,
    ) -> dict[str, Any]:
        """Get LLM review of the reasoning trace."""
        # Format reasoning steps
        steps_text = []
        for step in trace.steps[:10]:  # Limit to 10 steps
            steps_text.append(f"- [{step.step_type}] {step.description}")
            if step.conclusion:
                steps_text.append(f"  Conclusion: {step.conclusion}")
        
        # Format conflicts
        conflicts_text = "None detected."
        if trace.conflicts:
            conflicts_text = "\n".join([
                f"- {c.description} → Resolved: {c.resolution}"
                for c in trace.conflicts
            ])
        
        # Format resolution rules
        rules_text = "\n".join([
            f"- {r.rule_id}: {r.description}"
            for r in prompt_spec.market.resolution_rules.get_sorted_rules()
        ])
        
        # Format possible outcomes
        if prompt_spec.is_multi_choice:
            possible_outcomes_text = f"Multi-choice market. Allowed outcomes: {', '.join(prompt_spec.possible_outcomes)}"
        else:
            possible_outcomes_text = "Binary market. Allowed outcomes: YES, NO"

        # Format assumptions from prompt_spec.extra
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_text = "\n".join([f"- {a}" for a in assumptions_list]) if assumptions_list else "None"

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=prompt_spec.market.question,
            event_definition=prompt_spec.market.event_definition,
            assumptions=assumptions_text,
            preliminary_outcome=trace.preliminary_outcome or "UNCERTAIN",
            preliminary_confidence=trace.preliminary_confidence or 0.5,
            recommended_rule=trace.recommended_rule_id or "None",
            reasoning_summary=trace.reasoning_summary or "No summary available.",
            reasoning_steps="\n".join(steps_text),
            conflicts=conflicts_text,
            evidence_summary=trace.evidence_summary or f"{len(evidence_bundle.items)} evidence items.",
            resolution_rules=rules_text,
            possible_outcomes=possible_outcomes_text,
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        # Get LLM response
        response = ctx.llm.chat(messages)
        raw_output = response.content
        
        # Parse JSON with retries
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self._extract_json(raw_output)
            except json.JSONDecodeError as e:
                if attempt < self.MAX_RETRIES:
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": f"Invalid JSON: {e}. Please return valid JSON only."})
                    response = ctx.llm.chat(messages)
                    raw_output = response.content
                else:
                    # Return default review on failure
                    return {
                        "outcome": trace.preliminary_outcome or "INVALID",
                        "confidence": trace.preliminary_confidence or 0.5,
                        "resolution_rule_id": trace.recommended_rule_id or "R_DEFAULT",
                        "reasoning_valid": True,
                        "reasoning_issues": [],
                        "final_justification": "LLM review failed, using preliminary verdict.",
                    }
        
        return {}
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        
        # Try bare JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        
        return json.loads(text.strip())
    
    def _extract_outcome(self, review: dict[str, Any], trace: ReasoningTrace, prompt_spec: PromptSpec) -> str:
        """Extract outcome from review."""
        outcome = review.get("outcome", trace.preliminary_outcome)
        if outcome and prompt_spec.market.is_valid_outcome(outcome):
            return outcome
        return "INVALID"
    
    def _extract_confidence(self, review: dict[str, Any], trace: ReasoningTrace) -> float:
        """Extract confidence from review."""
        confidence = review.get("confidence", trace.preliminary_confidence or 0.5)
        try:
            return max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            return trace.preliminary_confidence or 0.5
    
    def _extract_rule_id(self, review: dict[str, Any], trace: ReasoningTrace) -> str:
        """Extract rule ID from review."""
        rule_id = review.get("resolution_rule_id", trace.recommended_rule_id)
        return rule_id or "R_DEFAULT"
    
    def _validate_verdict(
        self,
        verdict: DeterministicVerdict,
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        trace: ReasoningTrace,
    ) -> VerificationResult:
        """Validate the verdict."""
        checks: list[CheckResult] = []
        
        # Use validator
        is_valid, errors = self.validator.validate(verdict, prompt_spec, evidence_bundle, trace)
        
        if is_valid:
            checks.append(CheckResult.passed(
                check_id="verdict_valid",
                message="Verdict passes all validation checks",
            ))
        else:
            for error in errors:
                checks.append(CheckResult.failed(
                    check_id="validation_error",
                    message=error,
                ))
        
        # Check hashes present
        if verdict.prompt_spec_hash and verdict.evidence_root and verdict.reasoning_root:
            checks.append(CheckResult.passed(
                check_id="hashes_present",
                message="All artifact hashes computed",
            ))
        
        # Check confidence appropriate for outcome
        if verdict.outcome != "INVALID" and verdict.confidence < 0.55:
            checks.append(CheckResult.warning(
                check_id="confidence_threshold",
                message=f"Low confidence {verdict.confidence:.0%} for {verdict.outcome} outcome",
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="confidence_threshold",
                message=f"Confidence {verdict.confidence:.0%} appropriate for {verdict.outcome}",
            ))
        
        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)


class JudgeRuleBased(BaseAgent):
    """
    Rule-based Judge Agent.
    
    Uses deterministic rules to finalize verdicts.
    Fully deterministic - same input always produces same output.
    
    Used when:
    - LLM is unavailable
    - Deterministic behavior is required
    - Testing and development
    """
    
    _name = "JudgeRuleBased"
    _version = "v1"
    _capabilities = {AgentCapability.DETERMINISTIC, AgentCapability.REPLAY}
    
    def __init__(self, *, strict_mode: bool = True, **kwargs) -> None:
        """
        Initialize rule-based judge.
        
        Args:
            strict_mode: If True, exclude timestamps from hashing
        """
        super().__init__(**kwargs)
        self.builder = VerdictBuilder(strict_mode=strict_mode)
        self.validator = VerdictValidator()
        self.strict_mode = strict_mode
    
    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        reasoning_trace: ReasoningTrace,
    ) -> AgentResult:
        """
        Finalize verdict from reasoning trace.
        
        Args:
            ctx: Agent context
            prompt_spec: The prompt specification
            evidence_bundle: Collected evidence
            reasoning_trace: The reasoning trace from auditor
        
        Returns:
            AgentResult with DeterministicVerdict as output
        """
        ctx.info(f"JudgeRuleBased reviewing trace {reasoning_trace.trace_id}")
        
        try:
            # Apply rule-based review
            outcome, confidence, rule_id = self._apply_rules(
                prompt_spec,
                evidence_bundle,
                reasoning_trace,
            )
            
            # Build verdict
            verdict = self.builder.build(
                ctx,
                prompt_spec,
                evidence_bundle,
                reasoning_trace,
                override_outcome=outcome,
                override_confidence=confidence,
                override_rule_id=rule_id,
            )
            
            # Validate
            verification = self._validate_verdict(verdict, prompt_spec, evidence_bundle, reasoning_trace)
            
            ctx.info(f"Verdict finalized: {verdict.outcome} with {verdict.confidence:.0%} confidence")
            
            return AgentResult(
                output=verdict,
                verification=verification,
                receipts=ctx.get_receipt_refs(),
                metadata={
                    "judge": "rule_based",
                    "strict_mode": self.strict_mode,
                    "market_id": verdict.market_id,
                    "outcome": verdict.outcome,
                    "confidence": verdict.confidence,
                    "rule_id": verdict.resolution_rule_id,
                },
            )
            
        except Exception as e:
            ctx.error(f"Rule-based judge failed: {e}")
            return AgentResult.failure(error=str(e))
    
    def _apply_rules(
        self,
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        trace: ReasoningTrace,
    ) -> tuple[str, float, str]:
        """
        Apply deterministic rules to finalize verdict.

        Returns:
            Tuple of (outcome, confidence, rule_id)
        """
        market = prompt_spec.market

        # Start with preliminary values
        outcome = trace.preliminary_outcome or "UNCERTAIN"
        confidence = trace.preliminary_confidence or 0.5
        rule_id = trace.recommended_rule_id or "R_DEFAULT"

        # Rule 1: Convert UNCERTAIN to INVALID
        if outcome == "UNCERTAIN":
            outcome = "INVALID"
            rule_id = "R_INVALID_FALLBACK"

        # Rule 2: Enforce minimum confidence for definitive outcomes
        if outcome != "INVALID":
            if confidence < 0.55:
                outcome = "INVALID"
                rule_id = "R_INVALID_FALLBACK"

        # Rule 3: Check for unresolved conflicts
        unresolved_conflicts = [
            c for c in trace.conflicts
            if not c.winning_evidence_id
        ]
        if unresolved_conflicts and outcome != "INVALID":
            confidence -= 0.1 * len(unresolved_conflicts)
            confidence = max(0.0, confidence)

            if confidence < 0.55:
                outcome = "INVALID"
                rule_id = "R_CONFLICT"

        # Rule 4: Check evidence coverage
        valid_evidence = evidence_bundle.get_valid_evidence()
        if not valid_evidence and outcome != "INVALID":
            outcome = "INVALID"
            confidence = 0.3
            rule_id = "R_INVALID_FALLBACK"

        # Rule 5: Boost confidence for high-tier evidence
        if valid_evidence and outcome != "INVALID":
            max_tier = max(e.provenance.tier for e in valid_evidence)
            if max_tier >= 3:
                confidence = min(1.0, confidence + 0.05)

        # Ensure outcome is valid for this market
        if not market.is_valid_outcome(outcome):
            outcome = "INVALID"

        return outcome, confidence, rule_id
    
    def _validate_verdict(
        self,
        verdict: DeterministicVerdict,
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        trace: ReasoningTrace,
    ) -> VerificationResult:
        """Validate the verdict."""
        checks: list[CheckResult] = []
        
        # Use validator
        is_valid, errors = self.validator.validate(verdict, prompt_spec, evidence_bundle, trace)
        
        if is_valid:
            checks.append(CheckResult.passed(
                check_id="verdict_valid",
                message="Verdict passes all validation checks",
            ))
        else:
            for error in errors:
                checks.append(CheckResult.failed(
                    check_id="validation_error",
                    message=error,
                ))
        
        # Check deterministic format
        if verdict.resolution_time is None and self.strict_mode:
            checks.append(CheckResult.passed(
                check_id="deterministic_format",
                message="Verdict excludes timestamp for determinism",
            ))
        
        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)


def get_judge(ctx: "AgentContext", *, prefer_llm: bool = True) -> BaseAgent:
    """
    Get the appropriate judge based on context.
    
    Args:
        ctx: Agent context
        prefer_llm: If True and LLM is available, use LLM judge
    
    Returns:
        JudgeLLM or JudgeRuleBased
    """
    if prefer_llm and ctx.llm is not None:
        return JudgeLLM()
    return JudgeRuleBased()


def judge_verdict(
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundle: EvidenceBundle,
    reasoning_trace: ReasoningTrace,
    *,
    prefer_llm: bool = True,
) -> AgentResult:
    """
    Convenience function to finalize verdict.
    
    Args:
        ctx: Agent context
        prompt_spec: The prompt specification
        evidence_bundle: Collected evidence
        reasoning_trace: The reasoning trace from auditor
        prefer_llm: If True and LLM is available, use LLM judge
    
    Returns:
        AgentResult with DeterministicVerdict as output
    """
    judge = get_judge(ctx, prefer_llm=prefer_llm)
    return judge.run(ctx, prompt_spec, evidence_bundle, reasoning_trace)


# Register agents with the global registry
def _register_agents() -> None:
    """Register judge agents."""
    register_agent(
        step=AgentStep.JUDGE,
        name="JudgeLLM",
        factory=lambda ctx: JudgeLLM(),
        capabilities={AgentCapability.LLM},
        priority=100,  # Primary
    )
    
    register_agent(
        step=AgentStep.JUDGE,
        name="JudgeRuleBased",
        factory=lambda ctx: JudgeRuleBased(),
        capabilities={AgentCapability.DETERMINISTIC, AgentCapability.REPLAY},
        priority=50,  # Fallback
        is_fallback=True,
    )


# Auto-register on import
_register_agents()
