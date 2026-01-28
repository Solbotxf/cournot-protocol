"""
Module 07 - Judge Agent
Produces a DeterministicVerdict from (PromptSpec, EvidenceBundle, ReasoningTrace).

Owner: Protocol/Judging Engineer
Module ID: M07

The Judge is responsible for:
1. Interpreting MarketSpec.event_definition and resolution_rules
2. Consuming Auditor's evaluation_variables deterministically
3. Mapping to Outcome ∈ {YES, NO, INVALID}
4. Computing confidence ∈ [0,1] from defined policy
5. Emitting a DeterministicVerdict with references to commitments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from core.crypto.hashing import hash_canonical, to_hex
from core.merkle import build_merkle_root
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import (
    ChallengeRef,
    CheckResult,
    VerificationResult,
)

from .verification.deterministic_mapping import map_to_verdict
from .verification.schema_lock import SchemaLock


@dataclass
class AgentContext:
    """
    Context passed to agent operations.
    
    Can be extended with tracing, logging, or other cross-cutting concerns.
    """
    trace_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol for agent implementations."""
    role: str
    name: str


class JudgeAgent:
    """
    Judge Agent - produces deterministic verdicts.
    
    The Judge executes strictly-defined resolution rules and confidence policy
    to produce a DeterministicVerdict that is reproducible (same inputs → same verdict).
    
    Usage:
        judge = JudgeAgent()
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        if result.ok:
            print(f"Verdict: {verdict.outcome} with confidence {verdict.confidence}")
        else:
            print(f"Judge failed: {result.get_error_messages()}")
    """
    
    role: str = "judge"
    
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        require_strict_mode: bool = True,
        compute_commitments: bool = True,
    ):
        """
        Initialize the Judge agent.
        
        Args:
            name: Optional name for this agent instance
            require_strict_mode: Whether to require strict_mode in PromptSpec
            compute_commitments: Whether to compute hash commitments for the verdict
        """
        self.name = name or "judge_v1"
        self.require_strict_mode = require_strict_mode
        self.compute_commitments = compute_commitments
    
    def judge(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        trace: ReasoningTrace,
        *,
        ctx: Optional[AgentContext] = None,
        resolution_time: Optional[datetime] = None,
    ) -> tuple[DeterministicVerdict, VerificationResult]:
        """
        Produce a deterministic verdict from the inputs.
        
        Args:
            prompt_spec: The prompt specification defining the market and rules
            evidence: The evidence bundle from the Collector
            trace: The reasoning trace from the Auditor
            ctx: Optional agent context for tracing/logging
            resolution_time: Optional resolution timestamp (None for deterministic hashing)
        
        Returns:
            Tuple of (DeterministicVerdict, VerificationResult)
            
            The VerificationResult contains all check results and any challenge
            reference if the verdict is INVALID due to failures.
        """
        all_checks: list[CheckResult] = []
        challenge: Optional[ChallengeRef] = None
        
        # Step 1: Schema lock checks
        schema_checks = self._run_schema_checks(prompt_spec, trace)
        all_checks.extend(schema_checks)
        
        # Check if schema validation failed
        schema_failed = any(not c.ok for c in schema_checks)
        if schema_failed:
            # Find the first failing check to determine challenge type
            for check in schema_checks:
                if not check.ok:
                    challenge = self._determine_schema_challenge(check)
                    break
            
            # Return INVALID verdict
            verdict = self._create_invalid_verdict(
                prompt_spec=prompt_spec,
                resolution_time=resolution_time,
                rule_id="R_VALIDITY",
            )
            
            return verdict, VerificationResult.failure(
                checks=all_checks,
                challenge=challenge,
            )
        
        # Step 2: Map to verdict using deterministic mapping
        verdict, mapping_checks = map_to_verdict(
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            resolution_time=resolution_time,
        )
        all_checks.extend(mapping_checks)
        
        # Step 3: Compute commitments if enabled
        if self.compute_commitments:
            verdict = self._add_commitments(
                verdict=verdict,
                prompt_spec=prompt_spec,
                evidence=evidence,
                trace=trace,
            )
        
        # Step 4: Determine challenge reference if INVALID
        if verdict.outcome == "INVALID":
            challenge = self._determine_verdict_challenge(all_checks, trace)
        
        # Step 5: Build verification result
        all_ok = all(c.ok for c in all_checks) and verdict.outcome != "INVALID"
        
        return verdict, VerificationResult(
            ok=all_ok,
            checks=all_checks,
            challenge=challenge,
            error=None,
        )
    
    def _run_schema_checks(
        self,
        prompt_spec: PromptSpec,
        trace: ReasoningTrace,
    ) -> list[CheckResult]:
        """Run schema lock validation checks."""
        checks: list[CheckResult] = []
        
        # Validate PromptSpec
        prompt_checks = SchemaLock.verify_prompt_spec(
            prompt_spec,
            require_strict_mode=self.require_strict_mode,
        )
        checks.extend(prompt_checks)
        
        # Validate ReasoningTrace
        trace_checks = SchemaLock.verify_trace(
            trace,
            require_evaluation_variables=True,
        )
        checks.extend(trace_checks)
        
        return checks
    
    def _determine_schema_challenge(self, failed_check: CheckResult) -> ChallengeRef:
        """Determine the appropriate challenge reference for a schema failure."""
        check_id = failed_check.check_id
        
        if "trace" in check_id or "step" in check_id or "evaluation" in check_id:
            # Trace-related failure
            step_id = failed_check.details.get("step_id")
            return ChallengeRef.reasoning_challenge(
                step_id=step_id or "unknown",
                reason=failed_check.message,
            )
        elif "output_schema" in check_id or "strict_mode" in check_id:
            # PromptSpec configuration failure
            return ChallengeRef.bundle_challenge(
                reason=f"Schema lock failure: {failed_check.message}",
            )
        else:
            # Generic bundle challenge
            return ChallengeRef.bundle_challenge(
                reason=failed_check.message,
            )
    
    def _determine_verdict_challenge(
        self,
        checks: list[CheckResult],
        trace: ReasoningTrace,
    ) -> Optional[ChallengeRef]:
        """Determine the appropriate challenge reference for an INVALID verdict."""
        # Find the check that caused the INVALID outcome
        for check in checks:
            if not check.ok:
                # Check what kind of failure
                if "evaluation_variables" in check.check_id:
                    # Missing evaluation variables -> reasoning challenge
                    final_step = trace.steps[-1] if trace.steps else None
                    return ChallengeRef.reasoning_challenge(
                        step_id=final_step.step_id if final_step else "unknown",
                        reason=check.message,
                    )
                elif "validity" in check.check_id:
                    # Validity failure could be evidence or reasoning
                    if "insufficient_evidence" in check.message.lower():
                        return ChallengeRef.evidence_challenge(
                            evidence_id="bundle",
                            reason=check.message,
                        )
                    else:
                        return ChallengeRef.reasoning_challenge(
                            step_id="final",
                            reason=check.message,
                        )
                elif "conflict" in check.check_id:
                    # Conflict -> could be evidence issue
                    return ChallengeRef.evidence_challenge(
                        evidence_id="bundle",
                        reason=check.message,
                    )
                elif "binary_decision" in check.check_id:
                    # Decision failure -> reasoning challenge
                    return ChallengeRef.reasoning_challenge(
                        step_id="final",
                        reason=check.message,
                    )
        
        return None
    
    def _create_invalid_verdict(
        self,
        prompt_spec: PromptSpec,
        resolution_time: Optional[datetime],
        rule_id: str,
    ) -> DeterministicVerdict:
        """Create an INVALID verdict for early failures."""
        return DeterministicVerdict(
            market_id=prompt_spec.market.market_id,
            outcome="INVALID",
            confidence=0.0,
            resolution_time=resolution_time or datetime.now(timezone.utc),
            resolution_rule_id=rule_id,
        )
    
    def _add_commitments(
        self,
        verdict: DeterministicVerdict,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        trace: ReasoningTrace,
    ) -> DeterministicVerdict:
        """Add hash commitments to the verdict."""
        try:
            # Compute prompt_spec_hash
            # Note: Ensure created_at is None for deterministic hashing
            prompt_spec_hash = to_hex(hash_canonical(prompt_spec))
            
            # Compute evidence_root
            evidence_leaves = [hash_canonical(item) for item in evidence.items]
            evidence_root = to_hex(build_merkle_root(evidence_leaves))
            
            # Compute reasoning_root
            reasoning_leaves = [hash_canonical(step) for step in trace.steps]
            reasoning_root = to_hex(build_merkle_root(reasoning_leaves))
            
            # Return updated verdict with commitments
            return DeterministicVerdict(
                schema_version=verdict.schema_version,
                market_id=verdict.market_id,
                outcome=verdict.outcome,
                confidence=verdict.confidence,
                resolution_time=verdict.resolution_time,
                resolution_rule_id=verdict.resolution_rule_id,
                prompt_spec_hash=prompt_spec_hash,
                evidence_root=evidence_root,
                reasoning_root=reasoning_root,
                selected_leaf_refs=verdict.selected_leaf_refs,
                metadata=verdict.metadata,
            )
        except Exception:
            # If commitment computation fails, return original verdict
            # This shouldn't happen in normal operation
            return verdict
    
    def __repr__(self) -> str:
        return f"JudgeAgent(name={self.name!r}, strict_mode={self.require_strict_mode})"


__all__ = [
    "JudgeAgent",
    "AgentContext",
    "BaseAgent",
]