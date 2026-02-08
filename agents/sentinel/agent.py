"""
Sentinel Agent

Verifies complete proof bundles for correctness.

The Sentinel is the final verification step before on-chain settlement.
It ensures:
1. All artifacts are present and complete
2. All hashes match their computed values
3. All IDs and references are consistent
4. Evidence meets provenance requirements
5. Reasoning trace is valid and supports the verdict

Two implementations:
- SentinelStrict: Full verification (production)
- SentinelBasic: Quick validation (development/testing)
"""

from __future__ import annotations

import hashlib
from typing import Any, TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, AgentStep, BaseAgent
from agents.registry import register_agent
from core.schemas import (
    CheckResult,
    DeterministicVerdict,
    EvidenceBundle,
    ProofBundle,
    PromptSpec,
    ReasoningTrace,
    SentinelReport,
    ToolExecutionLog,
    ToolPlan,
    VerificationResult,
)

from .engine import VerificationEngine

if TYPE_CHECKING:
    from agents.context import AgentContext


class SentinelStrict(BaseAgent):
    """
    Strict Sentinel Agent.
    
    Performs comprehensive verification of proof bundles.
    Primary production implementation.
    
    Features:
    - Full completeness checks
    - Hash verification
    - Consistency validation
    - Provenance tier checks
    - Reasoning validation
    """
    
    _name = "SentinelStrict"
    _version = "v1"
    _capabilities = {AgentCapability.DETERMINISTIC, AgentCapability.REPLAY}
    
    def __init__(self, **kwargs) -> None:
        """Initialize strict sentinel."""
        super().__init__(**kwargs)
        self.engine = VerificationEngine(strict_mode=True)
    
    def run(
        self,
        ctx: "AgentContext",
        bundle: ProofBundle,
    ) -> AgentResult:
        """
        Verify a proof bundle.
        
        Args:
            ctx: Agent context
            bundle: The proof bundle to verify
        
        Returns:
            AgentResult with (VerificationResult, SentinelReport) as output
        """
        ctx.info(f"SentinelStrict verifying bundle {bundle.bundle_id}")
        
        try:
            # Run verification
            result, report = self.engine.verify(ctx, bundle)
            
            ctx.info(
                f"Verification complete: {report.passed_checks}/{report.total_checks} passed, "
                f"verified={report.verified}"
            )
            
            return AgentResult(
                output=(result, report),
                verification=result,
                receipts=ctx.get_receipt_refs(),
                metadata={
                    "sentinel": "strict",
                    "bundle_id": bundle.bundle_id,
                    "market_id": bundle.market_id,
                    "verified": report.verified,
                    "total_checks": report.total_checks,
                    "passed_checks": report.passed_checks,
                    "failed_checks": report.failed_checks,
                    "pass_rate": report.pass_rate,
                },
            )
            
        except Exception as e:
            ctx.error(f"Sentinel verification failed: {e}")
            return AgentResult.failure(error=str(e))
    
    def run_from_artifacts(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
        evidence_bundle: EvidenceBundle,
        reasoning_trace: ReasoningTrace,
        verdict: DeterministicVerdict,
        execution_log: ToolExecutionLog | None = None,
    ) -> AgentResult:
        """
        Verify using individual artifacts (builds ProofBundle internally).
        
        Convenience method when artifacts are available separately.
        """
        # Build proof bundle
        bundle = ProofBundle(
            bundle_id=self._generate_bundle_id(verdict.market_id),
            market_id=verdict.market_id,
            prompt_spec=prompt_spec,
            tool_plan=tool_plan,
            evidence_bundle=evidence_bundle,
            execution_log=execution_log,
            reasoning_trace=reasoning_trace,
            verdict=verdict,
        )
        
        return self.run(ctx, bundle)
    
    def _generate_bundle_id(self, market_id: str) -> str:
        """Generate proof bundle ID."""
        hash_bytes = hashlib.sha256(f"proof:{market_id}".encode()).digest()
        return f"proof_{hash_bytes[:8].hex()}"


class SentinelBasic(BaseAgent):
    """
    Basic Sentinel Agent.
    
    Performs quick validation checks without full hash verification.
    Useful for development and testing.
    
    Features:
    - Completeness checks only
    - Basic consistency checks
    - No hash re-computation
    """
    
    _name = "SentinelBasic"
    _version = "v1"
    _capabilities = {AgentCapability.DETERMINISTIC, AgentCapability.REPLAY}
    
    def run(
        self,
        ctx: "AgentContext",
        bundle: ProofBundle,
    ) -> AgentResult:
        """
        Perform basic verification of a proof bundle.
        
        Args:
            ctx: Agent context
            bundle: The proof bundle to verify
        
        Returns:
            AgentResult with (VerificationResult, SentinelReport) as output
        """
        ctx.info(f"SentinelBasic verifying bundle {bundle.bundle_id}")
        
        checks: list[CheckResult] = []
        report = SentinelReport(
            report_id=f"basic_{bundle.bundle_id}",
            bundle_id=bundle.bundle_id,
            market_id=bundle.market_id,
            verified=True,
        )
        
        # Basic completeness check
        if bundle.is_complete:
            report.add_check("completeness", "bundle_complete", True, "All artifacts present")
            checks.append(CheckResult.passed("bundle_complete", "All artifacts present"))
        else:
            report.add_check("completeness", "bundle_complete", False, "Bundle is incomplete")
            checks.append(CheckResult.failed("bundle_complete", "Bundle is incomplete"))
        
        # Basic market ID check
        ids_match = (
            bundle.prompt_spec.market_id == bundle.market_id and
            bundle.evidence_bundle.market_id == bundle.market_id and
            bundle.reasoning_trace.market_id == bundle.market_id and
            bundle.verdict.market_id == bundle.market_id
        )
        
        if ids_match:
            report.add_check("consistency", "market_ids_match", True, "All market IDs match")
            checks.append(CheckResult.passed("market_ids_match", "All market IDs match"))
        else:
            report.add_check("consistency", "market_ids_match", False, "Market IDs do not match")
            checks.append(CheckResult.failed("market_ids_match", "Market IDs do not match"))
        
        # Basic verdict check
        if bundle.verdict.outcome in ("YES", "NO", "INVALID"):
            report.add_check("verdict", "outcome_valid", True, f"Valid outcome: {bundle.verdict.outcome}")
            checks.append(CheckResult.passed("outcome_valid", f"Valid outcome: {bundle.verdict.outcome}"))
        else:
            report.add_check("verdict", "outcome_valid", False, f"Invalid outcome: {bundle.verdict.outcome}")
            checks.append(CheckResult.failed("outcome_valid", f"Invalid outcome: {bundle.verdict.outcome}"))
        
        report.created_at = ctx.now()
        
        ok = report.verified and report.failed_checks == 0
        result = VerificationResult(ok=ok, checks=checks)
        
        ctx.info(f"Basic verification complete: verified={report.verified}")
        
        return AgentResult(
            output=(result, report),
            verification=result,
            receipts=ctx.get_receipt_refs(),
            metadata={
                "sentinel": "basic",
                "bundle_id": bundle.bundle_id,
                "verified": report.verified,
            },
        )


def get_sentinel(ctx: "AgentContext", *, strict: bool = True) -> BaseAgent:
    """
    Get the appropriate sentinel based on requirements.
    
    Args:
        ctx: Agent context
        strict: If True, use strict verification
    
    Returns:
        SentinelStrict or SentinelBasic
    """
    if strict:
        return SentinelStrict()
    return SentinelBasic()


def verify_proof(
    ctx: "AgentContext",
    bundle: ProofBundle,
    *,
    strict: bool = True,
) -> AgentResult:
    """
    Convenience function to verify a proof bundle.
    
    Args:
        ctx: Agent context
        bundle: The proof bundle to verify
        strict: If True, use strict verification
    
    Returns:
        AgentResult with (VerificationResult, SentinelReport) as output
    """
    sentinel = get_sentinel(ctx, strict=strict)
    return sentinel.run(ctx, bundle)


def verify_artifacts(
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    tool_plan: ToolPlan,
    evidence_bundle: EvidenceBundle,
    reasoning_trace: ReasoningTrace,
    verdict: DeterministicVerdict,
    execution_log: ToolExecutionLog | None = None,
    *,
    strict: bool = True,
) -> AgentResult:
    """
    Convenience function to verify individual artifacts.
    
    Args:
        ctx: Agent context
        prompt_spec: The prompt specification
        tool_plan: The tool plan
        evidence_bundle: The evidence bundle
        reasoning_trace: The reasoning trace
        verdict: The final verdict
        execution_log: Optional execution log
        strict: If True, use strict verification
    
    Returns:
        AgentResult with (VerificationResult, SentinelReport) as output
    """
    sentinel = SentinelStrict()
    return sentinel.run_from_artifacts(
        ctx,
        prompt_spec,
        tool_plan,
        evidence_bundle,
        reasoning_trace,
        verdict,
        execution_log,
    )


def build_proof_bundle(
    prompt_spec: PromptSpec,
    tool_plan: ToolPlan,
    evidence_bundle: EvidenceBundle,
    reasoning_trace: ReasoningTrace,
    verdict: DeterministicVerdict,
    execution_log: ToolExecutionLog | None = None,
) -> ProofBundle:
    """
    Build a ProofBundle from individual artifacts.
    
    Args:
        prompt_spec: The prompt specification
        tool_plan: The tool plan
        evidence_bundle: The evidence bundle
        reasoning_trace: The reasoning trace
        verdict: The final verdict
        execution_log: Optional execution log
    
    Returns:
        Complete ProofBundle
    """
    hash_bytes = hashlib.sha256(f"proof:{verdict.market_id}".encode()).digest()
    bundle_id = f"proof_{hash_bytes[:8].hex()}"
    
    return ProofBundle(
        bundle_id=bundle_id,
        market_id=verdict.market_id,
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence_bundle=evidence_bundle,
        execution_log=execution_log,
        reasoning_trace=reasoning_trace,
        verdict=verdict,
    )


# Register agents with the global registry
def _register_agents() -> None:
    """Register sentinel agents."""
    register_agent(
        step=AgentStep.SENTINEL,
        name="SentinelStrict",
        factory=lambda ctx: SentinelStrict(),
        capabilities={AgentCapability.DETERMINISTIC, AgentCapability.REPLAY},
        priority=100,  # Primary
        metadata={"description": "Strict sentinel that verifies all pipeline artifacts — hash integrity, schema compliance, and cross-reference consistency. Fails on any violation."},
    )

    register_agent(
        step=AgentStep.SENTINEL,
        name="SentinelBasic",
        factory=lambda ctx: SentinelBasic(),
        capabilities={AgentCapability.DETERMINISTIC, AgentCapability.REPLAY},
        priority=50,  # Fallback
        is_fallback=True,
        metadata={"description": "Basic sentinel with minimal verification checks. Validates core artifact structure without exhaustive cross-referencing."},
    )


# Auto-register on import
_register_agents()
