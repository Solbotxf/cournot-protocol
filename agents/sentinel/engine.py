"""
Verification Engine

Performs comprehensive verification of proof bundles.
Handles:
- Completeness checks (all artifacts present)
- Hash verification (artifact hashes match)
- Consistency checks (IDs match across artifacts)
- Provenance checks (evidence meets tier requirements)
- Reasoning checks (reasoning trace is valid)
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, TYPE_CHECKING

from core.schemas import (
    CheckResult,
    DeterministicVerdict,
    EvidenceBundle,
    ProofBundle,
    PromptSpec,
    ReasoningTrace,
    SentinelReport,
    ToolPlan,
    VerificationResult,
)
from core.schemas.canonical import dumps_canonical

if TYPE_CHECKING:
    from agents.context import AgentContext


class VerificationEngine:
    """
    Engine for verifying proof bundles.
    
    Performs multiple categories of checks:
    1. Completeness - all required artifacts present
    2. Hash - artifact hashes match stored values
    3. Consistency - IDs and references align
    4. Provenance - evidence meets tier requirements
    5. Reasoning - reasoning trace is valid
    """
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize verification engine.
        
        Args:
            strict_mode: If True, apply stricter validation rules
        """
        self.strict_mode = strict_mode
    
    def verify(
        self,
        ctx: "AgentContext",
        bundle: ProofBundle,
    ) -> tuple[VerificationResult, SentinelReport]:
        """
        Verify a complete proof bundle.
        
        Args:
            ctx: Agent context
            bundle: The proof bundle to verify
        
        Returns:
            Tuple of (VerificationResult, SentinelReport)
        """
        start_time = time.time()
        
        # Initialize report
        report = SentinelReport(
            report_id=self._generate_report_id(bundle.bundle_id),
            bundle_id=bundle.bundle_id,
            market_id=bundle.market_id,
            verified=True,  # Start optimistic, set to False on failure
        )
        
        # Initialize verification result
        checks: list[CheckResult] = []
        
        # Run all verification categories
        self._check_completeness(bundle, report, checks)
        self._check_hashes(bundle, report, checks)
        self._check_consistency(bundle, report, checks)
        self._check_provenance(bundle, report, checks)
        self._check_reasoning(bundle, report, checks)
        
        # Finalize timing
        elapsed_ms = (time.time() - start_time) * 1000
        report.verification_time_ms = elapsed_ms
        report.created_at = ctx.now()
        
        # Build final result
        ok = report.verified and report.failed_checks == 0
        result = VerificationResult(ok=ok, checks=checks)
        
        return result, report
    
    def _check_completeness(
        self,
        bundle: ProofBundle,
        report: SentinelReport,
        checks: list[CheckResult],
    ) -> None:
        """Check that all required artifacts are present."""
        # Check prompt_spec
        if bundle.prompt_spec:
            report.add_check("completeness", "prompt_spec_present", True, "PromptSpec is present")
            checks.append(CheckResult.passed("prompt_spec_present", "PromptSpec is present"))
        else:
            report.add_check("completeness", "prompt_spec_present", False, "PromptSpec is missing")
            checks.append(CheckResult.failed("prompt_spec_present", "PromptSpec is missing"))
        
        # Check tool_plan
        if bundle.tool_plan:
            report.add_check("completeness", "tool_plan_present", True, "ToolPlan is present")
            checks.append(CheckResult.passed("tool_plan_present", "ToolPlan is present"))
        else:
            report.add_check("completeness", "tool_plan_present", False, "ToolPlan is missing")
            checks.append(CheckResult.failed("tool_plan_present", "ToolPlan is missing"))
        
        # Check evidence_bundle
        if bundle.evidence_bundle:
            report.add_check("completeness", "evidence_bundle_present", True, "EvidenceBundle is present")
            checks.append(CheckResult.passed("evidence_bundle_present", "EvidenceBundle is present"))
            
            # Check evidence has items
            if bundle.evidence_bundle.items:
                report.add_check(
                    "completeness", "evidence_items_present", True,
                    f"Evidence bundle has {len(bundle.evidence_bundle.items)} items"
                )
                checks.append(CheckResult.passed(
                    "evidence_items_present",
                    f"Evidence bundle has {len(bundle.evidence_bundle.items)} items"
                ))
            else:
                report.add_check(
                    "completeness", "evidence_items_present", False,
                    "Evidence bundle is empty", severity="warn"
                )
                checks.append(CheckResult.warning(
                    "evidence_items_present",
                    "Evidence bundle is empty"
                ))
        else:
            report.add_check("completeness", "evidence_bundle_present", False, "EvidenceBundle is missing")
            checks.append(CheckResult.failed("evidence_bundle_present", "EvidenceBundle is missing"))
        
        # Check reasoning_trace
        if bundle.reasoning_trace:
            report.add_check("completeness", "reasoning_trace_present", True, "ReasoningTrace is present")
            checks.append(CheckResult.passed("reasoning_trace_present", "ReasoningTrace is present"))
        else:
            report.add_check("completeness", "reasoning_trace_present", False, "ReasoningTrace is missing")
            checks.append(CheckResult.failed("reasoning_trace_present", "ReasoningTrace is missing"))
        
        # Check verdict
        if bundle.verdict:
            report.add_check("completeness", "verdict_present", True, "Verdict is present")
            checks.append(CheckResult.passed("verdict_present", "Verdict is present"))
        else:
            report.add_check("completeness", "verdict_present", False, "Verdict is missing")
            checks.append(CheckResult.failed("verdict_present", "Verdict is missing"))
    
    def _check_hashes(
        self,
        bundle: ProofBundle,
        report: SentinelReport,
        checks: list[CheckResult],
    ) -> None:
        """Verify artifact hashes match stored values."""
        if not bundle.verdict:
            return
        
        verdict = bundle.verdict
        
        # Check prompt_spec_hash
        if bundle.prompt_spec and verdict.prompt_spec_hash:
            computed_hash = self._compute_prompt_spec_hash(bundle.prompt_spec)
            if computed_hash == verdict.prompt_spec_hash:
                report.add_check("hash", "prompt_spec_hash", True, "PromptSpec hash matches")
                checks.append(CheckResult.passed("prompt_spec_hash", "PromptSpec hash matches"))
            else:
                report.add_check(
                    "hash", "prompt_spec_hash", False,
                    f"PromptSpec hash mismatch: {computed_hash[:16]}... != {verdict.prompt_spec_hash[:16]}..."
                )
                checks.append(CheckResult.failed(
                    "prompt_spec_hash",
                    "PromptSpec hash does not match verdict"
                ))
        
        # Check evidence_root
        if bundle.evidence_bundle and verdict.evidence_root:
            computed_root = self._compute_evidence_root(bundle.evidence_bundle)
            if computed_root == verdict.evidence_root:
                report.add_check("hash", "evidence_root", True, "Evidence root matches")
                checks.append(CheckResult.passed("evidence_root", "Evidence root matches"))
            else:
                report.add_check(
                    "hash", "evidence_root", False,
                    f"Evidence root mismatch: {computed_root[:16]}... != {verdict.evidence_root[:16]}..."
                )
                checks.append(CheckResult.failed(
                    "evidence_root",
                    "Evidence root does not match verdict"
                ))
        
        # Check reasoning_root
        if bundle.reasoning_trace and verdict.reasoning_root:
            computed_root = self._compute_reasoning_root(bundle.reasoning_trace)
            if computed_root == verdict.reasoning_root:
                report.add_check("hash", "reasoning_root", True, "Reasoning root matches")
                checks.append(CheckResult.passed("reasoning_root", "Reasoning root matches"))
            else:
                report.add_check(
                    "hash", "reasoning_root", False,
                    f"Reasoning root mismatch: {computed_root[:16]}... != {verdict.reasoning_root[:16]}..."
                )
                checks.append(CheckResult.failed(
                    "reasoning_root",
                    "Reasoning root does not match verdict"
                ))
    
    def _check_consistency(
        self,
        bundle: ProofBundle,
        report: SentinelReport,
        checks: list[CheckResult],
    ) -> None:
        """Check that IDs and references are consistent across artifacts."""
        market_id = bundle.market_id
        
        # Check market_id consistency
        ids_match = True
        mismatches = []
        
        if bundle.prompt_spec and bundle.prompt_spec.market_id != market_id:
            ids_match = False
            mismatches.append(f"prompt_spec.market_id={bundle.prompt_spec.market_id}")
        
        if bundle.evidence_bundle and bundle.evidence_bundle.market_id != market_id:
            ids_match = False
            mismatches.append(f"evidence_bundle.market_id={bundle.evidence_bundle.market_id}")
        
        if bundle.reasoning_trace and bundle.reasoning_trace.market_id != market_id:
            ids_match = False
            mismatches.append(f"reasoning_trace.market_id={bundle.reasoning_trace.market_id}")
        
        if bundle.verdict and bundle.verdict.market_id != market_id:
            ids_match = False
            mismatches.append(f"verdict.market_id={bundle.verdict.market_id}")
        
        if ids_match:
            report.add_check("consistency", "market_id_consistent", True, f"All market IDs match: {market_id}")
            checks.append(CheckResult.passed("market_id_consistent", f"All market IDs match: {market_id}"))
        else:
            report.add_check(
                "consistency", "market_id_consistent", False,
                f"Market ID inconsistent: expected {market_id}, found {mismatches}"
            )
            checks.append(CheckResult.failed(
                "market_id_consistent",
                f"Market ID mismatch across artifacts: {mismatches}"
            ))
        
        # Check plan_id consistency
        if bundle.tool_plan and bundle.evidence_bundle:
            if bundle.evidence_bundle.plan_id == bundle.tool_plan.plan_id:
                report.add_check("consistency", "plan_id_consistent", True, "Plan IDs match")
                checks.append(CheckResult.passed("plan_id_consistent", "Plan IDs match"))
            else:
                report.add_check(
                    "consistency", "plan_id_consistent", False,
                    f"Plan ID mismatch: {bundle.tool_plan.plan_id} != {bundle.evidence_bundle.plan_id}"
                )
                checks.append(CheckResult.failed(
                    "plan_id_consistent",
                    "Plan ID mismatch between tool_plan and evidence_bundle"
                ))
        
        # Check bundle_id consistency
        if bundle.reasoning_trace and bundle.evidence_bundle:
            if bundle.reasoning_trace.bundle_id == bundle.evidence_bundle.bundle_id:
                report.add_check("consistency", "bundle_id_consistent", True, "Bundle IDs match")
                checks.append(CheckResult.passed("bundle_id_consistent", "Bundle IDs match"))
            else:
                report.add_check(
                    "consistency", "bundle_id_consistent", False,
                    f"Bundle ID mismatch: {bundle.evidence_bundle.bundle_id} != {bundle.reasoning_trace.bundle_id}"
                )
                checks.append(CheckResult.failed(
                    "bundle_id_consistent",
                    "Bundle ID mismatch between evidence_bundle and reasoning_trace"
                ))
        
        # Check evidence references in reasoning
        if bundle.reasoning_trace and bundle.evidence_bundle:
            evidence_ids = {item.evidence_id for item in bundle.evidence_bundle.items}
            referenced_ids = set(bundle.reasoning_trace.get_evidence_refs())
            
            invalid_refs = referenced_ids - evidence_ids
            if not invalid_refs:
                report.add_check(
                    "consistency", "evidence_refs_valid", True,
                    f"All {len(referenced_ids)} evidence references are valid"
                )
                checks.append(CheckResult.passed(
                    "evidence_refs_valid",
                    f"All evidence references in reasoning trace are valid"
                ))
            else:
                report.add_check(
                    "consistency", "evidence_refs_valid", False,
                    f"Invalid evidence references: {invalid_refs}"
                )
                checks.append(CheckResult.failed(
                    "evidence_refs_valid",
                    f"Reasoning trace references unknown evidence: {invalid_refs}"
                ))
    
    def _check_provenance(
        self,
        bundle: ProofBundle,
        report: SentinelReport,
        checks: list[CheckResult],
    ) -> None:
        """Check that evidence meets provenance requirements."""
        if not bundle.evidence_bundle or not bundle.tool_plan:
            return
        
        min_tier = bundle.tool_plan.min_provenance_tier
        valid_evidence = bundle.evidence_bundle.get_valid_evidence()
        
        if not valid_evidence:
            report.add_check(
                "provenance", "has_valid_evidence", False,
                "No valid evidence in bundle", severity="warn"
            )
            checks.append(CheckResult.warning("has_valid_evidence", "No valid evidence in bundle"))
            return
        
        report.add_check(
            "provenance", "has_valid_evidence", True,
            f"Bundle has {len(valid_evidence)} valid evidence items"
        )
        checks.append(CheckResult.passed(
            "has_valid_evidence",
            f"Bundle has {len(valid_evidence)} valid evidence items"
        ))
        
        # Check tier requirements
        tiers = [e.provenance.tier for e in valid_evidence]
        max_tier = max(tiers)
        min_actual_tier = min(tiers)
        
        if max_tier >= min_tier:
            report.add_check(
                "provenance", "tier_requirement_met", True,
                f"Highest tier {max_tier} meets requirement {min_tier}"
            )
            checks.append(CheckResult.passed(
                "tier_requirement_met",
                f"Evidence tier {max_tier} meets minimum requirement {min_tier}"
            ))
        else:
            report.add_check(
                "provenance", "tier_requirement_met", False,
                f"Highest tier {max_tier} below requirement {min_tier}", severity="warn"
            )
            checks.append(CheckResult.warning(
                "tier_requirement_met",
                f"Evidence tier {max_tier} below minimum requirement {min_tier}"
            ))
        
        # Check for authoritative sources
        authoritative_count = sum(1 for e in valid_evidence if e.provenance.tier >= 3)
        if authoritative_count > 0:
            report.add_check(
                "provenance", "has_authoritative_source", True,
                f"{authoritative_count} authoritative sources (tier 3+)"
            )
            checks.append(CheckResult.passed(
                "has_authoritative_source",
                f"Bundle has {authoritative_count} authoritative sources"
            ))
        else:
            report.add_check(
                "provenance", "has_authoritative_source", False,
                "No authoritative sources (tier 3+)", severity="warn"
            )
            checks.append(CheckResult.warning(
                "has_authoritative_source",
                "No authoritative sources in evidence bundle"
            ))
    
    def _check_reasoning(
        self,
        bundle: ProofBundle,
        report: SentinelReport,
        checks: list[CheckResult],
    ) -> None:
        """Check that reasoning trace is valid."""
        if not bundle.reasoning_trace:
            return
        
        trace = bundle.reasoning_trace
        
        # Check has steps
        if trace.steps:
            report.add_check(
                "reasoning", "has_steps", True,
                f"Reasoning trace has {trace.step_count} steps"
            )
            checks.append(CheckResult.passed(
                "has_steps",
                f"Reasoning trace has {trace.step_count} steps"
            ))
        else:
            report.add_check("reasoning", "has_steps", False, "Reasoning trace has no steps")
            checks.append(CheckResult.failed("has_steps", "Reasoning trace has no steps"))
        
        # Check has conclusion
        conclusion_steps = trace.get_steps_by_type("conclusion")
        if conclusion_steps:
            report.add_check("reasoning", "has_conclusion", True, "Reasoning has conclusion step")
            checks.append(CheckResult.passed("has_conclusion", "Reasoning has conclusion step"))
        else:
            report.add_check(
                "reasoning", "has_conclusion", False,
                "Reasoning missing conclusion step", severity="warn"
            )
            checks.append(CheckResult.warning("has_conclusion", "Reasoning missing conclusion step"))
        
        # Check preliminary outcome matches verdict
        if bundle.verdict:
            # Allow UNCERTAIN → INVALID conversion
            preliminary = trace.preliminary_outcome
            final = bundle.verdict.outcome
            
            outcome_valid = (
                preliminary == final or
                (preliminary == "UNCERTAIN" and final == "INVALID") or
                (preliminary in ("YES", "NO") and trace.preliminary_confidence and 
                 trace.preliminary_confidence < 0.55 and final == "INVALID")
            )
            
            if outcome_valid:
                report.add_check(
                    "reasoning", "outcome_consistent", True,
                    f"Outcome transition valid: {preliminary} → {final}"
                )
                checks.append(CheckResult.passed(
                    "outcome_consistent",
                    f"Outcome transition valid: {preliminary} → {final}"
                ))
            else:
                report.add_check(
                    "reasoning", "outcome_consistent", False,
                    f"Unexpected outcome change: {preliminary} → {final}"
                )
                checks.append(CheckResult.failed(
                    "outcome_consistent",
                    f"Unexpected outcome change: {preliminary} → {final}"
                ))
        
        # Check confidence bounds
        if trace.preliminary_confidence is not None:
            if 0.0 <= trace.preliminary_confidence <= 1.0:
                report.add_check(
                    "reasoning", "confidence_valid", True,
                    f"Confidence {trace.preliminary_confidence:.2f} within bounds"
                )
                checks.append(CheckResult.passed(
                    "confidence_valid",
                    f"Confidence {trace.preliminary_confidence:.2f} within bounds"
                ))
            else:
                report.add_check(
                    "reasoning", "confidence_valid", False,
                    f"Confidence {trace.preliminary_confidence} out of bounds [0,1]"
                )
                checks.append(CheckResult.failed(
                    "confidence_valid",
                    f"Confidence out of bounds: {trace.preliminary_confidence}"
                ))
    
    def _compute_prompt_spec_hash(self, spec: PromptSpec) -> str:
        """Compute hash of prompt spec."""
        spec_dict = spec.model_dump(exclude={"created_at"})
        canonical = dumps_canonical(spec_dict)
        return f"0x{hashlib.sha256(canonical.encode()).hexdigest()}"
    
    def _compute_evidence_root(self, bundle: EvidenceBundle) -> str:
        """Compute evidence root hash."""
        evidence_ids = sorted([item.evidence_id for item in bundle.items])
        combined = "|".join(evidence_ids)
        return f"0x{hashlib.sha256(combined.encode()).hexdigest()}"
    
    def _compute_reasoning_root(self, trace: ReasoningTrace) -> str:
        """Compute reasoning root hash."""
        step_data = []
        for step in trace.steps:
            step_data.append({
                "step_id": step.step_id,
                "step_type": step.step_type,
                "conclusion": step.conclusion,
            })
        canonical = dumps_canonical(step_data)
        return f"0x{hashlib.sha256(canonical.encode()).hexdigest()}"
    
    def _generate_report_id(self, bundle_id: str) -> str:
        """Generate report ID from bundle ID."""
        hash_bytes = hashlib.sha256(f"report:{bundle_id}".encode()).digest()
        return f"report_{hash_bytes[:8].hex()}"
