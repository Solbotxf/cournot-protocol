"""
Module 06 - Auditor Agent

Purpose: Given PromptSpec and EvidenceBundle, produce a deterministic,
checkable ReasoningTrace that extracts claims, detects contradictions,
performs sanity checks, and produces evaluation variables for the Judge.

Owner: AI/Reasoning Engineer + Verification Engineer
Module ID: M06

The Auditor does not finalize the verdict; it produces the "audit trail"
for Judge + Sentinel verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from core.llm.determinism import DecodingPolicy
from core.llm.llm import LLMClient
from core.por.reasoning_trace import ReasoningStep, ReasoningTrace, TracePolicy
from core.schemas.evidence import EvidenceBundle, EvidenceItem
from core.schemas.prompts import PromptSpec
from core.schemas.verification import CheckResult, VerificationResult

from agents.base_agent import AgentContext, BaseAgent

from .reasoning.claim_extraction import ClaimExtractor, ClaimSet, extract_claims
from .reasoning.contradiction_checks import ContradictionChecker, check_contradictions
from .verification.logic_sanity import LogicSanityChecker, check_logic_sanity
from .verification.trace_policy import TracePolicyVerifier, verify_trace_policy

class AuditorAgent(BaseAgent):
    """
    Auditor Agent - produces deterministic reasoning traces.

    Given a PromptSpec and EvidenceBundle, the Auditor:
    1. Extracts structured claims from evidence
    2. Checks for contradictions across claims
    3. Performs logic sanity checks
    4. Builds a deterministic ReasoningTrace
    5. Verifies trace policy compliance
    6. Returns (ReasoningTrace, VerificationResult)
    """

    role = "auditor"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        name: Optional[str] = None,
        decoding_policy: Optional[DecodingPolicy] = None,
        strict_step_ids: bool = True,
        require_evaluation_variables: bool = True,
    ):
        """
        Initialize the Auditor agent.

        Args:
            llm: Optional LLM client for advanced reasoning (not required)
            name: Agent name for identification
            decoding_policy: Policy for LLM decoding (if used)
            strict_step_ids: Whether to enforce step_XXXX format
            require_evaluation_variables: Whether final step must have eval vars
        """
        super().__init__(name=name)
        self.llm = llm
        self.decoding_policy = decoding_policy or DecodingPolicy(
            temperature=0.0,
            seed=42,
            schema_lock=True,
        )
        self.strict_step_ids = strict_step_ids
        self.require_evaluation_variables = require_evaluation_variables

        # Initialize sub-components
        self.claim_extractor = ClaimExtractor()
        self.contradiction_checker = ContradictionChecker()
        self.logic_sanity_checker = LogicSanityChecker()
        self.trace_policy_verifier = TracePolicyVerifier(
            require_evaluation_variables=require_evaluation_variables,
            strict_step_id_format=strict_step_ids,
        )

    def _generate_trace_id(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
    ) -> str:
        """Generate a deterministic trace ID."""
        content = f"{prompt_spec.market.market_id}|{evidence.bundle_id}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        return "trace_" + hash_bytes.hex()[:12]

    def _generate_step_id(self, index: int) -> str:
        """Generate a deterministic step ID."""
        return f"step_{index:04d}"

    def _build_extraction_steps(
        self,
        evidence: EvidenceBundle,
        claims: ClaimSet,
        step_counter: int,
    ) -> tuple[list[ReasoningStep], int]:
        """
        Build extraction steps for each evidence item.

        Returns (steps, next_step_counter)
        """
        steps: list[ReasoningStep] = []

        for item in evidence.items:
            item_claims = claims.by_evidence_id(item.evidence_id)

            step = ReasoningStep(
                step_id=self._generate_step_id(step_counter),
                type="extract",
                inputs={
                    "evidence_id": item.evidence_id,
                    "content_type": item.content_type,
                    "source_id": item.source.source_id,
                },
                action=f"Extract claims from evidence item {item.evidence_id}",
                output={
                    "claim_count": len(item_claims),
                    "claims": [
                        {
                            "claim_id": c.claim_id,
                            "kind": c.kind,
                            "path": c.path,
                            "value": c.value,
                            "confidence": c.confidence,
                        }
                        for c in item_claims
                    ],
                },
                evidence_ids=[item.evidence_id],
                prior_step_ids=[],
            )
            steps.append(step)
            step_counter += 1

        return steps, step_counter

    def _build_check_steps(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        claims: ClaimSet,
        prior_step_ids: list[str],
        step_counter: int,
    ) -> tuple[list[ReasoningStep], int, list[CheckResult]]:
        """
        Build check steps for contradictions and logic sanity.

        Returns (steps, next_step_counter, all_checks)
        """
        steps: list[ReasoningStep] = []
        all_checks: list[CheckResult] = []

        # Contradiction check step
        contradiction_checks = self.contradiction_checker.check(claims)
        all_checks.extend(contradiction_checks)

        has_contradictions = any(not c.ok for c in contradiction_checks)
        contradiction_step = ReasoningStep(
            step_id=self._generate_step_id(step_counter),
            type="check",
            inputs={
                "check_type": "contradiction",
                "claim_count": len(claims),
            },
            action="Check for contradictions across claims from different sources",
            output={
                "has_contradictions": has_contradictions,
                "check_results": [
                    {
                        "check_id": c.check_id,
                        "ok": c.ok,
                        "severity": c.severity,
                        "message": c.message,
                    }
                    for c in contradiction_checks
                ],
            },
            evidence_ids=list(claims.get_unique_evidence_ids()),
            prior_step_ids=prior_step_ids,
        )
        steps.append(contradiction_step)
        step_counter += 1

        # Logic sanity check step
        sanity_checks = self.logic_sanity_checker.check(prompt_spec, evidence, claims)
        all_checks.extend(sanity_checks)

        has_sanity_issues = any(not c.ok for c in sanity_checks)
        sanity_step = ReasoningStep(
            step_id=self._generate_step_id(step_counter),
            type="check",
            inputs={
                "check_type": "logic_sanity",
                "event_definition": prompt_spec.market.event_definition,
            },
            action="Perform logic sanity checks against event definition and constraints",
            output={
                "has_issues": has_sanity_issues,
                "check_results": [
                    {
                        "check_id": c.check_id,
                        "ok": c.ok,
                        "severity": c.severity,
                        "message": c.message,
                    }
                    for c in sanity_checks
                ],
            },
            evidence_ids=list(evidence.evidence_ids),
            prior_step_ids=[contradiction_step.step_id],
        )
        steps.append(sanity_step)
        step_counter += 1

        return steps, step_counter, all_checks

    def _build_aggregate_step(
        self,
        prompt_spec: PromptSpec,
        claims: ClaimSet,
        prior_step_ids: list[str],
        step_counter: int,
    ) -> tuple[ReasoningStep, int]:
        """
        Build aggregation step that reconciles multiple sources.

        Returns (step, next_step_counter)
        """
        # Group claims by type
        numeric_claims = claims.by_kind("numeric")
        boolean_claims = claims.by_kind("boolean")

        # Calculate aggregated values
        numeric_values = {}
        for claim in numeric_claims:
            path = claim.path
            if path not in numeric_values:
                numeric_values[path] = []
            try:
                numeric_values[path].append(float(claim.value))
            except (ValueError, TypeError):
                pass

        # Average numeric values (simple aggregation)
        aggregated_numeric = {}
        for path, values in numeric_values.items():
            if values:
                aggregated_numeric[path] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # Aggregate boolean values (majority vote)
        boolean_values = {}
        for claim in boolean_claims:
            path = claim.path
            if path not in boolean_values:
                boolean_values[path] = {"true": 0, "false": 0}
            if claim.value is True:
                boolean_values[path]["true"] += 1
            elif claim.value is False:
                boolean_values[path]["false"] += 1

        aggregated_boolean = {}
        for path, counts in boolean_values.items():
            aggregated_boolean[path] = {
                "majority": counts["true"] >= counts["false"],
                "true_count": counts["true"],
                "false_count": counts["false"],
            }

        step = ReasoningStep(
            step_id=self._generate_step_id(step_counter),
            type="aggregate",
            inputs={
                "numeric_claim_count": len(numeric_claims),
                "boolean_claim_count": len(boolean_claims),
            },
            action="Aggregate claims from multiple sources using selection policy",
            output={
                "aggregated_numeric": aggregated_numeric,
                "aggregated_boolean": aggregated_boolean,
                "source_count": len(claims.get_unique_evidence_ids()),
            },
            evidence_ids=list(claims.get_unique_evidence_ids()),
            prior_step_ids=prior_step_ids,
        )

        return step, step_counter + 1

    def _build_evaluation_step(
        self,
        prompt_spec: PromptSpec,
        claims: ClaimSet,
        all_checks: list[CheckResult],
        prior_step_ids: list[str],
        step_counter: int,
    ) -> ReasoningStep:
        """
        Build the final evaluation step that produces evaluation_variables.

        This step outputs the derived state for Judge to use.
        """
        # Determine if we have insufficient evidence
        insufficient_evidence = (
            len(claims) == 0
            or any(c.check_id == "evidence_not_empty" and not c.ok for c in all_checks)
        )

        # Detect conflicts
        conflict_detected = any(
            "contradiction" in c.check_id and not c.ok
            for c in all_checks
        )

        # Try to determine event_observed based on claims
        event_observed: Optional[bool] = None
        numeric_value: Optional[float] = None

        # Simple heuristic: check if we have relevant numeric or boolean claims
        numeric_claims = claims.by_kind("numeric")
        boolean_claims = claims.by_kind("boolean")

        if numeric_claims and not conflict_detected:
            # Use first numeric claim as representative value
            try:
                numeric_value = float(numeric_claims[0].value)
            except (ValueError, TypeError):
                pass

        if boolean_claims and not conflict_detected:
            # Use majority of boolean claims
            true_count = sum(1 for c in boolean_claims if c.value is True)
            false_count = sum(1 for c in boolean_claims if c.value is False)
            if true_count > false_count:
                event_observed = True
            elif false_count > true_count:
                event_observed = False

        # Build evaluation variables
        evaluation_variables = {
            "event_observed": event_observed,
            "numeric_value": numeric_value,
            "source_summary": list(claims.get_unique_evidence_ids()),
            "conflict_detected": conflict_detected,
            "insufficient_evidence": insufficient_evidence,
            "claim_count": len(claims),
            "numeric_claim_count": len(numeric_claims),
            "boolean_claim_count": len(boolean_claims),
        }

        step = ReasoningStep(
            step_id=self._generate_step_id(step_counter),
            type="map",
            inputs={
                "event_definition": prompt_spec.market.event_definition,
                "check_summary": {
                    "total": len(all_checks),
                    "passed": sum(1 for c in all_checks if c.ok),
                    "failed": sum(1 for c in all_checks if not c.ok),
                },
            },
            action="Map aggregated results to evaluation variables for Judge",
            output={
                "evaluation_variables": evaluation_variables,
            },
            evidence_ids=list(claims.get_unique_evidence_ids()),
            prior_step_ids=prior_step_ids,
        )

        return step

    def _build_reasoning_trace(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        claims: ClaimSet,
        all_checks: list[CheckResult],
    ) -> ReasoningTrace:
        """
        Build the complete reasoning trace.
        """
        trace_id = self._generate_trace_id(prompt_spec, evidence)

        # Create trace policy
        policy = TracePolicy(
            decoding_policy="strict",
            allow_external_sources=False,
            max_steps=200,
            extra={
                "auditor_version": "1.0",
                "deterministic": True,
            },
        )

        steps: list[ReasoningStep] = []
        step_counter = 0

        # Build extraction steps
        extraction_steps, step_counter = self._build_extraction_steps(
            evidence, claims, step_counter
        )
        steps.extend(extraction_steps)

        # Build check steps
        check_steps, step_counter, _ = self._build_check_steps(
            prompt_spec,
            evidence,
            claims,
            [s.step_id for s in extraction_steps],
            step_counter,
        )
        steps.extend(check_steps)

        # Build aggregate step
        aggregate_step, step_counter = self._build_aggregate_step(
            prompt_spec,
            claims,
            [s.step_id for s in check_steps],
            step_counter,
        )
        steps.append(aggregate_step)

        # Build final evaluation step
        eval_step = self._build_evaluation_step(
            prompt_spec,
            claims,
            all_checks,
            [aggregate_step.step_id],
            step_counter,
        )
        steps.append(eval_step)

        # Build trace
        trace = ReasoningTrace(
            trace_id=trace_id,
            policy=policy,
            steps=steps,
            evidence_refs=list(evidence.evidence_ids),
        )

        return trace

    def audit(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        *,
        ctx: Optional[AgentContext] = None,
    ) -> tuple[ReasoningTrace, VerificationResult]:
        """
        Produce a deterministic reasoning trace from prompt spec and evidence.

        This is the main entry point for the Auditor agent.

        Args:
            prompt_spec: The prompt specification defining the market
            evidence: The collected evidence bundle
            ctx: Optional execution context

        Returns:
            Tuple of (ReasoningTrace, VerificationResult)

        The ReasoningTrace contains:
            - Extraction steps for each evidence item
            - Check steps for contradictions and sanity
            - Aggregate step for multi-source reconciliation
            - Final evaluation step with evaluation_variables

        The VerificationResult indicates whether the trace passes policy.
        """
        all_checks: list[CheckResult] = []

        # Step 1: Extract claims from evidence
        claims = self.claim_extractor.extract(prompt_spec, evidence)

        # Step 2: Check for contradictions
        contradiction_checks = self.contradiction_checker.check(claims)
        all_checks.extend(contradiction_checks)

        # Step 3: Perform logic sanity checks
        sanity_checks = self.logic_sanity_checker.check(prompt_spec, evidence, claims)
        all_checks.extend(sanity_checks)

        # Step 4: Build the reasoning trace
        trace = self._build_reasoning_trace(
            prompt_spec, evidence, claims, all_checks
        )

        # Step 5: Verify trace policy compliance
        verification_result = self.trace_policy_verifier.verify(
            trace, prompt_spec, evidence
        )

        # Merge all checks into the verification result
        for check in all_checks:
            verification_result.checks.append(check)

        # Update overall ok status
        verification_result.ok = verification_result.ok and all(
            c.ok or c.severity != "error" for c in all_checks
        )

        return trace, verification_result


# Convenience function for simple auditing
def audit(
    prompt_spec: PromptSpec,
    evidence: EvidenceBundle,
) -> tuple[ReasoningTrace, VerificationResult]:
    """
    Audit evidence against a prompt spec using default configuration.

    This is the main entry point for the Auditor module.
    """
    agent = AuditorAgent()
    return agent.audit(prompt_spec, evidence)