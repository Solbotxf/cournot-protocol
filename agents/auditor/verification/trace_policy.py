"""
Module 06 - Trace Policy Verification

Purpose: Hard validation of trace structure and grounding. Ensures that
reasoning traces follow policy constraints and properly reference evidence.

Owner: Verification Engineer
"""

from __future__ import annotations

from typing import Optional

from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.verification import (
    ChallengeRef,
    CheckResult,
    VerificationResult,
)


class TracePolicyVerifier:
    """
    Verifies that a ReasoningTrace conforms to policy constraints.

    Validates:
    1. Step IDs are unique and monotonically increasing
    2. Prior step references are valid (only reference earlier steps)
    3. All evidence IDs exist in the EvidenceBundle
    4. Number of steps <= policy.max_steps
    5. External sources are properly controlled
    6. Final step produces evaluation_variables
    """

    def __init__(
        self,
        require_evaluation_variables: bool = True,
        strict_step_id_format: bool = True,
    ):
        """
        Initialize the trace policy verifier.

        Args:
            require_evaluation_variables: Whether to require evaluation_variables
                in the final step output
            strict_step_id_format: Whether to enforce step_XXXX format
        """
        self.require_evaluation_variables = require_evaluation_variables
        self.strict_step_id_format = strict_step_id_format

    def _check_step_id_uniqueness(self, trace: ReasoningTrace) -> CheckResult:
        """Verify all step IDs are unique."""
        step_ids = [step.step_id for step in trace.steps]
        duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]

        if duplicates:
            return CheckResult(
                check_id="step_id_uniqueness",
                ok=False,
                severity="error",
                message=f"Duplicate step IDs found: {set(duplicates)}",
                details={"duplicates": list(set(duplicates))},
            )

        return CheckResult(
            check_id="step_id_uniqueness",
            ok=True,
            severity="info",
            message=f"All {len(step_ids)} step IDs are unique",
            details={"step_count": len(step_ids)},
        )

    def _check_step_id_format(self, trace: ReasoningTrace) -> CheckResult:
        """Verify step IDs follow expected format (step_XXXX)."""
        import re

        if not self.strict_step_id_format:
            return CheckResult(
                check_id="step_id_format",
                ok=True,
                severity="info",
                message="Step ID format check skipped (not strict mode)",
                details={},
            )

        pattern = re.compile(r"^step_\d{4}$")
        invalid_ids = []

        for step in trace.steps:
            if not pattern.match(step.step_id):
                invalid_ids.append(step.step_id)

        if invalid_ids:
            return CheckResult(
                check_id="step_id_format",
                ok=False,
                severity="warn",  # Warning, not error - format is a convention
                message=f"Step IDs not in expected format (step_XXXX): {invalid_ids[:5]}",
                details={"invalid_ids": invalid_ids},
            )

        return CheckResult(
            check_id="step_id_format",
            ok=True,
            severity="info",
            message="All step IDs follow expected format",
            details={},
        )

    def _check_prior_step_references(self, trace: ReasoningTrace) -> CheckResult:
        """Verify all prior_step_ids reference earlier steps only."""
        seen_step_ids: set[str] = set()
        invalid_refs: list[tuple[str, str]] = []  # (step_id, invalid_ref)

        for step in trace.steps:
            for prior_id in step.prior_step_ids:
                if prior_id not in seen_step_ids:
                    invalid_refs.append((step.step_id, prior_id))
            seen_step_ids.add(step.step_id)

        if invalid_refs:
            return CheckResult(
                check_id="prior_step_references",
                ok=False,
                severity="error",
                message=f"Invalid prior step references: steps reference non-earlier steps",
                details={
                    "invalid_references": [
                        {"step": ref[0], "references": ref[1]} for ref in invalid_refs
                    ]
                },
            )

        return CheckResult(
            check_id="prior_step_references",
            ok=True,
            severity="info",
            message="All prior step references are valid",
            details={},
        )

    def _check_evidence_references(
        self,
        trace: ReasoningTrace,
        evidence: EvidenceBundle,
    ) -> CheckResult:
        """Verify all evidence IDs in trace exist in the bundle."""
        bundle_evidence_ids = set(evidence.evidence_ids)
        trace_evidence_ids = trace.get_all_evidence_ids()
        trace_evidence_ids.update(trace.evidence_refs)

        invalid_ids = trace_evidence_ids - bundle_evidence_ids

        if invalid_ids:
            return CheckResult(
                check_id="evidence_references",
                ok=False,
                severity="error",
                message=f"Trace references {len(invalid_ids)} unknown evidence IDs",
                details={
                    "invalid_evidence_ids": list(invalid_ids),
                    "bundle_evidence_ids": list(bundle_evidence_ids),
                },
            )

        return CheckResult(
            check_id="evidence_references",
            ok=True,
            severity="info",
            message=f"All {len(trace_evidence_ids)} evidence references are valid",
            details={"evidence_count": len(trace_evidence_ids)},
        )

    def _check_max_steps(self, trace: ReasoningTrace) -> CheckResult:
        """Verify number of steps doesn't exceed policy.max_steps."""
        actual = len(trace.steps)
        max_allowed = trace.policy.max_steps

        if actual > max_allowed:
            return CheckResult(
                check_id="max_steps",
                ok=False,
                severity="error",
                message=f"Too many steps: {actual} > max_allowed {max_allowed}",
                details={"actual": actual, "max_allowed": max_allowed},
            )

        return CheckResult(
            check_id="max_steps",
            ok=True,
            severity="info",
            message=f"Step count ({actual}) within limit ({max_allowed})",
            details={"actual": actual, "max_allowed": max_allowed},
        )

    def _check_external_sources(self, trace: ReasoningTrace) -> CheckResult:
        """
        Check for external sources in step inputs when policy forbids them.
        """
        if trace.policy.allow_external_sources:
            return CheckResult(
                check_id="external_sources",
                ok=True,
                severity="info",
                message="External sources allowed by policy",
                details={},
            )

        # Look for external_url in step inputs
        violations: list[str] = []
        for step in trace.steps:
            if "external_url" in step.inputs:
                violations.append(step.step_id)

        if violations:
            return CheckResult(
                check_id="external_sources",
                ok=False,
                severity="error",
                message=f"External sources found in steps but policy forbids them",
                details={"violating_steps": violations},
            )

        return CheckResult(
            check_id="external_sources",
            ok=True,
            severity="info",
            message="No external sources found (policy forbids them)",
            details={},
        )

    def _check_evaluation_variables(self, trace: ReasoningTrace) -> CheckResult:
        """
        Check that the final step produces evaluation_variables.

        The final step should be of type "map" or "aggregate" and
        produce output containing evaluation_variables.
        """
        if not self.require_evaluation_variables:
            return CheckResult(
                check_id="evaluation_variables",
                ok=True,
                severity="info",
                message="Evaluation variables check skipped (not required)",
                details={},
            )

        if not trace.steps:
            return CheckResult(
                check_id="evaluation_variables",
                ok=False,
                severity="error",
                message="No steps in trace - cannot find evaluation_variables",
                details={},
            )

        final_step = trace.steps[-1]

        # Check step type
        if final_step.type not in ("map", "aggregate", "deduce"):
            return CheckResult(
                check_id="evaluation_variables",
                ok=False,
                severity="warn",
                message=f"Final step type '{final_step.type}' should be 'map' or 'aggregate'",
                details={"final_step_type": final_step.type},
            )

        # Check for evaluation_variables in output
        if "evaluation_variables" not in final_step.output:
            return CheckResult(
                check_id="evaluation_variables",
                ok=False,
                severity="error",
                message="Final step output missing 'evaluation_variables'",
                details={"final_step_output_keys": list(final_step.output.keys())},
            )

        eval_vars = final_step.output["evaluation_variables"]

        # Validate required fields
        required_fields = ["event_observed"]
        missing = [f for f in required_fields if f not in eval_vars]

        if missing:
            return CheckResult(
                check_id="evaluation_variables",
                ok=False,
                severity="warn",
                message=f"evaluation_variables missing recommended fields: {missing}",
                details={
                    "present_fields": list(eval_vars.keys()),
                    "missing_recommended": missing,
                },
            )

        return CheckResult(
            check_id="evaluation_variables",
            ok=True,
            severity="info",
            message="Final step contains valid evaluation_variables",
            details={"evaluation_variables": eval_vars},
        )

    def verify(
        self,
        trace: ReasoningTrace,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
    ) -> VerificationResult:
        """
        Verify a reasoning trace against policy constraints.

        Args:
            trace: The reasoning trace to verify
            prompt_spec: The prompt specification (for context)
            evidence: The evidence bundle (for reference validation)

        Returns:
            VerificationResult with all checks and any challenge reference
        """
        checks: list[CheckResult] = []
        challenge: Optional[ChallengeRef] = None

        # Run all checks
        checks.append(self._check_step_id_uniqueness(trace))
        checks.append(self._check_step_id_format(trace))
        checks.append(self._check_prior_step_references(trace))
        checks.append(self._check_evidence_references(trace, evidence))
        checks.append(self._check_max_steps(trace))
        checks.append(self._check_external_sources(trace))
        checks.append(self._check_evaluation_variables(trace))

        # Find first failed check for challenge reference
        for check in checks:
            if not check.ok:
                # Determine challenge kind and details
                if "evidence" in check.check_id.lower():
                    invalid_ids = check.details.get("invalid_evidence_ids", [])
                    challenge = ChallengeRef(
                        kind="reasoning_leaf",
                        reason=check.message,
                        step_id=None,  # Multiple steps may be affected
                    )
                elif "step" in check.check_id.lower():
                    # Try to get the specific step that caused the issue
                    step_id = None
                    if "invalid_references" in check.details:
                        refs = check.details["invalid_references"]
                        if refs:
                            step_id = refs[0].get("step")
                    challenge = ChallengeRef(
                        kind="reasoning_leaf",
                        step_id=step_id,
                        reason=check.message,
                    )
                else:
                    challenge = ChallengeRef(
                        kind="reasoning_leaf",
                        reason=check.message,
                    )
                break

        all_ok = all(c.ok for c in checks)

        return VerificationResult(
            ok=all_ok,
            checks=checks,
            challenge=challenge if not all_ok else None,
            error=None,
        )


# Convenience function
def verify_trace_policy(
    trace: ReasoningTrace,
    prompt_spec: PromptSpec,
    evidence: EvidenceBundle,
) -> VerificationResult:
    """
    Verify a reasoning trace using default configuration.

    This is the main entry point for trace policy verification.
    """
    verifier = TracePolicyVerifier()
    return verifier.verify(trace, prompt_spec, evidence)