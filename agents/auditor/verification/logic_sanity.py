"""
Module 06 - Logic Sanity Verification

Purpose: Validate that reasoning aligns with event_definition and time
window constraints. Performs sanity checks on the extracted claims
and their relationship to the market question.

Owner: Verification Engineer
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.verification import CheckResult

from ..reasoning.claim_extraction import ClaimSet


class LogicSanityChecker:
    """
    Performs sanity checks on reasoning logic and constraints.

    Validates:
    1. Time window sanity - evidence timestamps fall within resolution window
    2. Requirement coverage - each DataRequirement has evidence
    3. Event definition variables - required variables can be computed
    4. Confidence sanity - all confidence values in [0, 1]
    """

    def __init__(
        self,
        time_slack_seconds: int = 3600,  # 1 hour slack for timestamps
        require_all_requirements: bool = False,  # Strict requirement coverage
    ):
        """
        Initialize the logic sanity checker.

        Args:
            time_slack_seconds: Acceptable slack beyond resolution window
            require_all_requirements: Whether all requirements must have evidence
        """
        self.time_slack_seconds = time_slack_seconds
        self.require_all_requirements = require_all_requirements

    def _check_time_window_sanity(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        claims: ClaimSet,
    ) -> CheckResult:
        """
        Verify evidence timestamps fall within resolution window.

        Checks both retrieval timestamps and any timestamp claims.
        """
        window = prompt_spec.market.resolution_window
        window_start = window.start
        window_end = window.end

        # Add slack to the window
        from datetime import timedelta
        slack = timedelta(seconds=self.time_slack_seconds)
        effective_start = window_start - slack
        effective_end = window_end + slack

        violations: list[dict] = []

        # Check retrieval timestamps
        for item in evidence.items:
            if item.retrieval.retrieved_at:
                ts = item.retrieval.retrieved_at
                # Ensure timezone-aware comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                if ts < effective_start or ts > effective_end:
                    violations.append({
                        "type": "retrieval",
                        "evidence_id": item.evidence_id,
                        "timestamp": ts.isoformat(),
                        "issue": "outside_resolution_window",
                    })

        # Check timestamp claims
        for claim in claims.by_kind("timestamp"):
            try:
                # Parse the timestamp claim
                ts_str = str(claim.value)
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

                if ts < effective_start or ts > effective_end:
                    violations.append({
                        "type": "claim",
                        "claim_id": claim.claim_id,
                        "evidence_id": claim.evidence_id,
                        "timestamp": ts_str,
                        "issue": "outside_resolution_window",
                    })
            except (ValueError, TypeError):
                # Can't parse timestamp - not a violation, just skip
                pass

        if violations:
            return CheckResult(
                check_id="time_window_sanity",
                ok=False,
                severity="warn",
                message=f"Found {len(violations)} timestamp(s) outside resolution window",
                details={
                    "violations": violations,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "slack_seconds": self.time_slack_seconds,
                },
            )

        return CheckResult(
            check_id="time_window_sanity",
            ok=True,
            severity="info",
            message="All timestamps within resolution window",
            details={
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
            },
        )

    def _check_requirement_coverage(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
    ) -> CheckResult:
        """
        Verify each DataRequirement has at least one evidence item.
        """
        requirement_ids = {req.requirement_id for req in prompt_spec.data_requirements}

        # Track which requirements have evidence
        covered: set[str] = set()
        for item in evidence.items:
            if item.requirement_id and item.requirement_id in requirement_ids:
                covered.add(item.requirement_id)

        missing = requirement_ids - covered

        if missing:
            severity = "error" if self.require_all_requirements else "warn"
            return CheckResult(
                check_id="requirement_coverage",
                ok=not self.require_all_requirements,
                severity=severity,
                message=f"Missing evidence for {len(missing)} requirement(s): {missing}",
                details={
                    "missing_requirements": list(missing),
                    "covered_requirements": list(covered),
                    "total_requirements": len(requirement_ids),
                },
            )

        return CheckResult(
            check_id="requirement_coverage",
            ok=True,
            severity="info",
            message=f"All {len(requirement_ids)} requirements have evidence",
            details={"requirement_count": len(requirement_ids)},
        )

    def _check_event_definition_variables(
        self,
        prompt_spec: PromptSpec,
        claims: ClaimSet,
    ) -> CheckResult:
        """
        Verify that variables needed for event_definition can be computed.

        This is a heuristic check - looks for numeric claims if the
        event definition appears to involve numeric comparison.
        """
        event_def = prompt_spec.market.event_definition.lower()

        # Check if event definition involves numeric comparison
        numeric_indicators = [">", "<", "=", "above", "below", "greater", "less", "exceed"]
        needs_numeric = any(ind in event_def for ind in numeric_indicators)

        if needs_numeric:
            numeric_claims = claims.by_kind("numeric")
            if not numeric_claims:
                return CheckResult(
                    check_id="event_definition_variables",
                    ok=False,
                    severity="warn",
                    message="Event definition involves comparison but no numeric claims found",
                    details={
                        "event_definition": prompt_spec.market.event_definition,
                        "numeric_claim_count": 0,
                    },
                )

        # Check if event definition involves boolean/resolution check
        boolean_indicators = ["resolved", "confirmed", "announced", "declared"]
        needs_boolean = any(ind in event_def for ind in boolean_indicators)

        if needs_boolean:
            boolean_claims = claims.by_kind("boolean")
            if not boolean_claims:
                return CheckResult(
                    check_id="event_definition_variables",
                    ok=False,
                    severity="warn",
                    message="Event definition involves status check but no boolean claims found",
                    details={
                        "event_definition": prompt_spec.market.event_definition,
                        "boolean_claim_count": 0,
                    },
                )

        return CheckResult(
            check_id="event_definition_variables",
            ok=True,
            severity="info",
            message="Required variable types appear to be available",
            details={
                "numeric_claims": len(claims.by_kind("numeric")),
                "boolean_claims": len(claims.by_kind("boolean")),
                "total_claims": len(claims),
            },
        )

    def _check_confidence_sanity(
        self,
        evidence: EvidenceBundle,
        claims: ClaimSet,
    ) -> CheckResult:
        """
        Verify all confidence values are in [0, 1].
        """
        violations: list[dict] = []

        # Check evidence item confidence
        for item in evidence.items:
            if not 0.0 <= item.confidence <= 1.0:
                violations.append({
                    "type": "evidence",
                    "id": item.evidence_id,
                    "confidence": item.confidence,
                })

        # Check claim confidence
        for claim in claims:
            if not 0.0 <= claim.confidence <= 1.0:
                violations.append({
                    "type": "claim",
                    "id": claim.claim_id,
                    "confidence": claim.confidence,
                })

        if violations:
            return CheckResult(
                check_id="confidence_sanity",
                ok=False,
                severity="error",
                message=f"Found {len(violations)} invalid confidence value(s)",
                details={"violations": violations},
            )

        return CheckResult(
            check_id="confidence_sanity",
            ok=True,
            severity="info",
            message="All confidence values are valid",
            details={
                "evidence_count": len(evidence.items),
                "claim_count": len(claims),
            },
        )

    def _check_evidence_not_empty(
        self,
        evidence: EvidenceBundle,
    ) -> CheckResult:
        """Verify the evidence bundle is not empty."""
        if not evidence.items:
            return CheckResult(
                check_id="evidence_not_empty",
                ok=False,
                severity="error",
                message="Evidence bundle is empty - cannot perform reasoning",
                details={"item_count": 0},
            )

        return CheckResult(
            check_id="evidence_not_empty",
            ok=True,
            severity="info",
            message=f"Evidence bundle contains {len(evidence.items)} item(s)",
            details={"item_count": len(evidence.items)},
        )

    def check(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        claims: ClaimSet,
    ) -> list[CheckResult]:
        """
        Perform all logic sanity checks.

        Args:
            prompt_spec: The prompt specification
            evidence: The evidence bundle
            claims: The extracted claims

        Returns:
            List of CheckResult objects for all checks performed
        """
        results: list[CheckResult] = []

        # Run all checks
        results.append(self._check_evidence_not_empty(evidence))
        results.append(self._check_time_window_sanity(prompt_spec, evidence, claims))
        results.append(self._check_requirement_coverage(prompt_spec, evidence))
        results.append(self._check_event_definition_variables(prompt_spec, claims))
        results.append(self._check_confidence_sanity(evidence, claims))

        return results


# Convenience function
def check_logic_sanity(
    prompt_spec: PromptSpec,
    evidence: EvidenceBundle,
    claims: ClaimSet,
) -> list[CheckResult]:
    """
    Perform logic sanity checks using default configuration.

    This is the main entry point for logic sanity checking.
    """
    checker = LogicSanityChecker()
    return checker.check(prompt_spec, evidence, claims)