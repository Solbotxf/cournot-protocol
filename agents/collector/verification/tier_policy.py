"""
Module 05 - Tier Policy

Provenance tier classification and policy enforcement.
"""

from typing import Any, Optional

from core.schemas.errors import CournotError, ErrorCodes
from core.schemas.evidence import EvidenceBundle, EvidenceItem, ProvenanceProof
from core.schemas.prompts import DataRequirement, PromptSpec, SourceTarget
from core.schemas.verification import (
    ChallengeRef,
    CheckResult,
    VerificationResult,
)

from agents.collector.data_sources.base_source import FetchedArtifact


# Default tier mapping
DEFAULT_TIER_MAPPING = {
    "none": 0,
    "hashlog": 1,
    "signature": 2,
    "notary": 2,
    "zktls": 3,
}


class TierPolicy:
    """
    Provenance tier classification and enforcement.
    
    Determines required provenance tier for evidence and
    enforces that collected evidence meets requirements.
    """
    
    def __init__(
        self,
        tier_mapping: Optional[dict[str, int]] = None,
        strict_mode: bool = True
    ):
        """
        Initialize tier policy.
        
        Args:
            tier_mapping: Mapping from proof kind to tier level
            strict_mode: If True, fail on tier violations; else warn
        """
        self.tier_mapping = tier_mapping or DEFAULT_TIER_MAPPING
        self.strict_mode = strict_mode
    
    def required_tier(
        self,
        prompt_spec: PromptSpec,
        requirement: DataRequirement
    ) -> int:
        """
        Determine required provenance tier for a requirement.
        
        Takes the maximum of:
        - Market-level min_provenance_tier
        - Requirement-level min_provenance_tier
        
        Args:
            prompt_spec: Full prompt specification
            requirement: Specific data requirement
            
        Returns:
            Required tier level (0-3+)
        """
        market_tier = prompt_spec.market.min_provenance_tier
        requirement_tier = requirement.min_provenance_tier
        return max(market_tier, requirement_tier)
    
    def classify_provenance(
        self,
        target: SourceTarget,
        artifact: FetchedArtifact,
        *,
        proof_kind: Optional[str] = None,
        proof_blob: Optional[str] = None
    ) -> ProvenanceProof:
        """
        Classify the provenance tier for a fetched artifact.
        
        Args:
            target: Source target that was fetched
            artifact: Result of the fetch
            proof_kind: Override proof kind (default: infer from response)
            proof_blob: Optional proof data
            
        Returns:
            ProvenanceProof with appropriate tier
        """
        # Determine proof kind
        if proof_kind is None:
            proof_kind = self._infer_proof_kind(target, artifact)
        
        # Get tier from mapping
        tier = self.tier_mapping.get(proof_kind, 0)
        
        return ProvenanceProof(
            tier=tier,
            kind=proof_kind,  # type: ignore
            proof_blob=proof_blob,
            verifier=None,
            extra={
                "source_id": target.source_id,
                "uri": target.uri,
                "status_code": artifact.status_code,
            }
        )
    
    def _infer_proof_kind(
        self,
        target: SourceTarget,
        artifact: FetchedArtifact
    ) -> str:
        """
        Infer proof kind from target and artifact.
        
        Args:
            target: Source target
            artifact: Fetched artifact
            
        Returns:
            Inferred proof kind
        """
        # Check for signature in response headers
        if artifact.response_headers:
            # Look for common signature headers
            sig_headers = ["x-signature", "signature", "x-api-signature"]
            for header in sig_headers:
                if header.lower() in {h.lower() for h in artifact.response_headers}:
                    return "signature"
        
        # Check for zkTLS proof (would be in proof_blob)
        # For now, default to "none" for plain HTTP
        return "none"
    
    def check_item_tier(
        self,
        item: EvidenceItem,
        required_tier: int
    ) -> CheckResult:
        """
        Check if an evidence item meets tier requirement.
        
        Args:
            item: Evidence item to check
            required_tier: Required minimum tier
            
        Returns:
            CheckResult with pass/fail status
        """
        actual_tier = item.provenance.tier
        ok = actual_tier >= required_tier
        
        severity = "info" if ok else ("error" if self.strict_mode else "warn")
        
        return CheckResult(
            check_id=f"tier_check_{item.evidence_id}",
            ok=ok,
            severity=severity,
            message=(
                f"Evidence {item.evidence_id}: tier {actual_tier} "
                f">= required {required_tier}" if ok else
                f"Evidence {item.evidence_id}: tier {actual_tier} "
                f"< required {required_tier}"
            ),
            details={
                "evidence_id": item.evidence_id,
                "actual_tier": actual_tier,
                "required_tier": required_tier,
                "provenance_kind": item.provenance.kind,
            }
        )
    
    def enforce(
        self,
        bundle: EvidenceBundle,
        prompt_spec: PromptSpec
    ) -> VerificationResult:
        """
        Enforce tier policy on entire evidence bundle.
        
        Args:
            bundle: Evidence bundle to verify
            prompt_spec: Prompt specification with tier requirements
            
        Returns:
            VerificationResult with all checks
        """
        checks: list[CheckResult] = []
        all_ok = True
        failed_evidence_id: Optional[str] = None
        
        # Build requirement map
        requirement_map = {
            req.requirement_id: req
            for req in prompt_spec.data_requirements
        }
        
        for item in bundle.items:
            # Find requirement for this item
            req_id = item.requirement_id
            if req_id and req_id in requirement_map:
                required_tier = self.required_tier(
                    prompt_spec,
                    requirement_map[req_id]
                )
            else:
                # Use market-level tier if no specific requirement
                required_tier = prompt_spec.market.min_provenance_tier
            
            check = self.check_item_tier(item, required_tier)
            checks.append(check)
            
            if not check.ok:
                all_ok = False
                if failed_evidence_id is None:
                    failed_evidence_id = item.evidence_id
        
        # Build result
        if all_ok:
            return VerificationResult(ok=True, checks=checks)
        else:
            return VerificationResult(
                ok=False if self.strict_mode else True,
                checks=checks,
                challenge=ChallengeRef(
                    kind="provenance_tier",
                    evidence_id=failed_evidence_id,
                    reason="Provenance tier below required minimum"
                ),
                error=CournotError(
                    code=ErrorCodes.TIER_POLICY_VIOLATION,
                    message="One or more evidence items do not meet tier requirements",
                    details={"failed_count": sum(1 for c in checks if not c.ok)}
                ) if self.strict_mode else None
            )
    
    def get_tier_for_kind(self, kind: str) -> int:
        """Get tier level for a proof kind."""
        return self.tier_mapping.get(kind, 0)


class SelectionPolicyEnforcer:
    """
    Enforces evidence selection policies.
    
    Validates that collected evidence matches the selection
    policy defined in data requirements.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize enforcer.
        
        Args:
            strict_mode: If True, fail on policy violations
        """
        self.strict_mode = strict_mode
    
    def enforce_quorum(
        self,
        items: list[EvidenceItem],
        requirement: DataRequirement,
        required_tier: int
    ) -> VerificationResult:
        """
        Enforce quorum policy for a requirement.
        
        Args:
            items: Evidence items for this requirement
            requirement: Data requirement with policy
            required_tier: Required minimum tier
            
        Returns:
            VerificationResult
        """
        policy = requirement.selection_policy
        checks: list[CheckResult] = []
        
        # Count items meeting tier requirement
        qualifying_items = [
            item for item in items
            if item.provenance.tier >= required_tier
        ]
        
        # Check based on strategy
        if policy.strategy == "single_best":
            ok = len(qualifying_items) >= 1
            message = (
                f"single_best: found {len(qualifying_items)} qualifying items"
            )
        
        elif policy.strategy == "multi_source_quorum":
            ok = len(qualifying_items) >= policy.quorum
            message = (
                f"quorum: {len(qualifying_items)}/{policy.quorum} required"
            )
        
        elif policy.strategy == "fallback_chain":
            ok = len(qualifying_items) >= 1
            message = (
                f"fallback_chain: found {len(qualifying_items)} qualifying items"
            )
        
        else:
            ok = False
            message = f"Unknown selection strategy: {policy.strategy}"
        
        checks.append(CheckResult(
            check_id=f"selection_policy_{requirement.requirement_id}",
            ok=ok,
            severity="info" if ok else ("error" if self.strict_mode else "warn"),
            message=message,
            details={
                "requirement_id": requirement.requirement_id,
                "strategy": policy.strategy,
                "quorum": policy.quorum,
                "qualifying_count": len(qualifying_items),
                "total_count": len(items),
            }
        ))
        
        if ok:
            return VerificationResult(ok=True, checks=checks)
        else:
            return VerificationResult(
                ok=False if self.strict_mode else True,
                checks=checks,
                challenge=ChallengeRef(
                    kind="evidence_leaf",
                    reason=f"Selection policy not satisfied: {message}"
                ),
                error=CournotError(
                    code=ErrorCodes.QUORUM_NOT_MET if "quorum" in policy.strategy else ErrorCodes.SELECTION_POLICY_VIOLATION,
                    message=message
                ) if self.strict_mode else None
            )