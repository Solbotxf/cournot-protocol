"""
Module 05 - Collector Agent

Main collector agent that executes ToolPlan + PromptSpec.data_requirements
to produce a deterministic, policy-compliant EvidenceBundle.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from core.schemas.evidence import (
    EvidenceBundle,
    EvidenceItem,
    RetrievalReceipt,
    SourceDescriptor,
)
from core.schemas.prompts import DataRequirement, PromptSpec, SourceTarget
from core.schemas.transport import ToolPlan
from core.schemas.verification import VerificationResult

from agents.base_agent import AgentContext, BaseAgent
from agents.collector.data_sources import (
    BaseSource,
    FetchedArtifact,
    get_source_for_target,
)
from agents.collector.verification import (
    SelectionPolicyEnforcer,
    SignatureVerifier,
    TierPolicy,
    ZkTLSVerifier,
)
from agents.collector.collector_utils import (
    compute_bundle_id,
    compute_evidence_id,
    compute_request_fingerprint,
    compute_response_fingerprint,
    compute_tier_distribution,
    map_method,
    normalize_content,
)


@dataclass
class CollectorConfig:
    """Configuration for the collector agent."""
    default_timeout_s: int = 20
    strict_tier_policy: bool = True
    include_timestamps: bool = False
    collector_id: str = "collector_v1"


class CollectorAgent(BaseAgent[tuple[PromptSpec, ToolPlan], EvidenceBundle]):
    """
    Collector agent for fetching evidence with provenance.
    
    Executes explicit SourceTarget requests from data requirements,
    captures retrieval receipts, attaches provenance proofs, and
    enforces tier/selection policies.
    """
    
    agent_type: str = "collector"
    agent_version: str = "1.0.0"
    
    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        source_adapters: Optional[dict[str, BaseSource]] = None,
        context: Optional[AgentContext] = None
    ):
        super().__init__(context)
        self.collector_config = config or CollectorConfig()
        self.source_adapters = source_adapters or {}
        
        self.tier_policy = TierPolicy(
            strict_mode=self.collector_config.strict_tier_policy
        )
        self.selection_enforcer = SelectionPolicyEnforcer(
            strict_mode=self.collector_config.strict_tier_policy
        )
        self.signature_verifier = SignatureVerifier()
        self.zktls_verifier = ZkTLSVerifier()
    
    def run(
        self,
        input_data: tuple[PromptSpec, ToolPlan]
    ) -> EvidenceBundle:
        """Execute collection according to PromptSpec and ToolPlan."""
        prompt_spec, tool_plan = input_data
        self._validate_inputs(prompt_spec, tool_plan)
        
        all_items: list[EvidenceItem] = []
        policy_failures: list[dict[str, Any]] = []
        
        for requirement in prompt_spec.data_requirements:
            items, failures = self._collect_for_requirement(
                requirement, prompt_spec, tool_plan
            )
            all_items.extend(items)
            policy_failures.extend(failures)
        
        return self._build_bundle(prompt_spec, tool_plan, all_items, policy_failures)
    
    def collect(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan
    ) -> EvidenceBundle:
        """Convenience method matching Module 05 interface."""
        return self.run((prompt_spec, tool_plan))
    
    def _validate_inputs(self, prompt_spec: PromptSpec, tool_plan: ToolPlan) -> None:
        """Validate input specifications."""
        if not prompt_spec.data_requirements:
            raise ValueError("PromptSpec must have at least one data requirement")
        
        for req in prompt_spec.data_requirements:
            if not req.source_targets:
                raise ValueError(f"Requirement {req.requirement_id} has no source_targets")
    
    def _collect_for_requirement(
        self,
        requirement: DataRequirement,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        """Collect evidence for a single requirement."""
        required_tier = self.tier_policy.required_tier(prompt_spec, requirement)
        policy = requirement.selection_policy
        
        if policy.strategy == "fallback_chain":
            return self._collect_fallback_chain(requirement, prompt_spec, required_tier)
        elif policy.strategy == "multi_source_quorum":
            return self._collect_quorum(requirement, prompt_spec, required_tier)
        else:  # single_best
            return self._collect_single_best(requirement, prompt_spec, required_tier)
    
    def _collect_fallback_chain(
        self,
        requirement: DataRequirement,
        prompt_spec: PromptSpec,
        required_tier: int
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        """Collect using fallback chain strategy."""
        items: list[EvidenceItem] = []
        failures: list[dict[str, Any]] = []
        
        for idx, target in enumerate(requirement.source_targets):
            if not self._is_source_allowed(target, prompt_spec):
                failures.append({
                    "requirement_id": requirement.requirement_id,
                    "target_idx": idx,
                    "reason": f"Source {target.source_id} not allowed"
                })
                continue
            
            item, error = self._fetch_and_create_item(target, requirement, idx)
            
            if error:
                failures.append({
                    "requirement_id": requirement.requirement_id,
                    "target_idx": idx,
                    "reason": error
                })
                continue
            
            if item.provenance.tier >= required_tier:
                items.append(item)
                break  # Fallback chain stops on first success
            else:
                failures.append({
                    "requirement_id": requirement.requirement_id,
                    "target_idx": idx,
                    "reason": f"Tier {item.provenance.tier} < required {required_tier}"
                })
        
        return items, failures
    
    def _collect_quorum(
        self,
        requirement: DataRequirement,
        prompt_spec: PromptSpec,
        required_tier: int
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        """Collect using multi-source quorum strategy."""
        items: list[EvidenceItem] = []
        failures: list[dict[str, Any]] = []
        policy = requirement.selection_policy
        
        for idx, target in enumerate(requirement.source_targets):
            if len(items) >= policy.max_sources:
                break
            
            if not self._is_source_allowed(target, prompt_spec):
                failures.append({
                    "requirement_id": requirement.requirement_id,
                    "target_idx": idx,
                    "reason": f"Source {target.source_id} not allowed"
                })
                continue
            
            item, error = self._fetch_and_create_item(target, requirement, idx)
            
            if error:
                failures.append({
                    "requirement_id": requirement.requirement_id,
                    "target_idx": idx,
                    "reason": error
                })
                continue
            
            if item.provenance.tier >= required_tier:
                items.append(item)
            else:
                failures.append({
                    "requirement_id": requirement.requirement_id,
                    "target_idx": idx,
                    "reason": f"Tier {item.provenance.tier} < required {required_tier}"
                })
        
        return items, failures
    
    def _collect_single_best(
        self,
        requirement: DataRequirement,
        prompt_spec: PromptSpec,
        required_tier: int
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        """Collect using single best strategy."""
        return self._collect_fallback_chain(requirement, prompt_spec, required_tier)
    
    def _is_source_allowed(self, target: SourceTarget, prompt_spec: PromptSpec) -> bool:
        """Check if source is allowed by market policy."""
        for policy in prompt_spec.market.allowed_sources:
            if policy.source_id == target.source_id:
                return policy.allow
        return True  # Default to allowed if not explicitly listed
    
    def _fetch_and_create_item(
        self,
        target: SourceTarget,
        requirement: DataRequirement,
        target_idx: int
    ) -> tuple[Optional[EvidenceItem], Optional[str]]:
        """Fetch from target and create EvidenceItem."""
        try:
            adapter = self._get_adapter(target.source_id)
        except ValueError as e:
            return None, str(e)
        
        request_fingerprint = compute_request_fingerprint(target)
        artifact = adapter.fetch(target, timeout_s=self.collector_config.default_timeout_s)
        
        if not artifact.success:
            return None, artifact.error or "Fetch failed"
        
        response_fingerprint = compute_response_fingerprint(artifact)
        evidence_id = compute_evidence_id(
            requirement.requirement_id,
            request_fingerprint,
            response_fingerprint
        )
        provenance = self.tier_policy.classify_provenance(target, artifact)
        
        item = EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=requirement.requirement_id,
            source=SourceDescriptor(
                source_id=target.source_id,
                uri=target.uri,
                provider=target.source_id,
            ),
            retrieval=RetrievalReceipt(
                retrieved_at=(
                    datetime.now(timezone.utc)
                    if self.collector_config.include_timestamps
                    else None
                ),
                method=map_method(target.method),
                tool=f"{adapter.source_id}_adapter",
                request_fingerprint=request_fingerprint,
                response_fingerprint=response_fingerprint,
                status_code=artifact.status_code,
            ),
            provenance=provenance,
            content_type=target.expected_content_type,  # type: ignore
            content=artifact.parsed if artifact.parsed else artifact.raw_bytes.decode("utf-8", errors="replace"),
            normalized=normalize_content(artifact, requirement),
            confidence=1.0 if artifact.success else 0.0,
        )
        
        return item, None
    
    def _get_adapter(self, source_id: str) -> BaseSource:
        """Get source adapter for source_id."""
        if source_id in self.source_adapters:
            return self.source_adapters[source_id]
        return get_source_for_target(source_id)
    
    def _build_bundle(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
        items: list[EvidenceItem],
        policy_failures: list[dict[str, Any]]
    ) -> EvidenceBundle:
        """Build final EvidenceBundle."""
        bundle_id = compute_bundle_id(prompt_spec.market.market_id, tool_plan.plan_id)
        
        provenance_summary = {
            "total_items": len(items),
            "requirements_count": len(prompt_spec.data_requirements),
            "policy_failures": policy_failures,
            "adapters_used": list({item.source.source_id for item in items}),
            "tier_distribution": compute_tier_distribution(items),
        }
        
        if self.collector_config.include_timestamps:
            provenance_summary["collected_at"] = datetime.now(timezone.utc).isoformat()
        
        return EvidenceBundle(
            bundle_id=bundle_id,
            collector_id=self.collector_config.collector_id,
            collection_time=(
                datetime.now(timezone.utc)
                if self.collector_config.include_timestamps
                else None
            ),
            items=items,
            provenance_summary=provenance_summary,
        )
    
    def verify_bundle(
        self,
        bundle: EvidenceBundle,
        prompt_spec: PromptSpec
    ) -> VerificationResult:
        """Verify evidence bundle against prompt specification."""
        tier_result = self.tier_policy.enforce(bundle, prompt_spec)
        all_checks = list(tier_result.checks)
        
        # Group items by requirement
        items_by_req: dict[str, list[EvidenceItem]] = {}
        for item in bundle.items:
            req_id = item.requirement_id or "unknown"
            items_by_req.setdefault(req_id, []).append(item)
        
        # Check selection policy for each requirement
        for req in prompt_spec.data_requirements:
            items = items_by_req.get(req.requirement_id, [])
            required_tier = self.tier_policy.required_tier(prompt_spec, req)
            
            selection_result = self.selection_enforcer.enforce_quorum(
                items, req, required_tier
            )
            all_checks.extend(selection_result.checks)
        
        # Verify provenance proofs
        for item in bundle.items:
            if item.provenance.kind == "signature":
                sig_result = self.signature_verifier.verify(item)
                all_checks.extend(sig_result.checks)
            elif item.provenance.kind == "zktls":
                zk_result = self.zktls_verifier.verify(item)
                all_checks.extend(zk_result.checks)
        
        all_ok = all(c.ok for c in all_checks if c.severity == "error")
        
        return VerificationResult(
            ok=all_ok,
            checks=all_checks,
            challenge=tier_result.challenge if not tier_result.ok else None,
            error=tier_result.error if not tier_result.ok else None,
        )