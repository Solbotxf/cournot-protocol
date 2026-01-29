"""
Collector Agent

Executes ToolPlans to collect evidence from external sources.

Two implementations:
- CollectorHTTP: Makes real HTTP requests (production)
- CollectorMock: Returns mock data (testing)

Selection logic:
- If ctx.http is available → use HTTP collector
- Otherwise → use mock collector
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, AgentStep, BaseAgent
from agents.registry import register_agent
from core.schemas import (
    CheckResult,
    EvidenceBundle,
    PromptSpec,
    ToolExecutionLog,
    ToolPlan,
    VerificationResult,
)

from .engine import CollectionEngine
from .adapters import MockAdapter, get_adapter

if TYPE_CHECKING:
    from agents.context import AgentContext


class CollectorHTTP(BaseAgent):
    """
    HTTP-based Collector Agent.
    
    Makes real HTTP requests to collect evidence from sources.
    Primary production implementation.
    
    Features:
    - Executes ToolPlan from Prompt Engineer
    - Supports multiple selection strategies
    - Automatic retry on failure
    - Receipt recording for all requests
    """
    
    _name = "CollectorHTTP"
    _version = "v1"
    _capabilities = {AgentCapability.NETWORK}
    
    def __init__(
        self,
        *,
        max_retries: int = 2,
        continue_on_error: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize HTTP collector.
        
        Args:
            max_retries: Maximum retries per source
            continue_on_error: Whether to continue if a source fails
        """
        super().__init__(**kwargs)
        self.engine = CollectionEngine(
            max_retries=max_retries,
            continue_on_error=continue_on_error,
        )
    
    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """
        Execute collection plan.
        
        Args:
            ctx: Agent context with HTTP client
            prompt_spec: The prompt specification
            tool_plan: The tool plan to execute
        
        Returns:
            AgentResult with (EvidenceBundle, ToolExecutionLog) as output
        """
        ctx.info(f"CollectorHTTP executing plan {tool_plan.plan_id}")
        
        # Check HTTP client
        if not ctx.http:
            return AgentResult.failure(
                error="HTTP client not available",
            )
        
        try:
            # Execute collection
            bundle, execution_log = self.engine.execute(ctx, prompt_spec, tool_plan)
            
            # Validate results
            verification = self._validate_output(bundle, tool_plan)
            
            ctx.info(
                f"Collection complete: {bundle.total_sources_succeeded}/"
                f"{bundle.total_sources_attempted} sources succeeded, "
                f"{len(bundle.requirements_fulfilled)}/{len(tool_plan.requirements)} requirements fulfilled"
            )
            
            return AgentResult(
                output=(bundle, execution_log),
                verification=verification,
                receipts=ctx.get_receipt_refs(),
                success=bundle.has_evidence,
                error=None if bundle.has_evidence else "No evidence collected",
                metadata={
                    "collector": "http",
                    "bundle_id": bundle.bundle_id,
                    "total_sources_attempted": bundle.total_sources_attempted,
                    "total_sources_succeeded": bundle.total_sources_succeeded,
                    "requirements_fulfilled": bundle.requirements_fulfilled,
                    "requirements_unfulfilled": bundle.requirements_unfulfilled,
                },
            )
            
        except Exception as e:
            ctx.error(f"Collection failed: {e}")
            return AgentResult.failure(error=str(e))
    
    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        """Validate collection results."""
        checks: list[CheckResult] = []
        
        # Check 1: Has evidence
        if bundle.has_evidence:
            checks.append(CheckResult.passed(
                check_id="has_evidence",
                message=f"Collected {len(bundle.items)} evidence items",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_evidence",
                message="No evidence was collected",
            ))
        
        # Check 2: Requirements fulfilled
        unfulfilled = bundle.requirements_unfulfilled
        if not unfulfilled:
            checks.append(CheckResult.passed(
                check_id="requirements_fulfilled",
                message="All requirements fulfilled",
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="requirements_fulfilled",
                message=f"Unfulfilled requirements: {unfulfilled}",
            ))
        
        # Check 3: Success rate
        rate = bundle.success_rate
        if rate >= 0.5:
            checks.append(CheckResult.passed(
                check_id="success_rate",
                message=f"Success rate: {rate:.1%}",
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="success_rate",
                message=f"Low success rate: {rate:.1%}",
            ))
        
        # Check 4: Provenance tiers
        valid_evidence = bundle.get_valid_evidence()
        if valid_evidence:
            max_tier = max(e.provenance.tier for e in valid_evidence)
            min_tier = tool_plan.min_provenance_tier
            if max_tier >= min_tier:
                checks.append(CheckResult.passed(
                    check_id="provenance_tier",
                    message=f"Highest tier: {max_tier} (required: {min_tier})",
                ))
            else:
                checks.append(CheckResult.warning(
                    check_id="provenance_tier",
                    message=f"Highest tier {max_tier} below required {min_tier}",
                ))
        
        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)


class CollectorMock(BaseAgent):
    """
    Mock Collector Agent.
    
    Returns preset mock data without making real requests.
    Used for testing and development.
    
    Features:
    - Deterministic outputs
    - Configurable mock responses
    - No network dependencies
    """
    
    _name = "CollectorMock"
    _version = "v1"
    _capabilities = {AgentCapability.DETERMINISTIC, AgentCapability.REPLAY}
    
    def __init__(
        self,
        *,
        mock_responses: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize mock collector.
        
        Args:
            mock_responses: Map of URI patterns to mock responses
        """
        super().__init__(**kwargs)
        self.mock_responses = mock_responses or {}
    
    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """
        Execute collection with mock data.
        
        Args:
            ctx: Agent context
            prompt_spec: The prompt specification
            tool_plan: The tool plan to execute
        
        Returns:
            AgentResult with (EvidenceBundle, ToolExecutionLog) as output
        """
        ctx.info(f"CollectorMock executing plan {tool_plan.plan_id}")
        
        from .adapters import MockAdapter
        
        # Create mock adapter with responses
        mock_adapter = MockAdapter(responses=self.mock_responses)
        
        # Build bundle manually
        from core.schemas import EvidenceBundle, ToolExecutionLog, ToolCallRecord
        
        bundle = EvidenceBundle(
            bundle_id=f"mock_{tool_plan.plan_id}",
            market_id=prompt_spec.market_id,
            plan_id=tool_plan.plan_id,
        )
        
        execution_log = ToolExecutionLog(
            plan_id=tool_plan.plan_id,
            started_at=ctx.now().isoformat(),
        )
        
        # Process each requirement
        for req_id in tool_plan.requirements:
            requirement = prompt_spec.get_requirement_by_id(req_id)
            if not requirement:
                continue
            
            for target in requirement.source_targets:
                # Record tool call
                call_record = ToolCallRecord(
                    tool=f"mock:{target.source_id}",
                    input={"uri": target.uri, "requirement_id": req_id},
                    started_at=ctx.now().isoformat(),
                )
                
                # Get mock evidence
                evidence = mock_adapter.fetch(ctx, target, req_id)
                
                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "success": evidence.success,
                    "evidence_id": evidence.evidence_id,
                }
                
                execution_log.add_call(call_record)
                bundle.add_item(evidence)
                
                # For single_best, stop after first source
                if requirement.selection_policy.strategy == "single_best":
                    break
        
        bundle.collected_at = ctx.now()
        bundle.requirements_fulfilled = list(set(tool_plan.requirements))
        execution_log.ended_at = ctx.now().isoformat()
        
        return AgentResult(
            output=(bundle, execution_log),
            verification=VerificationResult.success([
                CheckResult.passed("mock_collection", "Mock collection completed"),
            ]),
            receipts=ctx.get_receipt_refs(),
            metadata={
                "collector": "mock",
                "bundle_id": bundle.bundle_id,
                "items_collected": len(bundle.items),
            },
        )


def get_collector(
    ctx: "AgentContext",
    *,
    prefer_http: bool = True,
    mock_responses: dict[str, Any] | None = None,
) -> BaseAgent:
    """
    Get the appropriate collector based on context.
    
    Args:
        ctx: Agent context
        prefer_http: If True and HTTP client available, use HTTP collector
        mock_responses: Mock responses for mock collector
    
    Returns:
        CollectorHTTP or CollectorMock
    """
    if prefer_http and ctx.http is not None:
        return CollectorHTTP()
    return CollectorMock(mock_responses=mock_responses)


def collect_evidence(
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    tool_plan: ToolPlan,
    *,
    prefer_http: bool = True,
) -> AgentResult:
    """
    Convenience function to collect evidence.
    
    Args:
        ctx: Agent context
        prompt_spec: The prompt specification
        tool_plan: The tool plan to execute
        prefer_http: If True, prefer HTTP collector
    
    Returns:
        AgentResult with (EvidenceBundle, ToolExecutionLog) as output
    """
    collector = get_collector(ctx, prefer_http=prefer_http)
    return collector.run(ctx, prompt_spec, tool_plan)


# Register agents with the global registry
def _register_agents() -> None:
    """Register collector agents."""
    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorHTTP",
        factory=lambda ctx: CollectorHTTP(),
        capabilities={AgentCapability.NETWORK},
        priority=100,  # Primary
    )
    
    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorMock",
        factory=lambda ctx: CollectorMock(),
        capabilities={AgentCapability.DETERMINISTIC, AgentCapability.REPLAY},
        priority=50,  # Fallback
        is_fallback=True,
    )


# Auto-register on import
_register_agents()
