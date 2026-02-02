"""
Collection Engine

Executes ToolPlans to collect evidence from sources.
Handles:
- Parallel vs sequential fetching
- Fallback chain execution
- Quorum checking
- Error recovery
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from core.schemas import (
    DataRequirement,
    EvidenceBundle,
    EvidenceItem,
    PromptSpec,
    ToolCallRecord,
    ToolExecutionLog,
    ToolPlan,
    SourceTarget
)
from .adapters import get_adapter

if TYPE_CHECKING:
    from agents.context import AgentContext


class CollectionEngine:
    """
    Engine for executing collection plans.
    
    Strategies:
    - single_best: Fetch from first source, stop on success
    - fallback_chain: Try sources in order until one succeeds
    - multi_source_quorum: Fetch from multiple sources, require quorum agreement
    """
    
    def __init__(
        self,
        *,
        max_retries: int = 2,
        continue_on_error: bool = True,
    ) -> None:
        """
        Initialize collection engine.
        
        Args:
            max_retries: Maximum retries per source
            continue_on_error: Whether to continue if a source fails
        """
        self.max_retries = max_retries
        self.continue_on_error = continue_on_error
    
    def execute(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> tuple[EvidenceBundle, ToolExecutionLog]:
        """
        Execute a collection plan.
        
        Args:
            ctx: Agent context
            prompt_spec: The prompt specification with data requirements
            tool_plan: The plan to execute
        
        Returns:
            Tuple of (EvidenceBundle, ToolExecutionLog)
        """
        # Initialize bundle
        bundle = EvidenceBundle(
            bundle_id=self._generate_bundle_id(tool_plan.plan_id),
            market_id=prompt_spec.market_id,
            plan_id=tool_plan.plan_id,
        )
        
        # Initialize execution log
        execution_log = ToolExecutionLog(
            plan_id=tool_plan.plan_id,
            started_at=ctx.now().isoformat(),
        )
        
        # Track unfulfilled requirements
        all_requirement_ids = set(tool_plan.requirements)
        fulfilled_ids: set[str] = set()
        
        # Execute each requirement
        for req_id in tool_plan.requirements:
            # Find the requirement in prompt_spec
            requirement = prompt_spec.get_requirement_by_id(req_id)
            if not requirement:
                ctx.warning(f"Requirement {req_id} not found in prompt_spec")
                continue
            
            # Execute based on selection strategy
            evidence_items = self._execute_requirement(
                ctx, requirement, execution_log
            )
            
            # Add to bundle
            for item in evidence_items:
                bundle.add_item(item)
                if item.is_valid:
                    fulfilled_ids.add(req_id)
        
        # Update bundle summary
        bundle.requirements_fulfilled = list(fulfilled_ids)
        bundle.requirements_unfulfilled = list(all_requirement_ids - fulfilled_ids)
        bundle.collected_at = ctx.now()
        
        # Complete execution log
        execution_log.ended_at = ctx.now().isoformat()
        
        return bundle, execution_log
    
    def _execute_requirement(
        self,
        ctx: "AgentContext",
        requirement: DataRequirement,
        execution_log: ToolExecutionLog,
    ) -> list[EvidenceItem]:
        """Execute a single requirement based on its selection policy."""
        strategy = requirement.selection_policy.strategy
        
        if strategy == "single_best":
            return self._execute_single_best(ctx, requirement, execution_log)
        elif strategy == "fallback_chain":
            return self._execute_fallback_chain(ctx, requirement, execution_log)
        elif strategy == "multi_source_quorum":
            return self._execute_quorum(ctx, requirement, execution_log)
        else:
            ctx.warning(f"Unknown strategy {strategy}, using single_best")
            return self._execute_single_best(ctx, requirement, execution_log)
    
    def _execute_single_best(
        self,
        ctx: "AgentContext",
        requirement: DataRequirement,
        execution_log: ToolExecutionLog,
    ) -> list[EvidenceItem]:
        """Execute single_best strategy: fetch from first source only."""
        if not requirement.source_targets:
            return []
        
        target = requirement.source_targets[0]
        ctx.debug(f"execute requirement with target:{target}")
        evidence = self._fetch_with_retry(ctx, target, requirement.requirement_id, execution_log)
        
        return [evidence] if evidence else []
    
    def _execute_fallback_chain(
        self,
        ctx: "AgentContext",
        requirement: DataRequirement,
        execution_log: ToolExecutionLog,
    ) -> list[EvidenceItem]:
        """Execute fallback_chain strategy: try sources until one succeeds."""
        results: list[EvidenceItem] = []
        
        for target in requirement.source_targets:
            evidence = self._fetch_with_retry(ctx, target, requirement.requirement_id, execution_log)
            
            if evidence:
                results.append(evidence)
                if evidence.is_valid:
                    # Success - stop trying more sources
                    break
        
        return results
    
    def _execute_quorum(
        self,
        ctx: "AgentContext",
        requirement: DataRequirement,
        execution_log: ToolExecutionLog,
    ) -> list[EvidenceItem]:
        """Execute multi_source_quorum strategy: fetch from multiple sources."""
        results: list[EvidenceItem] = []
        policy = requirement.selection_policy
        
        # Fetch from up to max_sources
        for target in requirement.source_targets[:policy.max_sources]:
            evidence = self._fetch_with_retry(ctx, target, requirement.requirement_id, execution_log)
            if evidence:
                results.append(evidence)
        
        # Check if we met min_sources
        valid_count = sum(1 for e in results if e.is_valid)
        if valid_count < policy.min_sources:
            ctx.warning(
                f"Requirement {requirement.requirement_id}: "
                f"only {valid_count}/{policy.min_sources} sources succeeded"
            )
        
        return results
    
    def _fetch_with_retry(
        self,
        ctx: AgentContext,
        target: SourceTarget,
        requirement_id: str,
        execution_log: ToolExecutionLog,
    ) -> EvidenceItem | None:
        """Fetch from a source with retries."""
        
        adapter = get_adapter(target.source_id)
        
        for attempt in range(self.max_retries + 1):
            # Record tool call start
            input_data = {
                "uri": target.uri,
                "method": target.method,
                "requirement_id": requirement_id,
                "attempt": attempt + 1,
            }
            if target.operation == "search" and target.search_query:
                input_data["operation"] = "search"
                input_data["search_query"] = target.search_query
            call_record = ToolCallRecord(
                tool=f"fetch:{target.source_id}",
                input=input_data,
                started_at=ctx.now().isoformat(),
            )
            
            try:
                # Execute fetch
                evidence = adapter.fetch(ctx, target, requirement_id)
                
                # Complete call record
                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "success": evidence.success,
                    "evidence_id": evidence.evidence_id,
                    "status_code": evidence.status_code,
                }
                if evidence.error:
                    call_record.error = evidence.error
                
                execution_log.add_call(call_record)
                
                if evidence.is_valid:
                    return evidence
                
                # If failed but we have more retries, continue
                if attempt < self.max_retries:
                    ctx.debug(f"Retry {attempt + 1} for {target.uri}")
                    continue
                
                # Last attempt failed
                return evidence
                
            except Exception as e:
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = str(e)
                execution_log.add_call(call_record)
                
                if not self.continue_on_error:
                    raise
                
                ctx.warning(f"Fetch error for {target.uri}: {e}")
                
                if attempt >= self.max_retries:
                    # Return error evidence
                    return EvidenceItem(
                        evidence_id=f"ev_error_{hashlib.sha256(target.uri.encode()).hexdigest()[:8]}",
                        requirement_id=requirement_id,
                        provenance=Provenance(
                            source_id=target.source_id,
                            source_uri=target.uri,
                            tier=0,
                        ),
                        success=False,
                        error=str(e),
                    )
        
        return None
    
    def _generate_bundle_id(self, plan_id: str) -> str:
        """Generate a deterministic bundle ID."""
        hash_bytes = hashlib.sha256(plan_id.encode()).digest()
        return f"bundle_{hash_bytes[:8].hex()}"


# Import Provenance for error evidence
from core.schemas import Provenance
