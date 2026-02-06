"""
Collector Agent

Executes ToolPlans to collect evidence from external sources.

Three implementations:
- CollectorLLM: Uses LLM web browsing to fetch and interpret content (preferred)
- CollectorHTTP: Makes real HTTP requests (production)
- CollectorMock: Returns mock data (testing)

Selection logic:
- If ctx.llm is available → use LLM collector
- If ctx.http is available → use HTTP collector
- Otherwise → use mock collector
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, AgentStep, BaseAgent
from agents.registry import register_agent
from core.schemas import (
    CheckResult,
    EvidenceBundle,
    EvidenceItem,
    Provenance,
    PromptSpec,
    ToolCallRecord,
    ToolExecutionLog,
    ToolPlan,
    VerificationResult,
)

from .engine import CollectionEngine
from .adapters import MockAdapter, get_adapter

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas import DataRequirement, SourceTarget


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


COLLECTOR_LLM_SYSTEM_PROMPT = (
    "You are a data collection agent. Visit the given URL and extract specific data. "
    "Return ONLY valid JSON with the following schema: "
    '{"success": true/false, "extracted_fields": {...}, "parsed_value": ..., '
    '"content_summary": "...", "error": null or "..."}'
)

DISCOVERY_QUERY_SYSTEM_PROMPT = """
You are a Search Strategy Specialist for a prediction market oracle.
Your goal is to generate targeted search queries that will yield definitive, verifiable evidence for a specific data requirement.

### OBJECTIVES
1. **Diversity:** Generate a mix of queries:
   - *Broad:* To catch general news and context.
   - *Specific:* Targeting official domains (e.g., "site:gov", "site:nytimes.com") or specific file types if applicable.
   - *Forensic:* Looking for primary source documents (e.g., "official statement", "press release", "PDF").
2. **Temporal Awareness:** If the requirement implies a specific time (e.g., "2026"), ensure queries include that year to filter noise.

### OUTPUT FORMAT
Return a single valid JSON object:
{
    "queries": [
        "precise query string 1",
        "precise query string 2",
        "precise query string 3"
    ]
}
"""

DEFERRED_DISCOVERY_SYSTEM_PROMPT = """
You are the Chief Evidence Analyst for a Prediction Market Resolution Engine.
Your task is to analyze search results to determine the definitive truth of a real-world event relative to the "Current Date/Time" provided in the user context.

### CORE PROTOCOLS
1. **Chronological Filtering:** You MUST compare the "DATE_PUBLISHED" of sources against the "Current Date/Time". Disregard outdated speculation if newer, confirming evidence exists.
2. **Definition Adherence:** You must strictly evaluate evidence against the provided "Event Definition" and "Assumptions". 
3. **Source Hierarchy:**
   - **Tier 1 (Authoritative):** Official government sites (.gov), primary company press releases, major wire services (Reuters, AP, Bloomberg).
   - **Tier 2 (Secondary):** Reputable mainstream news (NYT, BBC, WSJ).
   - **Tier 3 (Low Confidence):** Tabloids, opinion blogs, social media, or sources with no clear date.
4. **Ambiguity Resolution:** If sources conflict, Tier 1 overrides Tier 2. If Tier 1 sources conflict, the most recent one prevails.

### OUTPUT FORMAT
You MUST return a single, valid JSON object. Do not include markdown formatting (like ```json).

{
    "reasoning_trace": "A detailed step-by-step deduction. 1) Identify the latest date among sources. 2) Check if the event deadline has passed. 3) Weigh conflicting reports. 4) Final conclusion.",

    "resolution_status": "RESOLVED" | "AMBIGUOUS" | "UNRESOLVED",

    "parsed_value": "The specific answer string extracted from the text (e.g., '14 days', 'Yes', '$500'). Return null if unresolved.",

    "confidence_score": 0.0 to 1.0 (A float representing certainty. 1.0 = undeniable proof from Tier 1 source. <0.5 = conflicting or missing data.),

    "evidence_sources": [
        {
            "source_id": "[X]",
            "url": "...",
            "credibility_tier": "Tier 1/Tier 2/Tier 3",
            "relevance_reason": "Briefly explain why this source was selected (e.g. 'Official government statement dated yesterday')."
        }
    ]
}
"""


class CollectorLLM(BaseAgent):
    """
    LLM-based Collector Agent.

    Uses LLM web browsing capability to fetch and interpret URL content.
    Instead of raw HTTP fetching, passes URL + extraction instructions to the LLM,
    which browses and returns structured data.

    Features:
    - LLM-interpreted data extraction
    - Automatic JSON repair loop
    - Provenance tier 2 (LLM-interpreted)
    - Higher priority than CollectorHTTP for automatic selection
    """

    _name = "CollectorLLM"
    _version = "v1"
    _capabilities = {AgentCapability.LLM}
    MAX_RETRIES = 2

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """
        Execute collection plan using LLM web browsing.

        Args:
            ctx: Agent context with LLM client
            prompt_spec: The prompt specification
            tool_plan: The tool plan to execute

        Returns:
            AgentResult with (EvidenceBundle, ToolExecutionLog) as output
        """
        ctx.info(f"CollectorLLM executing plan {tool_plan.plan_id}")

        if ctx.llm is None:
            return AgentResult.failure(error="LLM client not available")

        bundle = EvidenceBundle(
            bundle_id=f"llm_{tool_plan.plan_id}",
            market_id=prompt_spec.market_id,
            plan_id=tool_plan.plan_id,
        )

        execution_log = ToolExecutionLog(
            plan_id=tool_plan.plan_id,
            started_at=ctx.now().isoformat(),
        )

        for req_id in tool_plan.requirements:
            requirement = prompt_spec.get_requirement_by_id(req_id)
            ctx.info(f"requirement: {requirement} with id {req_id}")
            if not requirement:
                continue

            # Handle deferred source discovery
            if requirement.deferred_source_discovery:
                evidence = self._discover_sources(ctx, requirement, prompt_spec)
                call_record = ToolCallRecord(
                    tool="llm:discover",
                    input={
                        "requirement_id": req_id,
                        "description": requirement.description,
                        "deferred": True,
                    },
                    started_at=ctx.now().isoformat(),
                )
                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "success": evidence.success,
                    "evidence_id": evidence.evidence_id,
                }
                if evidence.error:
                    call_record.error = evidence.error
                execution_log.add_call(call_record)
                bundle.add_item(evidence)
                continue

            for target in requirement.source_targets:
                call_record = ToolCallRecord(
                    tool=f"llm:{target.source_id}",
                    input={"uri": target.uri, "requirement_id": req_id},
                    started_at=ctx.now().isoformat(),
                )

                evidence = self._fetch_via_llm(ctx, target, requirement, prompt_spec)

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "success": evidence.success,
                    "evidence_id": evidence.evidence_id,
                }
                if evidence.error:
                    call_record.error = evidence.error

                execution_log.add_call(call_record)
                bundle.add_item(evidence)

                # For single_best, stop after first source
                if requirement.selection_policy.strategy == "single_best":
                    break

        bundle.collected_at = ctx.now()
        bundle.requirements_fulfilled = list(set(
            item.requirement_id for item in bundle.items if item.success
        ))
        bundle.requirements_unfulfilled = [
            r for r in tool_plan.requirements
            if r not in bundle.requirements_fulfilled
        ]
        execution_log.ended_at = ctx.now().isoformat()

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
                "collector": "llm",
                "bundle_id": bundle.bundle_id,
                "total_sources_attempted": bundle.total_sources_attempted,
                "total_sources_succeeded": bundle.total_sources_succeeded,
                "requirements_fulfilled": bundle.requirements_fulfilled,
                "requirements_unfulfilled": bundle.requirements_unfulfilled,
            },
        )

    def _fetch_via_llm(
        self,
        ctx: "AgentContext",
        target: "SourceTarget",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
    ) -> EvidenceItem:
        """
        Fetch and extract data from a URL using LLM web browsing.

        Args:
            ctx: Agent context
            target: Source target with URI
            requirement: Data requirement to fulfill
            prompt_spec: The prompt specification

        Returns:
            EvidenceItem with extracted data or error
        """
        semantics = prompt_spec.prediction_semantics
        evidence_id = self._generate_evidence_id(requirement.requirement_id, target)

        user_prompt = (
            f"Visit this URL: {target.uri}\n\n"
            f"Extract the following data:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Entity: {semantics.target_entity}\n"
            f"- Predicate: {semantics.predicate}\n"
            f"- Threshold: {semantics.threshold or 'N/A'}\n\n"
            f"Return ONLY valid JSON with this schema:\n"
            f'{{"success": true/false, "extracted_fields": {{...relevant fields...}}, '
            f'"parsed_value": <the key value extracted>, '
            f'"content_summary": "<brief summary of page content>", '
            f'"error": null}}'
        )

        messages = [
            {"role": "system", "content": COLLECTOR_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = ctx.llm.chat(messages)
        raw_output = response.content

        ctx.info(f"collector LLM response: {raw_output}")
        # Parse with retry loop
        last_error: str | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                parsed = self._extract_json(raw_output)
                return self._build_evidence_item(
                    ctx, target, requirement.requirement_id, parsed, raw_output
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = str(e)
                ctx.warning(f"JSON parse attempt {attempt + 1} failed: {last_error}")

                if attempt < self.MAX_RETRIES:
                    repair_prompt = (
                        f"The JSON was invalid: {last_error}. "
                        "Please fix and return valid JSON only."
                    )
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": repair_prompt})

                    response = ctx.llm.chat(messages)
                    raw_output = response.content

        # All retries exhausted — return error evidence
        ctx.error(f"LLM collection failed after {self.MAX_RETRIES + 1} attempts: {last_error}")
        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=requirement.requirement_id,
            provenance=Provenance(
                source_id=target.source_id,
                source_uri=target.uri,
                tier=2,
                fetched_at=ctx.now(),
                content_hash=self._hash_content(raw_output),
            ),
            success=False,
            error=f"JSON parse failed after {self.MAX_RETRIES + 1} attempts: {last_error}",
            raw_content=raw_output[:100],
        )

    def _discover_sources(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
    ) -> EvidenceItem:
        """
        Discover sources for a deferred requirement using LLM + web search.

        Flow:
        1. Ask LLM to generate search queries from the requirement context
        2. Execute queries via Serper API
        3. Feed search results to LLM to summarize and extract evidence (with temporal reasoning)
        """
        semantics = prompt_spec.prediction_semantics
        evidence_id = hashlib.sha256(
            f"discover:{requirement.requirement_id}".encode()
        ).hexdigest()[:16]
        event_def = prompt_spec.market.event_definition
        # Safely get assumptions list and join them into a string
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = "\n".join([f"- {a}" for a in assumptions_list]) if assumptions_list else "None"

        # Step 1: Generate search queries via LLM
        query_prompt = (
            f"Generate search queries for this data requirement:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Market question: {prompt_spec.market.question}\n"
            f"- Event Definition: {event_def}\n"
            f"- Key Assumptions: \n{assumptions_str}\n" # Helps generate queries like "OPM shutdown announcement" vs "holiday closure"
            f"- Entity: {semantics.target_entity}\n"
            f"- Predicate: {semantics.predicate}\n"
            f"- Threshold: {semantics.threshold or 'N/A'}\n"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": DISCOVERY_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": query_prompt},
        ]
        response = ctx.llm.chat(messages)

        try:
            content = response.content.strip()
            # Handle markdown-wrapped JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            query_data = json.loads(content)
            queries = query_data.get("queries", [])[:3]
        except (json.JSONDecodeError, KeyError):
            queries = [f"{semantics.target_entity} {semantics.predicate}"]

        if not queries:
            queries = [f"{semantics.target_entity} {semantics.predicate}"]

        # Step 2: Execute search queries via Serper
        search_results = self._execute_search_queries(ctx, queries)

        if not search_results:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="discover",
                    source_uri="serper:search",
                    tier=1,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="No search results found (Serper API may not be configured)",
            )

        # Step 3: Feed search results to LLM for synthesis
        # Use real system time (not ctx.now()) for temporal reasoning about current events
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


        # Format results with clear delimiters to prevent hallucination
        formatted_results = []
        for i, r in enumerate(search_results):
            source_block = (
                f"SOURCE_ID: [{i+1}]\n"
                f"TITLE: {r.get('title', 'Unknown')}\n"
                f"URL: {r.get('url', 'Unknown')}\n"
                f"DATE_PUBLISHED: {r.get('date', 'Unknown')}\n"
                f"SNIPPET: {r.get('snippet', '')}"
            )
            formatted_results.append(source_block)

        results_text = "\n\n---\n\n".join(formatted_results)

        synthesis_prompt = (
            f"### CONTEXT\n"
            f"Current Date/Time: {current_time_str}\n"
            f"Market Question: {prompt_spec.market.question}\n"
            f"Event Definition: {event_def}\n"
            f"Critical Assumptions (Must Follow):\n{assumptions_str}\n"
            f"Entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n"
            f"### DATA REQUIREMENT\n"
            f"{requirement.description}\n\n"
            f"### SEARCH RESULTS\n"
            f"{results_text}\n\n"
            f"### INSTRUCTIONS\n"
            f"1. Analyze the 'SOURCE_ID's above to answer the Market Question.\n"
            f"2. Compare the 'DATE_PUBLISHED' against 'Current Date/Time' to filter outdated info.\n"
            f"3. Extract the answer into the 'parsed_value' field.\n"
            f"4. If multiple sources conflict, explain why you chose the winner in 'reasoning_trace'.\n"
            f"5. Return ONLY the JSON object defined in the system prompt."
        )
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!synthesis_prompt:", synthesis_prompt)

        messages = [
            {"role": "system", "content": DEFERRED_DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ]
        response = ctx.llm.chat(messages)

        # Parse synthesis response
        try:
            content = response.content.strip()
            # Handle markdown-wrapped JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! getting result data", data)

            # Map the new schema to EvidenceItem
            # Store evidence_sources and metadata in extracted_fields
            extracted_fields = {
                "confidence_score": data.get("confidence_score"),
                "resolution_status": data.get("resolution_status"),
                "evidence_sources": data.get("evidence_sources", []),
            }

            # Determine success based on resolution_status
            resolution_status = data.get("resolution_status", "UNRESOLVED")
            success = resolution_status in ("RESOLVED", "AMBIGUOUS")

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="discover",
                    source_uri="serper:search",
                    tier=2,
                    fetched_at=ctx.now(),
                    content_hash=self._hash_content(response.content),
                ),
                raw_content=str(data.get("reasoning_trace", ""))[:500],
                parsed_value=data.get("parsed_value"),
                extracted_fields=extracted_fields,
                success=success,
                error=None if success else f"Resolution status: {resolution_status}",
            )

        except json.JSONDecodeError:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="discover",
                    source_uri="serper:search",
                    tier=1,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"LLM failed to produce valid JSON. Raw output: {response.content[:100]}...",
            )

    def _execute_search_queries(
        self,
        ctx: "AgentContext",
        queries: list[str],
    ) -> list[dict[str, str]]:
        """Execute search queries via Serper API and return combined results."""
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if ctx.config and hasattr(ctx.config, "serper") and ctx.config.serper:
            api_key = (ctx.config.serper.api_key or "").strip() or api_key

        if not api_key or not ctx.http:
            ctx.warning("Serper API not configured for deferred source discovery")
            return []

        all_results: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for query in queries:
            try:
                response = ctx.http.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query},
                    timeout=30.0,
                )
                if not response.ok:
                    ctx.warning(
                        f"Serper search failed for '{query}': HTTP {response.status_code}"
                    )
                    continue

                data = response.json()
                for item in data.get("organic", [])[:5]:
                    url = item.get("link", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": url,
                        })
            except Exception as e:
                ctx.warning(f"Serper search error for '{query}': {e}")

        return all_results

    def _build_evidence_item(
        self,
        ctx: "AgentContext",
        target: "SourceTarget",
        requirement_id: str,
        parsed: dict[str, Any],
        raw_output: str,
    ) -> EvidenceItem:
        """Build an EvidenceItem from parsed LLM response."""
        evidence_id = self._generate_evidence_id(requirement_id, target)
        content_summary = parsed.get("content_summary", "")
        extracted = self._truncate_extracted_fields(parsed.get("extracted_fields", {}))
        success = parsed.get("success", True)

        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=requirement_id,
            provenance=Provenance(
                source_id=target.source_id,
                source_uri=target.uri,
                tier=2,
                fetched_at=ctx.now(),
                content_hash=self._hash_content(raw_output),
            ),
            raw_content=str(content_summary)[:100],
            parsed_value=parsed.get("parsed_value"),
            extracted_fields=extracted,
            success=success,
            error=parsed.get("error"),
        )

    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        """Validate collection results."""
        checks: list[CheckResult] = []

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

        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)

    @staticmethod
    def _generate_evidence_id(requirement_id: str, target: "SourceTarget") -> str:
        """Generate a deterministic evidence ID."""
        key = f"{requirement_id}:{target.source_id}:{target.uri}:{target.operation}:{target.search_query}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_content(content: str) -> str:
        """Hash content using SHA256."""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try markdown code block first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        # Fall back to raw {…} extraction
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return json.loads(text.strip())

    @staticmethod
    def _truncate_extracted_fields(
        fields: dict[str, Any], max_value_len: int = 500
    ) -> dict[str, Any]:
        """Truncate extracted field values to prevent oversized evidence."""
        truncated = {}
        for key, value in fields.items():
            if isinstance(value, str) and len(value) > max_value_len:
                truncated[key] = value[:max_value_len] + "..."
            else:
                truncated[key] = value
        return truncated


# =============================================================================
# HyDE (Hypothetical Document Embeddings) Collector
# =============================================================================

HYDE_HYPOTHESIS_SYSTEM_PROMPT = """You are a Hypothetical Evidence Generator for a prediction market resolution system.

Your task is to generate a **hypothetical ideal document** that would perfectly answer the given data requirement. This hypothetical document will be used to search for real sources.

### INSTRUCTIONS
1. Imagine you are writing the IDEAL news article, press release, or official statement that would definitively answer the question.
2. Include specific details: dates, numbers, official names, and authoritative language.
3. Write as if you are a journalist or official spokesperson reporting the verified facts.
4. The document should be 2-4 paragraphs, realistic and authoritative.

### OUTPUT FORMAT
Return a single valid JSON object:
{
    "hypothetical_document": "The full text of the ideal document that would answer this question...",
    "key_entities": ["entity1", "entity2"],
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "search_queries": ["optimized search query 1", "optimized search query 2", "optimized search query 3"]
}

The search_queries should be derived from the hypothetical document to find real sources with similar content.
"""

HYDE_SYNTHESIS_SYSTEM_PROMPT = """You are an Evidence Synthesizer for a prediction market resolution system.

You were given a HYPOTHETICAL document describing ideal evidence, and now you have REAL search results.
Your task is to compare the real sources against the hypothetical expectation and extract verified facts.

### CORE PROTOCOLS
1. **Reality Check**: The hypothetical document is NOT real. Use it only as a template for what to look for.
2. **Source Verification**: Only extract facts that appear in the REAL search results.
3. **Confidence Calibration**:
   - If real sources MATCH the hypothesis → high confidence
   - If real sources CONTRADICT the hypothesis → report the contradiction
   - If real sources are SILENT → report as unverified
4. **Temporal Awareness**: Prefer the most recent sources. Check dates carefully.

### OUTPUT FORMAT
Return a single valid JSON object (no markdown):
{
    "hypothesis_match": "CONFIRMED" | "CONTRADICTED" | "PARTIAL" | "UNVERIFIED",
    "reasoning_trace": "Step-by-step analysis of how real sources compare to the hypothesis...",
    "parsed_value": "The specific answer extracted from REAL sources (not the hypothesis)",
    "confidence_score": 0.0 to 1.0,
    "evidence_sources": [
        {
            "source_id": "[X]",
            "url": "...",
            "credibility_tier": "Tier 1/Tier 2/Tier 3",
            "supports_hypothesis": true/false,
            "key_fact": "The specific fact extracted from this source"
        }
    ],
    "discrepancies": ["Any differences between hypothesis and reality"]
}
"""


class CollectorHyDE(BaseAgent):
    """
    HyDE (Hypothetical Document Embeddings) based Collector Agent.

    Uses the HyDE technique for improved evidence retrieval:
    1. Generate a hypothetical ideal document that would answer the requirement
    2. Use the hypothesis to generate semantically-rich search queries
    3. Search for real documents that match the hypothesis
    4. Synthesize real evidence by comparing against the hypothesis

    This approach often finds more relevant sources than direct query search
    because the hypothetical document captures the semantic intent better.

    Features:
    - Hypothesis-guided search for better semantic matching
    - Automatic comparison between expected and actual evidence
    - Confidence calibration based on hypothesis match
    - Works well for complex or nuanced requirements
    """

    _name = "CollectorHyDE"
    _version = "v1"
    _capabilities = {AgentCapability.LLM}
    MAX_RETRIES = 2

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """
        Execute collection using HyDE pattern.

        Args:
            ctx: Agent context with LLM client
            prompt_spec: The prompt specification
            tool_plan: The tool plan to execute

        Returns:
            AgentResult with (EvidenceBundle, ToolExecutionLog) as output
        """
        ctx.info(f"CollectorHyDE executing plan {tool_plan.plan_id}")

        if ctx.llm is None:
            return AgentResult.failure(error="LLM client not available")

        bundle = EvidenceBundle(
            bundle_id=f"hyde_{tool_plan.plan_id}",
            market_id=prompt_spec.market_id,
            plan_id=tool_plan.plan_id,
        )

        execution_log = ToolExecutionLog(
            plan_id=tool_plan.plan_id,
            started_at=ctx.now().isoformat(),
        )

        for req_id in tool_plan.requirements:
            requirement = prompt_spec.get_requirement_by_id(req_id)
            if not requirement:
                continue

            ctx.info(f"HyDE processing requirement: {req_id}")

            # Use HyDE for all requirements (both deferred and explicit)
            evidence = self._hyde_collect(ctx, requirement, prompt_spec)

            call_record = ToolCallRecord(
                tool="hyde:collect",
                input={
                    "requirement_id": req_id,
                    "description": requirement.description,
                },
                started_at=ctx.now().isoformat(),
            )
            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {
                "success": evidence.success,
                "evidence_id": evidence.evidence_id,
                "hypothesis_match": evidence.extracted_fields.get("hypothesis_match"),
            }
            if evidence.error:
                call_record.error = evidence.error

            execution_log.add_call(call_record)
            bundle.add_item(evidence)

        bundle.collected_at = ctx.now()
        bundle.requirements_fulfilled = [
            req_id for req_id in tool_plan.requirements
            if any(e.requirement_id == req_id and e.success for e in bundle.items)
        ]
        bundle.requirements_unfulfilled = [
            req_id for req_id in tool_plan.requirements
            if req_id not in bundle.requirements_fulfilled
        ]
        execution_log.ended_at = ctx.now().isoformat()

        return AgentResult(
            output=(bundle, execution_log),
            verification=self._validate_output(bundle, tool_plan),
            receipts=ctx.get_receipt_refs(),
            metadata={
                "collector": "hyde",
                "bundle_id": bundle.bundle_id,
                "items_collected": len(bundle.items),
            },
        )

    def _hyde_collect(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
    ) -> EvidenceItem:
        """
        Collect evidence using HyDE pattern.

        Flow:
        1. Generate hypothetical document that would answer the requirement
        2. Extract search queries from the hypothesis
        3. Search for real sources
        4. Synthesize by comparing real sources to hypothesis
        """
        semantics = prompt_spec.prediction_semantics
        evidence_id = hashlib.sha256(
            f"hyde:{requirement.requirement_id}".encode()
        ).hexdigest()[:16]

        event_def = prompt_spec.market.event_definition
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = "\n".join([f"- {a}" for a in assumptions_list]) if assumptions_list else "None"

        # Step 1: Generate hypothetical document
        ctx.info("HyDE Step 1: Generating hypothetical document...")
        hypothesis_prompt = (
            f"Generate a hypothetical ideal document for this requirement:\n\n"
            f"### REQUIREMENT\n"
            f"{requirement.description}\n\n"
            f"### CONTEXT\n"
            f"Market Question: {prompt_spec.market.question}\n"
            f"Event Definition: {event_def}\n"
            f"Assumptions:\n{assumptions_str}\n"
            f"Target Entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n"
            f"Threshold: {semantics.threshold or 'N/A'}\n\n"
            f"### TASK\n"
            f"Write a hypothetical news article or official statement that would "
            f"definitively answer whether '{semantics.target_entity}' '{semantics.predicate}'."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": HYDE_HYPOTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": hypothesis_prompt},
        ]

        ctx.debug(f"[Hypothesis] prompt {hypothesis_prompt}")
        response = ctx.llm.chat(messages)
        ctx.debug(f"[Hypothesis] result {response.content}")
        try:
            hypothesis_data = self._extract_json(response.content)
            hypothetical_doc = hypothesis_data.get("hypothetical_document", "")
            search_queries = hypothesis_data.get("search_queries", [])[:3]
            key_phrases = hypothesis_data.get("key_phrases", [])
        except (json.JSONDecodeError, KeyError) as e:
            ctx.warning(f"Failed to parse hypothesis: {e}")
            # Fallback: use requirement description as search query
            hypothetical_doc = requirement.description
            search_queries = [f"{semantics.target_entity} {semantics.predicate}"]
            key_phrases = []

        if not search_queries:
            search_queries = [f"{semantics.target_entity} {semantics.predicate}"]

        ctx.info(f"HyDE generated {len(search_queries)} search queries")
        ctx.debug(f"[Hypothesis] all search queries {search_queries}")

        # Step 2: Search for real sources using hypothesis-derived queries
        ctx.info("HyDE Step 2: Searching for real sources...")
        search_results = self._execute_search_queries(ctx, search_queries)

        if not search_results:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="hyde",
                    source_uri="serper:search",
                    tier=1,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="No search results found (Serper API may not be configured)",
                extracted_fields={
                    "hypothetical_document": hypothetical_doc[:500],
                    "hypothesis_match": "UNVERIFIED",
                },
            )

        # Step 3: Synthesize real evidence by comparing to hypothesis
        ctx.info("HyDE Step 3: Synthesizing evidence...")
        ctx.debug(f"[Hypothesis] search results {search_results}")
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Format search results
        formatted_results = []
        for i, r in enumerate(search_results):
            source_block = (
                f"SOURCE_ID: [{i+1}]\n"
                f"TITLE: {r.get('title', 'Unknown')}\n"
                f"URL: {r.get('url', 'Unknown')}\n"
                f"DATE_PUBLISHED: {r.get('date', 'Unknown')}\n"
                f"SNIPPET: {r.get('snippet', '')}"
            )
            formatted_results.append(source_block)

        results_text = "\n\n---\n\n".join(formatted_results)

        synthesis_prompt = (
            f"### HYPOTHETICAL DOCUMENT (for reference only - NOT real)\n"
            f"{hypothetical_doc[:1000]}\n\n"
            f"### KEY PHRASES TO LOOK FOR\n"
            f"{', '.join(key_phrases) if key_phrases else 'N/A'}\n\n"
            f"### CURRENT DATE/TIME\n"
            f"{current_time_str}\n\n"
            f"### MARKET QUESTION\n"
            f"{prompt_spec.market.question}\n\n"
            f"### REAL SEARCH RESULTS\n"
            f"{results_text}\n\n"
            f"### TASK\n"
            f"Compare the REAL search results above to the hypothetical document.\n"
            f"Extract ONLY verified facts from the real sources.\n"
            f"Report whether the hypothesis is CONFIRMED, CONTRADICTED, PARTIAL, or UNVERIFIED."
        )

        messages = [
            {"role": "system", "content": HYDE_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ]

        response = ctx.llm.chat(messages)

        # Parse synthesis response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)

            hypothesis_match = data.get("hypothesis_match", "UNVERIFIED")
            confidence_score = data.get("confidence_score", 0.5)

            # Determine success based on hypothesis match and confidence
            success = hypothesis_match in ("CONFIRMED", "PARTIAL") and confidence_score >= 0.5

            extracted_fields = {
                "hypothesis_match": hypothesis_match,
                "confidence_score": confidence_score,
                "evidence_sources": data.get("evidence_sources", []),
                "discrepancies": data.get("discrepancies", []),
                "hypothetical_document": hypothetical_doc[:300],
            }

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="hyde",
                    source_uri="serper:search",
                    tier=2,
                    fetched_at=ctx.now(),
                    content_hash=self._hash_content(response.content),
                ),
                raw_content=str(data.get("reasoning_trace", ""))[:500],
                parsed_value=data.get("parsed_value"),
                extracted_fields=extracted_fields,
                success=success,
                error=None if success else f"Hypothesis {hypothesis_match}, confidence {confidence_score:.2f}",
            )

        except json.JSONDecodeError:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="hyde",
                    source_uri="serper:search",
                    tier=1,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Failed to parse synthesis: {response.content[:100]}...",
                extracted_fields={
                    "hypothetical_document": hypothetical_doc[:300],
                    "hypothesis_match": "UNVERIFIED",
                },
            )

    def _execute_search_queries(
        self,
        ctx: "AgentContext",
        queries: list[str],
    ) -> list[dict[str, str]]:
        """Execute search queries via Serper API."""
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if ctx.config and hasattr(ctx.config, "serper") and ctx.config.serper:
            api_key = (ctx.config.serper.api_key or "").strip() or api_key

        if not api_key or not ctx.http:
            ctx.warning("Serper API not configured for HyDE search")
            return []

        all_results: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for query in queries:
            try:
                response = ctx.http.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query},
                    timeout=30.0,
                )
                if not response.ok:
                    ctx.warning(f"Serper search failed for '{query}': HTTP {response.status_code}")
                    continue

                data = response.json()
                for item in data.get("organic", [])[:5]:
                    url = item.get("link", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": url,
                            "date": item.get("date", "Unknown"),
                        })
            except Exception as e:
                ctx.warning(f"Serper search error for '{query}': {e}")

        return all_results

    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        """Validate collection results."""
        checks: list[CheckResult] = []

        if bundle.has_evidence:
            checks.append(CheckResult.passed(
                check_id="has_evidence",
                message=f"HyDE collected {len(bundle.items)} evidence items",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_evidence",
                message="No evidence was collected",
            ))

        # Check hypothesis match quality
        confirmed_count = sum(
            1 for e in bundle.items
            if e.extracted_fields.get("hypothesis_match") == "CONFIRMED"
        )
        if confirmed_count > 0:
            checks.append(CheckResult.passed(
                check_id="hypothesis_confirmed",
                message=f"{confirmed_count} items confirmed by real sources",
            ))

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

        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try markdown code block first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        # Fall back to raw {…} extraction
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return json.loads(text.strip())

    @staticmethod
    def _hash_content(content: str) -> str:
        """Hash content using SHA256."""
        return hashlib.sha256(content.encode()).hexdigest()


def get_collector(
    ctx: "AgentContext",
    *,
    prefer_hyde: bool = False,
    prefer_llm: bool = True,
    prefer_http: bool = True,
    mock_responses: dict[str, Any] | None = None,
) -> BaseAgent:
    """
    Get the appropriate collector based on context.

    Args:
        ctx: Agent context
        prefer_hyde: If True and LLM + HTTP available, use HyDE collector
        prefer_llm: If True and LLM client available, use LLM collector
        prefer_http: If True and HTTP client available, use HTTP collector
        mock_responses: Mock responses for mock collector

    Returns:
        CollectorHyDE, CollectorLLM, CollectorHTTP, or CollectorMock
    """
    if prefer_hyde and ctx.llm is not None and ctx.http is not None:
        return CollectorHyDE()
    if prefer_llm and ctx.llm is not None:
        return CollectorLLM()
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
        name="CollectorHyDE",
        factory=lambda ctx: CollectorHyDE(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=160,  # Highest - HyDE requires both LLM and HTTP
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorLLM",
        factory=lambda ctx: CollectorLLM(),
        capabilities={AgentCapability.LLM},
        priority=150,  # High - preferred when LLM available
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorHTTP",
        factory=lambda ctx: CollectorHTTP(),
        capabilities={AgentCapability.NETWORK},
        priority=100,  # Primary HTTP
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
