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
from .graphrag_engine import (
    GraphIndex,
    chunk_text_units,
    merge_elements_into_graph,
    detect_communities,
    build_local_context_pack,
    normalize_entity_name,
    rank_communities_by_query,
    infer_credibility_tier,
)

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
            "credibility_tier": 1 or 2 or 3 (integer: 1=authoritative, 2=reputable mainstream, 3=low confidence),
            "key_fact": "The specific fact extracted from this source",
            "supports": "YES" or "NO" or "N/A",
            "date_published": "YYYY-MM-DD" or null
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

            # Normalize evidence_sources to standard format
            raw_sources = data.get("evidence_sources", [])
            evidence_sources = _normalize_evidence_sources(raw_sources)

            extracted_fields = {
                "confidence_score": data.get("confidence_score"),
                "resolution_status": data.get("resolution_status"),
                "evidence_sources": evidence_sources,
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
            "credibility_tier": 1 or 2 or 3 (integer: 1=authoritative, 2=reputable mainstream, 3=low confidence),
            "key_fact": "The specific fact extracted from this source",
            "supports": "YES" or "NO" or "N/A",
            "date_published": "YYYY-MM-DD" or null
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
            evidence = self._hyde_collect(ctx, requirement, prompt_spec, execution_log)
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
        execution_log: ToolExecutionLog,
    ) -> EvidenceItem:
        """
        Collect evidence using HyDE pattern.

        Flow:
        1. Generate hypothetical document that would answer the requirement
        2. Extract search queries from the hypothesis
        3. Search for real sources
        4. Synthesize by comparing real sources to hypothesis
        """
        req_id = requirement.requirement_id
        semantics = prompt_spec.prediction_semantics
        evidence_id = hashlib.sha256(
            f"hyde:{req_id}".encode()
        ).hexdigest()[:16]

        event_def = prompt_spec.market.event_definition
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = "\n".join([f"- {a}" for a in assumptions_list]) if assumptions_list else "None"

        # ── Step 1: Generate hypothetical document ──────────────────
        ctx.info("HyDE Step 1: Generating hypothetical document...")
        hypothesis_record = ToolCallRecord(
            tool="hyde:hypothesis",
            input={"requirement_id": req_id},
            started_at=ctx.now().isoformat(),
        )

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

            hypothesis_record.ended_at = ctx.now().isoformat()
            hypothesis_record.output = {
                "num_queries": len(search_queries),
                "num_key_phrases": len(key_phrases),
                "doc_length": len(hypothetical_doc),
            }
        except (json.JSONDecodeError, KeyError) as e:
            ctx.warning(f"Failed to parse hypothesis: {e}")
            hypothetical_doc = requirement.description
            search_queries = [f"{semantics.target_entity} {semantics.predicate}"]
            key_phrases = []

            hypothesis_record.ended_at = ctx.now().isoformat()
            hypothesis_record.error = f"JSON parse failed, using fallback: {e}"
            hypothesis_record.output = {"num_queries": len(search_queries), "fallback": True}

        execution_log.add_call(hypothesis_record)

        if not search_queries:
            search_queries = [f"{semantics.target_entity} {semantics.predicate}"]

        ctx.info(f"HyDE generated {len(search_queries)} search queries")
        ctx.debug(f"[Hypothesis] all search queries {search_queries}")

        # ── Step 2: Search for real sources ─────────────────────────
        ctx.info("HyDE Step 2: Searching for real sources...")
        search_results = self._execute_search_queries(ctx, search_queries, execution_log)

        if not search_results:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
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

        # ── Step 3: Synthesize real evidence against hypothesis ─────
        ctx.info("HyDE Step 3: Synthesizing evidence...")
        ctx.debug(f"[Hypothesis] search results {search_results}")
        synthesis_record = ToolCallRecord(
            tool="hyde:synthesize",
            input={
                "requirement_id": req_id,
                "num_search_results": len(search_results),
            },
            started_at=ctx.now().isoformat(),
        )

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

            # Map hypothesis_match to standard resolution_status
            _MATCH_TO_STATUS = {
                "CONFIRMED": "RESOLVED",
                "PARTIAL": "AMBIGUOUS",
                "CONTRADICTED": "AMBIGUOUS",
                "UNVERIFIED": "UNRESOLVED",
            }
            resolution_status = _MATCH_TO_STATUS.get(hypothesis_match, "UNRESOLVED")

            # Normalize evidence_sources to standard format
            raw_sources = data.get("evidence_sources", [])
            evidence_sources = _normalize_evidence_sources(raw_sources)

            extracted_fields = {
                "confidence_score": confidence_score,
                "resolution_status": resolution_status,
                "evidence_sources": evidence_sources,
                "hypothesis_match": hypothesis_match,
                "discrepancies": data.get("discrepancies", []),
                "hypothetical_document": hypothetical_doc[:300],
            }

            synthesis_record.ended_at = ctx.now().isoformat()
            synthesis_record.output = {
                "hypothesis_match": hypothesis_match,
                "confidence_score": confidence_score,
                "resolution_status": resolution_status,
                "num_evidence_sources": len(evidence_sources),
            }
            execution_log.add_call(synthesis_record)

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
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
            synthesis_record.ended_at = ctx.now().isoformat()
            synthesis_record.error = "LLM returned invalid JSON for synthesis"
            execution_log.add_call(synthesis_record)

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
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
        execution_log: ToolExecutionLog,
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
            call_record = ToolCallRecord(
                tool="hyde:search",
                input={"query": query},
                started_at=ctx.now().isoformat(),
            )
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
                    call_record.ended_at = ctx.now().isoformat()
                    call_record.error = f"HTTP {response.status_code}"
                    execution_log.add_call(call_record)
                    continue

                data = response.json()
                results_found = 0
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
                        results_found += 1

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {"results_found": results_found}
                execution_log.add_call(call_record)

            except Exception as e:
                ctx.warning(f"Serper search error for '{query}': {e}")
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = str(e)
                execution_log.add_call(call_record)

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


# =============================================================================
# Agentic RAG System Prompts
# =============================================================================

AGENTIC_QUERY_PLANNER_SYSTEM_PROMPT = """
You are a Retrieval Strategist for a prediction market resolution oracle.
Given a data requirement, market question, and optionally previous gaps, produce a retrieval blueprint.

### RULES
1. Generate 2-4 diverse search queries (broad, specific, forensic).
2. Include domain hints for authoritative sources when applicable.
3. Include temporal hints when the requirement is time-sensitive.
4. If previous_gaps are provided, generate queries that specifically address those gaps.
5. Do NOT hallucinate facts — you are only planning queries, not answering the question.

### OUTPUT FORMAT
Return ONLY a single valid JSON object (no markdown, no commentary):
{
    "sub_questions": ["question the requirement decomposes into..."],
    "queries": ["search query 1", "search query 2", ...],
    "domain_hints": ["site:cdc.gov", ...],
    "recency_hint": "prefer last N days" or null,
    "must_have_terms": ["term1", ...],
    "avoid_terms": ["term1", ...]
}
"""

AGENTIC_RELEVANCE_ASSESSOR_SYSTEM_PROMPT = """
You are a Relevance & Credibility Assessor for a prediction market resolution oracle.
You receive a fetched web page fragment and must assess its relevance to a specific data requirement.

### RULES
1. Compare DATE_PUBLISHED (if present) against CURRENT_TIME. Outdated content is less relevant.
2. Assess credibility tier:
   - Tier 1: Official government (.gov), primary company releases, wire services (Reuters, AP, Bloomberg)
   - Tier 2: Reputable mainstream news (NYT, BBC, WSJ, major outlets)
   - Tier 3: Blogs, opinion pieces, social media, tabloids, undated sources
3. Determine if the content actually entails (proves/disproves) the requirement.
4. Extract the most relevant quote and any parseable value.
5. You MUST NOT hallucinate information not present in the fragment.
6. You MUST NOT invent URLs, dates, or facts.

### OUTPUT FORMAT
Return ONLY a single valid JSON object (no markdown, no commentary):
{
    "url": "the URL being assessed",
    "is_relevant": true or false,
    "entails_requirement": true or false,
    "credibility_tier": 1 or 2 or 3,
    "date_published": "YYYY-MM-DD" or null,
    "key_quote": "short excerpt from the text that supports assessment",
    "extracted_fields": {},
    "parsed_value": "extracted answer value or null",
    "reason": "brief explanation of why relevant or irrelevant",
    "confidence": 0.0 to 1.0
}
"""

AGENTIC_SYNTHESIS_SYSTEM_PROMPT = """
You are the Chief Evidence Synthesizer for a prediction market resolution oracle.
You receive a set of assessed evidence fragments and must fuse them into a single coherent resolution.

### RULES
1. **Source Hierarchy:** Tier 1 overrides Tier 2; Tier 2 overrides Tier 3.
   If same-tier sources conflict, the most recent one prevails.
2. **Temporal Awareness:** Compare dates against CURRENT_TIME. Recent evidence is preferred.
3. **Conflict Handling:** If sources conflict, document both sides. Keep opposing sources for transparency.
4. **Grounding:** You MUST NOT introduce facts not present in the assessed evidence.
   Every claim must cite a source URL from the input.
5. **Parsimony:** Provide the simplest conclusion supported by the strongest evidence.

### OUTPUT FORMAT
Return ONLY a single valid JSON object (no markdown, no commentary):
{
    "resolution_status": "RESOLVED" or "AMBIGUOUS" or "UNRESOLVED",
    "parsed_value": "the specific answer extracted or null",
    "confidence_score": 0.0 to 1.0,
    "evidence_sources": [
        {
            "url": "...",
            "credibility_tier": 1 or 2 or 3 (integer: 1=authoritative, 2=reputable mainstream, 3=low confidence),
            "key_fact": "...",
            "supports": "YES" or "NO" or "N/A",
            "date_published": "YYYY-MM-DD" or null
        }
    ],
    "conflicts": ["description of any conflicts between sources"],
    "missing_info": ["information still needed to fully resolve"],
    "reasoning_trace": "Step-by-step deduction: 1) ... 2) ... 3) ... Conclusion: ..."
}
"""


class CollectorAgenticRAG(BaseAgent):
    """
    Agentic RAG (Retrieval-Augmented Generation) Collector with deep reasoning.

    Implements iterative retrieval↔reasoning loops inspired by the "agentic RAG
    with deep reasoning" paradigm:

    1. **Query Planning** — LLM generates a retrieval blueprint with sub-questions,
       targeted queries, domain hints, and temporal constraints.
    2. **Retrieve Candidates** — Serper API search with global URL de-duplication.
    3. **Fetch & Extract** — HTTP fetch of top candidate pages, extract fragments.
    4. **Relevance Assessment** — LLM assesses each fragment for relevance,
       credibility tier, entailment, and confidence (integration enhancement).
    5. **Synthesis & Fusion** — LLM fuses filtered evidence into a coherent,
       conflict-aware result with structured provenance (integration enhancement).
    6. **Iterate** — If evidence is insufficient or conflicting, feed gaps back
       into query planning for another retrieval round.

    This produces higher-quality, more trustworthy evidence bundles than single-pass
    collectors by interleaving retrieval and reasoning.
    """

    _name = "CollectorAgenticRAG"
    _version = "v1"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}

    MAX_ITERS = 2
    TOP_K_FETCH = 5
    KEEP_K = 2
    MAX_RETRIES = 2
    MAX_FRAGMENT_BYTES = 15_000  # ~15 KB per page fragment

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """Execute agentic RAG collection with iterative retrieval↔reasoning."""
        if ctx.llm is None:
            return AgentResult.failure(error="LLM client not available")
        if ctx.http is None:
            return AgentResult.failure(error="HTTP client not available")

        ctx.info(f"CollectorAgenticRAG executing plan {tool_plan.plan_id}")

        bundle = EvidenceBundle(
            bundle_id=f"agentic_rag_{tool_plan.plan_id}",
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

            ctx.info(f"AgenticRAG processing requirement: {req_id}")
            evidence = self._collect_requirement(
                ctx, requirement, prompt_spec, execution_log,
            )
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
            success=bundle.has_evidence,
            error=None if bundle.has_evidence else "No evidence collected",
            metadata={
                "collector": "agentic_rag",
                "bundle_id": bundle.bundle_id,
                "items_collected": len(bundle.items),
                "requirements_fulfilled": bundle.requirements_fulfilled,
                "requirements_unfulfilled": bundle.requirements_unfulfilled,
            },
        )

    # ------------------------------------------------------------------
    # Core iterative loop for a single requirement
    # ------------------------------------------------------------------

    def _collect_requirement(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        execution_log: ToolExecutionLog,
    ) -> EvidenceItem:
        """Run the iterative agentic RAG loop for one requirement."""
        req_id = requirement.requirement_id
        market_id = prompt_spec.market_id
        semantics = prompt_spec.prediction_semantics
        event_def = prompt_spec.market.event_definition
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = (
            "\n".join([f"- {a}" for a in assumptions_list])
            if assumptions_list else "None"
        )
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        global_seen_urls: set[str] = set()
        previous_gaps: list[str] = []
        last_synthesis: dict[str, Any] | None = None

        for iteration in range(self.MAX_ITERS):
            ctx.info(f"AgenticRAG iteration {iteration + 1}/{self.MAX_ITERS} for {req_id}")
            evidence_id = hashlib.sha256(
                f"agentic:{req_id}:{iteration}:{market_id}".encode()
            ).hexdigest()[:16]

            # ── Step A: Query planning ──────────────────────────────
            plan = self._plan_queries(
                ctx, requirement, prompt_spec, semantics, event_def,
                assumptions_str, current_time_str, previous_gaps, execution_log,
            )
            queries = plan.get("queries", [])[:4]
            domain_hints = plan.get("domain_hints", [])

            # Append domain-hinted queries
            for hint in domain_hints[:2]:
                if queries:
                    queries.append(f"{queries[0]} {hint}")

            if not queries:
                queries = [f"{semantics.target_entity} {semantics.predicate}"]

            print(f"========================planning the queries: {queries}")
            # ── Step B: Retrieve candidates via Serper ───────────────
            search_results = self._execute_search_queries(
                ctx, queries, global_seen_urls, execution_log,
            )

            print(f"========================search results: {search_results}")
            if not search_results:
                if iteration == 0:
                    return EvidenceItem(
                        evidence_id=evidence_id,
                        requirement_id=req_id,
                        provenance=Provenance(
                            source_id="agentic_rag",
                            source_uri="serper:search",
                            tier=2,
                            fetched_at=ctx.now(),
                        ),
                        success=False,
                        error="No search results found (Serper API may not be configured)",
                    )
                # Later iterations: use whatever we have from prior iteration
                break

            # ── Step C: Fetch pages and extract fragments ────────────
            fragments = self._fetch_pages(
                ctx, search_results[:self.TOP_K_FETCH], execution_log,
            )

            if not fragments:
                previous_gaps.append("All fetched pages returned errors or empty content")
                continue

            # ── Step D: Relevance assessment & filtering ─────────────
            assessed = self._assess_relevance(
                ctx, fragments, requirement, prompt_spec,
                current_time_str, execution_log,
            )

            # Filter: keep relevant items with entailment or sufficient confidence
            kept = [
                a for a in assessed
                if a.get("is_relevant") and (
                    a.get("entails_requirement") or a.get("confidence", 0) >= 0.6
                )
            ]

            # Prefer higher tiers, then by confidence
            kept.sort(
                key=lambda a: (
                    -(a.get("credibility_tier", 3)),  # lower tier number = better
                    -(a.get("confidence", 0)),
                ),
            )

            # If conflicts exist, ensure we keep at least 2 opposing sources
            if len(kept) > self.KEEP_K:
                kept = kept[:self.KEEP_K]

            if not kept:
                previous_gaps.append("All fetched sources were irrelevant after assessment")
                continue

            # ── Step E: Synthesis & fusion ────────────────────────────
            synthesis = self._synthesize(
                ctx, kept, requirement, prompt_spec,
                current_time_str, execution_log,
            )
            last_synthesis = synthesis

            # ── Step F: Decide whether to iterate ────────────────────
            status = synthesis.get("resolution_status", "UNRESOLVED")
            confidence = synthesis.get("confidence_score", 0.0)
            missing = synthesis.get("missing_info", [])

            if status == "RESOLVED" and confidence >= 0.6:
                ctx.info(f"AgenticRAG resolved {req_id} with confidence {confidence}")
                break

            if status == "AMBIGUOUS" and confidence >= 0.5:
                ctx.info(f"AgenticRAG ambiguous {req_id} with confidence {confidence}")
                break

            # Feed gaps back for next iteration
            previous_gaps = list(missing) if missing else [
                f"Status {status} with low confidence {confidence}"
            ]

        # ── Build final EvidenceItem from last synthesis ──────────────
        if last_synthesis is None:
            evidence_id = hashlib.sha256(
                f"agentic:{req_id}:final:{market_id}".encode()
            ).hexdigest()[:16]
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="agentic_rag",
                    source_uri="serper:search+http:fetch",
                    tier=2,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="AgenticRAG failed to produce synthesis after all iterations",
            )

        return self._build_evidence_from_synthesis(
            ctx, req_id, market_id, last_synthesis,
        )

    # ------------------------------------------------------------------
    # Step A: Query planning
    # ------------------------------------------------------------------

    def _plan_queries(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        semantics: Any,
        event_def: str,
        assumptions_str: str,
        current_time_str: str,
        previous_gaps: list[str],
        execution_log: ToolExecutionLog,
    ) -> dict[str, Any]:
        """Generate a retrieval blueprint via LLM."""
        call_record = ToolCallRecord(
            tool="agentic_rag:plan",
            input={
                "requirement_id": requirement.requirement_id,
                "iteration_gaps": previous_gaps,
            },
            started_at=ctx.now().isoformat(),
        )

        gaps_text = ""
        if previous_gaps:
            gaps_text = (
                f"\n### PREVIOUS GAPS (address these specifically)\n"
                + "\n".join(f"- {g}" for g in previous_gaps)
            )

        prompt = (
            f"Generate a retrieval blueprint for this data requirement:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Market question: {prompt_spec.market.question}\n"
            f"- Event Definition: {event_def}\n"
            f"- Key Assumptions:\n{assumptions_str}\n"
            f"- Entity: {semantics.target_entity}\n"
            f"- Predicate: {semantics.predicate}\n"
            f"- Threshold: {semantics.threshold or 'N/A'}\n"
            f"- Current UTC time: {current_time_str}\n"
            f"{gaps_text}"
        )

        messages = [
            {"role": "system", "content": AGENTIC_QUERY_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = ctx.llm.chat(messages)

        try:
            plan = self._extract_json(response.content)
        except (json.JSONDecodeError, ValueError):
            plan = {
                "queries": [f"{semantics.target_entity} {semantics.predicate}"],
                "domain_hints": [],
            }

        call_record.ended_at = ctx.now().isoformat()
        call_record.output = {"queries": plan.get("queries", [])[:4]}
        execution_log.add_call(call_record)
        return plan

    # ------------------------------------------------------------------
    # Step B: Search via Serper
    # ------------------------------------------------------------------

    def _execute_search_queries(
        self,
        ctx: "AgentContext",
        queries: list[str],
        global_seen_urls: set[str],
        execution_log: ToolExecutionLog,
    ) -> list[dict[str, str]]:
        """Execute search queries via Serper API with global de-dupe."""
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if ctx.config and hasattr(ctx.config, "serper") and ctx.config.serper:
            api_key = (ctx.config.serper.api_key or "").strip() or api_key

        if not api_key or not ctx.http:
            ctx.warning("Serper API not configured for agentic RAG")
            return []

        all_results: list[dict[str, str]] = []

        for query in queries:
            call_record = ToolCallRecord(
                tool="agentic_rag:search",
                input={"query": query},
                started_at=ctx.now().isoformat(),
            )
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
                    call_record.ended_at = ctx.now().isoformat()
                    call_record.error = f"HTTP {response.status_code}"
                    execution_log.add_call(call_record)
                    continue

                data = response.json()
                results_found = 0
                for item in data.get("organic", [])[:7]:
                    url = item.get("link", "")
                    if url and url not in global_seen_urls:
                        global_seen_urls.add(url)
                        all_results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": url,
                            "date": item.get("date", ""),
                        })
                        results_found += 1

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {"results_found": results_found}
                execution_log.add_call(call_record)

            except Exception as e:
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = str(e)
                execution_log.add_call(call_record)

        return all_results

    # ------------------------------------------------------------------
    # Step C: Fetch pages
    # ------------------------------------------------------------------

    def _fetch_pages(
        self,
        ctx: "AgentContext",
        candidates: list[dict[str, str]],
        execution_log: ToolExecutionLog,
    ) -> list[dict[str, Any]]:
        """Fetch top candidate URLs and extract compact fragments."""
        fragments: list[dict[str, Any]] = []

        for candidate in candidates:
            url = candidate["url"]
            call_record = ToolCallRecord(
                tool="agentic_rag:fetch",
                input={"url": url},
                started_at=ctx.now().isoformat(),
            )
            try:
                response = ctx.http.get(url, timeout=20.0)
                status_code = response.status_code

                if not response.ok:
                    call_record.ended_at = ctx.now().isoformat()
                    call_record.error = f"HTTP {status_code}"
                    call_record.output = {"status_code": status_code}
                    execution_log.add_call(call_record)
                    continue

                raw_text = response.text[:self.MAX_FRAGMENT_BYTES]
                content_hash = self._hash_content(raw_text)

                fragments.append({
                    "url": url,
                    "title": candidate.get("title", ""),
                    "snippet": candidate.get("snippet", ""),
                    "date": candidate.get("date", ""),
                    "fragment": raw_text,
                    "content_hash": content_hash,
                    "status_code": status_code,
                })

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "status_code": status_code,
                    "content_hash": content_hash,
                    "fragment_length": len(raw_text),
                }
                execution_log.add_call(call_record)

            except Exception as e:
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = str(e)
                execution_log.add_call(call_record)

        return fragments

    # ------------------------------------------------------------------
    # Step D: Relevance assessment
    # ------------------------------------------------------------------

    def _assess_relevance(
        self,
        ctx: "AgentContext",
        fragments: list[dict[str, Any]],
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        current_time_str: str,
        execution_log: ToolExecutionLog,
    ) -> list[dict[str, Any]]:
        """Assess each fragment for relevance and credibility via LLM."""
        assessments: list[dict[str, Any]] = []

        for frag in fragments:
            call_record = ToolCallRecord(
                tool="agentic_rag:assess",
                input={"url": frag["url"]},
                started_at=ctx.now().isoformat(),
            )

            # Truncate fragment for the LLM prompt
            truncated_fragment = frag["fragment"][:8000]

            prompt = (
                f"### CONTEXT\n"
                f"CURRENT_TIME: {current_time_str}\n"
                f"Market Question: {prompt_spec.market.question}\n"
                f"Data Requirement: {requirement.description}\n\n"
                f"### WEB PAGE FRAGMENT\n"
                f"URL: {frag['url']}\n"
                f"TITLE: {frag.get('title', 'Unknown')}\n"
                f"DATE_FROM_SEARCH: {frag.get('date', 'Unknown')}\n"
                f"SNIPPET: {frag.get('snippet', '')}\n\n"
                f"CONTENT:\n{truncated_fragment}\n\n"
                f"### TASK\n"
                f"Assess this page's relevance and credibility for the data requirement above."
            )

            messages = [
                {"role": "system", "content": AGENTIC_RELEVANCE_ASSESSOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                response = ctx.llm.chat(messages)
                assessment = self._extract_json(response.content)
                assessment["url"] = frag["url"]  # ensure URL is correct
                assessment["content_hash"] = frag.get("content_hash", "")
                assessments.append(assessment)

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "is_relevant": assessment.get("is_relevant"),
                    "credibility_tier": assessment.get("credibility_tier"),
                    "confidence": assessment.get("confidence"),
                }
                execution_log.add_call(call_record)

            except (json.JSONDecodeError, ValueError):
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = "LLM returned invalid JSON for assessment"
                execution_log.add_call(call_record)

        return assessments

    # ------------------------------------------------------------------
    # Step E: Synthesis & fusion
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        ctx: "AgentContext",
        assessed: list[dict[str, Any]],
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        current_time_str: str,
        execution_log: ToolExecutionLog,
    ) -> dict[str, Any]:
        """Fuse assessed evidence into a coherent resolution via LLM."""
        call_record = ToolCallRecord(
            tool="agentic_rag:synthesize",
            input={
                "requirement_id": requirement.requirement_id,
                "num_sources": len(assessed),
            },
            started_at=ctx.now().isoformat(),
        )

        # Format assessed sources for the LLM
        formatted = []
        for i, a in enumerate(assessed):
            block = (
                f"SOURCE [{i + 1}]\n"
                f"URL: {a.get('url', 'Unknown')}\n"
                f"CREDIBILITY_TIER: {a.get('credibility_tier', 3)}\n"
                f"DATE_PUBLISHED: {a.get('date_published', 'Unknown')}\n"
                f"CONFIDENCE: {a.get('confidence', 0)}\n"
                f"KEY_QUOTE: {str(a.get('key_quote', ''))[:300]}\n"
                f"PARSED_VALUE: {a.get('parsed_value', 'null')}\n"
                f"REASON: {str(a.get('reason', ''))[:200]}"
            )
            formatted.append(block)

        sources_text = "\n\n---\n\n".join(formatted)

        semantics = prompt_spec.prediction_semantics
        prompt = (
            f"### CONTEXT\n"
            f"CURRENT_TIME: {current_time_str}\n"
            f"Market Question: {prompt_spec.market.question}\n"
            f"Event Definition: {prompt_spec.market.event_definition}\n"
            f"Entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n\n"
            f"### DATA REQUIREMENT\n"
            f"{requirement.description}\n\n"
            f"### ASSESSED EVIDENCE SOURCES\n"
            f"{sources_text}\n\n"
            f"### TASK\n"
            f"Synthesize the above sources into a single coherent resolution.\n"
            f"Apply the source hierarchy: Tier 1 > Tier 2 > Tier 3.\n"
            f"If sources conflict, document both sides."
        )

        messages = [
            {"role": "system", "content": AGENTIC_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = ctx.llm.chat(messages)
            synthesis = self._extract_json(response.content)

            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {
                "resolution_status": synthesis.get("resolution_status"),
                "confidence_score": synthesis.get("confidence_score"),
                "num_evidence_sources": len(synthesis.get("evidence_sources", [])),
            }
            execution_log.add_call(call_record)
            return synthesis

        except (json.JSONDecodeError, ValueError):
            call_record.ended_at = ctx.now().isoformat()
            call_record.error = "LLM returned invalid JSON for synthesis"
            execution_log.add_call(call_record)
            return {
                "resolution_status": "UNRESOLVED",
                "confidence_score": 0.0,
                "evidence_sources": [],
                "conflicts": [],
                "missing_info": ["Synthesis failed — LLM returned invalid JSON"],
                "reasoning_trace": "Synthesis parse error",
            }

    # ------------------------------------------------------------------
    # Build final EvidenceItem from synthesis
    # ------------------------------------------------------------------

    def _build_evidence_from_synthesis(
        self,
        ctx: "AgentContext",
        req_id: str,
        market_id: str,
        synthesis: dict[str, Any],
    ) -> EvidenceItem:
        """Convert synthesis output to a properly structured EvidenceItem."""
        evidence_id = hashlib.sha256(
            f"agentic:{req_id}:final:{market_id}".encode()
        ).hexdigest()[:16]

        resolution_status = synthesis.get("resolution_status", "UNRESOLVED")
        confidence_score = synthesis.get("confidence_score", 0.0)
        success = (
            resolution_status in ("RESOLVED", "AMBIGUOUS")
            and confidence_score >= 0.5
        )

        # Normalize evidence_sources to standard format
        raw_sources = synthesis.get("evidence_sources", synthesis.get("final_evidence", []))
        evidence_sources = _normalize_evidence_sources(raw_sources)

        extracted_fields = self._truncate_extracted_fields({
            "confidence_score": confidence_score,
            "resolution_status": resolution_status,
            "evidence_sources": evidence_sources,
            "conflicts": synthesis.get("conflicts", []),
            "missing_info": synthesis.get("missing_info", []),
        })

        reasoning_trace = str(synthesis.get("reasoning_trace", ""))[:500]

        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=req_id,
            provenance=Provenance(
                source_id="agentic_rag",
                source_uri="serper:search+http:fetch",
                tier=2,
                fetched_at=ctx.now(),
                content_hash=self._hash_content(reasoning_trace),
            ),
            raw_content=reasoning_trace,
            parsed_value=synthesis.get("parsed_value"),
            extracted_fields=extracted_fields,
            success=success,
            error=None if success else f"Resolution status: {resolution_status}",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        """Validate agentic RAG collection results."""
        checks: list[CheckResult] = []

        if bundle.has_evidence:
            checks.append(CheckResult.passed(
                check_id="has_evidence",
                message=f"AgenticRAG collected {len(bundle.items)} evidence items",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_evidence",
                message="No evidence was collected",
            ))

        # Check synthesis quality
        resolved_count = sum(
            1 for e in bundle.items
            if e.extracted_fields.get("resolution_status") == "RESOLVED"
        )
        if resolved_count > 0:
            checks.append(CheckResult.passed(
                check_id="synthesis_resolved",
                message=f"{resolved_count} requirements resolved via synthesis",
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

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return json.loads(text.strip())

    @staticmethod
    def _hash_content(content: str) -> str:
        """Hash content using SHA256."""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _truncate_extracted_fields(
        fields: dict[str, Any], max_value_len: int = 500
    ) -> dict[str, Any]:
        """Truncate extracted field values to prevent oversized evidence."""
        truncated: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, str) and len(value) > max_value_len:
                truncated[key] = value[:max_value_len] + "..."
            else:
                truncated[key] = value
        return truncated


# =============================================================================
# GraphRAG Collector — Local-to-Global approach
# =============================================================================

GRAPHRAG_ELEMENT_EXTRACTION_SYSTEM_PROMPT = """You extract structured graph elements from text for a retrieval+summarization oracle.
Return ONLY valid JSON.

OUTPUT SCHEMA:
{
  "entities": [{"name": "...", "type": "PERSON|ORG|EVENT|METRIC|LOCATION|CONCEPT", "description": "one-line description"}],
  "relations": [{"head": "...", "relation": "...", "tail": "...", "quote": "supporting excerpt"}],
  "claims": [{"claim": "short canonical statement", "quote": "supporting excerpt", "supports": "YES|NO|N/A"}],
  "date_published": "YYYY-MM-DD or null"
}

RULES:
- Extract ALL named entities, organisations, dates, metrics, and events.
- Relations must reference entity names that appear in the entities list.
- Claims are key factual assertions — canonicalize them into short statements.
- Quotes must come verbatim from the provided text (≤150 chars each).
- Do NOT invent facts not present in the text.
"""

GRAPHRAG_COMMUNITY_REPORT_SYSTEM_PROMPT = """You write a compact community report grounded in provided entities/relations/quotes only.
Return ONLY valid JSON.

OUTPUT SCHEMA:
{
  "community_title": "short descriptive title",
  "summary": "2-4 sentence summary of the community's key information",
  "key_entities": ["entity_name_1", "entity_name_2"],
  "key_facts": ["fact 1", "fact 2"],
  "citations": [{"url": "...", "quote": "..."}]
}

RULES:
- Ground every claim in the provided entities, relations, and quotes.
- Keep summary ≤ 1200 characters.
- Do NOT introduce facts not present in the input.
"""

GRAPHRAG_COMMUNITY_MAP_SYSTEM_PROMPT = """You produce a partial, query-focused answer from ONE community report.
Return ONLY valid JSON. Do not invent facts.

OUTPUT SCHEMA:
{
  "community_id": "c0",
  "supports": "YES|NO|N/A",
  "parsed_value": "extracted value or null",
  "confidence": 0.0 to 1.0,
  "key_fact": "the single most important fact from this community",
  "evidence_sources": [
    {
      "url": "...",
      "credibility_tier": 1 or 2 or 3,
      "key_fact": "...",
      "supports": "YES|NO|N/A",
      "date_published": "YYYY-MM-DD or null"
    }
  ],
  "missing_info": ["information still needed"]
}
"""

GRAPHRAG_REDUCE_SYSTEM_PROMPT = """You are the Chief Evidence Synthesizer for a prediction market resolution oracle.
You receive partial answers from community-level MAP steps and a local context pack of entity-level evidence.
Fuse them into a single coherent resolution.

### RULES
1. **Source Hierarchy:** Tier 1 > Tier 2 > Tier 3. Same-tier conflict → most recent wins.
2. **Temporal Awareness:** Compare dates against CURRENT_TIME. Recent evidence preferred.
3. **Conflict Handling:** Document both sides. Keep opposing sources for transparency.
4. **Grounding:** Do NOT introduce facts not present in the input. Every claim must cite a URL.
5. **Parsimony:** Simplest conclusion supported by strongest evidence.

### OUTPUT FORMAT
Return ONLY a single valid JSON object (no markdown, no commentary):
{
  "resolution_status": "RESOLVED" or "AMBIGUOUS" or "UNRESOLVED",
  "parsed_value": "the specific answer or null",
  "confidence_score": 0.0 to 1.0,
  "evidence_sources": [
    {
      "url": "...",
      "credibility_tier": 1 or 2 or 3,
      "key_fact": "...",
      "supports": "YES" or "NO" or "N/A",
      "date_published": "YYYY-MM-DD" or null
    }
  ],
  "conflicts": ["description of conflicts"],
  "missing_info": ["information still needed"],
  "reasoning_trace": "Step-by-step deduction: 1) ... 2) ... Conclusion: ..."
}
"""

GRAPHRAG_SEED_ENTITIES_SYSTEM_PROMPT = """Given a query about a prediction market, identify 5-10 seed entity names or must-have terms that are central to answering the query. Return ONLY valid JSON.

OUTPUT SCHEMA:
{
  "seed_entities": ["entity_or_term_1", "entity_or_term_2", ...]
}

RULES:
- Include proper nouns (people, organizations, places) relevant to the query.
- Include key metric names, dates, and event names.
- Keep terms concise — single words or short phrases.
"""


class CollectorGraphRAG(BaseAgent):
    """
    GraphRAG Collector — Local-to-Global query-focused summarization.

    Implements the GraphRAG approach:
    1. **Retrieve** — Search + fetch documents (reuses AgenticRAG patterns).
    2. **Index** — Chunk documents into text units, extract entities/relations
       via LLM, build an in-memory graph, detect communities, and generate
       community-level summary reports.
    3. **Query (Local)** — Identify seed entities, expand neighborhoods,
       build a compact local context pack.
    4. **Query (Global)** — Rank communities by relevance, MAP each to a
       partial answer, REDUCE into a final synthesis.
    5. **Emit** — Produce a standard EvidenceItem with graph_stats and
       community metadata.
    """

    _name = "CollectorGraphRAG"
    _version = "v1"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}

    MAX_RETRIES = 2
    TOP_K_FETCH = 5
    MAX_FRAGMENT_BYTES = 15_000
    TOP_M_COMMUNITIES = 3

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """Execute GraphRAG collection with graph-based indexing and query."""
        if ctx.llm is None:
            return AgentResult.failure(error="LLM client not available")
        if ctx.http is None:
            return AgentResult.failure(error="HTTP client not available")

        ctx.info(f"CollectorGraphRAG executing plan {tool_plan.plan_id}")

        bundle = EvidenceBundle(
            bundle_id=f"graphrag_{tool_plan.plan_id}",
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

            ctx.info(f"GraphRAG processing requirement: {req_id}")
            evidence = self._collect_requirement(
                ctx, requirement, prompt_spec, execution_log,
            )
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
            success=bundle.has_evidence,
            error=None if bundle.has_evidence else "No evidence collected",
            metadata={
                "collector": "graphrag",
                "bundle_id": bundle.bundle_id,
                "items_collected": len(bundle.items),
                "requirements_fulfilled": bundle.requirements_fulfilled,
                "requirements_unfulfilled": bundle.requirements_unfulfilled,
            },
        )

    # ------------------------------------------------------------------
    # Core per-requirement pipeline
    # ------------------------------------------------------------------

    def _collect_requirement(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        execution_log: ToolExecutionLog,
    ) -> EvidenceItem:
        """Run the full GraphRAG pipeline for one requirement."""
        req_id = requirement.requirement_id
        market_id = prompt_spec.market_id
        semantics = prompt_spec.prediction_semantics
        event_def = prompt_spec.market.event_definition
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = (
            "\n".join([f"- {a}" for a in assumptions_list])
            if assumptions_list else "None"
        )
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        evidence_id = hashlib.sha256(
            f"graphrag:{req_id}:{market_id}".encode()
        ).hexdigest()[:16]

        # ── A) Retrieval: plan queries + search + fetch ─────────────
        plan = self._plan_queries(
            ctx, requirement, prompt_spec, semantics, event_def,
            assumptions_str, current_time_str, execution_log,
        )
        queries = plan.get("queries", [])[:4]
        domain_hints = plan.get("domain_hints", [])
        for hint in domain_hints[:2]:
            if queries:
                queries.append(f"{queries[0]} {hint}")
        if not queries:
            queries = [f"{semantics.target_entity} {semantics.predicate}"]

        global_seen_urls: set[str] = set()
        search_results = self._execute_search_queries(
            ctx, queries, global_seen_urls, execution_log,
        )

        if not search_results:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="graphrag",
                    source_uri="serper:search",
                    tier=2,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="No search results found (Serper API may not be configured)",
            )

        fragments = self._fetch_pages(
            ctx, search_results[:self.TOP_K_FETCH], execution_log,
        )

        if not fragments:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="graphrag",
                    source_uri="serper:search+http:fetch",
                    tier=2,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="All fetched pages returned errors or empty content",
            )

        # ── B) GraphRAG index build ─────────────────────────────────
        graph = GraphIndex()

        # Chunk each document into text units
        for frag in fragments:
            units = chunk_text_units(
                frag["fragment"],
                frag["url"],
                frag.get("title", ""),
            )
            graph.text_units.extend(units)

        # Extract graph elements from each text unit via LLM
        for tu in graph.text_units:
            elements = self._extract_graph_elements(
                ctx, tu, requirement, prompt_spec, execution_log,
            )
            if elements:
                merge_elements_into_graph(graph, elements, tu.doc_url, tu.id)

        # Community detection
        comm_record = ToolCallRecord(
            tool="graphrag:communities",
            input={"num_entities": len(graph.entities)},
            started_at=ctx.now().isoformat(),
        )
        communities = detect_communities(graph, min_size=2)
        comm_record.ended_at = ctx.now().isoformat()
        comm_record.output = {"num_communities": len(communities)}
        execution_log.add_call(comm_record)

        # Generate community reports
        self._generate_community_reports(
            ctx, graph, requirement, prompt_spec, execution_log,
        )

        ctx.info(
            f"GraphRAG index built: {graph.stats}"
        )

        # ── C) Query: local + global ────────────────────────────────
        query_text = (
            f"{requirement.description} "
            f"{prompt_spec.market.question} "
            f"{event_def}"
        )

        # Local: seed entities → neighborhood pack
        local_pack = self._query_local(
            ctx, query_text, graph, execution_log,
        )

        # Global: community MAP → REDUCE
        synthesis = self._query_global_map_reduce(
            ctx, query_text, graph, local_pack,
            requirement, prompt_spec, current_time_str, execution_log,
        )

        # ── D) Build EvidenceItem ───────────────────────────────────
        return self._build_evidence_from_synthesis(
            ctx, req_id, market_id, synthesis, graph,
        )

    # ------------------------------------------------------------------
    # A-1) Query planning (reuses agentic pattern)
    # ------------------------------------------------------------------

    def _plan_queries(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        semantics: Any,
        event_def: str,
        assumptions_str: str,
        current_time_str: str,
        execution_log: ToolExecutionLog,
    ) -> dict[str, Any]:
        """Generate retrieval blueprint via LLM."""
        call_record = ToolCallRecord(
            tool="graphrag:plan",
            input={"requirement_id": requirement.requirement_id},
            started_at=ctx.now().isoformat(),
        )

        prompt = (
            f"Generate a retrieval blueprint for this data requirement:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Market question: {prompt_spec.market.question}\n"
            f"- Event Definition: {event_def}\n"
            f"- Key Assumptions:\n{assumptions_str}\n"
            f"- Entity: {semantics.target_entity}\n"
            f"- Predicate: {semantics.predicate}\n"
            f"- Threshold: {semantics.threshold or 'N/A'}\n"
            f"- Current UTC time: {current_time_str}\n"
        )

        messages = [
            {"role": "system", "content": AGENTIC_QUERY_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = ctx.llm.chat(messages)

        try:
            plan = self._extract_json(response.content)
        except (json.JSONDecodeError, ValueError):
            plan = {
                "queries": [f"{semantics.target_entity} {semantics.predicate}"],
                "domain_hints": [],
            }

        call_record.ended_at = ctx.now().isoformat()
        call_record.output = {"queries": plan.get("queries", [])[:4]}
        execution_log.add_call(call_record)
        return plan

    # ------------------------------------------------------------------
    # A-2) Serper search (with global de-dupe)
    # ------------------------------------------------------------------

    def _execute_search_queries(
        self,
        ctx: "AgentContext",
        queries: list[str],
        global_seen_urls: set[str],
        execution_log: ToolExecutionLog,
    ) -> list[dict[str, str]]:
        """Execute search queries via Serper API with global de-dupe."""
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if ctx.config and hasattr(ctx.config, "serper") and ctx.config.serper:
            api_key = (ctx.config.serper.api_key or "").strip() or api_key

        if not api_key or not ctx.http:
            ctx.warning("Serper API not configured for GraphRAG")
            return []

        all_results: list[dict[str, str]] = []

        for query in queries:
            call_record = ToolCallRecord(
                tool="graphrag:search",
                input={"query": query},
                started_at=ctx.now().isoformat(),
            )
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
                    call_record.ended_at = ctx.now().isoformat()
                    call_record.error = f"HTTP {response.status_code}"
                    execution_log.add_call(call_record)
                    continue

                data = response.json()
                results_found = 0
                for item in data.get("organic", [])[:7]:
                    url = item.get("link", "")
                    if url and url not in global_seen_urls:
                        global_seen_urls.add(url)
                        all_results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": url,
                            "date": item.get("date", ""),
                        })
                        results_found += 1

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {"results_found": results_found}
                execution_log.add_call(call_record)

            except Exception as e:
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = str(e)
                execution_log.add_call(call_record)

        return all_results

    # ------------------------------------------------------------------
    # A-3) Fetch pages
    # ------------------------------------------------------------------

    def _fetch_pages(
        self,
        ctx: "AgentContext",
        candidates: list[dict[str, str]],
        execution_log: ToolExecutionLog,
    ) -> list[dict[str, Any]]:
        """Fetch top candidate URLs and extract compact fragments."""
        fragments: list[dict[str, Any]] = []

        for candidate in candidates:
            url = candidate["url"]
            call_record = ToolCallRecord(
                tool="graphrag:fetch",
                input={"url": url},
                started_at=ctx.now().isoformat(),
            )
            try:
                response = ctx.http.get(url, timeout=20.0)
                status_code = response.status_code

                if not response.ok:
                    call_record.ended_at = ctx.now().isoformat()
                    call_record.error = f"HTTP {status_code}"
                    call_record.output = {"status_code": status_code}
                    execution_log.add_call(call_record)
                    continue

                raw_text = response.text[:self.MAX_FRAGMENT_BYTES]
                content_hash = self._hash_content(raw_text)

                fragments.append({
                    "url": url,
                    "title": candidate.get("title", ""),
                    "snippet": candidate.get("snippet", ""),
                    "date": candidate.get("date", ""),
                    "fragment": raw_text,
                    "content_hash": content_hash,
                    "status_code": status_code,
                })

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "status_code": status_code,
                    "content_hash": content_hash,
                    "fragment_length": len(raw_text),
                }
                execution_log.add_call(call_record)

            except Exception as e:
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = str(e)
                execution_log.add_call(call_record)

        return fragments

    # ------------------------------------------------------------------
    # B) Graph element extraction per text unit
    # ------------------------------------------------------------------

    def _extract_graph_elements(
        self,
        ctx: "AgentContext",
        text_unit: Any,
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        execution_log: ToolExecutionLog,
    ) -> dict[str, Any] | None:
        """Ask LLM to extract entities/relations/claims from a text unit."""
        call_record = ToolCallRecord(
            tool="graphrag:extract_elements",
            input={"text_unit_id": text_unit.id, "url": text_unit.doc_url},
            started_at=ctx.now().isoformat(),
        )

        semantics = prompt_spec.prediction_semantics
        prompt = (
            f"### DOCUMENT\n"
            f"URL: {text_unit.doc_url}\n"
            f"Title: {text_unit.doc_title}\n\n"
            f"### TEXT UNIT\n"
            f"{text_unit.content}\n\n"
            f"### CONTEXT\n"
            f"Market Question: {prompt_spec.market.question}\n"
            f"Requirement: {requirement.description}\n"
            f"Event Definition: {prompt_spec.market.event_definition}\n"
            f"Entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n\n"
            f"### TASK\n"
            f"Extract all entities, relations, and key claims from the text unit above."
        )

        messages = [
            {"role": "system", "content": GRAPHRAG_ELEMENT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_error: str | None = None
        raw_output = ""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt == 0:
                    response = ctx.llm.chat(messages)
                    raw_output = response.content
                elements = self._extract_json(raw_output)

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "entities": len(elements.get("entities", [])),
                    "relations": len(elements.get("relations", [])),
                    "claims": len(elements.get("claims", [])),
                }
                execution_log.add_call(call_record)
                return elements

            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                if attempt < self.MAX_RETRIES:
                    repair_prompt = (
                        f"The JSON was invalid: {last_error}. "
                        "Please fix and return valid JSON only."
                    )
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": repair_prompt})
                    response = ctx.llm.chat(messages)
                    raw_output = response.content

        call_record.ended_at = ctx.now().isoformat()
        call_record.error = f"JSON parse failed: {last_error}"
        execution_log.add_call(call_record)
        return None

    # ------------------------------------------------------------------
    # B) Community report generation
    # ------------------------------------------------------------------

    def _generate_community_reports(
        self,
        ctx: "AgentContext",
        graph: GraphIndex,
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        execution_log: ToolExecutionLog,
    ) -> None:
        """Generate LLM summary reports for each community."""
        for community in graph.communities:
            call_record = ToolCallRecord(
                tool="graphrag:community_report",
                input={"community_id": community.id, "num_entities": len(community.entity_ids)},
                started_at=ctx.now().isoformat(),
            )

            # Build community context
            ent_lines: list[str] = []
            all_urls: set[str] = set()
            for eid in community.entity_ids[:15]:
                ent = graph.entities.get(eid)
                if ent:
                    ent_lines.append(f"- {ent.name} ({ent.type}): {ent.description[:120]}")
                    all_urls.update(ent.source_urls)

            rel_lines: list[str] = []
            for rel in graph.relations:
                if rel.head_id in community.entity_ids and rel.tail_id in community.entity_ids:
                    head_name = graph.entities.get(rel.head_id)
                    tail_name = graph.entities.get(rel.tail_id)
                    if head_name and tail_name:
                        rel_lines.append(
                            f"- {head_name.name} --[{rel.relation}]--> {tail_name.name}"
                            f" | \"{rel.quote[:100]}\""
                        )

            # Collect supporting quotes from text units linked to community entities
            tu_ids: set[str] = set()
            for eid in community.entity_ids:
                ent = graph.entities.get(eid)
                if ent:
                    tu_ids.update(ent.text_unit_ids)

            quote_lines: list[str] = []
            for tu in graph.text_units:
                if tu.id in tu_ids and len(quote_lines) < 5:
                    snippet = tu.content[:200].replace("\n", " ")
                    quote_lines.append(f"- [{tu.doc_url}] {snippet}")

            context_text = (
                f"### ENTITIES\n" + "\n".join(ent_lines[:15]) + "\n\n"
                f"### RELATIONS\n" + "\n".join(rel_lines[:10]) + "\n\n"
                f"### SUPPORTING QUOTES\n" + "\n".join(quote_lines[:5])
            )

            prompt = (
                f"Write a community report for this group of related entities.\n\n"
                f"Market Question: {prompt_spec.market.question}\n"
                f"Requirement: {requirement.description}\n\n"
                f"{context_text}"
            )

            messages = [
                {"role": "system", "content": GRAPHRAG_COMMUNITY_REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                response = ctx.llm.chat(messages)
                report_data = self._extract_json(response.content)
                community.report = json.dumps(report_data, ensure_ascii=False)[:1200]
            except (json.JSONDecodeError, ValueError):
                community.report = f"Community {community.id}: {', '.join(community.entity_ids[:5])}"

            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {"report_length": len(community.report)}
            execution_log.add_call(call_record)

    # ------------------------------------------------------------------
    # C-1) Local query: seed entities → neighborhood pack
    # ------------------------------------------------------------------

    def _query_local(
        self,
        ctx: "AgentContext",
        query_text: str,
        graph: GraphIndex,
        execution_log: ToolExecutionLog,
    ) -> str:
        """Identify seed entities and build local context pack."""
        call_record = ToolCallRecord(
            tool="graphrag:local_pack",
            input={"query_length": len(query_text)},
            started_at=ctx.now().isoformat(),
        )

        # Ask LLM for seed entities
        messages = [
            {"role": "system", "content": GRAPHRAG_SEED_ENTITIES_SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ]

        seed_ids: list[str] = []
        try:
            response = ctx.llm.chat(messages)
            seed_data = self._extract_json(response.content)
            raw_seeds = seed_data.get("seed_entities", [])

            # Match seeds to graph entities by normalized name
            for seed in raw_seeds:
                normalized = normalize_entity_name(str(seed))
                if normalized in graph.entities:
                    seed_ids.append(normalized)
                else:
                    # Partial match: check if seed is a substring of entity names
                    for eid in graph.entities:
                        if normalized and normalized in eid and eid not in seed_ids:
                            seed_ids.append(eid)
                            break
        except (json.JSONDecodeError, ValueError):
            # Fallback: use query terms as seeds
            query_terms = set(normalize_entity_name(query_text).split())
            for eid in graph.entities:
                if any(t in eid for t in query_terms):
                    seed_ids.append(eid)

        local_pack = build_local_context_pack(graph, seed_ids[:10], max_chars=4000)

        call_record.ended_at = ctx.now().isoformat()
        call_record.output = {
            "seed_entities": seed_ids[:10],
            "pack_length": len(local_pack),
        }
        execution_log.add_call(call_record)
        return local_pack

    # ------------------------------------------------------------------
    # C-2) Global query: community MAP → REDUCE
    # ------------------------------------------------------------------

    def _query_global_map_reduce(
        self,
        ctx: "AgentContext",
        query_text: str,
        graph: GraphIndex,
        local_pack: str,
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        current_time_str: str,
        execution_log: ToolExecutionLog,
    ) -> dict[str, Any]:
        """Rank communities, MAP each to partial answer, REDUCE to synthesis."""
        semantics = prompt_spec.prediction_semantics

        # Rank communities by query overlap
        query_terms = set(normalize_entity_name(query_text).split())
        selected = rank_communities_by_query(
            graph.communities, query_terms, graph.entities,
            top_m=self.TOP_M_COMMUNITIES,
        )

        if not selected:
            # No communities — go straight to reduce with local pack only
            return self._reduce_step(
                ctx, [], local_pack, requirement, prompt_spec,
                current_time_str, execution_log,
            )

        # MAP step: partial answer per community
        map_results: list[dict[str, Any]] = []
        for community in selected:
            call_record = ToolCallRecord(
                tool="graphrag:map",
                input={"community_id": community.id},
                started_at=ctx.now().isoformat(),
            )

            prompt = (
                f"### QUERY\n"
                f"Market Question: {prompt_spec.market.question}\n"
                f"Requirement: {requirement.description}\n"
                f"Entity: {semantics.target_entity}\n"
                f"Predicate: {semantics.predicate}\n\n"
                f"### COMMUNITY REPORT (ID: {community.id})\n"
                f"{community.report}\n\n"
                f"### TASK\n"
                f"Produce a partial, query-focused answer from this community report."
            )

            messages = [
                {"role": "system", "content": GRAPHRAG_COMMUNITY_MAP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                response = ctx.llm.chat(messages)
                partial = self._extract_json(response.content)
                partial["community_id"] = community.id
                map_results.append(partial)

                call_record.ended_at = ctx.now().isoformat()
                call_record.output = {
                    "supports": partial.get("supports"),
                    "confidence": partial.get("confidence"),
                }
            except (json.JSONDecodeError, ValueError):
                call_record.ended_at = ctx.now().isoformat()
                call_record.error = "LLM returned invalid JSON for map step"

            execution_log.add_call(call_record)

        # REDUCE step
        return self._reduce_step(
            ctx, map_results, local_pack, requirement, prompt_spec,
            current_time_str, execution_log,
        )

    def _reduce_step(
        self,
        ctx: "AgentContext",
        map_results: list[dict[str, Any]],
        local_pack: str,
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        current_time_str: str,
        execution_log: ToolExecutionLog,
    ) -> dict[str, Any]:
        """REDUCE: combine MAP outputs + local pack into final synthesis."""
        call_record = ToolCallRecord(
            tool="graphrag:reduce",
            input={
                "num_map_results": len(map_results),
                "local_pack_length": len(local_pack),
            },
            started_at=ctx.now().isoformat(),
        )

        semantics = prompt_spec.prediction_semantics

        # Format MAP results
        map_text_parts: list[str] = []
        for i, mr in enumerate(map_results):
            block = (
                f"COMMUNITY [{mr.get('community_id', i)}]\n"
                f"SUPPORTS: {mr.get('supports', 'N/A')}\n"
                f"PARSED_VALUE: {mr.get('parsed_value', 'null')}\n"
                f"CONFIDENCE: {mr.get('confidence', 0)}\n"
                f"KEY_FACT: {str(mr.get('key_fact', ''))[:300]}\n"
                f"MISSING: {mr.get('missing_info', [])}"
            )
            map_text_parts.append(block)

        map_text = "\n\n---\n\n".join(map_text_parts) if map_text_parts else "(no community map results)"

        prompt = (
            f"### CONTEXT\n"
            f"CURRENT_TIME: {current_time_str}\n"
            f"Market Question: {prompt_spec.market.question}\n"
            f"Event Definition: {prompt_spec.market.event_definition}\n"
            f"Entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n\n"
            f"### DATA REQUIREMENT\n"
            f"{requirement.description}\n\n"
            f"### LOCAL EVIDENCE PACK (entity-centric)\n"
            f"{local_pack if local_pack else '(empty)'}\n\n"
            f"### GLOBAL COMMUNITY MAP RESULTS\n"
            f"{map_text}\n\n"
            f"### TASK\n"
            f"Synthesize the local evidence pack and the community-level partial "
            f"answers into a single coherent resolution.\n"
            f"Apply source hierarchy: Tier 1 > Tier 2 > Tier 3.\n"
            f"If sources conflict, document both sides."
        )

        messages = [
            {"role": "system", "content": GRAPHRAG_REDUCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = ctx.llm.chat(messages)
            synthesis = self._extract_json(response.content)

            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {
                "resolution_status": synthesis.get("resolution_status"),
                "confidence_score": synthesis.get("confidence_score"),
            }
            execution_log.add_call(call_record)
            return synthesis

        except (json.JSONDecodeError, ValueError):
            call_record.ended_at = ctx.now().isoformat()
            call_record.error = "LLM returned invalid JSON for reduce step"
            execution_log.add_call(call_record)
            return {
                "resolution_status": "UNRESOLVED",
                "confidence_score": 0.0,
                "evidence_sources": [],
                "conflicts": [],
                "missing_info": ["Reduce step failed — LLM returned invalid JSON"],
                "reasoning_trace": "Reduce parse error",
            }

    # ------------------------------------------------------------------
    # D) Build final EvidenceItem
    # ------------------------------------------------------------------

    def _build_evidence_from_synthesis(
        self,
        ctx: "AgentContext",
        req_id: str,
        market_id: str,
        synthesis: dict[str, Any],
        graph: GraphIndex,
    ) -> EvidenceItem:
        """Convert synthesis output + graph stats to an EvidenceItem."""
        evidence_id = hashlib.sha256(
            f"graphrag:{req_id}:final:{market_id}".encode()
        ).hexdigest()[:16]

        resolution_status = synthesis.get("resolution_status", "UNRESOLVED")
        confidence_score = synthesis.get("confidence_score", 0.0)
        success = (
            resolution_status in ("RESOLVED", "AMBIGUOUS")
            and confidence_score >= 0.5
        )

        raw_sources = synthesis.get("evidence_sources", [])
        evidence_sources = _normalize_evidence_sources(raw_sources)

        # Build compact community_reports dict (top 2)
        community_reports: dict[str, str] = {}
        for c in graph.communities[:2]:
            if c.report:
                community_reports[c.id] = c.report[:600]

        reasoning_trace = str(synthesis.get("reasoning_trace", ""))[:500]

        extracted_fields = self._truncate_extracted_fields({
            "confidence_score": confidence_score,
            "resolution_status": resolution_status,
            "evidence_sources": evidence_sources,
            "graph_stats": graph.stats,
            "top_entities": sorted(graph.entities.keys())[:10],
            "selected_communities": [c.id for c in graph.communities[:self.TOP_M_COMMUNITIES]],
            "community_reports": community_reports,
            "conflicts": synthesis.get("conflicts", []),
            "missing_info": synthesis.get("missing_info", []),
        })

        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=req_id,
            provenance=Provenance(
                source_id="graphrag",
                source_uri="serper:search+http:fetch+graphrag:index",
                tier=2,
                fetched_at=ctx.now(),
                content_hash=self._hash_content(reasoning_trace),
            ),
            raw_content=reasoning_trace,
            parsed_value=synthesis.get("parsed_value"),
            extracted_fields=extracted_fields,
            success=success,
            error=None if success else f"Resolution status: {resolution_status}",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        """Validate GraphRAG collection results."""
        checks: list[CheckResult] = []

        if bundle.has_evidence:
            checks.append(CheckResult.passed(
                check_id="has_evidence",
                message=f"GraphRAG collected {len(bundle.items)} evidence items",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_evidence",
                message="No evidence was collected",
            ))

        # Check graph stats present
        graph_items = [
            e for e in bundle.items
            if e.extracted_fields.get("graph_stats")
        ]
        if graph_items:
            checks.append(CheckResult.passed(
                check_id="graph_built",
                message=f"{len(graph_items)} items have graph stats",
            ))

        resolved_count = sum(
            1 for e in bundle.items
            if e.extracted_fields.get("resolution_status") == "RESOLVED"
        )
        if resolved_count > 0:
            checks.append(CheckResult.passed(
                check_id="synthesis_resolved",
                message=f"{resolved_count} requirements resolved via GraphRAG",
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

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return json.loads(text.strip())

    @staticmethod
    def _hash_content(content: str) -> str:
        """Hash content using SHA256."""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _truncate_extracted_fields(
        fields: dict[str, Any], max_value_len: int = 500
    ) -> dict[str, Any]:
        """Truncate extracted field values to prevent oversized evidence."""
        truncated: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, str) and len(value) > max_value_len:
                truncated[key] = value[:max_value_len] + "..."
            else:
                truncated[key] = value
        return truncated


def _normalize_evidence_sources(raw_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize LLM-produced evidence source entries to the standard format.

    Standard format (matching EvidenceSource schema):
        url, source_id, credibility_tier (int 1-3), key_fact, supports, date_published

    Handles variations across collector prompts:
        - credibility_tier as string ("Tier 1") or int (1)
        - supports as bool (supports_hypothesis) or string ("YES"/"NO")
        - relevance_reason → key_fact fallback
    """
    _TIER_MAP = {"tier 1": 1, "tier 2": 2, "tier 3": 3}
    normalized: list[dict[str, Any]] = []

    for src in raw_sources:
        if not isinstance(src, dict):
            continue

        # Normalize credibility_tier
        raw_tier = src.get("credibility_tier", 3)
        if isinstance(raw_tier, str):
            tier = _TIER_MAP.get(raw_tier.lower().strip(), 3)
        else:
            tier = int(raw_tier) if raw_tier in (1, 2, 3) else 3

        # Normalize supports
        raw_supports = src.get("supports", src.get("supports_hypothesis", "N/A"))
        if isinstance(raw_supports, bool):
            supports = "YES" if raw_supports else "NO"
        else:
            supports = str(raw_supports).upper() if raw_supports else "N/A"
            if supports not in ("YES", "NO", "N/A"):
                supports = "N/A"

        # key_fact fallback to relevance_reason
        key_fact = src.get("key_fact", src.get("relevance_reason", ""))

        normalized.append({
            "url": src.get("url", ""),
            "source_id": src.get("source_id"),
            "credibility_tier": tier,
            "key_fact": str(key_fact)[:300] if key_fact else "",
            "supports": supports,
            "date_published": src.get("date_published"),
        })

    return normalized


def get_collector(
    ctx: "AgentContext",
    *,
    prefer_gemini_grounded: bool = False,
    prefer_pan: bool = False,
    prefer_agentic: bool = False,
    prefer_graphrag: bool = False,
    prefer_hyde: bool = False,
    prefer_llm: bool = True,
    prefer_http: bool = True,
    mock_responses: dict[str, Any] | None = None,
    pan_config: Any | None = None,
    gemini_model: str | None = None,
) -> BaseAgent:
    """
    Get the appropriate collector based on context.

    Args:
        ctx: Agent context
        prefer_gemini_grounded: If True, use Gemini with Google Search grounding
        prefer_pan: If True and LLM + HTTP available, use PAN collector
        prefer_agentic: If True and LLM + HTTP available, use AgenticRAG collector
        prefer_graphrag: If True and LLM + HTTP available, use GraphRAG collector
        prefer_hyde: If True and LLM + HTTP available, use HyDE collector
        prefer_llm: If True and LLM client available, use LLM collector
        prefer_http: If True and HTTP client available, use HTTP collector
        mock_responses: Mock responses for mock collector
        pan_config: PANCollectorConfig for the PAN collector
        gemini_model: Override Gemini model for grounded collector

    Returns:
        Appropriate collector agent instance
    """
    if prefer_gemini_grounded:
        from .gemini_grounded_agent import CollectorGeminiGrounded
        kwargs: dict[str, Any] = {}
        if gemini_model:
            kwargs["model"] = gemini_model
        return CollectorGeminiGrounded(**kwargs)
    if prefer_pan and ctx.llm is not None and ctx.http is not None:
        from .pan_agent import PANCollectorAgent
        return PANCollectorAgent(pan_config=pan_config)
    if prefer_agentic and ctx.llm is not None and ctx.http is not None:
        return CollectorAgenticRAG()
    if prefer_graphrag and ctx.llm is not None and ctx.http is not None:
        return CollectorGraphRAG()
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
    from .pan_agent import PANCollectorAgent
    from .gemini_grounded_agent import CollectorGeminiGrounded
    from .crp_agent import CollectorCRP

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorGeminiGrounded",
        factory=lambda ctx: CollectorGeminiGrounded(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=200,  # 1st — single-call grounded search via Gemini
        metadata={
            "description": (
                "Gemini-grounded collector. Uses Google Gemini with built-in "
                "Google Search grounding to resolve market questions in a single "
                "LLM call. Gemini searches the web autonomously and returns a "
                "grounded answer with citations. Requires GOOGLE_API_KEY."
            ),
        },
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorCRP",
        factory=lambda ctx: CollectorCRP(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=195,  # 2nd — 4-phase Cognitive Resolution Pipeline
        metadata={
            "description": (
                "Cognitive Resolution Pipeline collector. 4-phase structured "
                "reasoning: Contract Clerk (parse) -> Investigator (search) -> "
                "Auditor (extract) -> Judge (adjudicate). Each phase is a "
                "separate LLM call with a focused prompt to prevent hallucination."
            ),
        },
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorHyDE",
        factory=lambda ctx: CollectorHyDE(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=190,  # 3rd — HyDE requires both LLM and HTTP
        metadata={"description": "Hypothetical Document Embeddings collector. Generates a hypothetical ideal answer first, then searches for real sources that match it. Good for complex or nuanced requirements."},
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorPAN",
        factory=lambda ctx: PANCollectorAgent(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=170,  # 4th — PAN wraps workflow with search
        metadata={
            "description": (
                "PAN (Program-of-thought with Adaptive search over Nondeterminism) collector. "
                "Runs the evidence-collection workflow through branchpoint search "
                "(beam / best-of-N) to explore multiple execution paths and keep "
                "the highest-scoring result. Configurable via search_algo, "
                "default_branching, and beam_width."
            ),
        },
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorAgenticRAG",
        factory=lambda ctx: CollectorAgenticRAG(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=160,  # 5th — AgenticRAG requires both LLM and HTTP
        metadata={"description": "Agentic RAG collector with iterative deep reasoning. Plans queries, retrieves candidates, assesses relevance via LLM, and synthesizes evidence with conflict handling. Produces the highest-quality evidence bundles."},
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorGraphRAG",
        factory=lambda ctx: CollectorGraphRAG(),
        capabilities={AgentCapability.LLM, AgentCapability.NETWORK},
        priority=150,  # 6th — GraphRAG local-to-global summarization
        metadata={"description": "GraphRAG collector implementing Local-to-Global query-focused summarization. Builds an entity-relation graph from retrieved documents, detects communities, generates community reports, and uses MAP-REDUCE over communities for synthesis."},
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorLLM",
        factory=lambda ctx: CollectorLLM(),
        capabilities={AgentCapability.LLM},
        priority=180,  # 3rd — preferred when LLM available
        metadata={"description": "LLM-based collector that uses web browsing to fetch and interpret URL content. Extracts structured data via LLM with automatic JSON repair. Supports deferred source discovery via search."},
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorHTTP",
        factory=lambda ctx: CollectorHTTP(),
        capabilities={AgentCapability.NETWORK},
        priority=100,  # 7th — direct HTTP fetch
        metadata={"description": "Direct HTTP collector that fetches data from explicit URLs in the tool plan. No LLM interpretation — returns raw API/page responses. Fast and deterministic."},
    )

    register_agent(
        step=AgentStep.COLLECTOR,
        name="CollectorMock",
        factory=lambda ctx: CollectorMock(),
        capabilities={AgentCapability.DETERMINISTIC, AgentCapability.REPLAY},
        priority=50,  # 8th — fallback mock
        is_fallback=True,
        metadata={"description": "Mock collector for testing and replay. Returns predetermined responses without making real network requests."},
    )


# Auto-register on import
_register_agents()
