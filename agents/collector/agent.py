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

DISCOVERY_QUERY_SYSTEM_PROMPT = (
    "You generate web search queries for a prediction market resolution system. "
    "Given a data requirement and market context, output 1-3 targeted search queries "
    "that would find authoritative, official, or reliable sources.\n\n"
    "Prefer queries that target:\n"
    "- Official government/organization websites\n"
    "- Major news agencies (Reuters, AP, BBC)\n"
    "- Domain-specific authoritative sources\n\n"
    'Return ONLY valid JSON: {"queries": ["query1", "query2", ...]}'
)

DEFERRED_DISCOVERY_SYSTEM_PROMPT = (
    "You are a source discovery agent for a prediction market resolution system. "
    "You will be given search results about a topic. "
    "Your task is to analyze the results, identify the most authoritative and relevant sources, "
    "and extract key evidence.\n\n"
    "Return ONLY valid JSON with this schema:\n"
    '{"success": true/false, '
    '"discovered_sources": [{"url": "...", "title": "...", "relevance": "high/medium/low"}], '
    '"extracted_fields": {...relevant data extracted from search results...}, '
    '"parsed_value": <the key finding>, '
    '"content_summary": "<synthesis of findings from all sources>", '
    '"confidence": 0.0-1.0, '
    '"error": null}'
)


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
        3. Feed search results to LLM to summarize and extract evidence
        """
        semantics = prompt_spec.prediction_semantics
        evidence_id = hashlib.sha256(
            f"discover:{requirement.requirement_id}".encode()
        ).hexdigest()[:16]

        # Step 1: Generate search queries via LLM
        query_prompt = (
            f"Generate search queries for this data requirement:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Market question: {prompt_spec.market.question}\n"
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
            query_data = self._extract_json(response.content)
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
        results_text = "\n\n".join(
            f"[{i+1}] {r['title']}\n{r['snippet']}\n{r['url']}"
            for i, r in enumerate(search_results)
        )

        synthesis_prompt = (
            f"Analyze these search results for the following requirement:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Market question: {prompt_spec.market.question}\n"
            f"- Entity: {semantics.target_entity}\n"
            f"- Predicate: {semantics.predicate}\n\n"
            f"Search results:\n{results_text}\n\n"
            f"Identify authoritative sources, extract relevant evidence, and provide "
            f"a confidence score (0.0-1.0) for how well the evidence addresses the requirement."
        )

        messages = [
            {"role": "system", "content": DEFERRED_DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ]
        response = ctx.llm.chat(messages)

        # Parse synthesis response (with retry)
        last_error: str | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                parsed = self._extract_json(response.content)
                extracted = self._truncate_extracted_fields(
                    parsed.get("extracted_fields", {})
                )

                # Include discovered_sources in extracted_fields
                discovered = parsed.get("discovered_sources", [])
                if discovered:
                    extracted["discovered_sources"] = discovered

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
                    raw_content=str(parsed.get("content_summary", ""))[:100],
                    parsed_value=parsed.get("parsed_value"),
                    extracted_fields=extracted,
                    success=parsed.get("success", True),
                    error=parsed.get("error"),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = str(e)
                if attempt < self.MAX_RETRIES:
                    repair_prompt = (
                        f"The JSON was invalid: {last_error}. "
                        "Please fix and return valid JSON only."
                    )
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": repair_prompt})
                    response = ctx.llm.chat(messages)

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
            error=f"Discovery synthesis failed: {last_error}",
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


def get_collector(
    ctx: "AgentContext",
    *,
    prefer_llm: bool = True,
    prefer_http: bool = True,
    mock_responses: dict[str, Any] | None = None,
) -> BaseAgent:
    """
    Get the appropriate collector based on context.

    Args:
        ctx: Agent context
        prefer_llm: If True and LLM client available, use LLM collector
        prefer_http: If True and HTTP client available, use HTTP collector
        mock_responses: Mock responses for mock collector

    Returns:
        CollectorLLM, CollectorHTTP, or CollectorMock
    """
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
        name="CollectorLLM",
        factory=lambda ctx: CollectorLLM(),
        capabilities={AgentCapability.LLM},
        priority=150,  # Highest - preferred when LLM available
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
