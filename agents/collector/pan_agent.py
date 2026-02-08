"""
PAN Collector Agent — inference-time scaling for evidence collection.

Wraps the collector workflow (query generation → web search → evidence
extraction → packaging) with the PAN runtime so that each unreliable step
becomes a branchpoint.  The search algorithm (beam / best-of-N) explores
multiple execution paths and keeps the one with the highest score.

The workflow is written *as if* every LLM call succeeds.  The runtime
handles exploration of alternatives — no retry loops in the workflow.

Compatible interface: same inputs/outputs as CollectorLLM.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, BaseAgent
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

from .pan_runtime import (
    BranchRequest,
    ExecutionPath,
    SearchAlgo,
    SearchConfig,
    search,
)

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas import DataRequirement

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Authoritative top-level domains / URL substrings.
_AUTHORITATIVE_PATTERNS = (
    ".gov", ".edu", "reuters.com", "apnews.com", "bloomberg.com",
    "nytimes.com", "bbc.com", "bbc.co.uk", "wsj.com",
)


def score_query_set(
    queries: list[str],
    requirement_desc: str,
    market_question: str,
) -> float:
    """Score a candidate set of search queries.

    Rewards:
    - More queries (up to 5)
    - Diversity (unique first words → different angles)
    - Longer queries (more specific, up to ~60 chars)
    - Queries that reuse terms from the requirement
    """
    if not queries:
        return 0.0

    n = min(len(queries), 5)
    count_score = n / 5.0  # 0.0 – 1.0

    # Diversity: ratio of unique first words
    first_words = {q.split()[0].lower() for q in queries if q.strip()}
    diversity = len(first_words) / max(n, 1)

    # Specificity: average normalised length (cap at 60 chars)
    avg_len = sum(min(len(q), 60) for q in queries) / max(n, 1)
    specificity = avg_len / 60.0

    # Overlap with requirement tokens
    req_tokens = set(requirement_desc.lower().split())
    overlap_hits = 0
    for q in queries:
        q_tokens = set(q.lower().split())
        if q_tokens & req_tokens:
            overlap_hits += 1
    overlap = overlap_hits / max(n, 1)

    return 0.25 * count_score + 0.25 * diversity + 0.20 * specificity + 0.30 * overlap


def score_search_results(results: list[dict[str, str]]) -> float:
    """Score a set of raw search results.

    Rewards:
    - Having results at all
    - Authoritative domains
    - Recency (has a date)
    - Snippet length (more content)
    """
    if not results:
        return 0.0

    n = len(results)
    count_score = min(n, 10) / 10.0

    auth_hits = 0
    dated = 0
    snippet_len_total = 0
    for r in results:
        url = r.get("url", "").lower()
        if any(pat in url for pat in _AUTHORITATIVE_PATTERNS):
            auth_hits += 1
        if r.get("date"):
            dated += 1
        snippet_len_total += min(len(r.get("snippet", "")), 300)

    auth_score = auth_hits / max(n, 1)
    date_score = dated / max(n, 1)
    snippet_score = (snippet_len_total / max(n, 1)) / 300.0

    return 0.30 * count_score + 0.30 * auth_score + 0.20 * date_score + 0.20 * snippet_score


def score_evidence_extraction(data: dict[str, Any]) -> float:
    """Score a parsed evidence-extraction dict from the LLM.

    Rewards:
    - Resolved status
    - High confidence_score
    - Has parsed_value
    - Has evidence_sources with URLs
    - Has reasoning_trace
    """
    if not data:
        return 0.0

    resolution = data.get("resolution_status", "UNRESOLVED")
    res_score = {"RESOLVED": 1.0, "AMBIGUOUS": 0.5}.get(resolution, 0.0)

    confidence = float(data.get("confidence_score", 0.0))
    conf_score = max(0.0, min(1.0, confidence))

    has_value = 1.0 if data.get("parsed_value") is not None else 0.0

    sources = data.get("evidence_sources", [])
    sources_with_url = sum(1 for s in sources if isinstance(s, dict) and s.get("url"))
    src_score = min(sources_with_url, 5) / 5.0

    has_trace = 1.0 if data.get("reasoning_trace") else 0.0

    return (
        0.25 * res_score
        + 0.25 * conf_score
        + 0.20 * has_value
        + 0.20 * src_score
        + 0.10 * has_trace
    )


def score_final(evidence_item: EvidenceItem) -> float:
    """Compute the final score for a complete EvidenceItem.

    This is the objective that the PAN search maximises.
    """
    if not evidence_item.success:
        return 0.0

    base = 0.3  # baseline for any successful item

    # Provenance tier contribution
    tier_score = evidence_item.provenance.tier / 4.0  # 0..1

    # extracted_fields richness
    ef = evidence_item.extracted_fields or {}
    field_count = min(len(ef), 5)
    fields_score = field_count / 5.0

    # Confidence if present
    confidence = float(ef.get("confidence_score", 0.5))
    conf_score = max(0.0, min(1.0, confidence))

    # Citations
    sources = ef.get("evidence_sources", [])
    citation_score = min(len(sources), 5) / 5.0

    # Has parsed_value
    has_value = 1.0 if evidence_item.parsed_value is not None else 0.0

    return base + 0.70 * (
        0.20 * tier_score
        + 0.20 * fields_score
        + 0.25 * conf_score
        + 0.20 * citation_score
        + 0.15 * has_value
    )


# ---------------------------------------------------------------------------
# PAN Collector Config
# ---------------------------------------------------------------------------

@dataclass
class PANCollectorConfig:
    """PAN-specific configuration, separate from SearchConfig."""
    search_algo: str = "beam"          # "bon_global" | "bon_local" | "beam"
    default_branching: int = 3         # N candidates per branchpoint
    beam_width: int = 2                # K beams to keep (beam search only)
    max_expansions: int = 50           # safety cap on generate_fn calls
    seed: int | None = None            # RNG seed for determinism

    def to_search_config(self) -> SearchConfig:
        algo_map = {
            "bon_global": SearchAlgo.BON_GLOBAL,
            "bon_local": SearchAlgo.BON_LOCAL,
            "beam": SearchAlgo.BEAM,
        }
        return SearchConfig(
            algo=algo_map.get(self.search_algo, SearchAlgo.BEAM),
            default_branching=self.default_branching,
            beam_width=self.beam_width,
            max_expansions=self.max_expansions,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# System prompts (reused from CollectorLLM)
# ---------------------------------------------------------------------------

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
    "reasoning_trace": "A detailed step-by-step deduction...",
    "resolution_status": "RESOLVED" | "AMBIGUOUS" | "UNRESOLVED",
    "parsed_value": "The specific answer string extracted from the text. Return null if unresolved.",
    "confidence_score": 0.0 to 1.0,
    "evidence_sources": [
        {
            "source_id": "[X]",
            "url": "...",
            "credibility_tier": 1 or 2 or 3,
            "key_fact": "The specific fact extracted from this source",
            "supports": "YES" or "NO" or "N/A",
            "date_published": "YYYY-MM-DD" or null
        }
    ]
}
"""


# ---------------------------------------------------------------------------
# PAN Collector Agent
# ---------------------------------------------------------------------------

class PANCollectorAgent(BaseAgent):
    """PAN (Program-of-thought with Adaptive search over Nondeterminism) Collector.

    Runs the evidence-collection workflow through the PAN runtime so that
    each unreliable step (LLM call) is a branchpoint.  A search algorithm
    (beam / best-of-N) explores multiple paths and keeps the best.

    Branchpoints:
        1. query_generation — generate N query sets, pick best
        2. snippet_selection — (identity when results are deterministic)
        3. evidence_extraction — generate N extraction summaries, pick best
        4. evidence_packaging — final packaging (deterministic, no branching)

    Compatible with the standard collector interface: returns
    ``(EvidenceBundle, ToolExecutionLog)`` in ``AgentResult.output``.
    """

    _name = "CollectorPAN"
    _version = "v1"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}

    def __init__(
        self,
        *,
        pan_config: PANCollectorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pan_config = pan_config or PANCollectorConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        """Execute collection with PAN search over execution paths."""
        ctx.info(f"CollectorPAN executing plan {tool_plan.plan_id} "
                 f"[algo={self.pan_config.search_algo}, "
                 f"N={self.pan_config.default_branching}, "
                 f"K={self.pan_config.beam_width}]")

        if ctx.llm is None:
            return AgentResult.failure(error="LLM client not available")

        bundle = EvidenceBundle(
            bundle_id=f"pan_{tool_plan.plan_id}",
            market_id=prompt_spec.market_id,
            plan_id=tool_plan.plan_id,
        )
        execution_log = ToolExecutionLog(
            plan_id=tool_plan.plan_id,
            started_at=ctx.now().isoformat(),
        )

        all_path_metadata: list[dict[str, Any]] = []

        for req_id in tool_plan.requirements:
            requirement = prompt_spec.get_requirement_by_id(req_id)
            if not requirement:
                continue

            evidence, path = self._collect_with_search(
                ctx, requirement, prompt_spec, execution_log,
            )
            bundle.add_item(evidence)
            all_path_metadata.append({
                "requirement_id": req_id,
                "score_breakdown": path.score_breakdown,
                "total_score": path.total_score,
                "final_score": path.final_score,
                "finished": path.finished,
                "error": path.error,
            })

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

        verification = self._validate_output(bundle, tool_plan)

        ctx.info(
            f"PAN collection complete: {bundle.total_sources_succeeded}/"
            f"{bundle.total_sources_attempted} sources succeeded, "
            f"{len(bundle.requirements_fulfilled)}/{len(tool_plan.requirements)} "
            f"requirements fulfilled"
        )

        return AgentResult(
            output=(bundle, execution_log),
            verification=verification,
            receipts=ctx.get_receipt_refs(),
            success=bundle.has_evidence,
            error=None if bundle.has_evidence else "No evidence collected",
            metadata={
                "collector": "pan",
                "bundle_id": bundle.bundle_id,
                "search_algo": self.pan_config.search_algo,
                "default_branching": self.pan_config.default_branching,
                "beam_width": self.pan_config.beam_width,
                "paths": all_path_metadata,
            },
        )

    # ------------------------------------------------------------------
    # Core: build the workflow generator and run search
    # ------------------------------------------------------------------

    def _collect_with_search(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        execution_log: ToolExecutionLog,
    ) -> tuple[EvidenceItem, ExecutionPath]:
        """Run the PAN search for a single requirement.

        The workflow generator accumulates ToolCallRecords locally
        (not on the shared execution_log) and returns them alongside
        the EvidenceItem.  Only the best path's records are appended
        to execution_log here, exactly once.
        """

        search_config = self.pan_config.to_search_config()

        # The workflow returns (EvidenceItem, list[ToolCallRecord]).
        def workflow_factory():
            return self._workflow(ctx, requirement, prompt_spec)

        result, path = search(workflow_factory, search_config)

        # Unpack (EvidenceItem, records) or handle None
        if result is not None:
            evidence, records = result
            for rec in records:
                execution_log.add_call(rec)
        else:
            evidence_id = hashlib.sha256(
                f"pan:{requirement.requirement_id}".encode()
            ).hexdigest()[:16]
            evidence = EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="pan",
                    source_uri="pan:search_failed",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"PAN search failed: {path.error or 'all paths failed'}",
            )

        return evidence, path

    # ------------------------------------------------------------------
    # Workflow generator (yields BranchRequests at each unreliable step)
    # ------------------------------------------------------------------

    def _workflow(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
    ) -> Generator[BranchRequest, Any, tuple[EvidenceItem, list[ToolCallRecord]]]:
        """The collector workflow written as a generator.

        Yields BranchRequest at each unreliable step.  The PAN runtime
        sends back the chosen candidate via ``generator.send(candidate)``.

        The workflow is written *as if* each LLM call succeeds perfectly.

        Returns:
            (EvidenceItem, records) — the evidence plus the ToolCallRecords
            accumulated *only* for this execution path.  The caller appends
            the records from the best path to the shared execution_log once.
        """
        req_id = requirement.requirement_id
        semantics = prompt_spec.prediction_semantics
        evidence_id = hashlib.sha256(
            f"pan:{req_id}".encode()
        ).hexdigest()[:16]

        market_question = prompt_spec.market.question
        event_def = prompt_spec.market.event_definition
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = (
            "\n".join(f"- {a}" for a in assumptions_list)
            if assumptions_list else "None"
        )

        # Local record accumulator — NOT the shared execution_log.
        records: list[ToolCallRecord] = []

        # ── Branchpoint 1: query generation ──────────────────────────
        queries: list[str] = yield BranchRequest(
            tag="query_generation",
            generate_fn=lambda: self._generate_queries(
                ctx, requirement, prompt_spec,
            ),
            score_fn=lambda qs: score_query_set(
                qs, requirement.description, market_question,
            ),
        )

        records.append(ToolCallRecord(
            tool="pan:query_generation",
            input={"requirement_id": req_id},
            output={"queries": queries},
            started_at=ctx.now().isoformat(),
            ended_at=ctx.now().isoformat(),
        ))

        ctx.info(f"PAN branchpoint 1 chose {len(queries)} queries: {queries}")

        # ── Branchpoint 2: search + snippet selection ────────────────
        search_results: list[dict[str, str]] = yield BranchRequest(
            tag="snippet_selection",
            generate_fn=lambda: self._execute_search(ctx, queries),
            score_fn=lambda sr: score_search_results(sr),
        )

        records.append(ToolCallRecord(
            tool="pan:web_search",
            input={"queries": queries},
            output={"num_results": len(search_results)},
            started_at=ctx.now().isoformat(),
            ended_at=ctx.now().isoformat(),
        ))

        ctx.info(f"PAN branchpoint 2 selected {len(search_results)} results")

        if not search_results:
            item = EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="pan",
                    source_uri="serper:search",
                    tier=1,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="No search results found",
            )
            return (item, records)

        # ── Branchpoint 3: evidence extraction ───────────────────────
        extraction_data: dict[str, Any] = yield BranchRequest(
            tag="evidence_extraction",
            generate_fn=lambda: self._extract_evidence(
                ctx, requirement, prompt_spec, search_results,
                event_def, assumptions_str,
            ),
            score_fn=lambda d: score_evidence_extraction(d),
        )

        records.append(ToolCallRecord(
            tool="pan:evidence_extraction",
            input={"requirement_id": req_id, "num_results": len(search_results)},
            output={
                "resolution_status": extraction_data.get("resolution_status"),
                "confidence_score": extraction_data.get("confidence_score"),
            },
            started_at=ctx.now().isoformat(),
            ended_at=ctx.now().isoformat(),
        ))

        ctx.info(
            f"PAN branchpoint 3 extracted evidence: "
            f"status={extraction_data.get('resolution_status')}, "
            f"confidence={extraction_data.get('confidence_score')}"
        )

        # ── Step 4: package evidence (deterministic, no branchpoint) ─
        evidence_item = self._package_evidence(
            ctx, requirement, evidence_id, extraction_data, queries, search_results,
        )

        # ── Record final score ────────────────────────────────────────
        # Yield one more BranchRequest to record the final score in
        # the path.  n_candidates=1 means no actual branching.
        _: Any = yield BranchRequest(
            tag="final",
            generate_fn=lambda: (evidence_item, records),
            score_fn=lambda pair: score_final(pair[0]),
            n_candidates=1,
        )

        return (evidence_item, records)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _generate_queries(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
    ) -> list[str]:
        """Generate search queries via LLM (one candidate set)."""
        semantics = prompt_spec.prediction_semantics
        event_def = prompt_spec.market.event_definition
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = (
            "\n".join(f"- {a}" for a in assumptions_list)
            if assumptions_list else "None"
        )

        query_prompt = (
            f"Generate search queries for this data requirement:\n"
            f"- Requirement: {requirement.description}\n"
            f"- Market question: {prompt_spec.market.question}\n"
            f"- Event Definition: {event_def}\n"
            f"- Key Assumptions:\n{assumptions_str}\n"
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
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            query_data = json.loads(content)
            queries = query_data.get("queries", [])[:5]
        except (json.JSONDecodeError, KeyError):
            queries = [f"{semantics.target_entity} {semantics.predicate}"]

        return queries or [f"{semantics.target_entity} {semantics.predicate}"]

    def _execute_search(
        self,
        ctx: "AgentContext",
        queries: list[str],
    ) -> list[dict[str, str]]:
        """Execute search queries via Serper API."""
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if ctx.config and hasattr(ctx.config, "serper") and ctx.config.serper:
            api_key = (ctx.config.serper.api_key or "").strip() or api_key

        if not api_key or not ctx.http:
            ctx.warning("Serper API not configured for PAN collector")
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
                            "date": item.get("date", ""),
                        })
            except Exception as e:
                ctx.warning(f"Serper search error for '{query}': {e}")

        return all_results

    def _extract_evidence(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        prompt_spec: PromptSpec,
        search_results: list[dict[str, str]],
        event_def: str,
        assumptions_str: str,
    ) -> dict[str, Any]:
        """Ask LLM to synthesise search results into structured evidence."""
        semantics = prompt_spec.prediction_semantics
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        formatted_results = []
        for i, r in enumerate(search_results):
            formatted_results.append(
                f"SOURCE_ID: [{i+1}]\n"
                f"TITLE: {r.get('title', 'Unknown')}\n"
                f"URL: {r.get('url', 'Unknown')}\n"
                f"DATE_PUBLISHED: {r.get('date', 'Unknown')}\n"
                f"SNIPPET: {r.get('snippet', '')}"
            )
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
            f"1. Analyze the SOURCE_IDs above to answer the Market Question.\n"
            f"2. Compare DATE_PUBLISHED against Current Date/Time to filter outdated info.\n"
            f"3. Extract the answer into the parsed_value field.\n"
            f"4. If multiple sources conflict, explain in reasoning_trace.\n"
            f"5. Return ONLY the JSON object defined in the system prompt."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": DEFERRED_DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ]
        response = ctx.llm.chat(messages)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, KeyError):
            return {
                "resolution_status": "UNRESOLVED",
                "confidence_score": 0.0,
                "parsed_value": None,
                "reasoning_trace": f"JSON parse failed: {response.content[:200]}",
                "evidence_sources": [],
            }

    def _package_evidence(
        self,
        ctx: "AgentContext",
        requirement: "DataRequirement",
        evidence_id: str,
        extraction_data: dict[str, Any],
        queries: list[str],
        search_results: list[dict[str, str]],
    ) -> EvidenceItem:
        """Build the final EvidenceItem from extraction data (deterministic)."""
        raw_sources = extraction_data.get("evidence_sources", [])
        evidence_sources = _normalize_evidence_sources(raw_sources)

        resolution_status = extraction_data.get("resolution_status", "UNRESOLVED")
        success = resolution_status in ("RESOLVED", "AMBIGUOUS")

        extracted_fields = {
            "confidence_score": extraction_data.get("confidence_score"),
            "resolution_status": resolution_status,
            "evidence_sources": evidence_sources,
            "pan_queries": queries,
            "pan_search_result_count": len(search_results),
        }

        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=requirement.requirement_id,
            provenance=Provenance(
                source_id="pan",
                source_uri="serper:search",
                tier=2,
                fetched_at=ctx.now(),
                content_hash=hashlib.sha256(
                    json.dumps(extraction_data, sort_keys=True, default=str).encode()
                ).hexdigest(),
            ),
            raw_content=str(extraction_data.get("reasoning_trace", ""))[:500],
            parsed_value=extraction_data.get("parsed_value"),
            extracted_fields=extracted_fields,
            success=success,
            error=None if success else f"Resolution status: {resolution_status}",
        )

    # ------------------------------------------------------------------
    # Validation (same as CollectorLLM)
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
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


# ---------------------------------------------------------------------------
# Normalize evidence sources (shared utility)
# ---------------------------------------------------------------------------

_TIER_MAP = {"tier 1": 1, "tier 2": 2, "tier 3": 3}


def _normalize_evidence_sources(raw_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize LLM-produced evidence source entries to the standard format."""
    normalized: list[dict[str, Any]] = []
    for src in raw_sources:
        if not isinstance(src, dict):
            continue

        raw_tier = src.get("credibility_tier", 3)
        if isinstance(raw_tier, str):
            tier = _TIER_MAP.get(raw_tier.lower().strip(), 3)
        else:
            tier = int(raw_tier) if raw_tier in (1, 2, 3) else 3

        raw_supports = src.get("supports", src.get("supports_hypothesis", "N/A"))
        if isinstance(raw_supports, bool):
            supports = "YES" if raw_supports else "NO"
        else:
            supports = str(raw_supports).upper() if raw_supports else "N/A"
            if supports not in ("YES", "NO", "N/A"):
                supports = "N/A"

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
