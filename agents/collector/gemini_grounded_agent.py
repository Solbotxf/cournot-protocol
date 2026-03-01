"""
Open Web Search Collector Agent (CollectorOpenSearch).

Uses Google Gemini with built-in Google Search grounding to resolve
prediction market questions in a single LLM call.  Gemini searches
the open web autonomously and returns a grounded answer with citations.

This is fundamentally different from other collectors:
- No Serper API needed — Gemini does the searching itself.
- Single LLM call per requirement (search + analysis combined).
- Grounding metadata provides authoritative citations automatically.

Requires: google-genai package (pip install google-genai)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

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

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas import DataRequirement


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

GEMINI_GROUNDED_SYSTEM_PROMPT = """\
You are an AI Oracle resolver for a prediction market resolution engine.

Your task:
1. Use Google Search to find the latest, most authoritative information.
2. Evaluate the evidence strictly against the resolution rules below.
3. Return a JSON verdict.

Rules:
- Search the web for current facts. Do NOT rely on training data alone.
- Follow the resolution rules exactly. If the rules specify a deadline,
  check whether the event occurred before that deadline.
- Prefer official / authoritative sources (government, company press
  releases, major wire services) over blogs or social media.
- If evidence is contradictory, side with the most authoritative and
  most recent source.

You MUST respond with ONLY the following JSON object and nothing else.
Do NOT use markdown fences. Do NOT add extra keys.
The JSON MUST have exactly these two keys: "outcome" and "reason".

{"outcome": "Yes or No", "reason": "Brief explanation citing the specific evidence found"}

- "outcome" must be exactly "Yes" or "No" (capitalized first letter).
- "reason" must be a brief string explaining your verdict with evidence.
"""


def _extract_required_domains(requirement: "DataRequirement") -> list[dict[str, str]]:
    """Extract domains and search queries from a requirement's source_targets.

    Returns a list of dicts with keys: domain, uri, search_query (optional).
    Only includes entries where a real domain can be parsed from the URI.
    """
    results: list[dict[str, str]] = []
    for target in requirement.source_targets:
        parsed = urlparse(target.uri)
        domain = parsed.netloc or parsed.path  # handle bare "fotmob.com"
        domain = domain.lower().lstrip("www.")
        if not domain:
            continue
        entry: dict[str, str] = {"domain": domain, "uri": target.uri}
        if target.search_query:
            entry["search_query"] = target.search_query
        results.append(entry)
    return results


def _sources_cover_domains(
    grounding_sources: list[dict[str, str]],
    required_domains: list[dict[str, str]],
) -> bool:
    """Check if any grounding source URIs match any of the required domains."""
    if not required_domains:
        return True  # nothing required — trivially covered
    for src in grounding_sources:
        uri = src.get("uri", "")
        parsed = urlparse(uri)
        host = (parsed.netloc or "").lower().lstrip("www.")
        for req in required_domains:
            if req["domain"] in host or host in req["domain"]:
                return True
    return False


def _build_user_prompt(
    prompt_spec: PromptSpec,
    requirement: "DataRequirement",
) -> str:
    """Build the user prompt from PromptSpec fields."""
    market = prompt_spec.market
    semantics = prompt_spec.prediction_semantics

    # Reconstruct a "rules" block from resolution_rules + event_definition
    rules_lines: list[str] = []
    rules_lines.append(f"Event definition: {market.event_definition}")
    for rule in market.resolution_rules.get_sorted_rules():
        rules_lines.append(f"- [{rule.rule_id}] {rule.description}")

    assumptions_list = prompt_spec.extra.get("assumptions", [])
    if assumptions_list:
        rules_lines.append("Assumptions:")
        for a in assumptions_list:
            rules_lines.append(f"  - {a}")

    rules_text = "\n".join(rules_lines)

    # Build preferred-sources hint from source_targets
    required_domains = _extract_required_domains(requirement)
    source_hint = ""
    if required_domains:
        domains_str = ", ".join(d["domain"] for d in required_domains)
        source_hint = (
            f"\nPreferred data sources (MUST search these first): {domains_str}\n"
            f"These are the authoritative sources specified in the market rules. "
            f"You SHOULD include results from these domains in your search.\n"
        )

    return (
        f"Market question: {market.question}\n\n"
        f"Resolution rules:\n{rules_text}\n\n"
        f"Data requirement: {requirement.description}\n"
        f"{source_hint}\n"
        f"Target entity: {semantics.target_entity}\n"
        f"Predicate: {semantics.predicate}\n"
        f"Threshold: {semantics.threshold or 'N/A'}\n\n"
        f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Search the web and resolve this market. Return ONLY the JSON object."
    )


def _build_targeted_source_prompt(
    prompt_spec: PromptSpec,
    requirement: "DataRequirement",
    required_domains: list[dict[str, str]],
    first_pass_result: dict[str, Any] | None = None,
) -> str:
    """Build a targeted prompt that focuses specifically on required data sources.

    This is used as a second pass when the initial search did not return
    evidence from the required domains.
    """
    market = prompt_spec.market
    semantics = prompt_spec.prediction_semantics

    # Build site:-scoped search instructions
    site_searches: list[str] = []
    for entry in required_domains:
        if entry.get("search_query"):
            site_searches.append(entry["search_query"])
        else:
            site_searches.append(
                f"site:{entry['domain']} {market.question}"
            )

    search_instructions = "\n".join(f"  - {q}" for q in site_searches)

    first_pass_note = ""
    if first_pass_result:
        outcome = first_pass_result.get("outcome", "unknown")
        reason = first_pass_result.get("reason", "")
        first_pass_note = (
            f"\nA previous search found: outcome={outcome}, reason: {reason}\n"
            f"Now verify this by searching the SPECIFIC authoritative sources below.\n"
        )

    return (
        f"Market question: {market.question}\n\n"
        f"Data requirement: {requirement.description}\n\n"
        f"Target entity: {semantics.target_entity}\n"
        f"Predicate: {semantics.predicate}\n"
        f"Threshold: {semantics.threshold or 'N/A'}\n"
        f"{first_pass_note}\n"
        f"IMPORTANT: You MUST search these SPECIFIC data sources:\n"
        f"{search_instructions}\n\n"
        f"Find the relevant page on these sites and extract the exact data needed.\n"
        f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Gemini-Grounded Collector
# ---------------------------------------------------------------------------

class CollectorOpenSearch(BaseAgent):
    """Open web search collector (Gemini only).

    Uses Google Gemini with built-in Google Search grounding to resolve
    market questions in a single LLM call.  Searches the open web
    autonomously and returns a grounded answer with citations.
    Requires GOOGLE_API_KEY.
    """

    _name = "CollectorOpenSearch"
    _version = "v1"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        ctx.info(f"CollectorOpenSearch executing plan {tool_plan.plan_id} "
                 f"[model={self._model}]")

        api_key = self._resolve_api_key(ctx)
        if not api_key:
            return AgentResult.failure(
                error="Google API key not available. "
                      "Set GOOGLE_API_KEY or configure llm.api_key with provider=google.",
            )

        client = self._get_client(api_key)

        bundle = EvidenceBundle(
            bundle_id=f"open_search_{tool_plan.plan_id}",
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

            evidence, record = self._collect_requirement(
                ctx, client, prompt_spec, requirement,
            )
            bundle.add_item(evidence)
            execution_log.add_call(record)

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
            f"GeminiGrounded collection complete: "
            f"{bundle.total_sources_succeeded}/{bundle.total_sources_attempted} succeeded, "
            f"{len(bundle.requirements_fulfilled)}/{len(tool_plan.requirements)} fulfilled"
        )

        return AgentResult(
            output=(bundle, execution_log),
            verification=verification,
            receipts=ctx.get_receipt_refs(),
            success=bundle.has_evidence,
            error=None if bundle.has_evidence else "No evidence collected",
            metadata={
                "collector": "open_search",
                "bundle_id": bundle.bundle_id,
                "model": self._model,
                "total_sources_attempted": bundle.total_sources_attempted,
                "total_sources_succeeded": bundle.total_sources_succeeded,
            },
        )

    # ------------------------------------------------------------------
    # Per-requirement collection
    # ------------------------------------------------------------------

    def _collect_requirement(
        self,
        ctx: "AgentContext",
        client: Any,
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
    ) -> tuple[EvidenceItem, ToolCallRecord]:
        req_id = requirement.requirement_id
        evidence_id = hashlib.sha256(
            f"open_search:{req_id}".encode()
        ).hexdigest()[:16]

        required_domains = _extract_required_domains(requirement)

        user_prompt = _build_user_prompt(prompt_spec, requirement)

        record = ToolCallRecord(
            tool="open_search:search_and_resolve",
            input={"requirement_id": req_id, "model": self._model},
            started_at=ctx.now().isoformat(),
        )

        try:
            ctx.info(f"[GeminiGrounded] Pass 1: general search for {req_id}")
            if required_domains:
                ctx.info(f"[GeminiGrounded] Required data-source domains: "
                         f"{[d['domain'] for d in required_domains]}")

            response = self._call_gemini(client, user_prompt)
            text = self._extract_text(response)
            grounding = self._extract_grounding(response)
            parsed = self._parse_json(text)
            parsed = self._normalize_parsed(parsed)

            # Check if grounding sources cover the required data-source domains
            sources_covered = _sources_cover_domains(
                grounding.get("sources", []), required_domains,
            )

            # --- Pass 2: targeted data-source search if domains were missed ---
            targeted_grounding: dict[str, Any] | None = None
            targeted_parsed: dict[str, Any] | None = None
            if required_domains and not sources_covered:
                ctx.info(
                    f"[GeminiGrounded] Pass 1 missed required domains "
                    f"{[d['domain'] for d in required_domains]}. "
                    f"Running targeted pass 2..."
                )
                targeted_prompt = _build_targeted_source_prompt(
                    prompt_spec, requirement, required_domains,
                    first_pass_result=parsed,
                )
                try:
                    targeted_response = self._call_gemini(client, targeted_prompt)
                    targeted_text = self._extract_text(targeted_response)
                    targeted_grounding = self._extract_grounding(targeted_response)
                    targeted_parsed = self._parse_json(targeted_text)
                    targeted_parsed = self._normalize_parsed(targeted_parsed)
                    ctx.info(
                        f"[GeminiGrounded] Pass 2 complete: "
                        f"outcome={targeted_parsed.get('outcome')}, "
                        f"sources={len(targeted_grounding.get('sources', []))}"
                    )
                except Exception as e2:
                    ctx.info(f"[GeminiGrounded] Pass 2 failed: {e2}")
                    # Fall through — we still have pass-1 results

            # --- Merge results: prefer targeted (data-source) over general ---
            # If targeted pass succeeded and found data-source evidence,
            # use it as the primary result (tier 1). Otherwise use pass 1.
            if targeted_parsed and targeted_grounding:
                targeted_sources_covered = _sources_cover_domains(
                    targeted_grounding.get("sources", []), required_domains,
                )
                if targeted_sources_covered:
                    # Targeted pass found the required sources — use it as primary
                    final_parsed = targeted_parsed
                    final_grounding = targeted_grounding
                    provenance_tier = 1  # tier 1 for data-source match
                    pass_used = "pass_2_targeted"
                else:
                    # Targeted pass also missed — use whichever has a valid outcome
                    final_parsed = targeted_parsed if targeted_parsed.get("outcome", "").lower() in ("yes", "no") else parsed
                    # Merge grounding sources from both passes (dedup by URI)
                    final_grounding = self._merge_grounding(grounding, targeted_grounding)
                    provenance_tier = 2
                    pass_used = "merged"
            else:
                final_parsed = parsed
                final_grounding = grounding
                provenance_tier = 1 if sources_covered and required_domains else 2
                pass_used = "pass_1"

            record.ended_at = ctx.now().isoformat()
            record.output = {
                "outcome": final_parsed.get("outcome"),
                "grounding_sources": len(final_grounding.get("sources", [])),
                "search_queries": final_grounding.get("search_queries", []),
                "pass_used": pass_used,
                "data_source_domains_required": [d["domain"] for d in required_domains],
                "data_source_covered": sources_covered or (targeted_grounding is not None and _sources_cover_domains(
                    (targeted_grounding or {}).get("sources", []), required_domains,
                )),
            }

            # Build evidence sources from grounding metadata
            evidence_sources = self._build_evidence_sources(
                final_grounding, required_domains,
            )

            outcome = final_parsed.get("outcome", "")
            reason = final_parsed.get("reason", "")
            combined_text = json.dumps(final_parsed)
            success = outcome.lower() in ("yes", "no")

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="gemini_grounded",
                    source_uri=f"gemini:{self._model}",
                    tier=provenance_tier,
                    fetched_at=ctx.now(),
                    content_hash=hashlib.sha256(combined_text.encode()).hexdigest(),
                ),
                raw_content=reason[:500] if reason else combined_text[:500],
                parsed_value=outcome,
                extracted_fields={
                    "outcome": outcome,
                    "reason": reason,
                    "evidence_sources": evidence_sources,
                    "grounding_search_queries": final_grounding.get("search_queries", []),
                    "grounding_source_count": len(evidence_sources),
                    "confidence_score": 0.9 if success else 0.0,
                    "resolution_status": "RESOLVED" if success else "UNRESOLVED",
                    "pass_used": pass_used,
                    "data_source_domains_required": [d["domain"] for d in required_domains],
                },
                success=success,
                error=None if success else f"Unexpected outcome: {outcome!r}",
            ), record

        except Exception as e:
            record.ended_at = ctx.now().isoformat()
            record.error = str(e)

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="gemini_grounded",
                    source_uri=f"gemini:{self._model}",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Gemini grounded call failed: {e}",
            ), record

    # ------------------------------------------------------------------
    # Gemini SDK helpers
    # ------------------------------------------------------------------

    def _resolve_api_key(self, ctx: "AgentContext") -> str | None:
        """Resolve Google API key from config or environment."""
        # From RuntimeConfig (if provider is google, use its key)
        if ctx.config and ctx.config.llm.provider == "google" and ctx.config.llm.api_key:
            return ctx.config.llm.api_key

        return os.getenv("GOOGLE_API_KEY")

    def _get_client(self, api_key: str) -> Any:
        """Create the google-genai client (lazy import)."""
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )
        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=80_000),
        )

    def _call_gemini(self, client: Any, prompt: str) -> Any:
        """Call Gemini with Google Search grounding enabled.

        A Python-level timeout (80s) ensures we return before
        Cloudflare's 100s proxy read timeout triggers a 524.
        """
        import concurrent.futures
        from google.genai import types

        def _do_call() -> Any:
            return client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=GEMINI_GROUNDED_SYSTEM_PROMPT,
                    temperature=0.0,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            return future.result(timeout=80)  # seconds

    @staticmethod
    def _merge_grounding(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge grounding metadata from two passes, deduplicating by URI."""
        seen_uris: set[str] = set()
        merged_sources: list[dict[str, str]] = []
        for src in second.get("sources", []) + first.get("sources", []):
            uri = src.get("uri", "")
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                merged_sources.append(src)
        merged_queries = list(dict.fromkeys(
            first.get("search_queries", []) + second.get("search_queries", [])
        ))
        return {"sources": merged_sources, "search_queries": merged_queries}

    @staticmethod
    def _build_evidence_sources(
        grounding: dict[str, Any],
        required_domains: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Build evidence source dicts from grounding, marking data-source matches as tier 1."""
        req_domain_set = {d["domain"] for d in required_domains}
        evidence_sources: list[dict[str, Any]] = []
        for src in grounding.get("sources", []):
            uri = src.get("uri", "")
            parsed = urlparse(uri)
            host = (parsed.netloc or "").lower().lstrip("www.")
            is_required = any(
                rd in host or host in rd for rd in req_domain_set
            )
            evidence_sources.append({
                "url": uri,
                "source_id": src.get("title", "")[:50],
                "credibility_tier": 1 if is_required else 2,
                "key_fact": src.get("title", ""),
                "supports": "N/A",
                "date_published": None,
                "is_required_data_source": is_required,
            })
        return evidence_sources

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from Gemini response."""
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", None)
                if text:
                    return text
        return ""

    @staticmethod
    def _extract_grounding(response: Any) -> dict[str, Any]:
        """Extract grounding metadata (search queries, source URLs) from response."""
        result: dict[str, Any] = {"sources": [], "search_queries": []}

        for candidate in getattr(response, "candidates", []):
            gm = getattr(candidate, "grounding_metadata", None)
            if gm is None:
                continue

            # Search queries used by Gemini
            queries = getattr(gm, "web_search_queries", None)
            if queries:
                result["search_queries"] = list(queries)

            # Grounding chunks (the web pages Gemini cited)
            chunks = getattr(gm, "grounding_chunks", None)
            if chunks:
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        result["sources"].append({
                            "uri": getattr(web, "uri", ""),
                            "title": getattr(web, "title", ""),
                        })

        return result

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from Gemini response text.

        Gemini sometimes returns extra text or even multiple JSON objects.
        We therefore:
        1) strip markdown fences if present
        2) locate the first "{" and then use JSONDecoder.raw_decode to parse
           the first JSON object only (ignoring trailing text)
        """
        text = text.strip()

        # Try stripping markdown code block (```json ... ```)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        start = text.find("{")
        if start >= 0:
            decoder = json.JSONDecoder()
            obj, _idx = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj

        # Fallback: attempt full parse
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Expected JSON object")
        return obj

    @staticmethod
    def _normalize_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
        """Normalize Gemini response keys to the expected {outcome, reason} format.

        Gemini sometimes ignores the prompt schema and returns alternative keys
        like ``resolution`` instead of ``outcome``, or ``evidence`` instead of
        ``reason``.  This maps known variants back to the canonical keys.
        """
        # --- outcome ---
        if "outcome" not in parsed:
            raw_outcome = parsed.get("resolution", "")
            if isinstance(raw_outcome, str) and raw_outcome.upper() in ("YES", "NO"):
                parsed["outcome"] = raw_outcome.capitalize()
            else:
                parsed["outcome"] = raw_outcome

        # Normalize casing: "YES"/"NO" -> "Yes"/"No"
        if isinstance(parsed.get("outcome"), str):
            upper = parsed["outcome"].strip().upper()
            if upper == "YES":
                parsed["outcome"] = "Yes"
            elif upper == "NO":
                parsed["outcome"] = "No"

        # --- reason ---
        if "reason" not in parsed:
            parsed["reason"] = parsed.get("evidence", "")

        return parsed

    # ------------------------------------------------------------------
    # Validation
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
