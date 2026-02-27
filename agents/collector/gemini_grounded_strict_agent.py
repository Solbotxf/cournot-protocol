"""
Gemini-Grounded Strict Collector Agent.

A strict variant of CollectorGeminiGrounded that ONLY searches within
the data-source domains specified in the requirement's source_targets.
Evidence from non-required domains is discarded.

Uses a two-phase approach:
1. **Serper discovery** — find actual page URLs on the required domain
   via the Serper API (google.serper.dev).  This reliably surfaces
   JS-heavy sites (e.g. Fotmob) that Gemini's built-in grounding misses.
2. **Gemini UrlContext + GoogleSearch** — pass the discovered URLs to
   Gemini via the ``UrlContext`` tool so it can ingest the full page
   content, alongside ``GoogleSearch`` for supplementary evidence.

If no source_targets are defined on the requirement, this collector
will fail — use CollectorGeminiGrounded instead for open-ended search.

Requires: google-genai package (pip install google-genai)
"""

from __future__ import annotations

import hashlib
import json
import os
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

from .gemini_grounded_agent import (
    GEMINI_GROUNDED_SYSTEM_PROMPT,
    _extract_required_domains,
    _sources_cover_domains,
    CollectorGeminiGrounded,
)

from .fotmob import (
    FotMobExtractionError,
    FotMobMatchData,
    fetch_match_stats as fotmob_fetch_match_stats,
    find_stat as fotmob_find_stat,
    match_team as fotmob_match_team,
)

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas import DataRequirement


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERPER_URL = "https://google.serper.dev/search"
_SERPER_MAX_URLS = 3  # top N Serper results to pass to UrlContext


# ---------------------------------------------------------------------------
# Strict prompt templates
# ---------------------------------------------------------------------------

STRICT_SYSTEM_PROMPT = """\
You are an AI Oracle resolver for a prediction market resolution engine.

Your task:
1. Use the provided URLs and Google Search to find the information
   needed to resolve this market.
2. Evaluate the evidence strictly against the resolution rules below.
3. Return a JSON verdict.

Rules:
- FIRST, read the provided URLs from the required data source domain
  and try to extract the exact data requested.
- If the specific data field is not visible in the provided URLs
  (e.g. because the page uses dynamic JavaScript rendering), use
  Google Search to find the same data from any reliable source.
- When using supplementary sources, cross-reference with the required
  domain page to ensure accuracy.
- Follow the resolution rules exactly.
- Do NOT rely on training data — search the web.

You MUST respond with ONLY the following JSON object and nothing else.
Do NOT use markdown fences. Do NOT add extra keys.
The JSON MUST have exactly these two keys: "outcome" and "reason".

{"outcome": "Yes or No", "reason": "Brief explanation citing the specific evidence found"}

- "outcome" must be exactly "Yes" or "No" (capitalized first letter).
- "reason" must be a brief string explaining your verdict with the
  exact data value found and its source.
"""


def _build_strict_prompt(
    prompt_spec: PromptSpec,
    requirement: "DataRequirement",
    required_domains: list[dict[str, str]],
    discovered_urls: list[dict[str, str]] | None = None,
) -> str:
    """Build a prompt that restricts search to required data-source domains only.

    When *discovered_urls* is provided (from Serper pre-search), the prompt
    instructs Gemini to read those specific pages first, before falling back
    to general search.
    """
    market = prompt_spec.market
    semantics = prompt_spec.prediction_semantics
    domains_str = ", ".join(d["domain"] for d in required_domains)

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

    # Resolution rules
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

    # Discovered URLs section
    url_section = ""
    if discovered_urls:
        url_lines = []
        for i, entry in enumerate(discovered_urls, 1):
            url_lines.append(f"  {i}. {entry['url']}")
            if entry.get("title"):
                url_lines[-1] += f"  ({entry['title']})"
        url_block = "\n".join(url_lines)
        url_section = (
            f"\nDISCOVERED PAGES on {domains_str} — read these URLs first:\n"
            f"{url_block}\n"
            f"Extract the exact data needed from these pages. "
            f"Look at ALL sections of the page, not just top-level stats.\n"
        )

    # Expected fields hint
    expected_fields = requirement.expected_fields or []
    fields_hint = ""
    if expected_fields:
        fields_str = ", ".join(f"'{f}'" for f in expected_fields)
        fields_hint = (
            f"\nExpected data fields to extract: {fields_str}\n"
            f"Search through ALL stat sections on the page to find these fields.\n"
        )

    return (
        f"Market question: {market.question}\n\n"
        f"Resolution rules:\n{rules_text}\n\n"
        f"Data requirement: {requirement.description}\n\n"
        f"Target entity: {semantics.target_entity}\n"
        f"Predicate: {semantics.predicate}\n"
        f"Threshold: {semantics.threshold or 'N/A'}\n"
        f"{fields_hint}"
        f"{url_section}\n"
        f"MANDATORY DATA SOURCES — search ONLY these domains: {domains_str}\n"
        f"Use these exact searches:\n"
        f"{search_instructions}\n\n"
        f"Do NOT use evidence from any other website.\n"
        f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Search the web and resolve this market. Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Strict Gemini-Grounded Collector
# ---------------------------------------------------------------------------

class CollectorGeminiGroundedStrict(CollectorGeminiGrounded):
    """Strict variant of the Gemini-grounded collector.

    Only searches within data-source domains specified in the
    requirement's ``source_targets``.  Evidence from other domains is
    discarded.  If no source_targets are defined, the collector fails.

    **Two-phase approach:**

    1. **Serper discovery** — uses the Serper API to find actual page
       URLs on the required domains.  This reliably surfaces JS-heavy
       sites that Gemini's grounding alone misses.
    2. **Gemini UrlContext + GoogleSearch** — passes the discovered URLs
       to Gemini via the ``UrlContext`` tool (full page ingestion)
       alongside ``GoogleSearch`` for supplementary evidence.

    The collector makes up to ``max_attempts`` Gemini calls per
    requirement, retrying until evidence from a required domain is found.
    """

    _name = "CollectorGeminiGroundedStrict"
    _version = "v2"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        max_attempts: int = 3,
        serper_max_urls: int = _SERPER_MAX_URLS,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._max_attempts = max_attempts
        self._serper_max_urls = serper_max_urls

    # ------------------------------------------------------------------
    # Serper URL discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _get_serper_api_key(ctx: "AgentContext") -> str:
        """Resolve Serper API key from config or environment."""
        api_key = ""
        if ctx.config and hasattr(ctx.config, "serper") and ctx.config.serper:
            api_key = (ctx.config.serper.api_key or "").strip()
        if not api_key:
            api_key = os.getenv("SERPER_API_KEY", "").strip()
        return api_key

    def _generate_discovery_query(
        self,
        client: Any,
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
        domain: str,
    ) -> str:
        """Use LLM to generate a concise page-discovery search query.

        Instead of fragile regex heuristics, we ask the LLM to distil the
        market question into a short search query suitable for finding the
        *event page* (not the specific stat) on the target domain.

        This is a cheap, fast call (no tools, low token count) that
        generalises to any market type — sports, crypto, politics, etc.
        """
        import concurrent.futures
        from google.genai import types

        market = prompt_spec.market
        question = market.question or ""
        event_def = market.event_definition or ""
        entity = prompt_spec.prediction_semantics.target_entity or ""
        description = requirement.description or ""

        llm_prompt = (
            "You are a search query generator. Given a prediction market "
            "question, produce a SHORT Google search query (max 12 words) "
            "that will find the relevant EVENT PAGE on a specific website.\n\n"
            "Rules:\n"
            "- The query MUST start with: site:{domain}\n"
            "- Focus on entity names, dates, and event identifiers.\n"
            "- Do NOT include stat names, thresholds, or numbers like '5+' or 'over 3'.\n"
            "- Do NOT include words like 'will', 'did', 'shots', 'corners', 'goals', 'price', 'above'.\n"
            "- For sports: include team names and date.\n"
            "- For crypto/finance: include asset name and exchange.\n"
            "- For politics/events: include key entity and date.\n\n"
            "Examples:\n"
            "  Q: Will Real Madrid make 5+ shots outside box vs Osasuna on Feb 21?\n"
            "  Domain: fotmob.com\n"
            "  Query: site:fotmob.com Real Madrid vs Osasuna February 21 2026\n\n"
            "  Q: Will Arsenal have 5 or more corners vs Tottenham on Feb 22?\n"
            "  Domain: fotmob.com\n"
            "  Query: site:fotmob.com Arsenal vs Tottenham February 22 2026\n\n"
            "  Q: Will BTC be above $100k on CoinGecko by end of March 2026?\n"
            "  Domain: coingecko.com\n"
            "  Query: site:coingecko.com Bitcoin price March 2026\n\n"
            f"Q: {question}\n"
            f"Event: {event_def}\n"
            f"Entity: {entity}\n"
            f"Domain: {domain}\n"
            "Query:"
        ).format(domain=domain)

        def _do_call() -> str:
            resp = client.models.generate_content(
                model=self._model,
                contents=llm_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            text = ""
            for candidate in getattr(resp, "candidates", []):
                content = getattr(candidate, "content", None)
                if content:
                    for part in getattr(content, "parts", []):
                        t = getattr(part, "text", None)
                        if t:
                            text = t.strip()
                            break
            return text

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_call)
                query = future.result(timeout=15)  # fast call, 15s timeout

            # Sanitise: ensure it starts with site:domain
            query = query.strip().strip('"').strip("'")
            if not query.startswith(f"site:{domain}"):
                query = f"site:{domain} {query}"

            # Cap length
            if len(query) > 200:
                query = query[:200]

            return query
        except Exception:
            # Fallback: use the market question directly (stripped of punctuation)
            fallback = f"site:{domain} {entity} {question}"[:200]
            return fallback

    def _serper_discover_urls(
        self,
        ctx: "AgentContext",
        client: Any,
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
        required_domains: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Use Serper API to discover actual page URLs on required domains.

        The discovery query is generated by the LLM (via *client*) to
        ensure it generalises across market types without brittle regex.

        Returns a list of ``{"url": ..., "title": ..., "snippet": ...}``
        dicts, up to ``self._serper_max_urls`` entries.
        """
        api_key = self._get_serper_api_key(ctx)
        if not api_key:
            ctx.info("[GeminiGroundedStrict] No Serper API key — skipping URL discovery")
            return []
        if not ctx.http:
            ctx.info("[GeminiGroundedStrict] No HTTP client — skipping URL discovery")
            return []

        discovered: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for entry in required_domains:
            domain = entry["domain"]
            query = self._generate_discovery_query(
                client, prompt_spec, requirement, domain,
            )

            ctx.info(f"[GeminiGroundedStrict] Serper discovery: {query!r}")

            try:
                response = ctx.http.post(
                    _SERPER_URL,
                    headers={
                        "X-API-KEY": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query},
                    timeout=30.0,
                )

                if not response.ok:
                    ctx.warning(
                        f"[GeminiGroundedStrict] Serper HTTP {response.status_code}"
                    )
                    continue

                data = response.json()
                organic = data.get("organic", [])

                for result in organic:
                    url = result.get("link", "")
                    if not url or url in seen_urls:
                        continue
                    # Verify the URL belongs to the required domain
                    parsed = urlparse(url)
                    host = (parsed.netloc or "").lower().lstrip("www.")
                    if domain not in host and host not in domain:
                        continue
                    seen_urls.add(url)
                    discovered.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                    })
                    if len(discovered) >= self._serper_max_urls:
                        break

                ctx.info(
                    f"[GeminiGroundedStrict] Serper found {len(discovered)} URLs "
                    f"on {domain}"
                )

            except Exception as e:
                ctx.warning(f"[GeminiGroundedStrict] Serper error: {e}")

            if len(discovered) >= self._serper_max_urls:
                break

        return discovered

    # ------------------------------------------------------------------
    # FotMob direct extraction (Phase 1.5)
    # ------------------------------------------------------------------

    def _try_fotmob_direct_extraction(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
        discovered_urls: list[dict[str, str]],
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Try direct stat extraction from a FotMob match page.

        Returns ``(outcome, reason, metadata)`` on success, or ``None``
        if extraction is not possible or fails.
        """
        # Find a fotmob match URL among discovered URLs
        match_url = None
        for entry in discovered_urls:
            url = entry.get("url", "")
            if "fotmob.com/matches/" in url:
                match_url = url
                break

        if not match_url:
            return None

        if not ctx.http:
            ctx.info("[GeminiGroundedStrict] No HTTP client for fotmob extraction")
            return None

        ctx.info(f"[GeminiGroundedStrict] Attempting fotmob direct extraction: {match_url}")

        try:
            match_data = fotmob_fetch_match_stats(match_url, ctx.http)
        except FotMobExtractionError as e:
            ctx.warning(f"[GeminiGroundedStrict] FotMob extraction failed: {e}")
            return None

        # Determine which stat to look for
        stat_titles = requirement.expected_fields or []
        stat = None
        used_title = ""
        for title in stat_titles:
            stat = fotmob_find_stat(match_data, title)
            if stat:
                used_title = title
                break

        if not stat:
            ctx.info(
                f"[GeminiGroundedStrict] Stat not found in fotmob data "
                f"(tried: {stat_titles})"
            )
            return None

        # Determine which team's value to use
        target = prompt_spec.prediction_semantics.target_entity
        side = fotmob_match_team(match_data, target)
        if not side:
            ctx.info(
                f"[GeminiGroundedStrict] Could not match entity {target!r} "
                f"to {match_data.home_team} / {match_data.away_team}"
            )
            return None

        value = stat.home_value if side == "home" else stat.away_value

        # Resolve against threshold
        threshold_str = prompt_spec.prediction_semantics.threshold
        try:
            threshold = float(threshold_str) if threshold_str else None
        except (ValueError, TypeError):
            threshold = None

        if threshold is not None:
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                ctx.info(
                    f"[GeminiGroundedStrict] Non-numeric stat value: {value!r}"
                )
                return None
            outcome = "Yes" if numeric_value >= threshold else "No"
        else:
            outcome = "Yes"
            numeric_value = value

        team_name = match_data.home_team if side == "home" else match_data.away_team
        reason = (
            f"{team_name} had {value} {stat.title.lower()} "
            f"(vs threshold {threshold_str}). "
            f"Source: fotmob.com match page {match_url}"
        )

        metadata = {
            "direct_extraction": True,
            "fotmob_url": match_url,
            "fotmob_match_id": match_data.match_id,
            "fotmob_stat_title": stat.title,
            "fotmob_stat_key": stat.key,
            "fotmob_stat_value": value,
            "fotmob_home_team": match_data.home_team,
            "fotmob_away_team": match_data.away_team,
            "fotmob_home_value": stat.home_value,
            "fotmob_away_value": stat.away_value,
            "fotmob_target_side": side,
        }

        ctx.info(
            f"[GeminiGroundedStrict] FotMob direct extraction SUCCESS: "
            f"{stat.title} = {value} ({team_name}), outcome = {outcome}"
        )

        return outcome, reason, metadata

    # ------------------------------------------------------------------
    # Per-requirement collection (override)
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
            f"gemini_grounded_strict:{req_id}".encode()
        ).hexdigest()[:16]

        required_domains = _extract_required_domains(requirement)

        record = ToolCallRecord(
            tool="gemini_grounded_strict:search_and_resolve",
            input={
                "requirement_id": req_id,
                "model": self._model,
                "required_domains": [d["domain"] for d in required_domains],
                "max_attempts": self._max_attempts,
            },
            started_at=ctx.now().isoformat(),
        )

        # Fail fast if no source_targets are defined
        if not required_domains:
            record.ended_at = ctx.now().isoformat()
            record.error = "No source_targets defined on requirement"
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="gemini_grounded_strict",
                    source_uri=f"gemini:{self._model}",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=(
                    "CollectorGeminiGroundedStrict requires source_targets "
                    "with at least one domain. Use CollectorGeminiGrounded "
                    "for open-ended search."
                ),
            ), record

        try:
            # --- Phase 1: Serper URL discovery ---
            discovered_urls = self._serper_discover_urls(
                ctx, client, prompt_spec, requirement, required_domains,
            )
            record.input["discovered_urls"] = [u["url"] for u in discovered_urls]

            # --- Phase 1.5: FotMob direct extraction ---
            fotmob_result = self._try_fotmob_direct_extraction(
                ctx, prompt_spec, requirement, discovered_urls,
            )
            if fotmob_result is not None:
                outcome, reason, fotmob_meta = fotmob_result
                combined_text = json.dumps({"outcome": outcome, "reason": reason})
                record.ended_at = ctx.now().isoformat()
                record.output = {
                    "outcome": outcome,
                    "method": "fotmob_direct_extraction",
                    "data_source_covered": True,
                    "attempts_made": 0,
                    "strict_mode": True,
                    "serper_discovered_urls": [u["url"] for u in discovered_urls],
                    **fotmob_meta,
                }

                return EvidenceItem(
                    evidence_id=evidence_id,
                    requirement_id=req_id,
                    provenance=Provenance(
                        source_id="fotmob_direct",
                        source_uri=fotmob_meta["fotmob_url"],
                        tier=1,
                        fetched_at=ctx.now(),
                        content_hash=hashlib.sha256(combined_text.encode()).hexdigest(),
                    ),
                    raw_content=reason[:500],
                    parsed_value=outcome,
                    extracted_fields={
                        "outcome": outcome,
                        "reason": reason,
                        "evidence_sources": [{
                            "url": fotmob_meta["fotmob_url"],
                            "source_id": "fotmob.com",
                            "credibility_tier": 1,
                            "key_fact": reason,
                            "supports": outcome,
                            "is_required_data_source": True,
                        }],
                        "confidence_score": 0.95,
                        "resolution_status": "RESOLVED",
                        "data_source_covered": True,
                        "data_source_domains_required": [d["domain"] for d in required_domains],
                        "strict_mode": True,
                        "serper_discovered_urls": [u["url"] for u in discovered_urls],
                        **fotmob_meta,
                    },
                    success=True,
                ), record

            # --- Phase 2: Gemini with UrlContext + GoogleSearch ---
            strict_prompt = _build_strict_prompt(
                prompt_spec, requirement, required_domains,
                discovered_urls=discovered_urls or None,
            )

            best_parsed: dict[str, Any] | None = None
            best_grounding: dict[str, Any] | None = None
            best_sources_covered = False
            all_grounding_sources: list[dict[str, str]] = []
            all_search_queries: list[str] = []
            all_url_context_statuses: list[dict[str, str]] = []
            attempts_made = 0

            for attempt in range(1, self._max_attempts + 1):
                attempts_made = attempt
                ctx.info(
                    f"[GeminiGroundedStrict] Attempt {attempt}/{self._max_attempts} "
                    f"for {req_id} (domains: {[d['domain'] for d in required_domains]}, "
                    f"discovered_urls: {len(discovered_urls)})"
                )

                response = self._call_gemini_strict(
                    client, strict_prompt,
                    discovered_urls=discovered_urls or None,
                )
                text = self._extract_text(response)
                grounding = self._extract_grounding(response)
                url_ctx_meta = self._extract_url_context_metadata(response)
                parsed = self._parse_json(text)
                parsed = self._normalize_parsed(parsed)

                # Accumulate metadata across attempts
                all_grounding_sources.extend(grounding.get("sources", []))
                all_search_queries.extend(grounding.get("search_queries", []))
                all_url_context_statuses.extend(url_ctx_meta)

                # Combine grounding sources + UrlContext sources for coverage check
                combined_sources = list(grounding.get("sources", []))
                for ucm in url_ctx_meta:
                    if ucm.get("status") == "success":
                        combined_sources.append({"uri": ucm["url"], "title": ""})

                # Filter to only required-domain sources
                filtered_sources = self._filter_to_required_domains(
                    combined_sources, required_domains,
                )

                sources_covered = len(filtered_sources) > 0

                ctx.info(
                    f"[GeminiGroundedStrict] Attempt {attempt}: "
                    f"outcome={parsed.get('outcome')}, "
                    f"total_sources={len(grounding.get('sources', []))}, "
                    f"url_context_ok={sum(1 for u in url_ctx_meta if u.get('status') == 'success')}, "
                    f"required_domain_sources={len(filtered_sources)}, "
                    f"covered={sources_covered}"
                )

                # Keep the best result (prefer one with required-domain coverage)
                if sources_covered and not best_sources_covered:
                    best_parsed = parsed
                    best_grounding = grounding
                    best_sources_covered = True
                    ctx.info(
                        f"[GeminiGroundedStrict] Found required-domain evidence "
                        f"on attempt {attempt}!"
                    )
                    break  # Got what we need
                elif best_parsed is None:
                    best_parsed = parsed
                    best_grounding = grounding

            # Final result assembly
            final_parsed = best_parsed or {"outcome": "", "reason": "No result"}
            final_grounding = best_grounding or {"sources": [], "search_queries": []}

            # Include UrlContext-fetched URLs in grounding sources
            merged_grounding = dict(final_grounding)
            merged_sources = list(merged_grounding.get("sources", []))
            seen_uris = {s.get("uri", "") for s in merged_sources}
            for ucm in all_url_context_statuses:
                if ucm.get("status") == "success" and ucm["url"] not in seen_uris:
                    merged_sources.append({
                        "uri": ucm["url"],
                        "title": f"[UrlContext] {ucm['url']}",
                    })
                    seen_uris.add(ucm["url"])
            merged_grounding["sources"] = merged_sources

            # Only include evidence sources from required domains
            evidence_sources = self._build_strict_evidence_sources(
                merged_grounding, required_domains,
            )

            # If no required-domain sources found, include all sources
            # for transparency but mark them as non-authoritative
            if not evidence_sources:
                evidence_sources = self._build_strict_evidence_sources(
                    {"sources": all_grounding_sources}, required_domains,
                )

            outcome = final_parsed.get("outcome", "")
            reason = final_parsed.get("reason", "")
            combined_text = json.dumps(final_parsed)
            success = outcome.lower() in ("yes", "no")

            # Deduplicate search queries
            unique_queries = list(dict.fromkeys(all_search_queries))

            record.ended_at = ctx.now().isoformat()
            record.output = {
                "outcome": outcome,
                "grounding_sources": len(evidence_sources),
                "search_queries": unique_queries,
                "data_source_domains_required": [d["domain"] for d in required_domains],
                "data_source_covered": best_sources_covered,
                "attempts_made": attempts_made,
                "strict_mode": True,
                "serper_discovered_urls": [u["url"] for u in discovered_urls],
                "url_context_statuses": all_url_context_statuses,
            }

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="gemini_grounded_strict",
                    source_uri=f"gemini:{self._model}",
                    tier=1 if best_sources_covered else 2,
                    fetched_at=ctx.now(),
                    content_hash=hashlib.sha256(combined_text.encode()).hexdigest(),
                ),
                raw_content=reason[:500] if reason else combined_text[:500],
                parsed_value=outcome,
                extracted_fields={
                    "outcome": outcome,
                    "reason": reason,
                    "evidence_sources": evidence_sources,
                    "grounding_search_queries": unique_queries,
                    "grounding_source_count": len(evidence_sources),
                    "confidence_score": 0.9 if (success and best_sources_covered) else (0.5 if success else 0.0),
                    "resolution_status": "RESOLVED" if success else "UNRESOLVED",
                    "data_source_domains_required": [d["domain"] for d in required_domains],
                    "data_source_covered": best_sources_covered,
                    "attempts_made": attempts_made,
                    "strict_mode": True,
                    "serper_discovered_urls": [u["url"] for u in discovered_urls],
                    "url_context_statuses": all_url_context_statuses,
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
                    source_id="gemini_grounded_strict",
                    source_uri=f"gemini:{self._model}",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Gemini grounded strict call failed: {e}",
            ), record

    # ------------------------------------------------------------------
    # Strict helpers
    # ------------------------------------------------------------------

    def _call_gemini_strict(
        self,
        client: Any,
        prompt: str,
        *,
        discovered_urls: list[dict[str, str]] | None = None,
    ) -> Any:
        """Call Gemini with UrlContext + GoogleSearch tools.

        When *discovered_urls* is provided, the ``UrlContext`` tool is
        included so Gemini can ingest the full content of those pages.
        """
        import concurrent.futures
        from google.genai import types

        tools: list[types.Tool] = []

        # Include UrlContext when we have discovered URLs
        if discovered_urls:
            tools.append(types.Tool(url_context=types.UrlContext()))

        # Always include GoogleSearch for supplementary evidence
        tools.append(types.Tool(google_search=types.GoogleSearch()))

        def _do_call() -> Any:
            return client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STRICT_SYSTEM_PROMPT,
                    temperature=0.0,
                    tools=tools,
                ),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            return future.result(timeout=80)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from Gemini response, concatenating all text parts.

        Gemini-2.5-flash (thinking model) may return multiple text parts:
        the first part contains reasoning/explanation, and a later part
        contains the actual JSON answer.  The parent's ``_extract_text``
        returns only the first non-empty part, which misses the JSON.

        This override concatenates all text parts so ``_parse_json`` can
        find the ``{...}`` block regardless of which part it's in.
        """
        texts: list[str] = []
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        return "\n".join(texts)

    @staticmethod
    def _extract_url_context_metadata(response: Any) -> list[dict[str, str]]:
        """Extract UrlContext metadata from the Gemini response.

        Returns a list of ``{"url": ..., "status": "success"|"failed"|...}``.
        """
        results: list[dict[str, str]] = []
        for candidate in getattr(response, "candidates", []):
            meta = getattr(candidate, "url_context_metadata", None)
            if meta is None:
                continue
            for url_meta in getattr(meta, "url_metadata", []):
                url = getattr(url_meta, "retrieved_url", "")
                status_enum = getattr(url_meta, "url_retrieval_status", None)
                status = "unknown"
                if status_enum is not None:
                    status_str = str(status_enum)
                    if "SUCCESS" in status_str:
                        status = "success"
                    elif "FAILED" in status_str:
                        status = "failed"
                    elif "UNSAFE" in status_str:
                        status = "unsafe"
                    else:
                        status = status_str
                results.append({"url": url, "status": status})
        return results

    @staticmethod
    def _filter_to_required_domains(
        sources: list[dict[str, str]],
        required_domains: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Return only sources whose URIs match a required domain."""
        req_domain_set = {d["domain"] for d in required_domains}
        filtered: list[dict[str, str]] = []
        for src in sources:
            uri = src.get("uri", "")
            parsed = urlparse(uri)
            host = (parsed.netloc or "").lower().lstrip("www.")
            if any(rd in host or host in rd for rd in req_domain_set):
                filtered.append(src)
        return filtered

    @staticmethod
    def _build_strict_evidence_sources(
        grounding: dict[str, Any],
        required_domains: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Build evidence sources, only including required-domain matches."""
        req_domain_set = {d["domain"] for d in required_domains}
        evidence_sources: list[dict[str, Any]] = []
        seen_uris: set[str] = set()
        for src in grounding.get("sources", []):
            uri = src.get("uri", "")
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            parsed = urlparse(uri)
            host = (parsed.netloc or "").lower().lstrip("www.")
            is_required = any(
                rd in host or host in rd for rd in req_domain_set
            )
            evidence_sources.append({
                "url": uri,
                "source_id": src.get("title", "")[:50],
                "credibility_tier": 1 if is_required else 3,
                "key_fact": src.get("title", ""),
                "supports": "N/A",
                "date_published": None,
                "is_required_data_source": is_required,
            })
        return evidence_sources
