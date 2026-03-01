"""
Site-Pinned Collector Agent (CollectorSitePinned).

Searches ONLY within data-source domains specified in the requirement's
source_targets.  Uses Serper for URL discovery, structured extractors
for direct data extraction, and Gemini UrlContext + GoogleSearch as
fallback.  Retries up to 3 times.  Requires GOOGLE_API_KEY.

Uses a two-phase approach:
1. **Serper discovery** — find actual page URLs on the required domain
   via the Serper API (google.serper.dev).  This reliably surfaces
   JS-heavy sites (e.g. Fotmob) that Gemini's built-in grounding misses.
2. **Gemini UrlContext + GoogleSearch** — pass the discovered URLs to
   Gemini via the ``UrlContext`` tool so it can ingest the full page
   content, alongside ``GoogleSearch`` for supplementary evidence.

If no source_targets are defined on the requirement, this collector
will fail — use CollectorOpenSearch instead for open-ended search.

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
    CollectorOpenSearch,
)

from .extractors import find_extractor
from .extractors.base import ExtractionError

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
- You MUST NOT use evidence from any other website. If you cannot
  access or extract the required data from the required domain(s),
  say so clearly in the reason.
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

# ---------------------------------------------------------------------------
# Domain-specific prompt hints for Phase 2 (Gemini UrlContext)
# ---------------------------------------------------------------------------
# Sites behind Cloudflare JS challenges (e.g. FBRef) cannot use structured
# HTTP extraction — they fall through to Phase 2.  These hints tell Gemini
# what to look for when reading the page via UrlContext.

_DOMAIN_PROMPT_HINTS: dict[str, str] = {
    "hltv.org": (
        "DOMAIN HINT (hltv.org):\n"
        "You are reading HLTV pages. For weapon statistics markets, you MUST locate the player stats/weapon table. "
        "Look for a row labeled 'AWP' and extract the number of kills/frags with the AWP for the specified event stage (playoffs if specified). "
        "If you cannot find the weapon table on the provided URLs, say so.\n"
    ),
    "fbref.com": (
        "DOMAIN HINT (fbref.com):\n"
        "You are reading a FBRef.com match report page. Extract ALL match data "
        "into a structured text summary. Include:\n"
        "1. Score line: \"HomeTeam X - Y AwayTeam\"\n"
        "2. Player stats table: For each team, list every player with their "
        "key stats (minutes, goals, assists, shots, shots on target, passes, "
        "tackles, interceptions, fouls, cards).\n"
        "3. Shot events: If a shot/goal log is visible, list each shot with "
        "minute, player, team, outcome (Goal/Saved/Blocked/Off Target), and "
        "body part.\n"
        "4. Team totals: Possession, total shots, shots on target, corners, "
        "fouls, offsides, cards — for both teams.\n"
        "Look at ALL sections of the page, not just top-level stats.\n"
    ),
}


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

    # Domain-specific prompt hints
    domain_hints = ""
    for entry in required_domains:
        hint = _DOMAIN_PROMPT_HINTS.get(entry["domain"], "")
        if hint:
            domain_hints += f"\n{hint}\n"

    return (
        f"Market question: {market.question}\n\n"
        f"Resolution rules:\n{rules_text}\n\n"
        f"Data requirement: {requirement.description}\n\n"
        f"Target entity: {semantics.target_entity}\n"
        f"Predicate: {semantics.predicate}\n"
        f"Threshold: {semantics.threshold or 'N/A'}\n"
        f"{fields_hint}"
        f"{url_section}"
        f"{domain_hints}\n"
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

class CollectorSitePinned(CollectorOpenSearch):
    """Site-pinned collector.

    Searches ONLY within data-source domains specified in the
    requirement's ``source_targets``.  Uses Serper for URL discovery,
    structured extractors for direct data extraction, and Gemini
    UrlContext + GoogleSearch as fallback.  Retries up to 3 times.
    Requires GOOGLE_API_KEY.
    """

    _name = "CollectorSitePinned"
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
            ctx.info("[SourcePinned] No Serper API key — skipping URL discovery")
            return []
        if not ctx.http:
            ctx.info("[SourcePinned] No HTTP client — skipping URL discovery")
            return []

        discovered: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for entry in required_domains:
            domain = entry["domain"]
            query = self._generate_discovery_query(
                client, prompt_spec, requirement, domain,
            )

            ctx.info(f"[SourcePinned] Serper discovery: {query!r}")

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
                        f"[SourcePinned] Serper HTTP {response.status_code}"
                    )
                    continue

                data = response.json()
                organic = data.get("organic") or []

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
                    f"[SourcePinned] Serper found {len(discovered)} URLs "
                    f"on {domain}"
                )

            except Exception as e:
                ctx.warning(f"[SourcePinned] Serper error: {e}")

            if len(discovered) >= self._serper_max_urls:
                break

        return discovered

    # ------------------------------------------------------------------
    # Direct extraction via extractor registry (Phase 1.5)
    # ------------------------------------------------------------------

    def _try_direct_extraction(
        self,
        ctx: "AgentContext",
        client: Any,
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
        discovered_urls: list[dict[str, str]],
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Try LLM-based resolution using structured data from a site extractor.

        Loops through discovered URLs, finds a matching extractor, fetches
        structured data, builds a text summary, then asks Gemini to resolve.

        Returns ``(outcome, reason, metadata)`` on success, or ``None``.
        """
        # Find a matching extractor among discovered URLs
        extractor = None
        match_url = None
        for entry in discovered_urls:
            url = entry.get("url", "")
            ext = find_extractor(url)
            if ext is not None:
                extractor = ext
                match_url = url
                break

        if extractor is None or match_url is None:
            return None

        if not ctx.http:
            ctx.info("[SourcePinned] No HTTP client for direct extraction")
            return None

        import concurrent.futures
        from google.genai import types

        ctx.info(
            f"[SourcePinned] Attempting {extractor.source_id} "
            f"direct extraction: {match_url}"
        )

        try:
            data_summary, ext_metadata = extractor.extract_and_summarize(
                match_url, ctx.http,
                gemini_client=client,
                gemini_model=self._model,
            )
        except ExtractionError as e:
            ctx.warning(
                f"[SourcePinned] {extractor.source_id} extraction failed: {e}"
            )
            return None

        # If extractor returned a deterministic numeric stat, resolve
        # without another LLM call.
        if extractor.source_id == "hltv" and ext_metadata.get("awp_kills") is not None:
            awp_kills = ext_metadata.get("awp_kills")
            outcome = "Yes" if int(awp_kills) > 0 else "No"
            reason = (
                f"HLTV weapon stats show AWP kills/frags = {awp_kills} "
                f"on {match_url}."
            )
            metadata = {
                "direct_extraction": True,
                "resolution_method": "deterministic_structured_data",
                "extractor_source_id": extractor.source_id,
                **ext_metadata,
            }
            ctx.info(
                f"[SourcePinned] {extractor.source_id} deterministic resolution SUCCESS: "
                f"outcome={outcome}, awp_kills={awp_kills}"
            )
            return outcome, reason, metadata

        # Build LLM prompt (domain-agnostic)
        market = prompt_spec.market
        semantics = prompt_spec.prediction_semantics

        rules_lines: list[str] = []
        rules_lines.append(f"Event definition: {market.event_definition}")
        for rule in market.resolution_rules.get_sorted_rules():
            rules_lines.append(f"- [{rule.rule_id}] {rule.description}")
        rules_text = "\n".join(rules_lines)

        llm_prompt = (
            f"You are resolving a prediction market using official data "
            f"from {extractor.source_id}.\n\n"
            f"Market question: {market.question}\n\n"
            f"Resolution rules:\n{rules_text}\n\n"
            f"Target entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n"
            f"Threshold: {semantics.threshold or 'N/A'}\n\n"
            f"=== DATA (from {extractor.source_id}) ===\n"
            f"{data_summary}\n"
            f"=== END DATA ===\n\n"
            "Based ONLY on the data above, resolve this market.\n"
            "You MUST respond with ONLY a JSON object, no markdown fences:\n"
            '{"outcome": "Yes or No", "reason": "Brief explanation citing specific data"}\n'
        )

        ctx.info(
            f"[SourcePinned] Calling LLM for {extractor.source_id} "
            f"data interpretation"
        )

        def _do_call() -> Any:
            return client.models.generate_content(
                model=self._model,
                contents=llm_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_call)
                response = future.result(timeout=30)
        except Exception as e:
            ctx.warning(
                f"[SourcePinned] {extractor.source_id} LLM call failed: {e}"
            )
            return None

        text = self._extract_text(response)
        parsed = self._parse_json(text)
        parsed = self._normalize_parsed(parsed)

        outcome = parsed.get("outcome", "")
        reason = parsed.get("reason", "")

        if outcome.lower() not in ("yes", "no"):
            ctx.warning(
                f"[SourcePinned] {extractor.source_id} LLM returned "
                f"invalid outcome: {outcome!r}"
            )
            return None

        metadata = {
            "direct_extraction": True,
            "resolution_method": "llm_with_structured_data",
            "extractor_source_id": extractor.source_id,
            **ext_metadata,
        }

        ctx.info(
            f"[SourcePinned] {extractor.source_id} LLM resolution SUCCESS: "
            f"outcome={outcome}, reason={reason[:100]}"
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
            f"source_pinned:{req_id}".encode()
        ).hexdigest()[:16]

        required_domains = _extract_required_domains(requirement)

        record = ToolCallRecord(
            tool="source_pinned:search_and_resolve",
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
                    source_id="source_pinned",
                    source_uri=f"gemini:{self._model}",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=(
                    "CollectorSitePinned requires source_targets "
                    "with at least one domain. Use CollectorOpenSearch "
                    "for open-ended search."
                ),
            ), record

        try:
            # --- Phase 0: Seed discovered URLs with explicit source_targets ---
            # If the PromptSpec already includes concrete URLs on the required
            # domain(s), use them directly (no need to discover via Serper).
            seeded_urls: list[dict[str, str]] = []
            for target in getattr(requirement, "source_targets", []) or []:
                uri = getattr(target, "uri", "") or ""
                if not uri:
                    continue
                parsed = urlparse(uri)
                host = (parsed.netloc or "").lower().lstrip("www.")
                if not host:
                    continue
                if any(rd.get("domain") in host or host in rd.get("domain") for rd in required_domains):
                    seeded_urls.append({"url": uri, "title": "[source_target]", "snippet": ""})

            # --- Phase 1: Serper URL discovery (optional) ---
            serper_urls = self._serper_discover_urls(
                ctx, client, prompt_spec, requirement, required_domains,
            )

            def _is_deep_source_target(u: str) -> bool:
                try:
                    p = urlparse(u)
                    path = (p.path or "")
                    return any(seg in path for seg in ("/stats/", "/en/matches/", "/matches/")) or bool(p.query)
                except Exception:
                    return False

            deep_seeded = [u for u in seeded_urls if _is_deep_source_target(u.get("url", ""))]

            # Prefer Serper discovery for finding the right page, but always
            # prepend deep explicit source_target URLs (e.g. HLTV stats pages)
            # since they are often the canonical contract source.
            discovered_urls: list[dict[str, str]] = []
            if deep_seeded:
                discovered_urls.extend(deep_seeded)
            for u in (serper_urls or seeded_urls):
                if u.get("url") and all(u["url"] != x.get("url") for x in discovered_urls):
                    discovered_urls.append(u)
            record.input["discovered_urls"] = [u["url"] for u in discovered_urls]

            # --- Phase 1.5: Direct extraction via extractor registry ---
            extraction_result = self._try_direct_extraction(
                ctx, client, prompt_spec, requirement, discovered_urls,
            )
            if extraction_result is not None:
                outcome, reason, ext_meta = extraction_result
                source_id = ext_meta.get("extractor_source_id", "direct")
                # Find the URL used for extraction from metadata
                ext_url = (
                    ext_meta.get("fotmob_url")
                    or ext_meta.get("source_url")
                    or f"extractor:{source_id}"
                )
                combined_text = json.dumps({"outcome": outcome, "reason": reason})
                record.ended_at = ctx.now().isoformat()
                record.output = {
                    "outcome": outcome,
                    "method": f"{source_id}_direct_extraction",
                    "data_source_covered": True,
                    "attempts_made": 0,
                    "strict_mode": True,
                    "serper_discovered_urls": [u["url"] for u in discovered_urls],
                    **ext_meta,
                }

                return EvidenceItem(
                    evidence_id=evidence_id,
                    requirement_id=req_id,
                    provenance=Provenance(
                        source_id=f"{source_id}_direct",
                        source_uri=ext_url,
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
                            "url": ext_url,
                            "source_id": f"{source_id}.com",
                            "credibility_tier": 1,
                            "key_fact": reason,
                            "supports": outcome,
                            "is_required_data_source": True,
                        }],
                        "confidence_score": 0.90,
                        "resolution_status": "RESOLVED",
                        "data_source_covered": True,
                        "data_source_domains_required": [d["domain"] for d in required_domains],
                        "strict_mode": True,
                        "serper_discovered_urls": [u["url"] for u in discovered_urls],
                        **ext_meta,
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
                    f"[SourcePinned] Attempt {attempt}/{self._max_attempts} "
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

                try:
                    parsed = self._parse_json(text)
                    parsed = self._normalize_parsed(parsed)
                except Exception as e:
                    ctx.warning(
                        f"[SourcePinned] Attempt {attempt}: failed to parse JSON response ({type(e).__name__}: {e}). "
                        f"text_head={text[:80]!r}"
                    )
                    # Still accumulate metadata, then retry next attempt
                    continue

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
                    f"[SourcePinned] Attempt {attempt}: "
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
                        f"[SourcePinned] Found required-domain evidence "
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

            # If no required-domain sources were found, keep evidence_sources
            # empty and rely on data_source_covered=False. This prevents
            # accidental off-domain resolution.
            if not evidence_sources:
                evidence_sources = []

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
                    source_id="source_pinned",
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
            import traceback
            tb = traceback.format_exc()
            ctx.error(f"[SourcePinned] Exception for {req_id}: {tb}")
            record.ended_at = ctx.now().isoformat()
            # Keep error short in artifacts, full traceback in logs
            record.error = f"{type(e).__name__}: {e}"

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="source_pinned",
                    source_uri=f"gemini:{self._model}",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Gemini grounded strict call failed: {type(e).__name__}: {e}",
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
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in (getattr(content, "parts", None) or []):
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts)

        # Fallback: google-genai sometimes exposes a convenience `.text` field
        # even when candidate parts are empty.
        t = getattr(response, "text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()

        return ""

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
            for url_meta in (getattr(meta, "url_metadata", None) or []):
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
