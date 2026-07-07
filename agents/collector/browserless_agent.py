"""
Browserless Collector Agent (CollectorBrowserless).

Two-phase collector:
1. **Serper discovery** — find page URLs on required domains.
2. **Browserless BQL** — fetch each page via stealth browser with
   Cloudflare solver, then resolve with LLM.

No direct extraction, no Jina Reader, no Gemini UrlContext/GoogleSearch.

Requires: ``BROWSERLESS_TOKEN`` environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

from agents.base import AgentCapability
from core.schemas import (
    EvidenceItem,
    Provenance,
    ToolCallRecord,
)
from .source_pinned_agent import (
    CollectorSitePinned,
    _CLOUDFLARE_MARKERS,
    _extract_required_domains,
    _SERPER_URL,
)

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas import DataRequirement, PromptSpec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROWSERLESS_URL = "https://production-sfo.browserless.io/stealth/bql"
_BROWSERLESS_TIMEOUT = 75.0  # fotmob-like SPAs need ~35s (30s goto timeout + solve + wait)


# ---------------------------------------------------------------------------
# CollectorBrowserless
# ---------------------------------------------------------------------------


class CollectorBrowserless(CollectorSitePinned):
    """Site-pinned collector using Browserless BQL only.

    Two-phase pipeline:
    1. Serper URL discovery on required domains.
    2. Browserless BQL stealth fetch + Cloudflare solver → LLM resolution.

    No direct extraction, no Jina Reader, no Gemini UrlContext/GoogleSearch.
    """

    _name = "CollectorBrowserless"
    _version = "v1"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}

    # ------------------------------------------------------------------
    # Discovery query override — better queries for match-report pages
    # ------------------------------------------------------------------

    def _generate_discovery_query(
        self,
        client: Any,
        prompt_spec: "PromptSpec",
        requirement: "DataRequirement",
        domain: str,
    ) -> str:
        """Generate a search query tuned for match-report / result pages.

        Overrides the parent to:
        1. Include ISO date format (YYYY-MM-DD) — fbref, ESPN, etc. index
           match reports by ISO date in the URL slug.
        2. Add "match report" keywords to steer Serper toward per-match
           pages instead of head-to-head history or team overview pages.
        """
        import concurrent.futures
        from google.genai import types

        market = prompt_spec.market
        question = market.question or ""
        event_def = market.event_definition or ""
        entity = prompt_spec.prediction_semantics.target_entity or ""

        llm_prompt = (
            "You are a search query generator. Given a prediction market "
            "question, produce a SHORT Google search query (max 16 words) "
            "that will find the specific MATCH REPORT or RESULT PAGE on a "
            "website.\n\n"
            "Rules:\n"
            "- The query MUST start with: site:{domain}\n"
            "- Focus on entity names, dates, and event identifiers.\n"
            "- Do NOT include stat names, thresholds, or numbers like "
            "'5+' or 'over 3'.\n"
            "- Do NOT include words like 'will', 'did', 'shots', "
            "'corners', 'goals', 'price', 'above'.\n"
            "- For sports: include BOTH the ISO date (YYYY-MM-DD) AND "
            "team names. Also add 'match report' to target per-match pages "
            "rather than head-to-head history or team profiles.\n"
            "- For crypto/finance: include asset name and exchange.\n"
            "- For politics/events: include key entity and date.\n\n"
            "Examples:\n"
            "  Q: Will Real Madrid make 5+ shots outside box vs Osasuna "
            "on Feb 21 2026?\n"
            "  Domain: fbref.com\n"
            "  Query: site:fbref.com Real Madrid Osasuna 2026-02-21 "
            "match report\n\n"
            "  Q: Will Arsenal have 5 or more corners vs Tottenham on "
            "Feb 22 2026?\n"
            "  Domain: fotmob.com\n"
            "  Query: site:fotmob.com Arsenal vs Tottenham 2026-02-22 "
            "match report\n\n"
            "  Q: Will Callum Hudson-Odoi be in the starting lineup "
            "for Nott'ham Forest vs Liverpool on March 15 2026?\n"
            "  Domain: fbref.com\n"
            "  Query: site:fbref.com Nottingham Forest Liverpool "
            "2026-03-15 lineup\n\n"
            "  Q: Will BTC be above $100k on CoinGecko by end of "
            "March 2026?\n"
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
                config=types.GenerateContentConfig(temperature=0.0),
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
                query = future.result(timeout=15)

            query = query.strip().strip('"').strip("'")
            if not query.startswith(f"site:{domain}"):
                query = f"site:{domain} {query}"

            if len(query) > 200:
                query = query[:200]

            return query
        except Exception:
            fallback = f"site:{domain} {entity} {question}"[:200]
            return fallback

    # ------------------------------------------------------------------
    # Browserless BQL fallback (Phase 1.75 override)
    # ------------------------------------------------------------------

    def _try_browserless_fallback(
        self,
        ctx: "AgentContext",
        client: Any,
        prompt_spec: "PromptSpec",
        requirement: "DataRequirement",
        discovered_urls: list[dict[str, str]],
    ) -> tuple[tuple[str, str, dict[str, Any]] | None, list[tuple[str, str]]]:
        """Fetch page content via Browserless BQL and resolve with LLM.

        Called when structured extractors fail (Phase 1.5 miss).
        Browserless BQL stealth mode renders JS and solves Cloudflare
        challenges, returning the page body text for LLM consumption.

        Returns ``(result, unresolved_reasons)`` where *result* is
        ``(outcome, reason, metadata)`` on success or ``None``, and
        *unresolved_reasons* is a list of ``(url, reason)`` for URLs
        where the LLM returned Unresolved.
        """
        unresolved_reasons: list[tuple[str, str]] = []

        if not ctx.http:
            return None, unresolved_reasons
        if not discovered_urls:
            return None, unresolved_reasons

        token = os.getenv("BROWSERLESS_TOKEN", "").strip()
        if not token:
            ctx.warning("[Browserless] No BROWSERLESS_TOKEN — skipping fallback")
            return None, unresolved_reasons

        import concurrent.futures
        import requests as _requests
        from google.genai import types

        bql_url = f"{_BROWSERLESS_URL}?token={token}"

        # Use a dedicated session with optional proxy for Browserless
        proxy = os.getenv("BROWSERLESS_PROXY", "").strip()
        session = _requests.Session()
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
            ctx.info(f"[Browserless] Using proxy: {proxy}")

        # Build shared LLM prompt parts once
        market = prompt_spec.market
        semantics = prompt_spec.prediction_semantics

        rules_lines: list[str] = []
        rules_lines.append(f"Event definition: {market.event_definition}")
        for rule in market.resolution_rules.get_sorted_rules():
            rules_lines.append(f"- [{rule.rule_id}] {rule.description}")
        rules_text = "\n".join(rules_lines)

        # Try each discovered URL: fetch via BQL → LLM resolve → next if Unresolved
        for entry in discovered_urls:
            url = entry.get("url", "")
            if not url:
                continue

            ctx.info(f"[Browserless] BQL fallback: {url}")

            bql_payload = {
                "query": (
                    'mutation Scrape($url: String!) {\n'
                    '  goto(url: $url, waitUntil: networkIdle) {\n'
                    '    status\n'
                    '    url\n'
                    '  }\n'
                    '  solve(type: cloudflare) {\n'
                    '    found\n'
                    '    solved\n'
                    '  }\n'
                    '  waitForTimeout(time: 3000) {\n'
                    '    time\n'
                    '  }\n'
                    '  text(selector: "body") {\n'
                    '    text\n'
                    '  }\n'
                    '}'
                ),
                "variables": {"url": url},
            }

            # --- Fetch page via BQL ---
            try:
                resp = session.post(
                    bql_url,
                    json=bql_payload,
                    timeout=_BROWSERLESS_TIMEOUT,
                )
                if resp.status_code >= 400:
                    ctx.warning(
                        f"[Browserless] HTTP {resp.status_code} for {url}"
                    )
                    continue

                data = resp.json()
                body_data = data.get("data", {}).get("text", {})
                body = (body_data.get("text", "") if isinstance(body_data, dict) else "").strip()

                if not body or len(body) < 50:
                    ctx.warning(
                        f"[Browserless] Too little content "
                        f"({len(body)} chars) for {url}"
                    )
                    continue

                # Check for Cloudflare markers
                body_lower = body[:4000].lower()
                is_cf = any(m.lower() in body_lower for m in _CLOUDFLARE_MARKERS)
                if is_cf:
                    ctx.warning(
                        f"[Browserless] Content still shows Cloudflare "
                        f"challenge for {url}"
                    )
                    continue

                ctx.info(
                    f"[Browserless] Fetched {len(body)} chars from {url}"
                )

            except Exception as e:
                ctx.warning(f"[Browserless] Error for {url}: {type(e).__name__}: {e}")
                continue

            # --- Truncate and send to LLM ---
            page_content = body
            max_content = 12_000
            if len(page_content) > max_content:
                page_content = page_content[:max_content] + "\n\n[... content truncated ...]"

            llm_prompt = (
                f"You are resolving a prediction market using data fetched "
                f"from {url}.\n\n"
                f"Market question: {market.question}\n\n"
                f"Resolution rules:\n{rules_text}\n\n"
                f"Target entity: {semantics.target_entity}\n"
                f"Predicate: {semantics.predicate}\n"
                f"Threshold: {semantics.threshold or 'N/A'}\n\n"
                f"=== PAGE CONTENT (from {url} via Browserless) ===\n"
                f"{page_content}\n"
                f"=== END PAGE CONTENT ===\n\n"
                "Based ONLY on the page content above, resolve this market.\n"
                "You MUST respond with ONLY a JSON object, no markdown fences:\n"
                '{"outcome": "Yes or No or Unresolved", "reason": "Brief explanation citing specific data"}\n'
                "Use 'Unresolved' if the page content does not contain enough information to determine the answer.\n"
            )

            ctx.info(
                f"[Browserless] Calling LLM with Browserless content "
                f"for {url}"
            )

            def _do_call(_prompt: str = llm_prompt) -> Any:
                return client.models.generate_content(
                    model=self._model,
                    contents=_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                    ),
                )

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_do_call)
                    response = future.result(timeout=30)
            except Exception as e:
                ctx.warning(f"[Browserless] LLM call failed for {url}: {e}")
                continue

            text = self._extract_text(response)
            parsed = self._parse_json(text)
            parsed = self._normalize_parsed(parsed)

            outcome = parsed.get("outcome", "")
            reason = parsed.get("reason", "")

            if outcome.lower() not in ("yes", "no", "unresolved"):
                ctx.warning(
                    f"[Browserless] LLM returned invalid outcome: {outcome!r}"
                )
                continue

            if outcome.lower() == "unresolved":
                ctx.info(
                    f"[Browserless] LLM returned Unresolved for {url} "
                    f"— trying next URL"
                )
                unresolved_reasons.append((url, reason))
                continue

            # --- Success ---
            metadata = {
                "direct_extraction": True,
                "resolution_method": "llm_with_browserless",
                "browserless_url": url,
                "browserless_content_length": len(page_content),
            }

            ctx.info(
                f"[Browserless] LLM resolution SUCCESS: "
                f"outcome={outcome}, reason={reason[:100]}"
            )

            return (outcome, reason, metadata), unresolved_reasons

        # All URLs exhausted
        return None, unresolved_reasons

    # ------------------------------------------------------------------
    # Serper search helper (reusable for retries)
    # ------------------------------------------------------------------

    def _serper_search(
        self,
        ctx: "AgentContext",
        api_key: str,
        query: str,
        domain: str,
        exclude_urls: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Run a single Serper search and return matching URLs.

        Returns a list of ``{"url": ..., "title": ..., "snippet": ...}``
        filtered to *domain* and excluding *exclude_urls*.
        """
        if not ctx.http:
            return []

        exclude = exclude_urls or set()
        results: list[dict[str, str]] = []

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
                ctx.warning(f"[Browserless] Serper HTTP {response.status_code}")
                return []

            data = response.json()
            for item in data.get("organic", []):
                url = item.get("link", "")
                if not url or url in exclude:
                    continue
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower().lstrip("www.")
                if domain not in host and host not in domain:
                    continue
                results.append({
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                })
                if len(results) >= self._serper_max_urls:
                    break

        except Exception as e:
            ctx.warning(f"[Browserless] Serper error: {e}")

        return results

    # ------------------------------------------------------------------
    # Refined query generation for retry round
    # ------------------------------------------------------------------

    def _generate_refined_query(
        self,
        client: Any,
        prompt_spec: "PromptSpec",
        requirement: "DataRequirement",
        domain: str,
        tried_urls: list[str],
        unresolved_reasons: list[tuple[str, str]],
    ) -> str:
        """Ask the LLM to produce a better search query after initial failure.

        Feeds back the URLs already tried and the reasons they were
        rejected so the LLM can steer toward a different page type.
        """
        import concurrent.futures
        from google.genai import types

        market = prompt_spec.market
        question = market.question or ""
        entity = prompt_spec.prediction_semantics.target_entity or ""

        reasons_block = "\n".join(
            f"  - {url}: {reason}" for url, reason in unresolved_reasons
        ) or "  (no specific reasons captured)"

        tried_block = "\n".join(f"  - {u}" for u in tried_urls) or "  (none)"

        llm_prompt = (
            "You are a search query generator. A previous search failed "
            "to find the right page. Given the feedback below, produce a "
            "BETTER Google search query (max 16 words) to find the "
            "specific match report or result page.\n\n"
            f"Market question: {question}\n"
            f"Entity: {entity}\n"
            f"Domain: {domain}\n\n"
            f"Previously tried URLs and why they failed:\n{reasons_block}\n\n"
            f"All URLs already tried (do NOT find these again):\n{tried_block}\n\n"
            "Rules:\n"
            "- The query MUST start with: site:{domain}\n"
            "- Try a DIFFERENT angle: different date format, different "
            "keywords, or more specific page type.\n"
            "- For sports: use ISO date (YYYY-MM-DD), try 'match report' "
            "or 'lineups' or 'stats' instead of generic team names.\n"
            "- Avoid keywords that led to the wrong pages above.\n\n"
            "Query:"
        ).format(domain=domain)

        def _do_call() -> str:
            resp = client.models.generate_content(
                model=self._model,
                contents=llm_prompt,
                config=types.GenerateContentConfig(temperature=0.3),
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
                query = future.result(timeout=15)

            query = query.strip().strip('"').strip("'")
            if not query.startswith(f"site:{domain}"):
                query = f"site:{domain} {query}"

            if len(query) > 200:
                query = query[:200]

            return query
        except Exception:
            fallback = f"site:{domain} {entity} match report"[:200]
            return fallback

    # ------------------------------------------------------------------
    # Per-requirement collection (full override)
    # Phases: Serper discovery → Browserless BQL → retry → done.
    # ------------------------------------------------------------------

    def _collect_requirement(
        self,
        ctx: "AgentContext",
        client: Any,
        prompt_spec: "PromptSpec",
        requirement: "DataRequirement",
        quality_feedback: dict[str, Any] | None = None,
    ) -> tuple[EvidenceItem, ToolCallRecord]:
        req_id = requirement.requirement_id
        evidence_id = hashlib.sha256(
            f"browserless:{req_id}".encode()
        ).hexdigest()[:16]

        required_domains = _extract_required_domains(requirement)

        record = ToolCallRecord(
            tool="browserless:search_and_resolve",
            input={
                "requirement_id": req_id,
                "model": self._model,
                "required_domains": [d["domain"] for d in required_domains],
            },
            started_at=ctx.now().isoformat(),
        )

        # Fail fast if no source_targets
        if not required_domains:
            record.ended_at = ctx.now().isoformat()
            record.error = "No source_targets defined on requirement"
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="browserless",
                    source_uri="browserless",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=(
                    "CollectorBrowserless requires source_targets "
                    "with at least one domain."
                ),
            ), record

        try:
            all_discovered_urls: list[dict[str, str]] = []
            all_unresolved_reasons: list[tuple[str, str]] = []

            # --- Round 1: Serper URL discovery → Browserless BQL ---
            discovered_urls = self._serper_discover_urls(
                ctx, client, prompt_spec, requirement, required_domains,
            )
            all_discovered_urls.extend(discovered_urls)
            record.input["discovered_urls"] = [u["url"] for u in discovered_urls]

            bl_result, unresolved = self._try_browserless_fallback(
                ctx, client, prompt_spec, requirement, discovered_urls,
            )
            all_unresolved_reasons.extend(unresolved)

            if bl_result is not None:
                outcome, reason, bl_meta = bl_result
                bl_url = bl_meta.get("browserless_url", "browserless")
                combined_text = json.dumps({"outcome": outcome, "reason": reason})
                record.ended_at = ctx.now().isoformat()
                record.output = {
                    "outcome": outcome,
                    "method": "browserless_bql",
                    "data_source_covered": True,
                    "strict_mode": True,
                    "serper_discovered_urls": [u["url"] for u in all_discovered_urls],
                    **bl_meta,
                }

                bl_host = (urlparse(bl_url).netloc or "").lower().lstrip("www.")
                return EvidenceItem(
                    evidence_id=evidence_id,
                    requirement_id=req_id,
                    provenance=Provenance(
                        source_id="browserless",
                        source_uri=bl_url,
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
                            "url": bl_url,
                            "source_id": "[1]",
                            "domain_name": bl_host or None,
                            "credibility_tier": 1,
                            "key_fact": reason,
                            "supports": outcome,
                            "date_published": None,
                            "is_required_data_source": True,
                        }],
                        "grounding_search_queries": [],
                        "grounding_source_count": 1,
                        "confidence_score": 0.85,
                        "resolution_status": "RESOLVED",
                        "pass_used": "browserless_bql",
                        "data_source_covered": True,
                        "data_source_domains_required": [d["domain"] for d in required_domains],
                        "strict_mode": True,
                        "serper_discovered_urls": [u["url"] for u in all_discovered_urls],
                        **bl_meta,
                    },
                    success=True,
                ), record

            # --- Round 2: Retry with refined query ---
            if all_unresolved_reasons:
                tried_urls = {u["url"] for u in all_discovered_urls}
                api_key = self._get_serper_api_key(ctx)

                for entry in required_domains:
                    domain = entry["domain"]
                    refined_query = self._generate_refined_query(
                        client, prompt_spec, requirement, domain,
                        list(tried_urls), all_unresolved_reasons,
                    )
                    ctx.info(
                        f"[Browserless] Retry with refined query: "
                        f"{refined_query!r}"
                    )

                    if api_key:
                        new_urls = self._serper_search(
                            ctx, api_key, refined_query, domain,
                            exclude_urls=tried_urls,
                        )
                    else:
                        new_urls = []

                    if not new_urls:
                        ctx.info(
                            f"[Browserless] No new URLs from retry on {domain}"
                        )
                        continue

                    all_discovered_urls.extend(new_urls)
                    record.input["retry_urls"] = [u["url"] for u in new_urls]

                    ctx.info(
                        f"[Browserless] Retry found {len(new_urls)} new URL(s) "
                        f"on {domain}"
                    )

                    bl_result_retry, retry_unresolved = (
                        self._try_browserless_fallback(
                            ctx, client, prompt_spec, requirement, new_urls,
                        )
                    )
                    all_unresolved_reasons.extend(retry_unresolved)

                    if bl_result_retry is not None:
                        outcome, reason, bl_meta = bl_result_retry
                        bl_url = bl_meta.get("browserless_url", "browserless")
                        bl_meta["retry_round"] = 2
                        combined_text = json.dumps({
                            "outcome": outcome, "reason": reason,
                        })
                        record.ended_at = ctx.now().isoformat()
                        record.output = {
                            "outcome": outcome,
                            "method": "browserless_bql",
                            "data_source_covered": True,
                            "strict_mode": True,
                            "serper_discovered_urls": [
                                u["url"] for u in all_discovered_urls
                            ],
                            **bl_meta,
                        }

                        retry_host = (urlparse(bl_url).netloc or "").lower().lstrip("www.")
                        return EvidenceItem(
                            evidence_id=evidence_id,
                            requirement_id=req_id,
                            provenance=Provenance(
                                source_id="browserless",
                                source_uri=bl_url,
                                tier=1,
                                fetched_at=ctx.now(),
                                content_hash=hashlib.sha256(
                                    combined_text.encode()
                                ).hexdigest(),
                            ),
                            raw_content=reason[:500],
                            parsed_value=outcome,
                            extracted_fields={
                                "outcome": outcome,
                                "reason": reason,
                                "evidence_sources": [{
                                    "url": bl_url,
                                    "source_id": "[1]",
                                    "domain_name": retry_host or None,
                                    "credibility_tier": 1,
                                    "key_fact": reason,
                                    "supports": outcome,
                                    "date_published": None,
                                    "is_required_data_source": True,
                                }],
                                "grounding_search_queries": [],
                                "grounding_source_count": 1,
                                "confidence_score": 0.85,
                                "resolution_status": "RESOLVED",
                                "pass_used": "browserless_bql_retry",
                                "data_source_covered": True,
                                "data_source_domains_required": [
                                    d["domain"] for d in required_domains
                                ],
                                "strict_mode": True,
                                "serper_discovered_urls": [
                                    u["url"] for u in all_discovered_urls
                                ],
                                **bl_meta,
                            },
                            success=True,
                        ), record

            # --- Both rounds failed — return Unresolved ---
            unresolved_reason = (
                f"Identified {len(all_discovered_urls)} URL(s) on required "
                f"domains across 2 rounds but could not extract usable content."
            )
            combined_text = json.dumps({
                "outcome": "Unresolved", "reason": unresolved_reason,
            })
            record.ended_at = ctx.now().isoformat()
            record.output = {
                "outcome": "Unresolved",
                "method": "browserless_bql",
                "data_source_covered": False,
                "serper_discovered_urls": [u["url"] for u in all_discovered_urls],
            }

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="browserless",
                    source_uri="browserless",
                    tier=2,
                    fetched_at=ctx.now(),
                    content_hash=hashlib.sha256(combined_text.encode()).hexdigest(),
                ),
                raw_content=unresolved_reason[:500],
                parsed_value=None,
                extracted_fields={
                    "outcome": "Unresolved",
                    "reason": unresolved_reason,
                    "evidence_sources": [],
                    "grounding_search_queries": [],
                    "grounding_source_count": 0,
                    "confidence_score": 0.0,
                    "resolution_status": "UNRESOLVED",
                    "pass_used": "browserless_bql",
                    "data_source_covered": False,
                    "data_source_domains_required": [d["domain"] for d in required_domains],
                    "serper_discovered_urls": [u["url"] for u in all_discovered_urls],
                },
                success=True,
                error=None,
            ), record

        except Exception as e:
            record.ended_at = ctx.now().isoformat()
            record.error = str(e)
            discovered_raw = record.input.get("discovered_urls", [])
            crash_reason = (
                f"Identified {len(discovered_raw)} URL(s) but failed: {e}"
            )
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="browserless",
                    source_uri="browserless",
                    tier=0,
                    fetched_at=ctx.now(),
                ),
                raw_content=crash_reason[:500],
                extracted_fields={
                    "outcome": "Unresolved",
                    "reason": crash_reason,
                    "evidence_sources": [],
                    "grounding_search_queries": [],
                    "grounding_source_count": 0,
                    "confidence_score": 0.0,
                    "resolution_status": "UNRESOLVED",
                    "pass_used": "browserless_bql",
                    "data_source_covered": False,
                    "data_source_domains_required": [],
                    "serper_discovered_urls": discovered_raw,
                },
                success=True,
                error=None,
            ), record
