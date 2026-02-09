"""
Gemini-Grounded Collector Agent.

Uses Google Gemini with built-in Google Search grounding to resolve
prediction market questions in a single LLM call.  Gemini searches
the web autonomously and returns a grounded answer with citations.

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

    return (
        f"Market question: {market.question}\n\n"
        f"Resolution rules:\n{rules_text}\n\n"
        f"Data requirement: {requirement.description}\n\n"
        f"Target entity: {semantics.target_entity}\n"
        f"Predicate: {semantics.predicate}\n"
        f"Threshold: {semantics.threshold or 'N/A'}\n\n"
        f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Search the web and resolve this market. Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Gemini-Grounded Collector
# ---------------------------------------------------------------------------

class CollectorGeminiGrounded(BaseAgent):
    """Collector that uses Gemini with Google Search grounding.

    A single Gemini call with ``GoogleSearch`` tool enabled handles both
    web retrieval and evidence synthesis.  Grounding metadata (URLs,
    titles, support segments) is extracted into the EvidenceItem.

    Capabilities: LLM + NETWORK (Gemini performs the network search).
    """

    _name = "CollectorGeminiGrounded"
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
        ctx.info(f"CollectorGeminiGrounded executing plan {tool_plan.plan_id} "
                 f"[model={self._model}]")

        api_key = self._resolve_api_key(ctx)
        if not api_key:
            return AgentResult.failure(
                error="Google API key not available. "
                      "Set GOOGLE_API_KEY or configure llm.api_key with provider=google.",
            )

        client = self._get_client(api_key)

        bundle = EvidenceBundle(
            bundle_id=f"gemini_grounded_{tool_plan.plan_id}",
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
                "collector": "gemini_grounded",
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
            f"gemini_grounded:{req_id}".encode()
        ).hexdigest()[:16]

        user_prompt = _build_user_prompt(prompt_spec, requirement)

        record = ToolCallRecord(
            tool="gemini_grounded:search_and_resolve",
            input={"requirement_id": req_id, "model": self._model},
            started_at=ctx.now().isoformat(),
        )

        try:
            print(f"---------------call gemini with user prompt: {user_prompt}")
            response = self._call_gemini(client, user_prompt)
            print(f"---------------gemini response: {response}")
            text = self._extract_text(response)
            print(f"---------------gemini text: {text}")
            grounding = self._extract_grounding(response)
            parsed = self._parse_json(text)
            parsed = self._normalize_parsed(parsed)

            print(f"---------------gemini grounding: {grounding}")
            print(f"---------------gemini parsed: {parsed}")
            record.ended_at = ctx.now().isoformat()
            record.output = {
                "outcome": parsed.get("outcome"),
                "grounding_sources": len(grounding.get("sources", [])),
                "search_queries": grounding.get("search_queries", []),
            }

            # Build evidence sources from grounding metadata
            evidence_sources = []
            for src in grounding.get("sources", []):
                evidence_sources.append({
                    "url": src.get("uri", ""),
                    "source_id": src.get("title", "")[:50],
                    "credibility_tier": 2,
                    "key_fact": src.get("title", ""),
                    "supports": "N/A",
                    "date_published": None,
                })

            outcome = parsed.get("outcome", "")
            reason = parsed.get("reason", "")
            success = outcome.lower() in ("yes", "no")

            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="gemini_grounded",
                    source_uri=f"gemini:{self._model}",
                    tier=2,
                    fetched_at=ctx.now(),
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                ),
                raw_content=reason[:500] if reason else text[:500],
                parsed_value=outcome,
                extracted_fields={
                    "outcome": outcome,
                    "reason": reason,
                    "evidence_sources": evidence_sources,
                    "grounding_search_queries": grounding.get("search_queries", []),
                    "grounding_source_count": len(evidence_sources),
                    "confidence_score": 0.9 if success else 0.0,
                    "resolution_status": "RESOLVED" if success else "UNRESOLVED",
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
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )
        return genai.Client(api_key=api_key)

    def _call_gemini(self, client: Any, prompt: str) -> Any:
        """Call Gemini with Google Search grounding enabled."""
        from google.genai import types

        return client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_GROUNDED_SYSTEM_PROMPT,
                temperature=0.0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

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
        """Parse JSON from Gemini response text, handling markdown fences."""
        text = text.strip()

        # Try stripping markdown code block (```json ... ```)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            return json.loads(fence_match.group(1).strip())

        # Try raw {…} extraction
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])

        return json.loads(text)

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
