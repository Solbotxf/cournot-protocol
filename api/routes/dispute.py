"""Dispute Route

POST /dispute      — structured dispute endpoint (full manual control)
POST /dispute/llm  — LLM-assisted dispute (3 user inputs → auto-structured)

Goal: allow *stateless* dispute-driven reruns of downstream pipeline steps.

Design notes (dashboard-driven MVP):
- Frontend must provide all necessary context (prompt_spec, evidence_bundle, and
  optionally reasoning_trace) so the protocol does not look up historical runs.
- Dispute context is injected into Auditor/Judge LLM prompts via ctx.extra["dispute_context"].
- MVP supports disputing judge reasoning and evidence interpretation.

This endpoint is intentionally conservative:
- It does NOT persist disputes.
- It does NOT fetch or look up artifacts from storage.

"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from html.parser import HTMLParser
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import get_agent_context
from api.errors import InvalidRequestError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dispute"])


# -----------------------------
# Schemas
# -----------------------------

ReasonCode = Literal[
    # Judge-focused
    "REASONING_ERROR",
    "LOGIC_GAP",
    # Evidence-focused
    "EVIDENCE_MISREAD",
    "EVIDENCE_INSUFFICIENT",
    # Generic
    "OTHER",
]


class DisputeTarget(BaseModel):
    """UI hint for what is being disputed.

    The backend does not hard-depend on this, but it is included in the LLM
    context injection to make the rerun dispute-aware.
    """

    artifact: Literal["evidence_bundle", "reasoning_trace", "verdict", "prompt_spec"] = Field(
        ..., description="Which artifact is disputed"
    )
    leaf_path: str | None = Field(
        default=None, description="Optional JSONPath-like pointer for the disputed leaf"
    )


class DisputePatch(BaseModel):
    """Optional patch/override operations.

    MVP supports:

    - evidence_items_append: list of EvidenceItem JSON objects to append to the
      provided evidence_bundle.items before rerun.

    - prompt_spec_override: partial PromptSpec JSON to deep-merge into the
      provided prompt_spec before rerun (stateless).

    Merge strategy:
    - dicts: deep-merge
    - lists: replace wholesale (frontend should include any items it wants to keep)
    """

    evidence_items_append: list[dict[str, Any]] | None = None
    prompt_spec_override: dict[str, Any] | None = None


class DisputeRequest(BaseModel):
    # Optional dashboard correlation only (stateless: no lookup)
    case_id: str | None = Field(default=None)

    reason_code: ReasonCode
    message: str = Field(..., min_length=1, max_length=8000)

    target: DisputeTarget | None = None

    # Full context artifacts (stateless contract)
    prompt_spec: dict[str, Any]
    evidence_bundle: dict[str, Any] | None = None

    # Optional: if present, allow judge-only rerun
    reasoning_trace: dict[str, Any] | None = None

    # Required for EVIDENCE_MISREAD (re-run original collectors)
    tool_plan: dict[str, Any] | None = None
    collectors: list[str] | None = None

    # Required for EVIDENCE_INSUFFICIENT (auto-collect from URLs/domains)
    evidence_urls: list[str] | None = Field(default=None, max_length=10)

    # Optional patch operations (e.g., append evidence items, prompt_spec_override)
    patch: DisputePatch | dict[str, Any] | None = None


class DisputeResponse(BaseModel):
    ok: Literal[True] = True
    case_id: str | None = None
    rerun_plan: list[str]
    artifacts: dict[str, Any]
    diff: dict[str, Any] | None = None


class DisputeLLMRequest(BaseModel):
    """Simplified dispute request: 3 user inputs + context artifacts.

    The LLM translates the natural language message into a structured
    DisputeRequest and delegates to the existing dispute logic.
    """

    reason_code: ReasonCode
    message: str = Field(..., min_length=1, max_length=4000)
    evidence_urls: list[str] | None = Field(default=None, max_length=5)

    # Context (frontend passes automatically)
    prompt_spec: dict[str, Any]
    evidence_bundle: dict[str, Any] | None = None
    reasoning_trace: dict[str, Any] | None = None
    tool_plan: dict[str, Any] | None = None
    collectors: list[str] | None = None


_LLM_DISPUTE_SYSTEM_PROMPT = """\
You are a dispute analysis assistant for a prediction-market resolution protocol.

Your job: given a user's natural-language dispute message, the reason code they
selected, and the existing evidence/prompt context, produce a structured JSON
object that maps the dispute onto protocol internals.

Return ONLY valid JSON with these fields:

{
  "target_artifact": "evidence_bundle" | "reasoning_trace" | "verdict" | "prompt_spec",
  "target_leaf_path": "<optional JSONPath-like pointer, e.g. items[0].extracted_fields.outcome>",
  "structured_message": "<enhanced, structured version of the user's message>",
  "extracted_urls": ["<any URLs or domains mentioned in the user's message>"],
  "evidence_assessments": [
    {
      "url_index": 0,
      "outcome": "<Yes/No/Unknown>",
      "reason": "<why this URL supports/refutes the market question>"
    }
  ]
}

Rules:
- "target_artifact": Pick the most relevant artifact being disputed.
  - EVIDENCE_MISREAD / EVIDENCE_INSUFFICIENT → "evidence_bundle"
  - REASONING_ERROR / LOGIC_GAP → "reasoning_trace"
  - OTHER → best guess, default "reasoning_trace"
- "target_leaf_path": If the user references a specific evidence item or field,
  provide the path. Otherwise null.
- "structured_message": Rewrite the user's message to be clear, specific, and
  reference the relevant evidence. Keep the user's intent intact.
- "extracted_urls": Extract any URLs (e.g. https://espn.com/article/123) or
  bare domains (e.g. espn.com) the user mentions in their message. Return an
  empty array if none are found.
- "evidence_assessments": One entry per URL provided (by index). If no URLs were
  provided, return an empty array. Each assessment should state the outcome the
  URL supports and why.
"""


# -----------------------------
# URL Fetching Helpers
# -----------------------------

_MAX_CONTENT_CHARS = 8000
_FETCH_TIMEOUT = 15.0


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def _is_wikipedia_url(url: str) -> bool:
    return bool(re.match(r"https?://[\w]+\.wikipedia\.org/wiki/", url))


def _wikipedia_api_url(url: str) -> str:
    """Convert a Wikipedia article URL to a MediaWiki parse API URL."""
    # Extract title from URL like https://en.wikipedia.org/wiki/Some_Article
    match = re.match(r"https?://([\w]+)\.wikipedia\.org/wiki/(.+?)(?:#.*)?$", url)
    if not match:
        return url
    lang, title = match.group(1), match.group(2)
    return f"https://{lang}.wikipedia.org/w/api.php?action=parse&page={title}&prop=wikitext&format=json&redirects=1"


async def _fetch_url_content(
    ctx: Any, url: str
) -> tuple[str | None, str | None]:
    """Fetch URL content, returning (text_content, error_msg).

    Uses the MediaWiki API for Wikipedia URLs, plain HTML fetch otherwise.
    Truncates to _MAX_CONTENT_CHARS.
    """
    try:
        if _is_wikipedia_url(url):
            api_url = _wikipedia_api_url(url)
            response = await asyncio.to_thread(
                ctx.http.get, api_url, timeout=_FETCH_TIMEOUT
            )
            if not response.ok:
                return None, f"HTTP {response.status_code} from Wikipedia API"
            data = response.json()
            wikitext = (
                data.get("parse", {})
                .get("wikitext", {})
                .get("*", "")
            )
            # Strip wikitext markup (basic: remove {{ }}, [[ ]], etc.)
            text = re.sub(r"\{\{[^}]*\}\}", "", wikitext)
            text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
            text = re.sub(r"'{2,}", "", text)
            return text[:_MAX_CONTENT_CHARS], None
        else:
            response = await asyncio.to_thread(
                ctx.http.get, url, timeout=_FETCH_TIMEOUT
            )
            if not response.ok:
                return None, f"HTTP {response.status_code}"
            text = _html_to_text(response.text)
            return text[:_MAX_CONTENT_CHARS], None
    except Exception as e:
        return None, str(e)


# -----------------------------
# Planning
# -----------------------------

_RERUN_PLAN: dict[ReasonCode, list[str]] = {
    "REASONING_ERROR": ["audit", "judge"],
    "LOGIC_GAP": ["audit", "judge"],
    "EVIDENCE_MISREAD": ["collect", "audit", "judge"],
    "EVIDENCE_INSUFFICIENT": ["collect", "audit", "judge"],
    "OTHER": ["audit", "judge"],
}


def _normalize_patch(patch: DisputePatch | dict[str, Any] | None) -> DisputePatch | None:
    if patch is None:
        return None
    if isinstance(patch, DisputePatch):
        return patch
    if isinstance(patch, dict):
        try:
            return DisputePatch(**patch)
        except Exception:
            # Treat as no-op patch if schema doesn't match; keep raw dict for LLM context injection.
            return None
    return None


def _deep_merge(base: Any, override: Any) -> Any:
    """Deep merge override onto base.

    - dict: recurse
    - list: replace (return override)
    - scalar/other: replace

    This keeps behavior predictable and makes overrides explicit.
    """
    if override is None:
        return base

    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(out.get(k), v)
        return out

    if isinstance(override, list):
        return override

    return override


# -----------------------------
# URL Classification + Synthetic Requirement
# -----------------------------


def _classify_url(raw: str) -> Literal["url", "domain"]:
    """Classify a raw string as a full URL or a bare domain.

    - If it starts with http(s):// and has a path beyond ``/`` → ``"url"``
    - Otherwise (bare domain or root-only URL) → ``"domain"``
    """
    from urllib.parse import urlparse

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        # Has a meaningful path beyond "/"
        if parsed.path and parsed.path.rstrip("/"):
            return "url"
        return "domain"
    # Bare domain like "espn.com"
    return "domain"


def _build_dispute_requirement(
    prompt_spec: Any,
    evidence_urls: list[str],
) -> Any:
    """Build a DataRequirement from dispute-provided URLs/domains."""
    from core.schemas.prompts import DataRequirement, SourceTarget, SelectionPolicy

    source_targets = []
    for raw_url in evidence_urls:
        kind = _classify_url(raw_url)
        uri = raw_url if raw_url.startswith("http") else f"https://{raw_url}/"
        source_targets.append(SourceTarget(
            source_id="web",
            uri=uri,
            method="GET",
            expected_content_type="html",
            operation="search" if kind == "domain" else "fetch",
        ))
    return DataRequirement(
        requirement_id="dispute_evidence",
        description=prompt_spec.market.question,
        source_targets=source_targets,
        selection_policy=SelectionPolicy(
            strategy="single_best",
            min_sources=1,
            max_sources=len(source_targets),
            quorum=1,
        ),
    )


# -----------------------------
# Route
# -----------------------------


@router.post("/dispute", response_model=DisputeResponse)
async def dispute(request: DisputeRequest) -> DisputeResponse:
    """Rerun collect/audit/judge based on a dispute reason_code."""

    rerun_plan = _RERUN_PLAN.get(request.reason_code, ["audit", "judge"])

    # Parse artifacts into core schemas (strict validation)
    try:
        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle, EvidenceItem
        from core.schemas.reasoning import ReasoningTrace

        # Apply patch operations (prompt_spec override + evidence append)
        normalized_patch = _normalize_patch(request.patch)

        merged_prompt_spec_dict = request.prompt_spec
        if normalized_patch and normalized_patch.prompt_spec_override:
            merged_prompt_spec_dict = _deep_merge(request.prompt_spec, normalized_patch.prompt_spec_override)

        prompt_spec = PromptSpec(**merged_prompt_spec_dict)

        evidence_bundle = EvidenceBundle(**request.evidence_bundle) if request.evidence_bundle is not None else None
        reasoning_trace = (
            ReasoningTrace(**request.reasoning_trace) if request.reasoning_trace else None
        )

        if evidence_bundle is not None and normalized_patch and normalized_patch.evidence_items_append:
            for raw_item in normalized_patch.evidence_items_append:
                evidence_bundle.items.append(EvidenceItem(**raw_item))

    except Exception as e:
        raise InvalidRequestError(f"Invalid artifacts in request: {e}")

    # Build ctx + inject dispute_context for LLM-aware reruns
    ctx = get_agent_context(with_llm=True, with_http=True, llm_override=None)
    ctx.extra["dispute_context"] = {
        "reason_code": request.reason_code,
        "message": request.message,
        "target": request.target.model_dump(mode="json") if request.target else None,
        "patch": request.patch if isinstance(request.patch, dict) else (request.patch.model_dump(mode="json") if request.patch else None),
    }

    from agents.auditor import get_auditor
    from agents.judge import get_judge

    # Evidence bundles for rerun
    evidence_bundles = [evidence_bundle] if evidence_bundle is not None else []

    executed_plan: list[str] = []

    # ------------------------------------------------------------------
    # Collect step (reason_code-driven)
    # ------------------------------------------------------------------
    if "collect" in rerun_plan:
        if request.reason_code == "EVIDENCE_MISREAD":
            # Re-run original collectors with dispute context
            if request.tool_plan is None:
                raise InvalidRequestError(
                    "tool_plan is required for reason_code=EVIDENCE_MISREAD"
                )
            if not request.collectors:
                raise InvalidRequestError(
                    "collectors is required for reason_code=EVIDENCE_MISREAD"
                )

            try:
                from core.schemas.transport import ToolPlan

                tool_plan = ToolPlan(**request.tool_plan)
            except Exception as e:
                raise InvalidRequestError(f"Invalid tool_plan: {e}")

            evidence_bundles = await _run_collectors(
                ctx, prompt_spec, tool_plan, request.collectors, request.message,
            )
            if not evidence_bundles:
                raise InvalidRequestError(
                    "No evidence collected for EVIDENCE_MISREAD dispute"
                )
            executed_plan.append("collect")

        elif request.reason_code == "EVIDENCE_INSUFFICIENT":
            # Auto-collect from user-provided evidence_urls
            if not request.evidence_urls:
                raise InvalidRequestError(
                    "evidence_urls is required for reason_code=EVIDENCE_INSUFFICIENT"
                )

            dispute_req = _build_dispute_requirement(prompt_spec, request.evidence_urls)

            # Build a synthetic ToolPlan containing the dispute requirement
            from core.schemas.transport import ToolPlan

            tool_plan = ToolPlan(
                plan_id="dispute_evidence_plan",
                requirements=[dispute_req.requirement_id],
                sources=["web"],
            )

            # Inject the dispute requirement into prompt_spec so collectors can find it
            prompt_spec.data_requirements.append(dispute_req)

            # Choose collectors based on URL classification
            collector_names: list[str] = []
            for url in request.evidence_urls:
                kind = _classify_url(url)
                if kind == "domain":
                    collector_names.append("CollectorSitePinned")
                else:
                    collector_names.append("CollectorWebPageReader")
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_collectors: list[str] = []
            for c in collector_names:
                if c not in seen:
                    seen.add(c)
                    unique_collectors.append(c)

            new_bundles = await _run_collectors(
                ctx, prompt_spec, tool_plan, unique_collectors, request.message,
            )

            if not new_bundles:
                raise InvalidRequestError(
                    "No evidence collected for EVIDENCE_INSUFFICIENT dispute"
                )

            # Merge newly collected evidence into existing bundles
            if evidence_bundles:
                # Append items from new bundles into the first existing bundle
                for nb in new_bundles:
                    evidence_bundles[0].items.extend(nb.items)
            else:
                evidence_bundles = new_bundles

            executed_plan.append("collect")

    # ------------------------------------------------------------------
    # Audit (if in rerun_plan or reasoning_trace missing)
    # ------------------------------------------------------------------
    if "audit" in rerun_plan or reasoning_trace is None:
        if not evidence_bundles:
            raise InvalidRequestError(
                "evidence_bundle is required for dispute"
            )
        auditor = get_auditor(ctx)
        audit_result = await asyncio.to_thread(auditor.run, ctx, prompt_spec, evidence_bundles)
        if not audit_result.success:
            raise InvalidRequestError(f"Audit failed: {audit_result.error}")
        reasoning_trace = audit_result.output
        executed_plan.append("audit")

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------
    judge = get_judge(ctx)
    judge_result = await asyncio.to_thread(judge.run, ctx, prompt_spec, evidence_bundles, reasoning_trace)
    if not judge_result.success:
        raise InvalidRequestError(f"Judge failed: {judge_result.error}")
    verdict = judge_result.output
    executed_plan.append("judge")

    primary_bundle = evidence_bundles[0] if evidence_bundles else None

    artifacts = {
        "prompt_spec": prompt_spec.model_dump(mode="json"),
        # Back-compat: include a single primary bundle
        "evidence_bundle": primary_bundle.model_dump(mode="json") if primary_bundle else None,
        # Preferred: include all bundles
        "evidence_bundles": [b.model_dump(mode="json") for b in evidence_bundles],
        "reasoning_trace": reasoning_trace.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
    }

    # Lightweight diff summary (frontend can compute richer diff)
    diff: dict[str, Any] = {
        "steps_rerun": executed_plan,
        "verdict_changed": None,
    }

    return DisputeResponse(
        case_id=request.case_id,
        rerun_plan=executed_plan,
        artifacts=artifacts,
        diff=diff,
    )


# ------------------------------------------------------------------
# Collector runner (shared by EVIDENCE_MISREAD + EVIDENCE_INSUFFICIENT)
# ------------------------------------------------------------------


async def _run_collectors(
    ctx: Any,
    prompt_spec: Any,
    tool_plan: Any,
    collector_names: list[str],
    dispute_message: str,
) -> list[Any]:
    """Instantiate and run collectors, injecting dispute context.

    Returns a list of successful EvidenceBundles (may be empty).
    """
    from agents.registry import get_registry
    from api.deps import build_llm_override, get_agent_context as _get_ctx

    registry = get_registry()

    collector_override = build_llm_override(None, None, agent_name="collector")
    collector_ctx = _get_ctx(with_llm=True, with_http=True, llm_override=collector_override)

    # Inject dispute context + quality feedback into collector context
    collector_ctx.extra["dispute_context"] = ctx.extra.get("dispute_context", {})
    collector_ctx.extra["quality_feedback"] = {
        "collector_guidance": dispute_message,
    }

    tasks = []
    task_names = []
    for collector_name in collector_names:
        try:
            if collector_name == "CollectorPAN":
                from agents.collector.pan_agent import PANCollectorAgent
                collector = PANCollectorAgent()
            elif collector_name == "CollectorOpenSearch":
                from agents.collector.gemini_grounded_agent import CollectorOpenSearch
                collector = CollectorOpenSearch()
            elif collector_name == "CollectorSitePinned":
                from agents.collector.source_pinned_agent import CollectorSitePinned
                collector = CollectorSitePinned()
            elif collector_name == "CollectorCRP":
                from agents.collector.crp_agent import CollectorCRP
                collector = CollectorCRP()
            elif collector_name == "CollectorAPIData":
                from agents.collector.api_data_agent import CollectorAPIData
                collector = CollectorAPIData()
            elif collector_name == "CollectorWebPageReader":
                from agents.collector.agent import CollectorWebPageReader
                collector = CollectorWebPageReader()
            else:
                collector = registry.get_agent_by_name(collector_name, collector_ctx)
        except Exception as e:
            raise InvalidRequestError(f"Collector not found: {collector_name} ({e})")

        tasks.append(asyncio.to_thread(collector.run, collector_ctx, prompt_spec, tool_plan))
        task_names.append(collector.name)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    evidence_bundles = []
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.warning("Collector %s failed: %s", name, result)
            continue
        if not getattr(result, "success", False):
            logger.warning("Collector %s failed: %s", name, getattr(result, "error", None))
            continue
        bundle, _ = result.output
        bundle.collector_name = name
        evidence_bundles.append(bundle)

    return evidence_bundles


# -----------------------------
# LLM-Assisted Dispute Route
# -----------------------------


def _build_llm_user_prompt(
    request: DisputeLLMRequest,
    fetched_contents: list[tuple[str, str | None, str | None]],
) -> str:
    """Build the user prompt for the LLM dispute parser.

    Args:
        request: The incoming LLM dispute request.
        fetched_contents: List of (url, content_or_none, error_or_none).
    """
    parts: list[str] = []

    # Reason code
    parts.append(f"## Reason Code\n{request.reason_code}")

    # User message
    parts.append(f"## User Dispute Message\n{request.message}")

    # Market question from prompt_spec
    market_q = request.prompt_spec.get("market_question", "")
    resolution_rules = request.prompt_spec.get("resolution_rules", "")
    if market_q:
        parts.append(f"## Market Question\n{market_q}")
    if resolution_rules:
        parts.append(f"## Resolution Rules\n{resolution_rules}")

    # Existing evidence summary
    if request.evidence_bundle:
        items = request.evidence_bundle.get("items", [])
        if items:
            summary_lines = []
            for i, item in enumerate(items):
                ef = item.get("extracted_fields", {})
                src = item.get("source_url", item.get("url", "unknown"))
                outcome = ef.get("outcome", "N/A")
                reason = ef.get("reason", "N/A")
                summary_lines.append(f"  [{i}] source={src}  outcome={outcome}  reason={reason}")
            parts.append("## Existing Evidence Items\n" + "\n".join(summary_lines))

    # Fetched URL contents
    if fetched_contents:
        url_parts = []
        for idx, (url, content, error) in enumerate(fetched_contents):
            if error:
                url_parts.append(f"  [{idx}] {url} — FETCH ERROR: {error}")
            else:
                # Truncate preview for prompt
                preview = (content or "")[:4000]
                url_parts.append(f"  [{idx}] {url}\n{preview}")
        parts.append("## User-Provided URL Contents\n" + "\n".join(url_parts))

    return "\n\n".join(parts)


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, tolerating markdown fences."""
    # Try to find JSON in markdown code blocks first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try the whole string as JSON
    # Find the first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        return json.loads(raw[start : end + 1])
    raise ValueError("No JSON object found in LLM response")


_LLM_FALLBACK_TARGETS: dict[str, str] = {
    "EVIDENCE_MISREAD": "evidence_bundle",
    "EVIDENCE_INSUFFICIENT": "evidence_bundle",
    "REASONING_ERROR": "reasoning_trace",
    "LOGIC_GAP": "reasoning_trace",
    "OTHER": "reasoning_trace",
}


@router.post("/dispute/llm", response_model=DisputeResponse)
async def dispute_llm(request: DisputeLLMRequest) -> DisputeResponse:
    """LLM-assisted dispute: translate natural language into structured dispute."""

    ctx = get_agent_context(with_llm=True, with_http=True, llm_override=None)

    # --- Step 1: Fetch URLs (if provided) ---
    fetched_contents: list[tuple[str, str | None, str | None]] = []
    if request.evidence_urls:
        fetch_tasks = [
            _fetch_url_content(ctx, url) for url in request.evidence_urls
        ]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for url, result in zip(request.evidence_urls, fetch_results):
            if isinstance(result, Exception):
                fetched_contents.append((url, None, str(result)))
            else:
                content, error = result
                fetched_contents.append((url, content, error))

    # --- Step 2: LLM Parse ---
    user_prompt = _build_llm_user_prompt(request, fetched_contents)

    try:
        llm_response = await asyncio.to_thread(
            ctx.llm.chat,
            [
                {"role": "system", "content": _LLM_DISPUTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = _parse_llm_response(llm_response.content)
    except Exception as e:
        logger.warning("LLM dispute parse failed, using fallback: %s", e)
        # Fallback: use reason_code to pick reasonable defaults
        parsed = {
            "target_artifact": _LLM_FALLBACK_TARGETS.get(request.reason_code, "reasoning_trace"),
            "target_leaf_path": None,
            "structured_message": request.message,
            "extracted_urls": [],
            "evidence_assessments": [],
        }

    # --- Step 3: Build DisputeRequest from LLM output ---
    target_artifact = parsed.get("target_artifact", "reasoning_trace")
    target_leaf_path = parsed.get("target_leaf_path")
    structured_message = parsed.get("structured_message", request.message)
    evidence_assessments = parsed.get("evidence_assessments", [])

    # Merge extracted_urls from LLM with explicitly provided evidence_urls (deduplicate)
    extracted_urls: list[str] = parsed.get("extracted_urls", [])
    explicit_urls: list[str] = list(request.evidence_urls) if request.evidence_urls else []
    all_urls_ordered: list[str] = []
    seen_urls: set[str] = set()
    for u in explicit_urls + extracted_urls:
        normalized = u.strip().rstrip("/").lower()
        if normalized and normalized not in seen_urls:
            seen_urls.add(normalized)
            all_urls_ordered.append(u.strip())

    # Build target
    target = DisputeTarget(artifact=target_artifact, leaf_path=target_leaf_path)

    # Build evidence_items_append from fetched URLs + LLM assessments
    evidence_items_append: list[dict[str, Any]] = []
    if fetched_contents:
        for idx, (url, content, error) in enumerate(fetched_contents):
            if error or content is None:
                continue  # Skip failed fetches

            # Find matching assessment from LLM
            assessment = next(
                (a for a in evidence_assessments if a.get("url_index") == idx),
                None,
            )

            extracted_fields = {}
            if assessment:
                extracted_fields["outcome"] = assessment.get("outcome", "Unknown")
                extracted_fields["reason"] = assessment.get("reason", "")

            evidence_items_append.append({
                "evidence_id": f"dispute-llm-url-{idx}",
                "requirement_id": "dispute-user-evidence",
                "provenance": {
                    "source_id": "dispute_llm_user_url",
                    "source_uri": url,
                    "tier": 2,
                },
                "raw_content": content[:_MAX_CONTENT_CHARS],
                "extracted_fields": extracted_fields,
            })

    # Build patch
    patch: DisputePatch | None = None
    if evidence_items_append:
        patch = DisputePatch(evidence_items_append=evidence_items_append)

    # Assemble the full DisputeRequest — reason_code drives what steps run
    dispute_request = DisputeRequest(
        reason_code=request.reason_code,
        message=structured_message,
        target=target,
        prompt_spec=request.prompt_spec,
        evidence_bundle=request.evidence_bundle,
        reasoning_trace=request.reasoning_trace,
        tool_plan=request.tool_plan,
        collectors=request.collectors,
        evidence_urls=all_urls_ordered or None,
        patch=patch,
    )

    # --- Step 4: Delegate to existing dispute logic ---
    return await dispute(dispute_request)
