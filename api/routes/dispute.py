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
    mode: Literal["reasoning_only", "full_rerun"] = Field(
        default="reasoning_only",
        description="reasoning_only reruns downstream steps using provided evidence. full_rerun re-collects evidence then reruns.",
    )

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

    # Optional: required for full_rerun (stateless re-collect)
    tool_plan: dict[str, Any] | None = None
    collectors: list[str] | None = None

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
  "mode": "reasoning_only" | "full_rerun",
  "target_artifact": "evidence_bundle" | "reasoning_trace" | "verdict" | "prompt_spec",
  "target_leaf_path": "<optional JSONPath-like pointer, e.g. items[0].extracted_fields.outcome>",
  "structured_message": "<enhanced, structured version of the user's message>",
  "evidence_assessments": [
    {
      "url_index": 0,
      "outcome": "<Yes/No/Unknown>",
      "reason": "<why this URL supports/refutes the market question>"
    }
  ]
}

Rules:
- "mode": Use "reasoning_only" unless the user explicitly asks for new evidence
  collection or the reason_code is EVIDENCE_INSUFFICIENT and the dispute clearly
  needs fresh evidence. Default to "reasoning_only".
- "target_artifact": Pick the most relevant artifact being disputed.
  - EVIDENCE_MISREAD / EVIDENCE_INSUFFICIENT → "evidence_bundle"
  - REASONING_ERROR / LOGIC_GAP → "reasoning_trace"
  - OTHER → best guess, default "reasoning_trace"
- "target_leaf_path": If the user references a specific evidence item or field,
  provide the path. Otherwise null.
- "structured_message": Rewrite the user's message to be clear, specific, and
  reference the relevant evidence. Keep the user's intent intact.
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
    "REASONING_ERROR": ["judge"],
    "LOGIC_GAP": ["judge"],
    "EVIDENCE_MISREAD": ["audit", "judge"],
    "EVIDENCE_INSUFFICIENT": ["audit", "judge"],
    "OTHER": ["judge"],
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
# Route
# -----------------------------


@router.post("/dispute", response_model=DisputeResponse)
async def dispute(request: DisputeRequest) -> DisputeResponse:
    """Rerun audit/judge based on a dispute, returning new artifacts."""

    if request.mode not in ("reasoning_only", "full_rerun"):
        raise InvalidRequestError("Invalid mode")

    rerun_plan = _RERUN_PLAN.get(request.reason_code, ["judge"])

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

    # Optional: full rerun includes evidence collection
    if request.mode == "full_rerun":
        if request.tool_plan is None:
            raise InvalidRequestError("tool_plan is required for mode=full_rerun")
        if not request.collectors:
            raise InvalidRequestError("collectors is required for mode=full_rerun")

        try:
            from core.schemas.transport import ToolPlan
            from core.schemas.evidence import EvidenceBundle
            from agents.registry import get_registry
            from api.deps import build_llm_override

            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            raise InvalidRequestError(f"Invalid tool_plan: {e}")

        registry = get_registry()

        collector_override = build_llm_override(None, None, agent_name="collector")
        collector_ctx = get_agent_context(with_llm=True, with_http=True, llm_override=collector_override)

        tasks = []
        task_names = []
        for collector_name in request.collectors:
            try:
                # Match /step/resolve collector instantiation patterns
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
                logger.warning(f"Collector {name} failed: {result}")
                continue
            if not getattr(result, "success", False):
                logger.warning(f"Collector {name} failed: {getattr(result, 'error', None)}")
                continue
            bundle, _ = result.output
            bundle.collector_name = name
            evidence_bundles.append(bundle)

        if not evidence_bundles:
            raise InvalidRequestError("No evidence collected in full_rerun")

        executed_plan.append("collect")

    # Audit (if requested or if reasoning_trace missing)
    if "audit" in rerun_plan or reasoning_trace is None:
        if not evidence_bundles:
            raise InvalidRequestError("evidence_bundle is required for reasoning_only, or enable full_rerun")
        auditor = get_auditor(ctx)
        audit_result = await asyncio.to_thread(auditor.run, ctx, prompt_spec, evidence_bundles)
        if not audit_result.success:
            raise InvalidRequestError(f"Audit failed: {audit_result.error}")
        reasoning_trace = audit_result.output
        executed_plan.append("audit")

    # Judge
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
            "mode": "reasoning_only",
            "target_artifact": _LLM_FALLBACK_TARGETS.get(request.reason_code, "reasoning_trace"),
            "target_leaf_path": None,
            "structured_message": request.message,
            "evidence_assessments": [],
        }

    # --- Step 3: Build DisputeRequest from LLM output ---
    mode = parsed.get("mode", "reasoning_only")
    if mode not in ("reasoning_only", "full_rerun"):
        mode = "reasoning_only"

    target_artifact = parsed.get("target_artifact", "reasoning_trace")
    target_leaf_path = parsed.get("target_leaf_path")
    structured_message = parsed.get("structured_message", request.message)
    evidence_assessments = parsed.get("evidence_assessments", [])

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

    # Assemble the full DisputeRequest
    dispute_request = DisputeRequest(
        mode=mode,
        reason_code=request.reason_code,
        message=structured_message,
        target=target,
        prompt_spec=request.prompt_spec,
        evidence_bundle=request.evidence_bundle,
        reasoning_trace=request.reasoning_trace,
        tool_plan=request.tool_plan if mode == "full_rerun" else None,
        collectors=request.collectors if mode == "full_rerun" else None,
        patch=patch,
    )

    # --- Step 4: Delegate to existing dispute logic ---
    return await dispute(dispute_request)
