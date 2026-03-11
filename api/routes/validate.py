"""Market Creation Validator Route

POST /validate — Validate a market query and compile it into a PromptSpec.

Combines two previously-separate steps into a single call:
  1. Prompt compilation (PromptSpec + ToolPlan)
  2. Market validation (classify → validate fields → assess resolvability)

Also probes any data sources mentioned in the query to check reachability,
so the creator gets early warning if a site is behind Cloudflare or otherwise
inaccessible to the AI resolution system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import build_llm_override, get_agent_context
from api.errors import InternalError, InvalidRequestError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["validate"])


# ---------------------------------------------------------------------------
# Market type enum
# ---------------------------------------------------------------------------

MarketType = Literal[
    "FINANCIAL_PRICE",
    "TEMPERATURE",
    "CRYPTO_THRESHOLD",
    "SPORTS_MATCH",
    "SPORTS_EXACT_SCORE",
    "SPORTS_PLAYER_PROP",
    "SPEECH_CONTENT",
    "GEOPOLITICAL_EVENT",
    "BINARY_EVENT",
    "MULTI_CHOICE_EVENT",
    "OPINION_NOVELTY",
    "UNKNOWN",
]

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    user_input: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The prediction market query to validate and compile (title + description combined)",
    )
    strict_mode: bool = Field(
        default=True,
        description="Enable strict mode for deterministic hashing",
    )
    llm_provider: str | None = Field(default=None, description="LLM provider override")
    llm_model: str | None = Field(default=None, description="LLM model override")


class MarketClassification(BaseModel):
    market_type: str = Field(..., description="Detected market type ID")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    detection_rationale: str = Field(..., description="Why this type was chosen")


class ValidationIssue(BaseModel):
    check_id: str = Field(..., description="Check ID (e.g. U-01, TEMP-04)")
    severity: Literal["error", "warning", "info"] = Field(..., description="Issue severity")
    message: str = Field(..., description="Human-readable issue description")
    suggestion: str = Field(..., description="Actionable fix suggestion for the creator")


class ValidationResult(BaseModel):
    checks_passed: list[str] = Field(default_factory=list, description="IDs of checks that passed")
    checks_failed: list[ValidationIssue] = Field(
        default_factory=list, description="Checks that failed with details"
    )


class RiskFactor(BaseModel):
    factor: str = Field(..., description="Description of the risk factor")
    points: int = Field(..., description="Risk points contributed")


class ResolvabilityAssessment(BaseModel):
    score: int = Field(..., ge=0, description="Total risk score")
    level: RiskLevel = Field(..., description="Risk level: LOW/MEDIUM/HIGH/VERY_HIGH")
    risk_factors: list[RiskFactor] = Field(
        default_factory=list, description="Individual risk factors"
    )
    recommendation: str = Field(..., description="Summary recommendation for the creator")


class SourceReachability(BaseModel):
    url: str = Field(..., description="URL that was probed")
    reachable: bool = Field(..., description="Whether the URL returned a successful response")
    status_code: int | None = Field(default=None, description="HTTP status code (if any)")
    error: str | None = Field(default=None, description="Error message if unreachable")
    method: str | None = Field(
        default=None,
        description="How the URL was reached: 'direct' (normal HTTP) or 'jina_reader' (via r.jina.ai proxy)",
    )


class ValidateResponse(BaseModel):
    ok: bool = Field(..., description="Whether both validation and prompt compilation succeeded")
    classification: MarketClassification
    validation: ValidationResult
    resolvability: ResolvabilityAssessment
    source_reachability: list[SourceReachability] = Field(
        default_factory=list,
        description="Reachability results for data sources mentioned in the query",
    )
    # Prompt compilation outputs
    market_id: str | None = Field(default=None, description="Generated market ID")
    prompt_spec: dict[str, Any] | None = Field(default=None, description="Compiled prompt specification")
    tool_plan: dict[str, Any] | None = Field(default=None, description="Tool execution plan")
    prompt_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Prompt compilation metadata"
    )
    errors: list[str] = Field(default_factory=list, description="Non-fatal errors")


# ---------------------------------------------------------------------------
# LLM system prompt — encodes all classification rules, checks, and scoring
# ---------------------------------------------------------------------------

_VALIDATE_SYSTEM_PROMPT = """\
You are a market creation validator for a prediction market resolution protocol.

Your job: given a market query (which may contain a title, description, resolution
criteria, or any combination), perform 3 stages of analysis and return a single
structured JSON result.

## STAGE 1: CLASSIFY MARKET TYPE

Classify into exactly one of these types based on the market query:

| Type ID | Detection Pattern |
|---|---|
| FINANCIAL_PRICE | Stock ticker + "closes", "above", "week of", price thresholds |
| TEMPERATURE | "highest temperature", "weather", city name + date |
| CRYPTO_THRESHOLD | Crypto pair + "above" + price + time |
| SPORTS_MATCH | Team vs Team, "who wins" |
| SPORTS_EXACT_SCORE | "exact score" |
| SPORTS_PLAYER_PROP | Player name + stat (tackles, fouls, shots, etc.) |
| SPEECH_CONTENT | "what will X say", "during event" |
| GEOPOLITICAL_EVENT | Country + action verb (strike, launch, capture) |
| BINARY_EVENT | Yes/No market about a future event |
| MULTI_CHOICE_EVENT | Multiple possible outcomes, not price-based |
| OPINION_NOVELTY | Comparative markets without factual resolution |
| UNKNOWN | Cannot determine type |

## STAGE 2: VALIDATE REQUIRED FIELDS

### Universal checks (apply to ALL markets):

| Check ID | What It Checks | Prompt to Creator |
|---|---|---|
| U-01 | Title contains ___ placeholder | "Your title contains '___'. Please rewrite the title to include the actual comparison, OR clearly state in the description that this is a multi-choice market where each outcome is evaluated independently." |
| U-02 | Description specifies at least one resolution data source | "Please specify the data source for resolution. Where should we look to determine the outcome? (e.g., 'Yahoo Finance', 'Wunderground', 'FotMob', 'FDIC website')" |
| U-03 | Description specifies a resolution deadline with timezone | "Please specify the exact deadline with timezone. e.g., 'February 25, 2026, 4:00 PM ET'" |
| U-04 | Description defines what "Yes" and "No" mean (for binary markets) | "Please define exactly what constitutes a 'Yes' resolution. Be specific about thresholds, actions, or conditions." |
| U-05 | Possible outcomes are explicitly listed (for multi-choice) | "Please list all possible outcomes. For multi-choice markets, each outcome must be mutually exclusive and collectively exhaustive." |
| U-06 | Description includes a fallback rule for event edge cases | "What happens if the event is cancelled, postponed, or otherwise does not occur as expected? Please add a fallback resolution rule." |
| U-07 | Description specifies a fallback data source or consensus-based resolution | "Your market specifies only one data source. If that source is unavailable (site down, blocked by Cloudflare, data not published), how should the market resolve? Please add a fallback such as: 'If the primary source is unavailable, resolution may use a consensus of credible reporting from major news agencies (AP, Reuters, BBC, etc.).' or specify an alternative data source URL." |

Apply U-04 only to binary markets. Apply U-05 only to multi-choice markets.

IMPORTANT for U-07:
- If the market specifies only ONE data source URL and does NOT include any consensus/fallback
  language, U-07 MUST FAIL. This is critical — single-source markets fail at ~2x the rate.
- If the market specifies multiple data sources, or includes consensus/news-based fallback
  language (e.g., "consensus of credible reporting", "multiple credible sources", "major news
  outlets"), U-07 PASSES.
- U-07 is separate from U-06. U-06 is about event edge cases (cancelled, postponed). U-07 is
  about data source redundancy.

### Type-specific checks:

#### FINANCIAL_PRICE
| FIN-01 | Ticker symbol is explicit and unambiguous | "Please confirm the exact ticker symbol and exchange (e.g., 'NVDA on NASDAQ')." |
| FIN-02 | For multi-choice "closes above ___": description explains resolution logic | "This appears to be a multi-choice market with price thresholds. Please clarify: should each threshold be evaluated independently (Yes/No per threshold), or should the market resolve to the single threshold closest to the actual price?" |
| FIN-03 | Primary data source is specified | "Which financial data provider should be used? (Yahoo Finance, Bloomberg, Google Finance)" |
| FIN-04 | "Closing price" definition: regular session or after-hours? | "Does 'closing price' mean the regular session close (4:00 PM ET) or does it include after-hours trading?" |
| FIN-05 | For "week of X": which day is the final trading day | "Which specific date is the final trading day of the week? (Normally Friday, but please confirm)" |

#### TEMPERATURE
| TEMP-01 | Weather station is specified by name | "Which weather station should be used? (e.g., 'LaGuardia Airport Station', 'Incheon Intl Airport Station')" |
| TEMP-02 | Unit is specified (°C or °F) | "Celsius or Fahrenheit?" |
| TEMP-03 | Primary data source specified | "Which weather data source? (Wunderground, AccuWeather, Weather.com, national meteorological agency)" |
| TEMP-04 | Fallback data source specified | "If data from the primary source is unavailable, which alternative source should be used? We recommend specifying at least 2 sources. In our data, 46 out of 132 temperature markets failed because Wunderground data was unavailable." |
| TEMP-05 | Date timing is reasonable for data availability | "Note: temperature markets can only be resolved after the date has passed and recorded data is available. Are you sure this date will have finalized data by your resolution deadline?" |

#### CRYPTO_THRESHOLD
| CRYPTO-01 | Exchange is specified | "Which exchange should be used for the price?" |
| CRYPTO-02 | Candle type is specified (open, close, high, low) | "Which candle price? (hourly open, hourly close, spot price at exact time)" |
| CRYPTO-03 | Timezone is explicit | "Please confirm the timezone." |
| CRYPTO-04 | Direct API data source preferred over web scraping | "Note: our system has difficulty scraping TradingView data reliably (9 out of 11 crypto/KRW markets failed). Consider specifying a direct API source (e.g., Binance API, CoinGecko API)." |
| CRYPTO-05 | Threshold number format is unambiguous | "Please confirm the exact threshold number to avoid ambiguity." |

#### SPORTS_MATCH / SPORTS_EXACT_SCORE
| SPORT-01 | League and match date specified | "Please confirm the league (e.g., Premier League) and exact match date/time." |
| SPORT-02 | Result type: 90 minutes only, or including extra time/penalties? | "Does the result include extra time and penalties, or regulation time only?" |
| SPORT-03 | Draw included as outcome for match result markets | "Is a draw a possible outcome? If so, please list 'Draw' as an explicit option." |
| SPORT-04 | Data source for results | "Which source for match results? (Official league website, ESPN, FotMob, Flashscore)" |

#### SPORTS_PLAYER_PROP
| PROP-01 | Player full name specified | "Please provide the player's full name as it appears on the stats source." |
| PROP-02 | Stat type precisely defined | "Please define the exact stat." |
| PROP-03 | Stats source specified | "Which stats provider? (FotMob, WhoScored, ESPN, SofaScore)" |
| PROP-04 | Resolvability warning | "Note: Player-level match stats can be difficult for our system to retrieve automatically. Consider providing the direct URL pattern for the stats page. Resolution may require manual verification." |

#### SPEECH_CONTENT
| SPEECH-01 | Event name and date specified | "Please provide the exact event name and scheduled date/time." |
| SPEECH-02 | Where to find transcript/recording | "Where will the transcript or recording be available? (C-SPAN, White House website, YouTube channel)" |
| SPEECH-03 | Definition of "say" — live speech only or written remarks too? | "Does 'say' mean spoken words during the live event only, or does it include prepared written remarks and press releases?" |
| SPEECH-04 | What counts as a match — exact word or variations? | "Does the speaker need to say the exact word, or do variations count?" |
| SPEECH-05 | Resolvability warning | "Note: Our system has difficulty accessing live event transcripts. 4 out of 4 speech-content markets in our data were misresolved. Consider providing a specific URL where the transcript will be posted, or flag this market for manual resolution." |

#### GEOPOLITICAL_EVENT
| GEO-01 | Precise definition of the event | "What exactly constitutes this event? Please list qualifying and non-qualifying actions." |
| GEO-02 | News consensus recommended over single source | "For geopolitical events, we recommend using news consensus (e.g., 'confirmed by at least 2 of: AP, Reuters, BBC, official government statement') rather than a single news source. Single sources are often behind Cloudflare or paywalls that prevent automated access." |
| GEO-03 | Minimum evidence threshold | "How many independent sources need to confirm the event for resolution? We recommend at least 2 authoritative sources." |
| GEO-04 | Timeline is clear with timezone | "Please specify the exact deadline with timezone." |

#### BINARY_EVENT
| BIN-01 | "Yes" criteria fully specified | "What specific, verifiable condition must be met for Yes?" |
| BIN-02 | Verification method clear | "How can the system verify this? Please provide a specific URL or data source." |
| BIN-03 | Edge cases addressed | "What happens if the event is partially true? (e.g., event happens but after the deadline)" |

#### OPINION_NOVELTY
| NOVEL-01 | Title matches actual resolution criteria | "Your title may be misleading. Consider making the title clearer to match the actual resolution criteria." |
| NOVEL-02 | Resolution is objectively verifiable | "Can this market be resolved using publicly available data? If it's purely opinion-based, AI resolution may not be possible." |

## STAGE 3: ASSESS RESOLVABILITY

Assign risk points based on these factors:

| Risk Factor | Points | Related Check |
|---|---|---|
| Single data source with no fallback (no consensus, no alternative URL) | +30 | U-07 fail |
| Data source requires web scraping (TradingView, specific URLs) | +20 | |
| Data source is Wunderground only | +25 | |
| Market requires real-time or near-real-time data | +15 | |
| Market requires transcript/video analysis | +35 | |
| Market has subjective resolution criteria | +40 | |
| Title contains ___ or unclear placeholder | +20 | U-01 fail |
| No fallback rule for cancelled/postponed events | +10 | U-06 fail |
| Event is inherently random (coin toss, exact sports score) | +15 | |
| Multiple independent conditions | +10 | |
| Data source likely behind Cloudflare/paywall (non-API news sites, etc.) | +20 | |

IMPORTANT: The "Single data source with no fallback" risk factor (+30) MUST be applied
when U-07 fails. These are linked — if U-07 fails, this risk factor applies. If U-07
passes (because the market has consensus language or multiple sources), this risk factor
MUST NOT be applied.

Risk levels:
- 0-15: LOW — approved for automated resolution
- 16-35: MEDIUM — warn creator about difficulty
- 36-55: HIGH — high risk of failing, show specific fixes
- 56+: VERY_HIGH — unlikely to resolve automatically, consider manual resolution

## OUTPUT FORMAT

Return ONLY valid JSON with this exact structure:

{
  "classification": {
    "market_type": "<TYPE_ID>",
    "confidence": <0.0-1.0>,
    "detection_rationale": "<why this type>"
  },
  "validation": {
    "checks_passed": ["<CHECK_ID>", ...],
    "checks_failed": [
      {
        "check_id": "<CHECK_ID>",
        "severity": "error" | "warning" | "info",
        "message": "<what's wrong>",
        "suggestion": "<actionable fix from the tables above>"
      }
    ]
  },
  "resolvability": {
    "score": <integer>,
    "level": "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH",
    "risk_factors": [
      {"factor": "<description>", "points": <integer>}
    ],
    "recommendation": "<summary recommendation>"
  }
}

Rules:
- Only run checks relevant to the detected market type + universal checks.
- For universal checks U-04 and U-05, only apply them when relevant (U-04 for binary, U-05 for multi-choice).
- severity: "error" for critical missing info that will cause INVALID, "warning" for likely issues, "info" for recommendations.
- Be thorough but do not invent checks beyond the ones listed above.
- The risk score must equal the sum of the risk_factors points.
- Do NOT include risk factors that don't apply.

## IMPORTANT: Recognizing fallback sources and consensus language

The AI resolution system can search the open web for news and information. Therefore,
the following language in a market description counts as a VALID FALLBACK source and
should NOT trigger "Single data source with no fallback":

- "consensus of credible reporting may also be used"
- "a consensus of credible news sources"
- "multiple credible sources"
- "major news outlets"
- "credible reporting from major news agencies (AP, Reuters, BBC, etc.)"
- Any similar phrasing that indicates open web news consensus as a secondary resolution method

When a market specifies BOTH a primary source (e.g., an official website URL) AND
consensus/news-based fallback language, this is an ideal configuration:
- The primary source gives the AI a specific place to look first
- The consensus fallback means the AI can search the open web if the primary source fails

In this case:
- U-02 (data source specified): PASS
- U-06 (fallback rule): PASS
- "Single data source with no fallback" risk factor: DO NOT APPLY
- The market should receive a LOWER risk score, not a higher one

Similarly, if a market says resolution uses "official government statements" or
"authoritative news sources" without a specific URL, this is still a valid source
because the AI can search for these. It is less ideal than a direct URL but should
not be treated as "no source specified".
"""


# ---------------------------------------------------------------------------
# Source reachability probing
# ---------------------------------------------------------------------------

_PROBE_TIMEOUT = 10.0


_CLOUDFLARE_MARKERS = [
    "cf-browser-verification",
    "cf_chl_opt",
    "challenges.cloudflare.com",
    "Checking if the site connection is secure",
    "Enable JavaScript and cookies to continue",
    "Attention Required! | Cloudflare",
    "Just a moment...",
    "Verify you are human",
]


_JINA_READER_PREFIX = "https://r.jina.ai/"


async def _probe_url_direct(http_client: Any, url: str) -> SourceReachability:
    """Probe a URL with a direct GET request.

    Returns reachability result including Cloudflare detection.
    """
    try:
        response = await asyncio.to_thread(
            http_client.get, url, timeout=_PROBE_TIMEOUT,
        )
        status = response.status_code

        # Check for Cloudflare on 403/503
        if status in (403, 503):
            body = response.text[:4000].lower()
            for marker in _CLOUDFLARE_MARKERS:
                if marker.lower() in body:
                    return SourceReachability(
                        url=url,
                        reachable=False,
                        status_code=status,
                        method="direct",
                        error="Blocked by Cloudflare challenge",
                    )

        if status < 400:
            return SourceReachability(url=url, reachable=True, status_code=status, method="direct")

        return SourceReachability(url=url, reachable=False, status_code=status, method="direct", error=f"HTTP {status}")

    except Exception as e:
        return SourceReachability(url=url, reachable=False, method="direct", error=str(e))


async def _probe_url_jina(http_client: Any, url: str) -> SourceReachability:
    """Probe a URL via Jina Reader (r.jina.ai) to bypass Cloudflare/JS walls.

    Jina Reader renders the page and returns markdown content, making
    JS-heavy and Cloudflare-protected sites readable by AI collectors.
    """
    jina_url = f"{_JINA_READER_PREFIX}{url}"
    try:
        response = await asyncio.to_thread(
            http_client.get, jina_url, timeout=_PROBE_TIMEOUT + 5,
        )
        status = response.status_code

        if status < 400:
            return SourceReachability(url=url, reachable=True, status_code=status, method="jina_reader")

        return SourceReachability(
            url=url, reachable=False, status_code=status, method="jina_reader",
            error=f"Jina Reader returned HTTP {status}",
        )

    except Exception as e:
        return SourceReachability(url=url, reachable=False, method="jina_reader", error=f"Jina Reader: {e}")


async def _probe_url(http_client: Any, url: str) -> SourceReachability:
    """Probe a URL for reachability, falling back to Jina Reader.

    1. Try direct HTTP GET — fast, works for most sites.
    2. If direct fails (Cloudflare, 403, 5xx, timeout), try Jina Reader
       which can bypass Cloudflare and render JS-heavy pages.
    """
    direct = await _probe_url_direct(http_client, url)
    if direct.reachable:
        return direct

    # Fallback to Jina Reader for blocked/failed URLs
    jina = await _probe_url_jina(http_client, url)
    if jina.reachable:
        return jina

    # Both failed — return direct error with note about Jina fallback
    return SourceReachability(
        url=url,
        reachable=False,
        status_code=direct.status_code,
        method="direct",
        error=f"{direct.error or 'Direct request failed'}. Jina Reader also failed: {jina.error or 'unknown'}",
    )


async def _probe_sources(http_client: Any, urls: list[str]) -> list[SourceReachability]:
    """Probe multiple URLs concurrently."""
    if not urls:
        return []
    tasks = [_probe_url(http_client, url) for url in urls]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_source_urls(prompt_spec_dict: dict[str, Any] | None) -> list[str]:
    """Extract source target URIs from a compiled PromptSpec dict.

    The prompt engineer populates data_requirements[].source_targets[].uri
    with the URLs it plans to fetch during evidence collection.
    """
    if not prompt_spec_dict:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for req in prompt_spec_dict.get("data_requirements", []):
        for target in req.get("source_targets", []):
            uri = target.get("uri", "")
            if uri and uri.startswith("http") and uri not in seen:
                urls.append(uri)
                seen.add(uri)
    return urls


def _build_validate_user_prompt(user_input: str) -> str:
    return f"## Market Query\n{user_input}"


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, tolerating markdown fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        return json.loads(raw[start : end + 1])
    raise ValueError("No JSON object found in LLM response")


def _level_from_score(score: int) -> RiskLevel:
    if score <= 15:
        return "LOW"
    if score <= 35:
        return "MEDIUM"
    if score <= 55:
        return "HIGH"
    return "VERY_HIGH"


def _build_validation_fields(
    parsed: dict[str, Any],
    reachability: list[SourceReachability],
) -> tuple[MarketClassification, ValidationResult, ResolvabilityAssessment]:
    """Convert raw LLM JSON into typed validation fields, with defensive defaults."""

    # Classification
    cls_raw = parsed.get("classification", {})
    classification = MarketClassification(
        market_type=cls_raw.get("market_type", "UNKNOWN"),
        confidence=cls_raw.get("confidence", 0.0),
        detection_rationale=cls_raw.get("detection_rationale", ""),
    )

    # Validation
    val_raw = parsed.get("validation", {})
    checks_passed = val_raw.get("checks_passed", [])
    checks_failed = []
    for issue in val_raw.get("checks_failed", []):
        checks_failed.append(
            ValidationIssue(
                check_id=issue.get("check_id", "UNKNOWN"),
                severity=issue.get("severity", "warning"),
                message=issue.get("message", ""),
                suggestion=issue.get("suggestion", ""),
            )
        )

    # Add reachability-based validation issues
    for probe in reachability:
        if not probe.reachable:
            checks_failed.append(
                ValidationIssue(
                    check_id="SRC-REACH",
                    severity="warning",
                    message=f"Data source unreachable: {probe.url} — {probe.error}",
                    suggestion="This source may be blocked by Cloudflare or require authentication. "
                    "Consider using an alternative API-based source, or for news/geopolitical markets, "
                    "use news consensus (e.g., 'confirmed by at least 2 of: AP, Reuters, BBC').",
                )
            )

    validation = ValidationResult(
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )

    # Resolvability — add extra points for unreachable sources
    res_raw = parsed.get("resolvability", {})
    risk_factors = []
    for rf in res_raw.get("risk_factors", []):
        risk_factors.append(
            RiskFactor(
                factor=rf.get("factor", ""),
                points=rf.get("points", 0),
            )
        )

    unreachable_count = sum(1 for p in reachability if not p.reachable)
    if unreachable_count > 0:
        risk_factors.append(
            RiskFactor(
                factor=f"{unreachable_count} data source(s) unreachable (Cloudflare/paywall/timeout)",
                points=20 * unreachable_count,
            )
        )

    score = sum(rf.points for rf in risk_factors)

    resolvability = ResolvabilityAssessment(
        score=score,
        level=_level_from_score(score),
        risk_factors=risk_factors,
        recommendation=res_raw.get("recommendation", ""),
    )

    return classification, validation, resolvability


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/validate", response_model=ValidateResponse)
async def validate_market(request: ValidateRequest) -> ValidateResponse:
    """Validate and compile a market query in a single call.

    Runs three things in parallel:
      1. LLM validation (classify, check fields, assess resolvability)
      2. Prompt compilation (PromptSpec + ToolPlan)

    Then probes any data source URLs found by the validator to check
    reachability (Cloudflare, paywalls, timeouts).
    """

    override = build_llm_override(
        request.llm_provider, request.llm_model, agent_name=None
    )
    prompt_override = build_llm_override(
        request.llm_provider, request.llm_model, agent_name="prompt_engineer",
    )

    # --- Phase 1: Run validation LLM + prompt compilation in parallel ---

    async def _run_validation() -> dict[str, Any]:
        ctx = get_agent_context(with_llm=True, llm_override=override)
        user_prompt = _build_validate_user_prompt(request.user_input)
        llm_response = await asyncio.to_thread(
            ctx.llm.chat,
            [
                {"role": "system", "content": _VALIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return _parse_llm_response(llm_response.content)

    async def _run_prompt_compile() -> tuple[Any, Any, dict[str, Any]]:
        from agents.prompt_engineer import PromptEngineerLLM

        ctx = get_agent_context(with_llm=True, llm_override=prompt_override)
        agent = PromptEngineerLLM(strict_mode=request.strict_mode)
        result = await asyncio.to_thread(agent.run, ctx, request.user_input)
        return result.success, result.output, result.metadata or {}, result.error

    validation_task = _run_validation()
    prompt_task = _run_prompt_compile()
    results = await asyncio.gather(validation_task, prompt_task, return_exceptions=True)

    # Handle validation result
    validation_error: str | None = None
    parsed: dict[str, Any] = {}
    if isinstance(results[0], Exception):
        logger.error("Validation LLM call failed: %s", results[0])
        validation_error = str(results[0])
        parsed = {}
    else:
        parsed = results[0]

    # Handle prompt compilation result
    prompt_error: str | None = None
    prompt_spec_dict: dict[str, Any] | None = None
    tool_plan_dict: dict[str, Any] | None = None
    market_id: str | None = None
    prompt_metadata: dict[str, Any] = {}

    if isinstance(results[1], Exception):
        logger.error("Prompt compilation failed: %s", results[1])
        prompt_error = str(results[1])
    else:
        success, output, metadata, error = results[1]
        prompt_metadata = metadata
        if success and output is not None:
            prompt_spec, tool_plan = output
            prompt_spec_dict = prompt_spec.model_dump(mode="json")
            tool_plan_dict = tool_plan.model_dump(mode="json")
            market_id = prompt_spec.market.market_id
        else:
            prompt_error = error

    # --- Phase 2: Probe source reachability using URLs from prompt_spec ---

    source_urls = _extract_source_urls(prompt_spec_dict)
    reachability: list[SourceReachability] = []

    if source_urls:
        ctx_http = get_agent_context(with_http=True)
        reachability = await _probe_sources(ctx_http.http, source_urls)

    # --- Build response ---

    errors: list[str] = []
    if validation_error:
        errors.append(f"Validation: {validation_error}")
    if prompt_error:
        errors.append(f"Prompt compilation: {prompt_error}")

    if parsed:
        classification, validation, resolvability = _build_validation_fields(
            parsed, reachability
        )
    else:
        # Fallback if validation LLM failed entirely
        classification = MarketClassification(
            market_type="UNKNOWN", confidence=0.0, detection_rationale="Validation LLM call failed"
        )
        validation = ValidationResult()
        resolvability = ResolvabilityAssessment(
            score=0, level="LOW", recommendation="Validation could not be performed"
        )

    return ValidateResponse(
        ok=not errors,
        classification=classification,
        validation=validation,
        resolvability=resolvability,
        source_reachability=reachability,
        market_id=market_id,
        prompt_spec=prompt_spec_dict,
        tool_plan=tool_plan_dict,
        prompt_metadata=prompt_metadata,
        errors=errors,
    )
