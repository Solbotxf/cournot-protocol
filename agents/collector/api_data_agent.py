"""
CollectorAPIData — routes data requirements to structured API providers.

Matches requirements to providers (Open-Meteo weather, CoinMarketCap crypto,
etc.) via keyword matching with optional LLM fallback, calls the matched API,
and returns standard ``EvidenceBundle`` output.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING, Any

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

from .api_providers import (
    APIProviderConfig,
    get_provider_by_id,
    match_providers,
)
from .api_providers.base import APICallResult, APIMatchResult

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas.prompts import DataRequirement

logger = logging.getLogger(__name__)


class CollectorAPIData(BaseAgent):
    """Fetches data from structured APIs (weather, crypto, etc.).

    Matches requirements to API providers via keyword matching, with an
    optional LLM fallback for ambiguous requirements.
    """

    _name = "CollectorAPIData"
    _version = "v1"
    _capabilities = {AgentCapability.NETWORK}
    MAX_RETRIES = 2

    def __init__(
        self,
        *,
        provider_configs: dict[str, APIProviderConfig] | None = None,
        use_llm_matching: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._provider_configs = provider_configs
        self._use_llm_matching = use_llm_matching

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        ctx.info(f"CollectorAPIData executing plan {tool_plan.plan_id}")

        configs = self._build_configs(ctx)

        bundle = EvidenceBundle(
            bundle_id=f"apidata_{tool_plan.plan_id}",
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
                ctx.info(f"Requirement {req_id} not found in prompt_spec, skipping")
                continue

            evidence, call_record = self._collect_requirement(
                ctx, prompt_spec, requirement, configs
            )
            execution_log.add_call(call_record)
            bundle.add_item(evidence)

        bundle.collected_at = ctx.now()
        bundle.requirements_fulfilled = [
            item.requirement_id
            for item in bundle.items
            if item.success
        ]
        bundle.requirements_unfulfilled = [
            req_id
            for req_id in tool_plan.requirements
            if req_id not in bundle.requirements_fulfilled
        ]
        execution_log.ended_at = ctx.now().isoformat()

        verification = self._validate_output(bundle, tool_plan)

        return AgentResult(
            output=(bundle, execution_log),
            verification=verification,
            receipts=ctx.get_receipt_refs(),
            metadata={
                "collector": "api_data",
                "bundle_id": bundle.bundle_id,
                "items_collected": len(bundle.items),
            },
        )

    # ------------------------------------------------------------------
    # Per-requirement collection
    # ------------------------------------------------------------------

    def _collect_requirement(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
        configs: dict[str, APIProviderConfig],
    ) -> tuple[EvidenceItem, ToolCallRecord]:
        description = requirement.description
        req_id = requirement.requirement_id

        call_record = ToolCallRecord(
            tool=f"api_data:{req_id}",
            input={"requirement_id": req_id, "description": description},
            started_at=ctx.now().isoformat(),
        )

        # Phase 1: keyword matching
        matches = match_providers(description, configs)

        # Phase 2: LLM fallback
        if not matches and self._use_llm_matching and ctx.llm is not None:
            matches = self._llm_match(ctx, description, configs)

        if not matches:
            evidence = EvidenceItem(
                evidence_id=f"apidata_{req_id}_{uuid.uuid4().hex[:8]}",
                requirement_id=req_id,
                provenance=Provenance(
                    source_id="api_data_no_match",
                    source_uri="n/a",
                    tier=3,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error="No API provider matched this requirement",
            )
            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {"success": False, "error": "no_match"}
            return evidence, call_record

        best = matches[0]
        provider = get_provider_by_id(best.provider_id)
        if provider is None:
            evidence = EvidenceItem(
                evidence_id=f"apidata_{req_id}_{uuid.uuid4().hex[:8]}",
                requirement_id=req_id,
                provenance=Provenance(
                    source_id=best.provider_id,
                    source_uri="n/a",
                    tier=3,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Provider '{best.provider_id}' not found in registry",
            )
            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {"success": False, "error": "provider_not_found"}
            return evidence, call_record

        # Merge suggested params with provider config
        cfg = configs.get(provider.provider_id, APIProviderConfig())
        params = dict(best.suggested_params)
        if cfg.api_key:
            params["api_key"] = cfg.api_key
        if cfg.base_url:
            params["base_url"] = cfg.base_url
        params.update(cfg.extra)

        # LLM-driven parameter extraction
        params = self._llm_extract_params(ctx, provider, description, params)

        # Validate params before making the API call
        ok, reason = provider.validate_params(params)
        if not ok:
            evidence = EvidenceItem(
                evidence_id=f"apidata_{req_id}_{uuid.uuid4().hex[:8]}",
                requirement_id=req_id,
                provenance=Provenance(
                    source_id=provider.provider_id,
                    source_uri="n/a",
                    tier=3,
                    fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Insufficient params: {reason}",
            )
            call_record.ended_at = ctx.now().isoformat()
            call_record.output = {
                "success": False,
                "error": "invalid_params",
                "reason": reason,
            }
            return evidence, call_record

        ctx.info(
            f"Calling {provider.display_name} for requirement {req_id} "
            f"(confidence={best.confidence:.2f}, keywords={best.matched_keywords})"
        )

        result: APICallResult = provider.call(cfg, description, params)

        # LLM-driven response interpretation
        interpretation = self._llm_interpret_response(
            ctx, description, provider, result
        )
        interp_text = ""
        interp_confidence = 0.0
        if interpretation is not None:
            result.extracted_fields["interpretation"] = interpretation.get(
                "interpretation"
            )
            result.extracted_fields["interpretation_confidence"] = (
                interpretation.get("confidence")
            )
            interp_text = interpretation.get("interpretation", "")
            interp_confidence = interpretation.get("confidence", 0.0)
            relevant = interpretation.get("relevant_data")
            if relevant and isinstance(relevant, dict):
                result.extracted_fields["relevant_data"] = relevant

        # Build canonical extracted_fields with provider-specific data nested
        provider_fields = dict(result.extracted_fields)
        source_uri = result.source_uri or "n/a"

        # Parse domain from source_uri for evidence_sources
        api_domain = None
        if source_uri.startswith("http"):
            from urllib.parse import urlparse as _urlparse
            api_domain = (_urlparse(source_uri).netloc or "").lower().lstrip("www.") or None

        canonical_fields: dict[str, Any] = {
            # --- Canonical fields (same as CollectorOpenSearch) ---
            "outcome": "Unresolved",  # API data doesn't resolve markets directly
            "reason": interp_text or f"Data fetched from {provider.display_name}",
            "evidence_sources": [{
                "url": source_uri,
                "source_id": "[1]",
                "domain_name": api_domain or provider.display_name,
                "credibility_tier": 1,
                "key_fact": interp_text[:300] if interp_text else f"API data from {provider.display_name}",
                "supports": "N/A",
                "date_published": None,
                "is_required_data_source": True,
            }],
            "grounding_search_queries": [],
            "grounding_source_count": 1,
            "confidence_score": interp_confidence if interp_confidence else (0.8 if result.success else 0.0),
            "resolution_status": "RESOLVED" if result.success else "UNRESOLVED",
            "pass_used": f"api_data:{provider.provider_id}",
            "data_source_domains_required": provider.domains[:5] if hasattr(provider, 'domains') else [],
            # --- Provider-specific fields ---
            "api_provider": provider.provider_id,
            "api_data": provider_fields,
        }

        evidence = EvidenceItem(
            evidence_id=f"apidata_{req_id}_{uuid.uuid4().hex[:8]}",
            requirement_id=req_id,
            provenance=Provenance(
                source_id=provider.provider_id,
                source_uri=source_uri,
                tier=3,
                fetched_at=ctx.now(),
                content_hash=(
                    provider._content_hash(result.raw_response)
                    if result.raw_response
                    else None
                ),
            ),
            raw_content=result.raw_response,
            content_type="application/json",
            parsed_value=result.data if result.success else None,
            success=result.success,
            error=result.error,
            status_code=result.status_code,
            extracted_fields=canonical_fields,
        )

        call_record.ended_at = ctx.now().isoformat()
        call_record.output = {
            "success": result.success,
            "provider": provider.provider_id,
            "evidence_id": evidence.evidence_id,
            "status_code": result.status_code,
        }

        return evidence, call_record

    # ------------------------------------------------------------------
    # LLM-driven parameter extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())

        # Try bare JSON (first { to last })
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])

        # Last resort: try whole text
        return json.loads(text.strip())

    def _llm_extract_params(
        self,
        ctx: "AgentContext",
        provider: "BaseAPIProvider",
        description: str,
        hint_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Use the LLM to extract API parameters from the requirement description.

        Falls back to *hint_params* when no LLM is available or extraction fails.
        """
        if not provider.param_schema or ctx.llm is None:
            return hint_params

        from core.llm.determinism import DecodingPolicy

        schema_str = json.dumps(provider.param_schema, indent=2)
        hint_str = json.dumps(hint_params, indent=2) if hint_params else "{}"
        today = ctx.now().strftime("%Y-%m-%d")
        prompt = (
            "Extract the API parameters from the following data requirement.\n\n"
            f"Today's date: {today}\n\n"
            f"Requirement: {description}\n\n"
            f"Parameter schema:\n{schema_str}\n\n"
        )
        if provider.param_extraction_hint:
            prompt += f"Instructions:\n{provider.param_extraction_hint}\n\n"
        prompt += (
            f"Default/hint values (use these if you cannot determine better values):\n"
            f"{hint_str}\n\n"
            "Respond with ONLY a JSON object containing the extracted parameters."
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = ctx.llm.chat(
                    messages=messages,
                    policy=DecodingPolicy(temperature=0.0, max_tokens=256),
                )
                parsed = self._extract_json(response.content)
                # Merge: LLM values take precedence over hints
                merged = dict(hint_params)
                merged.update(parsed)
                return merged
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < self.MAX_RETRIES:
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The JSON was invalid: {e}. "
                            "Please return ONLY a valid JSON object."
                        ),
                    })
                else:
                    ctx.warning(
                        f"LLM param extraction failed after {self.MAX_RETRIES + 1} "
                        f"attempts: {e}"
                    )
                    return hint_params

        return hint_params  # unreachable but satisfies type checker

    def _llm_interpret_response(
        self,
        ctx: "AgentContext",
        description: str,
        provider: "BaseAPIProvider",
        result: "APICallResult",
    ) -> dict[str, Any] | None:
        """Use the LLM to interpret API response data for the requirement.

        Returns a dict with ``interpretation``, ``relevant_data``, and
        ``confidence`` keys, or ``None`` on failure or when LLM is unavailable.
        """
        if ctx.llm is None or not result.success:
            return None

        from core.llm.determinism import DecodingPolicy

        fields_str = json.dumps(result.extracted_fields, default=str)
        if len(fields_str) > 2000:
            fields_str = fields_str[:2000] + "..."

        prompt = (
            "Interpret the following API response data in the context of "
            "the data requirement.\n\n"
            f"Requirement: {description}\n\n"
            f"Provider: {provider.display_name}\n\n"
            f"Extracted fields:\n{fields_str}\n\n"
            "Respond with a JSON object containing:\n"
            '- "interpretation": a concise natural-language summary of what '
            "the data means for the requirement\n"
            '- "relevant_data": a dict of the most relevant key-value pairs\n'
            '- "confidence": a float 0.0-1.0 indicating how well the data '
            "answers the requirement\n"
        )

        try:
            response = ctx.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                policy=DecodingPolicy(temperature=0.0, max_tokens=512),
            )
            return self._extract_json(response.content)
        except Exception:  # noqa: BLE001
            logger.debug("LLM response interpretation failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # LLM fallback matching
    # ------------------------------------------------------------------

    def _llm_match(
        self,
        ctx: "AgentContext",
        description: str,
        configs: dict[str, APIProviderConfig],
    ) -> list[APIMatchResult]:
        """Use the LLM to decide which provider (if any) fits."""
        from .api_providers import get_provider_registry

        available = []
        for p in get_provider_registry():
            cfg = configs.get(p.provider_id, APIProviderConfig())
            if p.is_available(cfg):
                available.append(
                    {
                        "provider_id": p.provider_id,
                        "display_name": p.display_name,
                        "domains": p.domains,
                        "keywords": p.keywords[:10],
                    }
                )

        if not available:
            return []

        prompt = (
            "Given the following data requirement and available API providers, "
            "return the provider_id that best matches. If none match, return "
            '"none".\n\n'
            f"Requirement: {description}\n\n"
            f"Providers:\n{json.dumps(available, indent=2)}\n\n"
            "Respond with ONLY the provider_id string (or \"none\")."
        )

        try:
            from core.llm.determinism import DecodingPolicy

            response = ctx.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                policy=DecodingPolicy(temperature=0.0, max_tokens=64),
            )
            chosen = response.content.strip().strip('"').strip("'")
            if chosen == "none" or not chosen:
                return []

            provider = get_provider_by_id(chosen)
            if provider is None:
                return []

            # Re-run match to extract params
            match = provider.match_requirement(description)
            if match is None:
                match = APIMatchResult(
                    provider_id=chosen,
                    confidence=0.5,
                    matched_keywords=["llm_match"],
                )
            else:
                match.confidence = max(match.confidence, 0.5)
            return [match]
        except Exception:  # noqa: BLE001
            logger.debug("LLM matching failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_output(
        bundle: EvidenceBundle, tool_plan: ToolPlan
    ) -> VerificationResult:
        checks: list[CheckResult] = []

        fulfilled = set(bundle.requirements_fulfilled)
        requested = set(tool_plan.requirements)

        if fulfilled == requested:
            checks.append(
                CheckResult.passed(
                    "api_data_completeness",
                    f"All {len(requested)} requirements fulfilled",
                )
            )
        elif fulfilled:
            checks.append(
                CheckResult.warning(
                    "api_data_completeness",
                    f"Fulfilled {len(fulfilled)}/{len(requested)} requirements",
                    details={"missing": list(requested - fulfilled)},
                )
            )
        else:
            checks.append(
                CheckResult.failed(
                    "api_data_completeness",
                    "No requirements fulfilled",
                )
            )

        success_items = [i for i in bundle.items if i.success]
        if success_items:
            checks.append(
                CheckResult.passed(
                    "api_data_items",
                    f"Collected {len(success_items)} evidence items",
                )
            )

        ok = any(c.ok for c in checks)
        if ok:
            return VerificationResult.success(checks)
        return VerificationResult.failure(checks)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _build_configs(
        self, ctx: "AgentContext"
    ) -> dict[str, APIProviderConfig]:
        """Build provider configs from explicit settings + env vars."""
        if self._provider_configs is not None:
            return dict(self._provider_configs)

        configs: dict[str, APIProviderConfig] = {}

        # Open-Meteo (free, always enabled)
        configs["open_meteo"] = APIProviderConfig(enabled=True)

        # CoinGecko (free, always enabled)
        configs["coingecko"] = APIProviderConfig(enabled=True)

        # Binance (free, optional key for higher rate limits)
        binance_key = os.getenv("BINANCE_API_KEY", "")
        configs["binance"] = APIProviderConfig(
            enabled=True,
            api_key=binance_key or None,
        )

        # CoinMarketCap (needs key)
        cmc_key = os.getenv("CMC_PRO_API_KEY", "")
        configs["coinmarketcap"] = APIProviderConfig(
            enabled=bool(cmc_key),
            api_key=cmc_key or None,
        )

        # Overlay from RuntimeConfig if available
        if ctx.config is not None:
            api_data_cfg = getattr(ctx.config, "api_data", None)
            if api_data_cfg is not None:
                for pid, pcfg in api_data_cfg.providers.items():
                    configs[pid] = APIProviderConfig(
                        enabled=pcfg.enabled,
                        api_key=pcfg.api_key,
                        base_url=pcfg.base_url,
                        timeout=pcfg.timeout,
                        extra=pcfg.extra,
                    )
                self._use_llm_matching = api_data_cfg.use_llm_matching

        return configs
