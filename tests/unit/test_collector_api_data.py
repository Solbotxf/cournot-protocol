"""
Tests for CollectorAPIData, API providers, and the provider registry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agents.collector.api_providers.base import (
    APICallResult,
    APIMatchResult,
    APIProviderConfig,
    BaseAPIProvider,
)
from agents.collector.api_providers.coinmarketcap import CoinMarketCapProvider
from agents.collector.api_providers.open_meteo import OpenMeteoProvider
from agents.collector.api_providers import (
    get_provider_by_id,
    get_provider_registry,
    match_providers,
    register_provider,
)
from agents.collector.api_data_agent import CollectorAPIData
from agents.context import AgentContext
from core.schemas import (
    DataRequirement,
    DisputePolicy,
    EvidenceBundle,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    ToolPlan,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

_NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_prompt_spec(
    requirement_id: str = "req_001",
    description: str = "Get current weather in London",
) -> PromptSpec:
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_api_test",
            question="Test question",
            event_definition="test event",
            resolution_deadline=_NOW,
            resolution_window=ResolutionWindow(start=_NOW, end=_NOW),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="rule", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="test",
            predicate="test predicate",
            threshold="n/a",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id=requirement_id,
                description=description,
                source_targets=[],
                deferred_source_discovery=True,
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
        ],
    )


def _make_tool_plan(requirements: list[str] | None = None) -> ToolPlan:
    return ToolPlan(
        plan_id="plan_api_test",
        requirements=requirements or ["req_001"],
    )


def _make_ctx() -> AgentContext:
    return AgentContext.create_minimal()


# ---------------------------------------------------------------
# Provider unit tests — OpenMeteo
# ---------------------------------------------------------------


class TestOpenMeteoProvider:
    def setup_method(self):
        self.provider = OpenMeteoProvider()

    def test_provider_id(self):
        assert self.provider.provider_id == "open_meteo"
        assert self.provider.requires_api_key is False

    def test_match_weather_terms(self):
        result = self.provider.match_requirement("What is the current weather in Paris?")
        assert result is not None
        assert result.provider_id == "open_meteo"
        assert result.confidence > 0
        assert "weather" in result.matched_keywords

    def test_match_temperature_terms(self):
        result = self.provider.match_requirement("Current temperature and wind speed in London")
        assert result is not None
        assert "temperature" in result.matched_keywords

    def test_no_match_unrelated(self):
        result = self.provider.match_requirement("What is the stock price of AAPL?")
        assert result is None

    def test_extract_city_london(self):
        result = self.provider.match_requirement("What is the weather in London?")
        assert result is not None
        assert "latitude" in result.suggested_params
        assert abs(result.suggested_params["latitude"] - 51.5074) < 0.01
        assert abs(result.suggested_params["longitude"] - (-0.1278)) < 0.01

    def test_extract_city_tokyo(self):
        result = self.provider.match_requirement("Tokyo forecast today")
        assert result is not None
        assert abs(result.suggested_params["latitude"] - 35.6762) < 0.01

    def test_extract_latlon_pattern(self):
        result = self.provider.match_requirement(
            "Weather at latitude 48.85 longitude 2.35"
        )
        assert result is not None
        assert abs(result.suggested_params["latitude"] - 48.85) < 0.01
        assert abs(result.suggested_params["longitude"] - 2.35) < 0.01

    def test_build_request(self):
        url, headers, params = self.provider.build_request(
            "weather", {"latitude": 51.5, "longitude": -0.1}
        )
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert params["latitude"] == 51.5
        assert params["longitude"] == -0.1
        assert params["current_weather"] == "true"

    def test_parse_response(self):
        raw = {
            "current_weather": {
                "temperature": 15.2,
                "windspeed": 12.5,
                "winddirection": 180,
                "weathercode": 1,
                "time": "2026-03-01T12:00",
            }
        }
        fields = self.provider.parse_response(raw)
        assert fields["temperature_c"] == 15.2
        assert fields["windspeed_kmh"] == 12.5
        assert fields["winddirection_deg"] == 180
        assert fields["weathercode"] == 1

    def test_is_available_no_key_needed(self):
        cfg = APIProviderConfig(enabled=True)
        assert self.provider.is_available(cfg) is True

    def test_is_available_disabled(self):
        cfg = APIProviderConfig(enabled=False)
        assert self.provider.is_available(cfg) is False


# ---------------------------------------------------------------
# Provider unit tests — CoinMarketCap
# ---------------------------------------------------------------


class TestCoinMarketCapProvider:
    def setup_method(self):
        self.provider = CoinMarketCapProvider()

    def test_provider_id(self):
        assert self.provider.provider_id == "coinmarketcap"
        assert self.provider.requires_api_key is True

    def test_match_crypto_terms(self):
        result = self.provider.match_requirement("What is the current Bitcoin price?")
        assert result is not None
        assert result.provider_id == "coinmarketcap"
        assert "bitcoin" in result.matched_keywords

    def test_match_btc_extracts_symbol(self):
        result = self.provider.match_requirement(
            "Get the current crypto price of Bitcoin and Ethereum"
        )
        assert result is not None
        assert "BTC" in result.suggested_params.get("symbols", [])
        assert "ETH" in result.suggested_params.get("symbols", [])

    def test_no_match_unrelated(self):
        result = self.provider.match_requirement("What is the weather in London?")
        assert result is None

    def test_extract_single_symbol(self):
        symbols = CoinMarketCapProvider._extract_symbols("Current Solana price")
        assert "SOL" in symbols

    def test_extract_uppercase_ticker(self):
        symbols = CoinMarketCapProvider._extract_symbols(
            "Check the BTC and ETH markets"
        )
        assert "BTC" in symbols
        assert "ETH" in symbols

    def test_build_request(self):
        url, headers, params = self.provider.build_request(
            "bitcoin price",
            {"symbols": ["BTC"], "api_key": "test_key"},
        )
        assert "quotes/latest" in url
        assert headers["X-CMC_PRO_API_KEY"] == "test_key"
        assert params["symbol"] == "BTC"
        assert params["convert"] == "USD"

    def test_parse_response(self):
        raw = {
            "data": {
                "BTC": {
                    "quote": {
                        "USD": {
                            "price": 95000.0,
                            "market_cap": 1800000000000,
                            "volume_24h": 35000000000,
                            "percent_change_24h": 2.5,
                        }
                    }
                }
            }
        }
        fields = self.provider.parse_response(raw)
        assert "BTC" in fields
        assert fields["BTC"]["price_usd"] == 95000.0
        assert fields["BTC"]["market_cap_usd"] == 1800000000000

    def test_is_available_needs_key(self):
        cfg = APIProviderConfig(enabled=True, api_key=None)
        assert self.provider.is_available(cfg) is False

    def test_is_available_with_key(self):
        cfg = APIProviderConfig(enabled=True, api_key="test_key")
        assert self.provider.is_available(cfg) is True


# ---------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------


class TestProviderRegistry:
    def test_match_weather_returns_open_meteo(self):
        configs = {
            "open_meteo": APIProviderConfig(enabled=True),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key="k"),
        }
        results = match_providers("What is the weather in London?", configs)
        assert len(results) > 0
        assert results[0].provider_id == "open_meteo"

    def test_match_crypto_returns_coinmarketcap(self):
        configs = {
            "open_meteo": APIProviderConfig(enabled=True),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key="k"),
        }
        results = match_providers("Current Bitcoin price in USD", configs)
        assert len(results) > 0
        assert results[0].provider_id == "coinmarketcap"

    def test_disabled_provider_excluded(self):
        configs = {
            "open_meteo": APIProviderConfig(enabled=False),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key="k"),
        }
        results = match_providers("weather in Paris", configs)
        assert all(r.provider_id != "open_meteo" for r in results)

    def test_missing_key_excludes_provider(self):
        configs = {
            "open_meteo": APIProviderConfig(enabled=True),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key=None),
        }
        results = match_providers("Bitcoin price", configs)
        assert all(r.provider_id != "coinmarketcap" for r in results)

    def test_unknown_requirement_returns_empty(self):
        configs = {
            "open_meteo": APIProviderConfig(enabled=True),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key="k"),
        }
        results = match_providers("Who won the 2024 presidential election?", configs)
        assert len(results) == 0

    def test_get_provider_by_id(self):
        p = get_provider_by_id("open_meteo")
        assert p is not None
        assert p.provider_id == "open_meteo"

    def test_get_provider_by_id_unknown(self):
        p = get_provider_by_id("nonexistent_provider")
        assert p is None

    def test_registry_contains_defaults(self):
        providers = get_provider_registry()
        ids = [p.provider_id for p in providers]
        assert "open_meteo" in ids
        assert "coinmarketcap" in ids
        assert "coingecko" in ids
        assert "binance" in ids

    def test_match_crypto_returns_coingecko_when_no_cmc_key(self):
        configs = {
            "open_meteo": APIProviderConfig(enabled=True),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key=None),
            "coingecko": APIProviderConfig(enabled=True),
        }
        results = match_providers("Current Bitcoin price in USD", configs)
        assert len(results) > 0
        assert results[0].provider_id == "coingecko"


# ---------------------------------------------------------------
# Provider unit tests — CoinGecko
# ---------------------------------------------------------------


class TestCoinGeckoProvider:
    def setup_method(self):
        from agents.collector.api_providers.coingecko import CoinGeckoProvider
        self.provider = CoinGeckoProvider()

    def test_provider_id(self):
        assert self.provider.provider_id == "coingecko"
        assert self.provider.requires_api_key is False

    def test_match_crypto_terms(self):
        result = self.provider.match_requirement("What is the current Bitcoin price?")
        assert result is not None
        assert result.provider_id == "coingecko"
        assert "bitcoin" in result.matched_keywords or "price" in result.matched_keywords

    def test_extract_coin_id_bitcoin(self):
        result = self.provider.match_requirement("Current Bitcoin price in USD")
        assert result is not None
        assert result.suggested_params.get("coin_id") == "bitcoin"

    def test_extract_coin_id_ethereum(self):
        result = self.provider.match_requirement("Ethereum crypto price today")
        assert result is not None
        assert result.suggested_params.get("coin_id") == "ethereum"

    def test_extract_coin_id_xrp(self):
        result = self.provider.match_requirement("XRP price on March 10?")
        assert result is not None
        assert result.suggested_params.get("coin_id") == "ripple"

    def test_no_match_unrelated(self):
        result = self.provider.match_requirement("What is the weather in London?")
        assert result is None

    def test_build_request_current(self):
        url, headers, params = self.provider.build_request(
            "bitcoin price", {"coin_id": "bitcoin", "mode": "current"}
        )
        assert url == "https://api.coingecko.com/api/v3/simple/price"
        assert params["ids"] == "bitcoin"
        assert params["vs_currencies"] == "usd"

    def test_build_request_historical(self):
        url, headers, params = self.provider.build_request(
            "bitcoin price on March 10",
            {"coin_id": "bitcoin", "mode": "historical", "date": "10-03-2026"},
        )
        assert "history" in url
        assert "bitcoin" in url
        assert params["date"] == "10-03-2026"

    def test_parse_historical_response(self):
        raw = {
            "id": "bitcoin",
            "market_data": {
                "current_price": {"usd": 68459.32},
                "market_cap": {"usd": 1350000000000},
                "total_volume": {"usd": 25000000000},
            },
        }
        fields = self.provider.parse_response(raw)
        assert fields["BTC"]["price_usd"] == 68459.32
        assert fields["price_usd"] == 68459.32
        assert fields["symbol"] == "BTC"

    def test_parse_current_response(self):
        raw = {
            "bitcoin": {
                "usd": 95000.0,
                "usd_market_cap": 1800000000000,
                "usd_24h_vol": 35000000000,
                "usd_24h_change": 2.5,
            }
        }
        fields = self.provider.parse_response(raw)
        assert fields["BTC"]["price_usd"] == 95000.0
        assert fields["BTC"]["percent_change_24h"] == 2.5

    def test_validate_params_ok(self):
        ok, reason = self.provider.validate_params(
            {"coin_id": "bitcoin", "mode": "current"}
        )
        assert ok is True

    def test_validate_params_missing_coin(self):
        ok, reason = self.provider.validate_params({"mode": "current"})
        assert ok is False

    def test_validate_params_historical_missing_date(self):
        ok, reason = self.provider.validate_params(
            {"coin_id": "bitcoin", "mode": "historical"}
        )
        assert ok is False

    def test_validate_params_historical_ok(self):
        ok, reason = self.provider.validate_params(
            {"coin_id": "bitcoin", "mode": "historical", "date": "10-03-2026"}
        )
        assert ok is True

    def test_is_available_no_key_needed(self):
        cfg = APIProviderConfig(enabled=True)
        assert self.provider.is_available(cfg) is True


# ---------------------------------------------------------------
# Collector integration tests (mocked HTTP)
# ---------------------------------------------------------------


class TestCollectorAPIData:
    """Integration tests with mocked HTTP calls."""

    def _mock_open_meteo_response(self) -> httpx.Response:
        data = {
            "current_weather": {
                "temperature": 18.5,
                "windspeed": 10.0,
                "winddirection": 270,
                "weathercode": 3,
                "time": "2026-03-01T12:00",
            }
        }
        return httpx.Response(
            status_code=200,
            json=data,
            request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
        )

    def _mock_cmc_response(self) -> httpx.Response:
        data = {
            "data": {
                "BTC": {
                    "quote": {
                        "USD": {
                            "price": 95000.0,
                            "market_cap": 1800000000000,
                            "volume_24h": 35000000000,
                            "percent_change_24h": 2.5,
                        }
                    }
                }
            }
        }
        return httpx.Response(
            status_code=200,
            json=data,
            request=httpx.Request(
                "GET", "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            ),
        )

    @patch("httpx.get")
    def test_weather_collection(self, mock_get: MagicMock):
        mock_get.return_value = self._mock_open_meteo_response()

        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(
            requirement_id="req_weather",
            description="Get current weather in London",
        )
        tool_plan = _make_tool_plan(requirements=["req_weather"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        assert result.success
        bundle, log = result.output
        assert isinstance(bundle, EvidenceBundle)
        assert len(bundle.items) == 1

        item = bundle.items[0]
        assert item.success
        assert item.requirement_id == "req_weather"
        assert item.extracted_fields["api_data"]["temperature_c"] == 18.5
        assert item.provenance.source_id == "open_meteo"
        assert item.provenance.tier == 3
        assert "req_weather" in bundle.requirements_fulfilled

    @patch("httpx.get")
    def test_crypto_collection(self, mock_get: MagicMock):
        mock_get.return_value = self._mock_cmc_response()

        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(
            requirement_id="req_crypto",
            description="Get current Bitcoin crypto price",
        )
        tool_plan = _make_tool_plan(requirements=["req_crypto"])

        configs = {
            "open_meteo": APIProviderConfig(enabled=True),
            "coinmarketcap": APIProviderConfig(enabled=True, api_key="test_key"),
        }
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        assert result.success
        bundle, log = result.output
        assert len(bundle.items) == 1

        item = bundle.items[0]
        assert item.success
        assert item.requirement_id == "req_crypto"
        assert "BTC" in item.extracted_fields["api_data"]
        assert item.extracted_fields["api_data"]["BTC"]["price_usd"] == 95000.0
        assert item.provenance.source_id == "coinmarketcap"

    def test_no_match_returns_error_evidence(self):
        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(
            requirement_id="req_unknown",
            description="Who won the 2024 presidential election?",
        )
        tool_plan = _make_tool_plan(requirements=["req_unknown"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        assert result.success  # agent itself succeeds, but evidence is failure
        bundle, log = result.output
        assert len(bundle.items) == 1
        assert bundle.items[0].success is False
        assert "No API provider matched" in bundle.items[0].error
        assert "req_unknown" in bundle.requirements_unfulfilled

    @patch("httpx.get")
    def test_http_error_produces_failed_evidence(self, mock_get: MagicMock):
        mock_get.return_value = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
        )

        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(
            requirement_id="req_fail",
            description="Weather forecast in London",
        )
        tool_plan = _make_tool_plan(requirements=["req_fail"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, log = result.output
        assert len(bundle.items) == 1
        assert bundle.items[0].success is False
        assert "500" in bundle.items[0].error

    @patch("httpx.get")
    def test_multiple_requirements(self, mock_get: MagicMock):
        """Collector handles multiple requirements in one plan."""
        mock_get.return_value = self._mock_open_meteo_response()

        ctx = _make_ctx()
        prompt_spec = PromptSpec(
            market=MarketSpec(
                market_id="mk_multi",
                question="Test",
                event_definition="test",
                resolution_deadline=_NOW,
                resolution_window=ResolutionWindow(start=_NOW, end=_NOW),
                resolution_rules=ResolutionRules(rules=[
                    ResolutionRule(rule_id="R1", description="r", priority=100),
                ]),
                dispute_policy=DisputePolicy(dispute_window_seconds=86400),
            ),
            prediction_semantics=PredictionSemantics(
                target_entity="test",
                predicate="test",
                threshold="n/a",
            ),
            data_requirements=[
                DataRequirement(
                    requirement_id="req_a",
                    description="Weather in Paris",
                    source_targets=[],
                    deferred_source_discovery=True,
                    selection_policy=SelectionPolicy(
                        strategy="single_best", min_sources=1, max_sources=1, quorum=1
                    ),
                ),
                DataRequirement(
                    requirement_id="req_b",
                    description="Temperature in Tokyo",
                    source_targets=[],
                    deferred_source_discovery=True,
                    selection_policy=SelectionPolicy(
                        strategy="single_best", min_sources=1, max_sources=1, quorum=1
                    ),
                ),
            ],
        )
        tool_plan = _make_tool_plan(requirements=["req_a", "req_b"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, log = result.output
        assert len(bundle.items) == 2
        assert all(item.success for item in bundle.items)
        assert len(log.calls) == 2

    def test_llm_fallback_mocked(self):
        """LLM fallback picks a provider when keywords don't match."""
        # LLM response 1: provider match → "open_meteo"
        # LLM response 2: param extraction JSON
        param_json = json.dumps({
            "latitude": 52.52,
            "longitude": 13.405,
            "mode": "current",
        })
        # LLM response 3: interpretation JSON
        interp_json = json.dumps({
            "interpretation": "Current temperature in Berlin is 5°C.",
            "relevant_data": {"temperature_c": 5},
            "confidence": 0.9,
        })
        ctx = AgentContext.create_mock(
            llm_responses=["open_meteo", param_json, interp_json]
        )

        prompt_spec = _make_prompt_spec(
            requirement_id="req_llm",
            description="What are atmospheric conditions in Berlin?",
        )
        tool_plan = _make_tool_plan(requirements=["req_llm"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=True
        )

        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(
                status_code=200,
                json={"current_weather": {"temperature": 5}},
                request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
            )
            result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        assert len(bundle.items) == 1
        # Should have successfully matched via LLM fallback
        assert bundle.items[0].provenance.source_id == "open_meteo"

    def test_verification_all_fulfilled(self):
        """Verification reports success when all requirements are fulfilled."""
        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(description="Weather in London")
        tool_plan = _make_tool_plan()

        with patch("httpx.get") as mock_get:
            mock_get.return_value = httpx.Response(
                status_code=200,
                json={"current_weather": {"temperature": 10}},
                request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
            )
            configs = {"open_meteo": APIProviderConfig(enabled=True)}
            collector = CollectorAPIData(
                provider_configs=configs, use_llm_matching=False
            )
            result = collector.run(ctx, prompt_spec, tool_plan)

        assert result.verification is not None
        assert result.verification.ok is True

    def test_verification_none_fulfilled(self):
        """Verification reports failure when no requirements are fulfilled."""
        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(
            description="Who won the 2024 election?"
        )
        tool_plan = _make_tool_plan()

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        assert result.verification is not None
        assert result.verification.ok is False


# ---------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------


class TestImports:
    def test_import_from_collector_package(self):
        from agents.collector import CollectorAPIData as C
        assert C._name == "CollectorAPIData"

    def test_import_from_api_data_agent(self):
        from agents.collector.api_data_agent import CollectorAPIData
        assert CollectorAPIData is not None


# ---------------------------------------------------------------
# Airport / alias location extraction tests
# ---------------------------------------------------------------


class TestOpenMeteoAirportExtraction:
    def setup_method(self):
        self.provider = OpenMeteoProvider()

    def test_icao_code_saez(self):
        result = self.provider._extract_location("temperature at SAEZ")
        assert abs(result["latitude"] - (-34.8222)) < 0.01
        assert abs(result["longitude"] - (-58.5358)) < 0.01

    def test_minister_pistarini_alias(self):
        result = self.provider._extract_location(
            "Minister Pistarini Airport Station"
        )
        assert abs(result["latitude"] - (-34.8222)) < 0.01
        assert abs(result["longitude"] - (-58.5358)) < 0.01

    def test_ezeiza_alias(self):
        result = self.provider._extract_location("ezeiza temperature")
        assert abs(result["latitude"] - (-34.8222)) < 0.01
        assert abs(result["longitude"] - (-58.5358)) < 0.01

    def test_heathrow_alias(self):
        result = self.provider._extract_location("weather at Heathrow airport")
        assert abs(result["latitude"] - 51.47) < 0.01
        assert abs(result["longitude"] - (-0.4543)) < 0.01

    def test_city_still_works(self):
        result = self.provider._extract_location("weather in London")
        assert abs(result["latitude"] - 51.5074) < 0.01


# ---------------------------------------------------------------
# Strict matching tests
# ---------------------------------------------------------------


class TestOpenMeteoStrictMatching:
    def setup_method(self):
        self.provider = OpenMeteoProvider()

    def test_no_location_matches_with_lower_confidence(self):
        """Weather keywords present but no extractable location → match with lower confidence."""
        result = self.provider.match_requirement("temperature recorded")
        assert result is not None
        assert result.provider_id == "open_meteo"
        assert "latitude" not in result.suggested_params

    def test_with_city_matches(self):
        result = self.provider.match_requirement("weather in London")
        assert result is not None
        assert result.provider_id == "open_meteo"

    def test_with_airport_matches(self):
        result = self.provider.match_requirement(
            "temperature at Minister Pistarini Airport"
        )
        assert result is not None
        assert abs(result.suggested_params["latitude"] - (-34.8222)) < 0.01


# ---------------------------------------------------------------
# Historical request building tests
# ---------------------------------------------------------------


class TestOpenMeteoHistoricalRequest:
    def setup_method(self):
        self.provider = OpenMeteoProvider()

    def test_historical_uses_archive_url(self):
        params = {
            "latitude": -34.8222,
            "longitude": -58.5358,
            "mode": "historical",
            "start_date": "2026-03-10",
            "end_date": "2026-03-10",
        }
        url, headers, query = self.provider.build_request("test", params)
        assert url == "https://archive-api.open-meteo.com/v1/archive"
        assert query["start_date"] == "2026-03-10"
        assert query["end_date"] == "2026-03-10"
        assert "daily" in query
        assert "temperature_2m_max" in query["daily"]

    def test_current_uses_forecast_url(self):
        params = {
            "latitude": 51.5,
            "longitude": -0.1,
            "mode": "current",
        }
        url, headers, query = self.provider.build_request("test", params)
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert query["current_weather"] == "true"


# ---------------------------------------------------------------
# Historical response parsing tests
# ---------------------------------------------------------------


class TestOpenMeteoHistoricalParsing:
    def setup_method(self):
        self.provider = OpenMeteoProvider()

    def test_parse_daily_response(self):
        data = {
            "daily": {
                "time": ["2026-03-10"],
                "temperature_2m_max": [27.0],
                "temperature_2m_min": [18.5],
                "precipitation_sum": [0.0],
            }
        }
        fields = self.provider.parse_response(data)
        assert fields["temperature_2m_max"] == 27.0
        assert fields["temperature_2m_min"] == 18.5
        assert fields["precipitation_sum"] == 0.0
        assert fields["date"] == "2026-03-10"

    def test_parse_current_response_still_works(self):
        data = {
            "current_weather": {
                "temperature": 15.0,
                "windspeed": 10.0,
                "winddirection": 180,
                "weathercode": 1,
                "time": "2026-03-12T12:00",
            }
        }
        fields = self.provider.parse_response(data)
        assert fields["temperature_c"] == 15.0
        assert fields["windspeed_kmh"] == 10.0


# ---------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------


class TestOpenMeteoValidation:
    def setup_method(self):
        self.provider = OpenMeteoProvider()

    def test_valid_current_params(self):
        ok, reason = self.provider.validate_params(
            {"latitude": 51.5, "longitude": -0.1, "mode": "current"}
        )
        assert ok is True
        assert reason == ""

    def test_missing_location_fails(self):
        ok, reason = self.provider.validate_params({"mode": "current"})
        assert ok is False
        assert "location" in reason.lower()

    def test_historical_missing_dates_fails(self):
        ok, reason = self.provider.validate_params(
            {"latitude": 51.5, "longitude": -0.1, "mode": "historical"}
        )
        assert ok is False
        assert "start_date" in reason.lower() or "end_date" in reason.lower()

    def test_historical_with_dates_passes(self):
        ok, reason = self.provider.validate_params({
            "latitude": -34.82,
            "longitude": -58.54,
            "mode": "historical",
            "start_date": "2026-03-10",
            "end_date": "2026-03-10",
        })
        assert ok is True


# ---------------------------------------------------------------
# Validation failure integration test
# ---------------------------------------------------------------


class TestValidationFailureIntegration:
    def test_no_location_weather_returns_failed_evidence(self):
        """Weather keywords match but no location → evidence fails at validate_params."""
        ctx = _make_ctx()
        prompt_spec = _make_prompt_spec(
            requirement_id="req_noloc",
            description="temperature recorded somewhere unknown",
        )
        tool_plan = _make_tool_plan(requirements=["req_noloc"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        assert len(bundle.items) == 1
        assert bundle.items[0].success is False
        assert "Insufficient params" in bundle.items[0].error
        assert "req_noloc" in bundle.requirements_unfulfilled


# ---------------------------------------------------------------
# Buenos Aires integration test (mocked HTTP)
# ---------------------------------------------------------------


class TestBuenosAiresIntegration:
    @patch("httpx.get")
    def test_minister_pistarini_historical(self, mock_get):
        """Full flow: Minister Pistarini + March 10 → archive API at Ezeiza coords."""
        archive_data = {
            "daily": {
                "time": ["2026-03-10"],
                "temperature_2m_max": [27.3],
                "temperature_2m_min": [19.1],
                "precipitation_sum": [0.2],
            }
        }
        mock_get.return_value = httpx.Response(
            status_code=200,
            json=archive_data,
            request=httpx.Request(
                "GET", "https://archive-api.open-meteo.com/v1/archive"
            ),
        )

        # LLM response 1: param extraction (historical date + Ezeiza coords)
        param_extraction_json = json.dumps({
            "latitude": -34.8222,
            "longitude": -58.5358,
            "mode": "historical",
            "start_date": "2026-03-10",
            "end_date": "2026-03-10",
        })
        # LLM response 2: interpretation
        interpretation_json = json.dumps({
            "interpretation": "The max temperature at Ezeiza on March 10 was 27.3°C.",
            "relevant_data": {"temperature_2m_max": 27.3},
            "confidence": 0.95,
        })
        ctx = AgentContext.create_mock(
            llm_responses=[param_extraction_json, interpretation_json]
        )

        prompt_spec = _make_prompt_spec(
            requirement_id="req_ba",
            description=(
                "Highest temperature recorded at Minister Pistarini Intl Airport "
                "Station on March 10, 2026"
            ),
        )
        tool_plan = _make_tool_plan(requirements=["req_ba"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        assert len(bundle.items) == 1
        item = bundle.items[0]
        assert item.success is True
        assert item.extracted_fields["api_data"]["temperature_2m_max"] == 27.3
        assert item.provenance.source_id == "open_meteo"

        # Verify the HTTP call used archive URL with correct coords
        call_args = mock_get.call_args
        assert "archive-api.open-meteo.com" in call_args.args[0]
        query_params = call_args.kwargs.get("params", {})
        assert abs(query_params["latitude"] - (-34.8222)) < 0.01
        assert abs(query_params["longitude"] - (-58.5358)) < 0.01
        assert query_params["start_date"] == "2026-03-10"
        assert query_params["end_date"] == "2026-03-10"


# ---------------------------------------------------------------
# LLM parameter extraction tests
# ---------------------------------------------------------------


class TestLLMParamExtraction:
    @patch("httpx.get")
    def test_llm_extracts_historical_date_params(self, mock_get):
        """LLM extracts historical date params for Buenos Aires."""
        archive_data = {
            "daily": {
                "time": ["2026-03-10"],
                "temperature_2m_max": [27.0],
                "temperature_2m_min": [18.5],
                "precipitation_sum": [0.0],
            }
        }
        mock_get.return_value = httpx.Response(
            status_code=200,
            json=archive_data,
            request=httpx.Request(
                "GET", "https://archive-api.open-meteo.com/v1/archive"
            ),
        )

        param_json = json.dumps({
            "latitude": -34.8222,
            "longitude": -58.5358,
            "mode": "historical",
            "start_date": "2026-03-10",
            "end_date": "2026-03-10",
        })
        interp_json = json.dumps({
            "interpretation": "Max temp was 27°C.",
            "relevant_data": {"temperature_2m_max": 27.0},
            "confidence": 0.9,
        })
        ctx = AgentContext.create_mock(llm_responses=[param_json, interp_json])

        prompt_spec = _make_prompt_spec(
            requirement_id="req_hist",
            description="Temperature at Buenos Aires on March 10, 2026",
        )
        tool_plan = _make_tool_plan(requirements=["req_hist"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        assert len(bundle.items) == 1
        item = bundle.items[0]
        assert item.success is True
        assert item.extracted_fields["api_data"]["temperature_2m_max"] == 27.0

    def test_llm_extraction_fallback_on_bad_json(self):
        """Invalid LLM JSON output → falls back to hint_params."""
        ctx = AgentContext.create_mock(
            llm_responses=["not valid json", "still bad", "nope"]
        )
        provider = OpenMeteoProvider()
        hint_params = {"latitude": 51.5, "longitude": -0.1}

        collector = CollectorAPIData(
            provider_configs={"open_meteo": APIProviderConfig(enabled=True)},
            use_llm_matching=False,
        )
        result = collector._llm_extract_params(
            ctx, provider, "weather in London", hint_params
        )
        # Should fall back to hint_params unchanged
        assert result == hint_params

    def test_no_llm_skips_extraction(self):
        """create_minimal() has no LLM → extraction step skipped entirely."""
        ctx = AgentContext.create_minimal()
        provider = OpenMeteoProvider()
        hint_params = {"latitude": 51.5, "longitude": -0.1, "mode": "current"}

        collector = CollectorAPIData(
            provider_configs={"open_meteo": APIProviderConfig(enabled=True)},
            use_llm_matching=False,
        )
        result = collector._llm_extract_params(
            ctx, provider, "weather in London", hint_params
        )
        assert result is hint_params  # exact same object, not a copy

    @patch("httpx.get")
    def test_llm_extracts_unlisted_city(self, mock_get):
        """LLM provides coords for a city not in the geocode cache."""
        mock_get.return_value = httpx.Response(
            status_code=200,
            json={"current_weather": {"temperature": 8.0}},
            request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
        )

        param_json = json.dumps({
            "latitude": 48.2082,
            "longitude": 16.3738,
            "mode": "current",
        })
        interp_json = json.dumps({
            "interpretation": "Current temperature in Vienna is 8°C.",
            "relevant_data": {"temperature_c": 8.0},
            "confidence": 0.85,
        })
        ctx = AgentContext.create_mock(llm_responses=[param_json, interp_json])

        prompt_spec = _make_prompt_spec(
            requirement_id="req_vienna",
            description="Current weather in Vienna",
        )
        tool_plan = _make_tool_plan(requirements=["req_vienna"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        assert len(bundle.items) == 1
        item = bundle.items[0]
        assert item.success is True

        # Verify LLM-provided coords were used
        call_args = mock_get.call_args
        query_params = call_args.kwargs.get("params", {})
        assert abs(query_params["latitude"] - 48.2082) < 0.01
        assert abs(query_params["longitude"] - 16.3738) < 0.01


# ---------------------------------------------------------------
# LLM response interpretation tests
# ---------------------------------------------------------------


class TestLLMResponseInterpretation:
    @patch("httpx.get")
    def test_interpretation_added_to_evidence(self, mock_get):
        """Interpretation fields are present in evidence when LLM is available."""
        mock_get.return_value = httpx.Response(
            status_code=200,
            json={"current_weather": {"temperature": 15.0}},
            request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
        )

        param_json = json.dumps({
            "latitude": 51.5074,
            "longitude": -0.1278,
            "mode": "current",
        })
        interp_json = json.dumps({
            "interpretation": "Current temperature in London is 15°C.",
            "relevant_data": {"temperature_c": 15.0},
            "confidence": 0.92,
        })
        ctx = AgentContext.create_mock(llm_responses=[param_json, interp_json])

        prompt_spec = _make_prompt_spec(
            requirement_id="req_interp",
            description="Current weather in London",
        )
        tool_plan = _make_tool_plan(requirements=["req_interp"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        item = bundle.items[0]
        assert item.success is True
        assert item.extracted_fields["api_data"]["interpretation"] == (
            "Current temperature in London is 15°C."
        )
        assert item.extracted_fields["api_data"]["interpretation_confidence"] == 0.92
        assert item.extracted_fields["api_data"]["relevant_data"] == {"temperature_c": 15.0}

    @patch("httpx.get")
    def test_interpretation_skipped_on_api_failure(self, mock_get):
        """No interpretation is added when the API call fails."""
        mock_get.return_value = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
        )

        param_json = json.dumps({
            "latitude": 51.5074,
            "longitude": -0.1278,
            "mode": "current",
        })
        # This response should never be consumed since API failed
        interp_json = json.dumps({
            "interpretation": "should not appear",
            "confidence": 0.5,
        })
        ctx = AgentContext.create_mock(llm_responses=[param_json, interp_json])

        prompt_spec = _make_prompt_spec(
            requirement_id="req_fail_interp",
            description="Weather in London",
        )
        tool_plan = _make_tool_plan(requirements=["req_fail_interp"])

        configs = {"open_meteo": APIProviderConfig(enabled=True)}
        collector = CollectorAPIData(
            provider_configs=configs, use_llm_matching=False
        )
        result = collector.run(ctx, prompt_spec, tool_plan)

        bundle, _ = result.output
        item = bundle.items[0]
        assert item.success is False
        assert "interpretation" not in item.extracted_fields


# ---------------------------------------------------------------
# Provider unit tests — Binance
# ---------------------------------------------------------------


class TestBinanceProvider:
    def setup_method(self):
        from agents.collector.api_providers.binance import BinanceProvider
        self.provider = BinanceProvider()

    # -- Identity --

    def test_provider_id(self):
        assert self.provider.provider_id == "binance"

    def test_display_name(self):
        assert self.provider.display_name == "Binance"

    def test_domains(self):
        assert "crypto" in self.provider.domains
        assert "finance" in self.provider.domains

    def test_keywords(self):
        kw = self.provider.keywords
        assert "binance" in kw
        assert "trading pair" in kw
        assert "spot" in kw
        assert "kline" in kw
        assert "candlestick" in kw
        assert "ohlcv" in kw
        assert "crypto" in kw

    def test_requires_api_key_false(self):
        assert self.provider.requires_api_key is False

    # -- Symbol extraction --

    def test_extract_symbol_bitcoin_name(self):
        symbol = self.provider._extract_symbol("Current Bitcoin price")
        assert symbol == "BTCUSDT"

    def test_extract_symbol_ethereum_name(self):
        symbol = self.provider._extract_symbol("What is the Ethereum price?")
        assert symbol == "ETHUSDT"

    def test_extract_symbol_ticker_btc(self):
        symbol = self.provider._extract_symbol("Check BTC markets")
        assert symbol == "BTCUSDT"

    def test_extract_symbol_ticker_eth(self):
        symbol = self.provider._extract_symbol("Get ETH data")
        assert symbol == "ETHUSDT"

    def test_extract_symbol_solana(self):
        symbol = self.provider._extract_symbol("Solana price today")
        assert symbol == "SOLUSDT"

    def test_extract_symbol_unrelated(self):
        symbol = self.provider._extract_symbol("weather in London")
        assert symbol is None

    # -- match_requirement --

    def test_match_crypto_terms(self):
        result = self.provider.match_requirement("What is the current Bitcoin price?")
        assert result is not None
        assert result.provider_id == "binance"
        assert "bitcoin" in result.matched_keywords

    def test_match_binance_keyword(self):
        result = self.provider.match_requirement("Get Binance spot price for ETH")
        assert result is not None
        assert result.suggested_params.get("symbol") == "ETHUSDT"

    def test_match_confidence_boosted_with_symbol(self):
        result = self.provider.match_requirement("Bitcoin crypto price")
        assert result is not None
        assert result.confidence >= 0.3  # boosted from symbol detection
        assert result.suggested_params.get("symbol") == "BTCUSDT"

    def test_no_match_unrelated(self):
        result = self.provider.match_requirement("What is the weather in London?")
        assert result is None

    # -- validate_params --

    def test_validate_params_ok(self):
        ok, reason = self.provider.validate_params(
            {"symbol": "BTCUSDT", "mode": "current"}
        )
        assert ok is True
        assert reason == ""

    def test_validate_params_missing_symbol(self):
        ok, reason = self.provider.validate_params({"mode": "current"})
        assert ok is False
        assert "trading pair" in reason.lower()

    def test_validate_params_historical_missing_start_date(self):
        ok, reason = self.provider.validate_params(
            {"symbol": "BTCUSDT", "mode": "historical"}
        )
        assert ok is False
        assert "start_date" in reason.lower()

    def test_validate_params_historical_ok(self):
        ok, reason = self.provider.validate_params(
            {"symbol": "BTCUSDT", "mode": "historical", "start_date": "2026-03-01"}
        )
        assert ok is True

    # -- build_request current --

    def test_build_request_current(self):
        url, headers, params = self.provider.build_request(
            "bitcoin price", {"symbol": "BTCUSDT", "mode": "current"}
        )
        assert url == "https://api.binance.com/api/v3/ticker/24hr"
        assert params["symbol"] == "BTCUSDT"
        assert headers == {}

    def test_build_request_current_with_api_key(self):
        url, headers, params = self.provider.build_request(
            "bitcoin price",
            {"symbol": "BTCUSDT", "mode": "current", "api_key": "test_key"},
        )
        assert url == "https://api.binance.com/api/v3/ticker/24hr"
        assert headers["X-MBX-APIKEY"] == "test_key"
        assert params["symbol"] == "BTCUSDT"

    # -- build_request historical --

    def test_build_request_historical(self):
        url, headers, params = self.provider.build_request(
            "bitcoin historical",
            {
                "symbol": "BTCUSDT",
                "mode": "historical",
                "interval": "1d",
                "start_date": "2026-03-01",
                "end_date": "2026-03-10",
            },
        )
        assert url == "https://api.binance.com/api/v3/klines"
        assert params["symbol"] == "BTCUSDT"
        assert params["interval"] == "1d"
        assert "startTime" in params
        assert "endTime" in params

    def test_build_request_historical_with_api_key(self):
        url, headers, params = self.provider.build_request(
            "btc klines",
            {
                "symbol": "BTCUSDT",
                "mode": "historical",
                "start_date": "2026-03-01",
                "api_key": "my_key",
            },
        )
        assert headers["X-MBX-APIKEY"] == "my_key"
        assert url == "https://api.binance.com/api/v3/klines"

    # -- parse_response 24hr ticker --

    def test_parse_response_ticker(self):
        raw = {
            "symbol": "BTCUSDT",
            "lastPrice": "95000.00",
            "priceChangePercent": "2.50",
            "volume": "12345.678",
            "quoteVolume": "1170000000.00",
            "highPrice": "96000.00",
            "lowPrice": "93000.00",
            "openPrice": "92500.00",
        }
        fields = self.provider.parse_response(raw)
        assert fields["symbol"] == "BTCUSDT"
        assert fields["last_price"] == "95000.00"
        assert fields["price_change_percent"] == "2.50"
        assert fields["volume"] == "12345.678"
        assert fields["quote_volume"] == "1170000000.00"
        assert fields["high_price"] == "96000.00"
        assert fields["low_price"] == "93000.00"
        assert fields["open_price"] == "92500.00"

    # -- parse_response klines --

    def test_parse_response_klines(self):
        raw = [
            [1709251200000, "95000", "96000", "93000", "95500", "1234.56"],
            [1709337600000, "95500", "97000", "94000", "96500", "2345.67"],
        ]
        fields = self.provider.parse_response(raw)
        assert fields["candle_count"] == 2
        assert len(fields["klines"]) == 2
        assert fields["klines"][0]["open"] == "95000"
        assert fields["klines"][0]["high"] == "96000"
        assert fields["klines"][0]["low"] == "93000"
        assert fields["klines"][0]["close"] == "95500"
        assert fields["klines"][0]["volume"] == "1234.56"
        assert fields["klines"][1]["open"] == "95500"

    def test_parse_response_empty_klines(self):
        fields = self.provider.parse_response([])
        assert fields["candle_count"] == 0
        assert fields["klines"] == []

    # -- is_available --

    def test_is_available_no_key_needed(self):
        cfg = APIProviderConfig(enabled=True)
        assert self.provider.is_available(cfg) is True

    def test_is_available_disabled(self):
        cfg = APIProviderConfig(enabled=False)
        assert self.provider.is_available(cfg) is False

    # -- date conversion --

    def test_date_to_ms(self):
        ms = self.provider._date_to_ms("2026-03-01")
        assert isinstance(ms, int)
        assert ms > 0
