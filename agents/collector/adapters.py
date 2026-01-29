"""
Source Adapters

Adapters for fetching data from different source types.
Each adapter knows how to:
1. Execute requests for its source type
2. Parse responses into evidence items
3. Determine provenance tier
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from core.schemas import (
    EvidenceItem,
    Provenance,
    SourceTarget,
)

if TYPE_CHECKING:
    from agents.context import AgentContext


class SourceAdapter(ABC):
    """
    Base class for source adapters.
    
    Each adapter handles a specific source type (http, polymarket, etc.)
    """
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """The source type this adapter handles."""
        ...
    
    @abstractmethod
    def fetch(
        self,
        ctx: "AgentContext",
        target: SourceTarget,
        requirement_id: str,
    ) -> EvidenceItem:
        """
        Fetch data from the source.
        
        Args:
            ctx: Agent context with HTTP client
            target: Source target specification
            requirement_id: ID of the requirement being fulfilled
        
        Returns:
            EvidenceItem with fetched data
        """
        ...
    
    def _generate_evidence_id(self, requirement_id: str, source_id: str, uri: str) -> str:
        """Generate a deterministic evidence ID."""
        content = f"{requirement_id}:{source_id}:{uri}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        return f"ev_{hash_bytes[:8].hex()}"
    
    def _hash_content(self, content: bytes | str) -> str:
        """Hash content for provenance tracking."""
        if isinstance(content, str):
            content = content.encode()
        return f"0x{hashlib.sha256(content).hexdigest()}"


class HttpAdapter(SourceAdapter):
    """
    Adapter for HTTP/HTTPS sources.
    
    Handles generic HTTP requests with JSON/text/HTML responses.
    """
    
    @property
    def source_type(self) -> str:
        return "http"
    
    def fetch(
        self,
        ctx: "AgentContext",
        target: SourceTarget,
        requirement_id: str,
    ) -> EvidenceItem:
        """Fetch data via HTTP."""
        evidence_id = self._generate_evidence_id(requirement_id, target.source_id, target.uri)
        
        # Check if we have HTTP client
        if not ctx.http:
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement_id,
                provenance=Provenance(
                    source_id=target.source_id,
                    source_uri=target.uri,
                    tier=0,
                ),
                success=False,
                error="HTTP client not available",
            )
        
        try:
            # Execute request
            method = target.method if target.method in ("GET", "POST") else "GET"
            
            if method == "GET":
                response = ctx.http.get(
                    target.uri,
                    headers=target.headers,
                    params=target.params,
                )
            else:
                response = ctx.http.post(
                    target.uri,
                    headers=target.headers,
                    params=target.params,
                    json=target.body if isinstance(target.body, dict) else None,
                    data=target.body if isinstance(target.body, str) else None,
                )
            
            # Parse response based on content type
            raw_content = response.text
            parsed_value = None
            extracted_fields = {}
            content_type = response.headers.get("content-type", "")
            
            if target.expected_content_type == "json" or "json" in content_type:
                try:
                    parsed_value = response.json()
                    # Extract nested values for common patterns
                    extracted_fields = self._extract_json_fields(parsed_value)
                except json.JSONDecodeError:
                    pass
            elif target.expected_content_type == "html" or "html" in content_type:
                parsed_value = raw_content
                extracted_fields = self._extract_html_fields(raw_content)
            else:
                parsed_value = raw_content
            
            # Build provenance
            provenance = Provenance(
                source_id=target.source_id,
                source_uri=target.uri,
                tier=self._determine_tier(target.source_id, target.uri),
                fetched_at=ctx.now(),
                receipt_id=response.receipt_id,
                content_hash=self._hash_content(response.content),
            )
            
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement_id,
                provenance=provenance,
                raw_content=raw_content[:10000] if len(raw_content) > 10000 else raw_content,
                content_type=content_type or "text/plain",
                parsed_value=parsed_value,
                success=response.ok,
                error=None if response.ok else f"HTTP {response.status_code}",
                status_code=response.status_code,
                extracted_fields=extracted_fields,
            )
            
        except Exception as e:
            ctx.warning(f"HTTP fetch failed for {target.uri}: {e}")
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement_id,
                provenance=Provenance(
                    source_id=target.source_id,
                    source_uri=target.uri,
                    tier=0,
                ),
                success=False,
                error=str(e),
            )
    
    def _determine_tier(self, source_id: str, uri: str) -> int:
        """Determine provenance tier based on source."""
        # Tier 3: Official APIs
        official_domains = [
            "api.coingecko.com",
            "api.coinbase.com",
            "api.binance.com",
            "api.polymarket.com",
            "clob.polymarket.com",
            "api.espn.com",
        ]
        for domain in official_domains:
            if domain in uri:
                return 3
        
        # Tier 2: News and aggregators
        news_domains = [
            "newsapi.org",
            "news.google.com",
            "reuters.com",
            "apnews.com",
        ]
        for domain in news_domains:
            if domain in uri:
                return 2
        
        # Tier 1: Generic HTTP
        return 1
    
    def _extract_json_fields(self, data: Any) -> dict[str, Any]:
        """Extract common fields from JSON response."""
        fields = {}
        
        if isinstance(data, dict):
            # Common price fields
            for key in ["price", "usd", "amount", "value", "last", "close"]:
                if key in data:
                    fields[key] = data[key]
            
            # Nested structures (CoinGecko style: {"bitcoin": {"usd": 50000}})
            for k, v in data.items():
                if isinstance(v, dict):
                    for inner_key in ["usd", "price", "value"]:
                        if inner_key in v:
                            fields[f"{k}_{inner_key}"] = v[inner_key]
            
            # Coinbase style: {"data": {"amount": "50000"}}
            if "data" in data and isinstance(data["data"], dict):
                for key in ["amount", "price", "value"]:
                    if key in data["data"]:
                        fields[key] = data["data"][key]
        
        return fields
    
    def _extract_html_fields(self, html: str) -> dict[str, Any]:
        """Extract fields from HTML (basic)."""
        fields = {}
        
        # Try to find price patterns
        price_pattern = r'\$[\d,]+(?:\.\d{2})?'
        prices = re.findall(price_pattern, html)
        if prices:
            fields["prices_found"] = prices[:5]  # First 5 prices
        
        # Try to find title
        title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            fields["title"] = title_match.group(1).strip()
        
        return fields


class PolymarketAdapter(SourceAdapter):
    """
    Adapter for Polymarket API.
    
    Specialized handling for Polymarket market data.
    """
    
    @property
    def source_type(self) -> str:
        return "polymarket"
    
    def fetch(
        self,
        ctx: "AgentContext",
        target: SourceTarget,
        requirement_id: str,
    ) -> EvidenceItem:
        """Fetch data from Polymarket."""
        # Delegate to HTTP adapter with Polymarket-specific post-processing
        http_adapter = HttpAdapter()
        evidence = http_adapter.fetch(ctx, target, requirement_id)
        
        # Override tier for Polymarket
        evidence.provenance.tier = 3
        
        # Extract Polymarket-specific fields
        if evidence.success and evidence.parsed_value:
            self._extract_polymarket_fields(evidence)
        
        return evidence
    
    def _extract_polymarket_fields(self, evidence: EvidenceItem) -> None:
        """Extract Polymarket-specific fields."""
        data = evidence.parsed_value
        if not isinstance(data, dict):
            return
        
        # Market data fields
        for key in ["outcome", "outcomePrices", "volume", "liquidity", "resolved", "resolutionSource"]:
            if key in data:
                evidence.extracted_fields[key] = data[key]
        
        # Handle outcome prices
        if "outcomePrices" in data:
            prices = data["outcomePrices"]
            if isinstance(prices, list) and len(prices) >= 2:
                evidence.extracted_fields["yes_price"] = prices[0]
                evidence.extracted_fields["no_price"] = prices[1]


class CryptoExchangeAdapter(SourceAdapter):
    """
    Adapter for cryptocurrency exchange APIs.
    
    Handles CoinGecko, Coinbase, Binance, etc.
    """
    
    @property
    def source_type(self) -> str:
        return "exchange"
    
    def fetch(
        self,
        ctx: "AgentContext",
        target: SourceTarget,
        requirement_id: str,
    ) -> EvidenceItem:
        """Fetch data from exchange API."""
        # Delegate to HTTP adapter
        http_adapter = HttpAdapter()
        evidence = http_adapter.fetch(ctx, target, requirement_id)
        
        # Extract price specifically
        if evidence.success and evidence.parsed_value:
            price = self._extract_price(evidence.parsed_value, target.uri)
            if price is not None:
                evidence.extracted_fields["price_usd"] = price
        
        return evidence
    
    def _extract_price(self, data: Any, uri: str) -> float | None:
        """Extract price from exchange response."""
        if not isinstance(data, dict):
            return None
        
        # CoinGecko: {"bitcoin": {"usd": 50000}}
        if "coingecko" in uri:
            for asset_data in data.values():
                if isinstance(asset_data, dict) and "usd" in asset_data:
                    return float(asset_data["usd"])
        
        # Coinbase: {"data": {"amount": "50000"}}
        if "coinbase" in uri:
            if "data" in data and "amount" in data["data"]:
                return float(data["data"]["amount"])
        
        # Binance: {"price": "50000.00"}
        if "binance" in uri:
            if "price" in data:
                return float(data["price"])
        
        # Generic fallback
        for key in ["price", "usd", "amount", "last", "close"]:
            if key in data:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    pass
        
        return None


class MockAdapter(SourceAdapter):
    """
    Mock adapter for testing.
    
    Returns preset responses without making actual requests.
    """
    
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        """
        Initialize mock adapter.
        
        Args:
            responses: Map of URI patterns to mock responses
        """
        self.responses = responses or {}
    
    @property
    def source_type(self) -> str:
        return "mock"
    
    def fetch(
        self,
        ctx: "AgentContext",
        target: SourceTarget,
        requirement_id: str,
    ) -> EvidenceItem:
        """Return mock data."""
        evidence_id = self._generate_evidence_id(requirement_id, target.source_id, target.uri)
        
        # Find matching response
        response_data = None
        for pattern, data in self.responses.items():
            if pattern in target.uri:
                response_data = data
                break
        
        if response_data is None:
            # Default mock response
            response_data = {"mock": True, "source": target.source_id}
        
        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=requirement_id,
            provenance=Provenance(
                source_id=target.source_id,
                source_uri=target.uri,
                tier=1,
                fetched_at=ctx.now(),
                content_hash=self._hash_content(json.dumps(response_data)),
            ),
            raw_content=json.dumps(response_data),
            content_type="application/json",
            parsed_value=response_data,
            success=True,
            status_code=200,
            extracted_fields=response_data if isinstance(response_data, dict) else {},
        )


# Adapter registry
_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "http": HttpAdapter,
    "web": HttpAdapter,
    "polymarket": PolymarketAdapter,
    "coingecko": CryptoExchangeAdapter,
    "coinbase": CryptoExchangeAdapter,
    "binance": CryptoExchangeAdapter,
    "exchange": CryptoExchangeAdapter,
    "mock": MockAdapter,
}


def get_adapter(source_id: str) -> SourceAdapter:
    """
    Get an adapter for the given source ID.
    
    Args:
        source_id: Source identifier
    
    Returns:
        Appropriate adapter instance
    """
    # Normalize source_id
    source_lower = source_id.lower()
    
    # Check for known adapters
    for key, adapter_class in _ADAPTERS.items():
        if key in source_lower:
            return adapter_class()
    
    # Default to HTTP adapter
    return HttpAdapter()


def register_adapter(source_type: str, adapter_class: type[SourceAdapter]) -> None:
    """Register a custom adapter."""
    _ADAPTERS[source_type] = adapter_class
