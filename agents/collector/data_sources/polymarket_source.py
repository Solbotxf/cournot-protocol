"""
Module 05 - Polymarket Source Adapter

Polymarket-specific data source for fetching market data.
Uses explicit URIs provided by Prompt Engineer - does not hardcode endpoints.
"""

from typing import Any, Optional

from core.schemas import SourceTarget

from agents.collector.data_sources.base_source import BaseSource, FetchedArtifact
from agents.collector.data_sources.http_source import HTTPSource


class PolymarketSource(BaseSource):
    """
    Polymarket data source adapter.
    
    Fetches market data from Polymarket APIs.
    Expects explicit URI to be provided in SourceTarget - does not
    discover or hardcode endpoints.
    """
    
    source_id: str = "polymarket"
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None
    ):
        """
        Initialize Polymarket source.
        
        Args:
            config: Configuration options
        """
        super().__init__(config)
        
        # Use HTTP source for underlying requests
        self._http_source = HTTPSource(config)
    
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch data from Polymarket API.
        
        The URI must be explicitly provided in the target - this adapter
        does not discover or construct endpoints.
        
        Args:
            target: Source target with explicit Polymarket API URI
            timeout_s: Request timeout in seconds
            
        Returns:
            FetchedArtifact with market data
        """
        if not target.uri:
            return self._create_error_artifact(
                "Polymarket source requires explicit URI in SourceTarget",
                target.expected_content_type
            )
        
        # Prepare headers for Polymarket API
        polymarket_headers = {
            "Accept": "application/json",
            "User-Agent": "CournotCollector/1.0",
        }
        
        # Merge with target headers
        merged_headers = {**polymarket_headers, **target.headers}
        
        # Create HTTP target
        http_target = SourceTarget(
            source_id="http",
            uri=target.uri,
            method=target.method,
            expected_content_type=target.expected_content_type,
            headers=merged_headers,
            params=target.params,
            body=target.body,
            auth_ref=target.auth_ref,
            cache_ttl_seconds=target.cache_ttl_seconds,
            notes=target.notes
        )
        
        # Fetch via HTTP
        artifact = self._http_source.fetch(http_target, timeout_s=timeout_s)
        
        # Validate response if successful
        if artifact.success and artifact.is_json:
            artifact = self._validate_response(artifact)
        
        return artifact
    
    def _validate_response(
        self,
        artifact: FetchedArtifact
    ) -> FetchedArtifact:
        """
        Validate Polymarket API response.
        
        Args:
            artifact: Fetched artifact to validate
            
        Returns:
            Validated artifact (may have error set if invalid)
        """
        if not artifact.parsed:
            return artifact
        
        data = artifact.parsed
        
        # Check for API error responses
        if isinstance(data, dict):
            if "error" in data:
                return FetchedArtifact(
                    raw_bytes=artifact.raw_bytes,
                    content_type=artifact.content_type,
                    parsed=artifact.parsed,
                    response_headers=artifact.response_headers,
                    status_code=artifact.status_code,
                    final_url=artifact.final_url,
                    error=f"API error: {data.get('error')}"
                )
        
        return artifact
    
    def supports_target(self, target: SourceTarget) -> bool:
        """Check if this adapter supports the target."""
        return target.source_id == "polymarket"
    
    def extract_market_data(
        self,
        artifact: FetchedArtifact
    ) -> Optional[dict[str, Any]]:
        """
        Extract normalized market data from response.
        
        Args:
            artifact: Fetched market data artifact
            
        Returns:
            Normalized market data dict or None
        """
        if not artifact.success or not artifact.is_json:
            return None
        
        data = artifact.parsed
        if not isinstance(data, dict):
            return None
        
        # Extract common market fields if present
        normalized = {}
        
        # Map common Polymarket fields
        field_mappings = {
            "id": ["id", "market_id", "marketId"],
            "question": ["question", "title"],
            "outcome": ["outcome", "resolution", "resolved"],
            "yes_price": ["yes_price", "yesPrice", "outcomePrices"],
            "no_price": ["no_price", "noPrice"],
            "volume": ["volume", "totalVolume"],
            "end_date": ["end_date", "endDate", "expirationDate"],
        }
        
        for key, aliases in field_mappings.items():
            for alias in aliases:
                if alias in data:
                    normalized[key] = data[alias]
                    break
        
        return normalized if normalized else None


class ExchangeSource(BaseSource):
    """
    Generic exchange data source adapter.
    
    For fetching price/market data from exchanges like Coinbase, Binance, etc.
    Uses explicit URIs provided in SourceTarget.
    """
    
    source_id: str = "exchange"
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None
    ):
        """
        Initialize exchange source.
        
        Args:
            config: Configuration options
        """
        super().__init__(config)
        self._http_source = HTTPSource(config)
    
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch data from exchange API.
        
        Args:
            target: Source target with explicit exchange API URI
            timeout_s: Request timeout in seconds
            
        Returns:
            FetchedArtifact with exchange data
        """
        if not target.uri:
            return self._create_error_artifact(
                "Exchange source requires explicit URI in SourceTarget",
                target.expected_content_type
            )
        
        # Prepare headers
        exchange_headers = {
            "Accept": "application/json",
        }
        merged_headers = {**exchange_headers, **target.headers}
        
        # Create HTTP target
        http_target = SourceTarget(
            source_id="http",
            uri=target.uri,
            method=target.method,
            expected_content_type=target.expected_content_type,
            headers=merged_headers,
            params=target.params,
            body=target.body,
            auth_ref=target.auth_ref,
            cache_ttl_seconds=target.cache_ttl_seconds,
            notes=target.notes
        )
        
        return self._http_source.fetch(http_target, timeout_s=timeout_s)
    
    def supports_target(self, target: SourceTarget) -> bool:
        """Check if this adapter supports the target."""
        return target.source_id == "exchange"