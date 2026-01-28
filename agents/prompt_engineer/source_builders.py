"""
Module 04 - Source Target Builders.

Provides builders for creating SourceTarget objects for common data sources
like exchanges, Polymarket, government APIs, and generic HTTP endpoints.
"""

from urllib.parse import urlparse

from core.schemas import SourceTarget


class SourceTargetBuilder:
    """Builds SourceTarget objects for common data sources."""
    
    @staticmethod
    def build_coinbase_target(symbol: str, date: str) -> SourceTarget:
        """Build Coinbase API target for price data."""
        # Coinbase Exchange API for candle data
        product_id = f"{symbol}-USD"
        return SourceTarget(
            source_id="exchange",
            uri=f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            method="GET",
            expected_content_type="json",
            params={
                "granularity": "86400",  # Daily candles
                "start": date,
                "end": date,
            },
            notes=f"Coinbase Exchange API for {product_id} daily candle data"
        )
    
    @staticmethod
    def build_binance_target(symbol: str, date: str) -> SourceTarget:
        """Build Binance API target for price data."""
        return SourceTarget(
            source_id="exchange",
            uri=f"https://api.binance.com/api/v3/klines",
            method="GET",
            expected_content_type="json",
            params={
                "symbol": f"{symbol}USDT",
                "interval": "1d",
                "startTime": date,
                "limit": "1",
            },
            notes=f"Binance API for {symbol}USDT daily kline data"
        )
    
    @staticmethod
    def build_polymarket_target(market_slug: str) -> SourceTarget:
        """Build Polymarket API target."""
        return SourceTarget(
            source_id="polymarket",
            uri=f"https://gamma-api.polymarket.com/markets/{market_slug}",
            method="GET",
            expected_content_type="json",
            notes="Polymarket Gamma API for market resolution"
        )
    
    @staticmethod
    def build_bls_target(series_id: str) -> SourceTarget:
        """Build BLS API target for economic data."""
        return SourceTarget(
            source_id="government",
            uri="https://api.bls.gov/publicAPI/v2/timeseries/data/",
            method="POST",
            expected_content_type="json",
            body={"seriesid": [series_id]},
            notes=f"BLS API for series {series_id}"
        )
    
    @staticmethod
    def build_generic_http_target(url: str) -> SourceTarget:
        """Build generic HTTP target from URL."""
        parsed = urlparse(url)
        content_type = "json" if "api" in parsed.netloc else "html"
        return SourceTarget(
            source_id="http",
            uri=url,
            method="GET",
            expected_content_type=content_type,
            notes=f"HTTP fetch from {parsed.netloc}"
        )