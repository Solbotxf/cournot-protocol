"""
Collector Data Sources Package

Data source adapters for fetching evidence from various sources.
"""

from agents.collector.data_sources.base_source import (
    BaseSource,
    FetchedArtifact,
    SourceProtocol,
)
from agents.collector.data_sources.http_source import (
    APISource,
    HTTPSource,
)
from agents.collector.data_sources.polymarket_source import (
    ExchangeSource,
    PolymarketSource,
)
from agents.collector.data_sources.web_source import (
    WebSource,
)

__all__ = [
    # Base
    "BaseSource",
    "FetchedArtifact",
    "SourceProtocol",
    # HTTP
    "HTTPSource",
    "APISource",
    # Web
    "WebSource",
    # Domain-specific
    "PolymarketSource",
    "ExchangeSource",
]


def get_source_for_target(target_source_id: str) -> BaseSource:
    """
    Get appropriate source adapter for a target source ID.
    
    Args:
        target_source_id: The source_id from SourceTarget
        
    Returns:
        Appropriate BaseSource instance
        
    Raises:
        ValueError: If no adapter found for source_id
    """
    adapters = {
        "http": HTTPSource,
        "https": HTTPSource,
        "api": APISource,
        "web": WebSource,
        "polymarket": PolymarketSource,
        "exchange": ExchangeSource,
    }
    
    adapter_class = adapters.get(target_source_id)
    if adapter_class is None:
        raise ValueError(f"No adapter found for source_id: {target_source_id}")
    
    return adapter_class()