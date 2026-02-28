"""Site extractor registry.

Maps discovered URLs to domain-specific extractors that fetch
structured data and produce text summaries for LLM resolution.
"""
from __future__ import annotations

from .base import ExtractionError, SiteExtractor
from .fotmob_ext import FotMobExtractor

__all__ = [
    "ExtractionError",
    "SiteExtractor",
    "FotMobExtractor",
    "find_extractor",
]

_REGISTRY: list[SiteExtractor] = [
    FotMobExtractor(),
]


def find_extractor(url: str) -> SiteExtractor | None:
    """Find a registered extractor that can handle the given URL."""
    for ext in _REGISTRY:
        if ext.can_handle(url):
            return ext
    return None
