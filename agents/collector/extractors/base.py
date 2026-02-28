"""Base class for site-specific data extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExtractionError(Exception):
    """Raised when data extraction from a site fails."""


class SiteExtractor(ABC):
    """Base for domain-specific structured data extractors.

    Each extractor knows how to fetch and summarize data from one
    website domain into a text summary suitable for LLM consumption.
    """

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Return True if this extractor handles the given URL."""

    @abstractmethod
    def extract_and_summarize(
        self, url: str, http_client: Any, **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Fetch structured data and return (summary_text, metadata).

        Args:
            url: The discovered URL to extract data from.
            http_client: An httpx-compatible HTTP client.
            **kwargs: Optional extra context. Extractors that need an
                LLM can accept ``gemini_client`` and ``gemini_model``.

        Returns:
            Tuple of (text_summary, metadata_dict).

        Raises:
            ExtractionError: If extraction fails.
        """

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Short identifier, e.g. 'fotmob'."""
