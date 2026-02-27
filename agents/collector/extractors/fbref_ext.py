"""FBRef extractor — stub, implemented in Task 3."""
from __future__ import annotations
from typing import Any
from .base import SiteExtractor, ExtractionError


class FBRefExtractor(SiteExtractor):
    source_id = "fbref"

    def can_handle(self, url: str) -> bool:
        return "fbref.com/en/matches/" in url

    def extract_and_summarize(self, url: str, http_client: Any) -> tuple[str, dict[str, Any]]:
        raise ExtractionError("Not implemented yet")
