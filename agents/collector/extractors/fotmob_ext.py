"""FotMob extractor — stub, implemented in Task 2."""
from __future__ import annotations
from typing import Any
from .base import SiteExtractor, ExtractionError


class FotMobExtractor(SiteExtractor):
    source_id = "fotmob"

    def can_handle(self, url: str) -> bool:
        return "fotmob.com/matches/" in url

    def extract_and_summarize(self, url: str, http_client: Any) -> tuple[str, dict[str, Any]]:
        raise ExtractionError("Not implemented yet")
