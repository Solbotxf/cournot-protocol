"""Tests for the site extractor registry."""
import pytest
from agents.collector.extractors.base import SiteExtractor
from agents.collector.extractors import find_extractor


class TestFindExtractor:
    def test_finds_fotmob_extractor(self):
        ext = find_extractor("https://www.fotmob.com/matches/arsenal-vs-brentford/389pak")
        assert ext is not None
        assert ext.source_id == "fotmob"

    def test_finds_fbref_extractor(self):
        ext = find_extractor("https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-January-1-2025-Premier-League")
        assert ext is not None
        assert ext.source_id == "fbref"

    def test_returns_none_for_unknown_url(self):
        ext = find_extractor("https://flashscore.com/match/123")
        assert ext is None

    def test_returns_none_for_empty_url(self):
        ext = find_extractor("")
        assert ext is None
