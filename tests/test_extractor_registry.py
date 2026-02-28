"""Tests for the site extractor registry."""
import pytest
from agents.collector.extractors.base import SiteExtractor
from agents.collector.extractors import find_extractor


class TestFindExtractor:
    def test_finds_fotmob_extractor(self):
        ext = find_extractor("https://www.fotmob.com/matches/arsenal-vs-brentford/389pak")
        assert ext is not None
        assert ext.source_id == "fotmob"

    def test_returns_none_for_fbref_url(self):
        """FBRef URLs should fall through to Phase 2 (no extractor registered)."""
        ext = find_extractor("https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-January-1-2025-Premier-League")
        assert ext is None

    def test_returns_none_for_unknown_url(self):
        ext = find_extractor("https://flashscore.com/match/123")
        assert ext is None

    def test_returns_none_for_empty_url(self):
        ext = find_extractor("")
        assert ext is None


# ---------------------------------------------------------------------------
# Tests: FotMobExtractor
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch
from agents.collector.extractors.fotmob_ext import FotMobExtractor
from agents.collector.extractors.base import ExtractionError
from agents.collector.fotmob import FotMobMatchData, FotMobStat, FotMobShot, FotMobEvent


def _make_fotmob_match_data() -> FotMobMatchData:
    return FotMobMatchData(
        match_id="4837357",
        home_team="Osasuna",
        away_team="Real Madrid",
        url="https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
        home_score=0,
        away_score=4,
        stats_by_category={
            "shots": [
                FotMobStat(title="Total shots", key="total_shots",
                           home_value=13, away_value=15, category="shots"),
            ],
        },
        shots=[
            FotMobShot(event_type="Goal", team_id=8633, player_name="Vinicius Jr",
                       situation="OpenPlay", shot_type="LeftFoot", minute=12,
                       expected_goals=0.45, is_home=False, on_target=True),
        ],
        events=[
            FotMobEvent(type="Goal", time=12, overload_time=None, is_home=False,
                        player_name="Vinicius Jr", assist_player="Bellingham"),
        ],
        event_types=["Goal"],
    )


class TestFotMobExtractor:
    def test_can_handle_match_url(self):
        ext = FotMobExtractor()
        assert ext.can_handle("https://www.fotmob.com/matches/arsenal-vs-brentford/389pak")

    def test_cannot_handle_team_url(self):
        ext = FotMobExtractor()
        assert not ext.can_handle("https://www.fotmob.com/teams/8633/overview/real-madrid")

    def test_extract_returns_summary_and_metadata(self):
        ext = FotMobExtractor()
        mock_http = MagicMock()
        match_data = _make_fotmob_match_data()

        with patch("agents.collector.extractors.fotmob_ext.fotmob_fetch_match_stats",
                   return_value=match_data):
            summary, meta = ext.extract_and_summarize(
                "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
                mock_http,
            )

        assert "Osasuna" in summary
        assert "Real Madrid" in summary
        assert meta["fotmob_match_id"] == "4837357"
        assert meta["fotmob_home_team"] == "Osasuna"
        assert meta["fotmob_away_team"] == "Real Madrid"
        assert meta["fotmob_shots_count"] == 1

    def test_extract_raises_on_fetch_error(self):
        ext = FotMobExtractor()
        mock_http = MagicMock()

        from agents.collector.fotmob import FotMobExtractionError
        with patch("agents.collector.extractors.fotmob_ext.fotmob_fetch_match_stats",
                   side_effect=FotMobExtractionError("Page broken")):
            with pytest.raises(ExtractionError, match="Page broken"):
                ext.extract_and_summarize(
                    "https://www.fotmob.com/matches/foo/bar", mock_http,
                )
