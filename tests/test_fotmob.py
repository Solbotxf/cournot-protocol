"""Tests for agents.collector.fotmob — FotMob stat extraction."""

import json
import pytest
from unittest.mock import MagicMock

from agents.collector.fotmob import (
    FotMobExtractionError,
    FotMobMatchData,
    FotMobStat,
    fetch_match_stats,
    find_stat,
    match_team,
)


# ---------------------------------------------------------------------------
# Fixture: minimal __NEXT_DATA__ JSON embedded in HTML
# ---------------------------------------------------------------------------

_SAMPLE_STATS = {
    "props": {
        "pageProps": {
            "general": {
                "matchId": "4837357",
                "homeTeam": {"name": "Osasuna", "id": 8371},
                "awayTeam": {"name": "Real Madrid", "id": 8633},
            },
            "content": {
                "stats": {
                    "Periods": {
                        "All": {
                            "stats": [
                                {
                                    "title": "Top stats",
                                    "key": "top_stats",
                                    "stats": [
                                        {
                                            "title": "Total shots",
                                            "key": "total_shots",
                                            "stats": [13, 15],
                                            "type": "text",
                                        },
                                    ],
                                },
                                {
                                    "title": "Shots",
                                    "key": "shots",
                                    "stats": [
                                        {
                                            "title": "Shots",
                                            "key": "shots",
                                            "stats": [None, None],
                                            "type": "title",
                                        },
                                        {
                                            "title": "Total shots",
                                            "key": "total_shots",
                                            "stats": [13, 15],
                                            "type": "text",
                                        },
                                        {
                                            "title": "Shots on target",
                                            "key": "ShotsOnTarget",
                                            "stats": [2, 5],
                                            "type": "text",
                                        },
                                        {
                                            "title": "Shots inside box",
                                            "key": "shots_inside_box",
                                            "stats": [11, 7],
                                            "type": "text",
                                        },
                                        {
                                            "title": "Shots outside box",
                                            "key": "shots_outside_box",
                                            "stats": [2, 8],
                                            "type": "text",
                                        },
                                    ],
                                },
                                {
                                    "title": "Passes",
                                    "key": "passes",
                                    "stats": [
                                        {
                                            "title": "Accurate passes",
                                            "key": "accurate_passes",
                                            "stats": ["293 (83%)", "511 (90%)"],
                                            "type": "text",
                                        },
                                    ],
                                },
                            ],
                        },
                    },
                },
            },
        },
    },
}


def _make_html(next_data: dict) -> str:
    """Wrap a __NEXT_DATA__ dict in minimal HTML."""
    return (
        '<!DOCTYPE html><html><head></head><body>'
        f'<script id="__NEXT_DATA__" type="application/json">'
        f'{json.dumps(next_data)}'
        f'</script></body></html>'
    )


SAMPLE_HTML = _make_html(_SAMPLE_STATS)


# ---------------------------------------------------------------------------
# Tests: fetch_match_stats
# ---------------------------------------------------------------------------

class TestFetchMatchStats:

    def test_parses_valid_html(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        data = fetch_match_stats(
            "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
            mock_http,
        )
        assert isinstance(data, FotMobMatchData)
        assert data.match_id == "4837357"
        assert data.home_team == "Osasuna"
        assert data.away_team == "Real Madrid"
        assert "shots" in data.stats_by_category
        assert "top_stats" in data.stats_by_category

    def test_raises_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        with pytest.raises(FotMobExtractionError, match="HTTP 403"):
            fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_raises_on_missing_next_data(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>No data here</body></html>"

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        with pytest.raises(FotMobExtractionError, match="__NEXT_DATA__"):
            fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_raises_on_missing_stats_path(self):
        bad_data = {"props": {"pageProps": {"general": {"matchId": "1"}}}}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _make_html(bad_data)

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        with pytest.raises(FotMobExtractionError, match="stats"):
            fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_sends_browser_user_agent(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "User-Agent" in headers
        assert "Mozilla" in headers["User-Agent"]


# ---------------------------------------------------------------------------
# Tests: find_stat
# ---------------------------------------------------------------------------

class TestFindStat:

    def _make_data(self) -> FotMobMatchData:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        return fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_finds_by_exact_title(self):
        data = self._make_data()
        stat = find_stat(data, "Shots outside box")
        assert stat is not None
        assert stat.home_value == 2
        assert stat.away_value == 8
        assert stat.category == "shots"

    def test_finds_by_title_case_insensitive(self):
        data = self._make_data()
        stat = find_stat(data, "shots OUTSIDE BOX")
        assert stat is not None
        assert stat.away_value == 8

    def test_finds_by_key_fallback(self):
        data = self._make_data()
        stat = find_stat(data, "shots_outside_box")
        assert stat is not None
        assert stat.away_value == 8

    def test_finds_stat_in_different_category(self):
        data = self._make_data()
        stat = find_stat(data, "Accurate passes")
        assert stat is not None
        assert stat.category == "passes"

    def test_returns_none_for_missing_stat(self):
        data = self._make_data()
        stat = find_stat(data, "Nonexistent stat")
        assert stat is None

    def test_skips_title_type_entries(self):
        """The 'Shots' title entry (type=title) should not be returned."""
        data = self._make_data()
        stat = find_stat(data, "Shots")
        # Should find the 'top_stats' category "Total shots" or similar,
        # NOT the title-type "Shots" entry
        if stat is not None:
            assert stat.home_value is not None  # not None like title entries


# ---------------------------------------------------------------------------
# Tests: match_team
# ---------------------------------------------------------------------------

class TestMatchTeam:

    def _make_data(self) -> FotMobMatchData:
        return FotMobMatchData(
            match_id="123",
            home_team="Osasuna",
            away_team="Real Madrid",
        )

    def test_exact_match_home(self):
        assert match_team(self._make_data(), "Osasuna") == "home"

    def test_exact_match_away(self):
        assert match_team(self._make_data(), "Real Madrid") == "away"

    def test_case_insensitive(self):
        assert match_team(self._make_data(), "real madrid") == "away"
        assert match_team(self._make_data(), "OSASUNA") == "home"

    def test_substring_match(self):
        assert match_team(self._make_data(), "Real") == "away"
        assert match_team(self._make_data(), "Madrid") == "away"

    def test_no_match(self):
        assert match_team(self._make_data(), "Barcelona") is None
