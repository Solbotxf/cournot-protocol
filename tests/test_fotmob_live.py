"""
Live integration test for FotMob extraction.

Skipped by default. Run with: pytest tests/test_fotmob_live.py -v -m live
"""

import pytest
from agents.collector.fotmob import fetch_match_stats, find_stat, match_team

pytestmark = pytest.mark.live


@pytest.fixture
def http_client():
    import httpx
    return httpx.Client()


class TestFotMobLive:

    def test_fetch_real_madrid_vs_osasuna(self, http_client):
        """Fetch the actual match page and verify 'Shots outside box' = 8."""
        url = "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz"
        data = fetch_match_stats(url, http_client)

        assert data.match_id == "4837357"
        assert data.home_team == "Osasuna"
        assert data.away_team == "Real Madrid"

        stat = find_stat(data, "Shots outside box")
        assert stat is not None
        assert stat.home_value == 2
        assert stat.away_value == 8

        side = match_team(data, "Real Madrid")
        assert side == "away"

        value = stat.away_value
        assert value == 8
        assert value >= 5  # market threshold

    def test_all_stat_categories_present(self, http_client):
        """Verify all expected stat categories are extracted."""
        url = "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz"
        data = fetch_match_stats(url, http_client)

        expected_cats = {"top_stats", "shots", "expected_goals", "passes", "defence", "duels"}
        assert expected_cats.issubset(set(data.stats_by_category.keys()))
