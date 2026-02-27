"""Tests for FBRef site extractor."""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from agents.collector.extractors.fbref_ext import FBRefExtractor, _parse_game_id, _infer_league_season
from agents.collector.extractors.base import ExtractionError


class TestParseGameId:
    def test_extracts_from_standard_url(self):
        url = "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-January-1-2025-Premier-League"
        assert _parse_game_id(url) == "7e6892e4"

    def test_extracts_from_url_with_trailing_slash(self):
        url = "https://fbref.com/en/matches/abc123de/Some-Match/"
        assert _parse_game_id(url) == "abc123de"

    def test_returns_none_for_non_match_url(self):
        url = "https://fbref.com/en/squads/18bb7c10/Arsenal"
        assert _parse_game_id(url) is None


class TestInferLeagueSeason:
    def test_infers_premier_league(self):
        url = "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-January-1-2025-Premier-League"
        league, season = _infer_league_season(url)
        assert league == "ENG-Premier League"

    def test_infers_la_liga(self):
        url = "https://fbref.com/en/matches/abc123/Osasuna-Real-Madrid-February-21-2026-La-Liga"
        league, season = _infer_league_season(url)
        assert league == "ESP-La Liga"

    def test_defaults_to_premier_league(self):
        url = "https://fbref.com/en/matches/abc123/Some-Match"
        league, season = _infer_league_season(url)
        assert league == "ENG-Premier League"


class TestFBRefExtractor:
    def test_can_handle_match_url(self):
        ext = FBRefExtractor()
        assert ext.can_handle("https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal")

    def test_cannot_handle_squad_url(self):
        ext = FBRefExtractor()
        assert not ext.can_handle("https://fbref.com/en/squads/18bb7c10/Arsenal")

    def test_cannot_handle_non_fbref(self):
        ext = FBRefExtractor()
        assert not ext.can_handle("https://fotmob.com/matches/abc123")

    def test_extract_builds_summary_from_dataframes(self):
        ext = FBRefExtractor()
        mock_http = MagicMock()

        # Mock player stats DataFrame
        player_idx = pd.MultiIndex.from_tuples(
            [
                ("ENG-Premier League", "2425", "2026-02-12 Brentford-Arsenal", "Arsenal", "Saka"),
                ("ENG-Premier League", "2425", "2026-02-12 Brentford-Arsenal", "Arsenal", "Havertz"),
                ("ENG-Premier League", "2425", "2026-02-12 Brentford-Arsenal", "Brentford", "Mbeumo"),
            ],
            names=["league", "season", "game", "team", "player"],
        )
        df_players = pd.DataFrame(
            {
                "shots_on_target": [3, 2, 1],
                "shots": [5, 4, 3],
                "goals": [1, 0, 0],
                "min": [90, 90, 90],
            },
            index=player_idx,
        )

        # Mock shot events DataFrame
        shot_idx = pd.MultiIndex.from_tuples(
            [("ENG-Premier League", "2425", "2026-02-12 Brentford-Arsenal")],
            names=["league", "season", "game"],
        )
        df_shots = pd.DataFrame(
            {
                "minute": [23],
                "player": ["Saka"],
                "team": ["Arsenal"],
                "outcome": ["Goal"],
                "distance": [12],
                "body_part": ["Right Foot"],
            },
            index=shot_idx,
        )

        with patch("agents.collector.extractors.fbref_ext._create_fbref_client") as mock_create:
            mock_fbref = MagicMock()
            mock_fbref.read_player_match_stats.return_value = df_players
            mock_fbref.read_shot_events.return_value = df_shots
            mock_create.return_value = mock_fbref

            summary, meta = ext.extract_and_summarize(
                "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-February-12-2026-Premier-League",
                mock_http,
            )

        assert "Arsenal" in summary
        assert "Saka" in summary
        assert meta["fbref_game_id"] == "7e6892e4"
        assert meta["source_url"] == "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-February-12-2026-Premier-League"

    def test_extract_raises_on_empty_data(self):
        ext = FBRefExtractor()
        mock_http = MagicMock()

        with patch("agents.collector.extractors.fbref_ext._create_fbref_client") as mock_create:
            mock_fbref = MagicMock()
            mock_fbref.read_player_match_stats.return_value = pd.DataFrame()
            mock_fbref.read_shot_events.return_value = pd.DataFrame()
            mock_create.return_value = mock_fbref

            with pytest.raises(ExtractionError, match="No data"):
                ext.extract_and_summarize(
                    "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-2026-Premier-League",
                    mock_http,
                )

    def test_extract_raises_on_bad_url(self):
        ext = FBRefExtractor()
        with pytest.raises(ExtractionError, match="game_id"):
            ext.extract_and_summarize(
                "https://fbref.com/en/squads/foo",
                MagicMock(),
            )
