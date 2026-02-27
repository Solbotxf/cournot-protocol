"""FBRef site extractor.

Uses the soccerdata library to fetch match data from fbref.com,
then builds a text summary for LLM consumption.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .base import ExtractionError, SiteExtractor


# League slug mapping (URL slug -> soccerdata league string)
_LEAGUE_MAP: dict[str, str] = {
    "Premier-League": "ENG-Premier League",
    "La-Liga": "ESP-La Liga",
    "Serie-A": "ITA-Serie A",
    "Bundesliga": "GER-Bundesliga",
    "Ligue-1": "FRA-Ligue 1",
}

# Default if we can't infer
_DEFAULT_LEAGUE = "ENG-Premier League"


def _parse_game_id(url: str) -> str | None:
    """Extract the FBRef game_id (hex hash) from a match URL.

    URL pattern: https://fbref.com/en/matches/{game_id}/...
    """
    m = re.search(r"fbref\.com/en/matches/([a-f0-9]+)", url)
    return m.group(1) if m else None


def _infer_league_season(url: str) -> tuple[str, str]:
    """Infer league and season from the FBRef match URL slug.

    Returns (league_string, season_string).
    Season is inferred from date in the URL slug if possible.
    """
    league = _DEFAULT_LEAGUE
    slug = url.split("/matches/")[1] if "/matches/" in url else ""
    # Slug looks like: 7e6892e4/Brentford-Arsenal-February-12-2026-Premier-League
    for url_slug, league_str in _LEAGUE_MAP.items():
        if url_slug in slug:
            league = league_str
            break

    # Infer season from year in slug — look for 4-digit year
    year_match = re.search(r"-(\d{4})-", slug)
    if year_match:
        year = int(year_match.group(1))
        # European seasons span two years: 2025-26 season runs Aug 2025 - May 2026
        # A match in Jan-May belongs to the season that started the prior year
        month_match = re.search(
            r"-(January|February|March|April|May|June|July|August|September|October|November|December)-",
            slug,
        )
        if month_match:
            month_name = month_match.group(1)
            early_months = {"January", "February", "March", "April", "May", "June", "July"}
            if month_name in early_months:
                start_year = year - 1
            else:
                start_year = year
        else:
            start_year = year
        season = f"{start_year}-{start_year + 1}"
    else:
        # Fallback: current season
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if now.month <= 7:
            start_year = now.year - 1
        else:
            start_year = now.year
        season = f"{start_year}-{start_year + 1}"

    return league, season


def _create_fbref_client(league: str, season: str) -> Any:
    """Create a soccerdata FBref client.

    Separated for easy mocking in tests.
    """
    from soccerdata import FBref
    return FBref(leagues=league, seasons=[season], no_cache=True)


def _summarize_player_stats(df: pd.DataFrame) -> str:
    """Build text summary from player match stats DataFrame."""
    lines: list[str] = []
    lines.append("=== PLAYER STATS (from fbref.com) ===")

    if df.empty:
        lines.append("  No player stats available.")
        return "\n".join(lines)

    # Group by team
    for team_name, team_df in df.groupby(level="team"):
        lines.append(f"\n  --- {team_name} ---")
        for idx, row in team_df.iterrows():
            player = idx[-1] if isinstance(idx, tuple) else str(idx)
            parts = [f"  {player}:"]
            for col in team_df.columns:
                val = row[col]
                if pd.notna(val):
                    parts.append(f"{col}={val}")
            lines.append(" ".join(parts))

    lines.append("")
    return "\n".join(lines)


def _summarize_shot_events(df: pd.DataFrame) -> str:
    """Build text summary from shot events DataFrame."""
    lines: list[str] = []
    lines.append("=== SHOT EVENTS (from fbref.com) ===")

    if df.empty:
        lines.append("  No shot events available.")
        return "\n".join(lines)

    for _, row in df.iterrows():
        parts: list[str] = []
        if "minute" in row and pd.notna(row["minute"]):
            parts.append(f"{int(row['minute'])}'")
        if "player" in row and pd.notna(row["player"]):
            parts.append(str(row["player"]))
        if "team" in row and pd.notna(row["team"]):
            parts.append(f"({row['team']})")
        if "outcome" in row and pd.notna(row["outcome"]):
            parts.append(f"— {row['outcome']}")
        if "distance" in row and pd.notna(row["distance"]):
            parts.append(f"dist={row['distance']}")
        if "body_part" in row and pd.notna(row["body_part"]):
            parts.append(row["body_part"])
        lines.append(f"  {' '.join(parts)}")

    lines.append("")
    return "\n".join(lines)


class FBRefExtractor(SiteExtractor):
    """Extracts match data from fbref.com using the soccerdata library."""

    source_id = "fbref"

    def can_handle(self, url: str) -> bool:
        return "fbref.com/en/matches/" in url

    def extract_and_summarize(
        self, url: str, http_client: Any,
    ) -> tuple[str, dict[str, Any]]:
        game_id = _parse_game_id(url)
        if not game_id:
            raise ExtractionError(f"Could not extract game_id from URL: {url}")

        league, season = _infer_league_season(url)
        fbref = _create_fbref_client(league, season)

        try:
            df_players = fbref.read_player_match_stats(
                stat_type="summary", match_id=game_id,
            )
        except Exception as e:
            raise ExtractionError(f"Failed to read player stats: {e}") from e

        try:
            df_shots = fbref.read_shot_events(match_id=game_id)
        except Exception:
            df_shots = pd.DataFrame()  # shots optional

        if df_players.empty and df_shots.empty:
            raise ExtractionError(f"No data returned from FBRef for game_id={game_id}")

        # Build text summary
        sections: list[str] = []
        sections.append(f"Match data from: {url}")
        sections.append(f"League: {league}, Season: {season}")
        sections.append("")
        sections.append(_summarize_player_stats(df_players))
        if not df_shots.empty:
            sections.append(_summarize_shot_events(df_shots))
        summary = "\n".join(sections)

        metadata = {
            "source_url": url,
            "fbref_game_id": game_id,
            "fbref_league": league,
            "fbref_season": season,
            "fbref_player_rows": len(df_players),
            "fbref_shot_rows": len(df_shots),
        }

        return summary, metadata
