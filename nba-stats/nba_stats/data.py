"""Download NBA box scores, schedules and injury reports (ESPN data via sportsdataverse) and cache them.

Seasons are labelled by the year they end: 2027 is the 2026-27 season.
"""
import io
from pathlib import Path

import pandas as pd
import requests

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
BOX_URL = BASE + "/espn_nba_player_boxscores/player_box_{season}.parquet"
SCHEDULE_URL = BASE + "/espn_nba_schedules/nba_schedule_{season}.parquet"
INJURIES_URL = BASE + "/espn_nba_injuries/injuries_{season}.parquet"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIRST_SEASON = 2016
REGULAR_SEASON = 2

BOX_COLS = ["game_id", "season", "season_type", "game_date", "athlete_id", "athlete_display_name",
            "athlete_position_abbreviation", "team_abbreviation", "opponent_team_abbreviation", "home_away",
            "minutes", "points", "field_goals_attempted", "three_point_field_goals_attempted", "free_throws_attempted",
            "assists", "rebounds", "offensive_rebounds", "defensive_rebounds", "turnovers", "starter", "did_not_play", "team_score", "opponent_team_score"]


def _cached(url: str, name: str, refresh: bool, columns: list[str] | None = None) -> pd.DataFrame | None:
    path = DATA_DIR / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content), columns=columns)
    except Exception as e:  # season not published yet
        print(f"skip {name}: {e}")
        return None
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(path)
    return df


def current_season() -> int:
    today = pd.Timestamp.today()
    return today.year + 1 if today.month >= 8 else today.year


def load_box(start: int = FIRST_SEASON, end: int | None = None, refresh_current: bool = False) -> pd.DataFrame:
    """Regular-season player games in which the player got on the floor."""
    end = end or current_season()
    frames = [f for s in range(start, end + 1)
              if (f := _cached(BOX_URL.format(season=s), f"box_v2_{s}", refresh_current and s == end, BOX_COLS)) is not None]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["season_type"] == REGULAR_SEASON) & ~df["did_not_play"].fillna(False) & (df["minutes"].fillna(0) > 0)]
    # All-Star and exhibition sides (EAST, WEST, ...) are listed as regular season; real teams play 20+ games
    team_games = df.groupby(["season", "team_abbreviation"])["game_id"].transform("nunique")
    current = df["season"] == df["season"].max()  # early in a season every team has only a few games
    df = df[(team_games >= 20) | (current & ~df["team_abbreviation"].isin(["EAST", "WEST"]))]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home"] = (df["home_away"] == "home").astype(float)
    df["starter"] = df["starter"].astype(float)
    pos = df["athlete_position_abbreviation"].fillna("")
    df["pos_group"] = pos.map(lambda p: "C" if "C" in p else ("G" if "G" in p else ("F" if "F" in p else "")))
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    s = _cached(SCHEDULE_URL.format(season=season), f"schedule_{season}", refresh)
    s = s[s["season_type"] == REGULAR_SEASON].copy()
    s["game_date"] = pd.to_datetime(s["game_date"])
    return s


def load_injuries(season: int) -> pd.DataFrame:
    """The latest injury report snapshot for the season (always fetched fresh)."""
    inj = _cached(INJURIES_URL.format(season=season), f"injuries_{season}", refresh=True)
    if inj is None or inj.empty:
        return pd.DataFrame(columns=["athlete_id", "status"])
    inj = inj[inj["as_of_date"] == inj["as_of_date"].max()]
    return inj[["athlete_id", "athlete_display_name", "status"]].drop_duplicates("athlete_id", keep="last")
