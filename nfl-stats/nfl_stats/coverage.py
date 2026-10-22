"""Man vs zone coverage splits for receivers, and man/zone rates for defenses, by season.

Built from nflverse play-by-play (targeted receiver, yards) joined to participation data (players on the field,
man/zone label). Coverage labels exist for 2018-2025. Per-player routes run aren't published, so every dropback a
player was on the field for counts as a route; this is accurate for WRs and overstates routes for TEs and RBs.

Features are attached from the *previous* season, so training rows and live predictions are built the same way.
"""
import io

import pandas as pd
import pyarrow.parquet as pq
import requests

from .data import DATA_DIR, TEAM_FIXES

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
PART_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{year}.csv"
FIRST_SEASON = 2018  # first season with man/zone labels
MIN_SNAPS = 30  # fewer pass snaps vs a coverage type than this -> rate is too noisy, leave missing

PLAYER_FEATURES = ["tprr_man", "tprr_zone", "yprr_man", "yprr_zone"]
DEFENSE_FEATURES = ["def_man_rate", "def_zone_rate"]
MATCHUP_FEATURES = ["matchup_tprr", "matchup_yprr"]
FEATURES = PLAYER_FEATURES + DEFENSE_FEATURES + MATCHUP_FEATURES


def _plays(year: int, refresh: bool) -> pd.DataFrame | None:
    """Regular-season dropbacks with a man/zone label: one row per play."""
    path = DATA_DIR / f"coverage_plays_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        r = requests.get(PBP_URL.format(year=year), timeout=300)
        r.raise_for_status()
        pbp = pq.read_table(io.BytesIO(r.content), columns=[
            "game_id", "play_id", "season", "season_type", "posteam", "defteam", "qb_dropback",
            "two_point_attempt", "receiver_player_id", "receiving_yards"]).to_pandas()
        part = pd.read_csv(PART_URL.format(year=year), low_memory=False,
                           usecols=["nflverse_game_id", "play_id", "offense_players", "defense_man_zone_type"])
    except Exception as e:
        print(f"skip coverage {year}: {e}")
        return None
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1) & (pbp["two_point_attempt"] != 1)]
    part = part[part["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
    plays = pbp.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"])
    plays["coverage"] = plays["defense_man_zone_type"].map({"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone"})
    plays["defteam"] = plays["defteam"].replace(TEAM_FIXES)
    plays = plays[["season", "defteam", "coverage", "offense_players", "receiver_player_id", "receiving_yards"]]
    DATA_DIR.mkdir(exist_ok=True)
    plays.to_parquet(path)
    return plays


def season_tables(last: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per player-season coverage splits and per defense-season man/zone rates."""
    frames = [p for y in range(FIRST_SEASON, last + 1) if (p := _plays(y, refresh)) is not None]
    plays = pd.concat(frames, ignore_index=True)

    # routes (pass snaps): one row per player on the field per play
    on_field = plays[["season", "coverage", "offense_players"]].assign(
        player_id=plays["offense_players"].str.split(";")).explode("player_id")
    snaps = on_field.groupby(["player_id", "season", "coverage"]).size().rename("snaps")
    tgt = plays[plays["receiver_player_id"].notna()].rename(columns={"receiver_player_id": "player_id"})
    targets = tgt.groupby(["player_id", "season", "coverage"]).agg(
        targets=("player_id", "size"), yards=("receiving_yards", lambda s: s.fillna(0).sum()))
    p = pd.concat([snaps, targets], axis=1).fillna({"targets": 0, "yards": 0})
    p = p[p["snaps"] >= MIN_SNAPS]
    p["tprr"], p["yprr"] = p["targets"] / p["snaps"], p["yards"] / p["snaps"]
    players = p[["tprr", "yprr"]].unstack("coverage")
    players.columns = [f"{stat}_{cov}" for stat, cov in players.columns]
    players = players.reset_index()

    rates = pd.crosstab([plays["defteam"], plays["season"]], plays["coverage"], normalize="index")
    defenses = rates.rename(columns={"man": "def_man_rate", "zone": "def_zone_rate"}).reset_index()
    defenses = defenses.rename(columns={"defteam": "opponent_team"})[["opponent_team", "season", *DEFENSE_FEATURES]]
    return players, defenses


def add_features(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Attach the previous season's coverage splits (player) and man/zone rates (opponent defense)."""
    players, defenses = season_tables(int(df["season"].max()) - 1, refresh)
    players["season"] += 1
    defenses["season"] += 1
    df = df.merge(players, on=["player_id", "season"], how="left")
    df = df.merge(defenses, on=["opponent_team", "season"], how="left")
    # what the player's splits imply against this particular defense's coverage mix
    df["matchup_tprr"] = df["def_man_rate"] * df["tprr_man"] + df["def_zone_rate"] * df["tprr_zone"]
    df["matchup_yprr"] = df["def_man_rate"] * df["yprr_man"] + df["def_zone_rate"] * df["yprr_zone"]
    return df
