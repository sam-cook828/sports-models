"""Leak-free features: every feature uses only games before the one being predicted.

One feature table serves all three stats (points, assists, rebounds); `feature_cols(target)` picks the
columns each model uses.
"""
import numpy as np
import pandas as pd

TARGETS = ["points", "assists", "rebounds"]
STATS = ["points", "minutes", "field_goals_attempted", "three_point_field_goals_attempted",
         "free_throws_attempted", "assists", "rebounds", "offensive_rebounds", "defensive_rebounds",
         "turnovers", "starter", "fga_share"]
WINDOWS = (5, 10)
VAC = ["vac_minutes", "vac_fga_share", "vac_points", "vac_assists", "vac_rebounds", "vac_players"]
MIN_AVG_MINUTES = 10  # prior 10-game average needed to be modeled


def _lagged(g, col: str, w: int | None) -> pd.Series:
    if w is None:
        return g[col].transform(lambda s: s.shift(1).expanding().mean())
    return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())


def _opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Each defense's last-10-game totals allowed (points, assists, rebounds), overall and by position group."""
    # a row's opponent is the defense; its team_score is what that defense allowed, opponent_team_score what it scored
    t = (df.groupby(["game_id", "game_date", "opponent_team_abbreviation"], as_index=False)
         .agg(pts_allowed=("team_score", "first"), pts_scored=("opponent_team_score", "first"))
         .rename(columns={"opponent_team_abbreviation": "defense"}))
    t["game_total"] = t["pts_allowed"] + t["pts_scored"]
    for stat in TARGETS:
        tot = df.groupby(["game_id", "opponent_team_abbreviation"])[stat].sum(min_count=1).rename(f"{stat}_allowed")
        by_pos = (df[df["pos_group"] != ""].pivot_table(index=["game_id", "opponent_team_abbreviation"],
                                                         columns="pos_group", values=stat, aggfunc="sum")
                  .add_prefix(f"{stat}_allowed_"))
        both = pd.concat([tot, by_pos], axis=1).reset_index().rename(columns={"opponent_team_abbreviation": "defense"})
        t = t.merge(both, on=["game_id", "defense"], how="left")
    t = t.sort_values(["defense", "game_date"])
    g = t.groupby("defense")
    out = t[["game_id", "defense"]].copy()
    for c in t.columns.drop(["game_id", "game_date", "defense", "pts_scored"]):
        out[f"opp_{c}_10"] = g[c].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    return out.rename(columns={"defense": "opponent_team_abbreviation"})


def _absences(df: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: the recent minutes, shot share, points, assists and rebounds of rotation players who sat.

    A rotation player is anyone who appeared for the team in its previous 10 games of the same season and was
    averaging 10+ minutes. Players who have since appeared for another team (traded, released) are not counted
    as absent. For upcoming games, players left out of the placeholders (listed Out) count as absent.
    """
    keep = ["minutes", "fga_share", "points", "assists", "rebounds"]
    app = df[["athlete_id", "team_abbreviation", "game_id", "game_date", "season", *keep]].copy()
    app = app.sort_values(["athlete_id", "game_date"])
    ga = app.groupby("athlete_id")
    for c in keep:  # form as of the end of each game played
        app[f"{c}_now"] = ga[c].transform(lambda s: s.rolling(10, min_periods=1).mean())
    app["next_team"], app["next_date"] = ga["team_abbreviation"].shift(-1), ga["game_date"].shift(-1)

    tg = app[["team_abbreviation", "game_id", "game_date", "season"]].drop_duplicates(["team_abbreviation", "game_id"])
    tg = tg.sort_values(["team_abbreviation", "game_date"])
    tg["k"] = tg.groupby("team_abbreviation").cumcount()
    app = app.merge(tg[["team_abbreviation", "game_id", "k"]], on=["team_abbreviation", "game_id"])

    # each appearance makes the player a candidate for the team's next 10 games
    now = [f"{c}_now" for c in keep]
    cand = app.loc[app["minutes_now"] >= 10, ["athlete_id", "team_abbreviation", "k", "game_date", "season",
                                              "next_team", "next_date", *now]]
    cand = cand.loc[cand.index.repeat(10)].copy()
    cand["k"] = cand["k"] + np.tile(np.arange(1, 11), len(cand) // 10)
    cand = cand.sort_values("game_date").drop_duplicates(["athlete_id", "team_abbreviation", "k"], keep="last")
    cand = cand.merge(tg.rename(columns={"game_date": "date_k", "season": "season_k"}), on=["team_abbreviation", "k"])
    cand = cand[cand["season"] == cand["season_k"]]  # offseason departures aren't absences
    played = app[["athlete_id", "team_abbreviation", "k"]].assign(played=True)
    cand = cand.merge(played, on=["athlete_id", "team_abbreviation", "k"], how="left")
    moved = (cand["next_team"].notna() & (cand["next_team"] != cand["team_abbreviation"])
             & (cand["next_date"] <= cand["date_k"]))
    absent = cand[cand["played"].isna() & ~moved]
    out = absent.groupby(["team_abbreviation", "game_id"]).agg(
        **{f"vac_{c}": (f"{c}_now", "sum") for c in keep}, vac_players=("athlete_id", "size")).reset_index()
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team_fga = df.groupby(["game_id", "team_abbreviation"])["field_goals_attempted"].transform("sum")
    df["fga_share"] = df["field_goals_attempted"] / team_fga.replace(0, np.nan)  # usage proxy
    df = df.sort_values(["athlete_id", "game_date"])
    g = df.groupby("athlete_id")
    gs = df.groupby(["athlete_id", "season"])
    cols = {}
    for c in STATS:
        for w in WINDOWS:
            cols[f"{c}_r{w}"] = _lagged(g, c, w)
        cols[f"{c}_season"] = _lagged(gs, c, None)
        cols[f"{c}_career"] = _lagged(g, c, None)
    for t in TARGETS:
        cols[f"last_game_{t}"] = g[t].shift(1)
        cols[f"{t}_std_10"] = g[t].transform(lambda s: s.shift(1).rolling(10, min_periods=2).std())
    cols["games_prior"] = g.cumcount()
    cols["season_games_prior"] = gs.cumcount()
    cols["last_game_minutes"] = g["minutes"].shift(1)
    rest = g["game_date"].diff().dt.days
    cols["rest_days"] = rest.clip(upper=10)
    cols["back_to_back"] = (rest == 1).astype(float)
    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    for t in TARGETS:  # production rate, separate from playing time
        for s in ("r5", "r10", "season"):
            df[f"{t}_per_min_{s}"] = df[f"{t}_{s}"] / df[f"minutes_{s}"].replace(0, np.nan)

    # teammates and opponents missing tonight
    vac = _absences(df)
    df = df.merge(vac, on=["team_abbreviation", "game_id"], how="left")
    opp_vac = vac.rename(columns={"team_abbreviation": "opponent_team_abbreviation", **{c: f"opp_{c}" for c in VAC}})
    df = df.merge(opp_vac, on=["opponent_team_abbreviation", "game_id"], how="left")
    df[VAC + [f"opp_{c}" for c in VAC]] = df[VAC + [f"opp_{c}" for c in VAC]].fillna(0)

    opp = _opponent(df)  # upcoming games are included so their rows get the latest defensive form
    df = df.merge(opp, on=["game_id", "opponent_team_abbreviation"], how="left")
    for t in TARGETS:  # what this defense allows to the player's position group
        df[f"opp_{t}_allowed_pos_10"] = np.select(
            [df["pos_group"] == p for p in ("G", "F", "C")],
            [df.get(f"opp_{t}_allowed_{p}_10", np.nan) for p in ("G", "F", "C")], default=np.nan)
    for p in ("G", "F", "C"):
        df[f"pos_{p}"] = (df["pos_group"] == p).astype(float)
    return df[df["minutes_r10"] >= MIN_AVG_MINUTES].dropna(subset=["points_r10"])


def feature_cols(target: str) -> list[str]:
    base = [f"{c}_{s}" for c in STATS for s in ("r5", "r10", "season", "career")]
    rates = [f"{t}_per_min_{s}" for t in TARGETS for s in ("r5", "r10", "season")]
    return (base + rates + VAC + [f"opp_{c}" for c in VAC]
            + [f"last_game_{target}", f"{target}_std_10", "last_game_minutes", "games_prior", "season_games_prior",
               "rest_days", "back_to_back", "home", "opp_pts_allowed_10", "opp_game_total_10",
               f"opp_{target}_allowed_10", f"opp_{target}_allowed_pos_10", "pos_G", "pos_F", "pos_C"])
