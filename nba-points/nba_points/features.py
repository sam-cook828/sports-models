"""Leak-free features: every feature uses only games before the one being predicted."""
import numpy as np
import pandas as pd

TARGET = "points"
STATS = ["points", "minutes", "field_goals_attempted", "three_point_field_goals_attempted",
         "free_throws_attempted", "assists", "rebounds", "turnovers", "starter", "fga_share"]
WINDOWS = (5, 10)
MIN_AVG_MINUTES = 10  # prior 10-game average needed to be modeled


def _lagged(g, col: str, w: int | None) -> pd.Series:
    if w is None:
        return g[col].transform(lambda s: s.shift(1).expanding().mean())
    return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())


def _opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Each defense's recent points allowed: overall, and to guards / forwards / centers."""
    games = df.groupby(["game_id", "game_date", "opponent_team_abbreviation"], as_index=False)["opponent_team_score"].first()
    # opponent_team_score here is the points the *player's* team allowed; flip to the defense's view
    allowed = df.groupby(["game_id", "game_date", "opponent_team_abbreviation"], as_index=False)["team_score"].first()
    allowed = allowed.rename(columns={"opponent_team_abbreviation": "defense", "team_score": "pts_allowed"})
    allowed = allowed.merge(games.rename(columns={"opponent_team_abbreviation": "defense", "opponent_team_score": "pts_scored"}),
                            on=["game_id", "game_date", "defense"])
    by_pos = (df[df["pos_group"] != ""].pivot_table(index=["game_id", "opponent_team_abbreviation"], columns="pos_group",
                                                     values="points", aggfunc="sum")
              .add_prefix("allowed_").reset_index().rename(columns={"opponent_team_abbreviation": "defense"}))
    t = allowed.merge(by_pos, on=["game_id", "defense"], how="left").sort_values(["defense", "game_date"])
    t["game_total"] = t["pts_allowed"] + t["pts_scored"]
    g = t.groupby("defense")
    out = t[["game_id", "defense"]].copy()
    for c in ["pts_allowed", "game_total", "allowed_G", "allowed_F", "allowed_C"]:
        if c in t:
            out[f"opp_{c}_10"] = g[c].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    return out.rename(columns={"defense": "opponent_team_abbreviation"})


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
    cols["games_prior"] = g.cumcount()
    cols["season_games_prior"] = gs.cumcount()
    cols["last_game_points"] = g[TARGET].shift(1)
    cols["points_std_10"] = g[TARGET].transform(lambda s: s.shift(1).rolling(10, min_periods=2).std())
    rest = g["game_date"].diff().dt.days
    cols["rest_days"] = rest.clip(upper=10)
    cols["back_to_back"] = (rest == 1).astype(float)
    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

    opp = _opponent(df)  # upcoming games are included so their rows get the latest defensive form
    df = df.merge(opp, on=["game_id", "opponent_team_abbreviation"], how="left")
    pos_cols = {"G": "opp_allowed_G_10", "F": "opp_allowed_F_10", "C": "opp_allowed_C_10"}
    df["opp_allowed_pos_10"] = [row.get(pos_cols.get(p, ""), np.nan) for p, row in
                                zip(df["pos_group"], df[list(pos_cols.values())].to_dict("records"))]
    for p in ("G", "F", "C"):
        df[f"pos_{p}"] = (df["pos_group"] == p).astype(float)
    df["baseline"] = df["points_r10"]
    return df[df["minutes_r10"] >= MIN_AVG_MINUTES].dropna(subset=["points_r10"])


def feature_cols() -> list[str]:
    base = [f"{c}_{s}" for c in STATS for s in ("r5", "r10", "season", "career")]
    return base + ["games_prior", "season_games_prior", "last_game_points", "points_std_10", "rest_days",
                   "back_to_back", "home", "opp_pts_allowed_10", "opp_game_total_10", "opp_allowed_pos_10",
                   "pos_G", "pos_F", "pos_C"]
