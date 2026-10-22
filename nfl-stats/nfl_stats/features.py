"""Leak-free feature engineering: every feature uses only games before the one being predicted."""
import pandas as pd

from . import coverage, injuries

# target -> (volume column, extra stat columns used as rolling features)
TARGETS = {
    "passing_yards": ("attempts", ["completions", "passing_air_yards", "passing_epa", "sacks_suffered", "rushing_yards", "offense_pct"]),
    "rushing_yards": ("carries", ["rushing_epa", "rushing_first_downs", "targets", "receiving_yards", "offense_pct", "target_share", "carry_share"]),
    "receiving_yards": ("targets", ["receptions", "receiving_air_yards", "receiving_yards_after_catch", "target_share", "air_yards_share", "wopr", "carries", "offense_pct"]),
    "receptions": ("targets", ["receiving_yards", "receiving_air_yards", "target_share", "air_yards_share", "wopr", "carries", "offense_pct"]),
    # rushing + receiving touchdowns (the anytime-scorer prop); touches = carries + targets
    "anytime_tds": ("touches", ["rushing_tds", "receiving_tds", "carries", "targets", "rushing_yards", "receiving_yards",
                                "rushing_first_downs", "receiving_first_downs", "target_share", "carry_share", "offense_pct"]),
    "passing_tds": ("attempts", ["passing_yards", "completions", "passing_air_yards", "passing_epa", "passing_interceptions",
                                 "sacks_suffered", "offense_pct"]),
}
MIN_AVG_VOLUME = {"attempts": 10, "carries": 3, "targets": 2, "touches": 3}  # prior-game average needed to be modeled
COVERAGE_TARGETS = ("receiving_yards", "receptions")
WINDOWS = (3, 8)
# known before kickoff, so used as-is (no lagging)
GAME_CONTEXT = ["is_home", "team_spread", "total_line", "team_implied", "opp_implied"]


def _roll(g: pd.core.groupby.DataFrameGroupBy, col: str, w: int) -> pd.Series:
    return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())


def _opp_allowed(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rolling yards of this type allowed by each defense in its previous games."""
    t = df.groupby(["season", "week", "opponent_team"], as_index=False)[target].sum()
    t = t.sort_values(["opponent_team", "season", "week"])
    g = t.groupby("opponent_team")
    for w in WINDOWS:
        t[f"opp_allowed_{w}"] = g[target].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    return t.drop(columns=target)


def build(df: pd.DataFrame, target: str, refresh_injuries: int | None = None) -> pd.DataFrame:
    vol, extras = TARGETS[target]
    df = df.sort_values(["player_id", "season", "week"]).copy()
    g = df.groupby("player_id")
    cols = {}
    for c in [target, vol, *extras]:
        for w in WINDOWS:
            cols[f"{c}_r{w}"] = _roll(g, c, w)
        cols[f"{c}_career"] = g[c].transform(lambda s: s.shift(1).expanding().mean())
    cols["games_prior"] = g.cumcount()
    cols["last_game_value"] = g[target].shift(1)
    cols["value_std_8"] = g[target].transform(lambda s: s.shift(1).rolling(8, min_periods=2).std())
    df = pd.concat([df, pd.DataFrame(cols)], axis=1)
    df = df.merge(_opp_allowed(df, target), on=["season", "week", "opponent_team"], how="left")
    df = pd.concat([df, pd.get_dummies(df["position"], prefix="pos", dtype=float)], axis=1)
    df["week_num"] = df["week"]
    df["baseline"] = df[f"{target}_r8"]
    if target in COVERAGE_TARGETS:
        df = coverage.add_features(df)
    df = injuries.add_features(df, refresh_injuries)

    df = df[df[f"{vol}_r8"] >= MIN_AVG_VOLUME[vol]].dropna(subset=[f"{target}_r8"])
    return df


def feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    vol, extras = TARGETS[target]
    base = [f"{c}_{s}" for c in [target, vol, *extras] for s in ("r3", "r8", "career")]
    return base + ["games_prior", "last_game_value", "value_std_8", "opp_allowed_3", "opp_allowed_8", "week_num", *GAME_CONTEXT] + [
        c for c in df.columns if c.startswith("pos_")
    ] + (coverage.FEATURES if target in COVERAGE_TARGETS else []) + injuries.FEATURES
