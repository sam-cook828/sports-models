"""Injury-report features from the weekly NFL injury reports (nflverse, 2016 onward).

Three kinds of feature, all known before kickoff:
- the player's own game status (Questionable / Doubtful / Out) and practice participation;
- teammates' absences: the recent target and carry share held by teammates ruled Out or Doubtful,
  volume that has to go to someone else;
- the opponent's defensive absences: the recent snap share of their defensive backs, and of their front seven,
  who are ruled Out or Doubtful.

"Recent" share means the player's average over his last 8 games before that week.
Players on injured reserve are not on the weekly report, so their absence is not counted.
"""
import pandas as pd

from .data import DATA_DIR, TEAM_FIXES, _cached, load_snaps

URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv"
STATUS = {"Questionable": 1, "Doubtful": 2, "Out": 3}
PRACTICE = {"Limited Participation in Practice": 1, "Did Not Participate In Practice": 2}
MISSING = 2  # Doubtful or worse counts as missing the game
DB = {"CB", "S", "SS", "FS", "DB"}
FRONT = {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB"}

PLAYER_FEATURES = ["inj_status", "inj_practice"]
TEAM_FEATURES = ["vac_target_share", "vac_carry_share"]
OPP_FEATURES = ["opp_db_out", "opp_front_out"]
FEATURES = PLAYER_FEATURES + TEAM_FEATURES + OPP_FEATURES


def load(start: int, end: int, refresh_season: int | None = None) -> pd.DataFrame:
    """One row per player-week on a report. `refresh_season` re-downloads that season (the current one)."""
    frames = []
    for y in range(start, end + 1):
        f = _cached(URL.format(year=y), f"injuries_{y}", y == refresh_season)
        if f is not None:
            frames.append(f)
    inj = pd.concat(frames, ignore_index=True)
    inj = inj[inj["game_type"] == "REG"].rename(columns={"gsis_id": "player_id"})
    inj["team"] = inj["team"].replace(TEAM_FIXES)
    inj["inj_status"] = inj["report_status"].map(STATUS).fillna(0)
    inj["inj_practice"] = inj["practice_status"].map(PRACTICE).fillna(0)
    # a player can appear twice in a week (e.g. a trade); keep the worse entry
    inj = inj.sort_values("inj_status").drop_duplicates(["player_id", "season", "week"], keep="last")
    return inj[["player_id", "season", "week", "team", "position", "inj_status", "inj_practice"]]


def _recent(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each player's 8-game average of `cols` as of the end of each game he played (rows keyed by time `t`)."""
    rows = rows.sort_values(["player_id", "t"])
    g = rows.groupby("player_id")
    out = rows[["player_id", "t"]].copy()
    for c in cols:
        out[c] = g[c].transform(lambda s: s.rolling(8, min_periods=1).mean())
    return out


def _as_of(inj: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """Attach each injured player's most recent averages from games before that week."""
    left = inj.sort_values("t")
    return pd.merge_asof(left, recent.sort_values("t"), on="t", by="player_id", allow_exact_matches=False)


def add_features(df: pd.DataFrame, refresh_season: int | None = None) -> pd.DataFrame:
    first, last = int(df["season"].min()), int(df["season"].max())
    inj = load(first, last, refresh_season)
    inj["t"] = inj["season"] * 100 + inj["week"]
    out = inj[inj["inj_status"] >= MISSING]

    # teammates' vacated volume
    stats = df[df["target_share"].notna() | df["carry_share"].notna()].assign(t=lambda d: d.season * 100 + d.week)
    off = _as_of(out, _recent(stats, ["target_share", "carry_share"]))
    team_vac = (off.groupby(["team", "season", "week"])[["target_share", "carry_share"]].sum()
                .rename(columns={"target_share": "vac_target_share", "carry_share": "vac_carry_share"}).reset_index())

    # opponent's missing defenders, by recent defensive snap share
    snaps = load_snaps(first, last).assign(t=lambda d: d.season * 100 + d.week)
    dfn = _as_of(out[out["position"].isin(DB | FRONT)], _recent(snaps, ["defense_pct"]))
    dfn["grp"] = dfn["position"].map(lambda p: "opp_db_out" if p in DB else "opp_front_out")
    opp_out = (dfn.pivot_table(index=["team", "season", "week"], columns="grp", values="defense_pct", aggfunc="sum")
               .reindex(columns=OPP_FEATURES).reset_index().rename(columns={"team": "opponent_team"}))

    df = df.merge(inj[["player_id", "season", "week", *PLAYER_FEATURES]], on=["player_id", "season", "week"], how="left")
    df = df.merge(team_vac, on=["team", "season", "week"], how="left")
    df = df.merge(opp_out, on=["opponent_team", "season", "week"], how="left")
    df[FEATURES] = df[FEATURES].fillna(0)  # not on the report = healthy / nobody missing
    # a player's own share shouldn't count as vacated for himself
    own = _as_of(df.loc[df["inj_status"] >= MISSING, ["player_id", "season", "week"]]
                 .assign(t=lambda d: d.season * 100 + d.week).reset_index(), _recent(stats, ["target_share", "carry_share"]))
    if len(own):
        own = own.set_index("index")
        df.loc[own.index, "vac_target_share"] -= own["target_share"].fillna(0)
        df.loc[own.index, "vac_carry_share"] -= own["carry_share"].fillna(0)
    df[TEAM_FEATURES] = df[TEAM_FEATURES].clip(0, 1)  # trades can double-count a share
    return df
