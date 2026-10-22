"""Predict yards for upcoming games.

Finds the next unplayed week in the nflverse schedule, adds a placeholder row per active player
against that week's opponent, and runs the trained models on features built from prior games only.
"""
import argparse

import joblib
import pandas as pd

from . import expected, injuries, probs
from .data import load_all, load_schedule, vegas_lines
from .features import TARGETS, build
from .train import MODELS_DIR

ID_COLS = ["player_id", "player_name", "player_display_name", "position", "position_group", "season", "week",
           "season_type", "team", "opponent_team"]


def next_week(season: int | None, week: int | None) -> tuple[pd.DataFrame, int, int]:
    """Unplayed games for the requested (or next) week."""
    sched = load_schedule(refresh=True)
    unplayed = sched[sched["home_score"].isna()]
    if season is None:
        season = int(unplayed["season"].min())
    unplayed = unplayed[unplayed["season"] == season]
    if unplayed.empty:
        raise SystemExit(f"no unplayed regular-season games found for {season}")
    if week is None:
        week = int(unplayed["week"].min())
    games = unplayed[unplayed["week"] == week]
    if games.empty:
        raise SystemExit(f"no unplayed games in {season} week {week}")
    return games, season, week


def placeholders(raw: pd.DataFrame, games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    opp = {**dict(zip(games["home_team"], games["away_team"])), **dict(zip(games["away_team"], games["home_team"]))}
    # active = appeared this season or last, so retired/cut players drop out
    latest = raw[raw["season"] >= season - 1].drop_duplicates("player_id", keep="last")
    latest = latest[latest["team"].isin(opp)]
    ph = latest[ID_COLS].copy()
    ph["season"], ph["week"], ph["season_type"] = season, week, "REG"
    ph["opponent_team"] = ph["team"].map(opp)
    already_played = raw[(raw["season"] == season) & (raw["week"] == week)]["player_id"]
    ph = ph.merge(vegas_lines(games), on=["season", "week", "team"], how="left")
    return ph[~ph["player_id"].isin(already_played)]


def predict(season: int | None = None, week: int | None = None, model: str = "all") -> pd.DataFrame:
    """Run the expected-yards model, the threshold-probability model, or both ("all")."""
    games, season, week = next_week(season, week)
    raw = load_all(refresh=True)
    full = pd.concat([raw, placeholders(raw, games, season, week)], ignore_index=True)
    injuries.load(season, season, refresh_season=season)  # this week's report, fetched fresh

    out = []
    for target in TARGETS:
        df = build(full, target)
        df = df[(df["season"] == season) & (df["week"] == week) & df[target].isna()].copy()
        df = df[df["inj_status"] < injuries.STATUS["Out"]]  # ruled out: not playing
        df["status"] = df["inj_status"].map({1: "Questionable", 2: "Doubtful"}).fillna("")
        cols = ["player_display_name", "position", "team", "opponent_team", "status", "stat", "recent_avg"]
        df["recent_avg"] = df["baseline"].round(1)
        df["stat"] = target.replace("_yards", "")
        if model in ("expected", "all"):
            bundle = joblib.load(MODELS_DIR / f"{target}_expected.joblib")
            exp, conf = expected.predict(bundle["model"], _align(df, bundle["features"]), bundle["features"])
            df["exp_yards"], df["confidence"] = exp.round(1), conf
            cols[5:5] = ["exp_yards", "confidence"]
        if model in ("probs", "all"):
            bundle = joblib.load(MODELS_DIR / f"{target}_probs.joblib")
            p = probs.predict(bundle["model"], _align(df, bundle["features"]), bundle["features"], target)
            df = df.join(p)
            cols += list(p.columns)
        out.append(df[cols])
    res = pd.concat(out, ignore_index=True)
    res.attrs["label"] = f"{season} week {week}"
    return res.rename(columns={"player_display_name": "player", "opponent_team": "opp"})


def _align(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Add any position dummies the model was trained with but this week's players lack."""
    missing = [c for c in feats if c not in df and c.startswith("pos_")]
    return df.assign(**{c: 0.0 for c in missing})


def sort_col(df: pd.DataFrame) -> str:
    """Sort by expected yards, or by the lowest threshold's probability when only probabilities were run."""
    return "exp_yards" if "exp_yards" in df else next(c for c in df if c.startswith("p_") and df[c].notna().any())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--stat", choices=[t.replace("_yards", "") for t in TARGETS], help="only this yard type")
    ap.add_argument("--player", help="case-insensitive name substring filter")
    ap.add_argument("--team", help="team abbreviation filter, e.g. KC")
    ap.add_argument("--top", type=int, default=15, help="rows per stat (default 15)")
    ap.add_argument("--model", choices=["expected", "probs", "all"], default="all",
                    help="expected yards, threshold probabilities, or both (default)")
    ap.add_argument("--csv", help="write all predictions to this path")
    args = ap.parse_args()

    res = predict(args.season, args.week, args.model)
    if args.csv:
        res.sort_values(["stat", sort_col(res)], ascending=[True, False]).to_csv(args.csv, index=False)
    if args.stat:
        res = res[res["stat"] == args.stat]
    if args.player:
        res = res[res["player"].str.contains(args.player, case=False, regex=False)]
    if args.team:
        res = res[(res["team"] == args.team.upper()) | (res["opp"] == args.team.upper())]
    print(f"Predictions for {res.attrs.get('label', '')}")
    for stat, g in res.groupby("stat", sort=False):
        print(f"\n== {stat} yards ==")
        g = g.sort_values(sort_col(g), ascending=False).head(args.top).drop(columns="stat").dropna(axis=1, how="all")
        pcols = [c for c in g if c.startswith("p_")]
        g[pcols] = g[pcols].apply(lambda col: (col * 100).round().astype(int).astype(str) + "%")
        g = g.rename(columns={c: c.replace("p_", "") + "+" for c in pcols})
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
