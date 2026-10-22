"""Predict points, assists and rebounds for upcoming NBA games.

Finds the next date with unplayed regular-season games (or --date), adds a placeholder row per likely player
against that night's opponent, and runs the trained model on features built from prior games only.
Players listed Out on the latest injury report are left out.
"""
import argparse

import joblib
import pandas as pd

from . import expected
from .data import current_season, load_box, load_injuries, load_schedule
from .features import TARGETS, build
from .train import MODELS_DIR

SHORT = ["pts", "ast", "reb"]  # column names for points, assists, rebounds
ID_COLS = ["athlete_id", "athlete_display_name", "athlete_position_abbreviation", "pos_group", "team_abbreviation"]


def upcoming(season: int, date: str | None) -> pd.DataFrame:
    sched = load_schedule(season, refresh=True)
    open_games = sched[~sched["status_type_completed"].astype(bool)]
    if date:
        games = open_games[open_games["game_date"] == pd.Timestamp(date)]
    else:
        games = open_games[open_games["game_date"] == open_games["game_date"].min()]
    if games.empty:
        raise SystemExit(f"no unplayed regular-season games found{' on ' + date if date else ''}")
    return games


def placeholders(box: pd.DataFrame, games: pd.DataFrame, season: int) -> pd.DataFrame:
    sides = pd.concat([
        games.assign(team_abbreviation=games["home_abbreviation"], opponent_team_abbreviation=games["away_abbreviation"], home=1.0),
        games.assign(team_abbreviation=games["away_abbreviation"], opponent_team_abbreviation=games["home_abbreviation"], home=0.0),
    ])[["game_id", "game_date", "team_abbreviation", "opponent_team_abbreviation", "home"]]
    recent = box[box["season"] >= season - 1]
    latest = recent.drop_duplicates("athlete_id", keep="last")
    # a player whose last game was last season is only kept if his team hasn't played yet this season;
    # otherwise he has likely moved, retired or fallen out of the rotation
    played_this_season = set(box.loc[box["season"] == season, "team_abbreviation"])
    latest = latest[(latest["season"] == season) | ~latest["team_abbreviation"].isin(played_this_season)]
    ph = latest[ID_COLS].merge(sides, on="team_abbreviation")
    return ph.assign(season=season)


def predict(date: str | None = None) -> pd.DataFrame:
    season = current_season()
    games = upcoming(season, date)
    box = load_box(refresh_current=True)
    ph = placeholders(box, games, season)
    out = set(load_injuries(season).query("status == 'Out'")["athlete_id"])
    ph = ph[~ph["athlete_id"].isin(out)]

    df = build(pd.concat([box, ph], ignore_index=True))
    df = df[df["game_id"].isin(games["game_id"]) & df["points"].isna()].copy()
    cols = ["athlete_display_name", "athlete_position_abbreviation", "team_abbreviation", "opponent_team_abbreviation"]
    for target, short in zip(TARGETS, SHORT):
        bundle = joblib.load(MODELS_DIR / f"{target}_expected.joblib")
        pred, conf = expected.predict(bundle["model"], df.assign(baseline=df[f"{target}_r10"]), bundle["features"])
        df[short], df[f"{short}_conf"], df[f"{short}_avg"] = pred.round(1), conf, df[f"{target}_r10"].round(1)
        cols += [short, f"{short}_conf", f"{short}_avg"]
    res = df[cols]
    res = res.rename(columns={"athlete_display_name": "player", "athlete_position_abbreviation": "pos",
                              "team_abbreviation": "team", "opponent_team_abbreviation": "opp"})
    res.attrs["label"] = f"{games['game_date'].iloc[0]:%a %b %-d, %Y}"
    return res.sort_values("pts", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="game date, e.g. 2026-10-24 (default: next date with unplayed games)")
    ap.add_argument("--team", help="team abbreviation filter, e.g. BOS")
    ap.add_argument("--player", help="case-insensitive name substring filter")
    ap.add_argument("--top", type=int, default=25, help="rows to show (default 25)")
    ap.add_argument("--sort", choices=SHORT, default="pts", help="sort by this stat (default pts)")
    ap.add_argument("--csv", help="write all predictions to this path")
    args = ap.parse_args()

    res = predict(args.date)
    if args.csv:
        res.to_csv(args.csv, index=False)
    if args.team:
        res = res[(res["team"] == args.team.upper()) | (res["opp"] == args.team.upper())]
    if args.player:
        res = res[res["player"].str.contains(args.player, case=False, regex=False)]
    print(f"Predictions for {res.attrs.get('label', '')} (expected value, confidence, 10-game average)\n")
    print(res.sort_values(args.sort, ascending=False).head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
