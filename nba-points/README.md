# nba-points

Predict NBA players' points per game, with a High / Medium / Low confidence label on each prediction.

```
uv sync
uv run python -m nba_points.train     # downloads data, trains, writes models/report.json
uv run python -m nba_points.predict   # the next date with unplayed games
uv run python -m nba_points.predict --date 2026-10-24 --team BOS
```

## Data

ESPN data published by [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data), cached as
parquet in `data/`. Seasons are labelled by the year they end (2027 is 2026-27).

| Source | Used for |
|---|---|
| `player_box_{season}` | Player box scores, regular season, 2015-16 onward |
| `nba_schedule_{season}` | Upcoming games and opponents, used by `predict` |
| `injuries_{season}` | The current injury report: `predict` leaves out players listed Out (current season only, so it isn't a training feature) |

Only players averaging at least 10 minutes over their previous 10 games are modeled.

## How the model works

All features use only games before the one being predicted.

- **Player form:** 5-game, 10-game, season-to-date and career averages of points, minutes, field-goal,
  three-point and free-throw attempts, assists, rebounds, turnovers, starts, and share of team shot attempts;
  points per minute (5-game, 10-game, season) and last game's minutes.
- **Missing teammates and opponents:** for each game, the recent minutes, shot share and points of rotation
  players who sat out (anyone who played for the team in its previous 10 games this season and was averaging
  10+ minutes). Box scores show who actually played, so this works for every past game; traded players are not
  counted as absent. In `predict`, players listed Out on the injury report are the absences.
- **Consistency and schedule:** last game's points, 10-game standard deviation, games played, rest days,
  back-to-backs, home/away.
- **Opponent:** the defense's last-10-game points allowed, its games' combined scoring (a pace proxy), and points
  allowed to the player's position group (guards, forwards, centers).

The model is a scikit-learn `HistGradientBoostingRegressor` predicting mean points, with the same
calibration and confidence-label method as `nfl-yards` (see its README). Seasons before 2023-24 train,
2023-24 tunes the number of boosting rounds, and 2024-25 onward is the held-out test set.

| Test set (47,202 player-games) | Model | 10-game average |
|---|---|---|
| RMSE (points) | 6.07 | 6.36 |
| Mean absolute error | 4.69 | 4.92 |

Average miss as a share of the projection, by confidence label: High 31%, Medium 44%, Low 55%.

All-Star games are listed as regular season in the source data and are removed. Feature groups were tested one at a
time on the held-out games; absences gave the biggest gain (RMSE 6.15 to 6.07), and two groups that didn't help
(games just missed, team scoring) were dropped.

## Limitations

- No projected minutes, starting lineups or Vegas lines yet; these are the likely next improvements.
- Absences only count players who have already played for the team this season, so in a team's first game of
  the season nobody is counted as missing.
- Early in a season the model leans on very few games and runs low for stars: in the test seasons, players
  averaging 24+ came out about 1.7 points low in their first three games (and about 0.6 high afterwards).
- In training, a player who sat for any reason (injury, rest, coach's decision) counts as absent; in `predict`,
  only players listed Out do. Late scratches and rest days aren't known until the injury report says so.
- A player who changed teams in the offseason appears under his new team only after his first game for it.
