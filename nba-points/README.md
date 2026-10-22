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
  three-point and free-throw attempts, assists, rebounds, turnovers, starts, and share of team shot attempts.
- **Consistency and schedule:** last game's points, 10-game standard deviation, games played, rest days,
  back-to-backs, home/away.
- **Opponent:** the defense's last-10-game points allowed, its games' combined scoring (a pace proxy), and points
  allowed to the player's position group (guards, forwards, centers).

The model is a scikit-learn `HistGradientBoostingRegressor` predicting mean points, with the same
calibration and confidence-label method as `nfl-yards` (see its README). Seasons before 2023-24 train,
2023-24 tunes the number of boosting rounds, and 2024-25 onward is the held-out test set.

| Test set (47,306 player-games) | Model | 10-game average |
|---|---|---|
| RMSE (points) | 6.20 | 6.43 |
| Mean absolute error | 4.79 | 4.96 |

Average miss as a share of the projection, by confidence label: High 32%, Medium 45%, Low 57%.

## Limitations

- No projected minutes, starting lineups, Vegas lines, or teammates' injuries yet; these are the likely next
  improvements. Players listed Out are dropped, but a teammate's absence doesn't raise anyone's projection.
- A player who changed teams in the offseason appears under his new team only after his first game for it.
