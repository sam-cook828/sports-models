# nfl-stats

Predict NFL players' weekly **passing, rushing and receiving yards**, **receptions**, **anytime touchdowns**
(rushing + receiving) and **passing touchdowns**, plus the **chance of reaching thresholds** (e.g. 60+ receiving
yards, 5+ receptions, 1+ TD), from public nflverse data.

## Quick start

```
uv sync
uv run python -m nfl_stats.train     # downloads data, trains every model, writes models/report.json
uv run python -m nfl_stats.predict   # predictions for the next unplayed week
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Trained models (`models/*.joblib`) and downloaded
data (`data/`) are not committed, so run `train` first. Training all twelve models takes about half an hour;
`train --target receptions` retrains one stat.

## Predicting upcoming games

```
uv run python -m nfl_stats.predict                            # next unplayed week, top 15 per stat
uv run python -m nfl_stats.predict --team KC --stat passing_yards
uv run python -m nfl_stats.predict --player mahomes --week 7
uv run python -m nfl_stats.predict --stat anytime_tds --top 30
uv run python -m nfl_stats.predict --model expected           # expected values only
uv run python -m nfl_stats.predict --model probs              # threshold probabilities only
uv run python -m nfl_stats.predict --csv preds.csv            # write every player to a file
```

Each run re-downloads the latest stats, schedule and injury report, so features reflect all completed games and
the current report. Players ruled Out are left out, and Questionable or Doubtful players are flagged in `status`.
Output per player: the `expected` value with a High / Medium / Low `confidence` label, their recent 8-game average
(`recent_avg`), and the probability of reaching each threshold:

| Stat (`--stat`) | Thresholds |
|---|---|
| `passing_yards` | 125, 150, 175, 200, 225, 250, 275, 300, 325, 350 |
| `rushing_yards`, `receiving_yards` | 25, 40, 50, 60, 70, 80, 90, 100 |
| `receptions` | 2, 3, 4, 5, 6, 7, 8 |
| `anytime_tds` | 1, 2 |
| `passing_tds` | 1, 2, 3, 4 |

To change the thresholds shown, edit `THRESHOLDS` in `nfl_stats/probs.py`; the model itself is trained on a finer
grid (`GRID`), so changes within the grid need no retraining.

## Data

All data comes from [nflverse](https://github.com/nflverse), downloaded and cached as parquet in `data/`.

| Source | Used for |
|---|---|
| `stats_player_week_{year}.csv` (`stats_player` release) | Per-player, per-game box score stats: yards, receptions, touchdowns, attempts, carries, targets, EPA, air yards, target share, WOPR, etc. Regular season only, 2016 to present. |
| `snap_counts_{year}.csv` (`snap_counts` release) | Offensive and defensive snap share per player-game. Joined to the stats via `players.csv`, which maps PFR ids to GSIS ids. |
| `injuries_{year}.csv` (`injuries` release) | Weekly injury reports: game status (Questionable, Doubtful, Out) and practice participation, 2016 to present. |
| `games.csv` (`nfldata` repo) | The schedule and closing Vegas lines (spread and over/under total) for every game. `predict` also uses it to find the next unplayed week and each team's opponent. |
| `play_by_play_{year}` and `pbp_participation_{year}` | Man vs zone coverage splits for receivers and man/zone rates for defenses, 2018-2025 (receiving yards and receptions only). |

Each row is one player in one game. Only players with meaningful recent usage are modeled: a prior 8-game
average of at least 10 pass attempts (passing yards and TDs), 3 carries (rushing), 2 targets (receiving yards and
receptions) or 3 touches, meaning carries plus targets (anytime TDs).

## How the models work

Each stat has two independent models that share the same features:

| Model | Module | Saved as | Answers |
|---|---|---|---|
| Expected value | `nfl_stats/expected.py` | `models/<stat>_expected.joblib` | What should we expect on average? |
| Threshold probabilities | `nfl_stats/probs.py` | `models/<stat>_probs.joblib` | What is the chance of reaching 50 yards, 5 catches, 1 TD...? |

Each can be trained (`train --model expected|probs`) and run (`predict --model expected|probs`) on its own, and each
has its own section in `models/report.json`.

### Features (`nfl_stats/features.py`)

All features use only information available **before** kickoff, so there is no leakage from the game itself.

- **Player form:** rolling 3-game and 8-game averages and the career average of the stat and of volume (attempts,
  carries, targets or touches), plus stat-specific inputs such as EPA, air yards, first downs, receptions,
  interceptions, sacks, and rushing and receiving touchdowns.
- **Role and usage:** snap share, target share and carry share (the player's fraction of his team's carries),
  each as rolling averages.
- **Consistency:** the last game's value, the 8-game standard deviation, and games played.
- **Opponent:** rolling 3- and 8-game average of the stat allowed by the opposing defense.
- **Vegas lines:** the team's point spread, the game's over/under total, each team's implied points
  (derived from the spread and total) and home/away. Implied points matter most for touchdowns; lines also
  capture game script (a big favorite runs more late, a team expected to trail passes more).
- **Injuries** (`nfl_stats/injuries.py`): the player's own game status and practice participation; the recent
  target and carry share of teammates ruled Out or Doubtful (volume someone else has to absorb); and the recent
  snap share of the opponent's defensive backs and front seven who are ruled Out or Doubtful.
- **Coverage** (`nfl_stats/coverage.py`, receiving yards and receptions): the player's previous-season targets and
  yards per route run against man and zone coverage, the opponent's previous-season man and zone rates, and the
  two combined for this matchup. Every pass snap counts as a route, since per-player routes aren't published.
- **Other:** position and week of the season.

### Expected value (`nfl_stats/expected.py`)

A scikit-learn `HistGradientBoostingRegressor` with squared-error loss, so it predicts the **mean**. Yardage is
right-skewed (occasional big games), so the expected value sits above the most likely outcome. For the most
likely range, use the threshold probabilities. For touchdowns the expected value is the average number of TDs,
e.g. 0.35 anytime TDs.

The trees alone pull the best players too far toward the middle and can carry a season-level bias (passing yards
once ran 8 yards high). A linear calibration step fixes most of this: it blends the tree prediction with the
player's 8-game average, with weights learned from out-of-sample predictions over the five seasons before the test
set (each season predicted by a model trained only on earlier seasons). The learned weights are in
`models/report.json` under `calibration`.

**Confidence labels.** Each expected value carries a High / Medium / Low label. A second model learns the
calibrated prediction's expected absolute miss from the same out-of-sample errors, and the miss is divided by the
projection. The labels split past predictions into equal thirds, High being the smallest relative miss. On the
test set, the average miss as a share of the projection:

| Stat | High | Medium | Low |
|---|---|---|---|
| Passing yards | 23% | 28% | 39% |
| Rushing yards | 45% | 60% | 77% |
| Receiving yards | 50% | 65% | 78% |
| Receptions | 43% | 53% | 61% |
| Anytime TDs | 117% | 148% | 167% |
| Passing TDs | 57% | 70% | 77% |

The label is about relative accuracy. High-volume players get more High labels but still miss by more in
absolute terms. Anytime TDs are mostly 0 or 1, so their expected value always misses by a lot; for TDs, use the
probability of 1+ instead.

### Threshold probabilities (`nfl_stats/probs.py`)

A `HistGradientBoostingClassifier` per stat. Every player-game is repeated once per threshold, with a label of
whether the player reached it, and the threshold (and its gap to the player's 8-game average) is added as a feature.
Both threshold features have a decreasing monotonic constraint, so P(60+) can never exceed P(50+). The model is
trained on a fine grid (every 5 yards; every 25 passing yards; every reception; 1-3 TDs; 1-5 passing TDs), so
`probs.prob_over` can give the chance of any line, such as a sportsbook's 92.5, by interpolating between grid points.

### Training and evaluation (`nfl_stats/train.py`)

Splits are by time: seasons before 2024 train, 2024 picks the number of boosting rounds, and the final models are
refit on both. Seasons after 2024 (2025 and the 2026 weeks played so far) are the held-out test set.

**Expected value vs. a baseline that predicts the player's 8-game average (test set):**

| Stat | RMSE model | RMSE baseline | MAE model | MAE baseline |
|---|---|---|---|---|
| Passing yards | 81.1 | 88.4 | 64.1 | 69.6 |
| Rushing yards | 28.5 | 30.0 | 21.0 | 22.3 |
| Receiving yards | 27.5 | 28.8 | 20.8 | 21.8 |
| Receptions | 1.98 | 2.06 | 1.54 | 1.61 |
| Anytime TDs | 0.56 | 0.60 | 0.42 | 0.43 |
| Passing TDs | 1.09 | 1.15 | 0.89 | 0.94 |

**Threshold probabilities vs. the historical base rate for each threshold (log loss, lower is better):**

| Stat | Model | Base rate |
|---|---|---|
| Passing yards | 0.448 | 0.516 |
| Rushing yards | 0.374 | 0.478 |
| Receiving yards | 0.363 | 0.443 |
| Receptions | 0.400 | 0.485 |
| Anytime TDs | 0.358 | 0.386 |
| Passing TDs | 0.447 | 0.482 |

The probabilities are well calibrated: across the test set, outcomes predicted at 30% happen about 25-35% of the
time, and similarly across other probability ranges. One soft spot: anytime-TD chances of 40-50% came in at 39%.
Full metrics, including the calibration tables, are in `models/report.json`.

## Limitations

- Weekly stats are noisy, touchdowns most of all. The gain over a rolling average is modest, and individual
  predictions can miss widely.
- The model does not see depth charts, red-zone usage or weather. Injuries come from the weekly report only:
  players on injured reserve aren't listed, so their absence isn't counted, and for Sunday games the report isn't
  final until Friday.
- Backup quarterbacks get passing projections from their spot-duty history; they only matter if the starter is out.
- Predictions are only as current as the last run. Rerun `predict` after the final injury report.
- Vegas lines for upcoming games appear in the schedule about a week ahead. Predicting further out leaves them
  missing, which the models handle but with less accuracy.
- The very top of the range is still uncertain: there are only a few dozen test games per stat for the top 2% of
  players, and elite receivers still come out a few yards low.
- `predict` uses the saved models and does not retrain. Retrain after new seasons accumulate
  (`uv run python -m nfl_stats.train --val-season <year>`).
- The two models are trained separately, so they can disagree slightly (e.g. expected yards above the 50% point
  of the probabilities, which is normal for skewed outcomes).

## Layout

```
nfl_stats/data.py       download and cache nflverse data, merge snap counts and Vegas lines
nfl_stats/features.py   leak-free rolling features and opponent strength
nfl_stats/injuries.py   injury-report features
nfl_stats/coverage.py   man/zone coverage features
nfl_stats/train.py      train either or both models, write models/report.json
nfl_stats/expected.py   expected-value model with calibration and confidence labels
nfl_stats/probs.py      threshold-probability model
nfl_stats/backtest.py   rolling-origin validation across seasons
nfl_stats/predict.py    predictions for upcoming games
```
