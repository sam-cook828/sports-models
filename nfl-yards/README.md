# nfl-yards

Predict NFL players' weekly **passing, rushing and receiving yards**, and the **chance of reaching yardage
thresholds** (e.g. 60+ receiving yards), from public nflverse data.

## Quick start

```
uv sync
uv run python -m nfl_yards.train     # downloads data, trains both models, writes models/report.json
uv run python -m nfl_yards.predict   # predictions for the next unplayed week
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Trained models (`models/*.joblib`) and downloaded
data (`data/`) are not committed, so run `train` first.

## Predicting upcoming games

```
uv run python -m nfl_yards.predict                          # next unplayed week, top 15 per stat
uv run python -m nfl_yards.predict --team KC --stat passing
uv run python -m nfl_yards.predict --player mahomes --week 7
uv run python -m nfl_yards.predict --model expected         # expected yards only
uv run python -m nfl_yards.predict --model probs            # threshold probabilities only
uv run python -m nfl_yards.predict --csv preds.csv          # write every player to a file
```

Each run re-downloads the latest stats, schedule and injury report, so features reflect all completed games and
the current report. Players ruled Out are left out, and Questionable or Doubtful players are flagged in `status`. Output per player:
expected yards (`exp_yards`) with a High / Medium / Low `confidence` label, their recent 8-game average
(`recent_avg`), and the probability of reaching each threshold:

| Yard type | Thresholds |
|---|---|
| Rushing, receiving | 25, 40, 50, 60, 70, 80, 90, 100 |
| Passing | 125, 150, 175, 200, 225, 250, 275, 300, 325, 350 |

To change thresholds, edit `THRESHOLDS` in `nfl_yards/probs.py` and run `uv run python -m nfl_yards.train --model probs`.

## Data

All data comes from [nflverse](https://github.com/nflverse), downloaded as CSV and cached as parquet in `data/`.

| Source | Used for |
|---|---|
| `stats_player_week_{year}.csv` (`stats_player` release) | Per-player, per-game box score stats: yards, attempts, carries, targets, EPA, air yards, target share, WOPR, etc. Regular season only, 2016 to present. |
| `snap_counts_{year}.csv` (`snap_counts` release) | Offensive snap share per player-game. Joined to the stats via `players.csv`, which maps PFR ids to GSIS ids. |
| `injuries_{year}.csv` (`injuries` release) | Weekly injury reports: game status (Questionable, Doubtful, Out) and practice participation, 2016 to present. |
| `games.csv` (`nfldata` repo) | The schedule and closing Vegas lines (spread and over/under total) for every game. `predict` also uses it to find the next unplayed week and each team's opponent. |

Each row is one player in one game. Only players with meaningful recent usage are modeled: a prior 8-game
average of at least 10 pass attempts (passing), 3 carries (rushing) or 2 targets (receiving).

## How the models work

Each yard type (passing, rushing, receiving) has two independent models that share the same features:

| Model | Module | Saved as | Answers |
|---|---|---|---|
| Expected yards | `nfl_yards/expected.py` | `models/<yard type>_expected.joblib` | How many yards should we expect on average? |
| Threshold probabilities | `nfl_yards/probs.py` | `models/<yard type>_probs.joblib` | What is the chance of reaching 50, 60, 70... yards? |

Each can be trained (`train --model expected|probs`) and run (`predict --model expected|probs`) on its own, and each
has its own section in `models/report.json`.

### Features (`nfl_yards/features.py`)

All features use only information available **before** kickoff, so there is no leakage from the game itself.

- **Player form:** rolling 3-game and 8-game averages and the career average of yards and of volume (attempts,
  carries or targets), plus yard-type-specific stats such as EPA, air yards, first downs, receptions and sacks.
- **Role and usage:** snap share, target share and carry share (the player's fraction of his team's carries),
  each as rolling averages.
- **Consistency:** the last game's yards, the 8-game standard deviation, and games played.
- **Opponent:** rolling 3- and 8-game average of that yard type allowed by the opposing defense.
- **Vegas lines:** the team's point spread, the game's over/under total, each team's implied points
  (derived from the spread and total) and home/away. Lines capture expected game script: a big favorite tends to
  run more late, and a team expected to trail tends to pass more.
- **Injuries** (`nfl_yards/injuries.py`): the player's own game status and practice participation; the recent
  target and carry share of teammates ruled Out or Doubtful (volume someone else has to absorb); and the recent
  snap share of the opponent's defensive backs and front seven who are ruled Out or Doubtful.
- **Other:** position and week of the season.

### Expected yards (`nfl_yards/expected.py`)

A scikit-learn `HistGradientBoostingRegressor` with squared-error loss, so it predicts the **mean** (expected)
yardage. Yardage is right-skewed (occasional big games), so the expected value sits above the most likely
outcome. For the most likely range, use the threshold probabilities.

The trees alone pull the best players too far toward the middle (receivers in the top 2% by recent form came out
7.5 yards low) and ran 8 yards high on passing. A linear calibration step fixes most of this: it blends the tree
prediction with the player's 8-game average, with weights learned from out-of-sample predictions over the five
seasons before the test set (each season predicted by a model trained only on earlier seasons). The learned
weights are in `models/report.json` under `calibration`.

**Confidence labels.** Each expected-yards prediction carries a High / Medium / Low label. A second model learns
the calibrated prediction's expected absolute miss from the same out-of-sample errors, and the miss is divided by
the projection. The labels split past predictions into equal thirds, High being the smallest relative miss.
On the test set, the average miss as a share of the projection:

| Yard type | High | Medium | Low |
|---|---|---|---|
| Passing | 25% | 28% | 40% |
| Rushing | 44% | 61% | 79% |
| Receiving | 52% | 65% | 77% |

The label is about relative accuracy. High-volume players get more High labels but still miss by more yards in
absolute terms, because they gain more yards.

### Threshold probabilities (`nfl_yards/probs.py`)

A `HistGradientBoostingClassifier` per yard type. Every player-game is repeated once per threshold, with a label of
whether the player reached it, and the threshold (and its gap to the player's 8-game average) is added as a feature.
Both threshold features have a decreasing monotonic constraint, so P(60+) can never exceed P(50+).

### Training and evaluation (`nfl_yards/train.py`)

Splits are by time: seasons before 2024 train, 2024 picks the number of boosting rounds, and the final models are
refit on both. Seasons after 2024 (2025 and the 2026 weeks played so far) are the held-out test set.

**Expected yards vs. a baseline that predicts the player's 8-game average (test set, in yards):**

| Yard type | RMSE model | RMSE baseline | MAE model | MAE baseline |
|---|---|---|---|---|
| Passing | 81.7 | 89.1 | 64.6 | 70.2 |
| Rushing | 28.7 | 30.2 | 21.1 | 22.5 |
| Receiving | 27.6 | 28.8 | 20.8 | 21.8 |

Average bias (prediction minus actual), before and after calibration: passing +8.1 → +1.5 yards overall;
receiving's top 2% by recent form −7.5 → −3.1.

**Threshold probabilities vs. the historical base rate for each threshold (log loss, lower is better):**

| Yard type | Model | Base rate |
|---|---|---|
| Passing | 0.450 | 0.518 |
| Rushing | 0.377 | 0.481 |
| Receiving | 0.364 | 0.442 |

The probabilities are well calibrated: across the test set, outcomes predicted at 30% happen about 25-35% of the
time, and similarly across other probability ranges. Full metrics, including the calibration tables, are in
`models/report.json`.

## Limitations

- Weekly yardage is noisy. The gain over a rolling average is modest, and individual predictions can miss widely.
- The model does not see depth charts or weather. Injuries come from the weekly report only: players on injured
  reserve aren't listed, so their absence isn't counted, and for Sunday games the report isn't final until Friday.
- Predictions are only as current as the last run. Rerun `predict` after the final injury report.
- Vegas lines for upcoming games appear in the schedule about a week ahead. Predicting further out leaves them
  missing, which the models handle but with less accuracy.
- The very top of the range is still uncertain: there are only a few dozen test games per yard type for the
  top 2% of players, and the calibration still leaves elite receivers about 3 yards low.
- Calibration gives some weight to recent averages, so a backup with a strong average from spot duty (or a
  starter who may not play) can get a misleading projection. Check who is starting.
- `predict` uses the saved models and does not retrain. Retrain after new seasons accumulate
  (`uv run python -m nfl_yards.train --val-season <year>`).
- The two models are trained separately, so they can disagree slightly (e.g. expected yards above the 50% point
  of the probabilities, which is normal for skewed outcomes).

## Layout

```
nfl_yards/data.py       download and cache nflverse data, merge snap counts and Vegas lines
nfl_yards/features.py   leak-free rolling features and opponent strength
nfl_yards/train.py      train either or both models, write models/report.json
nfl_yards/expected.py   expected-yards model
nfl_yards/probs.py      threshold-probability model
nfl_yards/predict.py    predictions for upcoming games
```
