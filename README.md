# sports-models

Player prop models for NFL and NBA, built on free public data. Each project is self-contained, with its own
environment, data cache and models.

| Project | Predicts | Data |
|---|---|---|
| [`nfl-stats/`](nfl-stats/) | Passing, rushing and receiving yards; receptions; anytime and passing touchdowns. Expected values, confidence labels and threshold probabilities | nflverse |
| [`nba-stats/`](nba-stats/) | Points, assists and rebounds per game, with confidence labels | ESPN via sportsdataverse |

```
cd nfl-stats   # or nba-stats
uv sync
uv run python -m nfl_stats.train     # nba_stats.train
uv run python -m nfl_stats.predict   # nba_stats.predict
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).
