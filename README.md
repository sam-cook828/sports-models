# sports-models

Player prop models for NFL and NBA, built on free public data. Each project is self-contained, with its own
environment, data cache and models.

| Project | Predicts | Data |
|---|---|---|
| [`nfl-yards/`](nfl-yards/) | Weekly passing, rushing and receiving yards, with threshold probabilities | nflverse |
| [`nba-points/`](nba-points/) | Points per game | ESPN via sportsdataverse |

```
cd nfl-yards   # or nba-points
uv sync
uv run python -m nfl_yards.train     # nba_points.train
uv run python -m nfl_yards.predict   # nba_points.predict
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).
