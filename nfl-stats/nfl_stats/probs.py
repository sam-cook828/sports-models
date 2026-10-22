"""Probability of reaching yardage thresholds, e.g. P(receiving yards >= 60).

One classifier per yard type is trained on rows stacked across thresholds (a `threshold` feature, label
yards >= threshold). The threshold features carry a decreasing monotonic constraint, so P(>=50) can never
fall below P(>=60).

The model is trained on a fine grid of thresholds (`GRID`) so it can answer for any line, e.g. a sportsbook's
92.5 (`prob_over`, interpolating between grid points). `THRESHOLDS` are the ones shown in reports, and test
metrics are scored on those.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

THRESHOLDS = {
    "passing_yards": [125, 150, 175, 200, 225, 250, 275, 300, 325, 350],
    "rushing_yards": [25, 40, 50, 60, 70, 80, 90, 100],
    "receiving_yards": [25, 40, 50, 60, 70, 80, 90, 100],
    "receptions": [2, 3, 4, 5, 6, 7, 8],
    "anytime_tds": [1, 2],
    "passing_tds": [1, 2, 3, 4],
}
GRID = {
    "passing_yards": list(range(100, 401, 25)),
    "rushing_yards": list(range(5, 151, 5)),
    "receiving_yards": list(range(5, 151, 5)),
    "receptions": list(range(1, 13)),
    "anytime_tds": [1, 2, 3],
    "passing_tds": [1, 2, 3, 4, 5],
}


def stack(df: pd.DataFrame, feats: list[str], target: str, thresholds: list[int]) -> pd.DataFrame:
    """One copy of the feature rows per threshold, plus the threshold and its gap to recent form."""
    parts = []
    for t in thresholds:
        x = df[feats].astype("float32")
        x["threshold"] = t
        x["thr_minus_r8"] = t - df[f"{target}_r8"]
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def labels(df: pd.DataFrame, target: str, thresholds: list[int]) -> np.ndarray:
    return np.concatenate([(df[target] >= t).to_numpy() for t in thresholds]).astype(int)


def make_model(columns: list[str], n_iter: int) -> HistGradientBoostingClassifier:
    mono = {c: -1 for c in ("threshold", "thr_minus_r8")}
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
        max_iter=n_iter, monotonic_cst=[mono.get(c, 0) for c in columns], random_state=0)


def fit(tr, va, te, feats, target):
    """Pick boosting rounds on `va`, refit on tr+va, score on `te` against base-rate-by-threshold."""
    th, grid = THRESHOLDS[target], GRID[target]
    Xtr, ytr = stack(tr, feats, target, grid), labels(tr, target, grid)
    Xva, yva = stack(va, feats, target, grid), labels(va, target, grid)
    columns = list(Xtr.columns)
    probe = make_model(columns, 400).fit(Xtr, ytr)
    del Xtr
    losses = [log_loss(yva, p[:, 1]) for p in probe.staged_predict_proba(Xva)]
    best_n = int(np.argmin(losses)) + 1

    trva = pd.concat([tr, va])
    final = make_model(columns, best_n).fit(stack(trva, feats, target, grid), labels(trva, target, grid))
    Xte, yte = stack(te, feats, target, th), labels(te, target, th)
    p = final.predict_proba(Xte)[:, 1]
    base = np.concatenate([np.full(len(te), (trva[target] >= t).mean()) for t in th])
    metrics = dict(best_iter=best_n, log_loss=round(log_loss(yte, p), 4), base_rate_log_loss=round(log_loss(yte, base), 4),
                   brier=round(brier_score_loss(yte, p), 4), base_rate_brier=round(brier_score_loss(yte, base), 4))
    # calibration: mean predicted vs observed hit rate in probability deciles
    cal = pd.DataFrame({"p": p, "y": yte})
    cal["bin"] = pd.cut(cal["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    metrics["calibration"] = {str(k): [round(v.p.mean(), 3), round(v.y.mean(), 3), len(v)]
                              for k, v in cal.groupby("bin", observed=True) if len(v)}
    return final, metrics


def predict(model, df: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    th = THRESHOLDS[target]
    p = model.predict_proba(stack(df, feats, target, th)[list(model.feature_names_in_)])[:, 1]
    return pd.DataFrame(p.reshape(len(th), len(df)).T, index=df.index, columns=[f"p_{t}" for t in th])


def prob_over(model, df: pd.DataFrame, feats: list[str], target: str, lines: pd.Series) -> np.ndarray:
    """P(yards > line) for each row's own line, interpolated between grid thresholds.

    Yards are whole numbers, so over 92.5 means at least 93.
    """
    grid = GRID[target]
    p = model.predict_proba(stack(df, feats, target, grid)[list(model.feature_names_in_)])[:, 1]
    p = p.reshape(len(grid), len(df)).T
    need = np.floor(lines.to_numpy(dtype=float)) + 1
    return np.array([np.interp(n, grid, row) if np.isfinite(n) else np.nan for n, row in zip(need, p)])
