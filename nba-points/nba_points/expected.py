"""Expected-points model: predicts a player's mean points for a game (squared-error loss).

Boosted trees shrink extreme players toward the middle (stars come out too low) and can carry a
season-level bias. A linear calibration step corrects both: it is fit on out-of-sample
predictions from the five seasons before the test set, each made by a model trained only on earlier seasons,
and blends the tree prediction with the player's 10-game average.

Each prediction also gets a High / Medium / Low confidence label. A second model learns how far off the
calibrated prediction tends to be (its expected absolute miss) from those same out-of-sample errors. The miss is
taken relative to the projection, and the labels split past predictions into equal thirds: High is the third with
the smallest expected miss.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

CALIBRATION_SEASONS = 5
LABELS = np.array(["High", "Medium", "Low"])


def _make(n_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, max_iter=n_iter, random_state=0)


class CalibratedRegressor:
    """Tree model plus a linear correction: points = a * tree prediction + b * 10-game average + c.

    Also carries the confidence model: expected absolute miss, its floor and the label cut points.
    """

    def __init__(self, trees, calibrator, error_model=None, floor=1.0, cuts=(np.inf, np.inf)):
        self.trees, self.calibrator = trees, calibrator
        self.error_model, self.floor, self.cuts = error_model, floor, cuts

    def predict(self, X: pd.DataFrame, baseline: pd.Series) -> np.ndarray:
        raw = self.trees.predict(X)
        return self.calibrator.predict(np.c_[raw, baseline.to_numpy()])

    def relative_miss(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        """Expected absolute miss as a share of the projection (the projection floored so tiny ones don't blow up)."""
        miss = self.error_model.predict(X.assign(pred=pred))
        return miss / np.maximum(pred, self.floor)

    def confidence(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        return LABELS[np.searchsorted(self.cuts, self.relative_miss(X, pred), side="right")]


def _out_of_sample(trva: pd.DataFrame, feats: list[str], target: str, n_iter: int) -> pd.DataFrame:
    """Predictions for each of the last few seasons from a model trained only on the seasons before it."""
    seasons = sorted(trva["season"].unique())[-CALIBRATION_SEASONS:]
    parts = []
    for s in seasons:
        past, cur = trva[trva.season < s], trva[trva.season == s]
        parts.append(cur.assign(raw_pred=_make(n_iter).fit(past[feats], past[target]).predict(cur[feats])))
    return pd.concat(parts)


def _confidence_parts(oof: pd.DataFrame, feats: list[str], target: str):
    """Model of the calibrated prediction's absolute miss, plus the floor and cut points for the labels."""
    X = oof[feats].assign(pred=oof["pred"])
    error_model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
        max_iter=300, early_stopping=True, validation_fraction=0.2, random_state=0,
    ).fit(X, (oof[target] - oof["pred"]).abs())
    floor = float(np.percentile(oof["pred"], 25))
    rel = error_model.predict(X) / np.maximum(oof["pred"], floor)
    cuts = tuple(np.percentile(rel, [100 / 3, 200 / 3]))
    return error_model, floor, cuts


def _scores(pred, te, target) -> dict:
    y, base = te[target].to_numpy(), te["baseline"].to_numpy()
    top = base >= np.percentile(base, 98)  # the very best players by recent form
    return dict(test_rmse=round(float(np.sqrt(np.mean((pred - y) ** 2))), 2),
                test_mae=round(float(np.abs(pred - y).mean()), 2),
                mean_bias=round(float((pred - y).mean()), 2),
                top2pct_bias=round(float((pred[top] - y[top]).mean()), 2))


def fit(tr, va, te, feats, target):
    """Pick boosting rounds on `va`, fit the calibration, refit on tr+va, score on `te` against the baseline."""
    probe = _make(1500).fit(tr[feats], tr[target])
    rmse = [np.sqrt(np.mean((p - va[target]) ** 2)) for p in probe.staged_predict(va[feats])]
    best_n = int(np.argmin(rmse)) + 1

    trva = pd.concat([tr, va])
    trees = _make(best_n).fit(trva[feats], trva[target])
    oof = _out_of_sample(trva, feats, target, best_n)
    calibrator = LinearRegression().fit(oof[["raw_pred", "baseline"]].to_numpy(), oof[target])
    oof["pred"] = calibrator.predict(oof[["raw_pred", "baseline"]].to_numpy())
    model = CalibratedRegressor(trees, calibrator, *_confidence_parts(oof, feats, target))

    y, base = te[target], te["baseline"]
    metrics = dict(n_train=len(tr), n_test=len(te), best_iter=best_n,
                   calibration=dict(tree_weight=round(float(model.calibrator.coef_[0]), 3),
                                    avg_weight=round(float(model.calibrator.coef_[1]), 3),
                                    intercept=round(float(model.calibrator.intercept_), 2)),
                   baseline_rmse=round(float(np.sqrt(np.mean((base - y) ** 2))), 2),
                   baseline_mae=round(float(np.abs(base - y).mean()), 2),
                   uncalibrated=_scores(trees.predict(te[feats]), te, target),
                   **_scores(model.predict(te[feats], base), te, target))
    # does the label separate reliable predictions from shaky ones on unseen games?
    pred = model.predict(te[feats], base)
    label = model.confidence(te[feats], pred)
    err = np.abs(pred - y.to_numpy())
    metrics["confidence"] = {
        str(lab): dict(n=int((label == lab).sum()), share=round(float((label == lab).mean()), 3),
                  mae=round(float(err[label == lab].mean()), 2),
                  relative_miss=round(float((err / np.maximum(pred, model.floor))[label == lab].mean()), 3))
        for lab in LABELS if (label == lab).any()}
    metrics["rmse_improvement_pct"] = round(100 * (1 - metrics["test_rmse"] / metrics["baseline_rmse"]), 1)
    return model, metrics


def predict(model: CalibratedRegressor, df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Expected points and a High / Medium / Low confidence label per row."""
    X = df.reindex(columns=feats)
    pred = model.predict(X, df["baseline"])
    return np.clip(pred, 0, None), model.confidence(X, pred)
