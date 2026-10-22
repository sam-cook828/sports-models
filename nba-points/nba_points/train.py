"""Train the expected-points model and score it against a 10-game-average baseline.

Seasons are split by time: seasons before --val-season train, --val-season tunes the number of boosting
rounds, and later seasons are the held-out test set.
"""
import argparse
import json
from pathlib import Path

import joblib

from . import expected
from .data import load_box
from .features import TARGET, build, feature_cols

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-season", type=int, default=2024, help="tuning season (2024 = 2023-24); later seasons are the test set")
    args = ap.parse_args()

    df = build(load_box())
    print(f"{len(df):,} player-games, seasons {df.season.min()}-{df.season.max()}")
    feats = feature_cols()
    tr, va, te = df[df.season < args.val_season], df[df.season == args.val_season], df[df.season > args.val_season]
    model, metrics = expected.fit(tr, va, te, feats, TARGET)
    print({k: v for k, v in metrics.items() if k != "confidence"})
    print("confidence:", metrics["confidence"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": feats}, MODELS_DIR / "points_expected.joblib")
    (MODELS_DIR / "report.json").write_text(json.dumps({"points": {"expected_points": metrics}}, indent=2))


if __name__ == "__main__":
    main()
