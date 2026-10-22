"""Train the expected points, assists and rebounds models; score each against its 10-game-average baseline.

Seasons are split by time: seasons before --val-season train, --val-season tunes the number of boosting
rounds, and later seasons are the held-out test set.
"""
import argparse
import json
from pathlib import Path

import joblib

from . import expected
from .data import load_box
from .features import TARGETS, build, feature_cols

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-season", type=int, default=2024, help="tuning season (2024 = 2023-24); later seasons are the test set")
    ap.add_argument("--target", choices=TARGETS, help="only this stat (default: all)")
    args = ap.parse_args()

    df = build(load_box())
    print(f"{len(df):,} player-games, seasons {df.season.min()}-{df.season.max()}")
    MODELS_DIR.mkdir(exist_ok=True)
    report_path = MODELS_DIR / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    for target in [args.target] if args.target else TARGETS:
        d = df.assign(baseline=df[f"{target}_r10"])
        feats = feature_cols(target)
        tr, va, te = d[d.season < args.val_season], d[d.season == args.val_season], d[d.season > args.val_season]
        model, metrics = expected.fit(tr, va, te, feats, target)
        print(target, {k: v for k, v in metrics.items() if k not in ("confidence", "calibration")}, flush=True)
        joblib.dump({"model": model, "features": feats}, MODELS_DIR / f"{target}_expected.joblib")
        report[target] = {"expected": metrics}

    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
