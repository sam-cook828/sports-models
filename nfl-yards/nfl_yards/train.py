"""Train the two models for each yard type: expected yards and threshold probabilities.

Each is independent: it has its own module, saved file (models/<target>_expected.joblib and
models/<target>_probs.joblib) and section in models/report.json.
"""
import argparse
import json
from pathlib import Path

import joblib

from . import expected, probs
from .data import load_all
from .features import TARGETS, build, feature_cols

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-season", type=int, default=2024, help="tuning season; later seasons are the test set")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--model", choices=["expected", "probs", "all"], default="all", help="which model to train")
    ap.add_argument("--target", choices=list(TARGETS), help="only this yard type (default: all)")
    args = ap.parse_args()

    raw = load_all(refresh=args.refresh)
    print(f"loaded {len(raw):,} player-weeks, seasons {raw.season.min()}-{raw.season.max()}")
    MODELS_DIR.mkdir(exist_ok=True)
    report_path = MODELS_DIR / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    for target in [args.target] if args.target else TARGETS:
        df = build(raw, target)
        feats = feature_cols(df, target)
        tr, va, te = df[df.season < args.val_season], df[df.season == args.val_season], df[df.season > args.val_season]
        section = report.setdefault(target, {})
        for name, module in (("expected", expected), ("probs", probs)):
            if args.model not in (name, "all"):
                continue
            model, metrics = module.fit(tr, va, te, feats, target)
            joblib.dump({"model": model, "features": feats}, MODELS_DIR / f"{target}_{name}.joblib")
            section["expected_yards" if name == "expected" else "probabilities"] = metrics
            print(target, name, {k: v for k, v in metrics.items() if k != "calibration"})

    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
