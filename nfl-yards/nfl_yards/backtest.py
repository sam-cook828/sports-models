"""Rolling-origin validation: re-run the full training recipe once per test season.

For each test season S: train on seasons before S-1, pick boosting rounds on S-1, refit on everything before S,
and score on S. This shows how stable the models' edge over the baselines is from season to season, rather than
relying on a single train/test split. Results go to models/backtest.json; no models are saved.
"""
import argparse
import json

import pandas as pd

from . import expected, probs
from .data import load_all
from .features import TARGETS, build, feature_cols
from .train import MODELS_DIR

KEYS = {
    "expected": ["test_rmse", "baseline_rmse", "test_mae", "baseline_mae", "mean_bias"],
    "probs": ["log_loss", "base_rate_log_loss", "brier", "base_rate_brier"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=2020, help="first test season")
    ap.add_argument("--last", type=int, default=2026, help="last test season (a partial season is fine)")
    ap.add_argument("--model", choices=["expected", "probs", "all"], default="all")
    args = ap.parse_args()

    raw = load_all()
    rows = []
    for target in TARGETS:
        df = build(raw, target)
        feats = feature_cols(df, target)
        for season in range(args.first, args.last + 1):
            tr, va, te = df[df.season < season - 1], df[df.season == season - 1], df[df.season == season]
            if te.empty:
                continue
            for name, module in (("expected", expected), ("probs", probs)):
                if args.model not in (name, "all"):
                    continue
                _, m = module.fit(tr, va, te, feats, target)
                row = {"target": target, "model": name, "season": season, "n_test": len(te), "best_iter": m["best_iter"],
                       **{k: m[k] for k in KEYS[name]}}
                rows.append(row)
                print(row, flush=True)

    res = pd.DataFrame(rows)
    out = {}
    for (target, name), g in res.groupby(["target", "model"], sort=False):
        cols = KEYS[name]
        # weight seasons by test size so a partial season counts less
        avg = {k: round(float((g[k] * g["n_test"]).sum() / g["n_test"].sum()), 4) for k in cols}
        model_col, base_col = cols[0], cols[1]
        wins = int((g[model_col] < g[base_col]).sum())
        out.setdefault(target, {})[name] = {
            "seasons": g.drop(columns=["target", "model"]).to_dict("records"),
            "weighted_avg": avg,
            "beats_baseline": f"{wins}/{len(g)} seasons",
        }
        print(f"\n{target} {name}: {avg} | beats baseline in {wins}/{len(g)} seasons")
    (MODELS_DIR / "backtest.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
