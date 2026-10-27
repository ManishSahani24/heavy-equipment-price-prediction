"""Train the full ensemble and write a Kaggle submission.

Examples
--------
Full run (reproduces the competition submission, ~several hours on CPU):
    python scripts/train.py --data-dir data/raw

Fast smoke test (fewer trees, two models):
    python scripts/train.py --data-dir data/synthetic --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heavy_equipment.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, TrainConfig  # noqa: E402
from heavy_equipment.pipeline import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder with train.csv and test.csv")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=["lgb1", "lgb2", "xgb", "cat"],
                   choices=["lgb1", "lgb2", "xgb", "cat"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--estimator-scale", type=float, default=1.0,
                   help="multiply the number of boosting rounds (e.g. 0.1 for a quick run)")
    p.add_argument("--quick", action="store_true",
                   help="smoke-test mode: 3 folds, 2%% of boosting rounds, LightGBM v1 + XGBoost only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        models=tuple(args.models),
        estimator_scale=args.estimator_scale,
    )
    if args.quick:
        cfg.n_folds = 3
        cfg.estimator_scale = 0.02
        cfg.early_stopping_rounds = 50
        cfg.models = ("lgb1", "xgb")
    run(cfg)


if __name__ == "__main__":
    main()
