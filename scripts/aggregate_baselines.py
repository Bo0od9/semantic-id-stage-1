"""Aggregate MostPop + Random baseline runs into results/baselines.csv."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import RESULTS_DIR, SAVED_DIR  # noqa: E402
from src.utils.aggregator import aggregate_seeds, write_results_csv  # noqa: E402


def main() -> None:
    mostpop = aggregate_seeds([SAVED_DIR / "mostpop"], model_name="MostPop")
    random = aggregate_seeds(
        [SAVED_DIR / f"random_seed{s}" for s in (42, 43, 44)],
        model_name="Random",
    )

    out = RESULTS_DIR / "baselines.csv"
    write_results_csv(out, [mostpop, random])
    print(f"wrote {out}")
    print()

    import pandas as pd  # noqa: PLC0415
    df = pd.read_csv(out)
    cols_show = [
        "model", "n_runs",
        "test/recall/10_mean", "test/recall/10_std",
        "test/recall/100_mean", "test/recall/100_std",
        "test/ndcg/10_mean", "test/ndcg/10_std",
        "test/hitrate/100_mean",
        "test/coverage/100_mean",
    ]
    present = [c for c in cols_show if c in df.columns]
    print(df[present].to_string(index=False))


if __name__ == "__main__":
    main()
