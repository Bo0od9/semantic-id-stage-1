"""Aggregate SASRec runs (3 seeds × ID/Content) into results CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.paths import RESULTS_DIR, SAVED_DIR  # noqa: E402
from src.utils.aggregator import aggregate_seeds, write_results_csv  # noqa: E402


def aggregate_model(model_slug: str, display_name: str, seeds: list[int]) -> pd.DataFrame:
    run_dirs = [SAVED_DIR / f"{model_slug}_seed{s}" for s in seeds]
    existing = [d for d in run_dirs if (d / "metrics.json").exists()]
    missing = [d for d in run_dirs if not (d / "metrics.json").exists()]
    if missing:
        print(f"[warn] {display_name}: missing runs {[d.name for d in missing]}")
    return aggregate_seeds(existing, model_name=display_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also include MostPop and Random baseline rows in the combined CSV.",
    )
    args = parser.parse_args()

    frames = []

    sasrec_id = aggregate_model("sasrec_id", "SASRec-ID", args.seeds)
    if not sasrec_id.empty:
        write_results_csv(RESULTS_DIR / "sasrec_id.csv", [sasrec_id])
        frames.append(sasrec_id)

    sasrec_content = aggregate_model("sasrec_content", "SASRec-Content", args.seeds)
    if not sasrec_content.empty:
        write_results_csv(RESULTS_DIR / "sasrec_content.csv", [sasrec_content])
        frames.append(sasrec_content)

    if args.include_baselines:
        mostpop = aggregate_seeds([SAVED_DIR / "mostpop"], model_name="MostPop")
        random_rec = aggregate_seeds(
            [SAVED_DIR / f"random_seed{s}" for s in args.seeds],
            model_name="Random",
        )
        frames = [mostpop, random_rec, *frames]

    if not frames:
        print("no runs to aggregate")
        return

    combined_path = RESULTS_DIR / "sasrec_summary.csv"
    write_results_csv(combined_path, frames)
    print(f"wrote {combined_path}")

    df = pd.read_csv(combined_path)
    cols = [
        "model", "n_runs",
        "test/recall/10_mean", "test/recall/10_std",
        "test/recall/100_mean",
        "test/ndcg/10_mean", "test/ndcg/10_std",
        "test/hitrate/10_mean",
        "test/coverage/100_mean",
    ]
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False))


if __name__ == "__main__":
    main()
