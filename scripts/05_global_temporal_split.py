"""Stage 5 of the data preparation pipeline: global temporal split.

Reads ``data/interim/listens_stage3.parquet`` and the cutoffs frozen on
stage 4 (``splits_metadata.json::temporal_cutoffs``), then materialises three
partitions: ``train``, ``val``, ``test``. Cold-start users and items (i.e.
users/items that appear in val/test but never in train) are dropped from the
side splits — cold-start is out-of-scope for the core experiment.

Outputs:

* ``data/processed/{train,val,test}.parquet``  (zstd)
* ``reports/07_temporal_split.md``
* ``data/filter_stats.json``   — merged ``stage5`` section (siblings preserved).
* ``data/splits_metadata.json`` — merged ``stage5_frozen_at_utc`` key.

**Read-only w.r.t. cutoffs.** If ``splits_metadata.json`` is missing, does not
contain ``temporal_cutoffs``, or any of the required sub-keys are missing, the
script fails fast. No fallback defaults.

Timestamps are seconds from the Yambda collection anchor (5s quantised) — not
a unix epoch. Same convention as stage 4.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    FILTER_STATS_PATH,
    LISTENS_STAGE3_PATH,
    PROCESSED_DIR,
    SPLITS_METADATA_PATH,
    TEMPORAL_SPLIT_REPORT,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
)
from utils.io import load_json, setup_logging, update_json_section  # noqa: E402
from utils.parquet import (  # noqa: E402
    atomic_write_parquet,
    counts,
    validate_listens_schema,
)

logger = logging.getLogger(__name__)

CUTOFF_MIN_INTERACTIONS: int = 10_000
SECONDS_PER_DAY: int = 86_400


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_cutoffs(meta_path: Path) -> tuple[int, int]:
    """Read ``T_val_seconds`` / ``T_test_seconds`` from splits metadata.

    Fails fast if the file, the ``temporal_cutoffs`` section, or either of
    the two keys is missing. No defaults.
    """
    if not meta_path.exists():
        raise RuntimeError(
            f"splits metadata not found at {meta_path}; stage 4 must run first"
        )
    payload = load_json(meta_path)
    if "temporal_cutoffs" not in payload:
        raise RuntimeError(
            f"{meta_path} has no 'temporal_cutoffs' section; stage 4 must run first"
        )
    section = payload["temporal_cutoffs"]
    for key in ("T_val_seconds", "T_test_seconds"):
        if key not in section:
            raise RuntimeError(
                f"{meta_path}::temporal_cutoffs missing required key {key!r}"
            )
    t_val = int(section["T_val_seconds"])
    t_test = int(section["T_test_seconds"])
    if not (t_val < t_test):
        raise RuntimeError(
            f"temporal_cutoffs invalid: T_val_seconds={t_val} >= T_test_seconds={t_test}"
        )
    logger.info(
        "loaded cutoffs T_val=%d (day %.3f) T_test=%d (day %.3f)",
        t_val,
        t_val / SECONDS_PER_DAY,
        t_test,
        t_test / SECONDS_PER_DAY,
    )
    return t_val, t_test


def read_stage3(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise RuntimeError(f"stage-3 parquet not found at {path}")
    df = pl.read_parquet(path)
    validate_listens_schema(df, origin=f"stage-3 parquet ({path})")
    logger.info("read stage-3 parquet: %d rows", df.height)
    return df


# ---------------------------------------------------------------------------
# Split + post-filter
# ---------------------------------------------------------------------------


def raw_split(
    df: pl.DataFrame, t_val: int, t_test: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("timestamp") < t_val)
    val = df.filter((pl.col("timestamp") >= t_val) & (pl.col("timestamp") < t_test))
    test = df.filter(pl.col("timestamp") >= t_test)
    logger.info(
        "raw split: train=%d, val=%d, test=%d", train.height, val.height, test.height
    )
    if train.height + val.height + test.height != df.height:
        raise RuntimeError(
            "raw-split row totals do not add up — timestamp-based partition bug"
        )
    return train, val, test


def _drop_cold(
    side: pl.DataFrame,
    train_users: pl.DataFrame,
    train_items: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Semi-join on both uid and item_id to drop cold-start rows.

    Reported ``users_dropped`` / ``items_dropped`` are the *actual* losses
    in distinct identities (raw minus final) — this accounts for cascade
    losses where a user/item exists in train yet loses all their side-split
    rows because every partner identity was cold.
    """
    before_rows = side.height
    before_users = int(side.get_column("uid").n_unique())
    before_items = int(side.get_column("item_id").n_unique())

    filtered = side.join(train_users, on="uid", how="semi").join(
        train_items, on="item_id", how="semi"
    )

    after_users = int(filtered.get_column("uid").n_unique()) if filtered.height else 0
    after_items = int(filtered.get_column("item_id").n_unique()) if filtered.height else 0

    breakdown = {
        "users_dropped": before_users - after_users,
        "items_dropped": before_items - after_items,
        "rows_dropped": before_rows - filtered.height,
    }
    return filtered, breakdown


def apply_cold_filter(
    train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, int]]]:
    train_users = train.select("uid").unique()
    train_items = train.select("item_id").unique()
    val_f, val_break = _drop_cold(val, train_users, train_items)
    test_f, test_break = _drop_cold(test, train_users, train_items)
    logger.info(
        "cold-start filter: val %d -> %d (users=%d items=%d)",
        val.height,
        val_f.height,
        val_break["users_dropped"],
        val_break["items_dropped"],
    )
    logger.info(
        "cold-start filter: test %d -> %d (users=%d items=%d)",
        test.height,
        test_f.height,
        test_break["users_dropped"],
        test_break["items_dropped"],
    )
    return val_f, test_f, {"val": val_break, "test": test_break}


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def assert_invariants(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    t_val: int,
    t_test: int,
) -> None:
    if train.height == 0:
        raise RuntimeError("train split is empty")
    if val.height < CUTOFF_MIN_INTERACTIONS:
        raise RuntimeError(
            f"val has {val.height} interactions (< {CUTOFF_MIN_INTERACTIONS})"
        )
    if test.height < CUTOFF_MIN_INTERACTIONS:
        raise RuntimeError(
            f"test has {test.height} interactions (< {CUTOFF_MIN_INTERACTIONS})"
        )

    train_max = int(train.get_column("timestamp").max())
    val_min = int(val.get_column("timestamp").min())
    val_max = int(val.get_column("timestamp").max())
    test_min = int(test.get_column("timestamp").min())

    if not (train_max < t_val <= val_min):
        raise RuntimeError(
            f"timestamp disjointness broken: train_max={train_max} t_val={t_val} val_min={val_min}"
        )
    if not (val_max < t_test <= test_min):
        raise RuntimeError(
            f"timestamp disjointness broken: val_max={val_max} t_test={t_test} test_min={test_min}"
        )

    train_users = set(train.get_column("uid").unique().to_list())
    train_items = set(train.get_column("item_id").unique().to_list())
    val_users = set(val.get_column("uid").unique().to_list())
    val_items = set(val.get_column("item_id").unique().to_list())
    test_users = set(test.get_column("uid").unique().to_list())
    test_items = set(test.get_column("item_id").unique().to_list())

    if not val_users.issubset(train_users):
        raise RuntimeError("val users not subset of train users after cold filter")
    if not test_users.issubset(train_users):
        raise RuntimeError("test users not subset of train users after cold filter")
    if not val_items.issubset(train_items):
        raise RuntimeError("val items not subset of train items after cold filter")
    if not test_items.issubset(train_items):
        raise RuntimeError("test items not subset of train items after cold filter")

    key_cols = ["uid", "item_id", "timestamp"]
    train_keys = train.select(key_cols)
    val_keys = val.select(key_cols)
    test_keys = test.select(key_cols)
    if train_keys.join(val_keys, on=key_cols, how="semi").height > 0:
        raise RuntimeError("row leakage: train ∩ val is non-empty on (uid,item,ts)")
    if train_keys.join(test_keys, on=key_cols, how="semi").height > 0:
        raise RuntimeError("row leakage: train ∩ test is non-empty on (uid,item,ts)")
    if val_keys.join(test_keys, on=key_cols, how="semi").height > 0:
        raise RuntimeError("row leakage: val ∩ test is non-empty on (uid,item,ts)")

    logger.info("all invariants passed")


# ---------------------------------------------------------------------------
# Write + report
# ---------------------------------------------------------------------------


def write_parquets(
    train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for df, path in (
        (train, TRAIN_PARQUET_PATH),
        (val, VAL_PARQUET_PATH),
        (test, TEST_PARQUET_PATH),
    ):
        atomic_write_parquet(df, path)
        logger.info("wrote %s (%d rows)", path, df.height)


def build_stage5_stats(
    pre: dict[str, int],
    cutoffs: tuple[int, int],
    raw: dict[str, dict[str, int]],
    cold: dict[str, dict[str, int]],
    final: dict[str, dict[str, int]],
) -> dict[str, Any]:
    t_val, t_test = cutoffs
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "description": (
            "Global temporal split with T_val / T_test from stage 4; "
            "cold-start users and items removed from val and test."
        ),
        "input_parquet": str(LISTENS_STAGE3_PATH),
        "cutoffs": {
            "T_val_seconds": t_val,
            "T_test_seconds": t_test,
            "T_val_days_from_start": t_val / SECONDS_PER_DAY,
            "T_test_days_from_start": t_test / SECONDS_PER_DAY,
            "source": "splits_metadata.json::temporal_cutoffs",
        },
        "pre": pre,
        "raw_split": raw,
        "cold_start_excluded": cold,
        "final": final,
        "invariants_checked": [
            "row_total_equals_pre",
            "timestamp_disjoint_train_val",
            "timestamp_disjoint_val_test",
            "val_users_subset_train",
            "test_users_subset_train",
            "val_items_subset_train",
            "test_items_subset_train",
            "no_row_leakage_train_val",
            "no_row_leakage_train_test",
            "no_row_leakage_val_test",
            "min_interactions_val",
            "min_interactions_test",
            "train_non_empty",
        ],
        "output": {
            "train_parquet": str(TRAIN_PARQUET_PATH),
            "val_parquet": str(VAL_PARQUET_PATH),
            "test_parquet": str(TEST_PARQUET_PATH),
        },
    }


def render_report(stats: dict[str, Any]) -> str:
    pre = stats["pre"]
    raw = stats["raw_split"]
    cold = stats["cold_start_excluded"]
    final = stats["final"]
    cutoffs = stats["cutoffs"]
    ts = stats["generated_at_utc"].replace("+00:00", " UTC")

    def _row(name: str, d: dict[str, int]) -> str:
        return (
            f"| {name} | {d['num_interactions']:,} | "
            f"{d['num_users']:,} | {d['num_items']:,} |"
        )

    lines = [
        "# Temporal Split (Stage 5)",
        "",
        f"_Generated: {ts}_",
        "",
        "## What this stage does",
        "",
        "Reads the stage-3 parquet and the cutoffs frozen on stage 4, materialises"
        " three partitions (`train`, `val`, `test`) and drops cold-start users and"
        " items from `val` / `test` (users/items that have no presence in `train`)."
        " The cutoffs themselves are **not** recomputed here — see"
        " [`reports/06_temporal_analysis.md`](06_temporal_analysis.md) for the"
        " rationale behind the 14+14-day hold-out.",
        "",
        "## Cutoffs (from `data/splits_metadata.json::temporal_cutoffs`)",
        "",
        f"- `T_val_seconds`  = **{cutoffs['T_val_seconds']:,}**"
        f" (day {cutoffs['T_val_days_from_start']:.3f} from dataset start)",
        f"- `T_test_seconds` = **{cutoffs['T_test_seconds']:,}**"
        f" (day {cutoffs['T_test_days_from_start']:.3f} from dataset start)",
        "",
        "## Pre (= stage-3 output)",
        "",
        "| Interactions | Users | Items |",
        "| ---: | ---: | ---: |",
        f"| {pre['num_interactions']:,} | {pre['num_users']:,} | {pre['num_items']:,} |",
        "",
        "## Raw split (before cold-start filter)",
        "",
        "| Split | Interactions | Users | Items |",
        "| :--- | ---: | ---: | ---: |",
        _row("train", raw["train"]),
        _row("val",   raw["val"]),
        _row("test",  raw["test"]),
        "",
        "## Cold-start drops",
        "",
        "| Split | Users dropped | Items dropped | Rows dropped |",
        "| :--- | ---: | ---: | ---: |",
        f"| val  | {cold['val']['users_dropped']:,}"
        f" | {cold['val']['items_dropped']:,}"
        f" | {cold['val']['rows_dropped']:,} |",
        f"| test | {cold['test']['users_dropped']:,}"
        f" | {cold['test']['items_dropped']:,}"
        f" | {cold['test']['rows_dropped']:,} |",
        "",
        "Train is authoritative: no rows are dropped from train at this stage."
        " Min-count filters are *not* re-applied on val/test — those thresholds"
        " belong to stage 3 and re-applying them on a hold-out would leak eval"
        " structure into training decisions.",
        "",
        "## Final splits",
        "",
        "| Split | Interactions | Users | Items |",
        "| :--- | ---: | ---: | ---: |",
        _row("train", final["train"]),
        _row("val",   final["val"]),
        _row("test",  final["test"]),
        "",
        "## Invariant checks",
        "",
        "All of the following are asserted at the end of the stage — the script"
        " fails fast otherwise:",
        "",
    ]
    for inv in stats["invariants_checked"]:
        lines.append(f"- ✓ `{inv}`")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{TRAIN_PARQUET_PATH.relative_to(TRAIN_PARQUET_PATH.parents[2])}` (zstd)",
        f"- `{VAL_PARQUET_PATH.relative_to(VAL_PARQUET_PATH.parents[2])}` (zstd)",
        f"- `{TEST_PARQUET_PATH.relative_to(TEST_PARQUET_PATH.parents[2])}` (zstd)",
        "- `data/filter_stats.json` (section `stage5`)",
        "- `data/splits_metadata.json` (key `stage5_frozen_at_utc`)",
        "",
        "## Next stage",
        "",
        "Stage 6 will carve out user-level subsamples (`train_subsample_10pct`,"
        " `train_subsample_1pct` and their matching val/test slices) from these"
        " three parquets for pilot and smoke-test runs.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage3-parquet", type=Path, default=LISTENS_STAGE3_PATH)
    p.add_argument("--splits-metadata", type=Path, default=SPLITS_METADATA_PATH)
    p.add_argument("--filter-stats", type=Path, default=FILTER_STATS_PATH)
    p.add_argument("--report", type=Path, default=TEMPORAL_SPLIT_REPORT)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    t_val, t_test = load_cutoffs(args.splits_metadata)
    df = read_stage3(args.stage3_parquet)
    pre = counts(df)
    logger.info("pre counts: %s", pre)

    train, val_raw, test_raw = raw_split(df, t_val, t_test)
    raw = {
        "train": counts(train),
        "val": counts(val_raw),
        "test": counts(test_raw),
    }

    val, test, cold = apply_cold_filter(train, val_raw, test_raw)

    assert_invariants(train, val, test, t_val, t_test)

    final = {"train": counts(train), "val": counts(val), "test": counts(test)}
    logger.info("final counts: %s", final)

    write_parquets(train, val, test)

    stats = build_stage5_stats(pre, (t_val, t_test), raw, cold, final)
    update_json_section(args.filter_stats, "stage5", stats)

    update_json_section(
        args.splits_metadata,
        "stage5_frozen_at_utc",
        datetime.now(tz=timezone.utc).isoformat(),
    )

    report_md = render_report(stats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md, encoding="utf-8")
    logger.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
