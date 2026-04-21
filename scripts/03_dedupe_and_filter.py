"""Stage 3 of the data preparation pipeline: dedup + played_ratio + min-count.

Reads ``data/interim/listens_stage2.parquet`` and produces
``data/interim/listens_stage3.parquet`` by applying, in fixed order:

1. **Exact dedup** — drop fully identical rows (all 6 columns equal). These
   come from two originally distinct events that fell into the same 5s bucket
   after the Yambda timestamp quantisation (Sec 3.3).
2. **Key-collision dedup** — within each ``(uid, timestamp, item_id)`` group,
   keep only the row with the maximum ``played_ratio_pct``. These groups are
   progress-snapshots of a single listen that got logged multiple times.
3. **played_ratio filter** — keep rows with ``played_ratio_pct ≥ 50``
   (matches Yambda paper's ``Listen_s`` metric, Sec 4.3).
4. **Iterative min-count filtering** — drop users with < 5 interactions and
   items with < 5 interactions, repeating until the set is stable.

Dedup must precede the played_ratio filter: progress-rows ``(played=0)`` and
``(played=53)`` for the same listen would otherwise both survive/die
incorrectly. Dedup must precede min-count filtering for the same reason —
double-counted interactions inflate the counters.

Outputs:

* ``data/interim/listens_stage3.parquet`` — filtered listens (zstd).
* ``data/filter_stats.json`` — JSON, merges a ``stage3`` section.
* ``data/splits_metadata.json`` — freezes the filter thresholds and seed.
* ``reports/03_dedup_and_filter.md`` — standalone markdown report.

See ``docs/instructions/dataset_prep.md`` (stage 3) for the spec.
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
    DEDUP_AND_FILTER_REPORT,
    FILTER_STATS_PATH,
    LISTENS_STAGE2_PATH,
    LISTENS_STAGE3_PATH,
    MIN_ITEM_INTERACTIONS,
    MIN_USER_INTERACTIONS,
    PLAYED_RATIO_MIN,
    RANDOM_SEED,
    SPLITS_METADATA_PATH,
)
from utils.io import (  # noqa: E402
    dump_json,
    load_json,
    setup_logging,
    update_json_section,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS: tuple[str, ...] = ("uid", "timestamp", "item_id")
MAX_MIN_COUNT_ITER: int = 20


def basic_counts(df: pl.DataFrame) -> dict[str, int]:
    return {
        "num_interactions": df.height,
        "num_unique_users": int(df["uid"].n_unique()),
        "num_unique_items": int(df["item_id"].n_unique()),
    }


def dedup_exact(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    before = df.height
    out = df.unique()
    removed = before - out.height
    logger.info("dedup_exact: removed %d full-duplicate rows", removed)
    return out, removed


def count_key_collision_groups(df: pl.DataFrame) -> int:
    grouped = df.group_by(list(KEY_COLUMNS)).agg(pl.len().alias("n"))
    return int(grouped.filter(pl.col("n") > 1).height)


def dedup_key_collisions(df: pl.DataFrame) -> tuple[pl.DataFrame, int, int]:
    num_groups = count_key_collision_groups(df)
    before = df.height
    out = (
        df.sort("played_ratio_pct")
        .unique(subset=list(KEY_COLUMNS), keep="last")
    )
    removed = before - out.height
    logger.info(
        "dedup_key_collisions: removed %d rows across %d colliding groups",
        removed,
        num_groups,
    )
    return out, removed, num_groups


def assert_key_unique(df: pl.DataFrame) -> None:
    grouped = (
        df.group_by(list(KEY_COLUMNS))
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if grouped.height != 0:
        raise RuntimeError(
            f"invariant violated: {grouped.height} (uid,timestamp,item_id) "
            "keys still have multiple rows after stage 3.2 dedup"
        )


def filter_played_ratio(
    df: pl.DataFrame, threshold: int
) -> tuple[pl.DataFrame, int]:
    before = df.height
    out = df.filter(pl.col("played_ratio_pct") >= threshold)
    removed = before - out.height
    logger.info(
        "played_ratio filter (>= %d): kept %d of %d (%.2f%%)",
        threshold,
        out.height,
        before,
        100.0 * out.height / before if before else 0.0,
    )
    return out, removed


def iterative_min_count_filter(
    df: pl.DataFrame,
    min_user: int,
    min_item: int,
    max_iter: int = MAX_MIN_COUNT_ITER,
) -> tuple[pl.DataFrame, list[dict[str, int]]]:
    snapshots: list[dict[str, int]] = []
    current = df
    for i in range(1, max_iter + 1):
        before_rows = current.height
        user_counts = current.group_by("uid").agg(pl.len().alias("n_u"))
        item_counts = current.group_by("item_id").agg(pl.len().alias("n_i"))
        good_users = user_counts.filter(pl.col("n_u") >= min_user).select("uid")
        good_items = item_counts.filter(pl.col("n_i") >= min_item).select(
            "item_id"
        )
        current = current.join(good_users, on="uid", how="inner").join(
            good_items, on="item_id", how="inner"
        )
        snap = {"iter": i, **basic_counts(current)}
        snapshots.append(snap)
        logger.info(
            "min_count iter %d: %d interactions, %d users, %d items",
            i,
            snap["num_interactions"],
            snap["num_unique_users"],
            snap["num_unique_items"],
        )
        if current.height == before_rows:
            break
    else:
        raise RuntimeError(
            f"min-count filter did not stabilise in {max_iter} iterations"
        )
    return current, snapshots


def build_stage3_stats(
    pre_counts: dict[str, int],
    after_dedup_exact: dict[str, int],
    num_exact_removed: int,
    after_dedup_keys: dict[str, int],
    num_keys_removed: int,
    num_collision_groups: int,
    after_played_ratio: dict[str, int],
    num_played_ratio_removed: int,
    snapshots: list[dict[str, int]],
    final_counts: dict[str, int],
    out_parquet: Path,
) -> dict[str, Any]:
    return {
        "description": (
            "Dedup (exact + key-collision), played_ratio ≥ 50, iterative "
            "min-count filter (min 5 per user, min 5 per item)."
        ),
        "input_parquet": str(
            LISTENS_STAGE2_PATH.relative_to(LISTENS_STAGE2_PATH.parents[2])
        ),
        "output_parquet": str(out_parquet.relative_to(out_parquet.parents[2])),
        "thresholds": {
            "played_ratio_min": PLAYED_RATIO_MIN,
            "min_user_interactions": MIN_USER_INTERACTIONS,
            "min_item_interactions": MIN_ITEM_INTERACTIONS,
        },
        "pre": pre_counts,
        "dedup_exact": {
            "rows_removed": num_exact_removed,
            "after": after_dedup_exact,
        },
        "dedup_key_collisions": {
            "rows_removed": num_keys_removed,
            "groups_affected": num_collision_groups,
            "after": after_dedup_keys,
        },
        "played_ratio_filter": {
            "threshold": PLAYED_RATIO_MIN,
            "rows_removed": num_played_ratio_removed,
            "after": after_played_ratio,
        },
        "min_count_iterations": snapshots,
        "final": final_counts,
    }


def render_report(stage3: dict[str, Any]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pre = stage3["pre"]
    de = stage3["dedup_exact"]
    dk = stage3["dedup_key_collisions"]
    pr = stage3["played_ratio_filter"]
    mc = stage3["min_count_iterations"]
    final = stage3["final"]
    th = stage3["thresholds"]

    lines: list[str] = [
        "# Dedup & Filter (Stage 3)",
        "",
        f"_Generated: {generated_at}_",
        "",
        "## What this stage does",
        "",
        "Reads the stage-2 parquet and applies, in fixed order:",
        "",
        "1. **Exact dedup** — drop rows that are identical in all six "
        "columns. These arise because Yambda quantises timestamps to a 5s "
        "grid (paper Sec 3.3), so two originally distinct events can collide.",
        "2. **Key-collision dedup** — within each `(uid, timestamp, item_id)` "
        "group, keep the row with the maximum `played_ratio_pct`. These are "
        "progress snapshots of one listen logged at multiple completion "
        "levels and collapsed by the 5s quantisation.",
        f"3. **played_ratio filter** — keep rows with "
        f"`played_ratio_pct ≥ {th['played_ratio_min']}`. Matches the "
        "`Listen_s` metric in the Yambda paper (Sec 4.3), so our numbers "
        "stay comparable to Table 6.",
        f"4. **Iterative min-count filter** — drop users with "
        f"< {th['min_user_interactions']} interactions and items with "
        f"< {th['min_item_interactions']} interactions, repeating until "
        "stable.",
        "",
        "Order matters: dedup must precede both the played_ratio filter "
        "and min-count filtering. Otherwise progress-rows of a single "
        "listen inflate interaction counters and, e.g., a "
        "`(played=0, played=53)` pair could drop the correct row.",
        "",
        "## Pre-stage counts (= stage 2 output)",
        "",
        "| Interactions | Unique users | Unique items |",
        "| ---: | ---: | ---: |",
        f"| {pre['num_interactions']:,} | {pre['num_unique_users']:,} | "
        f"{pre['num_unique_items']:,} |",
        "",
        "## 3.1 Exact dedup",
        "",
        f"- Removed rows: **{de['rows_removed']:,}**",
        f"- After: {de['after']['num_interactions']:,} interactions, "
        f"{de['after']['num_unique_users']:,} users, "
        f"{de['after']['num_unique_items']:,} items",
        "",
        "## 3.2 Key-collision dedup",
        "",
        f"- Colliding `(uid, timestamp, item_id)` groups: "
        f"**{dk['groups_affected']:,}**",
        f"- Removed rows: **{dk['rows_removed']:,}**",
        f"- After: {dk['after']['num_interactions']:,} interactions, "
        f"{dk['after']['num_unique_users']:,} users, "
        f"{dk['after']['num_unique_items']:,} items",
        "- Invariant checked: `(uid, timestamp, item_id)` is unique.",
        "",
        f"## 3.3 played_ratio ≥ {th['played_ratio_min']}",
        "",
        f"- Removed rows: **{pr['rows_removed']:,}**",
        f"- After: {pr['after']['num_interactions']:,} interactions, "
        f"{pr['after']['num_unique_users']:,} users, "
        f"{pr['after']['num_unique_items']:,} items",
        "",
        "## 3.4 Iterative min-count filter",
        "",
        "| Iter | Interactions | Users | Items |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for snap in mc:
        lines.append(
            f"| {snap['iter']} | {snap['num_interactions']:,} | "
            f"{snap['num_unique_users']:,} | {snap['num_unique_items']:,} |"
        )
    lines += [
        "",
        f"Converged in **{len(mc)} iteration(s)**.",
        "",
        "## Final",
        "",
        "| Interactions | Unique users | Unique items |",
        "| ---: | ---: | ---: |",
        f"| {final['num_interactions']:,} | {final['num_unique_users']:,} | "
        f"{final['num_unique_items']:,} |",
        "",
        "## Artifacts",
        "",
        f"- `{stage3['output_parquet']}` — filtered listens (zstd).",
        "- `data/filter_stats.json` (section `stage3`) — full machine-readable "
        "dump.",
        "- `data/splits_metadata.json` — frozen thresholds and seed for later "
        "stages.",
        "",
        "## Next stage",
        "",
        "Stage 4 will compute descriptive statistics (user history length, "
        "item popularity, temporal distribution) on top of this parquet and "
        "decide subgroup boundaries.",
        "",
    ]
    return "\n".join(lines)


def write_splits_metadata(path: Path) -> None:
    payload: dict[str, Any] = load_json(path) if path.exists() else {}
    payload["filters"] = {
        "played_ratio_min": PLAYED_RATIO_MIN,
        "min_user_interactions": MIN_USER_INTERACTIONS,
        "min_item_interactions": MIN_ITEM_INTERACTIONS,
    }
    payload["random_seed"] = RANDOM_SEED
    payload["stage3_frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-parquet", type=Path, default=LISTENS_STAGE2_PATH)
    parser.add_argument("--out-parquet", type=Path, default=LISTENS_STAGE3_PATH)
    parser.add_argument("--stats-out", type=Path, default=FILTER_STATS_PATH)
    parser.add_argument("--report-out", type=Path, default=DEDUP_AND_FILTER_REPORT)
    parser.add_argument(
        "--splits-metadata-out", type=Path, default=SPLITS_METADATA_PATH
    )
    args = parser.parse_args(argv)

    setup_logging()

    if not args.in_parquet.exists():
        raise SystemExit(
            f"stage 2 parquet not found at {args.in_parquet}; run stage 2 first."
        )

    logger.info("reading stage 2 parquet: %s", args.in_parquet)
    df = pl.read_parquet(args.in_parquet)
    pre_counts = basic_counts(df)
    logger.info(
        "pre: %d interactions, %d users, %d items",
        pre_counts["num_interactions"],
        pre_counts["num_unique_users"],
        pre_counts["num_unique_items"],
    )

    df, num_exact_removed = dedup_exact(df)
    after_dedup_exact = basic_counts(df)

    df, num_keys_removed, num_collision_groups = dedup_key_collisions(df)
    after_dedup_keys = basic_counts(df)
    assert_key_unique(df)

    df, num_played_ratio_removed = filter_played_ratio(df, PLAYED_RATIO_MIN)
    after_played_ratio = basic_counts(df)
    if df.height == 0:
        raise RuntimeError("played_ratio filter dropped all rows")

    df, snapshots = iterative_min_count_filter(
        df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS
    )
    final_counts = basic_counts(df)
    if df.height == 0:
        raise RuntimeError("min-count filter dropped all rows")

    logger.info("writing stage 3 parquet: %s", args.out_parquet)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out_parquet, compression="zstd")

    stage3 = build_stage3_stats(
        pre_counts=pre_counts,
        after_dedup_exact=after_dedup_exact,
        num_exact_removed=num_exact_removed,
        after_dedup_keys=after_dedup_keys,
        num_keys_removed=num_keys_removed,
        num_collision_groups=num_collision_groups,
        after_played_ratio=after_played_ratio,
        num_played_ratio_removed=num_played_ratio_removed,
        snapshots=snapshots,
        final_counts=final_counts,
        out_parquet=args.out_parquet,
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **stage3,
    }
    update_json_section(args.stats_out, "stage3", payload)

    logger.info("writing splits_metadata: %s", args.splits_metadata_out)
    write_splits_metadata(args.splits_metadata_out)

    logger.info("writing markdown report: %s", args.report_out)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(stage3), encoding="utf-8")

    logger.info("stage 3 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
