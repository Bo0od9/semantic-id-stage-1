"""Stage 6 of the data preparation pipeline: nested user subsamples.

Carves two nested user-level subsamples of the stage-5 ``train`` split (10%
and 1%) plus matching ``val`` / ``test`` slices. The 1% set is drawn from
inside the 10% set — not independently — so pilot runs on ``10pct`` and
smoke runs on ``1pct`` stay directly comparable.

Design choices (frozen):

* **Seed** — read-only from ``splits_metadata.json::random_seed``. No new
  seeds are minted here.
* **Nesting** — ``users_1pct ⊂ users_10pct ⊂ train.users``.
* **1% floor** — if ``round(0.01·n_train_users) < SUBSAMPLE_MIN_USERS``
  (=100), the effective user count is raised to 100. The fraction that
  ended up being applied is recorded in ``stage6`` stats.
* **Val/test subsample = row-level semi-join** on ``train_sub.uid``
  followed by a cold-item cascade filter. When ``train`` shrinks to ``X%``,
  its item set shrinks too; val/test-sub rows whose ``item_id`` no longer
  appears in ``train_sub`` are dropped (analogous to stage 5's cold-start
  filter, but applied against the *subsample* train). The full ``train``
  subsample itself is authoritative for its subset and never filtered.

Outputs:

* ``data/processed/{train,val,test}_subsample_{10pct,1pct}.parquet`` (zstd)
* ``reports/08_subsamples.md``
* ``data/filter_stats.json``   — merged ``stage6`` section.
* ``data/splits_metadata.json`` — merged ``stage6_frozen_at_utc`` and
  ``subsamples.{10pct,1pct}.user_ids`` (full uid lists).

Fails fast on any invariant violation. No silent fallbacks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    FILTER_STATS_PATH,
    PROCESSED_DIR,
    SPLITS_METADATA_PATH,
    SUBSAMPLE_10PCT_FRACTION,
    SUBSAMPLE_10PCT_MIN_USERS_GUARD,
    SUBSAMPLE_1PCT_FRACTION,
    SUBSAMPLE_1PCT_MIN_USERS_GUARD,
    SUBSAMPLE_MIN_USERS,
    SUBSAMPLES_REPORT,
    TEST_PARQUET_PATH,
    TEST_SUBSAMPLE_10PCT_PARQUET_PATH,
    TEST_SUBSAMPLE_1PCT_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    TRAIN_SUBSAMPLE_10PCT_PARQUET_PATH,
    TRAIN_SUBSAMPLE_1PCT_PARQUET_PATH,
    VAL_PARQUET_PATH,
    VAL_SUBSAMPLE_10PCT_PARQUET_PATH,
    VAL_SUBSAMPLE_1PCT_PARQUET_PATH,
)
from utils.io import load_json, setup_logging, update_json_section  # noqa: E402
from utils.parquet import (  # noqa: E402
    atomic_write_parquet,
    counts,
    validate_listens_schema,
)

logger = logging.getLogger(__name__)

PCT_KEYS: tuple[str, ...] = ("10pct", "1pct")
PCT_FRACTIONS: dict[str, float] = {
    "10pct": SUBSAMPLE_10PCT_FRACTION,
    "1pct": SUBSAMPLE_1PCT_FRACTION,
}
PCT_MIN_GUARD: dict[str, int] = {
    "10pct": SUBSAMPLE_10PCT_MIN_USERS_GUARD,
    "1pct": SUBSAMPLE_1PCT_MIN_USERS_GUARD,
}
PCT_PATHS: dict[str, dict[str, Path]] = {
    "10pct": {
        "train": TRAIN_SUBSAMPLE_10PCT_PARQUET_PATH,
        "val": VAL_SUBSAMPLE_10PCT_PARQUET_PATH,
        "test": TEST_SUBSAMPLE_10PCT_PARQUET_PATH,
    },
    "1pct": {
        "train": TRAIN_SUBSAMPLE_1PCT_PARQUET_PATH,
        "val": VAL_SUBSAMPLE_1PCT_PARQUET_PATH,
        "test": TEST_SUBSAMPLE_1PCT_PARQUET_PATH,
    },
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_seed(meta_path: Path) -> int:
    if not meta_path.exists():
        raise RuntimeError(f"splits metadata not found at {meta_path}")
    payload = load_json(meta_path)
    if "random_seed" not in payload:
        raise RuntimeError(f"{meta_path} missing 'random_seed'; stage 3 must run first")
    seed = int(payload["random_seed"])
    logger.info("loaded seed=%d from %s", seed, meta_path)
    return seed


def read_split(path: Path, name: str) -> pl.DataFrame:
    if not path.exists():
        raise RuntimeError(f"stage-5 {name} parquet not found at {path}")
    df = pl.read_parquet(path)
    validate_listens_schema(df, origin=f"stage-5 {name} ({path})")
    logger.info("read stage-5 %s: %d rows", name, df.height)
    return df


# ---------------------------------------------------------------------------
# User selection
# ---------------------------------------------------------------------------


def _resolve_effective_count(
    pct_key: str, n_train_users: int
) -> tuple[int, float, bool]:
    requested = PCT_FRACTIONS[pct_key]
    n_raw = int(round(requested * n_train_users))
    floored = False
    n = n_raw
    if n < SUBSAMPLE_MIN_USERS:
        n = SUBSAMPLE_MIN_USERS
        floored = True
    if n > n_train_users:
        raise RuntimeError(
            f"{pct_key}: effective count {n} exceeds train population {n_train_users}"
        )
    effective = n / n_train_users
    logger.info(
        "%s: requested=%.4f raw=%d floored=%s effective=%d (%.4f)",
        pct_key,
        requested,
        n_raw,
        floored,
        n,
        effective,
    )
    return n, effective, floored


def pick_nested_users(
    train_uids: list[int], rng: np.random.Generator
) -> dict[str, dict[str, Any]]:
    """Draw 10% users from full train, then 1% users from inside the 10% set.

    Returns a dict keyed by pct with ``user_ids`` (sorted ints), ``num_users``,
    ``effective_fraction`` and ``floor_applied`` fields.
    """
    n_train = len(train_uids)
    if n_train == 0:
        raise RuntimeError("train has zero users; cannot subsample")

    # Deterministic base order: sort so the seeded draw is reproducible
    # regardless of parquet row ordering.
    base = np.array(sorted(train_uids), dtype=np.int64)

    n10, frac10, floor10 = _resolve_effective_count("10pct", n_train)
    idx10 = rng.choice(n_train, size=n10, replace=False)
    users_10pct_arr = np.sort(base[idx10])

    n1, frac1, floor1 = _resolve_effective_count("1pct", n_train)
    if n1 > n10:
        raise RuntimeError(
            f"nesting impossible: 1pct effective count {n1} > 10pct {n10}"
        )
    idx1 = rng.choice(n10, size=n1, replace=False)
    users_1pct_arr = np.sort(users_10pct_arr[idx1])

    return {
        "10pct": {
            "user_ids": [int(u) for u in users_10pct_arr.tolist()],
            "num_users": n10,
            "effective_fraction": frac10,
            "floor_applied": floor10,
            "requested_fraction": PCT_FRACTIONS["10pct"],
        },
        "1pct": {
            "user_ids": [int(u) for u in users_1pct_arr.tolist()],
            "num_users": n1,
            "effective_fraction": frac1,
            "floor_applied": floor1,
            "requested_fraction": PCT_FRACTIONS["1pct"],
        },
    }


# ---------------------------------------------------------------------------
# Materialise subsamples
# ---------------------------------------------------------------------------


def _drop_cold_items(
    side: pl.DataFrame, train_items: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Drop rows whose item_id is not present in ``train_items``.

    Reports row/user/item losses as distinct-identity deltas.
    """
    before_rows = side.height
    before_users = int(side.get_column("uid").n_unique()) if before_rows else 0
    before_items = int(side.get_column("item_id").n_unique()) if before_rows else 0

    filtered = side.join(train_items, on="item_id", how="semi")

    after_users = (
        int(filtered.get_column("uid").n_unique()) if filtered.height else 0
    )
    after_items = (
        int(filtered.get_column("item_id").n_unique()) if filtered.height else 0
    )
    return filtered, {
        "users_dropped": before_users - after_users,
        "items_dropped": before_items - after_items,
        "rows_dropped": before_rows - filtered.height,
    }


def materialise_subsample(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    user_ids: list[int],
) -> dict[str, Any]:
    users_df = pl.DataFrame({"uid": pl.Series("uid", user_ids, dtype=pl.UInt32)})

    train_sub = train.join(users_df, on="uid", how="semi")
    val_by_user = val.join(users_df, on="uid", how="semi")
    test_by_user = test.join(users_df, on="uid", how="semi")

    train_items = train_sub.select("item_id").unique()
    val_sub, val_cold = _drop_cold_items(val_by_user, train_items)
    test_sub, test_cold = _drop_cold_items(test_by_user, train_items)

    logger.info(
        "subsample: train=%d val=%d->%d (cold rows=%d) test=%d->%d (cold rows=%d)",
        train_sub.height,
        val_by_user.height,
        val_sub.height,
        val_cold["rows_dropped"],
        test_by_user.height,
        test_sub.height,
        test_cold["rows_dropped"],
    )
    return {
        "train": train_sub,
        "val_raw": val_by_user,
        "val": val_sub,
        "val_cold": val_cold,
        "test_raw": test_by_user,
        "test": test_sub,
        "test_cold": test_cold,
    }


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def assert_min_sizes(users_by_pct: dict[str, dict[str, Any]]) -> None:
    for key, guard in PCT_MIN_GUARD.items():
        n = users_by_pct[key]["num_users"]
        if n < guard:
            raise RuntimeError(
                f"{key}: num_users={n} below min-size guard {guard}"
            )


def assert_nesting(users_by_pct: dict[str, dict[str, Any]], train_uids: set[int]) -> None:
    u10 = set(users_by_pct["10pct"]["user_ids"])
    u1 = set(users_by_pct["1pct"]["user_ids"])
    if not u10.issubset(train_uids):
        raise RuntimeError("users_10pct is not a subset of train users")
    if not u1.issubset(u10):
        raise RuntimeError("users_1pct is not a subset of users_10pct")
    if len(u10) != users_by_pct["10pct"]["num_users"]:
        raise RuntimeError("users_10pct has duplicates")
    if len(u1) != users_by_pct["1pct"]["num_users"]:
        raise RuntimeError("users_1pct has duplicates")


def assert_subsample_invariants(
    pct_key: str,
    frames: dict[str, Any],
) -> None:
    train_sub: pl.DataFrame = frames["train"]
    val_sub: pl.DataFrame = frames["val"]
    test_sub: pl.DataFrame = frames["test"]

    if val_sub.height == 0:
        raise RuntimeError(f"{pct_key}: val subsample is empty after cold-item filter")
    if test_sub.height == 0:
        raise RuntimeError(f"{pct_key}: test subsample is empty after cold-item filter")

    validate_listens_schema(train_sub, origin=f"{pct_key} train_sub")
    validate_listens_schema(val_sub, origin=f"{pct_key} val_sub")
    validate_listens_schema(test_sub, origin=f"{pct_key} test_sub")

    train_users = set(train_sub.get_column("uid").unique().to_list())
    train_items = set(train_sub.get_column("item_id").unique().to_list())

    val_users = set(val_sub.get_column("uid").unique().to_list())
    val_items = set(val_sub.get_column("item_id").unique().to_list())
    test_users = set(test_sub.get_column("uid").unique().to_list())
    test_items = set(test_sub.get_column("item_id").unique().to_list())

    if not val_users.issubset(train_users):
        raise RuntimeError(f"{pct_key}: val users not subset of train_sub users")
    if not test_users.issubset(train_users):
        raise RuntimeError(f"{pct_key}: test users not subset of train_sub users")
    if not val_items.issubset(train_items):
        raise RuntimeError(f"{pct_key}: val items not subset of train_sub items")
    if not test_items.issubset(train_items):
        raise RuntimeError(f"{pct_key}: test items not subset of train_sub items")

    key_cols = ["uid", "item_id", "timestamp"]
    train_keys = train_sub.select(key_cols)
    val_keys = val_sub.select(key_cols)
    test_keys = test_sub.select(key_cols)
    if train_keys.join(val_keys, on=key_cols, how="semi").height > 0:
        raise RuntimeError(f"{pct_key}: row leakage train ∩ val")
    if train_keys.join(test_keys, on=key_cols, how="semi").height > 0:
        raise RuntimeError(f"{pct_key}: row leakage train ∩ test")
    if val_keys.join(test_keys, on=key_cols, how="semi").height > 0:
        raise RuntimeError(f"{pct_key}: row leakage val ∩ test")


INVARIANT_NAMES: tuple[str, ...] = (
    "min_sizes_per_pct",
    "nesting_1pct_in_10pct",
    "nesting_10pct_in_train",
    "val_users_subset_train_sub",
    "test_users_subset_train_sub",
    "val_items_subset_train_sub",
    "test_items_subset_train_sub",
    "val_non_empty",
    "test_non_empty",
    "schema_identical_to_stage5",
    "row_key_disjoint_train_val_test",
)


# ---------------------------------------------------------------------------
# Stats + report
# ---------------------------------------------------------------------------


def build_pct_stats(
    users_entry: dict[str, Any], frames: dict[str, Any]
) -> dict[str, Any]:
    return {
        "requested_fraction": users_entry["requested_fraction"],
        "effective_fraction": users_entry["effective_fraction"],
        "floor_applied": users_entry["floor_applied"],
        "num_users": users_entry["num_users"],
        "train": counts(frames["train"]),
        "val_raw": counts(frames["val_raw"]),
        "val_cold_drops": frames["val_cold"],
        "val": counts(frames["val"]),
        "test_raw": counts(frames["test_raw"]),
        "test_cold_drops": frames["test_cold"],
        "test": counts(frames["test"]),
    }


def build_stage6_stats(
    seed: int,
    pre: dict[str, dict[str, int]],
    pct_stats: dict[str, dict[str, Any]],
    nesting: dict[str, int],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "description": (
            "Nested user subsamples (10% and 1%) of stage-5 train/val/test "
            "with cold-item cascade filter on hold-outs."
        ),
        "source": {
            "train_parquet": str(TRAIN_PARQUET_PATH),
            "val_parquet": str(VAL_PARQUET_PATH),
            "test_parquet": str(TEST_PARQUET_PATH),
        },
        "seed": seed,
        "min_users_floor": SUBSAMPLE_MIN_USERS,
        "min_users_guards": PCT_MIN_GUARD,
        "pre": pre,
        "subsamples": pct_stats,
        "nesting": nesting,
        "invariants_checked": list(INVARIANT_NAMES),
        "output": {
            pct: {split: str(path) for split, path in splits.items()}
            for pct, splits in PCT_PATHS.items()
        },
    }


def render_report(stats: dict[str, Any]) -> str:
    pre = stats["pre"]
    seed = stats["seed"]
    ts = stats["generated_at_utc"].replace("+00:00", " UTC")

    def _counts_row(name: str, d: dict[str, int]) -> str:
        return (
            f"| {name} | {d['num_interactions']:,} | "
            f"{d['num_users']:,} | {d['num_items']:,} |"
        )

    lines: list[str] = [
        "# Subsamples (Stage 6)",
        "",
        f"_Generated: {ts}_",
        "",
        "## What this stage does",
        "",
        "Carves two nested user-level subsamples of the stage-5 train/val/test "
        "parquets so pilot and smoke runs are reproducible and comparable. "
        "10% users are drawn from the full train population; 1% users are "
        "drawn from inside the 10% set (not independently). Val/test "
        "subsamples are row-level semi-joins on `train_sub.uid`, followed "
        "by a **cold-item cascade filter** — rows whose `item_id` no longer "
        "appears in the shrunken `train_sub` are dropped. Train subsamples "
        "themselves are authoritative for their own subset and never filtered. "
        "The seed (`random_seed` = "
        f"**{seed}**) is inherited from `splits_metadata.json`; no new seed "
        "is minted.",
        "",
        "## Inputs (= stage-5 final)",
        "",
        "| Split | Interactions | Users | Items |",
        "| :--- | ---: | ---: | ---: |",
        _counts_row("train", pre["train"]),
        _counts_row("val", pre["val"]),
        _counts_row("test", pre["test"]),
        "",
    ]

    for pct in PCT_KEYS:
        sub = stats["subsamples"][pct]
        lines += [
            f"## {pct} subsample",
            "",
            f"- Requested fraction: **{sub['requested_fraction']:.4f}**",
            f"- Effective fraction: **{sub['effective_fraction']:.4f}**"
            + ("  (floor applied)" if sub["floor_applied"] else ""),
            f"- Users: **{sub['num_users']:,}**",
            "",
            "| Split | Interactions | Users | Items |",
            "| :--- | ---: | ---: | ---: |",
            _counts_row("train", sub["train"]),
            _counts_row("val_raw", sub["val_raw"]),
            _counts_row("val", sub["val"]),
            _counts_row("test_raw", sub["test_raw"]),
            _counts_row("test", sub["test"]),
            "",
            "Cold-item cascade drops (rows / users / items lost on hold-outs "
            "because the referenced item no longer exists in the shrunken "
            "`train_sub`):",
            "",
            "| Split | Users dropped | Items dropped | Rows dropped |",
            "| :--- | ---: | ---: | ---: |",
            f"| val  | {sub['val_cold_drops']['users_dropped']:,}"
            f" | {sub['val_cold_drops']['items_dropped']:,}"
            f" | {sub['val_cold_drops']['rows_dropped']:,} |",
            f"| test | {sub['test_cold_drops']['users_dropped']:,}"
            f" | {sub['test_cold_drops']['items_dropped']:,}"
            f" | {sub['test_cold_drops']['rows_dropped']:,} |",
            "",
        ]

    lines += [
        "## Nesting",
        "",
        f"- `|users_1pct ∩ users_10pct|` = "
        f"{stats['nesting']['intersection_size']:,}"
        f" (must equal |users_1pct| = {stats['nesting']['users_1pct']:,}) — ✓",
        f"- `|users_10pct ∩ users_train|` = "
        f"{stats['nesting']['users_10pct']:,}"
        f" (must equal |users_10pct|) — ✓",
        "",
        "## Invariant checks",
        "",
        "All of the following are asserted at the end of the stage — the "
        "script fails fast otherwise:",
        "",
    ]
    for name in stats["invariants_checked"]:
        lines.append(f"- ✓ `{name}`")

    lines += [
        "",
        "## Statistical-power warning",
        "",
        f"`train_subsample_1pct` has {stats['subsamples']['1pct']['num_users']:,}"
        f" users and {stats['subsamples']['1pct']['train']['num_interactions']:,}"
        " train interactions. This is intended for smoke tests "
        "only (does the pipeline run, are there NaNs, are all ranks "
        "populated) — it is **not** statistically meaningful for model "
        "comparison. Use `train_subsample_10pct` for pilot experiments and "
        "the full `train` for final numbers.",
        "",
        "## Artifacts",
        "",
    ]
    for pct in PCT_KEYS:
        for split, path in PCT_PATHS[pct].items():
            rel = path.relative_to(path.parents[2])
            lines.append(f"- `{rel}` (zstd)")
    lines += [
        "- `data/filter_stats.json` (section `stage6`)",
        "- `data/splits_metadata.json` (keys `stage6_frozen_at_utc`,"
        " `subsamples.{10pct,1pct}.user_ids`)",
        "",
        "## Next stage",
        "",
        "Stage 7 (embeddings sanity check) and stage 8 (push to hub) can "
        "now consume either the full splits or any of the subsamples.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--val-parquet", type=Path, default=VAL_PARQUET_PATH)
    p.add_argument("--test-parquet", type=Path, default=TEST_PARQUET_PATH)
    p.add_argument("--splits-metadata", type=Path, default=SPLITS_METADATA_PATH)
    p.add_argument("--filter-stats", type=Path, default=FILTER_STATS_PATH)
    p.add_argument("--report", type=Path, default=SUBSAMPLES_REPORT)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    seed = load_seed(args.splits_metadata)
    train = read_split(args.train_parquet, "train")
    val = read_split(args.val_parquet, "val")
    test = read_split(args.test_parquet, "test")

    pre = {"train": counts(train), "val": counts(val), "test": counts(test)}
    logger.info("pre counts: %s", pre)

    train_uids = train.get_column("uid").unique().to_list()
    train_uids_set = set(int(u) for u in train_uids)

    rng = np.random.default_rng(seed)
    users_by_pct = pick_nested_users(train_uids, rng)
    assert_min_sizes(users_by_pct)
    assert_nesting(users_by_pct, train_uids_set)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames_by_pct: dict[str, dict[str, Any]] = {}
    pct_stats: dict[str, dict[str, Any]] = {}

    for pct in PCT_KEYS:
        logger.info("==> materialising %s", pct)
        frames = materialise_subsample(
            train, val, test, users_by_pct[pct]["user_ids"]
        )
        assert_subsample_invariants(pct, frames)
        frames_by_pct[pct] = frames
        pct_stats[pct] = build_pct_stats(users_by_pct[pct], frames)

        for split in ("train", "val", "test"):
            path = PCT_PATHS[pct][split]
            atomic_write_parquet(frames[split], path)
            logger.info("wrote %s (%d rows)", path, frames[split].height)

    u10 = set(users_by_pct["10pct"]["user_ids"])
    u1 = set(users_by_pct["1pct"]["user_ids"])
    nesting = {
        "users_10pct": len(u10),
        "users_1pct": len(u1),
        "intersection_size": len(u10 & u1),
    }

    stats = build_stage6_stats(seed, pre, pct_stats, nesting=nesting)
    update_json_section(args.filter_stats, "stage6", stats)

    update_json_section(
        args.splits_metadata,
        "subsamples",
        {
            pct: {
                "requested_fraction": users_by_pct[pct]["requested_fraction"],
                "effective_fraction": users_by_pct[pct]["effective_fraction"],
                "floor_applied": users_by_pct[pct]["floor_applied"],
                "num_users": users_by_pct[pct]["num_users"],
                "user_ids": users_by_pct[pct]["user_ids"],
            }
            for pct in PCT_KEYS
        },
    )
    update_json_section(
        args.splits_metadata,
        "stage6_frozen_at_utc",
        datetime.now(tz=timezone.utc).isoformat(),
    )

    report_md = render_report(stats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md, encoding="utf-8")
    logger.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
