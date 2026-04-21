"""Stage 2 of the data preparation pipeline: filter listens by embedding coverage.

Drops every listen whose ``item_id`` is not present in the embeddings table.
Users whose entire history consisted of tracks without embeddings are removed
as well. No other filters are applied here — ``played_ratio_pct`` threshold and
min-count filtering belong to stage 3.

Outputs:

* ``data/interim/listens_stage2.parquet`` — filtered listens.
* ``data/filter_stats.json`` — JSON with a ``stage2`` section (merged with any
  existing content so later stages can append their own sections).
* ``reports/02_embedding_filter.md`` — standalone markdown report for the
  thesis "Data" chapter.

Pre-filter numbers are cross-checked against ``data/raw_stats.json`` (written
by stage 1) to catch silent drift if the cached parquet files change between
runs. See ``docs/instructions/dataset_prep.md`` (stage 2) for the spec.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    EMBEDDING_FILTER_REPORT,
    EMBEDDINGS_PARQUET_REL,
    FILTER_STATS_PATH,
    LISTENS_PARQUET_REL,
    LISTENS_STAGE2_PATH,
    LOST_HISTORY_HEAVY_THRESHOLD,
    RAW_CACHE_DIR,
    RAW_STATS_PATH,
    YAMBDA_REPO,
)
from utils.io import load_json, setup_logging, update_json_section  # noqa: E402

logger = logging.getLogger(__name__)

LOST_FRACTION_PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 99)


def download_parquet(repo_filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ensuring %s is cached (repo=%s)", repo_filename, YAMBDA_REPO)
    local = hf_hub_download(
        repo_id=YAMBDA_REPO,
        filename=repo_filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local)


def assert_matches_stage1(
    raw_stats: dict[str, Any],
    pre: dict[str, int],
) -> None:
    expected_listens = raw_stats["listens"]["num_interactions"]
    expected_users = raw_stats["listens"]["num_unique_users"]
    expected_items = raw_stats["listens"]["num_unique_items"]
    mismatches = []
    if pre["num_listens"] != expected_listens:
        mismatches.append(
            f"num_listens {pre['num_listens']} != raw_stats {expected_listens}"
        )
    if pre["num_unique_users"] != expected_users:
        mismatches.append(
            f"num_unique_users {pre['num_unique_users']} != raw_stats {expected_users}"
        )
    if pre["num_unique_items"] != expected_items:
        mismatches.append(
            f"num_unique_items {pre['num_unique_items']} != raw_stats {expected_items}"
        )
    if mismatches:
        raise RuntimeError(
            "stage 2 pre-filter numbers diverged from stage 1 raw_stats.json: "
            + "; ".join(mismatches)
            + " — re-run stage 1 before continuing."
        )


def compute_pre_stats(listens: pl.DataFrame) -> dict[str, int]:
    return {
        "num_listens": listens.height,
        "num_unique_users": int(listens["uid"].n_unique()),
        "num_unique_items": int(listens["item_id"].n_unique()),
    }


def filter_by_embeddings(
    listens: pl.DataFrame, emb_ids: pl.DataFrame
) -> pl.DataFrame:
    return listens.join(emb_ids, on="item_id", how="inner")


def compute_lost_fraction(
    listens: pl.DataFrame, filtered: pl.DataFrame
) -> pl.DataFrame:
    before = listens.group_by("uid").agg(pl.len().alias("n_before"))
    after = filtered.group_by("uid").agg(pl.len().alias("n_after"))
    joined = before.join(after, on="uid", how="left").with_columns(
        pl.col("n_after").fill_null(0),
    )
    return joined.with_columns(
        (1.0 - pl.col("n_after") / pl.col("n_before")).alias("lost_fraction"),
    )


def build_stage2_stats(
    raw_stats: dict[str, Any],
    pre: dict[str, int],
    post: dict[str, int],
    loss_df: pl.DataFrame,
    out_parquet: Path,
) -> dict[str, Any]:
    num_listened_items = pre["num_unique_items"]
    num_items_with_emb = post["num_unique_items"]
    num_items_without_emb = num_listened_items - num_items_with_emb

    num_users_fully_lost = int(
        loss_df.filter(pl.col("n_after") == 0).height
    )
    surviving = loss_df.filter(pl.col("n_after") > 0)
    num_users_heavy_loss = int(
        surviving.filter(
            pl.col("lost_fraction") >= LOST_HISTORY_HEAVY_THRESHOLD
        ).height
    )

    lost_series = surviving["lost_fraction"]
    percentiles: dict[str, float] = {}
    if lost_series.len() > 0:
        for p in LOST_FRACTION_PERCENTILES:
            val = lost_series.quantile(p / 100, interpolation="linear")
            percentiles[f"p{p}"] = float(val) if val is not None else 0.0
        percentiles["max"] = float(lost_series.max())  # type: ignore[arg-type]
    else:
        for p in LOST_FRACTION_PERCENTILES:
            percentiles[f"p{p}"] = 0.0
        percentiles["max"] = 0.0

    row_coverage = post["num_listens"] / pre["num_listens"]
    item_coverage = num_items_with_emb / num_listened_items

    return {
        "description": (
            "Keep only listens whose item_id has an embedding; drop users "
            "whose entire history was removed."
        ),
        "input": {
            "source": str(RAW_STATS_PATH.relative_to(RAW_STATS_PATH.parents[1])),
        },
        "output_parquet": str(out_parquet.relative_to(out_parquet.parents[2])),
        "items": {
            "listened_total": num_listened_items,
            "with_embedding": num_items_with_emb,
            "without_embedding": num_items_without_emb,
            "item_coverage": item_coverage,
        },
        "interactions": {
            "before": pre["num_listens"],
            "after": post["num_listens"],
            "row_coverage": row_coverage,
        },
        "users": {
            "before": pre["num_unique_users"],
            "after": post["num_unique_users"],
            "fully_lost": num_users_fully_lost,
            "heavy_loss_ge_80pct": num_users_heavy_loss,
        },
        "lost_fraction_percentiles": percentiles,
        "cross_check_raw_stats": {
            "expected_num_listens_after": raw_stats["embeddings"][
                "num_listens_with_embedding"
            ],
            "expected_num_unique_items_after": raw_stats["embeddings"][
                "num_listen_items_with_embedding"
            ],
        },
    }


def render_report(stage2: dict[str, Any]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    items = stage2["items"]
    inter = stage2["interactions"]
    users = stage2["users"]
    pct = stage2["lost_fraction_percentiles"]

    lines = [
        "# Embedding Filter (Stage 2)",
        "",
        f"_Generated: {generated_at}_",
        "",
        "## What this stage does",
        "",
        "Keeps only those listens whose `item_id` has a content-based "
        "embedding in Yambda's `embeddings.parquet`. Users whose entire "
        "history was removed are dropped. No `played_ratio_pct` or min-count "
        "filters are applied here — those belong to stage 3.",
        "",
        "## Before / after",
        "",
        "| Metric | Before | After | Coverage |",
        "| --- | ---: | ---: | ---: |",
        f"| Listens | {inter['before']:,} | {inter['after']:,} | "
        f"{inter['row_coverage']:.2%} |",
        f"| Unique users | {users['before']:,} | {users['after']:,} | "
        f"{users['after'] / users['before']:.2%} |",
        f"| Unique items | {items['listened_total']:,} | "
        f"{items['with_embedding']:,} | {items['item_coverage']:.2%} |",
        "",
        "## Items dropped",
        "",
        f"- Listened items without embedding: **{items['without_embedding']:,}** "
        f"of {items['listened_total']:,} "
        f"({items['without_embedding'] / items['listened_total']:.2%}).",
        "",
        "## Users affected",
        "",
        f"- Users fully lost (all their tracks had no embedding): "
        f"**{users['fully_lost']:,}**.",
        f"- Users with heavy loss (≥ {int(LOST_HISTORY_HEAVY_THRESHOLD * 100)}% "
        f"of their history removed, excluding fully lost): "
        f"**{users['heavy_loss_ge_80pct']:,}**.",
        "",
        "### Lost-fraction percentiles (surviving users only)",
        "",
        "| Percentile | Lost fraction |",
        "| --- | ---: |",
    ]
    for key in ("p50", "p75", "p90", "p95", "p99", "max"):
        lines.append(f"| {key} | {pct[key]:.4f} |")
    lines += [
        "",
        "## Cross-check",
        "",
        "Pre-filter counts were validated against `data/raw_stats.json` "
        "(stage 1). Post-filter row count matches "
        f"`embeddings.num_listens_with_embedding` = "
        f"**{stage2['cross_check_raw_stats']['expected_num_listens_after']:,}**.",
        "",
        "## Output",
        "",
        f"- `{stage2['output_parquet']}` — filtered listens parquet, zstd-compressed.",
        "- `data/filter_stats.json` (section `stage2`) — machine-readable dump.",
        "",
        "## Next stage",
        "",
        "Stage 3 will apply `played_ratio_pct ≥ 50` and min-count "
        "filtering (min 5 interactions per user and per item, iterated until "
        "stable).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=RAW_CACHE_DIR)
    parser.add_argument("--out-parquet", type=Path, default=LISTENS_STAGE2_PATH)
    parser.add_argument("--stats-out", type=Path, default=FILTER_STATS_PATH)
    parser.add_argument("--report-out", type=Path, default=EMBEDDING_FILTER_REPORT)
    parser.add_argument("--raw-stats", type=Path, default=RAW_STATS_PATH)
    args = parser.parse_args(argv)

    setup_logging()

    if not args.raw_stats.exists():
        raise SystemExit(
            f"stage 1 raw stats not found at {args.raw_stats}; run stage 1 first."
        )
    raw_stats = load_json(args.raw_stats)

    listens_path = download_parquet(LISTENS_PARQUET_REL, args.cache_dir)
    embeddings_path = download_parquet(EMBEDDINGS_PARQUET_REL, args.cache_dir)

    logger.info("reading listens parquet: %s", listens_path)
    listens = pl.read_parquet(listens_path)
    logger.info("reading embeddings parquet (item_id column only): %s", embeddings_path)
    emb_ids = pl.read_parquet(embeddings_path, columns=["item_id"]).unique()

    pre = compute_pre_stats(listens)
    logger.info(
        "pre-filter: %d listens, %d users, %d items",
        pre["num_listens"],
        pre["num_unique_users"],
        pre["num_unique_items"],
    )
    assert_matches_stage1(raw_stats, pre)

    logger.info("filtering listens by embedding presence")
    filtered = filter_by_embeddings(listens, emb_ids)
    if filtered.height == 0:
        raise RuntimeError("filtered dataframe is empty — embedding coverage is broken")
    if filtered["item_id"].null_count() or filtered["uid"].null_count():
        raise RuntimeError("filtered dataframe has nulls in item_id or uid")

    logger.info("computing per-user lost fraction")
    loss_df = compute_lost_fraction(listens, filtered)

    # Drop users that lost their entire history.
    surviving_users = loss_df.filter(pl.col("n_after") > 0).select("uid")
    num_fully_lost = loss_df.height - surviving_users.height
    if num_fully_lost > 0:
        logger.info("dropping %d users with zero remaining history", num_fully_lost)
        filtered = filtered.join(surviving_users, on="uid", how="inner")

    post = compute_pre_stats(filtered)
    logger.info(
        "post-filter: %d listens, %d users, %d items",
        post["num_listens"],
        post["num_unique_users"],
        post["num_unique_items"],
    )

    expected_listens_after = raw_stats["embeddings"]["num_listens_with_embedding"]
    if post["num_listens"] != expected_listens_after:
        raise RuntimeError(
            f"post-filter listens count {post['num_listens']} != "
            f"stage 1 prediction {expected_listens_after}"
        )

    logger.info("writing filtered parquet: %s", args.out_parquet)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(args.out_parquet, compression="zstd")

    stage2 = build_stage2_stats(raw_stats, pre, post, loss_df, args.out_parquet)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **stage2,
    }
    update_json_section(args.stats_out, "stage2", payload)

    logger.info("writing markdown report: %s", args.report_out)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(stage2), encoding="utf-8")

    logger.info("stage 2 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
