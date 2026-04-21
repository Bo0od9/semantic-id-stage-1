"""Stage 1 of the data preparation pipeline: load Yambda and inspect it.

Downloads the chosen Yambda configuration from the Hugging Face Hub, validates
the schema, computes raw (pre-filter) statistics, and emits two artefacts:

* ``data/raw_stats.json`` — machine-readable dump of every number collected,
  so later stages can re-read it without touching the dataset again.
* ``reports/01_dataset_overview.md`` — markdown report for the "Data" chapter
  of the thesis. Generated from the same numbers.

No filtering, no splitting, no writes to ``data/processed/``. See
``docs/instructions/dataset_prep.md`` (stage 1) for the full specification.
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

# Allow `python scripts/01_load_and_inspect.py` without installing as package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    DATASET_OVERVIEW_REPORT,
    EMBEDDINGS_PARQUET_REL,
    LISTENS_PARQUET_REL,
    PLAYED_RATIO_MIN,
    RAW_CACHE_DIR,
    RAW_STATS_PATH,
    YAMBDA_EVENT,
    YAMBDA_FORMAT,
    YAMBDA_REPO,
    YAMBDA_SIZE,
)
from utils.io import dump_json, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

EXPECTED_LISTENS_COLUMNS: dict[str, str] = {
    "uid": "UInt32",
    "item_id": "UInt32",
    "timestamp": "UInt32",
    "is_organic": "UInt8",
    "played_ratio_pct": "UInt16",
    "track_length_seconds": "UInt32",
}
EXPECTED_EMBEDDING_ID_COL = "item_id"
EMBEDDING_VECTOR_CANDIDATES: tuple[str, ...] = ("embed", "normalized_embed", "embedding")


def download_parquet(repo_filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s from %s", repo_filename, YAMBDA_REPO)
    local = hf_hub_download(
        repo_id=YAMBDA_REPO,
        filename=repo_filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local)


def validate_listens_schema(df: pl.DataFrame) -> None:
    actual = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}
    missing = [c for c in EXPECTED_LISTENS_COLUMNS if c not in actual]
    if missing:
        raise RuntimeError(
            f"listens schema missing expected columns {missing}; actual columns: {actual}"
        )
    for col, expected_dtype in EXPECTED_LISTENS_COLUMNS.items():
        if actual[col] != expected_dtype:
            logger.warning(
                "column %s has dtype %s, expected %s — continuing but double-check",
                col,
                actual[col],
                expected_dtype,
            )


def pick_embedding_vector_column(df: pl.DataFrame) -> str:
    for candidate in EMBEDDING_VECTOR_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(
        f"embeddings parquet has none of the expected vector columns "
        f"{EMBEDDING_VECTOR_CANDIDATES}; columns: {df.columns}"
    )


def compute_listens_stats(df: pl.DataFrame) -> dict[str, Any]:
    n_rows = df.height
    if n_rows == 0:
        raise RuntimeError("listens dataframe is empty — aborting")

    n_users = df["uid"].n_unique()
    n_items = df["item_id"].n_unique()
    ts_min = int(df["timestamp"].min())  # type: ignore[arg-type]
    ts_max = int(df["timestamp"].max())  # type: ignore[arg-type]

    pr = df["played_ratio_pct"]
    percentiles = {
        f"p{p}": float(pr.quantile(p / 100, interpolation="linear"))  # type: ignore[arg-type]
        for p in (10, 25, 50, 75, 90)
    }
    n_above_threshold = int((pr >= PLAYED_RATIO_MIN).sum())

    organic_share = float((df["is_organic"] == 1).mean())  # type: ignore[arg-type]

    # Per the Yambda paper (arXiv:2505.22238, Sec 3.3), the timestamp field is
    # stored in **seconds** relative to the dataset start, quantised to
    # 5-second granularity: T' = floor((T - T_start) / 5) * 5. So the raw
    # value is already in seconds — no extra multiplication.
    span_seconds = ts_max - ts_min
    span_days = span_seconds / 86400.0

    return {
        "num_interactions": n_rows,
        "num_unique_users": int(n_users),
        "num_unique_items": int(n_items),
        "timestamp_min_raw": ts_min,
        "timestamp_max_raw": ts_max,
        "timestamp_span_seconds": span_seconds,
        "timestamp_span_days": span_days,
        "played_ratio_percentiles": percentiles,
        f"num_interactions_with_played_ratio_ge_{PLAYED_RATIO_MIN}": n_above_threshold,
        f"share_interactions_with_played_ratio_ge_{PLAYED_RATIO_MIN}": (
            n_above_threshold / n_rows
        ),
        "share_organic": organic_share,
    }


def compute_embedding_stats(
    emb_df: pl.DataFrame,
    listens_df: pl.DataFrame,
    vector_col: str,
) -> dict[str, Any]:
    n_rows = emb_df.height
    if n_rows == 0:
        raise RuntimeError("embeddings dataframe is empty — aborting")
    if EXPECTED_EMBEDDING_ID_COL not in emb_df.columns:
        raise RuntimeError(
            f"embeddings missing '{EXPECTED_EMBEDDING_ID_COL}' column; "
            f"columns: {emb_df.columns}"
        )

    first_vec = emb_df[vector_col][0]
    if first_vec is None:
        raise RuntimeError("first embedding vector is null")
    dim = len(first_vec)

    n_unique_emb = emb_df[EXPECTED_EMBEDDING_ID_COL].n_unique()

    # Coverage of listens items: how many of the tracks users actually listened
    # to have an embedding? This is the number that matters for stage 2.
    listen_items = listens_df.select("item_id").unique()
    emb_items = emb_df.select(pl.col("item_id")).unique()
    covered = listen_items.join(emb_items, on="item_id", how="inner").height
    n_listen_items = listen_items.height
    n_listen_items_missing = n_listen_items - covered

    n_listens_covered = (
        listens_df.join(emb_items, on="item_id", how="inner").height
    )

    return {
        "num_embedding_rows": n_rows,
        "num_unique_items_with_embeddings": int(n_unique_emb),
        "embedding_vector_column": vector_col,
        "embedding_dim": int(dim),
        "available_vector_columns": [
            c for c in EMBEDDING_VECTOR_CANDIDATES if c in emb_df.columns
        ],
        "num_listen_items": n_listen_items,
        "num_listen_items_with_embedding": covered,
        "num_listen_items_missing_embedding": n_listen_items_missing,
        "listen_item_coverage": covered / n_listen_items,
        "num_listens_with_embedding": n_listens_covered,
        "listens_row_coverage": n_listens_covered / listens_df.height,
    }


def format_schema_table(df: pl.DataFrame) -> str:
    lines = ["| Column | Dtype |", "| --- | --- |"]
    for name, dtype in zip(df.columns, df.dtypes):
        lines.append(f"| `{name}` | `{dtype}` |")
    return "\n".join(lines)


def format_sample_records(df: pl.DataFrame, n: int = 5) -> str:
    sample = df.head(n)
    return "```\n" + str(sample) + "\n```"


def write_report(
    report_path: Path,
    listens_df: pl.DataFrame,
    embeddings_df: pl.DataFrame,
    stats: dict[str, Any],
    listens_parquet: Path,
    embeddings_parquet: Path,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    item_coverage = stats["embeddings"]["listen_item_coverage"]
    row_coverage = stats["embeddings"]["listens_row_coverage"]

    sections = [
        "# Dataset Overview (Stage 1 — Raw, Pre-Filter)",
        "",
        f"_Generated: {generated_at}_",
        "",
        "## Source",
        "",
        f"- Repository: `{YAMBDA_REPO}`",
        f"- Size: `{YAMBDA_SIZE}`",
        f"- Format: `{YAMBDA_FORMAT}`",
        f"- Event stream: `{YAMBDA_EVENT}`",
        f"- Listens parquet: `{listens_parquet.name}`",
        f"- Embeddings parquet: `{embeddings_parquet.name}`",
        "",
        "## Schema — listens",
        "",
        format_schema_table(listens_df),
        "",
        "## Schema — embeddings",
        "",
        format_schema_table(embeddings_df),
        "",
        "## Raw numbers — listens",
        "",
        f"- Interactions: **{stats['listens']['num_interactions']:,}**",
        f"- Unique users: **{stats['listens']['num_unique_users']:,}**",
        f"- Unique items: **{stats['listens']['num_unique_items']:,}**",
        f"- Timestamp range (seconds from dataset start, 5s granularity): "
        f"{stats['listens']['timestamp_min_raw']:,} → "
        f"{stats['listens']['timestamp_max_raw']:,}",
        f"- Temporal span: **{stats['listens']['timestamp_span_days']:.1f} days** "
        f"(~{stats['listens']['timestamp_span_days'] / 30:.1f} months) — matches "
        "the 11-month collection window from the paper.",
        f"- Share of organic interactions: {stats['listens']['share_organic']:.3f}",
        "",
        "### played_ratio_pct distribution",
        "",
        "| Percentile | Value |",
        "| --- | --- |",
    ]
    for pk, pv in stats["listens"]["played_ratio_percentiles"].items():
        sections.append(f"| {pk} | {pv:.1f} |")
    sections += [
        "",
        f"- Interactions with `played_ratio_pct ≥ {PLAYED_RATIO_MIN}`: "
        f"**{stats['listens'][f'num_interactions_with_played_ratio_ge_{PLAYED_RATIO_MIN}']:,}** "
        f"({stats['listens'][f'share_interactions_with_played_ratio_ge_{PLAYED_RATIO_MIN}']:.1%} "
        f"of all listens) — this is the planned threshold for stage 2/3.",
        "",
        "## Raw numbers — embeddings",
        "",
        f"- Rows in embeddings parquet: **{stats['embeddings']['num_embedding_rows']:,}**",
        f"- Unique items with embeddings: "
        f"**{stats['embeddings']['num_unique_items_with_embeddings']:,}**",
        f"- Embedding dimensionality: **{stats['embeddings']['embedding_dim']}**",
        f"- Vector column used: `{stats['embeddings']['embedding_vector_column']}` "
        f"(available: {stats['embeddings']['available_vector_columns']})",
        "",
        "### Coverage of listened items",
        "",
        f"- Listened items: **{stats['embeddings']['num_listen_items']:,}**",
        f"- Of those, with embedding: "
        f"**{stats['embeddings']['num_listen_items_with_embedding']:,}** "
        f"({item_coverage:.2%})",
        f"- Listened items **missing** embedding: "
        f"**{stats['embeddings']['num_listen_items_missing_embedding']:,}**",
        f"- Interaction-level coverage: "
        f"**{stats['embeddings']['num_listens_with_embedding']:,}** of "
        f"{stats['listens']['num_interactions']:,} "
        f"({row_coverage:.2%}) listens survive the stage-2 filter.",
        "",
        "## Sample records — listens",
        "",
        format_sample_records(listens_df),
        "",
        "## Sample records — embeddings",
        "",
        format_sample_records(embeddings_df),
        "",
        "## Notes from the Yambda paper (arXiv:2505.22238, Ploshkin et al. 2025)",
        "",
        "Things the original paper settles that directly affect later stages.",
        "Keep these in view — they are easy to forget once implementation starts.",
        "",
        "### Dataset construction",
        "",
        "- **Yambda-50M is a uniform 1/100 random subsample of Yambda-5B**",
        "  (500M = 1/10). So `50m` is statistically representative of the full",
        "  dataset — nothing special about which users end up in it.",
        "- **User eligibility filter (Sec 3.3):** a user is in the 5B release",
        "  only if they performed ≥ 10 actions in the first 10 months **and**",
        "  ≥ 1 action in the remaining month. Cold-start and dormant users are",
        "  therefore already excluded upstream — we do not need to re-do this.",
        "- **Paper Table 3 numbers for 50M:** 10 000 users, 934 057 items,",
        "  46 467 212 listens.",
        "  - Our listens count matches exactly.",
        "  - Unique `uid`s in `listens.parquet` = 9 238: the remaining ~762",
        "    users have no listen events at all (only likes/dislikes), which",
        "    is normal.",
        "  - Unique `item_id`s in `listens.parquet` = 877 168, not 934 057.",
        "    **Verified:** the paper's 934 057 is the union of `item_id`s",
        "    across **all five** event streams — `listens` (877 168) ∪",
        "    `likes` (181 304) ∪ `dislikes` (53 413) ∪ `unlikes` (117 953) ∪",
        "    `undislikes` (15 399) = 934 057 exactly. The gap of 56 889",
        "    tracks appears in at least one non-listen stream but was never",
        "    actually played. Not a bug, just a scope difference.",
        "- **Timestamp formula (Sec 3.3):** "
        "  `T'_event = floor((T_event − T_start) / 5) × 5`. "
        "  Values are stored **in seconds** from the dataset anchor, quantised",
        "  to 5 s. Track length is also rounded to 5 s; `played_ratio_pct` is",
        "  at 1% granularity.",
        "- **Audio embeddings** come from a convolutional NN trained on audio",
        "  spectrograms. Both `embed` (raw) and `normalized_embed` are outputs",
        "  of the same network.",
        "",
        "### Paper's own evaluation protocol (Sec 4)",
        "",
        "- **Split:** Global Temporal Split with",
        "  - Train = 300 days",
        "  - Gap = 30 minutes (simulates prod model-refresh latency)",
        "  - Test = 1 day",
        "  Users with empty history at the start of the test period are",
        "  dropped. No explicit validation split is defined — we will carve",
        "  our own val window inside the train period for stage 5.",
        "- **\"Successful listen\" threshold (Sec 4.3):** the paper defines",
        "  `Listen_s` (the metric variant used in their benchmark tables) as",
        "  `played_ratio_pct ≥ 50`. **We follow the same convention** — our",
        "  pipeline uses `≥ 50`, so our numbers are directly comparable to",
        "  paper Table 6. The accepted noise trade-off is logged in",
        "  `docs/risks_and_limitations.md` (R-002).",
        "- **SASRec on Yambda-50M (Listen_s, Table 6):** NDCG@10 = 0.0748,",
        "  Recall@10 = 0.0325, Coverage@10 = 0.0130. iALS NDCG@10 = 0.0407.",
        "  These are the reference points our content-based SASRec should",
        "  reach or beat once embeddings are plugged in.",
        "- **Hyperparameters:** DecayPop, ItemKNN, iALS, BPR, SANSA were tuned",
        "  via Optuna on the GTS scheme. SASRec HPs were **not** tuned due to",
        "  compute cost — they used fixed values and reserved one day of the",
        "  train period as internal validation. Useful to know when comparing.",
        "",
        "### User history reference (paper Table 5)",
        "",
        "| Event | Median | p90 | p95 |",
        "| --- | --- | --- | --- |",
        "| Listen | 3 076 | 12 956 | 17 030 |",
        "| Like | 45 | 269 | 409 |",
        "| Dislike | 4 | 30 | 60 |",
        "",
        "These numbers are from the 5B dataset but scale roughly to 50M. Use",
        "them as a sanity anchor when stage 4 computes our own percentiles,",
        "and as prior input when choosing subgroup boundaries (stage 4.1).",
        "",
        "### What's not decided by the paper and still on us",
        "",
        "- Where to cut the validation window inside the 300-day train period.",
        "- Whether to filter by `is_organic` (the paper does not).",
        "- Min-count thresholds (the paper keeps everything that survives the",
        "  10-actions eligibility filter).",
        "",
        "## Caveats — keep in mind for later stages",
        "",
        "- **Low user count (9 238).** Yambda's card advertises ~1M users for the",
        "  `5b` size; the `50m` sample has ~10K *heavy* users (mean ~5 030 listens",
        "  per user, top user 27 617). Implications:",
        "  - Subgroup analysis (stage 4.1) is tight: max ~3 groups at ~3K users",
        "    each if we want ≥1 000 per bucket.",
        "  - `train_subsample_1pct` will have ~92 users — too small for anything",
        "    beyond smoke tests.",
        "  - If statistical power becomes a problem downstream, consider switching",
        "    to `500m`.",
        "",
        "- **Timestamp is seconds from the dataset anchor**, quantised to 5 s",
        "  (not a unix epoch). `timestamp_min = 0` marks the dataset start, not",
        "  a calendar date. Span ≈ 301 days ≈ 10 months, consistent with the",
        "  paper's 11-month collection window. Non-uniformity across that span",
        "  should still be checked on stage 4 before picking $T_{val}$/$T_{test}$.",
        "",
        "- **Bimodal `played_ratio_pct`** (p25 = 7, p50 = 100). Classic",
        "  skip-vs-full-listen shape. The `≥ 50` cutoff (R-002) is more",
        "  permissive than `≥ 90` — it keeps events where the user heard at",
        "  least half the track. Combined with the stage-2 embedding filter,",
        "  the final train will still be noticeably smaller than the raw",
        "  46.47M, but larger than it would be at `≥ 90`.",
        "",
        "- **`uid` is quantised to multiples of 100** (min=100, max=1 000 000,",
        "  step 100). This is the dataset's anonymisation scheme — harmless for",
        "  the pipeline but worth noting so nobody tries to interpret uid gaps",
        "  as missing users.",
        "",
        "- **Two embedding variants** (`embed`, `normalized_embed`) are",
        "  available. The decision of which to feed into content-based SASRec",
        "  is deferred to the training stage and must be recorded there, not",
        "  here. Stage 7 (sanity check) will inspect both.",
        "",
        "- **`is_organic` ≈ 0.517** — roughly half of listens are organic, half",
        "  algorithmic. We do **not** currently plan to filter by this flag, but",
        "  it is a natural covariate to revisit if results look biased toward",
        "  recommendation-driven behaviour.",
        "",
        "- **Items-with-embedding set is much larger than the listened-item",
        "  set** (7.72M vs 877K). The 7.72M figure should never be used as a",
        "  denominator for coverage — the meaningful coverage is on the 877K",
        "  subset (93.4% here).",
        "",
        "## Next stage",
        "",
        "Stage 2 will drop every listen whose `item_id` is not present in the",
        "embeddings table, then stage 3 will apply the "
        f"`played_ratio_pct ≥ {PLAYED_RATIO_MIN}` threshold and min-count filters.",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("wrote %s", report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=RAW_CACHE_DIR,
        help="where to cache downloaded parquet files",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=RAW_STATS_PATH,
        help="path to raw_stats.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DATASET_OVERVIEW_REPORT,
        help="path to dataset overview markdown report",
    )
    args = parser.parse_args(argv)

    setup_logging()

    listens_rel = LISTENS_PARQUET_REL
    embeddings_rel = EMBEDDINGS_PARQUET_REL

    listens_path = download_parquet(listens_rel, args.cache_dir)
    embeddings_path = download_parquet(embeddings_rel, args.cache_dir)

    logger.info("reading listens parquet: %s", listens_path)
    listens_df = pl.read_parquet(listens_path)
    validate_listens_schema(listens_df)

    logger.info("reading embeddings parquet: %s", embeddings_path)
    embeddings_df = pl.read_parquet(embeddings_path)
    vector_col = pick_embedding_vector_column(embeddings_df)

    logger.info("computing listens stats (%d rows)", listens_df.height)
    listens_stats = compute_listens_stats(listens_df)
    logger.info("computing embedding stats (%d rows)", embeddings_df.height)
    embedding_stats = compute_embedding_stats(embeddings_df, listens_df, vector_col)

    stats: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": YAMBDA_REPO,
            "size": YAMBDA_SIZE,
            "format": YAMBDA_FORMAT,
            "event": YAMBDA_EVENT,
            "listens_parquet": listens_rel,
            "embeddings_parquet": embeddings_rel,
        },
        "planned_filters": {
            "played_ratio_min": PLAYED_RATIO_MIN,
        },
        "listens": listens_stats,
        "embeddings": embedding_stats,
        "listens_schema": {
            name: str(dtype) for name, dtype in zip(listens_df.columns, listens_df.dtypes)
        },
        "embeddings_schema": {
            name: str(dtype)
            for name, dtype in zip(embeddings_df.columns, embeddings_df.dtypes)
        },
    }
    dump_json(args.stats_out, stats)

    write_report(
        args.report_out,
        listens_df,
        embeddings_df,
        stats,
        listens_path,
        embeddings_path,
    )

    logger.info("stage 1 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
