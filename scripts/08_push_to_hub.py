"""Stage 8: publish processed dataset to HuggingFace Hub.

Builds `item_embeddings.parquet` from the train split joined against raw
Yambda audio embeddings, then pushes two configs to a private repo:

  - `interactions`     — 9 splits (train/val/test + 10pct/1pct subsamples)
  - `item_embeddings`  — train items only (item_id, embed, normalized_embed,
                         popularity)

Metadata JSONs (`splits_metadata.json`, `filter_stats.json`, `raw_stats.json`)
and a curated README.md are uploaded separately via `HfApi.upload_file`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download, whoami

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    EMBEDDING_DIM_EXPECTED,
    EMBEDDINGS_PARQUET_REL,
    FILTER_STATS_PATH,
    HF_REPO_DEFAULT_PRIVATE,
    HF_REPO_NAME,
    HUB_PUBLISH_REPORT,
    ITEM_EMBEDDINGS_PARQUET_PATH,
    PROCESSED_DIR,
    RAW_CACHE_DIR,
    RAW_STATS_PATH,
    REPORTS_DIR,
    SPLITS_METADATA_PATH,
    TEST_PARQUET_PATH,
    TEST_SUBSAMPLE_1PCT_PARQUET_PATH,
    TEST_SUBSAMPLE_10PCT_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    TRAIN_SUBSAMPLE_1PCT_PARQUET_PATH,
    TRAIN_SUBSAMPLE_10PCT_PARQUET_PATH,
    VAL_PARQUET_PATH,
    VAL_SUBSAMPLE_1PCT_PARQUET_PATH,
    VAL_SUBSAMPLE_10PCT_PARQUET_PATH,
    YAMBDA_REPO,
)
from utils.io import setup_logging

logger = logging.getLogger(__name__)

INTERACTIONS_SPLITS: dict[str, Path] = {
    "train": TRAIN_PARQUET_PATH,
    "validation": VAL_PARQUET_PATH,
    "test": TEST_PARQUET_PATH,
    "train_subsample_10pct": TRAIN_SUBSAMPLE_10PCT_PARQUET_PATH,
    "validation_subsample_10pct": VAL_SUBSAMPLE_10PCT_PARQUET_PATH,
    "test_subsample_10pct": TEST_SUBSAMPLE_10PCT_PARQUET_PATH,
    "train_subsample_1pct": TRAIN_SUBSAMPLE_1PCT_PARQUET_PATH,
    "validation_subsample_1pct": VAL_SUBSAMPLE_1PCT_PARQUET_PATH,
    "test_subsample_1pct": TEST_SUBSAMPLE_1PCT_PARQUET_PATH,
}

METADATA_FILES: list[Path] = [
    SPLITS_METADATA_PATH,
    FILTER_STATS_PATH,
    RAW_STATS_PATH,
]

README_LOCAL_PATH: Path = PROCESSED_DIR / "README.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-name", default=HF_REPO_NAME)
    p.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=HF_REPO_DEFAULT_PRIVATE,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build artifacts locally, print summary, skip push/upload/verify.",
    )
    p.add_argument(
        "--skip-embeddings-extract",
        action="store_true",
        help="Reuse an existing data/processed/item_embeddings.parquet.",
    )
    return p.parse_args()


def auth_check() -> str:
    try:
        info = whoami()
    except Exception as err:
        raise RuntimeError(
            "Not authenticated to HuggingFace Hub. Run `hf auth login` first."
        ) from err
    username = info.get("name") if isinstance(info, dict) else None
    if not username:
        raise RuntimeError(f"whoami() returned no username: {info!r}")
    logger.info("Authenticated as %s", username)
    return username


def extract_item_embeddings() -> pl.DataFrame:
    logger.info("Reading train split for item_id + popularity...")
    train_df = pl.read_parquet(TRAIN_PARQUET_PATH, columns=["item_id"])
    n_train_rows = train_df.height
    logger.info("train rows = %d", n_train_rows)

    popularity = (
        train_df.group_by("item_id")
        .len()
        .rename({"len": "popularity"})
        .with_columns(pl.col("popularity").cast(pl.UInt32))
    )
    n_items = popularity.height
    logger.info("train unique items = %d", n_items)

    pop_sum = int(popularity["popularity"].sum())
    if pop_sum != n_train_rows:
        raise RuntimeError(
            f"Popularity sanity failed: sum={pop_sum} != train rows={n_train_rows}"
        )

    logger.info("Downloading raw Yambda embeddings (cached if present)...")
    emb_local = hf_hub_download(
        repo_id=YAMBDA_REPO,
        filename=EMBEDDINGS_PARQUET_REL,
        repo_type="dataset",
        cache_dir=str(RAW_CACHE_DIR),
    )
    emb_df = pl.read_parquet(
        emb_local, columns=["item_id", "embed", "normalized_embed"]
    )
    logger.info("Raw embeddings rows = %d", emb_df.height)

    joined = (
        emb_df.join(popularity, on="item_id", how="inner")
        .select(["item_id", "embed", "normalized_embed", "popularity"])
        .sort("item_id")
    )
    if joined.height != n_items:
        raise RuntimeError(
            f"Inner join shrunk: train items {n_items}, after join "
            f"{joined.height}. All train items must have embeddings."
        )

    first_embed = np.asarray(joined.item(0, "embed"), dtype=np.float64)
    if first_embed.shape != (EMBEDDING_DIM_EXPECTED,):
        raise RuntimeError(
            f"Expected embed shape ({EMBEDDING_DIM_EXPECTED},), "
            f"got {first_embed.shape}"
        )
    if joined.schema["item_id"] != pl.UInt32:
        raise RuntimeError(f"item_id dtype: {joined.schema['item_id']}")
    if joined.schema["popularity"] != pl.UInt32:
        raise RuntimeError(f"popularity dtype: {joined.schema['popularity']}")

    logger.info(
        "item_embeddings built: %d rows x {item_id, embed, normalized_embed, popularity}",
        joined.height,
    )
    return joined


def write_item_embeddings(df: pl.DataFrame) -> None:
    ITEM_EMBEDDINGS_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ITEM_EMBEDDINGS_PARQUET_PATH.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(ITEM_EMBEDDINGS_PARQUET_PATH)
    size_mb = ITEM_EMBEDDINGS_PARQUET_PATH.stat().st_size / 1024**2
    logger.info("Wrote %s (%.1f MB)", ITEM_EMBEDDINGS_PARQUET_PATH, size_mb)


def build_interactions_dataset() -> DatasetDict:
    splits: dict[str, Dataset] = {}
    for name, path in INTERACTIONS_SPLITS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Interactions split '{name}' missing: {path}"
            )
        logger.info("Loading '%s' as HF Dataset", name)
        splits[name] = Dataset.from_parquet(str(path))
    return DatasetDict(splits)


def build_item_embeddings_dataset() -> DatasetDict:
    if not ITEM_EMBEDDINGS_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"item_embeddings.parquet missing: {ITEM_EMBEDDINGS_PARQUET_PATH}"
        )
    logger.info("Loading item_embeddings as HF Dataset")
    return DatasetDict(
        {"train": Dataset.from_parquet(str(ITEM_EMBEDDINGS_PARQUET_PATH))}
    )


def log_summary(
    repo_id: str, interactions: DatasetDict, embeddings: DatasetDict
) -> None:
    logger.info("=== Will push to %s ===", repo_id)
    logger.info("-- interactions --")
    for name, ds in interactions.items():
        logger.info("   %-32s %12d rows", name, len(ds))
    logger.info("-- item_embeddings --")
    for name, ds in embeddings.items():
        logger.info("   %-32s %12d rows", name, len(ds))


def push_configs(
    interactions: DatasetDict,
    embeddings: DatasetDict,
    repo_id: str,
    private: bool,
) -> None:
    logger.info("Pushing interactions config to %s ...", repo_id)
    interactions.push_to_hub(
        repo_id, config_name="interactions", private=private
    )
    logger.info("Pushing item_embeddings config to %s ...", repo_id)
    embeddings.push_to_hub(
        repo_id, config_name="item_embeddings", private=private
    )


def upload_extra_files(api: HfApi, repo_id: str) -> None:
    for src in METADATA_FILES:
        if not src.exists():
            raise FileNotFoundError(f"Metadata file missing: {src}")
        dst = f"metadata/{src.name}"
        logger.info("Uploading %s -> %s", src.name, dst)
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"stage8: upload {src.name}",
        )
    if not README_LOCAL_PATH.exists():
        raise FileNotFoundError(f"README.md missing: {README_LOCAL_PATH}")
    logger.info("Uploading curated README.md (overwrites auto-generated)")
    api.upload_file(
        path_or_fileobj=str(README_LOCAL_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="stage8: curated README with configs and limitations",
    )


def verify_push(repo_id: str) -> None:
    logger.info("Streaming first row of interactions.train for verification")
    ds = load_dataset(repo_id, "interactions", split="train", streaming=True)
    first = next(iter(ds))
    missing = {"uid", "timestamp", "item_id"} - set(first.keys())
    if missing:
        raise RuntimeError(f"interactions.train missing columns: {missing}")
    logger.info(
        "interactions.train OK (uid=%s, item_id=%s)",
        first["uid"],
        first["item_id"],
    )

    logger.info("Streaming first row of item_embeddings.train for verification")
    ds_emb = load_dataset(
        repo_id, "item_embeddings", split="train", streaming=True
    )
    first_emb = next(iter(ds_emb))
    missing_e = {"item_id", "embed", "normalized_embed", "popularity"} - set(
        first_emb.keys()
    )
    if missing_e:
        raise RuntimeError(
            f"item_embeddings.train missing columns: {missing_e}"
        )
    emb_dim = len(first_emb["embed"])
    if emb_dim != EMBEDDING_DIM_EXPECTED:
        raise RuntimeError(f"embed dim {emb_dim} != {EMBEDDING_DIM_EXPECTED}")
    logger.info(
        "item_embeddings.train OK (item_id=%s, embed dim=%d)",
        first_emb["item_id"],
        emb_dim,
    )


def write_report(
    repo_id: str,
    interactions: DatasetDict,
    embeddings: DatasetDict,
    private: bool,
    dry_run: bool,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        "# 10 — HuggingFace Hub publish",
        "",
        f"- **Timestamp (UTC)**: `{ts}`",
        f"- **Repo**: `{repo_id}`",
        f"- **Visibility**: `{'private' if private else 'public'}`",
        f"- **Mode**: `{'dry-run' if dry_run else 'push'}`",
        f"- **URL**: https://huggingface.co/datasets/{repo_id}",
        "",
        "## Config `interactions`",
        "",
        "| Split | Rows |",
        "| :--- | ---: |",
    ]
    for name, ds in interactions.items():
        lines.append(f"| `{name}` | {len(ds):,} |")

    lines.extend(
        [
            "",
            "## Config `item_embeddings`",
            "",
            "| Split | Rows |",
            "| :--- | ---: |",
        ]
    )
    for name, ds in embeddings.items():
        lines.append(f"| `{name}` | {len(ds):,} |")

    size_mb = ITEM_EMBEDDINGS_PARQUET_PATH.stat().st_size / 1024**2
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `data/processed/item_embeddings.parquet` — {size_mb:,.1f} MB",
            f"- `metadata/{SPLITS_METADATA_PATH.name}`",
            f"- `metadata/{FILTER_STATS_PATH.name}`",
            f"- `metadata/{RAW_STATS_PATH.name}`",
            "- `README.md`",
            "",
        ]
    )

    HUB_PUBLISH_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", HUB_PUBLISH_REPORT)


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.dry_run:
        try:
            username = auth_check()
        except RuntimeError as err:
            logger.warning(
                "HF auth unavailable (%s) — dry run continues with placeholder username",
                err,
            )
            username = "<unauthenticated>"
    else:
        username = auth_check()
    repo_id = f"{username}/{args.repo_name}"
    logger.info(
        "Target: %s (private=%s, dry_run=%s)",
        repo_id,
        args.private,
        args.dry_run,
    )

    if args.skip_embeddings_extract and ITEM_EMBEDDINGS_PARQUET_PATH.exists():
        logger.info("Reusing existing %s", ITEM_EMBEDDINGS_PARQUET_PATH)
    else:
        df = extract_item_embeddings()
        write_item_embeddings(df)

    interactions = build_interactions_dataset()
    embeddings = build_item_embeddings_dataset()
    log_summary(repo_id, interactions, embeddings)

    if args.dry_run:
        logger.info("Dry run: skipping push, upload, and verify")
        write_report(
            repo_id, interactions, embeddings, args.private, dry_run=True
        )
        logger.info("Dry run done.")
        return

    api = HfApi()
    logger.info("Ensuring repo exists: %s", repo_id)
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    push_configs(interactions, embeddings, repo_id, args.private)
    upload_extra_files(api, repo_id)
    verify_push(repo_id)
    write_report(
        repo_id, interactions, embeddings, args.private, dry_run=False
    )
    logger.info("Done. URL: https://huggingface.co/datasets/%s", repo_id)


if __name__ == "__main__":
    main()
