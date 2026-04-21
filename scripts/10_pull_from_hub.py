"""Stage 10: pull processed dataset back from HuggingFace Hub.

Reverse of stage 8. Downloads two configs (`interactions`, `item_embeddings`)
and three metadata JSONs into the local `data/` tree so that downstream
training scripts can load from `data/processed/*.parquet` as if the pipeline
had run locally.

After this script, run `python scripts/09_build_item_id_map.py` to regenerate
`artifacts/item_id_map.json` from the freshly pulled `data/processed/train.parquet`.

Usage:
    hf auth login
    python scripts/10_pull_from_hub.py                   # infer username from whoami()
    python scripts/10_pull_from_hub.py --username alice  # explicit username
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import polars as pl
from datasets import load_dataset
from huggingface_hub import hf_hub_download, whoami

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FILTER_STATS_PATH,
    HF_REPO_NAME,
    ITEM_EMBEDDINGS_PARQUET_PATH,
    PROCESSED_DIR,
    RAW_STATS_PATH,
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
)
from utils.io import setup_logging

logger = logging.getLogger(__name__)

INTERACTIONS_SPLIT_TO_LOCAL: dict[str, Path] = {
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

METADATA_REMOTE_TO_LOCAL: dict[str, Path] = {
    f"metadata/{SPLITS_METADATA_PATH.name}": SPLITS_METADATA_PATH,
    f"metadata/{FILTER_STATS_PATH.name}": FILTER_STATS_PATH,
    f"metadata/{RAW_STATS_PATH.name}": RAW_STATS_PATH,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-name", default=HF_REPO_NAME)
    p.add_argument(
        "--username",
        default=None,
        help="HF username; if omitted, resolved via whoami()",
    )
    p.add_argument(
        "--skip-interactions",
        action="store_true",
        help="Skip downloading the interactions config",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip downloading the item_embeddings config",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip downloading metadata JSONs",
    )
    return p.parse_args()


def resolve_repo_id(args: argparse.Namespace) -> str:
    if args.username:
        return f"{args.username}/{args.repo_name}"
    try:
        info = whoami()
    except Exception as err:
        raise RuntimeError(
            "HuggingFace auth missing. Run `hf auth login` or pass --username."
        ) from err
    username = info.get("name") if isinstance(info, dict) else None
    if not username:
        raise RuntimeError(f"whoami() returned no username: {info!r}")
    return f"{username}/{args.repo_name}"


def _write_dataset_to_parquet(ds, target: Path, label: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    table = ds.data.table
    df = pl.from_arrow(table)
    if not isinstance(df, pl.DataFrame):
        df = df.to_frame()
    tmp = target.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(target)
    size_mb = target.stat().st_size / 1024**2
    logger.info("%-40s %10d rows   %8.1f MB   %s", label, df.height, size_mb, target)


def pull_interactions(repo_id: str) -> None:
    logger.info("Loading interactions config from %s ...", repo_id)
    ds = load_dataset(repo_id, "interactions")
    expected = set(INTERACTIONS_SPLIT_TO_LOCAL)
    got = set(ds.keys())
    missing = expected - got
    if missing:
        raise RuntimeError(
            f"Config interactions is missing splits: {sorted(missing)}"
        )
    extra = got - expected
    if extra:
        logger.warning(
            "Config interactions has extra splits (ignored): %s", sorted(extra)
        )
    for split_name, local_path in INTERACTIONS_SPLIT_TO_LOCAL.items():
        _write_dataset_to_parquet(ds[split_name], local_path, split_name)


def pull_item_embeddings(repo_id: str) -> None:
    logger.info("Loading item_embeddings config from %s ...", repo_id)
    ds = load_dataset(repo_id, "item_embeddings")
    if "train" not in ds:
        raise RuntimeError(
            f"Config item_embeddings missing 'train' split, got {list(ds.keys())}"
        )
    _write_dataset_to_parquet(ds["train"], ITEM_EMBEDDINGS_PARQUET_PATH, "item_embeddings")


def pull_metadata(repo_id: str) -> None:
    for remote_path, local_path in METADATA_REMOTE_TO_LOCAL.items():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s → %s", remote_path, local_path)
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
        )
        shutil.copy2(cached, local_path)


def main() -> None:
    setup_logging()
    args = parse_args()
    repo_id = resolve_repo_id(args)
    logger.info("Pulling from %s into %s", repo_id, PROCESSED_DIR.parent)

    if not args.skip_interactions:
        pull_interactions(repo_id)
    if not args.skip_embeddings:
        pull_item_embeddings(repo_id)
    if not args.skip_metadata:
        pull_metadata(repo_id)

    logger.info(
        "Done. Next: run `python scripts/09_build_item_id_map.py` to regenerate "
        "artifacts/item_id_map.json from train.parquet."
    )


if __name__ == "__main__":
    main()
