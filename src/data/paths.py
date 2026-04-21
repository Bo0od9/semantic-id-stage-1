from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
SAVED_DIR: Path = REPO_ROOT / "saved"
RESULTS_DIR: Path = REPO_ROOT / "results"

TRAIN_PARQUET: Path = PROCESSED_DIR / "train.parquet"
VAL_PARQUET: Path = PROCESSED_DIR / "val.parquet"
TEST_PARQUET: Path = PROCESSED_DIR / "test.parquet"
ITEM_EMBEDDINGS_PARQUET: Path = PROCESSED_DIR / "item_embeddings.parquet"

ITEM_ID_MAP_PATH: Path = ARTIFACTS_DIR / "item_id_map.json"
SPLITS_METADATA_PATH: Path = DATA_DIR / "splits_metadata.json"
USER_VECTORS_DIR: Path = ARTIFACTS_DIR / "user_vectors"


_SPLIT_TEMPLATES = {
    "full": ("train.parquet", "val.parquet", "test.parquet"),
    "subsample_10pct": (
        "train_subsample_10pct.parquet",
        "val_subsample_10pct.parquet",
        "test_subsample_10pct.parquet",
    ),
    "subsample_1pct": (
        "train_subsample_1pct.parquet",
        "val_subsample_1pct.parquet",
        "test_subsample_1pct.parquet",
    ),
}


def resolve_split_parquet(split_set: str, split: str) -> Path:
    if split_set not in _SPLIT_TEMPLATES:
        raise ValueError(
            f"Unknown split_set {split_set!r}; options: {sorted(_SPLIT_TEMPLATES)}"
        )
    names = dict(zip(("train", "val", "test"), _SPLIT_TEMPLATES[split_set], strict=True))
    if split not in names:
        raise ValueError(f"Unknown split {split!r}; options: train|val|test")
    return PROCESSED_DIR / names[split]
