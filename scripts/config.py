"""Central configuration for the data preparation pipeline.

All paths are absolute and derived from the repository root so scripts can be
invoked from any working directory. All constants listed here are the single
source of truth for subsequent stages — do not hardcode the same values
elsewhere.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
RAW_CACHE_DIR: Path = DATA_DIR / "raw_cache"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = REPO_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

YAMBDA_REPO: str = "yandex/yambda"
YAMBDA_SIZE: str = "50m"
YAMBDA_FORMAT: str = "flat"
YAMBDA_EVENT: str = "listens"

LISTENS_PARQUET_REL: str = f"{YAMBDA_FORMAT}/{YAMBDA_SIZE}/{YAMBDA_EVENT}.parquet"
EMBEDDINGS_PARQUET_REL: str = "embeddings.parquet"

# Planned filter thresholds. Applied on later stages, recorded here for
# single-source-of-truth semantics.
PLAYED_RATIO_MIN: int = 50
MIN_USER_INTERACTIONS: int = 5
MIN_ITEM_INTERACTIONS: int = 5
LOST_HISTORY_HEAVY_THRESHOLD: float = 0.80

RANDOM_SEED: int = 42

RAW_STATS_PATH: Path = DATA_DIR / "raw_stats.json"
FILTER_STATS_PATH: Path = DATA_DIR / "filter_stats.json"
SPLITS_METADATA_PATH: Path = DATA_DIR / "splits_metadata.json"
LISTENS_STAGE2_PATH: Path = INTERIM_DIR / "listens_stage2.parquet"
LISTENS_STAGE3_PATH: Path = INTERIM_DIR / "listens_stage3.parquet"
DATASET_OVERVIEW_REPORT: Path = REPORTS_DIR / "01_dataset_overview.md"
EMBEDDING_FILTER_REPORT: Path = REPORTS_DIR / "02_embedding_filter.md"
DEDUP_AND_FILTER_REPORT: Path = REPORTS_DIR / "03_dedup_and_filter.md"
USER_HISTORY_REPORT: Path = REPORTS_DIR / "04_user_history_distribution.md"
ITEM_POPULARITY_REPORT: Path = REPORTS_DIR / "05_item_popularity.md"
TEMPORAL_ANALYSIS_REPORT: Path = REPORTS_DIR / "06_temporal_analysis.md"
TEMPORAL_SPLIT_REPORT: Path = REPORTS_DIR / "07_temporal_split.md"
SUBSAMPLES_REPORT: Path = REPORTS_DIR / "08_subsamples.md"
EMBEDDINGS_REPORT: Path = REPORTS_DIR / "09_embeddings_analysis.md"
HUB_PUBLISH_REPORT: Path = REPORTS_DIR / "10_hub_publish.md"

# Stage 7: item embeddings sanity check.
EMBEDDING_DIM_EXPECTED: int = 128
EMBEDDING_NN_SAMPLE_SIZE: int = 20
EMBEDDING_NN_TOP_K: int = 5
EMBEDDING_PCA_HTML_SAMPLE_SIZE: int = 30_000
EMBEDDING_PCA_HTML_TOP_K: int = 5
EMBEDDINGS_PCA_FIG_STEM: str = "fig_09_embeddings_pca"
EMBEDDINGS_NORMS_EMBED_STEM: str = "fig_09_embedding_norms_embed"
EMBEDDINGS_NORMS_NORMALIZED_STEM: str = "fig_09_embedding_norms_normalized"
EMBEDDINGS_PCA_HTML_PATH: Path = FIGURES_DIR / "fig_09_embeddings_pca.html"
EMBEDDINGS_PCA_3D_HTML_PATH: Path = FIGURES_DIR / "fig_09_embeddings_pca_3d.html"

TRAIN_PARQUET_PATH: Path = PROCESSED_DIR / "train.parquet"
VAL_PARQUET_PATH: Path = PROCESSED_DIR / "val.parquet"
TEST_PARQUET_PATH: Path = PROCESSED_DIR / "test.parquet"
ITEM_EMBEDDINGS_PARQUET_PATH: Path = PROCESSED_DIR / "item_embeddings.parquet"

# Stage 6: nested user-level subsamples of (train, val, test).
SUBSAMPLE_10PCT_FRACTION: float = 0.10
SUBSAMPLE_1PCT_FRACTION: float = 0.01
SUBSAMPLE_MIN_USERS: int = 100
SUBSAMPLE_10PCT_MIN_USERS_GUARD: int = 800
SUBSAMPLE_1PCT_MIN_USERS_GUARD: int = 100

TRAIN_SUBSAMPLE_10PCT_PARQUET_PATH: Path = PROCESSED_DIR / "train_subsample_10pct.parquet"
VAL_SUBSAMPLE_10PCT_PARQUET_PATH: Path = PROCESSED_DIR / "val_subsample_10pct.parquet"
TEST_SUBSAMPLE_10PCT_PARQUET_PATH: Path = PROCESSED_DIR / "test_subsample_10pct.parquet"
TRAIN_SUBSAMPLE_1PCT_PARQUET_PATH: Path = PROCESSED_DIR / "train_subsample_1pct.parquet"
VAL_SUBSAMPLE_1PCT_PARQUET_PATH: Path = PROCESSED_DIR / "val_subsample_1pct.parquet"
TEST_SUBSAMPLE_1PCT_PARQUET_PATH: Path = PROCESSED_DIR / "test_subsample_1pct.parquet"

# Stage 8: HuggingFace Hub publish.
HF_REPO_NAME: str = "yambda-semantic-user-ids"
HF_REPO_DEFAULT_PRIVATE: bool = True
