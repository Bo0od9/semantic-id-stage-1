"""Stage 9: build dense item_id map from train.parquet.

Reads `data/processed/train.parquet`, enumerates unique `item_id` values,
sorts them ascending and writes `artifacts/item_id_map.json` with:

    {"n_items": N, "dense_to_raw": [raw_id_0, raw_id_1, ..., raw_id_{N-1}]}

The map is fixed once on the train split and reused by every downstream
script — cold-start filter (stage 5) guarantees val/test item ids ⊆ train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    ITEM_ID_MAP_PATH,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)


def main() -> None:
    assert TRAIN_PARQUET.exists(), f"Missing {TRAIN_PARQUET}"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_items = (
        pl.scan_parquet(TRAIN_PARQUET).select("item_id").unique().sort("item_id").collect()
    )
    raw_ids = train_items["item_id"].to_list()
    n_items = len(raw_ids)

    for name, path in [("val", VAL_PARQUET), ("test", TEST_PARQUET)]:
        split_items = set(
            pl.scan_parquet(path).select("item_id").unique().collect()["item_id"].to_list()
        )
        train_set = set(raw_ids)
        missing = split_items - train_set
        assert not missing, (
            f"{name} contains {len(missing)} items absent from train "
            f"(first: {sorted(missing)[:5]})"
        )

    payload = {"n_items": n_items, "dense_to_raw": raw_ids}
    with open(ITEM_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"wrote {ITEM_ID_MAP_PATH} (n_items={n_items})")


if __name__ == "__main__":
    main()
