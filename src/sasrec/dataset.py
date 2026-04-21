"""SASRec datasets: training sequences and A2 eval prefixes.

Conventions:
- Every tensor / array of ``item_id`` values is in **dense** space ``[0, n_items)``.
  The ``SASRec`` model shifts ids to ``[1, n_items]`` internally (row 0 is padding).
- ``TrainSequenceDataset`` emits next-item pairs: ``items[:-1]`` → input,
  ``items[1:]`` → positives, both clipped to the last ``max_seq_len`` events.
- ``EvalStateDataset`` emits a single per-user prefix for A2 frozen state:
  val prefix = train interactions < ``T_val``; test prefix = (train ∪ val)
  interactions < ``T_test``. A data-leak assertion is enforced on load.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch

from ..data.dataset import ItemIdMap
from ..data.paths import SPLITS_METADATA_PATH, resolve_split_parquet


def load_temporal_cutoffs(path: str | Path = SPLITS_METADATA_PATH) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    cutoffs = meta["temporal_cutoffs"]
    return {
        "T_val": int(cutoffs["T_val_seconds"]),
        "T_test": int(cutoffs["T_test_seconds"]),
    }


def load_user_sequences(
    parquet_paths: Iterable[str | Path],
    item_id_map: ItemIdMap,
) -> dict[int, np.ndarray]:
    """Load flat interactions, group per uid, sort by timestamp ASC, remap to dense.

    Order is guaranteed via ``pl.col("item_id").sort_by("timestamp")`` *inside*
    ``agg`` — a pre-sort on the outer LazyFrame is not preserved by ``group_by``
    per Polars semantics.
    """
    paths = list(parquet_paths)
    lfs = [pl.scan_parquet(p).select("uid", "timestamp", "item_id") for p in paths]
    df = pl.concat(lfs) if len(lfs) > 1 else lfs[0]
    grouped = (
        df.group_by("uid", maintain_order=False)
        .agg(pl.col("item_id").sort_by("timestamp"))
        .sort("uid")
        .collect()
    )
    sequences: dict[int, np.ndarray] = {}
    uids = grouped["uid"].to_list()
    items_lists = grouped["item_id"].to_list()
    for uid, raw_items in zip(uids, items_lists, strict=True):
        raw = np.asarray(raw_items, dtype=np.int64)
        dense = item_id_map.to_dense(raw)
        sequences[int(uid)] = dense
    return sequences


def load_prefix_sequences(
    split_set: str,
    eval_split: Literal["val", "test"],
    item_id_map: ItemIdMap,
) -> dict[int, np.ndarray]:
    """Build A2 prefix sequences for val/test evaluation.

    - ``val``: prefix = train interactions < ``T_val``.
    - ``test``: prefix = (train ∪ val) interactions < ``T_test``.

    Raises on data-leak (max timestamp in prefix ≥ cutoff).
    """
    cutoffs = load_temporal_cutoffs()
    if eval_split == "val":
        paths = [resolve_split_parquet(split_set, "train")]
        cutoff = cutoffs["T_val"]
    elif eval_split == "test":
        paths = [resolve_split_parquet(split_set, "train"), resolve_split_parquet(split_set, "val")]
        cutoff = cutoffs["T_test"]
    else:
        raise ValueError(f"Unknown eval_split {eval_split!r}; expected val|test")

    for p in paths:
        max_ts = (
            pl.scan_parquet(p)
            .select(pl.col("timestamp").max().alias("mx"))
            .collect()["mx"][0]
        )
        assert int(max_ts) < cutoff, (
            f"Data leak: prefix parquet {p} has max timestamp {max_ts} ≥ cutoff {cutoff} "
            f"for eval_split={eval_split!r}"
        )

    return load_user_sequences(paths, item_id_map)


class TrainSequenceDataset(torch.utils.data.Dataset):
    """Per-user causal next-item training samples.

    For a user with chronological dense history ``seq`` of length ``L``:
    ``items = seq[:-1][-max_seq_len:]``, ``positives = seq[1:][-max_seq_len:]``.
    Users with ``L < min_len`` are dropped.
    """

    def __init__(
        self,
        sequences: dict[int, np.ndarray],
        max_seq_len: int,
        min_len: int = 2,
    ) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.user_ids = sorted(uid for uid, s in sequences.items() if s.shape[0] >= min_len)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        uid = self.user_ids[idx]
        seq = self.sequences[uid]
        items = seq[:-1][-self.max_seq_len :]
        positives = seq[1:][-self.max_seq_len :]
        return {"uid": uid, "items": items, "positives": positives}


class EvalStateDataset(torch.utils.data.Dataset):
    """One sample per eligible eval user: dense prefix clipped to ``max_seq_len``."""

    def __init__(
        self,
        prefix_sequences: dict[int, np.ndarray],
        max_seq_len: int,
        user_ids_filter: Iterable[int] | None = None,
    ) -> None:
        self.prefix_sequences = prefix_sequences
        self.max_seq_len = max_seq_len
        keys = set(prefix_sequences.keys())
        if user_ids_filter is not None:
            keys &= set(int(u) for u in user_ids_filter)
        self.user_ids = sorted(u for u in keys if prefix_sequences[u].shape[0] >= 1)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        uid = self.user_ids[idx]
        prefix = self.prefix_sequences[uid][-self.max_seq_len :]
        return {"uid": uid, "items": prefix}
