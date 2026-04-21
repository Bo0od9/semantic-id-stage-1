from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..metrics.ranking import Targets
from .paths import ITEM_ID_MAP_PATH, resolve_split_parquet


@dataclasses.dataclass
class ItemIdMap:
    """Dense ↔ raw item id mapping.

    ``dense_to_raw[i]`` is the raw ``uint32`` id for dense id ``i``.
    Dense ids are contiguous in ``[0, n_items)`` with ``raw_ids`` sorted
    ascending — so ``searchsorted`` recovers dense ids from raw ids.
    """

    dense_to_raw: np.ndarray

    @property
    def n_items(self) -> int:
        return int(self.dense_to_raw.shape[0])

    def to_dense(self, raw: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.dense_to_raw, raw)
        if not np.all(self.dense_to_raw[idx] == raw):
            missing = raw[self.dense_to_raw[idx] != raw]
            raise KeyError(f"{len(missing)} raw item ids not in map, e.g. {missing[:5].tolist()}")
        return idx

    def to_raw(self, dense: np.ndarray) -> np.ndarray:
        return self.dense_to_raw[dense]


def load_item_id_map(path: str | Path = ITEM_ID_MAP_PATH) -> ItemIdMap:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    arr = np.asarray(data["dense_to_raw"], dtype=np.int64)
    assert arr.ndim == 1 and arr.shape[0] == int(data["n_items"])
    assert np.all(arr[:-1] < arr[1:]), "dense_to_raw must be strictly ascending"
    return ItemIdMap(dense_to_raw=arr)


def build_targets(
    split_set: str,
    split: str,
    item_id_map: ItemIdMap,
    device: torch.device | str = "cpu",
) -> Targets:
    """Read val/test parquet, group by uid, remap item_ids to dense, build Targets."""
    assert split in ("val", "test"), "targets are constructed for val/test only"
    path = resolve_split_parquet(split_set, split)

    grouped = (
        pl.scan_parquet(path)
        .select("uid", "item_id")
        .group_by("uid", maintain_order=False)
        .agg("item_id")
        .sort("uid")
        .collect()
    )
    user_ids_np = grouped["uid"].to_numpy().astype(np.int64)
    user_ids = torch.from_numpy(user_ids_np).to(device=device, dtype=torch.long)

    item_tensors: list[torch.Tensor] = []
    for raw_list in grouped["item_id"].to_list():
        raw_arr = np.asarray(raw_list, dtype=np.int64)
        dense_arr = item_id_map.to_dense(raw_arr)
        item_tensors.append(torch.from_numpy(dense_arr).to(device=device, dtype=torch.long))

    return Targets(user_ids=user_ids, item_ids=item_tensors)


def load_user_ids(
    split_set: str,
    split: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    path = resolve_split_parquet(split_set, split)
    uids = (
        pl.scan_parquet(path)
        .select("uid")
        .unique()
        .sort("uid")
        .collect()["uid"]
        .to_numpy()
        .astype(np.int64)
    )
    return torch.from_numpy(uids).to(device=device, dtype=torch.long)


def load_popularity(
    item_embeddings_path: str | Path,
    item_id_map: ItemIdMap,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return popularity vector of shape (n_items,) indexed by dense item_id."""
    df = pl.scan_parquet(item_embeddings_path).select("item_id", "popularity").collect()
    raw_ids = df["item_id"].to_numpy().astype(np.int64)
    pops = df["popularity"].to_numpy().astype(np.int64)

    dense = item_id_map.to_dense(raw_ids)
    pop_arr = np.zeros(item_id_map.n_items, dtype=np.int64)
    pop_arr[dense] = pops
    return torch.from_numpy(pop_arr).to(device=device, dtype=torch.long)
