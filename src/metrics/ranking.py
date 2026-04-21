"""Ranking primitives: Embeddings, Targets, Ranked, rank_items.

Adapted from yandex/yambda benchmark code (Apache License 2.0):
https://huggingface.co/datasets/yandex/yambda/tree/main/benchmarks/yambda/evaluation/ranking.py

Modifications relative to the upstream version:
- ``rank_items`` uses a stable descending ``argsort`` over the full score
  tensor instead of ``torch.topk``. Combined with the ``Embeddings``
  post-invariant (``ids`` are sorted ascending), this yields deterministic
  tie-breaking by ``(score DESC, item_id ASC)``.
- ``Targets.__post_init__`` deduplicates each user's target item list
  (per eval_protocol §6: multi-target set semantics).
- ``batch_size`` default preserved; exposed as an argument.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property

import numpy as np
import polars as pl
import torch
from tqdm import tqdm


@dataclasses.dataclass
class Embeddings:
    ids: torch.Tensor
    embeddings: torch.Tensor

    def __post_init__(self) -> None:
        assert self.ids.dim() == 1
        assert self.embeddings.dim() == 2
        assert self.ids.shape[0] == self.embeddings.shape[0]
        assert self.ids.device == self.embeddings.device

        if not torch.all(self.ids[:-1] <= self.ids[1:]):
            order = torch.argsort(self.ids, descending=False)
            self.embeddings = self.embeddings[order, :]
            self.ids = self.ids[order]

        assert torch.all(self.ids[:-1] < self.ids[1:]), "ids must be unique and sorted"

    @property
    def device(self) -> torch.device:
        return self.ids.device

    def save(self, file_path: str) -> None:
        np.savez(
            file_path,
            ids=self.ids.cpu().numpy(),
            embeddings=self.embeddings.cpu().numpy(),
        )

    @classmethod
    def load(cls, file_path: str, device: torch.device | str = "cpu") -> "Embeddings":
        with np.load(file_path) as data:
            ids_np = data["ids"]
            embeddings_np = data["embeddings"]
        return cls(
            ids=torch.from_numpy(ids_np).to(device),
            embeddings=torch.from_numpy(embeddings_np).to(device),
        )


@dataclasses.dataclass
class Targets:
    user_ids: torch.Tensor
    item_ids: list[torch.Tensor]

    def __post_init__(self) -> None:
        assert len(self.item_ids) > 0
        assert self.user_ids.dim() == 1
        assert self.user_ids.shape[0] == len(self.item_ids)
        assert all(x.dim() == 1 for x in self.item_ids), "all target lists must be 1-D"

        dev = self.item_ids[0].device
        assert all(x.device == dev for x in self.item_ids), "all target lists must share device"
        assert self.user_ids.device == dev

        if not torch.all(self.user_ids[:-1] <= self.user_ids[1:]):
            order = torch.argsort(self.user_ids, descending=False)
            self.item_ids = [self.item_ids[i] for i in order.tolist()]
            self.user_ids = self.user_ids[order]

        assert torch.all(self.user_ids[:-1] < self.user_ids[1:]), "user_ids must be unique"

        self.item_ids = [torch.unique(x) for x in self.item_ids]

    @cached_property
    def lengths(self) -> torch.Tensor:
        return torch.tensor(
            [ids.shape[0] for ids in self.item_ids],
            device=self.item_ids[0].device,
        )

    def __len__(self) -> int:
        return len(self.item_ids)

    @property
    def device(self) -> torch.device:
        return self.user_ids.device

    @classmethod
    def from_sequential(
        cls,
        df: pl.LazyFrame | pl.DataFrame,
        device: torch.device | str,
    ) -> "Targets":
        grouped = (
            df.lazy()
            .select("uid", "item_id")
            .group_by("uid", maintain_order=False)
            .agg("item_id")
            .sort("uid")
            .collect()
        )
        user_ids_np = grouped["uid"].to_numpy().astype(np.int64)
        user_ids = torch.from_numpy(user_ids_np).to(device=device, dtype=torch.long)
        item_lists = [
            torch.tensor(x, device=device, dtype=torch.long)
            for x in grouped["item_id"].to_list()
        ]
        return cls(user_ids=user_ids, item_ids=item_lists)


@dataclasses.dataclass
class Ranked:
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    scores: torch.Tensor | None = None
    num_item_ids: int | None = None

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = torch.arange(
                self.item_ids.shape[1], 0, -1,
                device=self.item_ids.device, dtype=torch.float32,
            ).expand((self.user_ids.shape[0], self.item_ids.shape[1]))

        assert self.user_ids.dim() == 1
        assert self.scores.dim() == 2
        assert self.scores.shape == self.item_ids.shape
        assert self.user_ids.shape[0] == self.scores.shape[0]
        assert self.user_ids.device == self.scores.device == self.item_ids.device
        assert torch.all(self.scores[:, :-1] >= self.scores[:, 1:]), "scores must be sorted descending"

        if not torch.all(self.user_ids[:-1] <= self.user_ids[1:]):
            order = torch.argsort(self.user_ids, descending=False)
            self.item_ids = self.item_ids[order, :]
            self.scores = self.scores[order, :]
            self.user_ids = self.user_ids[order]

    @property
    def device(self) -> torch.device:
        return self.user_ids.device


def rank_items(
    users: Embeddings,
    items: Embeddings,
    num_items: int,
    batch_size: int = 128,
    show_progress: bool = True,
) -> Ranked:
    assert users.device == items.device
    assert num_items <= items.ids.shape[0], "num_items exceeds catalog size"

    num_users = users.ids.shape[0]
    device = users.embeddings.device
    dtype = users.embeddings.dtype

    scores = torch.empty((num_users, num_items), device=device, dtype=dtype)
    item_ids = torch.empty((num_users, num_items), device=device, dtype=torch.long)

    n_batches = (num_users + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Ranking items")

    for batch_idx in iterator:
        s, e = batch_idx * batch_size, (batch_idx + 1) * batch_size
        batch_scores = users.embeddings[s:e, :] @ items.embeddings.T

        sort_idx = torch.argsort(batch_scores, dim=-1, descending=True, stable=True)
        topk_idx = sort_idx[:, :num_items]

        scores[s:e] = torch.gather(batch_scores, dim=-1, index=topk_idx)
        item_ids[s:e] = torch.gather(
            items.ids.expand(topk_idx.shape[0], items.ids.shape[0]),
            dim=-1,
            index=topk_idx,
        )

    return Ranked(
        user_ids=users.ids,
        item_ids=item_ids,
        scores=scores,
        num_item_ids=items.ids.shape[0],
    )
