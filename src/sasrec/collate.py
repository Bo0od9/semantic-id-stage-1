from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def collate_train(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
    items_flat = np.concatenate([s["items"] for s in batch])
    positives_flat = np.concatenate([s["positives"] for s in batch])
    lengths = np.fromiter((len(s["items"]) for s in batch), dtype=np.int64, count=len(batch))
    uids = np.fromiter((s["uid"] for s in batch), dtype=np.int64, count=len(batch))
    return {
        "items": torch.from_numpy(items_flat).long(),
        "positives": torch.from_numpy(positives_flat).long(),
        "lengths": torch.from_numpy(lengths).long(),
        "uids": torch.from_numpy(uids).long(),
    }


def collate_eval(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
    items_flat = np.concatenate([s["items"] for s in batch])
    lengths = np.fromiter((len(s["items"]) for s in batch), dtype=np.int64, count=len(batch))
    uids = np.fromiter((s["uid"] for s in batch), dtype=np.int64, count=len(batch))
    return {
        "items": torch.from_numpy(items_flat).long(),
        "lengths": torch.from_numpy(lengths).long(),
        "uids": torch.from_numpy(uids).long(),
    }
