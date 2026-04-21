"""Evaluation loop: compute Recall/NDCG/HitRate/Coverage + per-user primary."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import ItemIdMap, build_targets, load_user_ids
from ..metrics.metrics import calc_metrics, per_user_primary
from ..metrics.ranking import Embeddings, Targets, rank_items
from ..utils.seed import free_device_memory
from .dataset import EvalStateDataset, load_prefix_sequences
from .extract import extract_item_matrix, infer_user_vectors
from .model import SASRec


@dataclass
class EvalContext:
    """Pre-loaded data for one eval split. Reuse across all val/test evaluations."""

    prefix_sequences: dict[int, np.ndarray]
    eval_users: set[int]
    targets: Targets

    @classmethod
    def build(
        cls,
        item_id_map: ItemIdMap,
        split_set: str,
        eval_split: str,
    ) -> "EvalContext":
        prefix = load_prefix_sequences(split_set, eval_split, item_id_map)
        users = set(load_user_ids(split_set, eval_split).cpu().numpy().tolist())
        targets = build_targets(split_set, eval_split, item_id_map, device="cpu")
        return cls(prefix_sequences=prefix, eval_users=users, targets=targets)


def evaluate_with_context(
    model: SASRec,
    item_id_map: ItemIdMap,
    ctx: EvalContext,
    *,
    max_seq_len: int,
    batch_size: int,
    device: torch.device | str,
    metric_names: list[str],
    primary_k: int,
    show_progress: bool = False,
) -> dict[str, object]:
    """Evaluate with pre-loaded EvalContext.

    Inference runs on ``device``; ranking/metrics on CPU to avoid MPS
    allocator fragmentation (see original OOM investigation).
    """
    ds = EvalStateDataset(
        ctx.prefix_sequences, max_seq_len=max_seq_len, user_ids_filter=ctx.eval_users
    )
    uids, z = infer_user_vectors(model, ds, batch_size=batch_size, device=device)
    free_device_memory(device)

    z_t = torch.from_numpy(z)
    user_ids_t = torch.from_numpy(uids).to(dtype=torch.long)
    users_emb = Embeddings(ids=user_ids_t, embeddings=z_t)

    item_matrix = extract_item_matrix(model, device=device)
    free_device_memory(device)
    item_matrix_t = torch.from_numpy(item_matrix)
    item_ids_dense = torch.arange(item_id_map.n_items, dtype=torch.long)
    items_emb = Embeddings(ids=item_ids_dense, embeddings=item_matrix_t)

    max_k = max(int(m.split("@")[1]) for m in metric_names)
    ranked = rank_items(users_emb, items_emb, num_items=max_k, show_progress=show_progress)

    metrics = calc_metrics(ranked, ctx.targets, metric_names, show_progress=show_progress)
    pu = per_user_primary(ranked, ctx.targets, k=primary_k, show_progress=False)

    out = {
        "metrics": metrics,
        "per_user": {
            "user_ids": pu["user_ids"].numpy().astype(np.int64),
            "recall": pu["recall"].numpy().astype(np.float32),
            "ndcg": pu["ndcg"].numpy().astype(np.float32),
        },
    }

    del ranked, users_emb, items_emb, item_matrix_t, z_t, item_matrix
    gc.collect()
    free_device_memory(device)
    return out


def evaluate_split(
    model: SASRec,
    item_id_map: ItemIdMap,
    split_set: str,
    eval_split: str,
    *,
    max_seq_len: int,
    batch_size: int,
    device: torch.device | str,
    metric_names: list[str],
    primary_k: int,
    show_progress: bool = False,
) -> dict[str, object]:
    """One-shot evaluation (loads context fresh). Use ``evaluate_with_context``
    with a pre-built ``EvalContext`` inside training loops to avoid re-reading
    parquets between epochs.
    """
    ctx = EvalContext.build(item_id_map, split_set, eval_split)
    return evaluate_with_context(
        model,
        item_id_map,
        ctx,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
        metric_names=metric_names,
        primary_k=primary_k,
        show_progress=show_progress,
    )
