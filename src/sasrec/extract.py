"""Extract A2 user vectors and scoring item matrix after SASRec training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import ItemIdMap, load_user_ids
from ..data.paths import resolve_split_parquet
from ..utils.seed import free_device_memory
from .collate import collate_eval
from .dataset import EvalStateDataset, load_prefix_sequences, load_user_sequences
from .model import SASRec


@torch.no_grad()
def infer_user_vectors(
    model: SASRec,
    dataset: EvalStateDataset,
    batch_size: int,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(user_ids, z_u)`` with users sorted ASC by id."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    use_cuda = dev.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_eval,
        shuffle=False,
        pin_memory=use_cuda,
    )
    model.eval()

    uid_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    for batch in loader:
        items = batch["items"].to(dev, non_blocking=use_cuda)
        lengths = batch["lengths"].to(dev, non_blocking=use_cuda)
        z = model.encode_full_history(items, lengths)
        uid_chunks.append(batch["uids"].numpy())
        z_chunks.append(z.detach().cpu().numpy())

    uids = np.concatenate(uid_chunks).astype(np.int64)
    vecs = np.concatenate(z_chunks, axis=0).astype(np.float32)
    assert np.all(uids[:-1] < uids[1:]), "user_ids from EvalStateDataset must be sorted ASC"
    return uids, vecs


@torch.no_grad()
def extract_item_matrix(model: SASRec, device: torch.device | str) -> np.ndarray:
    model.eval()
    mat = model.item_matrix().detach().to(device).cpu().numpy().astype(np.float32)
    return mat


def extract_and_save(
    model: SASRec,
    item_id_map: ItemIdMap,
    split_set: str,
    out_dir: Path,
    max_seq_len: int,
    batch_size: int,
    device: torch.device | str,
) -> None:
    """Build train/val/test z_u and item_matrix, save into ``out_dir``.

    For SASRec downstream use (RQ-VAE, fusion): we produce one z_u per user
    per split. ``train`` prefix = user's full train history (same as in loss);
    ``val`` / ``test`` prefixes follow A2 with their respective cutoffs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict[int, np.ndarray]] = {
        "train": load_user_sequences(
            [resolve_split_parquet(split_set, "train")], item_id_map
        ),
        "val": load_prefix_sequences(split_set, "val", item_id_map),
        "test": load_prefix_sequences(split_set, "test", item_id_map),
    }

    for split_name, prefix_dict in splits.items():
        eval_users = set(
            load_user_ids(split_set, split_name).cpu().numpy().tolist()
        )
        ds = EvalStateDataset(prefix_dict, max_seq_len=max_seq_len, user_ids_filter=eval_users)
        uids, vecs = infer_user_vectors(model, ds, batch_size=batch_size, device=device)
        np.save(out_dir / f"{split_name}_user_ids.npy", uids)
        np.save(out_dir / f"{split_name}_vecs.npy", vecs)
        free_device_memory(device)

    item_matrix = extract_item_matrix(model, device=device)
    np.save(out_dir / "item_matrix.npy", item_matrix)
    free_device_memory(device)
