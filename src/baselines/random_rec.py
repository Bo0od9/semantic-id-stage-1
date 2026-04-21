from __future__ import annotations

import torch

from ..metrics.ranking import Ranked


def rank_random(
    user_ids: torch.Tensor,
    n_items: int,
    k: int,
    seed: int,
    batch_size: int = 256,
) -> Ranked:
    """Per-user uniform top-K without replacement.

    Generates random scores per user on CPU (seeded), then ``topk`` picks a
    uniform k-subset without replacement (ties have probability 0).
    """
    assert k <= n_items
    device = user_ids.device
    n_users = user_ids.shape[0]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    item_ids_buf = torch.empty((n_users, k), dtype=torch.long)
    scores_buf = torch.empty((n_users, k), dtype=torch.float32)

    for s in range(0, n_users, batch_size):
        e = min(s + batch_size, n_users)
        rand = torch.rand(e - s, n_items, generator=gen)
        topk = rand.topk(k, dim=-1, sorted=True)
        item_ids_buf[s:e] = topk.indices
        scores_buf[s:e] = topk.values

    return Ranked(
        user_ids=user_ids.to(torch.long),
        item_ids=item_ids_buf.to(device=device, dtype=torch.long),
        scores=scores_buf.to(device=device),
        num_item_ids=n_items,
    )
