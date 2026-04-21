from __future__ import annotations

import torch

from ..metrics.ranking import Ranked


def rank_mostpop(
    popularity: torch.Tensor,
    user_ids: torch.Tensor,
    k: int,
) -> Ranked:
    """Same top-K popular items for every user.

    ``popularity[i]`` is the count for dense item id ``i``; since the popularity
    tensor is indexed by dense id, a stable descending argsort breaks ties by
    item_id ascending — matching eval_protocol §6.
    """
    assert popularity.dim() == 1
    n_items = popularity.shape[0]
    assert k <= n_items, f"k={k} exceeds catalog size {n_items}"

    sort_idx = torch.argsort(popularity, descending=True, stable=True)
    top_ids = sort_idx[:k].to(device=user_ids.device, dtype=torch.long)
    top_scores = popularity[sort_idx[:k]].to(device=user_ids.device, dtype=torch.float32)

    n_users = user_ids.shape[0]
    item_ids_b = top_ids.unsqueeze(0).expand(n_users, k).contiguous()
    scores_b = top_scores.unsqueeze(0).expand(n_users, k).contiguous()

    return Ranked(
        user_ids=user_ids.to(torch.long),
        item_ids=item_ids_b,
        scores=scores_b,
        num_item_ids=n_items,
    )
