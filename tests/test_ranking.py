from __future__ import annotations

import polars as pl
import pytest
import torch

from src.metrics.ranking import Embeddings, Targets, rank_items


def test_rank_items_matches_brute_force():
    torch.manual_seed(0)
    n_users, n_items, d = 7, 12, 4
    user_emb = torch.randn(n_users, d)
    item_emb = torch.randn(n_items, d)

    users = Embeddings(ids=torch.arange(n_users, dtype=torch.long), embeddings=user_emb)
    items = Embeddings(ids=torch.arange(n_items, dtype=torch.long), embeddings=item_emb)

    ranked = rank_items(users, items, num_items=5, batch_size=3, show_progress=False)

    brute = user_emb @ item_emb.T
    brute_top = brute.topk(5, dim=-1, sorted=True)

    # scores should match
    assert torch.allclose(ranked.scores, brute_top.values, atol=1e-5)
    # item_ids may differ only on ties (probability 0 for random float data) — compare ignoring tiebreak
    assert torch.equal(ranked.item_ids, brute_top.indices)


def test_tie_breaking_item_id_ascending():
    # All items have identical score → ranker must break by item_id ASC
    n_items, d = 6, 2
    item_emb = torch.ones(n_items, d)  # identical
    user_emb = torch.ones(1, d)
    users = Embeddings(ids=torch.tensor([0]), embeddings=user_emb)
    items = Embeddings(ids=torch.arange(n_items, dtype=torch.long), embeddings=item_emb)

    ranked = rank_items(users, items, num_items=n_items, show_progress=False)
    assert ranked.item_ids[0].tolist() == list(range(n_items)), (
        f"tie-break should give ASC item_ids, got {ranked.item_ids[0].tolist()}"
    )


def test_embeddings_sorts_by_id():
    ids = torch.tensor([3, 1, 2], dtype=torch.long)
    emb = torch.tensor([[3.0], [1.0], [2.0]])
    e = Embeddings(ids=ids, embeddings=emb)
    assert e.ids.tolist() == [1, 2, 3]
    assert e.embeddings.flatten().tolist() == [1.0, 2.0, 3.0]


def test_rank_items_k_equals_n_items():
    n_users, n_items, d = 3, 5, 2
    user_emb = torch.randn(n_users, d)
    item_emb = torch.randn(n_items, d)
    users = Embeddings(ids=torch.arange(n_users, dtype=torch.long), embeddings=user_emb)
    items = Embeddings(ids=torch.arange(n_items, dtype=torch.long), embeddings=item_emb)

    ranked = rank_items(users, items, num_items=n_items, show_progress=False)
    assert ranked.item_ids.shape == (n_users, n_items)
    for u in range(n_users):
        assert sorted(ranked.item_ids[u].tolist()) == list(range(n_items))


def test_rank_items_num_items_exceeds_catalog_asserts():
    n_users, n_items, d = 2, 3, 2
    users = Embeddings(ids=torch.arange(n_users, dtype=torch.long), embeddings=torch.randn(n_users, d))
    items = Embeddings(ids=torch.arange(n_items, dtype=torch.long), embeddings=torch.randn(n_items, d))
    with pytest.raises(AssertionError):
        rank_items(users, items, num_items=n_items + 1, show_progress=False)


def test_targets_from_sequential_groups_by_uid():
    df = pl.DataFrame(
        {
            "uid": [0, 0, 1, 1, 1, 2],
            "item_id": [10, 20, 30, 40, 30, 50],
        }
    )
    targets = Targets.from_sequential(df, device="cpu")
    assert targets.user_ids.tolist() == [0, 1, 2]
    assert sorted(targets.item_ids[0].tolist()) == [10, 20]
    assert sorted(targets.item_ids[1].tolist()) == [30, 40]  # dedup of 30
    assert targets.item_ids[2].tolist() == [50]
