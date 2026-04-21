from __future__ import annotations

import pytest
import torch

from src.baselines.popularity import rank_mostpop
from src.baselines.random_rec import rank_random
from src.data import (
    ITEM_EMBEDDINGS_PARQUET,
    build_targets,
    load_item_id_map,
    load_popularity,
    load_user_ids,
)
from src.metrics.metrics import calc_metrics


SPLIT_SET = "subsample_1pct"
K = [10, 100]


def test_mostpop_same_topk_for_all_users():
    pop = torch.tensor([10, 50, 5, 30, 20], dtype=torch.long)
    users = torch.tensor([0, 1, 7], dtype=torch.long)
    ranked = rank_mostpop(pop, users, k=3)
    assert ranked.item_ids.shape == (3, 3)
    for u in range(3):
        assert ranked.item_ids[u].tolist() == [1, 3, 4], ranked.item_ids[u].tolist()


def test_mostpop_tie_breaks_by_item_id_asc():
    pop = torch.tensor([7, 7, 7, 3], dtype=torch.long)
    users = torch.tensor([0], dtype=torch.long)
    ranked = rank_mostpop(pop, users, k=3)
    assert ranked.item_ids[0].tolist() == [0, 1, 2]


def test_mostpop_preserves_int64_precision_above_float32_limit():
    # Counts above 2^24 collide in float32; sort must operate in int64 to keep
    # tie-breaking honest.
    pop = torch.tensor([20_000_000, 20_000_001, 20_000_002], dtype=torch.long)
    users = torch.tensor([0], dtype=torch.long)
    ranked = rank_mostpop(pop, users, k=3)
    assert ranked.item_ids[0].tolist() == [2, 1, 0], ranked.item_ids[0].tolist()


def test_mostpop_k_equals_n_items():
    pop = torch.tensor([10, 50, 5, 30], dtype=torch.long)
    users = torch.tensor([0], dtype=torch.long)
    ranked = rank_mostpop(pop, users, k=4)
    assert ranked.item_ids[0].tolist() == [1, 3, 0, 2]


def test_mostpop_k_greater_than_n_items_asserts():
    pop = torch.tensor([10, 50, 5], dtype=torch.long)
    users = torch.tensor([0], dtype=torch.long)
    with pytest.raises(AssertionError):
        rank_mostpop(pop, users, k=4)


def test_random_k_equals_n_items():
    users = torch.arange(2, dtype=torch.long)
    ranked = rank_random(users, n_items=5, k=5, seed=42)
    for u in range(2):
        assert sorted(ranked.item_ids[u].tolist()) == [0, 1, 2, 3, 4]


def test_random_k_greater_than_n_items_asserts():
    users = torch.arange(2, dtype=torch.long)
    with pytest.raises(AssertionError):
        rank_random(users, n_items=3, k=5, seed=42)


def test_random_uniform_recall_close_to_k_over_n():
    m = load_item_id_map()
    users = load_user_ids(SPLIT_SET, "test")
    targets = build_targets(SPLIT_SET, "test", m)
    ranked = rank_random(users, n_items=m.n_items, k=100, seed=42)
    out = calc_metrics(ranked, targets, ["recall@10", "hitrate@100"], show_progress=False)
    # Recall@10 for uniform random ≈ 10 / n_items × mean(|target_capped|)
    # On 1% subsample that is ~10/260927 ≈ 3.8e-5. Allow generous bound.
    assert out["recall"][10] < 0.01, f"Random Recall@10 too high: {out['recall'][10]}"
    assert out["hitrate"][100] < 0.25, f"Random HitRate@100 too high: {out['hitrate'][100]}"


def test_mostpop_dominates_random_on_recall_100():
    m = load_item_id_map()
    pop = load_popularity(ITEM_EMBEDDINGS_PARQUET, m)
    users = load_user_ids(SPLIT_SET, "test")
    targets = build_targets(SPLIT_SET, "test", m)

    r_pop = rank_mostpop(pop, users, k=100)
    r_rnd = rank_random(users, n_items=m.n_items, k=100, seed=42)

    pop_metrics = calc_metrics(r_pop, targets, ["recall@100"], show_progress=False)
    rnd_metrics = calc_metrics(r_rnd, targets, ["recall@100"], show_progress=False)

    assert pop_metrics["recall"][100] > 5 * max(rnd_metrics["recall"][100], 1e-6), (
        f"MostPop should beat Random by ≥5× on Recall@100, "
        f"got MostPop={pop_metrics['recall'][100]} vs Random={rnd_metrics['recall'][100]}"
    )
