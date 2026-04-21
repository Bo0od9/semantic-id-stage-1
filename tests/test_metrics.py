from __future__ import annotations

import math

import torch

from src.metrics.metrics import (
    Coverage,
    DCG,
    HitRate,
    NDCG,
    Recall,
    calc_metrics,
    create_target_mask,
    per_user_primary,
)
from src.metrics.ranking import Ranked, Targets


def _make_case(rank_item_ids: list[list[int]], target_item_ids: list[list[int]], n_items: int):
    user_ids = torch.arange(len(rank_item_ids), dtype=torch.long)
    item_ids_t = torch.tensor(rank_item_ids, dtype=torch.long)
    scores = torch.arange(item_ids_t.shape[1], 0, -1, dtype=torch.float32).expand(item_ids_t.shape).contiguous()
    ranked = Ranked(user_ids=user_ids, item_ids=item_ids_t, scores=scores, num_item_ids=n_items)
    targets = Targets(
        user_ids=user_ids.clone(),
        item_ids=[torch.tensor(x, dtype=torch.long) for x in target_item_ids],
    )
    return ranked, targets


def test_ndcg_ideal_equals_one():
    # User 0: target = {5, 7, 9}, ranked top-3 = [5, 7, 9] — ideal order
    ranked, targets = _make_case(
        rank_item_ids=[[5, 7, 9, 1, 2, 3, 4, 6, 8, 0]],
        target_item_ids=[[5, 7, 9]],
        n_items=20,
    )
    mask = create_target_mask(ranked, targets, show_progress=False)
    ndcg = NDCG()(ranked, targets, mask, ks=[3, 10])
    assert math.isclose(ndcg[3], 1.0, abs_tol=1e-6), f"NDCG@3 should be 1, got {ndcg[3]}"
    assert math.isclose(ndcg[10], 1.0, abs_tol=1e-6), f"NDCG@10 should be 1, got {ndcg[10]}"


def test_ndcg_less_than_one_for_imperfect():
    # User 0: target = {7}, ranked = [5, 7, ...] → target at rank 2 → NDCG@2 = 1/log2(3)
    ranked, targets = _make_case(
        rank_item_ids=[[5, 7, 9, 1]],
        target_item_ids=[[7]],
        n_items=20,
    )
    mask = create_target_mask(ranked, targets, show_progress=False)
    ndcg = NDCG()(ranked, targets, mask, ks=[2])
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert math.isclose(ndcg[2], expected, abs_tol=1e-6), f"NDCG@2 expected {expected}, got {ndcg[2]}"


def test_hitrate():
    ranked, targets = _make_case(
        rank_item_ids=[[5, 7, 9, 1], [0, 1, 2, 3]],
        target_item_ids=[[7], [99]],
        n_items=100,
    )
    mask = create_target_mask(ranked, targets, show_progress=False)
    hr = HitRate()(ranked, targets, mask, ks=[2, 4])
    # user 0 hits at k=2; user 1 never hits
    assert math.isclose(hr[2], 0.5, abs_tol=1e-6), f"HitRate@2 expected 0.5, got {hr[2]}"
    assert math.isclose(hr[4], 0.5, abs_tol=1e-6), f"HitRate@4 expected 0.5, got {hr[4]}"


def test_recall_capping():
    # user 0: |target|=5, K=10 → denom=min(5,10)=5; hits in top-10 = 5 → 5/5 = 1.0
    # user 1: |target|=20, K=10 → denom=min(20,10)=10; hits in top-10 = 3 → 3/10 = 0.3
    ranked_items = [
        [0, 1, 2, 3, 4, 20, 21, 22, 23, 24],
        [100, 101, 102, 50, 51, 52, 53, 54, 99, 98],
    ]
    targets_list = [
        list(range(5)),
        list(range(20)),
    ]
    ranked, targets = _make_case(
        rank_item_ids=ranked_items,
        target_item_ids=targets_list,
        n_items=200,
    )
    mask = create_target_mask(ranked, targets, show_progress=False)
    recall = Recall()(ranked, targets, mask, ks=[10])
    # user 1: none of [100,101,102,50,51,52,53,54,99,98] ∈ range(20) → 0 hits, 0/10 = 0
    # mean = (1.0 + 0.0) / 2 = 0.5
    assert math.isclose(recall[10], 0.5, abs_tol=1e-6), f"Recall@10 expected 0.5, got {recall[10]}"


def test_coverage():
    # Union of top-3 per user over 2 users = 5 unique items / 100 total = 0.05
    ranked, targets = _make_case(
        rank_item_ids=[[0, 1, 2, 3], [0, 1, 4, 5]],
        target_item_ids=[[0], [0]],
        n_items=100,
    )
    cov = Coverage()(ranked, targets, None, ks=[3])
    # union of {0,1,2} ∪ {0,1,4} = {0,1,2,4} → 4 / 100 = 0.04
    assert math.isclose(cov[3], 0.04, abs_tol=1e-6), f"Coverage@3 expected 0.04, got {cov[3]}"


def test_dcg():
    # User 0: [5, 7, 9], target = {5, 9} → hits at positions 0 and 2
    # DCG@3 = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    ranked, targets = _make_case(
        rank_item_ids=[[5, 7, 9, 1]],
        target_item_ids=[[5, 9]],
        n_items=20,
    )
    mask = create_target_mask(ranked, targets, show_progress=False)
    dcg = DCG()(ranked, targets, mask, ks=[3])
    assert math.isclose(dcg[3], 1.5, abs_tol=1e-6), f"DCG@3 expected 1.5, got {dcg[3]}"


def test_calc_metrics_end_to_end():
    ranked, targets = _make_case(
        rank_item_ids=[[5, 7, 9, 1, 2]],
        target_item_ids=[[5, 7]],
        n_items=20,
    )
    out = calc_metrics(
        ranked, targets,
        ["recall@5", "ndcg@5", "hitrate@5", "coverage@5"],
        show_progress=False,
    )
    assert "recall" in out and 5 in out["recall"]
    assert math.isclose(out["hitrate"][5], 1.0, abs_tol=1e-6)


def test_per_user_primary_shapes():
    ranked, targets = _make_case(
        rank_item_ids=[[5, 7, 9, 1], [0, 1, 2, 3]],
        target_item_ids=[[7], [99]],
        n_items=100,
    )
    pu = per_user_primary(ranked, targets, k=2)
    assert pu["user_ids"].shape == (2,)
    assert pu["recall"].shape == (2,)
    assert pu["ndcg"].shape == (2,)
    # user 0 recall@2 = 1/1 = 1.0; user 1 = 0
    assert math.isclose(pu["recall"][0].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(pu["recall"][1].item(), 0.0, abs_tol=1e-6)


def test_target_dedup():
    # duplicates in target list should collapse via Targets.__post_init__
    user_ids = torch.tensor([0], dtype=torch.long)
    targets = Targets(
        user_ids=user_ids,
        item_ids=[torch.tensor([5, 5, 7, 7, 9], dtype=torch.long)],
    )
    assert targets.lengths.tolist() == [3], f"expected dedup to 3, got {targets.lengths.tolist()}"
