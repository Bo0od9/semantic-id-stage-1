"""Ranking metrics: Recall, NDCG, HitRate, Coverage, etc.

Adapted from yandex/yambda benchmark code (Apache License 2.0):
https://huggingface.co/datasets/yandex/yambda/tree/main/benchmarks/yambda/evaluation/metrics.py

Modifications relative to the upstream version:
- ``NDCG``: uses ``ideal_target_mask`` for the denominator (upstream had a
  bug where ``ideal_dcg`` was computed from ``target_mask`` instead of
  ``ideal_target_mask``, making NDCG == 1 whenever DCG > 0).
  With the corrected ``ideal_dcg``, the ``divide()`` helper now returns
  the per-user mean ``(1/n) Σ dcg_u / idcg_u``, which is the conventional
  aggregation.
- ``HitRate`` class added (absent upstream) — reported per eval_protocol §6.
- ``REGISTERED_METRIC_FN`` extended with ``"hitrate"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from tqdm import tqdm

from .ranking import Ranked, Targets


def cut_off_ranked(ranked: Ranked, targets: Targets) -> Ranked:
    mask = torch.isin(ranked.user_ids, targets.user_ids)
    assert ranked.scores is not None
    trimmed = Ranked(
        user_ids=ranked.user_ids[mask],
        scores=ranked.scores[mask, :],
        item_ids=ranked.item_ids[mask, :],
        num_item_ids=ranked.num_item_ids,
    )
    assert trimmed.item_ids.shape[0] == len(targets), "Ranked does not cover all target user_ids"
    return trimmed


class Metric(ABC):
    @abstractmethod
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets | None,
        target_mask: torch.Tensor | None,
        ks: Iterable[int],
    ) -> dict[int, float]:
        ...


class Recall(Metric):
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets,
        target_mask: torch.Tensor,
        ks: Iterable[int],
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)
        values: dict[int, float] = {}
        num_positives = targets.lengths.to(torch.float32)
        num_positives = torch.where(num_positives == 0, torch.inf, num_positives)
        for k in ks:
            hits = target_mask[:, :k].to(torch.float32).sum(dim=-1)
            denom = torch.clamp(num_positives, max=float(k))
            values[k] = torch.mean(hits / denom).item()
        return values


class Precision(Metric):
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets | None,
        target_mask: torch.Tensor,
        ks: Iterable[int],
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)
        values: dict[int, float] = {}
        for k in ks:
            values[k] = (target_mask[:, :k].to(torch.float32).sum(dim=-1) / k).mean().item()
        return values


class HitRate(Metric):
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets | None,
        target_mask: torch.Tensor,
        ks: Iterable[int],
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)
        values: dict[int, float] = {}
        for k in ks:
            hit = (target_mask[:, :k].sum(dim=-1) > 0).to(torch.float32)
            values[k] = torch.mean(hit).item()
        return values


class MRR(Metric):
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets,
        target_mask: torch.Tensor,
        ks: Iterable[int],
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)
        values: dict[int, float] = {}
        num_positives = targets.lengths.to(torch.float32)
        for k in ks:
            first_rank = torch.argmax(target_mask[:, :k].to(torch.float32), dim=-1).to(torch.float32) + 1.0
            hit_any = target_mask[:, :k].sum(dim=-1) > 0
            rr = torch.where(hit_any, 1.0 / first_rank, torch.zeros_like(first_rank))
            rr = torch.where(num_positives == 0, torch.zeros_like(rr), rr)
            values[k] = torch.mean(rr).item()
        return values


class DCG(Metric):
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets | None,
        target_mask: torch.Tensor,
        ks: Iterable[int],
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)
        discounts = 1.0 / torch.log2(
            torch.arange(2, target_mask.shape[1] + 2, device=target_mask.device, dtype=torch.float32)
        )
        values: dict[int, float] = {}
        for k in ks:
            dcg_k = torch.sum(target_mask[:, :k] * discounts[:k], dim=1)
            values[k] = torch.mean(dcg_k).item()
        return values


class NDCG(Metric):
    def __call__(
        self,
        ranked: Ranked | None,
        targets: Targets,
        target_mask: torch.Tensor,
        ks: Iterable[int],
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        discounts = 1.0 / torch.log2(
            torch.arange(2, target_mask.shape[1] + 2, device=target_mask.device, dtype=torch.float32)
        )

        def calc_dcg(mask: torch.Tensor) -> dict[int, torch.Tensor]:
            return {k: torch.sum(mask[:, :k] * discounts[:k], dim=1) for k in ks}

        actual_dcg = calc_dcg(target_mask)

        ideal_target_mask = (
            torch.arange(target_mask.shape[1], device=target_mask.device)[None, :]
            < targets.lengths[:, None]
        ).to(torch.float32)
        assert target_mask.shape == ideal_target_mask.shape

        ideal_dcg = calc_dcg(ideal_target_mask)

        values: dict[int, float] = {}
        for k in ks:
            x, y = actual_dcg[k], ideal_dcg[k]
            per_user = torch.where(y == 0, torch.zeros_like(x), x / y)
            values[k] = per_user.mean().item()
        return values


class Coverage(Metric):
    def __init__(self, cut_off: bool = False) -> None:
        self.cut_off = cut_off

    def __call__(
        self,
        ranked: Ranked,
        targets: Targets | None,
        target_mask: torch.Tensor | None,
        ks: Iterable[int],
    ) -> dict[int, float]:
        if self.cut_off:
            assert targets is not None
            ranked = cut_off_ranked(ranked, targets)
        assert all(0 < k <= ranked.item_ids.shape[1] for k in ks)
        assert ranked.num_item_ids is not None
        values: dict[int, float] = {}
        for k in ks:
            values[k] = ranked.item_ids[:, :k].flatten().unique().shape[0] / ranked.num_item_ids
        return values


REGISTERED_METRIC_FN: dict[str, Metric] = {
    "recall": Recall(),
    "precision": Precision(),
    "hitrate": HitRate(),
    "mrr": MRR(),
    "dcg": DCG(),
    "ndcg": NDCG(),
    "coverage": Coverage(cut_off=False),
}


def _parse_metrics(metric_names: list[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for metric in metric_names:
        parts = metric.split("@")
        assert len(parts) == 2, f"Invalid metric: {metric!r}, expected name@k"
        grouped[parts[0]].append(int(parts[1]))
    return dict(grouped)


def create_target_mask(ranked: Ranked, targets: Targets, show_progress: bool = True) -> torch.Tensor:
    ranked = cut_off_ranked(ranked, targets)
    assert ranked.device == targets.device
    assert ranked.item_ids.shape[0] == len(targets)

    target_mask = ranked.item_ids.new_zeros(ranked.item_ids.shape, dtype=torch.float32)
    iterator = enumerate(targets.item_ids)
    if show_progress:
        iterator = enumerate(tqdm(targets.item_ids, desc="Target mask"))
    for i, target in iterator:
        target_mask[i, torch.isin(ranked.item_ids[i], target)] = 1.0
    return target_mask


def per_user_primary(
    ranked: Ranked,
    targets: Targets,
    k: int,
    show_progress: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-user Recall@k and NDCG@k for paired significance testing.

    Returns a dict with ``user_ids`` (aligned with ``targets`` order),
    ``recall`` and ``ndcg`` — all 1-D tensors of shape ``(n_eval_users,)``.
    """
    mask = create_target_mask(ranked, targets, show_progress=show_progress)

    num_positives = targets.lengths.to(torch.float32)
    num_positives = torch.where(num_positives == 0, torch.inf, num_positives)
    denom = torch.clamp(num_positives, max=float(k))
    recall = mask[:, :k].sum(dim=-1) / denom

    discounts = 1.0 / torch.log2(
        torch.arange(2, mask.shape[1] + 2, device=mask.device, dtype=torch.float32)
    )
    dcg = (mask[:, :k] * discounts[:k]).sum(dim=1)

    ideal = (
        torch.arange(mask.shape[1], device=mask.device)[None, :]
        < targets.lengths[:, None]
    ).to(torch.float32)
    idcg = (ideal[:, :k] * discounts[:k]).sum(dim=1)
    ndcg = torch.where(idcg == 0, torch.zeros_like(dcg), dcg / idcg)

    return {"user_ids": targets.user_ids, "recall": recall, "ndcg": ndcg}


def calc_metrics(
    ranked: Ranked,
    targets: Targets,
    metrics: list[str],
    show_progress: bool = True,
) -> dict[str, dict[int, float]]:
    grouped = _parse_metrics(metrics)
    target_mask = create_target_mask(ranked, targets, show_progress=show_progress)

    result: dict[str, dict[int, float]] = {}
    for name, ks in grouped.items():
        if name not in REGISTERED_METRIC_FN:
            raise KeyError(f"Unknown metric {name!r}; registered: {list(REGISTERED_METRIC_FN)}")
        fn: Any = REGISTERED_METRIC_FN[name]
        result[name] = fn(ranked, targets, target_mask, ks=sorted(set(ks)))
    return result
