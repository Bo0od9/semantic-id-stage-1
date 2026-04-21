"""Sampled softmax with log-Q correction and mixed negatives.

Implements the training loss from eval_protocol §7 / ARGUS §3.1:

    f(h_t, e_i) = ⟨h_t, e_i⟩ / exp(τ) − log Q(i)
    L = − log[ exp(f_pos) / (exp(f_pos) + Σ_{i∈N} exp(f_i)) ]

Negatives N = N_in_batch ∪ N_uniform.

- ``N_in_batch``: positives of all other positions in the batch, with a
  per-user boolean mask zeroing out same-user positions (they would be
  false negatives).
- ``N_uniform``: ``n_uniform`` ids drawn uniformly from ``[0, n_items)``
  on each step.

log-Q correction is applied only to sampled negatives (the positive is
ground truth and does not receive correction). For in-batch negatives
``q_inbatch(i) ∝ popularity(i)``; for uniform negatives ``q_uniform(i)
= 1 / n_items``. The temperature scalar ``τ`` is learnable; ``exp(τ)``
is hard-clamped to ``[min_exp_tau, max_exp_tau]`` (CLIP-style).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampledSoftmaxLoss(nn.Module):
    def __init__(
        self,
        item_encoder: nn.Module,
        n_items: int,
        popularity: torch.Tensor,
        n_uniform: int = 4096,
        init_temperature: float = 0.1,
        min_exp_tau: float = 0.01,
        max_exp_tau: float = 100.0,
        min_q: float = 1e-9,
        max_n_pos_per_step: int = 0,
    ) -> None:
        super().__init__()
        assert popularity.dim() == 1 and popularity.shape[0] == n_items
        # Store item_encoder without registering it as submodule to avoid
        # duplicate parameters in the optimizer (it already belongs to the SASRec model).
        self._item_encoder_ref = [item_encoder]
        self.n_items = n_items
        self.n_uniform = n_uniform
        self.min_exp_tau = float(min_exp_tau)
        self.max_exp_tau = float(max_exp_tau)
        # OOM-safety: если в батче случайно оказалось слишком много валидных
        # позиций, квадратичная матрица (n_pos, n_pos) in-batch logits может
        # не влезть в память девайса. При ``max_n_pos_per_step > 0`` сабсэмплим
        # valid позиции равномерно: это несмещённая оценка loss (positives и
        # negatives остаются при своих распределениях, log-Q не трогается).
        self.max_n_pos_per_step = int(max_n_pos_per_step)

        self.log_tau = nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))

        # Compute log Q_inbatch on CPU in float64 (MPS has no fp64), then store as float32.
        pop_cpu = popularity.detach().to(device="cpu", dtype=torch.float64)
        total = float(pop_cpu.sum().item())
        assert total > 0.0, "popularity sum must be positive"
        q_inbatch = (pop_cpu / total).clamp(min=min_q).to(torch.float32)
        self.register_buffer("log_q_inbatch", torch.log(q_inbatch))
        self.register_buffer(
            "log_q_uniform",
            torch.tensor(math.log(1.0 / n_items), dtype=torch.float32),
        )

    @property
    def item_encoder(self) -> nn.Module:
        return self._item_encoder_ref[0]

    def _exp_tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(self.min_exp_tau, self.max_exp_tau)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden.device
        queries = hidden[mask]
        n_pos = queries.shape[0]
        assert positives.shape[0] == n_pos, (
            f"positives ({positives.shape[0]}) must align with valid positions ({n_pos})"
        )

        lengths = mask.sum(dim=-1)
        sample_idx = torch.repeat_interleave(
            torch.arange(mask.shape[0], device=device), lengths
        )

        if self.max_n_pos_per_step > 0 and n_pos > self.max_n_pos_per_step:
            keep = torch.randperm(n_pos, device=device)[: self.max_n_pos_per_step]
            queries = queries[keep]
            positives = positives[keep]
            sample_idx = sample_idx[keep]
            n_pos = self.max_n_pos_per_step

        pos_emb = self.item_encoder(positives)
        tau = self._exp_tau()

        # einsum не создаёт промежуточный (n_pos, d) tensor из element-wise product.
        pos_logit = torch.einsum("bd,bd->b", queries, pos_emb) / tau

        inbatch_logits = (queries @ pos_emb.T) / tau
        inbatch_logits = inbatch_logits - self.log_q_inbatch[positives].unsqueeze(0)
        user_eq = sample_idx.unsqueeze(0) == sample_idx.unsqueeze(1)
        inbatch_logits = inbatch_logits.masked_fill(user_eq, float("-inf"))

        uniform_ids = torch.randint(0, self.n_items, (self.n_uniform,), device=device)
        uniform_emb = self.item_encoder(uniform_ids)
        uniform_logits = (queries @ uniform_emb.T) / tau - self.log_q_uniform

        all_logits = torch.cat(
            [pos_logit.unsqueeze(-1), inbatch_logits, uniform_logits],
            dim=-1,
        )
        log_probs = F.log_softmax(all_logits, dim=-1)
        loss = -log_probs[:, 0].mean()
        return loss

    def stats(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                "tau": float(self._exp_tau().item()),
                "log_tau": float(self.log_tau.item()),
            }
