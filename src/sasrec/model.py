"""SASRec encoder for sequential recommendation.

Adapted from yandex/yambda benchmark code (Apache License 2.0):
https://huggingface.co/datasets/yandex/yambda/tree/main/benchmarks/models/sasrec/model.py

Modifications relative to the upstream version:
- ``forward`` always returns hidden states ``(B, L, d_model)`` plus the
  valid-event mask. Training loss (sampled softmax + log-Q) is computed
  externally in ``src/sasrec/loss.py``.
- ``item_source`` parameter selects between a classical trainable item
  embedding table (SASRec-ID, plan §1.1) and a frozen pretrained audio
  embedding followed by a learnable linear projection (SASRec-Content,
  plan §1.2).
- ``encode_full_history`` performs a no-grad forward pass and returns the
  last-token hidden state — the A2 frozen user state from eval_protocol §2.
- ``item_matrix`` exposes the scoring matrix (L2-normalised for Content,
  raw weights for ID) per eval_protocol §5.
- Padding id ``0`` is reserved; item embedding at row 0 is zero and frozen.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_masked_tensor(
    data: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a flat concatenation of variable-length sequences into a padded tensor.

    Events are placed **left-aligned**: positions ``[0, lengths[i])`` of sample
    ``i`` are filled from ``data``; the rest is zero. ``mask[i, j] = True``
    iff position ``j`` is a valid event of sample ``i``.
    """
    batch_size = lengths.shape[0]
    max_seq_len = int(lengths.max().item())
    device = data.device

    if data.dim() == 1:
        padded = torch.zeros(batch_size, max_seq_len, dtype=data.dtype, device=device)
    else:
        padded = torch.zeros(batch_size, max_seq_len, *data.shape[1:], dtype=data.dtype, device=device)

    mask = torch.arange(max_seq_len, device=lengths.device)[None] < lengths[:, None]
    padded[mask] = data
    return padded, mask


class TrainableItemEncoder(nn.Module):
    """Classical SASRec: learnable item embedding table (plan §1.1).

    Internally stores a ``(n_items + 1, d_model)`` table with row ``0``
    reserved for padding. Accepts ``dense_ids ∈ [0, n_items)`` and shifts
    them by ``+1`` before the lookup, so row 0 is never indexed by valid
    data.
    """

    def __init__(self, n_items: int, d_model: int, init_range: float = 0.02) -> None:
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        nn.init.trunc_normal_(
            self.emb.weight, std=init_range, a=-2 * init_range, b=2 * init_range
        )
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, dense_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(dense_ids + 1)

    def item_matrix(self) -> torch.Tensor:
        return self.emb.weight[1:]


class PretrainedItemEncoder(nn.Module):
    """SASRec-Content: frozen audio embedding + trainable linear projection (plan §1.2).

    Expects ``dense_ids ∈ [0, n_items)``; shifts by ``+1`` internally so row
    0 (frozen zero vector) is never indexed.
    """

    def __init__(
        self,
        audio_embeddings: torch.Tensor,
        d_model: int,
        init_range: float = 0.02,
    ) -> None:
        super().__init__()
        assert audio_embeddings.dim() == 2, "audio_embeddings must be (n_items, audio_dim)"
        n_items, audio_dim = audio_embeddings.shape
        self.n_items = n_items
        self.audio_dim = audio_dim
        self.d_model = d_model

        padded = torch.zeros(n_items + 1, audio_dim, dtype=audio_embeddings.dtype)
        padded[1:] = audio_embeddings
        self.audio_emb = nn.Embedding.from_pretrained(padded, freeze=True, padding_idx=0)

        self.proj = nn.Linear(audio_dim, d_model, bias=True)
        nn.init.trunc_normal_(
            self.proj.weight, std=init_range, a=-2 * init_range, b=2 * init_range
        )
        nn.init.zeros_(self.proj.bias)

    def forward(self, dense_ids: torch.Tensor) -> torch.Tensor:
        # L2-normalised per eval_protocol §7: item embeddings live on the unit
        # sphere both in training (sampled-softmax loss) and in eval (item_matrix).
        return F.normalize(self.proj(self.audio_emb(dense_ids + 1)), dim=-1)

    def item_matrix(self) -> torch.Tensor:
        raw = self.proj(self.audio_emb.weight[1:])
        return F.normalize(raw, dim=-1)


class SASRec(nn.Module):
    """SASRec encoder parameterised by item source.

    ``item_source``:
    - ``"trainable"`` — SASRec-ID (§1.1 plan).
    - ``"pretrained"`` — SASRec-Content (§1.2 plan); requires ``audio_embeddings``.

    Positional encoding is reverse-order (newest event → position 0) as in
    the upstream yambda SASRec; this is preserved because the sequence data
    is left-aligned but pos-embeddings are picked from ``arange(L-1, -1, -1)``
    and filtered by ``positions < lengths``.
    """

    def __init__(
        self,
        n_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        item_source: Literal["trainable", "pretrained"] = "trainable",
        audio_embeddings: torch.Tensor | None = None,
        dim_feedforward: int | None = None,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-9,
        init_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.item_source = item_source

        if item_source == "trainable":
            assert audio_embeddings is None
            self.item_encoder: nn.Module = TrainableItemEncoder(n_items, d_model, init_range)
        elif item_source == "pretrained":
            assert audio_embeddings is not None, "audio_embeddings required for pretrained mode"
            assert audio_embeddings.shape[0] == n_items
            self.item_encoder = PretrainedItemEncoder(audio_embeddings, d_model, init_range)
        else:
            raise ValueError(f"Unknown item_source {item_source!r}")

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        nn.init.trunc_normal_(
            self.pos_emb.weight, std=init_range, a=-2 * init_range, b=2 * init_range
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        # enable_nested_tensor=False — MPS не поддерживает NestedTensor, а
        # в fast-path TransformerEncoder он использовался бы для сжатия
        # паддингов. На CUDA отключение не критично: SDPA (flash/memeff)
        # сработает и при явном bool-маске, если is_causal=True.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Кешированная causal-маска до max_seq_len — берётся срезом в _encode,
        # не пересоздаётся на каждый forward (persistent=False: не попадает
        # в state_dict, восстанавливается из max_seq_len).
        causal_full = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask_full", causal_full, persistent=False)

        self._init_transformer_weights(init_range)

    @torch.no_grad()
    def _init_transformer_weights(self, init_range: float) -> None:
        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                else:
                    nn.init.trunc_normal_(
                        param, std=init_range, a=-2 * init_range, b=2 * init_range
                    )
            elif "bias" in name:
                nn.init.zeros_(param)

    def item_matrix(self) -> torch.Tensor:
        """Return the scoring I-matrix, shape ``(n_items, d_model)``."""
        return self.item_encoder.item_matrix()

    def _encode(
        self,
        items_flat: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core encoder: flat items + lengths → padded hidden states + mask."""
        item_emb = self.item_encoder(items_flat)
        item_emb_padded, mask = create_masked_tensor(item_emb, lengths)

        seq_len = mask.shape[1]
        device = mask.device

        # Reverse positional encoding: на валидных позициях i-го пользователя
        # (left-aligned, [0, l_i)) кладётся pos_emb(l_i - 1 - j).
        # Эквивалент старой схемы (positions < lengths → flat → create_masked_tensor),
        # но без второй аллокации (B, L, d): один embedding lookup + element-wise
        # mask для зануления паддингов.
        arange_l = torch.arange(seq_len, device=device)
        pos_indices = (lengths[:, None] - 1 - arange_l[None]).clamp(min=0)
        pos_emb_padded = self.pos_emb(pos_indices) * mask.unsqueeze(-1)

        x = item_emb_padded + pos_emb_padded
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        causal_mask = self._causal_mask_full[:seq_len, :seq_len]
        # is_causal=True — подсказка для SDPA (flash/mem-efficient kernel на CUDA);
        # на MPS остаётся явный bool-mask путь.
        x = self.encoder(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=~mask,
            is_causal=True,
        )
        return x, mask

    def forward(
        self,
        items_flat: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(hidden_states (B, L, d_model), mask (B, L))``."""
        return self._encode(items_flat, lengths)

    @torch.no_grad()
    def encode_full_history(
        self,
        items_flat: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """A2 frozen user state: last-token hidden state of each sequence."""
        was_training = self.training
        self.eval()
        try:
            x, mask = self._encode(items_flat, lengths)
            batch_size, seq_len, d_model = x.shape
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(batch_size, device=x.device)
            z_u = x[batch_idx, last_idx]
            empty = lengths == 0
            if empty.any():
                z_u[empty] = 0.0
            return z_u
        finally:
            if was_training:
                self.train()
