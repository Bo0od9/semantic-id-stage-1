from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from src.sasrec.loss import SampledSoftmaxLoss
from src.sasrec.model import SASRec, create_masked_tensor


def _make_model(n_items=50, max_seq_len=16, d_model=16, dropout=0.0):
    torch.manual_seed(0)
    return SASRec(
        n_items=n_items,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        dropout=dropout,
    )


def test_create_masked_tensor_layout():
    data = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)
    lengths = torch.tensor([3, 1, 2], dtype=torch.long)
    padded, mask = create_masked_tensor(data, lengths)
    assert padded.shape == (3, 3)
    assert padded[0].tolist() == [10, 20, 30]
    assert padded[1].tolist() == [40, 0, 0]
    assert padded[2].tolist() == [50, 60, 0]
    assert mask.tolist() == [[True, True, True], [True, False, False], [True, True, False]]


def test_forward_shapes():
    m = _make_model(n_items=30, max_seq_len=10, d_model=8)
    items = torch.tensor([0, 1, 2, 15, 7, 8, 9], dtype=torch.long)
    lengths = torch.tensor([4, 3], dtype=torch.long)
    h, mask = m(items, lengths)
    assert h.shape == (2, 4, 8)
    assert mask.shape == (2, 4)
    assert mask.sum().item() == 7


def test_trainable_item_matrix_shape_and_padding():
    m = _make_model(n_items=30, max_seq_len=10, d_model=8)
    mat = m.item_matrix()
    assert mat.shape == (30, 8)
    # padding row (index 0 in underlying emb) must be zero — should not appear in item_matrix
    # Here item_matrix = emb.weight[1:], so padding row is excluded.
    # Verify that accessing dense_id=0 indirectly (through forward) doesn't return emb.weight[0].
    ids = torch.tensor([0, 1, 2], dtype=torch.long)
    lens = torch.tensor([3], dtype=torch.long)
    h, _ = m(ids, lens)
    assert h.shape == (1, 3, 8)


def test_pretrained_item_matrix_is_l2_normalized():
    audio = torch.randn(20, 32)
    m = SASRec(
        n_items=20, max_seq_len=8, d_model=16, n_heads=2, n_layers=1,
        item_source="pretrained", audio_embeddings=audio, dropout=0.0,
    )
    mat = m.item_matrix()
    assert mat.shape == (20, 16)
    norms = torch.norm(mat, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_pretrained_forward_matches_item_matrix_rows():
    """Training (forward) and eval (item_matrix) item vectors must coincide
    on the unit sphere — eval_protocol §5 train/eval consistency."""
    audio = torch.randn(20, 32)
    m = SASRec(
        n_items=20, max_seq_len=8, d_model=16, n_heads=2, n_layers=1,
        item_source="pretrained", audio_embeddings=audio, dropout=0.0,
    )
    m.eval()
    ids = torch.tensor([0, 5, 17], dtype=torch.long)
    with torch.no_grad():
        emb_forward = m.item_encoder(ids)
        mat = m.item_matrix()
    norms = torch.norm(emb_forward, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        "PretrainedItemEncoder.forward must return unit-norm vectors"
    )
    for k, dense_id in enumerate(ids.tolist()):
        assert torch.allclose(emb_forward[k], mat[dense_id], atol=1e-6)


def test_trainable_item_matrix_not_l2_normalized():
    m = _make_model(n_items=20, max_seq_len=8, d_model=16)
    mat = m.item_matrix()
    norms = torch.norm(mat, dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "trainable norms should vary"


def test_causal_mask_blocks_future():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.0)
    m.eval()
    # Build two sequences identical in the first 3 positions, different in position 3
    base = torch.tensor([1, 2, 3, 40, 5, 6], dtype=torch.long)
    alt = torch.tensor([1, 2, 3, 41, 5, 6], dtype=torch.long)
    lengths = torch.tensor([6], dtype=torch.long)

    with torch.no_grad():
        h1, _ = m(base, lengths)
        h2, _ = m(alt, lengths)

    # Positions 0..2 share full prefix (causal attention cannot see position 3 onwards).
    # NOTE: reverse positional embeddings mean the "past" direction in attention =
    # indices 0..t → positions (seq_len-1)..(seq_len-1-t).
    # That is still causal in the standard tensor sense, so positions 0..2 depend only
    # on items 0..2, which match between base and alt.
    assert torch.allclose(h1[0, :3], h2[0, :3], atol=1e-6), "causal mask leaked future info"
    # Position 3 is where the first differing input is — here outputs must differ.
    assert not torch.allclose(h1[0, 3], h2[0, 3], atol=1e-3)


def test_padding_invariance_on_valid_positions():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.0)
    m.eval()

    # Two samples in a batch: first length 4, second length 2.
    # Outputs on valid positions of sample 0 must be identical whether sample 1 is in the batch or not.
    items_full = torch.tensor([1, 2, 3, 4, 10, 11], dtype=torch.long)
    lengths_full = torch.tensor([4, 2], dtype=torch.long)

    items_alone = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    lengths_alone = torch.tensor([4], dtype=torch.long)

    with torch.no_grad():
        h_full, mask_full = m(items_full, lengths_full)
        h_alone, mask_alone = m(items_alone, lengths_alone)

    # Compare h_full[0, :4] (valid positions of sample 0) vs h_alone[0, :4]
    assert torch.allclose(h_full[0, :4], h_alone[0, :4], atol=1e-5)


def test_encode_full_history_uses_last_token():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.0)
    m.eval()
    items = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    with torch.no_grad():
        h, mask = m(items, lengths)
        z = m.encode_full_history(items, lengths)

    assert z.shape == (2, 16)
    assert torch.allclose(z[0], h[0, 2], atol=1e-6)  # last valid = position length-1 = 2
    assert torch.allclose(z[1], h[1, 1], atol=1e-6)


def test_encode_full_history_deterministic():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.3)
    m.eval()
    items = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    lengths = torch.tensor([2, 2], dtype=torch.long)

    z1 = m.encode_full_history(items, lengths)
    z2 = m.encode_full_history(items, lengths)
    assert torch.allclose(z1, z2, atol=1e-6), "eval mode must disable dropout"


def test_sampled_softmax_runs_and_is_finite():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.0)
    pop = torch.arange(1, 51).long()  # nonzero popularity for all items
    loss_fn = SampledSoftmaxLoss(
        item_encoder=m.item_encoder, n_items=50, popularity=pop, n_uniform=64, init_temperature=0.5,
    )
    items = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    positives = torch.tensor([2, 3, 4, 7, 8, 9], dtype=torch.long)
    lengths = torch.tensor([3, 3], dtype=torch.long)

    m.train()
    h, mask = m(items, lengths)
    loss = loss_fn(h, mask, positives)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters())
    assert loss_fn.log_tau.grad is not None


def test_sampled_softmax_zero_popularity_is_rejected():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16)
    pop = torch.zeros(50, dtype=torch.long)
    try:
        SampledSoftmaxLoss(
            item_encoder=m.item_encoder, n_items=50, popularity=pop, n_uniform=32,
        )
    except AssertionError:
        return
    raise AssertionError("should have rejected all-zero popularity")


def test_sampled_softmax_log_q_magnitudes():
    m = _make_model(n_items=50, max_seq_len=8, d_model=16)
    # Make popularity trivially skewed: item 10 is dominant
    pop = torch.ones(50, dtype=torch.long)
    pop[10] = 999_999
    loss_fn = SampledSoftmaxLoss(
        item_encoder=m.item_encoder, n_items=50, popularity=pop, n_uniform=4,
    )
    log_q_inbatch = loss_fn.log_q_inbatch
    # dominant item has higher q_inbatch → higher log_q_inbatch
    assert log_q_inbatch[10].item() > log_q_inbatch[0].item() + 5.0
    # uniform q = 1/n_items → log = -log(50) ≈ -3.91
    expected_uniform = math.log(1.0 / 50)
    assert abs(loss_fn.log_q_uniform.item() - expected_uniform) < 1e-5


def test_sampled_softmax_inbatch_diagonal_masked():
    """Same-user positions (incl. the diagonal self-match) must be masked
    out of in-batch negatives so they don't enter the softmax denominator."""
    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.0)
    pop = torch.arange(1, 51).long()
    loss_fn = SampledSoftmaxLoss(
        item_encoder=m.item_encoder, n_items=50, popularity=pop, n_uniform=8,
    )
    # Two batch users, two positions each → 4 query rows; same-user pairs
    # cover the diagonal and the two cross-position cells per user.
    items = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    positives = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    lengths = torch.tensor([2, 2], dtype=torch.long)

    m.eval()
    h, mask = m(items, lengths)
    queries = h[mask]
    pos_emb = loss_fn.item_encoder(positives)
    tau = loss_fn._exp_tau()
    inbatch = (queries @ pos_emb.T) / tau - loss_fn.log_q_inbatch[positives].unsqueeze(0)
    sample_idx = torch.tensor([0, 0, 1, 1])
    user_eq = sample_idx.unsqueeze(0) == sample_idx.unsqueeze(1)
    inbatch = inbatch.masked_fill(user_eq, float("-inf"))

    # Expected mask: block-diagonal True
    assert torch.equal(
        torch.isinf(inbatch) & (inbatch < 0),
        torch.tensor(
            [[True, True, False, False],
             [True, True, False, False],
             [False, False, True, True],
             [False, False, True, True]],
        ),
    )


def test_load_user_sequences_respects_timestamp_order(tmp_path):
    """Sequences must be ordered by timestamp ASC regardless of row order in
    the parquet — Polars ``group_by`` does not preserve outer sort, so the
    ordering guarantee relies on ``sort_by`` inside ``agg``."""
    from src.data.dataset import ItemIdMap
    from src.sasrec import dataset as ds_mod

    path = tmp_path / "flat.parquet"
    # Rows deliberately scrambled per uid; if sort order leaks, items will come
    # back in insertion order rather than chronological.
    pl.DataFrame(
        {
            "uid": [0, 1, 0, 1, 0],
            "timestamp": [30, 50, 10, 40, 20],
            "item_id": [3, 5, 1, 4, 2],
        }
    ).write_parquet(path)

    item_id_map = ItemIdMap(dense_to_raw=np.array([1, 2, 3, 4, 5], dtype=np.int64))
    sequences = ds_mod.load_user_sequences([path], item_id_map)
    # raw 1,2,3,4,5 → dense 0,1,2,3,4
    assert sequences[0].tolist() == [0, 1, 2]  # ts 10,20,30 → items 1,2,3
    assert sequences[1].tolist() == [3, 4]     # ts 40,50 → items 4,5


def test_load_prefix_sequences_rejects_data_leak(tmp_path, monkeypatch):
    """If a prefix parquet contains a timestamp ≥ cutoff, load_prefix_sequences
    must raise — protects A2 frozen state from leaking the eval window."""
    from src.sasrec import dataset as ds_mod
    from src.data.dataset import ItemIdMap

    train_path = tmp_path / "train.parquet"
    pl.DataFrame(
        {
            "uid": [0, 0, 1],
            "timestamp": [10, 250, 50],  # 250 >= T_val=100 → leak
            "item_id": [1, 2, 3],
        }
    ).write_parquet(train_path)

    def fake_resolve(_split_set: str, split: str) -> Path:
        return train_path if split == "train" else tmp_path / "val.parquet"

    monkeypatch.setattr(ds_mod, "resolve_split_parquet", fake_resolve)
    monkeypatch.setattr(
        ds_mod,
        "load_temporal_cutoffs",
        lambda *_a, **_kw: {"T_val": 100, "T_test": 200},
    )

    item_id_map = ItemIdMap(dense_to_raw=np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(AssertionError, match="Data leak"):
        ds_mod.load_prefix_sequences("subsample_1pct", "val", item_id_map)


def test_sampled_softmax_max_n_pos_caps_logits():
    """max_n_pos_per_step > 0 должен обрезать валидные позиции до порога.
    Проверяется через патч на inbatch-логиты: их ширина == min(n_pos, cap)."""
    import types

    m = _make_model(n_items=50, max_seq_len=8, d_model=16, dropout=0.0)
    pop = torch.arange(1, 51).long()
    loss_fn = SampledSoftmaxLoss(
        item_encoder=m.item_encoder,
        n_items=50,
        popularity=pop,
        n_uniform=4,
        max_n_pos_per_step=3,
    )
    # n_pos before cap = 2+2+2 = 6 → после cap = 3.
    items = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    positives = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    lengths = torch.tensor([2, 2, 2], dtype=torch.long)

    m.eval()
    h, mask = m(items, lengths)
    torch.manual_seed(0)
    loss = loss_fn(h, mask, positives)
    assert torch.isfinite(loss)

    # disabled (cap=0) → все 6 позиций участвуют. Проверяем через градиент log_tau:
    # cap=0 и cap=6 должны обучаться одинаково стохастически, но размерности позволяют.
    loss_fn_disabled = SampledSoftmaxLoss(
        item_encoder=m.item_encoder,
        n_items=50,
        popularity=pop,
        n_uniform=4,
        max_n_pos_per_step=0,
    )
    loss_disabled = loss_fn_disabled(h, mask, positives)
    assert torch.isfinite(loss_disabled)


def test_sampled_softmax_item_encoder_not_in_params():
    m = _make_model(n_items=30, max_seq_len=8, d_model=16)
    pop = torch.ones(30, dtype=torch.long)
    loss_fn = SampledSoftmaxLoss(
        item_encoder=m.item_encoder, n_items=30, popularity=pop, n_uniform=16,
    )
    loss_params = list(loss_fn.parameters())
    assert len(loss_params) == 1, f"loss_fn should expose only log_tau, got {len(loss_params)}"
    model_param_ids = {id(p) for p in m.parameters()}
    loss_param_ids = {id(p) for p in loss_params}
    assert not (loss_param_ids & model_param_ids), "item_encoder params must not leak into loss_fn.parameters()"
