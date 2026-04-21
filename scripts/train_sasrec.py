"""Train SASRec-ID or SASRec-Content.

Usage:
    uv run python scripts/train_sasrec.py --config-name sasrec_id
    uv run python scripts/train_sasrec.py --config-name sasrec_content
    uv run python scripts/train_sasrec.py --config-name sasrec_id -m trainer.seed=42,43,44
    uv run python scripts/train_sasrec.py --config-name sasrec_id data.split_set=subsample_1pct
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import (  # noqa: E402
    ITEM_EMBEDDINGS_PARQUET,
    SAVED_DIR,
    USER_VECTORS_DIR,
    load_item_id_map,
    load_popularity,
    resolve_split_parquet,
)
from src.sasrec import (  # noqa: E402
    EvalContext,
    SASRec,
    SASRecTrainer,
    SampledSoftmaxLoss,
    TrainerConfig,
    TrainSequenceDataset,
    collate_train,
    evaluate_with_context,
    extract_and_save,
    load_user_sequences,
)
from src.utils.io import save_metrics  # noqa: E402
from src.utils.seed import resolve_device, set_seed  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_audio_embeddings(item_id_map) -> torch.Tensor:
    df = (
        pl.scan_parquet(ITEM_EMBEDDINGS_PARQUET)
        .select("item_id", "normalized_embed")
        .collect()
    )
    raw_ids = df["item_id"].to_numpy().astype(np.int64)
    embeds_list = df["normalized_embed"].to_list()
    embeds = torch.tensor(np.asarray(embeds_list, dtype=np.float32))
    dense = item_id_map.to_dense(raw_ids)
    aligned = torch.zeros(item_id_map.n_items, embeds.shape[1], dtype=torch.float32)
    aligned[dense] = embeds
    return aligned


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:
    cudnn_det = bool(OmegaConf.select(cfg, "trainer.cudnn_deterministic", default=True))
    set_seed(cfg.trainer.seed, cudnn_deterministic=cudnn_det)
    device = resolve_device(cfg.trainer.device)
    use_cuda = device.type == "cuda"
    logger.info(
        f"model={cfg.model_name} split_set={cfg.data.split_set} "
        f"seed={cfg.trainer.seed} device={device}"
    )

    item_map = load_item_id_map()
    popularity = load_popularity(ITEM_EMBEDDINGS_PARQUET, item_map)

    audio_embeddings = (
        _load_audio_embeddings(item_map) if cfg.item_source == "pretrained" else None
    )

    model = SASRec(
        n_items=item_map.n_items,
        max_seq_len=cfg.model.max_seq_len,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        item_source=cfg.item_source,
        audio_embeddings=audio_embeddings,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        layer_norm_eps=cfg.model.layer_norm_eps,
        init_range=cfg.model.init_range,
    ).to(device)

    loss_fn = SampledSoftmaxLoss(
        item_encoder=model.item_encoder,
        n_items=item_map.n_items,
        popularity=popularity,
        n_uniform=cfg.trainer.n_uniform_negatives,
        init_temperature=cfg.trainer.init_temperature,
        max_n_pos_per_step=int(
            OmegaConf.select(cfg, "trainer.max_n_pos_per_step", default=0)
        ),
    ).to(device)

    train_sequences = load_user_sequences(
        [resolve_split_parquet(cfg.data.split_set, "train")], item_map
    )
    train_ds = TrainSequenceDataset(train_sequences, max_seq_len=cfg.model.max_seq_len)
    num_workers = int(cfg.trainer.num_workers) if use_cuda else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        collate_fn=collate_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(use_cuda and num_workers > 0),
        drop_last=False,
    )

    run_dir = SAVED_DIR / cfg.run_name
    vectors_dir = USER_VECTORS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    writer = None
    if cfg.wandb.enabled:
        from src.logger import WandBWriter

        project_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        writer = WandBWriter(
            project_config=project_config,
            project_name=cfg.wandb.project,
            entity=cfg.wandb.entity,
            run_name=cfg.run_name,
            mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags),
        )

    val_ctx = EvalContext.build(item_map, cfg.data.split_set, "val")

    def val_eval_fn(inst: torch.nn.Module) -> dict[str, float]:
        out = evaluate_with_context(
            inst,
            item_id_map=item_map,
            ctx=val_ctx,
            max_seq_len=cfg.model.max_seq_len,
            batch_size=cfg.trainer.eval_batch_size,
            device=device,
            metric_names=[cfg.trainer.tuning_metric],
            primary_k=int(cfg.data.primary_k),
            show_progress=False,
        )
        flat = {}
        for name, per_k in out["metrics"].items():
            for k, v in per_k.items():
                flat[f"{name}@{k}"] = float(v)
        return flat

    trainer = SASRecTrainer(
        model=model,
        loss_fn=loss_fn,
        eval_fn=val_eval_fn,
        cfg=TrainerConfig(
            epochs=cfg.trainer.epochs,
            lr=cfg.trainer.lr,
            weight_decay=cfg.trainer.weight_decay,
            grad_clip_norm=cfg.trainer.grad_clip_norm,
            eval_every=cfg.trainer.eval_every,
            early_stopping_patience=cfg.trainer.early_stopping_patience,
            tuning_metric=cfg.trainer.tuning_metric,
            log_every=cfg.trainer.log_every,
        ),
        save_dir=run_dir,
        device=device,
        writer=writer,
    )

    try:
        trainer.fit(train_loader)

        k_values = list(cfg.data.k_values)
        metric_names = [f"{m}@{k}" for m in cfg.data.metrics for k in k_values]

        test_ctx = EvalContext.build(item_map, cfg.data.split_set, "test")

        results: dict[str, object] = {}
        per_user_rows: list[dict[str, float | int | str]] = []
        for split, ctx in (("val", val_ctx), ("test", test_ctx)):
            out = evaluate_with_context(
                model,
                item_id_map=item_map,
                ctx=ctx,
                max_seq_len=cfg.model.max_seq_len,
                batch_size=cfg.trainer.eval_batch_size,
                device=device,
                metric_names=metric_names,
                primary_k=int(cfg.data.primary_k),
                show_progress=False,
            )
            results[split] = out["metrics"]
            uids = out["per_user"]["user_ids"]
            recall = out["per_user"]["recall"]
            ndcg = out["per_user"]["ndcg"]
            for uid, r, n in zip(uids.tolist(), recall.tolist(), ndcg.tolist(), strict=True):
                per_user_rows.append(
                    {"split": split, "uid": int(uid), "recall": float(r), "ndcg": float(n)}
                )

        save_metrics(run_dir / "metrics.json", results)
        pl.DataFrame(per_user_rows).write_parquet(run_dir / "per_user.parquet")

        if writer is not None:
            writer.set_step(trainer.step, mode="final")
            flat_final: dict[str, float] = {}
            for split_name, split_metrics in results.items():
                for metric_name, per_k in split_metrics.items():
                    for k, v in per_k.items():
                        flat_final[f"{split_name}/{metric_name}@{k}"] = float(v)
            flat_final["best_val_metric"] = float(trainer.best_metric)
            flat_final["best_step"] = float(trainer.best_step)
            flat_final["total_steps"] = float(trainer.step)
            writer.add_scalars(flat_final)
            writer.set_summary(
                {
                    **{f"final/{key}": value for key, value in flat_final.items()},
                    "best/val_metric": float(trainer.best_metric),
                    "best/step": int(trainer.best_step),
                }
            )

        extract_and_save(
            model=model,
            item_id_map=item_map,
            split_set=cfg.data.split_set,
            out_dir=vectors_dir,
            max_seq_len=cfg.model.max_seq_len,
            batch_size=cfg.trainer.eval_batch_size,
            device=device,
        )

        logger.info(
            f"DONE. best val {cfg.trainer.tuning_metric}={trainer.best_metric:.5f}. "
            f"run_dir={run_dir} vectors_dir={vectors_dir}"
        )
    finally:
        if writer is not None:
            writer.finish()


if __name__ == "__main__":
    main()
