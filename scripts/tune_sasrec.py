"""Optuna-тюнинг гиперпараметров SASRec-ID.

Usage:
    uv run python scripts/tune_sasrec.py --config-name tune_sasrec
    uv run python scripts/tune_sasrec.py --config-name tune_sasrec tune.n_trials=60
    uv run python scripts/tune_sasrec.py --config-name tune_sasrec data.split_set=subsample_10pct

Протокол (eval_protocol §8):
- tuning metric: val NDCG@10 (из cfg.trainer.tuning_metric)
- seed: 42 на время тюнинга; финальный multi-seed — отдельно
- test split не используется
- тюнинг делается только на SASRec-ID; HP переносятся на SASRec-Content
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import (  # noqa: E402
    ITEM_EMBEDDINGS_PARQUET,
    SAVED_DIR,
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
    load_user_sequences,
)
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
    set_seed(cfg.trainer.seed, cudnn_deterministic=bool(cfg.trainer.cudnn_deterministic))
    device = resolve_device(cfg.trainer.device)
    use_cuda = device.type == "cuda"
    logger.info(
        f"tune: model={cfg.model_name} split_set={cfg.data.split_set} "
        f"seed={cfg.trainer.seed} device={device} n_trials={cfg.tune.n_trials}"
    )

    # Тяжёлые объекты — один раз на всё исследование.
    item_map = load_item_id_map()
    popularity = load_popularity(ITEM_EMBEDDINGS_PARQUET, item_map)
    audio_embeddings = (
        _load_audio_embeddings(item_map) if cfg.item_source == "pretrained" else None
    )
    train_sequences = load_user_sequences(
        [resolve_split_parquet(cfg.data.split_set, "train")], item_map
    )
    val_ctx = EvalContext.build(item_map, cfg.data.split_set, "val")

    study_dir = SAVED_DIR / cfg.tune.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    storage_path = study_dir / "optuna.db"
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=cfg.tune.study_name,
        storage=storage_url,
        sampler=TPESampler(seed=int(cfg.tune.sampler.seed)),
        pruner=MedianPruner(
            n_startup_trials=int(cfg.tune.pruner.n_startup_trials),
            n_warmup_steps=int(cfg.tune.pruner.n_warmup_steps),
            interval_steps=int(cfg.tune.pruner.interval_steps),
        ),
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [64, 128])
        n_layers = trial.suggest_categorical("n_layers", [2, 3])
        max_seq_len = trial.suggest_categorical("max_seq_len", [128, 200, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

        logger.info(
            f"[trial {trial.number}] d_model={d_model} n_layers={n_layers} "
            f"max_seq_len={max_seq_len} dropout={dropout:.3f} "
            f"lr={lr:.2e} wd={weight_decay:.2e}"
        )

        model = SASRec(
            n_items=item_map.n_items,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=cfg.model.n_heads,
            n_layers=n_layers,
            item_source=cfg.item_source,
            audio_embeddings=audio_embeddings,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=dropout,
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

        train_ds = TrainSequenceDataset(train_sequences, max_seq_len=max_seq_len)
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

        def val_eval_fn(inst: torch.nn.Module) -> dict[str, float]:
            out = evaluate_with_context(
                inst,
                item_id_map=item_map,
                ctx=val_ctx,
                max_seq_len=max_seq_len,
                batch_size=cfg.trainer.eval_batch_size,
                device=device,
                metric_names=[cfg.trainer.tuning_metric],
                primary_k=int(cfg.data.primary_k),
                show_progress=False,
            )
            flat: dict[str, float] = {}
            for name, per_k in out["metrics"].items():
                for k, v in per_k.items():
                    flat[f"{name}@{k}"] = float(v)
            return flat

        def eval_callback(step: int, metrics: dict[str, float]) -> None:
            value = float(metrics.get(cfg.trainer.tuning_metric, -math.inf))
            if not math.isfinite(value):
                value = -1.0
            trial.report(value, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_dir = study_dir / f"trial_{trial.number:04d}"
        trainer = SASRecTrainer(
            model=model,
            loss_fn=loss_fn,
            eval_fn=val_eval_fn,
            cfg=TrainerConfig(
                epochs=int(cfg.trainer.epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                grad_clip_norm=float(cfg.trainer.grad_clip_norm),
                eval_every=int(cfg.trainer.eval_every),
                early_stopping_patience=int(cfg.trainer.early_stopping_patience),
                tuning_metric=str(cfg.trainer.tuning_metric),
                log_every=int(cfg.trainer.log_every),
            ),
            save_dir=trial_dir,
            device=device,
            writer=None,
            eval_callback=eval_callback,
        )

        try:
            trainer.fit(train_loader)
        except RuntimeError as e:
            # Non-finite loss или другой hard-фейл — помечаем триал как failed.
            logger.warning(f"[trial {trial.number}] fail: {e}")
            raise optuna.TrialPruned() from e

        return float(trainer.best_metric)

    study.optimize(
        objective,
        n_trials=int(cfg.tune.n_trials),
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # Итоговый отчёт
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    logger.info(
        f"study done: total={len(study.trials)} "
        f"completed={len(completed)} pruned={len(pruned)} failed={len(failed)}"
    )

    if not completed:
        logger.error("ни один триал не завершился — нечего выбирать")
        return

    best = study.best_trial
    logger.info(
        f"best trial={best.number} "
        f"{cfg.trainer.tuning_metric}={best.value:.5f} params={best.params}"
    )

    report = {
        "study_name": str(cfg.tune.study_name),
        "tuning_metric": str(cfg.trainer.tuning_metric),
        "n_trials_total": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_failed": len(failed),
        "best_trial": int(best.number),
        "best_value": float(best.value),
        "best_params": {k: _to_jsonable(v) for k, v in best.params.items()},
    }
    with open(study_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Hydra-совместимый override для ручного переноса в configs/model/sasrec.yaml
    # и configs/trainer/sasrec.yaml.
    override_lines = [
        "# Скопируйте эти значения в configs/model/sasrec.yaml и configs/trainer/sasrec.yaml.",
        f"# best val {cfg.trainer.tuning_metric}={best.value:.5f} (trial {best.number})",
        f"model.d_model={best.params['d_model']}",
        f"model.n_layers={best.params['n_layers']}",
        f"model.max_seq_len={best.params['max_seq_len']}",
        f"model.dropout={best.params['dropout']:.4f}",
        f"trainer.lr={best.params['lr']:.6e}",
        f"trainer.weight_decay={best.params['weight_decay']:.6e}",
    ]
    (study_dir / "best_overrides.txt").write_text("\n".join(override_lines) + "\n")


def _to_jsonable(v: object) -> object:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


if __name__ == "__main__":
    main()
